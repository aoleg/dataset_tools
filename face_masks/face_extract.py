#!/usr/bin/env python3
"""
Cut close-up face crops out of a folder of images, for character training.

For every face detected in IMG_DIR, writes OUT_DIR/<tier>/<basename>_face<N>.png
(OUT_DIR defaults to IMG_DIR/faces; <tier> is 512, 768 or 1024).

Each crop is framed around the face and hair, cut at the aspect ratio of a
standard musubi-tuner bucket and resized to that bucket. The tier is the largest
one the natural crop can fill (the same rule as k2prep, up to 1.15x upscale), so
1024 holds every face whose crop reaches 1024 or more. A crop too small for 512
is made looser until it reaches 512. If that is impossible, because the image
is too small or the face would fill less than --min-fill of the crop, the face
is skipped.

Detector: YOLOv8 face model from the adetailer repo (Bingsu/adetailer on HF).
Usage:    python face_extract.py <img_dir> [<out_dir>] [--fill 0.5] [--min-fill 0.25]
          [--ratio auto] [--conf 0.5] [--largest-only] [--overwrite]
"""
import argparse
import math
import os
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache

from PIL import Image, ImageOps

from make_face_masks import DEFAULT_MODEL, IMG_EXTS, face_boxes, face_region, load_model

# ---------------------------------------------------------------------------
# Buckets. Copied from k2prep.py (../k2prep), which ports
# musubi-tuner's BucketSelector. Keep the two in step.
# ---------------------------------------------------------------------------

TIERS = [1024, 768, 512]          # nominal resolution, descending
UPSCALE_TOLERANCE = 1.15          # max permitted linear upscale into a tier
RESO_STEPS = 16

AR_FAMILIES = ["9:16", "2:3", "4:5", "1:1", "5:4", "3:2", "16:9"]
AR_NOMINAL = [0.5647, 0.6667, 0.8028, 1.0000, 1.2456, 1.5000, 1.7708]


def divisible_by(n: int, d: int) -> int:
    return n - n % d


@lru_cache(maxsize=None)
def generate_buckets(resolution: int, steps: int = RESO_STEPS) -> list:
    area = resolution * resolution
    sqrt_size = int(math.sqrt(area))
    min_size = divisible_by(sqrt_size // 2, steps)
    out = []
    for w in range(min_size, sqrt_size + steps, steps):
        h = divisible_by(area // w, steps)
        out.append((w, h))
        out.append((h, w))
    return sorted(set(out))


@lru_cache(maxsize=None)
def bucket_for(tier: int, family: str):
    nominal = AR_NOMINAL[AR_FAMILIES.index(family)]
    return min(generate_buckets(tier), key=lambda b: abs(b[0] / b[1] - nominal))


HEADSHOT_FAMILIES = ["2:3", "4:5", "1:1"]   # preferred by --ratio auto


def needed_area(bucket) -> float:
    """Smallest source crop area that fills the bucket within UPSCALE_TOLERANCE."""
    return bucket[0] * bucket[1] / UPSCALE_TOLERANCE ** 2


# ---------------------------------------------------------------------------
# Crop planning
# ---------------------------------------------------------------------------

def clip_box(box, img_w, img_h):
    x0, y0, x1, y1 = box
    return max(0.0, x0), max(0.0, y0), min(float(img_w), x1), min(float(img_h), y1)


def box_area(box) -> float:
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def intersect(a, b):
    return max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])


def fit_dims(ar: float, area: float, img_w: int, img_h: int):
    """Crop of ratio ar and at least `area` pixels, shrunk to fit inside the image."""
    w = math.sqrt(area * ar)
    h = w / ar
    if w > img_w:
        w, h = img_w, img_w / ar
    if h > img_h:
        w, h = img_h * ar, img_h
    return max(1, min(img_w, math.ceil(w - 1e-6))), max(1, min(img_h, math.ceil(h - 1e-6)))


FACE_CENTRE_Y = 0.42              # face centre, as a share of crop height from the top


def place(face_box, w: int, h: int, img_w: int, img_h: int):
    """Put a w x h crop around the face, then shift it inside the image.

    Horizontally centred on the face. Vertically, the face centre sits at
    FACE_CENTRE_Y, as in a portrait. The face and hair region is not used here:
    its hair allowance is generous, and placing by it leaves bald or short-haired
    heads under a band of empty background. The crop is at least as tall as the
    region, which leaves room for normal hair above the face.
    """
    cx, cy = (face_box[0] + face_box[2]) / 2, (face_box[1] + face_box[3]) / 2
    left = min(max(0, int(round(cx - w / 2))), img_w - w)
    top = min(max(0, int(round(cy - FACE_CENTRE_Y * h))), img_h - h)
    return left, top, left + w, top + h


def plan_for_family(face_box, region, family, img_w, img_h, fill, min_fill):
    """-> (tier, bucket, crop box, actual fill) or (None, reason)."""
    rw, rh = region[2] - region[0], region[3] - region[1]
    r_area = rw * rh

    def crop_at(ar, area):
        # The crop must contain the whole region, whatever the ratio.
        area = max(area, rw * rw / ar, rh * rh * ar)
        w, h = fit_dims(ar, area, img_w, img_h)
        return place(face_box, w, h, img_w, img_h)

    chosen = None
    for tier in TIERS:
        bucket = bucket_for(tier, family)
        crop = crop_at(bucket[0] / bucket[1], r_area / fill)
        if box_area(crop) >= needed_area(bucket):
            chosen = tier, bucket, crop
            break
    if chosen is None:
        # Too small for 512 at the preferred framing: loosen the cut to reach 512.
        bucket = bucket_for(512, family)
        crop = crop_at(bucket[0] / bucket[1], max(r_area / fill, needed_area(bucket)))
        if box_area(crop) < needed_area(bucket):
            return None, "image too small"
        chosen = 512, bucket, crop

    tier, bucket, crop = chosen
    # Size of the face against the crop. Not the overlap: placement may cut some
    # of the hair allowance, and that must not make a face count as smaller.
    actual = r_area / box_area(crop)
    if actual < min_fill:
        return None, f"face too small (fills {actual:.1%} of the crop)"
    return tier, bucket, crop, actual


def plan_crop(face_box, region, img_w, img_h, ratio, fill, min_fill):
    """Try the requested family, or in auto mode every family, head-shot ratios first.

    Auto ranks by the detected face box, not by the face and hair region: the hair
    extension makes the region tall, and matching it gives 9:16 strips. Profile
    faces have narrow boxes too, so the head-shot ratios come first and the others
    are only a fallback for faces that cannot fit them.
    """
    if ratio == "auto":
        f_ar = (face_box[2] - face_box[0]) / (face_box[3] - face_box[1])

        def by_closeness(fams):
            return sorted(fams, key=lambda f: abs(AR_NOMINAL[AR_FAMILIES.index(f)] - f_ar))
        families = by_closeness(HEADSHOT_FAMILIES) + by_closeness(
            [f for f in AR_FAMILIES if f not in HEADSHOT_FAMILIES])
    else:
        families = [ratio]
    first_reason = None
    for family in families:
        plan = plan_for_family(face_box, region, family, img_w, img_h, fill, min_fill)
        if plan[0] is not None:
            return plan
        first_reason = first_reason or plan[1]
    return None, first_reason


# ---------------------------------------------------------------------------
# Output bookkeeping
# ---------------------------------------------------------------------------

OUT_NAME = re.compile(r"^(?P<stem>.+)_face\d+\.(png|jpg)$", re.IGNORECASE)


def existing_outputs(out_dir: str):
    """stem -> list of output paths already written by an earlier run."""
    found = defaultdict(list)
    for tier in TIERS:
        d = os.path.join(out_dir, str(tier))
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            m = OUT_NAME.match(f)
            if m:
                found[m.group("stem")].append(os.path.join(d, f))
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("img_dir")
    ap.add_argument("out_dir", nargs="?", default=None, help="output dir (default: <img_dir>/faces)")
    ap.add_argument("--fill", type=float, default=0.5,
                    help="preferred share of the crop covered by the face and hair area")
    ap.add_argument("--min-fill", type=float, default=0.25,
                    help="skip a face when a looser cut would drop its share below this")
    ap.add_argument("--ratio", default="auto", choices=["auto"] + AR_FAMILIES,
                    help="bucket aspect ratio; auto picks 2:3, 4:5 or 1:1 by the face shape")
    ap.add_argument("--grow", type=float, default=1.35, help="scale factor on the detected face box")
    ap.add_argument("--conf", type=float, default=0.5, help="detector confidence threshold")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="adetailer face model file name")
    ap.add_argument("--largest-only", action="store_true", help="keep only the largest face of each image")
    ap.add_argument("--allow-others", action="store_true",
                    help="keep crops that also contain half or more of another detected face")
    ap.add_argument("--keep-size", action="store_true",
                    help="save the crop at its source pixel size instead of resizing it to the bucket")
    ap.add_argument("--format", default="png", choices=["png", "jpg"], help="output file format")
    ap.add_argument("--overwrite", action="store_true",
                    help="redo images that already have crops (their old crops are deleted)")
    args = ap.parse_args()

    if not os.path.isdir(args.img_dir):
        sys.exit(f"not a directory: {args.img_dir}")
    if not 0 < args.min_fill <= args.fill <= 1:
        sys.exit("need 0 < --min-fill <= --fill <= 1")
    if args.out_dir is None:
        args.out_dir = os.path.join(args.img_dir, "faces")
        print(f"out_dir not given, writing faces to {args.out_dir}")

    files = sorted(
        f for f in os.listdir(args.img_dir) if os.path.splitext(f)[1].lower() in IMG_EXTS
    )
    if not files:
        sys.exit(f"no images in {args.img_dir}")

    done = existing_outputs(args.out_dir)
    model = load_model(args.model)
    per_tier = Counter()
    skipped = []            # (file, face number, reason)
    no_face = []
    already = 0
    for i, f in enumerate(files, 1):
        stem = os.path.splitext(f)[0]
        if stem in done:
            if not args.overwrite:
                already += 1
                continue
            for p in done[stem]:
                os.remove(p)

        img = ImageOps.exif_transpose(Image.open(os.path.join(args.img_dir, f))).convert("RGB")
        img_w, img_h = img.size
        all_boxes = sorted(face_boxes(model, img, args.conf), key=box_area, reverse=True)
        if not all_boxes:
            no_face.append(f)
            print(f"[{i}/{len(files)}] {f}: no face", flush=True)
            continue
        boxes = all_boxes[:1] if args.largest_only else all_boxes

        results = []
        for n, box in enumerate(boxes, 1):
            region = clip_box(face_region(box, args.grow, include_hair=True), img_w, img_h)
            plan = plan_crop(box, region, img_w, img_h, args.ratio, args.fill, args.min_fill)
            if plan[0] is None:
                skipped.append((f, n, plan[1]))
                results.append(f"face{n} skipped: {plan[1]}")
                continue
            tier, bucket, crop, actual = plan
            if not args.allow_others:
                other = next((b for b in all_boxes if b is not box
                              and box_area(intersect(b, crop)) >= 0.5 * box_area(b)), None)
                if other is not None:
                    skipped.append((f, n, "another face in the crop"))
                    results.append(f"face{n} skipped: another face in the crop")
                    continue
            out = img.crop(crop)
            if not args.keep_size:
                out = out.resize(bucket, Image.LANCZOS)
            tier_dir = os.path.join(args.out_dir, str(tier))
            os.makedirs(tier_dir, exist_ok=True)
            path = os.path.join(tier_dir, f"{stem}_face{n}.{args.format}")
            if args.format == "jpg":
                out.save(path, quality=95)
            else:
                out.save(path)
            per_tier[tier] += 1
            results.append(f"face{n} -> {tier} {bucket[0]}x{bucket[1]} (fill {actual:.0%})")
        print(f"[{i}/{len(files)}] {f}: " + "; ".join(results), flush=True)

    print(f"\nWritten: " + ", ".join(f"{t}: {per_tier[t]}" for t in TIERS)
          + f"  (total {sum(per_tier.values())})")
    if already:
        print(f"{already} image(s) already had crops and were not redone (use --overwrite).")
    if skipped:
        print(f"\n{len(skipped)} face(s) skipped:")
        for f, n, reason in skipped:
            print(f"   {f} face{n}: {reason}")
    if no_face:
        print(f"\n{len(no_face)} image(s) with no detected face:")
        for f in no_face:
            print("  ", f)


if __name__ == "__main__":
    main()
