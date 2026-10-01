#!/usr/bin/env python3
"""
Find duplicate images with perceptual hashes and move the worse copies out.

Every folder given is scanned recursively, and all of them form one pool. Two
images are copies when their perceptual hashes are close, or when local image
features (ORB) show the same picture under another crop, border, scale or a
mock-up. Copies connected through such matches form one group. In each group the
best copy stays; every other copy, and its .txt caption, moves to
<root>/_duplicates/<same relative path>, where <root> is the folder given on the
command line that contains it. No prompt.

Quality is judged the k2prep way: each copy is scored as the trainer would see it,
cropped and resized to the 512, 768 or 1024 bucket it reaches. A copy that reaches
a larger bucket wins, unless it scores GUARD or more points lower. Equal copies:
the one with a caption wins, then the oldest file (modification time).

Usage:    python dedup.py <folder> [<folder> ...] [--match loose|strict|exact]
          [--dry-run] [--undo] [--exclude NAME] [--workers N]
Install:  pip install Pillow numpy imagehash opencv-python-headless
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None     # large scans are normal input, not an attack

OUT_DIRNAME = "_duplicates"
HASHES_NAME = "hashes.json"
VERIFIED_NAME = "verified.json"
PLAN_NAME = "plan.json"
MOVES_NAME = "moves.jsonl"
CACHE_VERSION = 1

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".avif"}
DEFAULT_EXCLUDES = ["masks", "faces"]   # face_masks output: near-identical by design

# --- matching, calibrated on 690 real poster and photo downloads -------------
# pHash and dHash are 64-bit; distances are Hamming distances. pHash distances
# are always even (each hash has exactly 32 bits set).
#   loose : hash rule, plus ORB feature verification of a wider candidate window:
#           other scans, colour casts, borders, frames, wall mock-ups, watermarks,
#           different crops. Known false match: two different works that share
#           the same central photo (photomontage).
#   strict: resized, recompressed and lightly watermarked copies only; no ORB.
#   exact : byte-identical files only.
MATCH_RULES = {
    "loose": [(14, 16), (16, 8)],      # (max phash, max dhash); any rule may match
    "strict": [(8, 10)],
    "exact": [],
}
# Candidate window for the feature check (loose only): each image's nearest
# CAND_PER_IMAGE neighbours by phash + dhash within these distances. Every true
# copy in the calibration set was within rank 22 of its partner.
CAND_MAX_PHASH = 22
CAND_MAX_DHASH = 22
CAND_PER_IMAGE = 30
# Feature check: ORB keypoints on a FEATURE_SIDE image, ratio test, RANSAC
# homography. A copy needs MIN_INLIERS consistent matches spread over MIN_COVER
# of both images, and the centre of one, warped onto the other, must correlate
# at MIN_NCC or more. The last test rejects identical mock-up templates (frames,
# walls, shop labels) holding different pictures: their matches lie on the
# template and the centres do not agree. Calibration: false template matches
# 0.02-0.27, true copies 0.74 and up. MIN_NCC also vetoes hash-rule matches.
FEATURE_SIDE = 640
ORB_FEATURES = 1500
MIN_INLIERS = 40
MIN_COVER = 0.36
MIN_NCC = 0.5
FLAT_LUMA_STD = 6.0      # below this an image is near-uniform: exact matches only
# Colour: a pair is two different images when one copy is clearly in colour and
# the other has less than COLOUR_RATIO of its colourfulness (black and white,
# sepia, toned). Real colour duplicate pairs measured >= 0.36, toned copies <= 0.06.
COLOUR_MIN = 8.0
COLOUR_RATIO = 0.25

# --- ranking -----------------------------------------------------------------
GUARD = 3.0              # a lower-tier copy wins if it scores this much higher
TIE = 0.1                # scores closer than this are equal


# ---------------------------------------------------------------------------
# k2prep scoring. Copied from k2prep.py (T:/claude/github2/k2prep), two-pass
# rendered scoring; keep the two in step. Q (JPEG quality) is not used: k2prep
# reports it but keeps it out of the composite.
# ---------------------------------------------------------------------------

TIERS = [1024, 768, 512]
UPSCALE_TOLERANCE = 1.15
RESO_STEPS = 16
EPS = 1e-6

AR_FAMILIES = ["9:16", "2:3", "4:5", "1:1", "5:4", "3:2", "16:9"]
AR_NOMINAL = [0.5647, 0.6667, 0.8028, 1.0000, 1.2456, 1.5000, 1.7708]

B_RENDERED_BANDS = [(0.010, 10), (0.014, 9), (0.018, 8), (0.023, 7), (0.030, 6),
                    (0.038, 5), (0.048, 4), (0.060, 3), (0.075, 2)]
D_RENDERED_BANDS = [(0.140, 10), (0.110, 9), (0.085, 8), (0.065, 7), (0.048, 6),
                    (0.035, 5), (0.025, 4), (0.017, 3), (0.010, 2)]
MIN_BLOCK_PERIOD = 3.0


def divisible_by(n: int, d: int) -> int:
    return n - n % d


@lru_cache(maxsize=None)
def generate_buckets(resolution: int, steps: int = RESO_STEPS):
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


def assign_family(src_ar: float) -> str:
    idx = min(range(len(AR_NOMINAL)), key=lambda i: abs(AR_NOMINAL[i] - src_ar))
    return AR_FAMILIES[idx]


def crop_dims(w: int, h: int, target_ar: float):
    if w / h > target_ar:
        cw, ch = int(round(h * target_ar)), h
    else:
        cw, ch = w, int(round(w / target_ar))
    return max(1, min(cw, w)), max(1, min(ch, h))


def crop_box(src_w: int, src_h: int, target_ar: float):
    cw, ch = crop_dims(src_w, src_h, target_ar)
    left = (src_w - cw) // 2
    top = int((src_h - ch) * (1 / 3 if target_ar < 1.0 else 1 / 2))
    return (left, top, left + cw, top + ch)


def assign_tier(src_w: int, src_h: int, family: str):
    for tier in TIERS:
        bw, bh = bucket_for(tier, family)
        cw, ch = crop_dims(src_w, src_h, bw / bh)
        if cw * ch >= (bw * bh) / (UPSCALE_TOLERANCE ** 2):
            return tier, (bw, bh), (cw, ch)
    return None


def to_rgb(img: Image.Image) -> Image.Image:
    """RGBA/LA/transparent P over white, 16-bit and float scaled, else RGB."""
    mode = img.mode
    if mode == "RGB":
        return img
    if mode in ("RGBA", "LA") or (mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    if mode == "I" or mode.startswith("I;16"):
        a = np.asarray(img).astype(np.float32) / 256.0
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "L").convert("RGB")
    if mode == "F":
        a = np.asarray(img, dtype=np.float32)
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) / max(hi - lo, EPS) * 255.0
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "L").convert("RGB")
    return img.convert("RGB")


def block_period_energy(luma: Image.Image, period: float):
    if period < MIN_BLOCK_PERIOD:
        return None
    arr = np.asarray(luma, dtype=np.float32)
    best = None
    for axis in (arr, arr.T):
        diff = np.abs(axis[:, 1:] - axis[:, :-1]).mean(axis=0)
        n = diff.size
        if n < 32 or period > n / 4:
            continue
        mean = float(diff.mean())
        if mean <= EPS:
            continue
        centred = diff - mean
        x = np.arange(n)
        amp = 2.0 * abs(complex(np.sum(centred * np.exp(-2j * math.pi * x / period)))) / n
        value = float(amp / mean)
        best = value if best is None else max(best, value)
    return best


def detail_ratio(luma: Image.Image) -> float:
    w, h = luma.size
    if w < 4 or h < 4:
        return 0.0
    small = luma.resize((max(1, w // 2), max(1, h // 2)), Image.BOX)
    back = small.resize((w, h), Image.BILINEAR)
    a = np.asarray(luma, dtype=np.float32)
    b = np.asarray(back, dtype=np.float32)
    hf = float(np.abs(a - b).mean())
    return float(hf / max(float(a.std()), EPS))


def band_score(value, bands, lower_is_better):
    for edge, s in bands:
        if (value <= edge) if lower_is_better else (value >= edge):
            return s
    return 1


def fine_score(value: float, bands, lower_is_better: bool) -> float:
    if lower_is_better:
        first_hi = bands[0][0]
        if value <= first_hi:
            frac = 1.0 - (value / first_hi if first_hi > 0 else 0.0)
            return bands[0][1] + min(0.999, max(0.0, frac))
        for i in range(1, len(bands)):
            hi, s = bands[i]
            if value <= hi:
                lo = bands[i - 1][0]
                return s + (hi - value) / max(hi - lo, EPS)
        last = bands[-1][0]
        return 1.0 + (min(0.999, last / value) if value > 0 else 0.0)
    first_lo = bands[0][0]
    if value >= first_lo:
        growth = first_lo / max(bands[1][0], EPS)
        span = math.log(growth) if growth > 1.0 else 1.0
        t = math.log(max(value, first_lo) / first_lo) / span
        return bands[0][1] + t / (1.0 + t)
    for i in range(1, len(bands)):
        lo, s = bands[i]
        if value >= lo:
            hi = bands[i - 1][0]
            return s + (value - lo) / max(hi - lo, EPS)
    last = bands[-1][0]
    return 1.0 + (min(0.999, value / last) if last > 0 else 0.0)


def score_file(path: str) -> dict:
    """k2prep rendered score at the bucket this copy reaches.

    A copy too small for 512 gets tier 0 and is scored at its own size, cropped
    to its family ratio but not resized (k2prep's rule for images below a tier:
    upscaling first would invent detail and flatter the result).
    """
    try:
        with Image.open(path) as im:
            qtables = getattr(im, "quantization", None)
            fmt = im.format or ""
            img = to_rgb(ImageOps.exif_transpose(im))
        w, h = img.size
        family = assign_family(w / h)
        fit = assign_tier(w, h, family)
        if fit:
            tier, bucket, crop = fit
            box = crop_box(w, h, bucket[0] / bucket[1])
            out = img.resize(bucket, resample=Image.LANCZOS, box=box)
        else:
            tier = 0
            bw, bh = bucket_for(TIERS[-1], family)
            box = crop_box(w, h, bw / bh)
            out = img.crop(box)
            crop = bucket = out.size
        luma = out.convert("L")
        d_ratio = detail_ratio(luma)
        scores = [band_score(d_ratio, D_RENDERED_BANDS, False)]
        fine = [fine_score(d_ratio, D_RENDERED_BANDS, False)]
        b_ratio = None
        if qtables:
            b_ratio = block_period_energy(luma, 8.0 * (bucket[0] / crop[0]))
            if b_ratio is not None:
                scores.append(band_score(b_ratio, B_RENDERED_BANDS, True))
                fine.append(fine_score(b_ratio, B_RENDERED_BANDS, True))
        return {"tier": tier, "bucket": list(bucket), "score": round(min(fine), 4),
                "d_ratio": round(d_ratio, 5), "b_ratio": None if b_ratio is None else round(b_ratio, 5),
                "format": fmt}
    except Exception as ex:  # noqa: BLE001 - one bad file must not stop the run
        return {"error": f"{type(ex).__name__}: {ex}"}


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def colourfulness(img: Image.Image) -> float:
    """RMS chroma left after removing the part that luma predicts.

    A toned monochrome image (sepia, blue, yellowed paper) has chroma that is a
    function of brightness, so this is near 0 for it; a colour image keeps its
    chroma. Plain chroma spread cannot tell a sepia print from a pale colour photo.
    """
    a = np.asarray(img.resize((48, 48), Image.BOX).convert("YCbCr"), dtype=np.float32)
    y = a[..., 0].ravel()
    x = np.stack([np.ones_like(y), y, y * y], 1)
    total = 0.0
    for c in (a[..., 1].ravel() - 128.0, a[..., 2].ravel() - 128.0):
        coef, *_ = np.linalg.lstsq(x, c, rcond=None)
        total += float(np.mean((c - x @ coef) ** 2))
    return math.sqrt(total)


def hash_file(path: str) -> dict:
    import imagehash
    try:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                sha.update(chunk)
        with Image.open(path) as im:
            w, h = im.size
            try:
                orientation = im.getexif().get(274, 1) or 1
            except Exception:  # noqa: BLE001
                orientation = 1
            animated = (getattr(im, "n_frames", 1) or 1) > 1
            im.draft("RGB", (512, 512))          # JPEG: decode at reduced scale
            img = to_rgb(ImageOps.exif_transpose(im))
        if orientation in (5, 6, 7, 8):
            w, h = h, w
        img.thumbnail((256, 256), Image.LANCZOS)
        luma = img.convert("L")
        return {
            "sha256": sha.hexdigest(), "w": w, "h": h, "animated": animated,
            "phash": str(imagehash.phash(luma)), "dhash": str(imagehash.dhash(luma)),
            "luma_std": round(float(np.asarray(luma.resize((32, 32), Image.BOX), dtype=np.float32).std()), 2),
            "colour": round(colourfulness(img), 2),
        }
    except Exception as ex:  # noqa: BLE001
        return {"error": f"{type(ex).__name__}: {ex}"}


# ---------------------------------------------------------------------------
# Scanning and cache
# ---------------------------------------------------------------------------

class Item:
    __slots__ = ("idx", "root_idx", "root", "rel", "path", "size", "mtime", "caption", "data")

    def __init__(self, idx, root_idx, root, rel, path, size, mtime):
        self.idx, self.root_idx, self.root, self.rel, self.path = idx, root_idx, root, rel, path
        self.size, self.mtime = size, mtime
        self.caption = None
        self.data = {}


def scan(roots, excludes):
    """All images under the roots; folders starting with "_" and excluded names are skipped."""
    items = []
    excl = {e.casefold() for e in excludes}
    for ri, root in enumerate(roots):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames
                                 if not d.startswith("_") and d.casefold() not in excl)
            stems = {}
            for fn in filenames:
                stem, ext = os.path.splitext(fn)
                if ext.lower() == ".txt":
                    stems.setdefault(stem.casefold(), {})["txt"] = fn
            for fn in sorted(filenames):
                if os.path.splitext(fn)[1].lower() not in IMAGE_EXTS:
                    continue
                p = Path(dirpath) / fn
                try:
                    st = p.stat()
                except OSError:
                    continue
                rel = p.relative_to(root).as_posix()
                it = Item(len(items), ri, root, rel, p, st.st_size, st.st_mtime_ns)
                txt = stems.get(os.path.splitext(fn)[0].casefold(), {}).get("txt")
                it.caption = Path(dirpath) / txt if txt else None
                items.append(it)
    return items


def load_cache(root: Path) -> dict:
    p = root / OUT_DIRNAME / HASHES_NAME
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("version") == CACHE_VERSION:
            return data.get("files", {})
    except (OSError, ValueError):
        pass
    return {}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def load_verified(roots) -> dict:
    """Feature-check results, keyed by the content of both images."""
    out = {}
    for root in roots:
        try:
            data = json.loads((root / OUT_DIRNAME / VERIFIED_NAME).read_text(encoding="utf-8"))
            if data.get("version") == CACHE_VERSION:
                out.update(data.get("pairs", {}))
        except (OSError, ValueError):
            pass
    return out


def save_verified(roots, items, cache) -> None:
    """Each root keeps the results for pairs that involve one of its images."""
    for ri, root in enumerate(roots):
        mine = {it.data["sha256"][:20] for it in items if it.root_idx == ri and "sha256" in it.data}
        pairs = {k: v for k, v in cache.items() if k.split("|")[0] in mine or k.split("|")[1] in mine}
        write_json(root / OUT_DIRNAME / VERIFIED_NAME,
                   {"version": CACHE_VERSION, "tool": "dedup.py", "pairs": pairs})


def save_caches(roots, items) -> None:
    for ri, root in enumerate(roots):
        files = {it.rel: {"size": it.size, "mtime_ns": it.mtime, **it.data}
                 for it in items if it.root_idx == ri and it.data}
        write_json(root / OUT_DIRNAME / HASHES_NAME,
                   {"version": CACHE_VERSION, "tool": "dedup.py",
                    "written": datetime.now().isoformat(timespec="seconds"), "files": files})


class Progress:
    """"label: done/total" on one line in a console; at most one line per
    PROGRESS_EVERY seconds when the output is redirected."""
    PROGRESS_EVERY = 5.0

    def __init__(self, label, total):
        self.label, self.total, self.t0, self.last = label, total, time.time(), 0.0
        self.tty = sys.stdout.isatty()

    def update(self, done):
        now = time.time()
        if done < self.total and now - self.last < (0.2 if self.tty else self.PROGRESS_EVERY):
            return
        self.last = now
        if self.tty:
            print(f"\r  {self.label}: {done}/{self.total}", end="", flush=True)
        elif done < self.total:
            print(f"  {self.label}: {done}/{self.total}", flush=True)

    def finish(self):
        line = f"  {self.label}: {self.total}/{self.total} ({time.time() - self.t0:.0f}s)"
        print(("\r" if self.tty else "") + line, flush=True)


def run_pool(fn, items, workers, label):
    """Run fn(path) for each item in worker processes; returns results in order."""
    if not items:
        return []
    prog = Progress(label, len(items))
    out = [None] * len(items)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, res in enumerate(ex.map(fn, [str(it.path) for it in items], chunksize=4), 1):
            out[k - 1] = res
            prog.update(k)
    prog.finish()
    return out


# ---------------------------------------------------------------------------
# Matching and grouping
# ---------------------------------------------------------------------------

def hash_rule_ok(p, d, rules):
    return any(p <= mp and d <= md for mp, md in rules)


def find_candidates(items, max_p, max_d, per_image):
    """(i, j, phash distance, dhash distance) for i < j: pairs within the window,
    each image limited to its per_image nearest (None: no limit). Plain, animated
    and colour-vs-monochrome pairs are left out."""
    cand = [it for it in items if not it.data.get("animated") and it.data["luma_std"] >= FLAT_LUMA_STD]
    n = len(cand)
    if n < 2:
        return []
    ph = np.array([int(it.data["phash"], 16) for it in cand], dtype=np.uint64)
    dh = np.array([int(it.data["dhash"], 16) for it in cand], dtype=np.uint64)
    col = np.array([it.data["colour"] for it in cand], dtype=np.float32)
    ids = np.array([it.idx for it in cand])
    found = {}
    # The XOR blocks are block x n uint64 each; keep them near 160 MB whatever n is.
    block = max(16, min(1024, 10_000_000 // n))
    for start in range(0, n, block):
        stop = min(start + block, n)
        dp = np.bitwise_count(ph[start:stop, None] ^ ph[None, :]).astype(np.int16)
        dd = np.bitwise_count(dh[start:stop, None] ^ dh[None, :]).astype(np.int16)
        ok = (dp <= max_p) & (dd <= max_d)
        ok[np.arange(stop - start), np.arange(start, stop)] = False
        if per_image is not None and per_image < n:
            score = np.where(ok, dp + dd, 9999)
            top = np.argpartition(score, per_image, axis=1)[:, :per_image]
            keep = np.zeros_like(ok)
            np.put_along_axis(keep, top, True, axis=1)
            ok &= keep
        rows, cols = np.nonzero(ok)
        gi = rows + start
        ca, cb = col[gi], col[cols]
        hi, lo = np.maximum(ca, cb), np.minimum(ca, cb)
        same_colour = ~((hi >= COLOUR_MIN) & (lo < COLOUR_RATIO * hi))
        for a, b, p, d in zip(gi[same_colour], cols[same_colour],
                              dp[rows, cols][same_colour], dd[rows, cols][same_colour]):
            i, j = int(ids[a]), int(ids[b])
            found[(min(i, j), max(i, j))] = (int(p), int(d))
    return [(i, j, p, d) for (i, j), (p, d) in found.items()]


# --- feature check, run in worker processes --------------------------------

_FEATURES = {}           # per-process cache: path -> (points, descriptors, gray)
_FEATURES_MAX = 160


def image_features(path):
    import cv2
    hit = _FEATURES.pop(path, None)
    if hit is not None:
        _FEATURES[path] = hit                        # most recently used last
        return hit
    with Image.open(path) as im:
        im.draft("L", (FEATURE_SIDE * 2, FEATURE_SIDE * 2))
        img = to_rgb(ImageOps.exif_transpose(im)).convert("L")
    img.thumbnail((FEATURE_SIDE, FEATURE_SIDE), Image.LANCZOS)
    gray = np.asarray(img)
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES, scaleFactor=1.2, nlevels=8, fastThreshold=10)
    kp, des = orb.detectAndCompute(gray, None)
    pts = np.array([k.pt for k in kp], dtype=np.float32) if kp else np.zeros((0, 2), np.float32)
    feat = (pts, des, gray)
    _FEATURES[path] = feat
    while len(_FEATURES) > _FEATURES_MAX:
        _FEATURES.pop(next(iter(_FEATURES)))
    return feat


def compare_features(fa, fb):
    """-> [inliers, cover, ncc]; ncc is None when no homography could be fitted."""
    import cv2
    (pa, da, ga), (pb, db, gb) = fa, fb
    if da is None or db is None or len(pa) < 10 or len(pb) < 10:
        return [0, 0.0, None]
    good = [m[0] for m in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
            if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < 8:
        return [0, 0.0, None]
    src = pa[[m.queryIdx for m in good]]
    dst = pb[[m.trainIdx for m in good]]
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
    if H is None:
        return [0, 0.0, None]
    inl = mask.ravel().astype(bool)
    n = int(inl.sum())

    def cover(pts, gray):
        if len(pts) < 2:
            return 0.0
        (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
        return float((x1 - x0) * (y1 - y0) / (gray.shape[0] * gray.shape[1]))
    cov = min(cover(src[inl], ga), cover(dst[inl], gb))

    # Warp A onto B and correlate the central half of B where A lands.
    h, w = gb.shape
    warped = cv2.warpPerspective(ga, H, (w, h))
    valid = cv2.warpPerspective(np.full_like(ga, 255), H, (w, h)) > 0
    win = np.zeros_like(valid)
    win[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True
    m = valid & win
    if m.sum() < 500:
        return [n, round(cov, 3), 0.0]
    x = warped[m].astype(np.float32)
    y = gb[m].astype(np.float32)
    x -= x.mean()
    y -= y.mean()
    ncc = float((x * y).sum() / max(math.sqrt(float((x * x).sum()) * float((y * y).sum())), 1e-6))
    return [n, round(cov, 3), round(ncc, 3)]


def verify_task(task):
    """(anchor path, [(other path, key), ...]) -> [(key, [inliers, cover, ncc]), ...]"""
    anchor, others = task
    out = []
    try:
        fa = image_features(anchor)
    except Exception:  # noqa: BLE001 - unreadable here: no verdict, the hash rule decides
        return [(key, None) for _, key in others]
    for path, key in others:
        try:
            out.append((key, compare_features(fa, image_features(path))))
        except Exception:  # noqa: BLE001
            out.append((key, None))
    return out


def pair_key(a, b):
    x, y = sorted((a.data["sha256"][:20], b.data["sha256"][:20]))
    return f"{x}|{y}"


def verify_pairs(items, pairs, cache, workers):
    """Feature check for (i, j) pairs; results cached by content in `cache`."""
    todo = {}
    for i, j in pairs:
        key = pair_key(items[i], items[j])
        if key not in cache:
            todo.setdefault(i, []).append((str(items[j].path), key))
    n_pairs = sum(len(v) for v in todo.values())
    print(f"feature check: {n_pairs} pair(s), {len(pairs) - n_pairs} from cache")
    if not n_pairs:
        return
    tasks = [(str(items[i].path), others) for i, others in sorted(todo.items())]
    prog, done = Progress("checked", n_pairs), 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(verify_task, tasks, chunksize=1):
            for key, v in res:
                if v is not None:
                    cache[key] = v
            done += len(res)
            prog.update(done)
    prog.finish()


def find_edges(items, mode, cache, workers):
    """{(i, j): (kind, details)} for every pair judged a copy."""
    edges = {}
    by_sha = {}
    for it in items:
        by_sha.setdefault(it.data["sha256"], []).append(it.idx)
    for group in by_sha.values():
        for a in group:
            for b in group:
                if a < b:
                    edges[(a, b)] = ("exact", {})
    rules = MATCH_RULES[mode]
    if not rules:
        return edges
    features = mode == "loose"
    if features:
        cands = find_candidates(items, CAND_MAX_PHASH, CAND_MAX_DHASH, CAND_PER_IMAGE)
    else:
        cands = find_candidates(items, max(r[0] for r in rules), max(r[1] for r in rules), None)
    cands = [c for c in cands if (c[0], c[1]) not in edges]
    print(f"{len(cands)} candidate pair(s) from the hashes")
    if features:
        verify_pairs(items, [(i, j) for i, j, _, _ in cands], cache, workers)
    for i, j, p, d in cands:
        v = cache.get(pair_key(items[i], items[j])) if features else None
        details = {"phash": p, "dhash": d}
        if v is not None:
            details.update(inliers=v[0], cover=v[1], ncc=v[2])
        if v is not None and v[2] is not None and v[2] < MIN_NCC:
            continue                                  # the centres disagree: not the same picture
        if hash_rule_ok(p, d, rules):
            edges[(i, j)] = ("hash", details)
        elif v is not None and v[0] >= MIN_INLIERS and v[1] >= MIN_COVER and (v[2] or 0) >= MIN_NCC:
            edges[(i, j)] = ("features", details)
    return edges


def components(nodes, adj):
    seen, out = set(), []
    for n in nodes:
        if n in seen:
            continue
        stack, comp = [n], []
        seen.add(n)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj.get(x, ()):
                if y in nodes and y not in seen:
                    seen.add(y)
                    stack.append(y)
        out.append(comp)
    return out


def pick_keeper(members, items):
    """Best copy: largest tier, then score; a lower-tier copy that scores GUARD
    or more above wins instead. Ties (score within TIE): caption, then oldest
    modification time, then path order. Returns (keeper, reason)."""
    def s(i):
        return items[i].data["score_info"]

    def tier_score(i):
        return (s(i)["tier"], s(i)["score"])

    top = max(members, key=tier_score)
    reason = "larger bucket" if len({s(i)["tier"] for i in members}) > 1 else "higher score"
    while True:
        better = [i for i in members
                  if s(i)["tier"] < s(top)["tier"] and s(i)["score"] >= s(top)["score"] + GUARD]
        if not better:
            break
        top = max(better, key=tier_score)
        reason = f"guard: smaller copy scores {GUARD:g}+ points higher"
    ties = [i for i in members
            if s(i)["tier"] == s(top)["tier"] and abs(s(i)["score"] - s(top)["score"]) < TIE]
    if len(ties) > 1:
        def tie_key(i):
            it = items[i]
            return (it.caption is None, it.mtime, it.root_idx, it.rel.casefold())
        top = min(ties, key=tie_key)
        others = [items[i] for i in ties if i != top]
        if items[top].caption is not None and any(o.caption is None for o in others):
            reason = "equal quality; has a caption"
        else:
            reason = "equal quality; oldest file"
    return top, reason


def plan_groups(items, edges, scorer):
    """Groups of copies: every image connected through matches is in the group,
    and all but the keeper move. Images that could not be scored stay in place
    and do not connect others."""
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    scorer(sorted(adj))
    ok = {i for i in adj if "error" not in items[i].data["score_info"]}
    adj = {i: {j for j in nb if j in ok} for i, nb in adj.items() if i in ok}
    groups = []
    for comp in components(set(adj), adj):
        if len(comp) < 2:
            continue
        keeper, reason = pick_keeper(comp, items)
        # steps from the keeper, for the plan: 1 = matches the keeper directly
        steps, frontier = {keeper: 0}, [keeper]
        while frontier:
            nxt = []
            for x in frontier:
                for y in adj[x]:
                    if y not in steps:
                        steps[y] = steps[x] + 1
                        nxt.append(y)
            frontier = nxt
        losers = sorted(i for i in comp if i != keeper)
        groups.append((keeper, losers, reason, steps))
    return groups


# ---------------------------------------------------------------------------
# Moving
# ---------------------------------------------------------------------------

def free_dest(dest: Path, stem_peer: Path | None = None) -> Path:
    """dest, or dest with " (N)" before the extension if taken."""
    if not dest.exists() and (stem_peer is None or not stem_peer.exists()):
        return dest
    n = 1
    while True:
        cand = dest.with_name(f"{dest.stem} ({n}){dest.suffix}")
        peer = stem_peer.with_name(f"{dest.stem} ({n}){stem_peer.suffix}") if stem_peer else None
        if not cand.exists() and (peer is None or not peer.exists()):
            return cand
        n += 1


def move_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def caption_shared(it, items, moving) -> bool:
    """True if another image with the same stem stays in the folder; its caption stays too."""
    if it.caption is None:
        return False
    return any(o.caption == it.caption and o.idx != it.idx and o.idx not in moving for o in items)


def describe(it):
    s = it.data.get("score_info", {})
    return {"path": str(it.path), "w": it.data.get("w"), "h": it.data.get("h"),
            "tier": s.get("tier"), "score": s.get("score"), "caption": it.caption is not None,
            "mtime": datetime.fromtimestamp(it.mtime / 1e9).isoformat(timespec="seconds"),
            "bytes": it.size}


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------

def undo(roots) -> int:
    failed = 0
    for root in roots:
        restored = skipped = 0
        log = root / OUT_DIRNAME / MOVES_NAME
        if not log.is_file():
            print(f"{root}: no {MOVES_NAME}, nothing to undo")
            continue
        entries = [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for e in reversed(entries):
            src, dst = Path(e["from"]), Path(e["to"])
            if dst.exists() and not src.exists():
                move_file(dst, src)
                restored += 1
            else:
                skipped += 1
                print(f"  not restored (moved back already, or the original path is taken): {src}")
        log.rename(log.with_name(f"moves-undone-{datetime.now():%Y%m%d-%H%M%S}.jsonl"))
        print(f"{root}: restored {restored} file(s)")
        failed += skipped
    return 0 if not failed else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", help="folders to scan recursively; all form one pool")
    ap.add_argument("--match", choices=list(MATCH_RULES), default="loose",
                    help="loose (default): hashes plus a feature check; finds other scans, crops, "
                         "borders, frames, mock-ups and watermarks. strict: resized and recompressed "
                         "copies, hashes only. exact: identical files only")
    ap.add_argument("--dry-run", action="store_true", help="write hashes.json and plan.json, move nothing")
    ap.add_argument("--undo", action="store_true", help="move the files of the last run back")
    ap.add_argument("--exclude", action="append", default=[], metavar="NAME",
                    help=f"skip folders with this name (repeatable); always skipped: names starting "
                         f"with _ and {', '.join(DEFAULT_EXCLUDES)}")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="worker processes (default: CPU count - 1)")
    args = ap.parse_args(argv)

    roots = []
    for f in args.folders:
        p = Path(f).resolve()
        if not p.is_dir():
            ap.error(f"not a folder: {f}")
        roots.append(p)
    for a in roots:
        for b in roots:
            if a != b and b.is_relative_to(a):
                ap.error(f"{b} is inside {a}; give only the outer folder")
    if len(set(roots)) != len(roots):
        ap.error("the same folder is given twice")

    if args.undo:
        return undo(roots)

    t0 = time.time()
    items = scan(roots, DEFAULT_EXCLUDES + args.exclude)
    print(f"{len(items)} image(s) in {len(roots)} folder(s)")
    if not items:
        return 0

    # 1. hashes, from cache where size and mtime are unchanged
    caches = [load_cache(r) for r in roots]
    todo = []
    for it in items:
        c = caches[it.root_idx].get(it.rel)
        if c and c.get("size") == it.size and c.get("mtime_ns") == it.mtime and "phash" in c:
            it.data = {k: v for k, v in c.items() if k not in ("size", "mtime_ns")}
        else:
            todo.append(it)
    print(f"hashing {len(todo)} new or changed file(s), {len(items) - len(todo)} from cache")
    for it, res in zip(todo, run_pool(hash_file, todo, args.workers, "hashed")):
        it.data = res
    errors = [it for it in items if "error" in it.data]
    good = [it for it in items if "error" not in it.data]
    for it in errors:
        print(f"  unreadable, skipped: {it.path}: {it.data['error']}")

    # 2. matches and groups; scores only for images that have a copy
    for k, it in enumerate(good):
        it.idx = k
    verified = load_verified(roots)
    edges = find_edges(good, args.match, verified, args.workers)
    if args.match == "loose":
        save_verified(roots, good, verified)
    kinds = {}
    for kind, _ in edges.values():
        kinds[kind] = kinds.get(kind, 0) + 1
    print(f"{len(edges)} matching pair(s)" +
          (": " + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())) if kinds else ""))

    def scorer(idxs):
        need = [good[i] for i in idxs if "score_info" not in good[i].data]
        # Identical bytes give identical scores: render each content once.
        first = {}
        for it in need:
            first.setdefault(it.data["sha256"], it)
        uniq = list(first.values())
        print(f"scoring {len(need)} image(s) that have duplicates "
              f"({len(idxs) - len(need)} from cache, {len(need) - len(uniq)} identical copies)")
        by_sha = dict(zip((it.data["sha256"] for it in uniq),
                          run_pool(score_file, uniq, args.workers, "scored")))
        for it in need:
            it.data["score_info"] = by_sha[it.data["sha256"]]
        for i in idxs:
            if "error" in good[i].data["score_info"]:
                print(f"  could not score, left in place: {good[i].path}: {good[i].data['score_info']['error']}")

    groups = plan_groups(good, edges, scorer)
    save_caches(roots, items)

    # 3. plan
    moving = {i for _, losers, _, _ in groups for i in losers}
    plan = {"written": datetime.now().isoformat(timespec="seconds"), "match": args.match,
            "folders": [str(r) for r in roots], "groups": []}
    caption_lost = []
    for keeper, losers, reason, steps in sorted(groups, key=lambda g: (good[g[0]].root_idx, good[g[0]].rel)):
        k = good[keeper]
        entry = {"keep": describe(k), "reason": reason, "move": []}
        for i in losers:
            it = good[i]
            p = int(bin(int(k.data["phash"], 16) ^ int(it.data["phash"], 16)).count("1"))
            d = int(bin(int(k.data["dhash"], 16) ^ int(it.data["dhash"], 16)).count("1"))
            direct = edges.get((min(keeper, i), max(keeper, i)))
            entry["move"].append({**describe(it), "exact": it.data["sha256"] == k.data["sha256"],
                                  "phash_distance": p, "dhash_distance": d,
                                  "steps_from_kept": steps.get(i),
                                  "match": direct[0] if direct else "through other copies",
                                  **({"features": {x: direct[1][x] for x in ("inliers", "cover", "ncc")
                                                   if x in direct[1]}} if direct and "inliers" in direct[1] else {})})
            if it.caption is not None and k.caption is None:
                caption_lost.append((k, it))
        plan["groups"].append(entry)
    for root in roots:
        write_json(root / OUT_DIRNAME / PLAN_NAME, plan)

    n_moves = len(moving)
    n_bytes = sum(good[i].size for i in moving)
    print(f"{len(groups)} group(s); {n_moves} worse cop{'y' if n_moves == 1 else 'ies'} to move, "
          f"{n_bytes / 1e6:.1f} MB")
    sizes = {}
    for g in groups:
        sizes[len(g[1]) + 1] = sizes.get(len(g[1]) + 1, 0) + 1
    if sizes:
        print("copies per group: " + ", ".join(f"{n} x{c}" for n, c in sorted(sizes.items())))
    guarded = [g for g in groups if g[2].startswith("guard")]
    if guarded:
        print(f"{len(guarded)} group(s) keep a smaller copy because the larger one scores much lower")
    if caption_lost:
        print(f"{len(caption_lost)} moved cop{'y' if len(caption_lost) == 1 else 'ies'} take a caption "
              f"along while the kept image has none:")
        for k, it in caption_lost[:20]:
            print(f"   {it.path}  (kept: {k.path})")
        if len(caption_lost) > 20:
            print(f"   ... see {PLAN_NAME}")

    if args.dry_run:
        print(f"Dry run: nothing moved. Plan: {roots[0] / OUT_DIRNAME / PLAN_NAME}")
        return 0
    if not groups:
        print(f"Nothing to move. Done in {time.time() - t0:.0f}s.")
        return 0

    # 4. move
    moved = {r: [0, 0, 0] for r in roots}          # images, captions, bytes
    logs = {}
    try:
        for keeper, losers, reason, _ in groups:
            for i in losers:
                it = good[i]
                if not it.path.exists():
                    print(f"  gone before the move, skipped: {it.path}")
                    continue
                out = it.root / OUT_DIRNAME
                cap = it.caption if it.caption and it.caption.exists() and not caption_shared(it, good, moving) else None
                dst = free_dest(out / it.rel, (out / it.rel).with_suffix(".txt") if cap else None)
                log = logs.get(it.root)
                if log is None:
                    log = logs[it.root] = open(out / MOVES_NAME, "a", encoding="utf-8")
                stamp = datetime.now().isoformat(timespec="seconds")
                move_file(it.path, dst)
                log.write(json.dumps({"time": stamp, "from": str(it.path), "to": str(dst),
                                      "kept": str(good[keeper].path), "reason": reason},
                                     ensure_ascii=False) + "\n")
                moved[it.root][0] += 1
                moved[it.root][2] += it.size
                if cap:
                    cdst = dst.with_suffix(".txt")
                    move_file(cap, cdst)
                    log.write(json.dumps({"time": stamp, "from": str(cap), "to": str(cdst),
                                          "kept": str(good[keeper].path), "reason": "caption of a moved copy"},
                                         ensure_ascii=False) + "\n")
                    moved[it.root][1] += 1
                log.flush()
    finally:
        for log in logs.values():
            log.close()

    for root, (n_img, n_cap, n_b) in moved.items():
        print(f"{root}: moved {n_img} image(s) and {n_cap} caption(s), {n_b / 1e6:.1f} MB "
              f"-> {root / OUT_DIRNAME}")
    print(f"Done in {time.time() - t0:.0f}s. Undo with --undo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
