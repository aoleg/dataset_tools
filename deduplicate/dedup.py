#!/usr/bin/env python3
"""
Find duplicate images with perceptual hashes and move the worse copies out.

Every folder given is scanned recursively, and all of them form one pool. Two
images are copies when their perceptual hashes are close, or when local image
features (ORB) show the same picture under another crop, border, scale or a
mock-up. Copies connected through such matches form one group. In each group the
best copy stays; every other copy, and its .txt caption, moves to
<root>/_duplicates/<same relative path>, where <root> is the folder given on the
command line that contains it. If the kept copy has no caption and a moved copy
has one, that caption is also copied next to the kept copy, under its name.
No prompt.

Quality is judged the k2prep way: each copy is scored as the trainer would see it,
cropped and resized to the 512, 768 or 1024 bucket it reaches. A copy that reaches
a larger bucket wins, unless it scores GUARD or more points lower. Equal copies:
one in a sorted folder wins, then the one with a caption, then the oldest file.

Sorted folders (--sorted) are curated trees. A better unsorted copy moves into the
sorted copy's place under the sorted name (promotion), when it is a bucket up or
PROMOTE_MARGIN points better and matches by hash. Sorted captions are never moved
or overwritten. The same picture in several sorted folders keeps every folder's
slot and gives each the best copy (--sorted-copies keep), or keeps one (one).
Copies that match by features only are listed for review and left alone.

Usage:    python dedup.py <folder> [<folder> ...] [--sorted FOLDER] [--sorted-copies keep|one]
          [--promote-margin X] [--match loose|strict|exact] [--dry-run] [--undo]
          [--exclude NAME] [--workers N] [--gpu] [--review]
Install:  pip install Pillow numpy imagehash opencv-python-headless
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
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
# Folders never scanned, at any depth: the output folders of the dataset tools.
# _duplicates is this tool's own, _prep is k2prep's, _classify and _embeddings
# are classify's, masks and faces are face_masks' (near-identical images by
# design), _backup is remove_borders' (the originals of cropped images). Any
# other folder is scanned, whatever its name; --exclude adds names.
DEFAULT_EXCLUDES = ["_duplicates", "_prep", "_classify", "_embeddings", "masks", "faces", "_backup"]

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
# 0.02-0.27, true copies 0.74 and up. MIN_NCC also vetoes hash-rule matches,
# and so does a feature check that finds no homography at all: two pictures
# that hash alike but share no geometry are two pictures (on 2,000 downloaded
# posters all nine such pairs were, for example different state emblems
# printed on one card template). An unwarped centre NCC cannot replace the homography: it
# reached 0.81 on different emblems and fell to 0.26 on a true copy with a
# small border.
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
# Sorted folders (--sorted): an unsorted copy displaces a sorted copy of the same
# picture only when it reaches a larger bucket, or scores PROMOTE_MARGIN or more
# higher in the same bucket, and only on an exact or hash match. Calibrated on
# 59 mixed groups of one real collection: at 1.0, 23 promotions, 18 of them a
# bucket step; the same-bucket cases below 1.0 were resolution bumps inside the
# 1024 bucket that the bucketed score cannot see. Reporting only until phase 1.
PROMOTE_MARGIN = 1.0


# ---------------------------------------------------------------------------
# k2prep scoring. Copied from k2prep.py (../k2prep), two-pass
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
    __slots__ = ("idx", "root_idx", "root", "rel", "path", "size", "mtime", "caption", "data", "role")

    def __init__(self, idx, root_idx, root, rel, path, size, mtime, role=0):
        self.idx, self.root_idx, self.root, self.rel, self.path = idx, root_idx, root, rel, path
        self.size, self.mtime = size, mtime
        self.caption = None
        self.data = {}
        self.role = role                 # 0 unsorted (raw downloads), 1 sorted (curated tree)


def scan(roots, excludes, roles=None):
    """All images under the roots, at any depth; folders with an excluded name are skipped.
    roles: one entry per root, 0 unsorted or 1 sorted; all unsorted when None."""
    items = []
    excl = {e.casefold() for e in excludes}
    for ri, root in enumerate(roots):
        role = roles[ri] if roles else 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d.casefold() not in excl)
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
                it = Item(len(items), ri, root, rel, p, st.st_size, st.st_mtime_ns, role)
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


def write_cache(root: Path, files: dict) -> None:
    write_json(root / OUT_DIRNAME / HASHES_NAME,
               {"version": CACHE_VERSION, "tool": "dedup.py",
                "written": datetime.now().isoformat(timespec="seconds"), "files": files})


def save_caches(roots, items) -> None:
    for ri, root in enumerate(roots):
        write_cache(root, {it.rel: {"size": it.size, "mtime_ns": it.mtime, **it.data}
                           for it in items if it.root_idx == ri and it.data})


def carry_cache(root: Path, entries) -> int:
    """Add cache entries for files that a run placed under `root` (promoted and
    synced copies), so the next run does not hash and score them again.
    entries: (path, data) pairs; data is the hashed item's data."""
    files = load_cache(root)
    n = 0
    for path, data in entries:
        try:
            st = path.stat()
        except OSError:
            continue
        files[path.relative_to(root).as_posix()] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, **data}
        n += 1
    if n:
        write_cache(root, files)
    return n


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

# ---------------------------------------------------------------------------
# Feature check. Features are extracted once per image, in a parallel pass, and
# shared with the pair workers through memory-mapped files in a temporary
# folder. The centre check needs the grey thumbnails, so it runs only for pairs
# that can still become a match: hash rule met, or enough inliers and cover.
# Each worker pins OpenCV to one thread; the pool provides the parallelism.
# ---------------------------------------------------------------------------

def gray_thumb(path):
    """The grey thumbnail (longest side FEATURE_SIDE) that features are computed on."""
    with Image.open(path) as im:
        im.draft("L", (FEATURE_SIDE * 2, FEATURE_SIDE * 2))
        img = to_rgb(ImageOps.exif_transpose(im)).convert("L")
    img.thumbnail((FEATURE_SIDE, FEATURE_SIDE), Image.LANCZOS)
    return np.asarray(img)


def extract_task(path):
    """-> (points float32 [k, 2], descriptors uint8 [k, 32], (h, w)), or None when unreadable."""
    import cv2
    cv2.setNumThreads(1)
    try:
        gray = gray_thumb(path)
    except Exception:  # noqa: BLE001
        return None
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES, scaleFactor=1.2, nlevels=8, fastThreshold=10)
    kp, des = orb.detectAndCompute(gray, None)
    if des is None or len(kp) < 10:
        return np.zeros((0, 2), np.float32), np.zeros((0, 32), np.uint8), gray.shape
    return np.array([k.pt for k in kp], dtype=np.float32), des, gray.shape


class FeatureStore:
    """Keypoints and descriptors of the images in the feature check, written by
    the parent into files that every pair worker maps read-only."""

    def __init__(self, folder: Path, n: int):
        self.folder = folder
        self.pts = np.lib.format.open_memmap(folder / "pts.npy", mode="w+", dtype=np.float32,
                                             shape=(n, ORB_FEATURES, 2))
        self.des = np.lib.format.open_memmap(folder / "des.npy", mode="w+", dtype=np.uint8,
                                             shape=(n, ORB_FEATURES, 32))
        self.meta = np.full((n, 3), -1, np.int32)        # count (-1: unreadable), height, width

    def put(self, k: int, feat) -> None:
        pts, des, (h, w) = feat
        c = min(len(pts), ORB_FEATURES)
        self.pts[k, :c] = pts[:c]
        self.des[k, :c] = des[:c]
        self.meta[k] = (c, h, w)

    def finish(self, paths) -> None:
        self.pts.flush()
        self.des.flush()
        del self.pts, self.des
        np.save(self.folder / "meta.npy", self.meta)
        (self.folder / "paths.json").write_text(json.dumps(paths, ensure_ascii=False), encoding="utf-8")


_STORE = None            # per worker: (pts, des, meta, paths), mapped read-only
_GRAYS = {}              # per worker: store index -> grey thumbnail, most recently used last
_GRAYS_MAX = 64


def _open_store(folder):
    global _STORE
    import cv2
    cv2.setNumThreads(1)
    folder = Path(folder)
    _STORE = (np.load(folder / "pts.npy", mmap_mode="r"), np.load(folder / "des.npy", mmap_mode="r"),
              np.load(folder / "meta.npy"), json.loads((folder / "paths.json").read_text(encoding="utf-8")))


def gray_of(k: int):
    hit = _GRAYS.pop(k, None)
    if hit is None:
        hit = gray_thumb(_STORE[3][k])
    _GRAYS[k] = hit
    while len(_GRAYS) > _GRAYS_MAX:
        _GRAYS.pop(next(iter(_GRAYS)))
    return hit


def compare_indexed(a: int, b: int, need_ncc: bool, matches=None):
    """-> [inliers, cover, ncc] for two store indices, or None when an image was
    unreadable (no verdict). ncc is None when no homography could be fitted, or
    when the pair cannot become a match anyway and the centre check was skipped.
    matches: (query indices, train indices) already matched on the GPU, else
    the descriptors are matched here."""
    import cv2
    pts, des, meta, _ = _STORE
    (ca, ha, wa), (cb, hb, wb) = meta[a], meta[b]
    if ca < 0 or cb < 0:
        return None
    if ca < 10 or cb < 10:
        return [0, 0.0, None]
    pa, pb = np.asarray(pts[a, :ca]), np.asarray(pts[b, :cb])
    if matches is None:
        da, db = np.ascontiguousarray(des[a, :ca]), np.ascontiguousarray(des[b, :cb])
        good = [m[0] for m in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
                if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        if len(good) < 8:
            return [0, 0.0, None]
        q, t = [m.queryIdx for m in good], [m.trainIdx for m in good]
    else:
        q, t = matches
        if len(q) < 8:
            return [0, 0.0, None]
    src, dst = pa[q], pb[t]
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
    if H is None:
        return [0, 0.0, None]
    inl = mask.ravel().astype(bool)
    n = int(inl.sum())

    def cover(p, h, w):
        if len(p) < 2:
            return 0.0
        (x0, y0), (x1, y1) = p.min(0), p.max(0)
        return float((x1 - x0) * (y1 - y0) / (h * w))
    cov = min(cover(src[inl], ha, wa), cover(dst[inl], hb, wb))
    if not need_ncc and not (n >= MIN_INLIERS and cov >= MIN_COVER):
        return [n, round(cov, 3), None]           # cannot become a match: no centre check needed

    # Warp A onto B and correlate the central half of B where A lands.
    ga, gb = gray_of(a), gray_of(b)
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
    """(anchor index, [(other index, key, need_ncc[, matches]), ...])
    -> [(key, [inliers, cover, ncc] or None), ...]"""
    anchor, others = task
    out = []
    for k, key, need, *rest in others:
        try:
            if rest and rest[0] is None:                 # matched on the GPU: fewer than 8 matches
                out.append((key, [0, 0.0, None] if _STORE[2][anchor][0] >= 0 and _STORE[2][k][0] >= 0 else None))
            else:
                out.append((key, compare_indexed(anchor, k, need, rest[0] if rest else None)))
        except Exception:  # noqa: BLE001 - no verdict: the hash rule decides
            out.append((key, None))
    return out


def gpu_match(folder: Path, pairs, batch: int = 256):
    """k=2 Hamming matching with the ratio test for every (a, b) store pair, on
    the GPU with torch. Returns one (query indices, train indices) pair of int32
    arrays per input pair, in order, or None when fewer than 8 matches remain.
    The distances are exact: the descriptor bits become 0/1 values in fp16 and
    every partial sum is a small integer. Needs about 50 KB of GPU memory per
    image for the descriptors, plus the batch."""
    import torch
    des = np.load(folder / "des.npy")                # the whole array: [n, ORB_FEATURES, 32] uint8
    meta = np.load(folder / "meta.npy")
    dev = torch.device("cuda")
    D = torch.from_numpy(des).to(dev)
    counts = torch.from_numpy(np.maximum(meta[:, 0], 0)).to(dev)
    shifts = torch.arange(8, device=dev, dtype=torch.uint8)
    col = torch.arange(ORB_FEATURES, device=dev)

    def bits(idx):
        x = D[idx]                                                      # [B, F, 32]
        return ((x.unsqueeze(-1) >> shifts) & 1).reshape(x.shape[0], ORB_FEATURES, 256).half()

    out = []
    prog = Progress("matched", len(pairs))
    for s in range(0, len(pairs), batch):
        chunk = pairs[s:s + batch]
        ia = torch.tensor([a for a, _ in chunk], device=dev)
        ib = torch.tensor([b for _, b in chunk], device=dev)
        A, B = bits(ia), bits(ib)
        # Hamming(a, b) = |a| + |b| - 2 a.b
        dist = A.sum(-1, keepdim=True) + B.sum(-1)[:, None, :] - 2 * torch.bmm(A, B.transpose(1, 2))
        ca, cb = counts[ia], counts[ib]
        dist.masked_fill_(col[None, None, :] >= cb[:, None, None], 1000.0)   # padded train rows
        d, idx = dist.topk(2, dim=2, largest=False)
        ok = (d[..., 0] < 0.75 * d[..., 1]) & (col[None, :] < ca[:, None]) & (cb[:, None] >= 2)
        nz = ok.nonzero()                                               # [M, 2]: (pair, query)
        tr = idx[nz[:, 0], nz[:, 1], 0]
        nz, tr = nz.cpu().numpy(), tr.cpu().numpy()
        for k in range(len(chunk)):
            sel = nz[:, 0] == k
            q = nz[sel, 1].astype(np.int32)
            out.append((q, tr[sel].astype(np.int32)) if len(q) >= 8 else None)
        prog.update(min(s + batch, len(pairs)))
    prog.finish()
    del D, A, B, dist
    torch.cuda.empty_cache()
    return out


def pair_key(a, b):
    x, y = sorted((a.data["sha256"][:20], b.data["sha256"][:20]))
    return f"{x}|{y}"


def verify_pairs(items, cands, rules, cache, workers, gpu=False):
    """Feature check for candidate pairs (i, j, phash distance, dhash distance);
    results are cached by content in `cache`. With gpu, the descriptors are
    matched on the GPU and the workers do the rest."""
    if gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("no CUDA device")
        except Exception as e:  # noqa: BLE001
            print(f"--gpu: torch with CUDA is not available in this venv ({e}); matching on the CPU")
            gpu = False
    todo = {}
    for i, j, p, d in cands:
        key = pair_key(items[i], items[j])
        if key not in cache:
            todo.setdefault(i, []).append((j, key, hash_rule_ok(p, d, rules)))
    n_pairs = sum(len(v) for v in todo.values())
    print(f"feature check: {n_pairs} pair(s), {len(cands) - n_pairs} from cache")
    if not n_pairs:
        return
    involved = sorted({i for i in todo} | {j for v in todo.values() for j, _, _ in v})
    index = {i: k for k, i in enumerate(involved)}
    tmp = Path(tempfile.mkdtemp(prefix="dedup-features-"))
    try:
        store = FeatureStore(tmp, len(involved))
        feats = run_pool(extract_task, [items[i] for i in involved], workers, "features")
        for k, f in enumerate(feats):
            if f is not None:
                store.put(k, f)
        store.finish([str(items[i].path) for i in involved])
        del feats
        anchors = sorted(todo.items())
        if gpu:
            flat = [(index[i], index[j]) for i, others in anchors for j, _, _ in others]
            matched = iter(gpu_match(tmp, flat))
            tasks = [(index[i], [(index[j], key, need, next(matched)) for j, key, need in others])
                     for i, others in anchors]
        else:
            tasks = [(index[i], [(index[j], key, need) for j, key, need in others]) for i, others in anchors]
        prog, done = Progress("checked", n_pairs), 0
        with ProcessPoolExecutor(max_workers=workers, initializer=_open_store, initargs=(str(tmp),)) as ex:
            for res in ex.map(verify_task, tasks, chunksize=1):
                for key, v in res:
                    if v is not None:
                        cache[key] = v
                done += len(res)
                prog.update(done)
        prog.finish()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if tmp.exists():                             # a worker may still hold a map for a moment
            time.sleep(1)
            shutil.rmtree(tmp, ignore_errors=True)


def find_edges(items, mode, cache, workers, gpu=False):
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
        verify_pairs(items, cands, rules, cache, workers, gpu)
    vetoed = {"no shared geometry": 0, "centres disagree": 0}
    for i, j, p, d in cands:
        v = cache.get(pair_key(items[i], items[j])) if features else None
        details = {"phash": p, "dhash": d}
        if v is not None:
            details.update(inliers=v[0], cover=v[1], ncc=v[2])
        if v is not None and (v[2] is None or v[2] < MIN_NCC):
            # No homography at all, or the centres disagree: not the same
            # picture, however close the hashes are.
            if hash_rule_ok(p, d, rules):
                vetoed["no shared geometry" if v[2] is None else "centres disagree"] += 1
            continue
        if hash_rule_ok(p, d, rules):
            edges[(i, j)] = ("hash", details)
        elif v is not None and v[0] >= MIN_INLIERS and v[1] >= MIN_COVER and v[2] >= MIN_NCC:
            edges[(i, j)] = ("features", details)
    n_vetoed = sum(vetoed.values())
    if n_vetoed:
        print(f"{n_vetoed} hash match(es) vetoed by the feature check: "
              + ", ".join(f"{n} {why}" for why, n in vetoed.items() if n))
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
    or more above wins instead. Ties (score within TIE): a copy in a sorted
    folder, then caption, then oldest modification time, then path order.
    Returns (keeper, reason)."""
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
            return (it.role == 0, it.caption is None, it.mtime, it.root_idx, it.rel.casefold())
        top = min(ties, key=tie_key)
        others = [items[i] for i in ties if i != top]
        if items[top].role == 1 and any(o.role == 0 for o in others):
            reason = "equal quality; in a sorted folder"
        elif items[top].caption is not None and any(o.caption is None for o in others):
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
    try:
        shutil.move(str(src), str(dst))
    except PermissionError:
        # across volumes shutil.move copies and then deletes the source, which
        # fails for a read-only file
        os.chmod(src, 0o666)
        shutil.move(str(src), str(dst))


def caption_shared(it, items, moving) -> bool:
    """True if another image with the same stem stays in the folder; its caption stays too."""
    if it.caption is None:
        return False
    return any(o.caption == it.caption and o.idx != it.idx and o.idx not in moving for o in items)


def describe(it):
    s = it.data.get("score_info", {})
    return {"path": str(it.path), "role": "sorted" if it.role == 1 else "unsorted",
            "w": it.data.get("w"), "h": it.data.get("h"),
            "tier": s.get("tier"), "score": s.get("score"), "caption": it.caption is not None,
            "mtime": datetime.fromtimestamp(it.mtime / 1e9).isoformat(timespec="seconds"),
            "bytes": it.size}


def role_report(keeper, losers, items, edges, margin):
    """What the sorted/unsorted rules would do with this group. Reporting only
    (phase 0 of the feature): nothing acts on the verdict yet.

    Mixed groups: the quality deltas between the best copy of each role, the
    match kind to the slot the keeper would take, and a verdict: promote,
    within margin, sorted copy kept, or review. Sorted-only groups: whether
    the copies share one directory, and whether every match is an exact or
    hash match (slot sync would apply) or a feature match (review).
    """
    members = [keeper] + list(losers)
    s = [i for i in members if items[i].role == 1]
    u = [i for i in members if items[i].role == 0]
    if not s:
        return {"roles": "unsorted"}

    def rank(i):
        si = items[i].data["score_info"]
        return (si["tier"], si["score"])

    def kind(a, b):
        e = edges.get((min(a, b), max(a, b)))
        return e[0] if e else "indirect"

    if not u:
        dirs = {items[i].path.parent for i in members}
        rep = {"roles": "sorted", "layout": "same directory" if len(dirs) == 1 else "cross directory"}
        kinds = {kind(keeper, i) for i in losers}
        if len(dirs) > 1 and kinds - {"exact", "hash"}:
            rep["note"] = "framing differs" if "features" in kinds else "indirect match"
        return rep

    best_s = max(s, key=rank)
    u_ref = keeper if items[keeper].role == 0 else max(u, key=rank)
    (ts, ss), (tu, su) = rank(best_s), rank(u_ref)
    rep = {"roles": "mixed", "best_sorted": str(items[best_s].path), "best_unsorted": str(items[u_ref].path),
           "tier_delta": tu - ts, "score_delta": round(su - ss, 2)}
    if items[keeper].role == 1:
        rep.update(match=kind(keeper, u_ref), verdict="sorted copy kept")
        return rep
    # The slot the keeper would take: the best sorted copy it matches directly
    # by hash or content; failing that, the best one it matches directly at all.
    direct = {i: kind(keeper, i) for i in s}
    clean = [i for i in s if direct[i] in ("exact", "hash")]
    any_direct = [i for i in s if direct[i] != "indirect"]
    slot = max(clean or any_direct or s, key=rank)
    m = direct[slot]
    rep.update(slot=str(items[slot].path), match=m)
    if m == "indirect":
        rep["verdict"] = "review: indirect match"
    elif m == "features":
        rep["verdict"] = "review: framing differs"
    elif tu < ts:
        rep["verdict"] = "review: guard keeper in a smaller bucket"
    elif tu > ts:
        rep["verdict"] = "promote: larger bucket"
    elif su - ss >= margin:
        rep["verdict"] = f"promote: score +{su - ss:.2f}"
    else:
        rep["verdict"] = "within margin"
    return rep


def caption_source(keeper, losers, items):
    """The moved copy whose caption the kept copy inherits, or None.

    Only when the kept copy has no caption. The best captioned copy by the same
    ranking (tier, score), then the oldest file.
    """
    if items[keeper].caption is not None:
        return None
    captioned = [items[i] for i in losers if items[i].caption is not None and items[i].caption.exists()]
    if not captioned:
        return None
    return min(captioned, key=lambda it: (-it.data["score_info"]["tier"], -it.data["score_info"]["score"],
                                          it.mtime, it.root_idx, it.rel.casefold()))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Names:
    """The file system as it will be after the planned moves: names freed by
    move-outs and names this plan has already given away. Destinations are
    chosen against it, so the dry run shows the final names and the real run
    uses the same ones."""

    def __init__(self):
        self.freed, self.taken = set(), set()
        self.copies = {}                 # planned caption copy: destination -> source

    def exists(self, p: Path) -> bool:
        return p in self.taken or (p not in self.freed and p.exists())

    def free_dest(self, dest: Path, stem_peer: Path | None = None) -> Path:
        """dest, or dest with " (N)" before the extension if taken; the chosen
        name and its peer are taken from then on."""
        cand, peer, n = dest, stem_peer, 0
        while self.exists(cand) or (peer is not None and self.exists(peer)):
            n += 1
            cand = dest.with_name(f"{dest.stem} ({n}){dest.suffix}")
            peer = stem_peer.with_name(f"{dest.stem} ({n}){stem_peer.suffix}") if stem_peer else None
        self.taken.add(cand)
        if peer is not None:
            self.taken.add(peer)
        return cand


def steps_from(start, members, edges):
    """Steps from `start` to every member over the group's matches."""
    adj = {m: set() for m in members}
    for a, b in edges:
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    steps, frontier = {start: 0}, [start]
    while frontier:
        nxt = []
        for x in frontier:
            for y in adj[x]:
                if y not in steps:
                    steps[y] = steps[x] + 1
                    nxt.append(y)
        frontier = nxt
    return steps


def decide_group(group, items, edges, with_roles, margin, policy="keep"):
    """What to do with one group. Without sorted folders: the keeper stays and
    the losers move, as always. With them (see README, "Sorted folders"):

    - the best copy of the group is the keeper by quality, except that an
      unsorted keeper within the promotion margin of the best sorted copy, or
      matching it by features only, does not displace it
    - a review verdict (framing differs, indirect match, guard keeper) skips
      the whole group: nothing moves
    - sorted slots: with policy "keep", one slot per directory survives, the
      best one there; with policy "one", a single slot survives in all sorted
      folders. Non-surviving sorted copies move out like any loser.
    - every surviving slot gets the best copy: the promoted unsorted keeper
      moves into its primary slot, the other slots get a copy of the best
      copy (slot sync) when their image matches it by hash. A slot whose image
      is byte-identical, or matches by features only or indirectly, is left
      as it is and reported.
    - every unsorted copy except a promoted keeper moves out

    Returns a dict: keeper (the best copy), losers (everything that moves out,
    including replaced slot images), slot_like (losers whose caption stays for
    the copy that replaces them), promote (the primary slot of a promoted
    keeper), sync (slots that get a copy), slots (the report per sorted copy),
    reason, steps, report, and skip (why nothing moves) for review groups.
    """
    keeper, losers, reason, steps = group
    d = {"keeper": keeper, "losers": list(losers), "reason": reason, "steps": steps,
         "report": {}, "skip": None, "promote": None, "sync": [], "slots": [], "slot_like": set()}
    if not with_roles:
        return d
    rep = role_report(keeper, losers, items, edges, margin)
    d["report"] = rep
    if rep["roles"] == "unsorted":
        return d
    members = [keeper] + list(losers)
    s = [i for i in members if items[i].role == 1]

    def rank(i):
        si = items[i].data["score_info"]
        return (si["tier"], si["score"])

    def kind(a, b):
        e = edges.get((min(a, b), max(a, b)))
        return e[0] if e else "indirect"

    best, primary = keeper, None
    if rep["roles"] == "mixed":
        v = rep["verdict"]
        if v.startswith("review"):
            d["skip"] = v
            return d
        if v == "within margin":
            best = max(s, key=rank)
            d["reason"] = f"sorted copy kept; the unsorted copy is within the margin ({rep['score_delta']:+.2f})"
        elif v.startswith("promote"):
            primary = next(i for i in s if str(items[i].path) == rep["slot"])
            d["promote"] = primary
            d["reason"] = "promoted into the sorted folder: " + v.split(": ", 1)[1]
    if policy == "one":
        survivors = {primary if primary is not None else best}
    else:
        by_dir = {}
        for i in s:
            by_dir.setdefault(items[i].path.parent, []).append(i)
        survivors = set()
        for idxs in by_dir.values():
            if primary in idxs:
                survivors.add(primary)
            elif best in idxs:
                survivors.add(best)
            else:
                survivors.add(max(idxs, key=rank))
    for i in sorted(s, key=lambda i: (items[i].root_idx, items[i].rel.casefold())):
        if i == best:
            action = "best copy; stays"
        elif i == primary:
            action = "takes the promoted copy"
        elif i not in survivors:
            action = "moves out: " + ("one slot per picture" if policy == "one" else "a better copy stays in this folder")
        else:
            k = kind(best, i)
            if k == "exact":
                action = "identical to the best copy; stays"
            elif k == "hash":
                action = "gets a copy of the best copy"
                d["sync"].append(i)
            else:
                action = "left as it is: " + ("framing differs" if k == "features" else "indirect match")
        d["slots"].append({"path": str(items[i].path), "action": action})
    d["keeper"] = best
    d["losers"] = sorted(i for i in members if i != best
                         and (items[i].role == 0 or i not in survivors or i in d["sync"] or i == primary))
    d["slot_like"] = set(d["sync"]) | ({primary} if primary is not None else set())
    if best != keeper:
        d["steps"] = steps_from(best, members, edges)
    return d


def plan_placement(best, slot, items, names, leaving, action, best_caption=None):
    """Where the best copy goes for one slot: a move for the promoted keeper
    (action "move"), a copy for slot sync (action "copy"). The slot's image is
    already planned out, so its name is free unless another image of the same
    stem stays. A caption that belongs to the slot is never moved or
    overwritten; best_caption is the caption the best copy will have, for a
    synced slot that has none."""
    b, sl = items[best], items[slot]
    dest = sl.path.parent / (sl.path.stem + b.path.suffix)
    renamed = names.exists(dest)
    if renamed:
        dest = names.free_dest(dest, dest.with_suffix(".txt"))
    else:
        names.taken.add(dest)
    slot_cap = sl.caption if sl.caption and sl.caption.exists() else None
    captions = []
    if slot_cap is not None and renamed:
        captions.append({"action": "copy", "from": str(slot_cap), "to": str(dest.with_suffix(".txt")),
                         "reason": "caption of the slot, under the new name"})
    if action == "move":
        own_cap = b.caption if b.caption and b.caption.exists() else None
        own_shared = own_cap is not None and caption_shared(b, items, leaving)
        if slot_cap is not None:
            if own_cap is not None and not own_shared:
                park = names.free_dest((b.root / OUT_DIRNAME / b.rel).with_suffix(".txt"))
                captions.append({"action": "move", "from": str(own_cap), "to": str(park),
                                 "reason": "caption of a promoted copy; the caption of the slot wins"})
        elif own_cap is not None:
            captions.append({"action": "copy" if own_shared else "move", "from": str(own_cap),
                             "to": str(dest.with_suffix(".txt")), "reason": "caption of a promoted copy"})
        has_caption = slot_cap is not None or own_cap is not None
    else:
        if slot_cap is None and best_caption is not None:
            captions.append({"action": "copy", "from": str(best_caption), "to": str(dest.with_suffix(".txt")),
                             "reason": "caption of the best copy, for a slot without one"})
        has_caption = slot_cap is not None or best_caption is not None
    if has_caption:
        names.taken.add(dest.with_suffix(".txt"))
    return {"action": action, "from": str(b.path), "to": str(dest), "renamed": renamed,
            "slot_of": str(sl.path), "slot_idx": slot, "has_caption": has_caption, "captions": captions}


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
            if e.get("action") == "copy":
                # A caption the tool copied to a kept image: remove it, but only
                # if nobody has edited it since.
                if dst.exists() and file_sha256(dst) == e.get("sha256"):
                    os.chmod(dst, 0o666)             # a copy of a read-only file is read-only too
                    dst.unlink()
                    restored += 1
                elif dst.exists():
                    skipped += 1
                    print(f"  copied caption was edited since, left in place: {dst}")
                continue
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
    ap.add_argument("folders", nargs="*", help="unsorted folders (raw downloads) to scan recursively; "
                                               "all folders form one pool")
    ap.add_argument("--sorted", action="append", default=[], metavar="FOLDER",
                    help="a sorted (curated) folder; repeatable. A better unsorted copy moves into "
                         "the sorted copy's place under its name; sorted captions are never touched; "
                         "a sorted copy wins a tie. Copies that match by features only, and pictures "
                         "held in several sorted folders, are listed for review and left alone")
    ap.add_argument("--sorted-copies", choices=["keep", "one"], default="keep",
                    help="the same picture in several sorted folders: keep (default) keeps every "
                         "folder's copy and gives each the best copy; one keeps a single copy and "
                         "moves the others out")
    ap.add_argument("--promote-margin", type=float, default=PROMOTE_MARGIN, metavar="X",
                    help=f"same-bucket score margin an unsorted copy needs to displace a sorted copy "
                         f"(default {PROMOTE_MARGIN:g}); a larger bucket always qualifies")
    ap.add_argument("--match", choices=list(MATCH_RULES), default="loose",
                    help="loose (default): hashes plus a feature check; finds other scans, crops, "
                         "borders, frames, mock-ups and watermarks. strict: resized and recompressed "
                         "copies, hashes only. exact: identical files only")
    ap.add_argument("--dry-run", action="store_true", help="write hashes.json and plan.json, move nothing")
    ap.add_argument("--review", action="store_true",
                    help="after the run, open the review tool (review.py) on the groups: check them by eye "
                         "and change the kept copy; not after --dry-run, because nothing has moved")
    ap.add_argument("--undo", action="store_true", help="move the files of the last run back")
    ap.add_argument("--exclude", action="append", default=[], metavar="NAME",
                    help=f"skip folders with this name, at any depth (repeatable); always skipped: "
                         f"{', '.join(DEFAULT_EXCLUDES)}")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="worker processes (default: CPU count - 1)")
    ap.add_argument("--gpu", action="store_true",
                    help="match the feature descriptors on the GPU with torch; needs the torch that one of "
                         "the GPU tools installed into the shared venv, else the CPU is used")
    args = ap.parse_args(argv)

    roots, roles = [], []
    for role, folders in ((0, args.folders), (1, args.sorted)):
        for f in folders:
            p = Path(f).resolve()
            if not p.is_dir():
                ap.error(f"not a folder: {f}")
            roots.append(p)
            roles.append(role)
    if not roots:
        ap.error("give at least one folder")
    for a in roots:
        for b in roots:
            if a != b and b.is_relative_to(a):
                ap.error(f"{b} is inside {a}; give only the outer folder")
    if len(set(roots)) != len(roots):
        ap.error("the same folder is given twice")

    if args.undo:
        return undo(roots)

    t0 = time.time()
    items = scan(roots, DEFAULT_EXCLUDES + args.exclude, roles)
    print(f"{len(items)} image(s) in {len(roots)} folder(s)"
          + (f", {sum(1 for it in items if it.role == 1)} of them in {len(args.sorted)} sorted folder(s)"
             if args.sorted else ""))
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
    edges = find_edges(good, args.match, verified, args.workers, args.gpu)
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

    # 3. plan: decide every group, then choose every destination against the
    #    planned state. Move-outs are planned before promotions, because they
    #    run before them (a promotion may take a name a move-out frees).
    with_roles = bool(args.sorted)
    decisions = [decide_group(g, good, edges, with_roles, args.promote_margin, args.sorted_copies)
                 for g in sorted(groups, key=lambda g: (good[g[0]].root_idx, good[g[0]].rel))]
    acted = [d for d in decisions if d["skip"] is None]
    moving = {i for d in acted for i in d["losers"]}
    leaving = moving | {d["keeper"] for d in acted if d["promote"] is not None}
    names = Names()
    for d in acted:
        d["moves"] = []
        for i in d["losers"]:
            it = good[i]
            cap = it.caption if it.caption and it.caption.exists() else None
            if cap is None:
                mode = None
            elif i in d["slot_like"]:
                mode = "copy"                       # the slot keeps its caption for the copy that replaces it
            elif caption_shared(it, good, leaving):
                mode = None                         # stays with the image of the same stem
            else:
                mode = "move"
            out = it.root / OUT_DIRNAME / it.rel
            # Two slot images that share one caption may share its copy too.
            peer = out.with_suffix(".txt") if mode and names.copies.get(out.with_suffix(".txt")) != cap else None
            dst = names.free_dest(out, peer)
            names.freed.add(it.path)
            if mode == "move":
                names.freed.add(cap)
            elif mode == "copy":
                names.copies[dst.with_suffix(".txt")] = cap
            d["moves"].append({"idx": i, "to": dst, "cap": cap if mode else None, "cap_mode": mode,
                               "cap_to": dst.with_suffix(".txt") if mode else None})
    for d in acted:
        k = good[d["keeper"]]
        d["promotion"] = (plan_placement(d["keeper"], d["promote"], good, names, leaving, "move")
                          if d["promote"] is not None else None)
        final = Path(d["promotion"]["to"]) if d["promotion"] else k.path
        has_caption = d["promotion"]["has_caption"] if d["promotion"] else (k.caption is not None and k.caption.exists())
        d["copy_caption"] = None
        if not has_caption:
            src = caption_source(d["keeper"], d["losers"], good)
            if src is not None:
                d["copy_caption"] = (src, final.with_suffix(".txt"))
                names.taken.add(final.with_suffix(".txt"))
                has_caption = True
        best_caption = final.with_suffix(".txt") if has_caption else None
        d["syncs"] = []
        for i in d["sync"]:
            pc = plan_placement(d["keeper"], i, good, names, leaving, "copy", best_caption)
            pc["from"] = str(final)
            d["syncs"].append(pc)

    plan = {"written": datetime.now().isoformat(timespec="seconds"), "match": args.match,
            "folders": [str(r) for r, ro in zip(roots, roles) if ro == 0],
            "sorted_folders": [str(r) for r, ro in zip(roots, roles) if ro == 1],
            "promote_margin": args.promote_margin, "sorted_copies": args.sorted_copies, "groups": []}
    verdicts, layouts = {}, {}

    def copy_info(k, i, steps):
        it = good[i]
        p = int(bin(int(k.data["phash"], 16) ^ int(it.data["phash"], 16)).count("1"))
        dd = int(bin(int(k.data["dhash"], 16) ^ int(it.data["dhash"], 16)).count("1"))
        direct = edges.get((min(k.idx, i), max(k.idx, i)))
        return {**describe(it), "exact": it.data["sha256"] == k.data["sha256"],
                "phash_distance": p, "dhash_distance": dd, "steps_from_kept": steps.get(i),
                "match": direct[0] if direct else "through other copies",
                **({"features": {x: direct[1][x] for x in ("inliers", "cover", "ncc") if x in direct[1]}}
                   if direct and "inliers" in direct[1] else {})}

    for d in decisions:
        k = good[d["keeper"]]
        entry = {"keep": describe(k), "reason": d["reason"], **d["report"]}
        rep = d["report"]
        if rep.get("roles") == "mixed":
            v = rep["verdict"].split(":")[0]
            verdicts[v] = verdicts.get(v, 0) + 1
        elif rep.get("roles") == "sorted":
            lay = rep["layout"] + (" (a slot left as it is)" if "note" in rep else "")
            layouts[lay] = layouts.get(lay, 0) + 1
        if d["skip"] is not None:
            entry["review"] = d["skip"]
            entry["members"] = [copy_info(k, i, d["steps"]) for i in d["losers"]]
        else:
            if d["slots"]:
                entry["slots"] = d["slots"]
            if d["promotion"]:
                entry["promote"] = {x: v for x, v in d["promotion"].items() if x not in ("action", "slot_idx")}
            if d["syncs"]:
                entry["sync"] = [{x: v for x, v in pc.items() if x not in ("action", "slot_idx")} for pc in d["syncs"]]
            if d["copy_caption"]:
                src, cdst = d["copy_caption"]
                entry["copy_caption"] = {"from": str(src.caption), "to": str(cdst)}
            entry["move"] = [{**copy_info(k, m["idx"], d["steps"]), "to": str(m["to"]),
                              **({"caption_to": str(m["cap_to"]), "caption_action": m["cap_mode"]} if m["cap_mode"] else {})}
                             for m in d["moves"]]
        plan["groups"].append(entry)

    n_moves = len(moving)
    n_bytes = sum(good[i].size for i in moving)
    skipped = [d for d in decisions if d["skip"] is not None]
    promotions = [d for d in acted if d["promotion"]]
    caption_copies = [d for d in acted if d["copy_caption"]]
    n_ren = sum(1 for d in promotions if d["promotion"]["renamed"])
    n_sync = sum(len(d["syncs"]) for d in acted)
    n_left = sum(1 for d in acted for sl in d["slots"] if sl["action"].startswith("left"))
    n_out = sum(1 for d in acted for sl in d["slots"] if sl["action"].startswith("moves out"))
    plan["summary"] = {"groups": len(groups), "copies_to_move": n_moves, "bytes_to_move": n_bytes,
                       "review_groups": len(skipped), "captions_copied_to_kept": len(caption_copies),
                       **({"mixed_verdicts": verdicts, "sorted_only_layouts": layouts,
                           "promotions": len(promotions), "promotions_renamed": n_ren, "slots_synced": n_sync,
                           "slots_left_as_they_are": n_left, "sorted_copies_out": n_out} if with_roles else {})}
    for root in roots:
        write_json(root / OUT_DIRNAME / PLAN_NAME, plan)
    print(f"{len(groups)} group(s); {n_moves} worse cop{'y' if n_moves == 1 else 'ies'} to move, "
          f"{n_bytes / 1e6:.1f} MB"
          + (f"; {len(skipped)} group(s) left for review, nothing moves there" if skipped else ""))
    sizes = {}
    for g in groups:
        sizes[len(g[1]) + 1] = sizes.get(len(g[1]) + 1, 0) + 1
    if sizes:
        print("copies per group: " + ", ".join(f"{n} x{c}" for n, c in sorted(sizes.items())))
    guarded = [d for d in acted if d["reason"].startswith("guard")]
    if guarded:
        print(f"{len(guarded)} group(s) keep a smaller copy because the larger one scores much lower")
    if caption_copies:
        print(f"{len(caption_copies)} kept image(s) without a caption get a copy of the caption "
              f"of a moved copy")
    if with_roles:
        n_mixed = sum(verdicts.values())
        print(f"sorted/unsorted: {n_mixed} group(s) with copies in both"
              + (": " + ", ".join(f"{n} {v}" for v, n in sorted(verdicts.items())) if n_mixed else "")
              + f" (margin {args.promote_margin:g})")
        n_sorted = sum(layouts.values())
        print(f"  {n_sorted} group(s) inside sorted folders only"
              + (": " + ", ".join(f"{n} {v}" for v, n in sorted(layouts.items())) if n_sorted else ""))
        if promotions:
            print(f"  {len(promotions)} unsorted cop{'y moves' if len(promotions) == 1 else 'ies move'} "
                  f"into sorted folders" + (f", {n_ren} under a new name" if n_ren else ""))
        if n_sync:
            print(f"  {n_sync} slot(s) in other sorted folders get a copy of the best copy "
                  f"(policy {args.sorted_copies})")
        if n_left:
            print(f"  {n_left} sorted slot(s) left as they are: framing differs or indirect match")
        if n_out:
            print(f"  {n_out} sorted cop{'y moves' if n_out == 1 else 'ies move'} out"
                  + (" (one slot per picture)" if args.sorted_copies == "one" else " (a better copy stays in the same folder)"))

    if args.dry_run:
        print(f"Dry run: nothing moved. Plan: {roots[0] / OUT_DIRNAME / PLAN_NAME}")
        if args.review:
            print("--review needs a real run; nothing has moved yet")
        return 0
    if not acted:
        print(f"Nothing to move. Done in {time.time() - t0:.0f}s.")
        return 0

    # 4. move, in passes: captions copied to kept images, move-outs, promotions,
    #    then the caption actions of the promotions. A promotion is logged in the
    #    sorted root's log, after the move-out it depends on, so --undo replays
    #    the sequence in the right order.
    moved = {r: [0, 0, 0, 0, 0, 0] for r in roots}  # images, captions, bytes, captions copied, promoted in, synced
    logs = {}

    def log_for(root):
        if root not in logs:
            (root / OUT_DIRNAME).mkdir(parents=True, exist_ok=True)
            logs[root] = open(root / OUT_DIRNAME / MOVES_NAME, "a", encoding="utf-8")
        return logs[root]

    def log(root, **entry):
        f = log_for(root)
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), **entry},
                           ensure_ascii=False) + "\n")
        f.flush()

    def copy_caption(src: Path, dst: Path, root: Path, reason: str) -> bool:
        if dst.exists() or not src.exists():        # never overwrite a caption
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        log(root, action="copy", **{"from": str(src), "to": str(dst)}, sha256=file_sha256(dst), reason=reason)
        return True

    def final_path(d):
        return d["promotion"]["to"] if d["promotion"] else str(good[d["keeper"]].path)

    try:
        for d in acted:                                            # pass 0: captions for kept images
            if d["copy_caption"]:
                src, cdst = d["copy_caption"]
                root = next(r for r, ro in zip(roots, roles) if cdst.is_relative_to(r))
                if copy_caption(src.caption, cdst, root, "kept image had no caption"):
                    moved[root][3] += 1
        for d in acted:                                            # pass 1: move-outs
            kept = final_path(d)
            for m in d["moves"]:
                it = good[m["idx"]]
                if not it.path.exists():
                    print(f"  gone before the move, skipped: {it.path}")
                    continue
                dst = m["to"]
                if dst.exists():
                    dst = free_dest(dst, dst.with_suffix(".txt") if m["cap_mode"] else None)
                    print(f"  planned name was taken, used {dst.name}: {it.path}")
                move_file(it.path, dst)
                log(it.root, **{"from": str(it.path), "to": str(dst)}, kept=kept, reason=d["reason"])
                moved[it.root][0] += 1
                moved[it.root][2] += it.size
                if m["cap_mode"] == "move" and m["cap"].exists():
                    move_file(m["cap"], dst.with_suffix(".txt"))
                    log(it.root, **{"from": str(m["cap"]), "to": str(dst.with_suffix(".txt"))}, kept=kept,
                        reason="caption of a moved copy")
                    moved[it.root][1] += 1
                elif m["cap_mode"] == "copy":
                    copy_caption(m["cap"], dst.with_suffix(".txt"), it.root,
                                 "caption of the slot; a copy for the moved image")
        for d in acted:                                            # pass 2: promotions
            if not d["promotion"]:
                continue
            k, slot = good[d["keeper"]], good[d["promote"]]
            src, dst = Path(d["promotion"]["from"]), Path(d["promotion"]["to"])
            if not src.exists():
                print(f"  GONE before the move, slot left empty, run --undo: {src} -> {dst}")
                continue
            if dst.exists():
                dst = free_dest(dst, dst.with_suffix(".txt"))
                print(f"  planned name was taken, used {dst.name}: {src}")
            move_file(src, dst)
            log(slot.root, action="promote", **{"from": str(src), "to": str(dst)}, slot_of=str(slot.path),
                reason=d["reason"])
            moved[slot.root][4] += 1
            d["promotion"]["to"] = str(dst)
        for d in acted:                                            # pass 3: captions of promotions
            if not d["promotion"]:
                continue
            slot = good[d["promote"]]
            for c in d["promotion"]["captions"]:
                src, dst = Path(c["from"]), Path(c["to"])
                if c["action"] == "copy":
                    copy_caption(src, dst, slot.root, c["reason"])
                elif src.exists() and not dst.exists():
                    move_file(src, dst)
                    log(slot.root, **{"from": str(src), "to": str(dst)}, kept=d["promotion"]["to"],
                        reason=c["reason"])
        for d in acted:                                            # pass 4: slot sync copies
            for pc in d["syncs"]:
                slot = good[pc["slot_idx"]]
                src, dst = Path(d["promotion"]["to"] if d["promotion"] else pc["from"]), Path(pc["to"])
                if not src.exists():
                    print(f"  GONE before the copy, slot left empty, run --undo: {src} -> {dst}")
                    continue
                if dst.exists():
                    dst = free_dest(dst, dst.with_suffix(".txt"))
                    print(f"  planned name was taken, used {dst.name}: {dst}")
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                log(slot.root, action="copy", **{"from": str(src), "to": str(dst)}, sha256=file_sha256(dst),
                    slot_of=str(slot.path), reason="slot sync: the best copy of this picture")
                moved[slot.root][5] += 1
                pc["to"] = str(dst)
                for c in pc["captions"]:
                    copy_caption(Path(c["from"]), Path(c["to"]), slot.root, c["reason"])
    finally:
        for f in logs.values():
            f.close()

    # 5. the hashes and scores of promoted and synced files are known: carry
    #    them to the cache of the folder they now live in
    carried = {}
    for d in acted:
        data = good[d["keeper"]].data
        if d["promotion"]:
            carried.setdefault(good[d["promote"]].root, []).append((Path(d["promotion"]["to"]), data))
        for pc in d["syncs"]:
            carried.setdefault(good[pc["slot_idx"]].root, []).append((Path(pc["to"]), data))
    for root, entries in carried.items():
        carry_cache(root, entries)

    for root, (n_img, n_cap, n_b, n_copied, n_in, n_sync) in moved.items():
        print(f"{root}: moved {n_img} image(s) and {n_cap} caption(s), {n_b / 1e6:.1f} MB "
              f"-> {root / OUT_DIRNAME}" + (f"; copied {n_copied} caption(s) to kept images" if n_copied else "")
              + (f"; {n_in} image(s) promoted into it" if n_in else "")
              + (f"; {n_sync} slot(s) synced" if n_sync else ""))
    print(f"Done in {time.time() - t0:.0f}s. Undo with --undo.")
    if args.review:
        import subprocess
        return subprocess.call([sys.executable, str(Path(__file__).with_name("review.py")), str(roots[0])])
    return 0


if __name__ == "__main__":
    sys.exit(main())
