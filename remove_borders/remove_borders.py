#!/usr/bin/env python3
"""
Cut borders off the images of a dataset, in place.

Every folder given is a root and is scanned at any depth. Light and dark frames,
thin edge lines and banners with text at the top or bottom are cut off. JPEG
files are cropped losslessly (whole DCT blocks); PNG and other lossless formats
to the exact pixel; other lossy formats are cropped into a PNG. Before a crop,
the original and its sidecars are copied to <root>/_backup/<same relative path>.
An image whose crop would be smaller than --min-area is not cropped; it moves to
the same _backup tree with its sidecars.

Without --dry-run the crops are made: originals go to _backup first, every
step is logged in _backup/_remove_borders/log.jsonl, and --undo puts the last
run of each folder back.

Usage:    python remove_borders.py <folder> [<folder> ...] [--dry-run [--no-sheets]]
          [--sheets] [--undo] [--exclude NAME] [--sidecars LIST] [--min-area N]
          [--no-dark] [--reanalyse] [--threads N]
Install:  pip install Pillow numpy jpeglib
"""
import argparse
import csv
import filecmp
import json
import math
import os
import shutil
import signal
import stat
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None     # large scans are normal input, not an attack

BACKUP_DIRNAME = "_backup"
RUN_DIRNAME = "_remove_borders"   # inside _backup: plan, log, report, cache, sheets
REPORT_NAME = "report.csv"
PLAN_NAME = "plan.json"
SHEETS_DIRNAME = "sheets"
LOG_NAME = "log.jsonl"            # every real run and undo, appended; --undo reads it
CACHE_NAME = "cache.json"         # analysis results by file, reused while a file is unchanged
PROGRESS_EVERY = 500              # images between progress lines
MASKS_DIRNAME = "masks"           # face_masks: <dir>/masks/<stem>.png, the size of its image

IMAGE_EXTS = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff",
              ".gif", ".avif", ".heic", ".heif"}
# Folders never scanned, at any depth: the output folders of the dataset tools.
# _backup is this tool's own, _duplicates is deduplicate's, _prep is k2prep's,
# _classify and _embeddings are classify's, masks and faces are face_masks'.
# Any other folder is scanned, whatever its name; --exclude adds names.
DEFAULT_EXCLUDES = [BACKUP_DIRNAME, "_duplicates", "_prep", "_classify", "_embeddings", "masks", "faces"]
DEFAULT_SIDECARS = ".txt"
DEFAULT_MIN_AREA = 256 * 256      # smaller than this after a crop: moved to _backup, not cropped
DEFAULT_THREADS = 8

# How a crop can be made, by file format.
KIND_JPEG = "jpeg"                # lossless DCT crop
KIND_LOSSLESS = "lossless"        # exact crop, saved in the same format
KIND_LOSSY = "lossy"              # exact crop of the decoded pixels, saved as PNG
LOSSLESS_FORMATS = {"PNG", "BMP", "GIF", "TGA", "PPM", "PCX", "SGI", "QOI", "ICO", "DIB"}

REPORT_FIELDS = ["path", "format", "kind", "mode", "bits", "width", "height", "mcu", "orientation",
                 "frames", "sidecars", "mask", "action", "write", "new_width", "new_height",
                 "cut_top", "cut_bottom", "cut_left", "cut_right", "mcu_lost", "cuts", "reason", "result"]


@dataclass
class Item:
    root: Path
    path: Path
    rel: str                                  # relative to root, forward slashes
    sidecars: list[Path] = field(default_factory=list)
    shared_stem: bool = False                 # another image in the folder has the same stem
    mask: Path | None = None
    head: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)  # analyse_file
    key: list | None = None                       # file_key when analysed, for the cache

    @property
    def action(self) -> str:
        """skip, review, crop, too small or "" (nothing to do)."""
        return action_of(self.head, self.result)


def action_of(head: dict, result: dict) -> str:
    """What the analysis of one image leads to: skip, review, crop, too small
    or "" (nothing to do). result is analyse_array's dict with its "plan"."""
    if skip_reason(head) or result.get("error"):
        return "skip"
    if result.get("review"):
        return "review"
    if not result.get("cuts"):
        return ""
    write = result.get("plan", {}).get("write", "")
    return "too small" if write == "too small" else "crop" if write else ""


# --- scan ---------------------------------------------------------------------

def scan(root: Path, excludes, sidecar_exts) -> list[Item]:
    """The images under root at any depth, with their sidecars and masks.
    Folders are skipped by exact name (case-insensitive), at any depth."""
    excl = {e.casefold() for e in excludes}
    exts = {e.casefold() for e in sidecar_exts}
    items = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        masks_dir = next((d for d in dirnames if d.casefold() == MASKS_DIRNAME), None)
        dirnames[:] = sorted(d for d in dirnames if d.casefold() not in excl)
        side_by_stem: dict[str, list[str]] = {}
        images_by_stem: dict[str, int] = {}
        for fn in filenames:
            stem, ext = os.path.splitext(fn)
            if ext.casefold() in exts:
                side_by_stem.setdefault(stem.casefold(), []).append(fn)
            elif ext.casefold() in IMAGE_EXTS:
                images_by_stem[stem.casefold()] = images_by_stem.get(stem.casefold(), 0) + 1
        masks = {}
        if masks_dir:
            try:
                masks = {os.path.splitext(f)[0].casefold(): f for f in os.listdir(here / masks_dir)
                         if f.lower().endswith(".png")}
            except OSError:
                masks = {}
        for fn in sorted(filenames):
            stem, ext = os.path.splitext(fn)
            if ext.casefold() not in IMAGE_EXTS:
                continue
            key = stem.casefold()
            p = here / fn
            items.append(Item(
                root=root, path=p, rel=p.relative_to(root).as_posix(),
                sidecars=[here / s for s in sorted(side_by_stem.get(key, []))],
                shared_stem=images_by_stem.get(key, 0) > 1,
                mask=(here / masks_dir / masks[key]) if key in masks else None))
    return items


# --- headers ------------------------------------------------------------------

def png_ihdr(path: Path):
    """(bit depth, colour type) from the PNG header, or (None, None)."""
    try:
        with open(path, "rb") as f:
            b = f.read(33)
    except OSError:
        return None, None
    if b[:8] != b"\x89PNG\r\n\x1a\n" or b[12:16] != b"IHDR":
        return None, None
    return b[24], b[25]


def webp_is_lossless(path: Path) -> bool:
    """True when the first image chunk of a WebP file is VP8L (lossless)."""
    with open(path, "rb") as f:
        head = f.read(12)
        if head[:4] != b"RIFF" or head[8:12] != b"WEBP":
            return False
        while True:
            chunk = f.read(8)
            if len(chunk) < 8:
                return False
            fourcc, size = chunk[:4], int.from_bytes(chunk[4:], "little")
            if fourcc == b"VP8L":
                return True
            if fourcc == b"VP8 ":
                return False
            f.seek(size + (size & 1), 1)          # chunks are padded to an even size


def read_header(path: Path) -> dict:
    """Format, crop kind, mode, bit depth, size, JPEG MCU size, EXIF orientation
    and frame count, from the file header only (no pixel decoding)."""
    h = {"format": "", "kind": "", "mode": "", "bits": 8, "width": 0, "height": 0, "mcu": None,
         "orientation": 1, "frames": 1, "error": ""}
    try:
        with Image.open(path) as im:
            fmt = im.format or ""
            h.update(format=fmt, mode=im.mode, width=im.size[0], height=im.size[1])
            try:
                o = im.getexif().get(274, 1) or 1
                h["orientation"] = o if o in range(1, 9) else 1
            except Exception:  # noqa: BLE001 - a damaged EXIF block is not fatal
                pass
            # MPO is a JPEG with more images appended (iPhone depth and gain maps),
            # not an animation: the first image is cropped as an ordinary JPEG
            h["frames"] = 1 if fmt == "MPO" else (getattr(im, "n_frames", 1) or 1)
            if fmt in ("JPEG", "MPO"):
                h["kind"] = KIND_JPEG
                layer = getattr(im, "layer", None)
                if layer:
                    h["mcu"] = [8 * max(c[1] for c in layer), 8 * max(c[2] for c in layer)]
            elif fmt == "WEBP":
                h["kind"] = KIND_LOSSLESS if webp_is_lossless(path) else KIND_LOSSY
            elif fmt == "TIFF":
                comp = str(im.info.get("compression", ""))
                h["kind"] = KIND_LOSSY if "jpeg" in comp else KIND_LOSSLESS
            elif fmt in LOSSLESS_FORMATS:
                h["kind"] = KIND_LOSSLESS
            else:                                 # AVIF, HEIF, JPEG 2000 and the rest
                h["kind"] = KIND_LOSSY
            if im.mode.startswith("I;16") or im.mode in ("I", "F"):
                h["bits"] = 16 if im.mode.startswith("I;16") else 32
    except Exception as e:  # noqa: BLE001 - any unreadable file is reported, not fatal
        h["error"] = f"{type(e).__name__}: {e}".strip()
        return h
    if h["format"] == "PNG":
        depth, _ = png_ihdr(path)
        if depth == 16:
            h["bits"] = 16                        # Pillow opens 16-bit RGB as 8-bit RGB
    return h


def skip_reason(h: dict) -> str:
    if h["error"]:
        return "unreadable: " + h["error"]
    if h["frames"] > 1:
        return "animated"
    return ""


# --- detection ----------------------------------------------------------------
# Thresholds were set on the sample folder; PLAN.md sections 2 and 6 give the
# measurements behind them. Distances are per channel, 8 bit, in stored pixels
# (no EXIF rotation), so a box found here is what the crop needs.

DETECTOR_VERSION = 1
TOL = 24                          # a pixel within this of the border colour, in every channel, is border
LIGHT_MIN = 160                   # mean of a light border colour
DARK_MAX = 70                     # mean of a dark border colour; mid-tones are never borders
MAX_DEPTH_SHARE = 0.25            # borders are searched in the outer quarter of each side
BANNER_MAX_SHARE = 0.12           # a banner is at most this share of the height
BANNER_TEXT_SHARE = 0.90          # a row of text leaves at most this share in the band colour; texture
                                  # (a white desk at the bottom of a photo) stays above it
MAX_SLOPE = math.tan(math.radians(2.0))   # a slanted frame edge, at most 2 degrees
FIT_TOL = 2                       # px: a column lies on the edge line within this distance
RANSAC_PAIRS = 200                # random column pairs tried for a slanted edge line
HOLE_TOL = 4                      # px: a column whose picture starts this much before the line is a hole
                                  # (less is the blur of a soft scan or JPEG edge)
FRAME_MIN_INLIERS = 0.30          # share of a side on its edge line; deeper columns are light content
SLANT_MIN_INLIERS = 0.60          # a slanted edge is a rotated print: straight along most of the side
CLEAN_INLIERS = 0.95              # sides this clean may leave one inner corner in border colour
SOFT_CORNER = (6, 0.01)           # px, share of the short side: a blank this small at an inner corner
                                  # is a rounded print corner or blur, and the corner counts as picture
FRAME_MAX_HOLES = 0.05            # share of a side where the picture starts before the line
STRICT_MIN_INLIERS = 0.70         # the same for a border on only 1 or 2 sides
STRICT_MAX_HOLES = 0.01
STRICT_MAX_DEPTH_SHARE = 0.05     # a 1 or 2 sided border is at most this share of the side
BACKGROUND_SHARE = 0.90           # another edge this much in the border colour: background, not border
OPEN_SIDE_SHARE = 0.80            # the open side of a 3 sided frame this much in the frame colour is
                                  # a fourth frame side (with up to OPEN_SIDE_HOLES) or a background
OPEN_SIDE_HOLES = 0.10
NEUTRAL_SPREAD = 16               # a studio background is neutral (white, grey, black): channels within this;
                                  # a tinted border (cream or sepia card) is never taken for a background
THIN_LINE = 3                     # px: a 1 or 2 sided border this thin is a "line" (k2prep test on JPEG)
MIN_KEEP = 0.50                   # a crop that keeps less of the area is left for review
MAX_GROW = 3                      # px: a cut grows while its new edge is still border colour
MAX_ROUNDS = 4                    # a frame with a keyline needs two or three
SIDES = "TBLR"
NEIGHBOURS = {"T": "LR", "B": "LR", "L": "TB", "R": "TB"}


def load_pixels(path) -> np.ndarray:
    """The stored pixels as RGB uint8, transparency composited over white."""
    with Image.open(path) as im:
        mode = im.mode
        if mode.startswith("I;16"):
            g = (np.asarray(im, dtype=np.uint16) >> 8).astype(np.uint8)
            return np.repeat(g[..., None], 3, axis=2)
        if mode in ("I", "F"):
            g = np.asarray(im, dtype=np.float64)
            g = np.clip(g * (255.0 / max(float(g.max()), 1.0)), 0, 255).astype(np.uint8)
            return np.repeat(g[..., None], 3, axis=2)
        if mode in ("RGBA", "LA", "PA", "RGBa", "La") or "transparency" in im.info:
            rgba = im.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return np.asarray(Image.alpha_composite(bg, rgba).convert("RGB"))
        return np.asarray(im.convert("RGB"))


def side_view(a: np.ndarray, side: str) -> np.ndarray:
    """A view of a with the given side as row 0 and the side's length along axis 1."""
    if side == "T":
        return a
    if side == "B":
        return a[::-1]
    if side == "L":
        return a.transpose(1, 0, 2)
    return a.transpose(1, 0, 2)[::-1]


def near(block: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Pixels within TOL of colour c in every channel. Compared in uint8 against
    the bounds, without a 16-bit copy of the block."""
    c = np.asarray(c, dtype=np.int16)
    lo = np.clip(c - TOL, 0, 255).astype(np.uint8)
    hi = np.clip(c + TOL, 0, 255).astype(np.uint8)
    return ((block >= lo) & (block <= hi)).all(axis=-1)


def tone_of(c: np.ndarray, dark: bool) -> str:
    lum = float(c.mean())
    if lum >= LIGHT_MIN:
        return "light"
    if dark and lum <= DARK_MAX:
        return "dark"
    return ""


def neutral(c) -> bool:
    return float(np.max(c) - np.min(c)) <= NEUTRAL_SPREAD


def edge_colour(view: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return np.round(np.median(view[0, lo:hi], axis=0)).astype(np.int16)


def depth_profile(view: np.ndarray, lo: int, hi: int, c: np.ndarray, maxd: int, chunk: int = 32) -> np.ndarray:
    """For each column of [lo, hi), the first depth whose pixel leaves colour c
    (maxd when none does within maxd). Read in chunks of rows and stopped as
    soon as every column has its depth, so most images read a few rows only."""
    n = hi - lo
    d = np.full(n, maxd, dtype=np.int32)
    open_ = np.ones(n, dtype=bool)
    for y0 in range(0, maxd, chunk):
        y1 = min(maxd, y0 + chunk)
        cols = np.flatnonzero(open_)
        bad = ~near(view[y0:y1, lo:hi][:, cols], c)
        hit = bad.any(axis=0)
        d[cols[hit]] = y0 + bad.argmax(axis=0)[hit]
        open_[cols[hit]] = False
        if not open_.any():
            break
    return d


def fit_edge(d: np.ndarray, maxd: int, slanted: bool = True) -> dict | None:
    """The inner edge of a border as a line depth = a + b*x through the depth
    profile: horizontal at the most common depth, or slanted (up to MAX_SLOPE)
    when a robust fit holds clearly more columns. None when too few columns end
    within maxd."""
    n = len(d)
    x = np.arange(n)
    valid = (d >= 1) & (d < maxd)
    if valid.sum() < FRAME_MIN_INLIERS * n:
        return None
    xs, ys = x[valid], d[valid].astype(np.float64)
    hist = np.bincount(d[valid], minlength=maxd + 1)
    win = np.convolve(hist, np.ones(2 * FIT_TOL + 1), mode="same")
    k = 1 + int(np.argmax(win[1:]))          # depth 0 is no border
    in_win = ys[np.abs(ys - k) <= FIT_TOL]
    mode = int(round(float(np.median(in_win))))  # the middle of the best window: soft scan edges spread

    def score(a, b):
        return int(np.count_nonzero(np.abs(ys - (a + b * xs)) <= FIT_TOL))

    flat = score(mode, 0.0)
    a, b, best = float(mode), 0.0, flat
    # A slanted line is tried only when the flat one leaves room to do clearly better.
    if slanted and len(xs) >= 8 and flat * 1.05 + 2 < len(xs):
        rng = np.random.default_rng(12345)
        i, j = rng.integers(0, len(xs), (2, RANSAC_PAIRS))
        dx = xs[j] - xs[i]
        ok = np.abs(dx) >= n / 4
        bb = np.divide(ys[j] - ys[i], dx, out=np.zeros(RANSAC_PAIRS), where=ok)
        ok &= (bb != 0) & (np.abs(bb) <= MAX_SLOPE)
        if ok.any():
            bb = bb[ok]
            aa = ys[i][ok] - bb * xs[i][ok]
            scores = (np.abs(ys[None, :] - (aa[:, None] + bb[:, None] * xs[None, :])) <= FIT_TOL).sum(axis=1)
            k = int(np.argmax(scores))
            if scores[k] > flat * 1.05 + 2:
                a, b, best = float(aa[k]), float(bb[k]), int(scores[k])
        if b != 0.0:
            inl = np.abs(ys - (a + b * xs)) <= FIT_TOL
            b2, a2 = np.polyfit(xs[inl], ys[inl], 1)
            if abs(b2) <= MAX_SLOPE and score(a2, b2) >= best:
                a, b = float(a2), float(b2)
    pred = a + b * x
    inliers = float(np.mean(valid & (np.abs(d - pred) <= FIT_TOL)))
    holes = float(np.mean((d == 0) | (d < pred - HOLE_TOL)))
    return {"a": a, "b": b, "inliers": inliers, "holes": holes,
            "deep": float(pred.max()), "shallow": float(pred.min())}


def banner_height(view: np.ndarray) -> int:
    """Height of a banner at row 0 of view: a band of one neutral colour (grey,
    white or black) with text in it, ending in a sharp straight edge. 0 when
    there is none. A band without text is a frame side, not a banner (the white
    floor of a product shot must not be cut as one); a coloured band with text
    is part of a poster's design."""
    h = view.shape[0]
    n = max(int(h * BANNER_MAX_SHARE), 6)
    if h < n + 3:
        return 0
    c = edge_colour(view, 0, view.shape[1])
    if not neutral(c):
        return 0
    share = near(view[:n + 3], c).mean(axis=1)
    if share[:2].min() < 0.95:
        return 0
    best, best_drop = 0, 0.3
    for k in range(4, n):
        if share[:k].min() < 0.5:
            break
        drop = share[k - 1] - share[k:k + 2].min()
        if share[k - 1] >= 0.95 and drop >= best_drop:
            best, best_drop = k, drop
    if not best or share[:best].min() > BANNER_TEXT_SHARE:   # no edge, or no text
        return 0
    return best


def measure_sides(a: np.ndarray, dark: bool) -> dict:
    """Colour, tone and depth profile of each side over its inner span, the
    span between the borders of the two neighbouring sides (found in a first
    pass over the full length)."""
    h, w = a.shape[:2]
    views = {s: side_view(a, s) for s in SIDES}
    maxd = {s: max(int((h if s in "TB" else w) * MAX_DEPTH_SHARE), 1) for s in SIDES}
    rough = {}
    for s, v in views.items():
        c = edge_colour(v, 0, v.shape[1])
        fit = fit_edge(depth_profile(v, 0, v.shape[1], c, maxd[s]), maxd[s]) if tone_of(c, dark) else None
        rough[s] = int(math.ceil(fit["deep"])) + FIT_TOL if fit else 0
    out = {}
    for s, v in views.items():
        n1, n2 = NEIGHBOURS[s]
        lo, hi = rough[n1], v.shape[1] - rough[n2]
        if hi - lo < 16:
            lo, hi = 0, v.shape[1]
        c = edge_colour(v, lo, hi)
        tone = tone_of(c, dark)
        m = {"c": c, "tone": tone, "lo": lo, "hi": hi, "maxd": maxd[s], "d": None}
        if tone:
            m["d"] = depth_profile(v, lo, hi, c, maxd[s])
        out[s] = m
    return out


def cut_depth(view: np.ndarray, m: dict, deep: float) -> int:
    """Where to cut a side whose border ends at depth deep: past the soft edge,
    and further while the new edge is still border colour."""
    cut = int(math.ceil(deep)) + (1 if deep <= 6 else 2)
    for _ in range(MAX_GROW):
        if cut >= view.shape[0] or near(view[cut, m["lo"]:m["hi"]], m["c"]).mean() <= 0.5:
            break
        cut += 1
    return cut


def find_frame(a: np.ndarray, ms: dict, outer: np.ndarray | None = None) -> tuple[dict | None, str, dict | None]:
    """Cuts {side: (depth, slope degrees)} for a frame on 3 or 4 sides, or None
    and why not. When only the inner-corner test fails, the cuts come back as
    the third value: the caller looks for a second frame inside (a print on a
    card, scanned on a white bed), which a product shot never has.
    With outer (the colour of a frame already found around a), a frame on 1 or
    2 sides is enough, if it has that colour and clean edges (CLEAN_INLIERS, STRICT_MAX_HOLES): the card of a
    rotated print shows only where the outer cuts did not reach the photo."""
    h, w = a.shape[:2]
    fits = {}
    for s, m in ms.items():
        if m["d"] is None:
            continue
        f = fit_edge(m["d"], m["maxd"])
        if (f and f["inliers"] >= (SLANT_MIN_INLIERS if f["b"] else FRAME_MIN_INLIERS)
                and f["holes"] <= FRAME_MAX_HOLES and f["shallow"] >= 0.5):
            fits[s] = f
    if outer is not None:
        fits = {s: f for s, f in fits.items() if f["inliers"] >= CLEAN_INLIERS
                and f["holes"] <= STRICT_MAX_HOLES and np.abs(ms[s]["c"] - outer).max() <= TOL}
    if len(fits) < (1 if outer is not None else 3):
        return None, f"{len(fits)} frame sides", None
    tones = {ms[s]["tone"] for s in fits}
    colours = np.array([ms[s]["c"] for s in fits])
    if len(tones) > 1 or (colours.max(axis=0) - colours.min(axis=0)).max() > 2 * TOL:
        return None, "frame colours differ", None
    if len(fits) == 3 and outer is None and neutral(np.median(colours, axis=0)):
        o = next(s for s in SIDES if s not in fits)
        if near(side_view(a, o)[0], np.median(colours, axis=0)).mean() >= OPEN_SIDE_SHARE:
            f = fit_edge(ms[o]["d"], ms[o]["maxd"]) if ms[o]["d"] is not None else None
            if not (f and f["inliers"] >= (SLANT_MIN_INLIERS if f["b"] else FRAME_MIN_INLIERS)
                    and f["holes"] <= OPEN_SIDE_HOLES and f["shallow"] >= 0.5):
                return None, "background, not a frame: the open side has the frame colour too", None
            fits[o] = f
    cuts = {s: (cut_depth(side_view(a, s), ms[s], f["deep"]), math.degrees(math.atan(f["b"]))) for s, f in fits.items()}
    t, b, l, r = (cuts.get(s, (0, 0))[0] for s in SIDES)
    if t + b >= h - 2 or l + r >= w - 2:
        return None, "nothing left inside", None
    fc = np.median(colours, axis=0)
    soft = max(SOFT_CORNER[0], int(SOFT_CORNER[1] * min(h, w)))
    corners = {"TL": (t, l, 1, 1), "TR": (t, w - r - 1, 1, -1),
               "BL": (h - b - 1, l, -1, 1), "BR": (h - b - 1, w - r - 1, -1, -1)}
    framed = [k for k in corners if k[0] in fits and k[1] in fits]
    content = 0
    for k in framed:
        y, x, dy, dx = corners[k]
        n = np.arange(min(soft + 1, h - t - b, w - l - r))
        diag = a[y + dy * n, x + dx * n].astype(np.int16)
        content += int((np.abs(diag - fc).max(axis=1) > TOL).any())
    clean = all(f["inliers"] >= CLEAN_INLIERS for f in fits.values())
    if content < len(framed) - (1 if len(framed) == 4 or clean else 0):
        return None, f"inner corners are border colour ({content} of {len(framed)} are picture)", cuts
    return cuts, "", None


def find_border(a: np.ndarray, ms: dict) -> tuple[dict | None, str]:
    """Cuts for a border on 1 or 2 sides: one depth (no slant), few holes, thin,
    and not a background that runs round the whole picture."""
    h, w = a.shape[:2]
    found = {}
    for s, m in ms.items():
        if m["d"] is None:
            continue
        f = fit_edge(m["d"], m["maxd"], slanted=False)
        if (f and f["inliers"] >= STRICT_MIN_INLIERS and f["holes"] <= STRICT_MAX_HOLES
                and f["deep"] <= STRICT_MAX_DEPTH_SHARE * (h if s in "TB" else w)):
            found[s] = f["deep"]
    if not found:
        return None, "no border"
    if len(found) > 2:
        return None, f"{len(found)} sides look like a border but are no frame"
    for s in found:
        if not neutral(ms[s]["c"]):
            continue
        for o in SIDES:
            if o not in found and near(side_view(a, o)[0], ms[s]["c"]).mean() >= BACKGROUND_SHARE:
                return None, "background, not a border: another edge has the same colour"
    return {s: (cut_depth(side_view(a, s), ms[s], D), 0.0) for s, D in found.items()}, ""


def nested_frame(a: np.ndarray, outer: dict, ms_outer: dict, dark: bool):
    """(measures, cuts) of a frame inside the box the outer cuts leave, or None."""
    h, w = a.shape[:2]
    t, b, l, r = (outer.get(s, (0, 0))[0] for s in SIDES)
    inner = a[t:h - b, l:w - r]
    if min(inner.shape[:2]) < 16:
        return None
    ms = measure_sides(inner, dark)
    colour = np.median(np.array([m["c"] for m in ms_outer.values()]), axis=0)
    got, _, _ = find_frame(inner, ms, outer=colour)
    return (ms, got) if got else None


def analyse_array(a: np.ndarray, dark: bool = True) -> dict:
    """The box to keep, in stored pixels, and the cuts that give it."""
    H, W = a.shape[:2]
    y0, y1, x0, x1 = 0, H, 0, W
    cuts = []
    note = ""
    for rnd in range(1, MAX_ROUNDS + 1):
        found = False
        for s in "BT":
            sub = a[y0:y1, x0:x1]
            bh = banner_height(side_view(sub, s))
            if bh:
                cuts.append({"round": rnd, "side": s, "kind": "banner", "tone": "", "depth": bh, "slope": 0.0})
                y0, y1 = (y0, y1 - bh) if s == "B" else (y0 + bh, y1)
                found = True
        sub = a[y0:y1, x0:x1]
        if min(sub.shape[:2]) < 16:
            break
        ms = measure_sides(sub, dark)
        got, why, pending = find_frame(sub, ms)
        kind = "frame"
        if pending:
            nested = nested_frame(sub, pending, {s: ms[s] for s in pending}, dark)
            if nested:
                for s, (depth, slope) in pending.items():
                    cuts.append({"round": rnd, "side": s, "kind": "frame", "tone": ms[s]["tone"],
                                 "depth": int(depth), "slope": round(slope, 2)})
                t, b, l, r = (pending.get(s, (0, 0))[0] for s in SIDES)
                y0, y1, x0, x1 = y0 + t, y1 - b, x0 + l, x1 - r
                sub = a[y0:y1, x0:x1]
                ms, got = nested
        if not got:
            got, why2 = find_border(sub, ms)
            kind = "border"
            note = why if why2 == "no border" else why2
        if got:
            for s, (depth, slope) in got.items():
                k = "line" if kind == "border" and depth <= THIN_LINE else kind
                cuts.append({"round": rnd, "side": s, "kind": k, "tone": ms[s]["tone"],
                             "depth": int(depth), "slope": round(slope, 2)})
            t, b, l, r = (got.get(s, (0, 0))[0] for s in SIDES)
            y0, y1, x0, x1 = y0 + t, y1 - b, x0 + l, x1 - r
            found = True
        if not found:
            break
    box = [x0, y0, x1, y1]
    review = ""
    if cuts and (x1 - x0) * (y1 - y0) < MIN_KEEP * W * H:
        review = f"the crop would keep {(x1 - x0) * (y1 - y0) / (W * H):.0%} of the image"
    return {"size": [W, H], "box": box, "cuts": cuts, "review": review, "note": "" if cuts else note}


def analyse_file(path: str, dark: bool) -> dict:
    """analyse_array on a file; runs in a worker process."""
    try:
        return analyse_array(load_pixels(path), dark)
    except Exception as e:  # noqa: BLE001 - one bad file is reported, not fatal
        return {"error": f"{type(e).__name__}: {e}".strip()}


def describe_cuts(cuts) -> str:
    return "; ".join(f"{c['side']} {c['kind']}{' ' + c['tone'] if c['tone'] else ''} {c['depth']}"
                     + (f" {c['slope']:+.1f}deg" if c["slope"] else "") for c in cuts)


# --- crop plan ----------------------------------------------------------------
# The detector's box becomes the box that is written: a JPEG crop starts on the
# MCU grid, a thin line on a JPEG is cut only when k2prep keeps the bucket of
# the image, and a crop below --min-area is not made (the image moves out).

# Aspect-ratio families, tiers and buckets, copied from k2prep.py
# (../k2prep), which ports musubi-tuner's BucketSelector.
AR_FAMILIES = ["9:16", "2:3", "4:5", "1:1", "5:4", "3:2", "16:9"]
AR_NOMINAL = [0.5647, 0.6667, 0.8028, 1.0000, 1.2456, 1.5000, 1.7708]
TIERS = [1024, 768, 512]
UPSCALE_TOLERANCE = 1.15
RESO_STEPS = 16


@lru_cache(maxsize=None)
def generate_buckets(resolution: int, steps: int = RESO_STEPS):
    area = resolution * resolution
    sqrt_size = int(math.sqrt(area))
    min_size = sqrt_size // 2 - (sqrt_size // 2) % steps
    out = []
    for bw in range(min_size, sqrt_size + steps, steps):
        bh = (area // bw) - (area // bw) % steps
        out.append((bw, bh))
        out.append((bh, bw))
    return sorted(set(out))


@lru_cache(maxsize=None)
def bucket_for(tier: int, family: str):
    nominal = AR_NOMINAL[AR_FAMILIES.index(family)]
    return min(generate_buckets(tier), key=lambda b: abs(b[0] / b[1] - nominal))


def assign_family(ratio: float) -> str:
    return AR_FAMILIES[min(range(len(AR_NOMINAL)), key=lambda i: abs(AR_NOMINAL[i] - ratio))]


def crop_dims(w: int, h: int, target_ar: float):
    """k2prep: minimal crop to the target ratio, in source pixels."""
    if w / h > target_ar:
        cw, ch = int(round(h * target_ar)), h
    else:
        cw, ch = w, int(round(w / target_ar))
    return max(1, min(cw, w)), max(1, min(ch, h))


def k2_bucket(w: int, h: int) -> tuple[str, int | None]:
    """k2prep's family and tier for an upright image of w x h; tier None when
    the image does not reach the 512 tier."""
    family = assign_family(w / h)
    for tier in TIERS:
        bw, bh = bucket_for(tier, family)
        cw, ch = crop_dims(w, h, bw / bh)
        if cw * ch >= (bw * bh) / (UPSCALE_TOLERANCE ** 2):
            return family, tier
    return family, None


def box_from_cuts(size, cuts, skip_kinds=()) -> list[int]:
    """The box the cuts leave: each cut takes its depth off its side."""
    W, H = size
    t = {s: sum(c["depth"] for c in cuts if c["side"] == s and c["kind"] not in skip_kinds) for s in SIDES}
    return [t["L"], t["T"], W - t["R"], H - t["B"]]


def align_to_mcu(box, mcu) -> list[int]:
    """A lossless JPEG crop starts on the MCU grid: the left and top edges move
    inward to the next multiple, so the border goes completely. The right and
    bottom edges stay where they are (the last block is cut by the image size)."""
    mw, mh = mcu
    x0, y0, x1, y1 = box
    return [-(-x0 // mw) * mw, -(-y0 // mh) * mh, x1, y1]


def crop_plan(head: dict, result: dict, min_area: int) -> dict:
    """What to write: {"box", "write" (jpeg, exact, png, too small or ""),
    "lost" (px of picture lost at the left and top to the MCU grid), "note"}."""
    W, H = result["size"]
    cuts = result.get("cuts", [])
    box = list(result["box"])
    plan = {"box": box, "write": "", "lost": [0, 0], "note": ""}
    if not cuts or result.get("review"):
        return plan
    jpeg = head.get("kind") == KIND_JPEG and head.get("mcu")

    def finish(b):
        return align_to_mcu(b, head["mcu"]) if jpeg else list(b)

    box = finish(box)
    if jpeg and any(c["kind"] == "line" for c in cuts):
        before = finish(box_from_cuts((W, H), cuts, skip_kinds=("line",)))
        rot = head.get("orientation", 1) in (5, 6, 7, 8)

        def upright(b):
            w, h = b[2] - b[0], b[3] - b[1]
            return (h, w) if rot else (w, h)

        b0, b1 = k2_bucket(*upright(before)), k2_bucket(*upright(box))
        if b0[1] is not None and b1 != b0:
            plan["note"] = (f"line kept: k2prep bucket {b0[0]} {b0[1]} would become "
                            f"{b1[0]} {b1[1] if b1[1] else 'none'}")
            box = before
    plan["box"] = box
    plan["lost"] = [box[0] - result["box"][0], box[1] - result["box"][1]] if jpeg else [0, 0]
    x0, y0, x1, y1 = box
    if box == [0, 0, W, H] or x1 - x0 < 1 or y1 - y0 < 1:
        return plan
    if (x1 - x0) * (y1 - y0) < min_area:
        plan["write"] = "too small"
    else:
        plan["write"] = {KIND_JPEG: "jpeg", KIND_LOSSLESS: "exact", KIND_LOSSY: "png"}[head["kind"]]
    return plan


# --- per-image API --------------------------------------------------------------
# What the pipeline tool calls, one image at a time, on an image it decoded
# itself: plan() on the upright (EXIF-transposed) pixels and the header, then
# apply() for the box in the coordinates of that array. The analysis runs on
# the stored pixels, as the command line does, so both give the same box.

def to_stored(a: np.ndarray, orientation: int) -> np.ndarray:
    """The upright pixels as the file stores them: the inverse of the EXIF
    transpose (ImageOps.exif_transpose). A view where numpy allows one."""
    if orientation == 2:
        return a[:, ::-1]
    if orientation == 3:
        return a[::-1, ::-1]
    if orientation == 4:
        return a[::-1]
    if orientation == 5:
        return a.transpose(1, 0, 2)
    if orientation == 6:
        return np.rot90(a, 1)
    if orientation == 7:
        return a.transpose(1, 0, 2)[::-1, ::-1]
    if orientation == 8:
        return np.rot90(a, -1)
    return a


def to_upright(a: np.ndarray, orientation: int) -> np.ndarray:
    """The stored pixels turned the way the EXIF orientation shows them,
    exactly as ImageOps.exif_transpose turns the image."""
    if orientation == 6:
        return np.rot90(a, -1)
    if orientation == 8:
        return np.rot90(a, 1)
    return to_stored(a, orientation)                 # the flips and transposes are their own inverse


def plan(array: np.ndarray, head: dict, options: dict | None = None) -> dict:
    """The border crop of one image. array: the upright RGB uint8 pixels
    (transparency composited over white, as load_pixels gives them); head: the
    file's read_header dict; options: "dark" (cut dark borders too, default
    True) and "min_area" (default DEFAULT_MIN_AREA).
    -> {"action": action_of, "result": the analysis with its crop plan,
        "box": the crop in stored pixels or None, "display_box": the same in
        upright pixels, "write": jpeg, exact, png, too small or ""}"""
    opts = {"dark": True, "min_area": DEFAULT_MIN_AREA, **(options or {})}
    o = head.get("orientation", 1)
    result = analyse_array(to_stored(array, o), opts["dark"])
    if result.get("cuts"):
        result["plan"] = crop_plan(head, result, opts["min_area"])
    out = {"action": action_of(head, result), "result": result, "box": None, "display_box": None, "write": ""}
    if out["action"] in ("crop", "too small"):
        box = result["plan"]["box"]
        W, H = result["size"]
        out.update(box=list(box), display_box=stored_to_display(box, o, W, H), write=result["plan"]["write"])
    return out


def apply(array: np.ndarray, plan: dict) -> list[int]:
    """The box to keep, in the coordinates of the upright array: plan's
    display box when it crops, else the whole array."""
    h, w = array.shape[:2]
    if plan.get("action") != "crop" or not plan.get("display_box"):
        return [0, 0, w, h]
    x0, y0, x1, y1 = plan["display_box"]
    return [max(0, x0), max(0, y0), min(w, x1), min(h, y1)]


# --- writing ------------------------------------------------------------------
# Every write goes to <name>.part and replaces the target in one step. The
# target keeps the modification time and the read-only flag of the source.

PNG_LEVEL = 6                     # zlib level of written PNGs: the default balance of size and speed
TIFF_KEEP_COMPRESSION = {"raw", "tiff_lzw", "tiff_adobe_deflate", "tiff_deflate", "packbits", "group3", "group4"}


def stored_to_display(box, orientation, sw, sh):
    """A box in stored pixels -> the same box in upright (display) pixels."""
    x0, y0, x1, y1 = box
    if orientation == 2:
        return [sw - x1, y0, sw - x0, y1]
    if orientation == 3:
        return [sw - x1, sh - y1, sw - x0, sh - y0]
    if orientation == 4:
        return [x0, sh - y1, x1, sh - y0]
    if orientation == 5:
        return [y0, x0, y1, x1]
    if orientation == 6:
        return [sh - y1, x0, sh - y0, x1]
    if orientation == 7:
        return [sh - y1, sw - x1, sh - y0, sw - x0]
    if orientation == 8:
        return [y0, sw - x1, y1, sw - x0]
    return [x0, y0, x1, y1]


def exif_for_output(exif_bytes: bytes | None, width: int, height: int) -> bytes | None:
    """The source EXIF without its thumbnail (it shows the whole image) and
    with the new pixel size. The orientation stays: the stored pixels keep it."""
    if not exif_bytes:
        return None
    try:
        ex = Image.Exif()
        ex.load(exif_bytes)
        sub = ex.get_ifd(0x8769)
        if sub:
            sub[0xA002], sub[0xA003] = width, height
        return ex.tobytes()
    except Exception:  # noqa: BLE001 - a damaged EXIF block is dropped, not fatal
        return None


def rewrite_jpeg_header(data: bytes, exif: bytes | None) -> bytes:
    """Replace the EXIF segment (APP1 "Exif") of a JPEG file and drop the MPF
    index (APP2 "MPF"), working on the marker segments before the image data.
    Done on the file bytes, not through jpeglib's marker objects: replacing a
    marker there crashed the process (heap corruption) in reframe."""
    if data[:2] != b"\xff\xd8":
        return data
    out, pos, done_exif = [data[:2]], 2, False
    while pos + 4 <= len(data) and data[pos] == 0xFF:
        marker = data[pos + 1]
        if marker == 0xDA:                           # start of scan: the rest is image data
            break
        if 0xD0 <= marker <= 0xD7 or marker == 0x01:
            out.append(data[pos:pos + 2])
            pos += 2
            continue
        length = int.from_bytes(data[pos + 2:pos + 4], "big")
        seg = data[pos:pos + 2 + length]
        body = seg[4:]
        if marker == 0xE1 and body[:6] == b"Exif\x00\x00" and not done_exif:
            done_exif = True
            if exif and len(exif) + 2 <= 0xFFFF:
                seg = b"\xff\xe1" + (len(exif) + 2).to_bytes(2, "big") + exif
        elif marker == 0xE2 and body[:4] == b"MPF\x00":
            seg = b""                                # points at images the crop does not carry
        out.append(seg)
        pos += 2 + length
    out.append(data[pos:])
    return b"".join(out)


def part_path(dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    return dst.with_name(dst.name + ".part")


def finish_write(tmp: Path, dst: Path, src_stat: os.stat_result) -> None:
    """Move tmp over dst; dst gets the source's times and read-only flag."""
    if dst.exists() and not os.access(dst, os.W_OK):
        os.chmod(dst, stat.S_IWRITE | stat.S_IREAD)
    os.replace(tmp, dst)
    os.utime(dst, ns=(src_stat.st_atime_ns, src_stat.st_mtime_ns))
    if not src_stat.st_mode & stat.S_IWRITE:
        os.chmod(dst, stat.S_IREAD)


def jpeglib_dir() -> Path:
    """A new ASCII-only folder for jpeglib's files; the caller removes it.
    libjpeg opens paths as narrow strings: a Cyrillic path fails to read, and a
    write lands under a garbled name (UTF-8 bytes read as the ANSI code page)
    instead of the target."""
    base = Path(tempfile.gettempdir())
    if not str(base).isascii():
        base = Path(Path.cwd().anchor or "C:/")
    return Path(tempfile.mkdtemp(prefix="remove_borders_", dir=base))


def write_jpeg(src: Path, dst: Path, box) -> None:
    """Crop whole DCT blocks with jpeglib: no decoding, no re-encoding. The box
    is in stored pixels and starts on the MCU grid. jpeglib only sees ASCII
    paths in jpeglib_dir(); the result reaches dst through Python. Run in worker
    processes only: jpeglib corrupts the heap when Pillow runs in other threads."""
    import jpeglib
    st = os.stat(src)
    work = jpeglib_dir()
    try:
        shutil.copyfile(src, work / "in.jpg")
        im = jpeglib.read_dct(str(work / "in.jpg"))
        crop_dct(im, box)
        im.write_dct(str(work / "out.jpg"))
        data = (work / "out.jpg").read_bytes()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    w, h = box[2] - box[0], box[3] - box[1]
    with Image.open(src) as orig:
        exif = exif_for_output(orig.info.get("exif"), w, h)
    tmp = part_path(dst)
    tmp.write_bytes(rewrite_jpeg_header(data, exif))
    finish_write(tmp, dst, st)


def crop_dct(im, box) -> None:
    """Cut the DCT coefficient arrays of a jpeglib image to the box."""
    sf = np.asarray(im.samp_factor)                  # rows: (vertical, horizontal) per component
    maxv, maxh = int(sf[:, 0].max()), int(sf[:, 1].max())
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    for i, name in enumerate(("Y", "Cb", "Cr", "K")):
        arr = getattr(im, name, None)
        if arr is None or i >= len(sf):
            continue
        v, hh = int(sf[i][0]), int(sf[i][1])
        bx0, by0 = x0 * hh // maxh // 8, y0 * v // maxv // 8
        bw, bh = math.ceil(w * hh / maxh / 8), math.ceil(h * v / maxv / 8)
        setattr(im, name, arr[by0:by0 + bh, bx0:bx0 + bw].copy())
    im.width, im.height = w, h


def save_kwargs(im: Image.Image, fmt: str, w: int, h: int) -> dict:
    """The metadata of im that the saved crop keeps."""
    info = im.info
    kw = {}
    exif = exif_for_output(info.get("exif"), w, h)
    if exif and fmt in ("PNG", "WEBP", "TIFF"):
        kw["exif"] = exif
    if info.get("icc_profile") and fmt in ("PNG", "WEBP", "TIFF"):
        kw["icc_profile"] = info["icc_profile"]
    if info.get("dpi") and fmt in ("PNG", "TIFF", "BMP"):
        kw["dpi"] = info["dpi"]
    if "transparency" in info and fmt in ("PNG", "GIF"):
        kw["transparency"] = info["transparency"]
    if fmt == "PNG":
        from PIL import PngImagePlugin
        text = getattr(im, "text", None) or {}
        if text:
            meta = PngImagePlugin.PngInfo()
            for k, v in text.items():
                meta.add_text(k, v if isinstance(v, str) else str(v))
            kw["pnginfo"] = meta
        kw["compress_level"] = PNG_LEVEL
    elif fmt == "WEBP":
        kw.update(lossless=True, quality=100, method=4, exact=True)
    elif fmt == "TIFF":
        comp = str(info.get("compression", "raw"))
        kw["compression"] = comp if comp in TIFF_KEEP_COMPRESSION else "tiff_lzw"
    return kw


def write_exact(src: Path, dst: Path, box, bits: int = 8) -> None:
    """Crop a lossless image to the exact pixel and save it in its own format
    and mode. 16-bit colour PNG goes through OpenCV: Pillow reads it as 8 bit."""
    st = os.stat(src)
    tmp = part_path(dst)
    x0, y0, x1, y1 = box
    with Image.open(src) as im:
        fmt, mode = im.format, im.mode
        if fmt == "PNG" and bits > 8 and not mode.startswith("I"):
            import cv2
            arr = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            ok, buf = cv2.imencode(".png", np.ascontiguousarray(arr[y0:y1, x0:x1]))
            if not ok:
                raise OSError(f"OpenCV could not encode {src.name}")
            tmp.write_bytes(buf.tobytes())
            finish_write(tmp, dst, st)
            return
        im.load()
        out = im.crop((x0, y0, x1, y1))
        out.info = dict(im.info)
        kw = save_kwargs(im, fmt, x1 - x0, y1 - y0)
    out.save(tmp, format=fmt, **kw)
    finish_write(tmp, dst, st)


def write_png(src: Path, dst: Path, box) -> None:
    """Crop a lossy image (WebP, AVIF, HEIF, JPEG-in-TIFF) to the exact pixel of
    its decoded image and save it as PNG, with its ICC profile and EXIF."""
    st = os.stat(src)
    tmp = part_path(dst)
    x0, y0, x1, y1 = box
    with Image.open(src) as im:
        im.load()
        out = im.crop((x0, y0, x1, y1))
        if out.mode not in ("RGB", "RGBA", "L", "LA", "P", "I;16", "I"):
            out = out.convert("RGBA" if "A" in out.mode else "RGB")
        kw = save_kwargs(im, "PNG", x1 - x0, y1 - y0)
    out.save(tmp, format="PNG", **kw)
    finish_write(tmp, dst, st)


def png_target(src: Path) -> Path:
    """Where a lossy image goes as PNG: next to it, same stem."""
    return src.with_suffix(".png")


def write_mask(mask: Path, dst: Path, display_box) -> str:
    """Crop a face_masks mask (made from the upright image) with the image's
    box in upright pixels. "" when done, else why not."""
    with Image.open(mask) as m:
        x0, y0, x1, y1 = display_box
        if x1 > m.width or y1 > m.height:
            return f"mask is {m.width}x{m.height}, smaller than the crop box"
    write_exact(mask, dst, display_box)
    return ""


def write_crop(src: str, plan: dict, head: dict, mask: str | None = None) -> dict:
    """Write one planned crop in place (a lossy format: as PNG next to it) and
    crop its mask. Runs in a worker process. -> {"out": path written, "mask": note}
    or {"error": ...}."""
    try:
        src_p = Path(src)
        kind = plan["write"]
        if kind == "jpeg":
            dst = src_p
            write_jpeg(src_p, dst, plan["box"])
        elif kind == "exact":
            dst = src_p
            write_exact(src_p, dst, plan["box"], head.get("bits", 8))
        elif kind == "png":
            dst = png_target(src_p)
            write_png(src_p, dst, plan["box"])
        else:
            return {"error": f"nothing to write for {kind!r}"}
        out = {"out": str(dst), "mask": ""}
        if mask:
            W, H = head["width"], head["height"]
            out["mask"] = write_mask(Path(mask), Path(mask),
                                     stored_to_display(plan["box"], head.get("orientation", 1), W, H))
        return out
    except Exception as e:  # noqa: BLE001 - one bad file is reported, not fatal
        return {"error": f"{type(e).__name__}: {e}".strip()}


# --- report -------------------------------------------------------------------

def final_box(result: dict) -> list[int]:
    return result.get("plan", {}).get("box") or result["box"]


def side_totals(result: dict) -> list[int]:
    """Pixels cut at the top, bottom, left and right, after the crop plan."""
    if not result.get("cuts"):
        return [0, 0, 0, 0]
    (W, H), (x0, y0, x1, y1) = result["size"], final_box(result)
    return [y0, H - y1, x0, W - x1]


def report_rows(items) -> list[list]:
    rows = []
    for it in items:
        h, r = it.head, it.result
        mcu = f"{h['mcu'][0]}x{h['mcu'][1]}" if h.get("mcu") else ""
        action = it.action
        plan = r.get("plan", {})
        reason = (skip_reason(h) or r.get("error", "") or r.get("review", "") or plan.get("note", "")
                  or r.get("note", ""))
        if action == "skip" and r.get("error"):
            reason = "analysis failed: " + r["error"]
        box = final_box(r) if r.get("box") else None
        nw, nh = (box[2] - box[0], box[3] - box[1]) if box and action in ("crop", "too small") else ("", "")
        totals = side_totals(r) if action in ("crop", "review", "too small") else ["", "", "", ""]
        lost = plan.get("lost", [0, 0])
        rows.append([it.rel, h["format"], h["kind"], h["mode"], h["bits"], h["width"], h["height"], mcu,
                     h["orientation"], h["frames"], ";".join(p.name for p in it.sidecars),
                     it.mask.name if it.mask else "", action, plan.get("write", "") if action == "crop" else "",
                     nw, nh, *totals, f"{lost[0]},{lost[1]}" if any(lost) else "",
                     describe_cuts(r.get("cuts", [])), reason, r.get("done", "")])
    return rows


def write_report(root: Path, items) -> Path:
    run_dir = root / BACKUP_DIRNAME / RUN_DIRNAME
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / REPORT_NAME
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(REPORT_FIELDS)
        w.writerows(report_rows(items))
    os.replace(tmp, path)
    return path


def write_plan(root: Path, items, dark: bool) -> Path:
    path = root / BACKUP_DIRNAME / RUN_DIRNAME / PLAN_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    data ={"detector": DETECTOR_VERSION, "dark": dark,
            "images": {it.rel: {"action": it.action, **it.result} for it in items if it.result}}
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return path


def summary(root: Path, items, min_area: int) -> list[str]:
    def count(pred):
        return sum(1 for it in items if pred(it))
    by_fmt: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for it in items:
        if it.head["error"]:
            continue
        by_fmt[it.head["format"]] = by_fmt.get(it.head["format"], 0) + 1
        by_kind[it.head["kind"]] = by_kind.get(it.head["kind"], 0) + 1
    ok = [it for it in items if not it.head["error"]]
    areas = sorted(it.head["width"] * it.head["height"] for it in ok)
    lines = [f"{root}: {len(items)} images"]
    if by_fmt:
        lines.append("  formats: " + ", ".join(f"{k} {v}" for k, v in sorted(by_fmt.items(), key=lambda kv: -kv[1])))
        lines.append("  crop:    " + ", ".join(f"{k} {v}" for k, v in sorted(by_kind.items(), key=lambda kv: -kv[1])))
    if areas:
        lines.append(f"  area:    min {areas[0]:,}, median {areas[len(areas) // 2]:,}, max {areas[-1]:,} px; "
                     f"{sum(a < min_area for a in areas)} below {min_area:,} already")
    lines.append(f"  sidecars: {count(lambda it: it.sidecars)} images with, {count(lambda it: not it.sidecars)} without; "
                 f"{count(lambda it: it.shared_stem)} share their stem with another image")
    lines.append(f"  masks:   {count(lambda it: it.mask)}")
    extra = [f"{count(lambda it: it.head['frames'] > 1 and not it.head['error'])} animated (skipped)",
             f"{count(lambda it: it.head['error'])} unreadable (skipped)",
             f"{count(lambda it: it.head['bits'] > 8)} with more than 8 bits per sample",
             f"{count(lambda it: it.head['orientation'] != 1)} with EXIF rotation"]
    lines.append("  other:   " + ", ".join(extra))
    acts = {a: count(lambda it, a=a: it.action == a) for a in ("crop", "review", "skip", "too small")}
    kinds: dict[str, int] = {}
    for it in items:
        if it.action == "crop":
            for k in sorted({c["kind"] for c in it.result["cuts"]}):
                kinds[k] = kinds.get(k, 0) + 1
    lines.append(f"  borders: {acts['crop']} to crop"
                 + (" (" + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())) + ")" if kinds else "")
                 + f", {acts['too small']} too small after the crop, {acts['review']} for review, "
                 f"{acts['skip']} skipped")
    kept = count(lambda it: it.result.get("cuts") and not it.result.get("review")
                 and it.result.get("plan", {}).get("note", "").startswith("line kept"))
    if kept:
        lines.append(f"  lines:   {kept} thin JPEG lines left in place (k2prep bucket would change)")
    return lines


# --- contact sheets -------------------------------------------------------------
# One tile per image: a thumbnail with the box that stays drawn in red, and the
# four corners of that box magnified, so a cut of 1 or 2 px can be judged too.

THUMB = 300                       # px, longest side of the thumbnail
ZOOM_SRC, ZOOM = 36, 4            # each corner zoom shows 36x36 source px at 4x
TILE_W, TILE_H = THUMB + 8 + 2 * ZOOM_SRC * ZOOM + 4, THUMB + 38
SHEET_COLS, SHEET_ROWS = 3, 4
SHEET_GROUPS = ("frame", "border", "banner", "review")


def tile_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def make_tile(path: str, result: dict, label: str) -> tuple:
    """One sheet tile as (width, height, RGB bytes); runs in a worker process."""
    try:
        im = Image.fromarray(load_pixels(path))
    except Exception as e:  # noqa: BLE001 - a tile that cannot be drawn says why
        tile = Image.new("RGB", (TILE_W, TILE_H), (40, 40, 48))
        ImageDraw.Draw(tile).text((6, 6), f"{label}: {type(e).__name__}: {e}", fill=(255, 120, 120),
                                  font=tile_font(13))
        return tile.size[0], tile.size[1], tile.tobytes()
    return tile_of(im, result, label)


def tile_of(im: Image.Image, result: dict, label: str) -> tuple:
    """The sheet tile of an image already decoded (stored pixels, RGB) and its
    analysis: the thumbnail with the box, the four corners magnified, the label."""
    tile = Image.new("RGB", (TILE_W, TILE_H), (40, 40, 48))
    d = ImageDraw.Draw(tile)
    W, H = im.size
    x0, y0, x1, y1 = final_box(result)
    sc = min(THUMB / W, THUMB / H)
    tile.paste(im.resize((max(1, round(W * sc)), max(1, round(H * sc))), Image.BILINEAR), (0, 0))
    d.rectangle((x0 * sc, y0 * sc, x1 * sc - 1, y1 * sc - 1), outline=(255, 0, 0))
    half = ZOOM_SRC // 2
    for i, (cx, cy) in enumerate(((x0, y0), (x1, y0), (x0, y1), (x1, y1))):
        ox, oy = cx - half, cy - half
        z = Image.new("RGB", (ZOOM_SRC, ZOOM_SRC), (60, 0, 60))
        z.paste(im.crop((max(ox, 0), max(oy, 0), min(ox + ZOOM_SRC, W), min(oy + ZOOM_SRC, H))),
                (max(-ox, 0), max(-oy, 0)))
        z = z.resize((ZOOM_SRC * ZOOM, ZOOM_SRC * ZOOM), Image.NEAREST)
        zd = ImageDraw.Draw(z)
        for xx in (x0, x1):
            if ox <= xx <= ox + ZOOM_SRC:
                zd.line(((xx - ox) * ZOOM, 0, (xx - ox) * ZOOM, ZOOM_SRC * ZOOM), fill=(255, 0, 0))
        for yy in (y0, y1):
            if oy <= yy <= oy + ZOOM_SRC:
                zd.line((0, (yy - oy) * ZOOM, ZOOM_SRC * ZOOM, (yy - oy) * ZOOM), fill=(255, 0, 0))
        tile.paste(z, (THUMB + 8 + (i % 2) * (ZOOM_SRC * ZOOM + 4), (i // 2) * (ZOOM_SRC * ZOOM + 4)))
    font = tile_font(13)
    d.text((4, THUMB + 3), label, fill=(255, 230, 0), font=font)
    note = result.get("review") or ("too small after the crop" if result.get("plan", {}).get("write") == "too small"
                                     else describe_cuts(result["cuts"]))
    d.text((4, THUMB + 20), note[:110],
           fill=(190, 255, 190), font=font)
    return tile.size[0], tile.size[1], tile.tobytes()


def sheet_group(it) -> str:
    if it.action in ("review", "too small"):
        return "review"
    kinds = {c["kind"] for c in it.result["cuts"]}
    return "frame" if "frame" in kinds else "border" if kinds & {"border", "line"} else "banner"


def write_sheets(root: Path, items, threads: int) -> tuple[Path, int]:
    """Contact sheets of every image to crop or review, one set per group:
    sheets/frame_001.jpg, border_..., banner_..., review_... The sheets of the
    last run are replaced."""
    out = root / BACKUP_DIRNAME / RUN_DIRNAME / SHEETS_DIRNAME
    if out.is_dir():
        for f in out.glob("*.jpg"):
            f.unlink()
    out.mkdir(parents=True, exist_ok=True)
    chosen = [it for it in items if it.action in ("crop", "review", "too small")]
    groups = {g: [it for it in chosen if sheet_group(it) == g] for g in SHEET_GROUPS}
    order = [it for g in SHEET_GROUPS for it in groups[g]]
    tiles = iter(run_pool(make_tile, [[str(it.path) for it in order], [it.result for it in order],
                                      [f"{it.rel}  {it.head['width']}x{it.head['height']}" for it in order]],
                          threads))
    per = SHEET_COLS * SHEET_ROWS
    written = 0
    for g in SHEET_GROUPS:
        members = groups[g]
        for start in range(0, len(members), per):
            chunk = members[start:start + per]
            rows = (len(chunk) + SHEET_COLS - 1) // SHEET_COLS
            sheet = Image.new("RGB", (SHEET_COLS * (TILE_W + 8), rows * (TILE_H + 8)), (20, 20, 24))
            for k in range(len(chunk)):
                w, h, data = next(tiles)
                sheet.paste(Image.frombytes("RGB", (w, h), data),
                            ((k % SHEET_COLS) * (TILE_W + 8) + 4, (k // SHEET_COLS) * (TILE_H + 8) + 4))
            sheet.save(out / f"{g}_{start // per + 1:03d}.jpg", quality=88)
            written += 1
    return out, written


# --- real run -------------------------------------------------------------------
# Order per image: backup copies first, then the write, so the original is
# always in _backup before the image changes. A backup that exists already is
# never overwritten (it holds the first original). Every image's job is logged
# before it starts and again when it ends, so an interrupted run can be undone.


def same_bytes(a: Path, b: Path) -> bool:
    try:
        return a.stat().st_size == b.stat().st_size and filecmp.cmp(a, b, shallow=False)
    except OSError:
        return False


def make_writable(path: Path) -> None:
    if path.exists() and not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)


def copy_keep(src: Path, dst: Path) -> str:
    """Copy src to dst unless dst exists. -> copied, same (dst has these bytes
    already) or kept (dst holds other bytes: an earlier original, left alone)."""
    if dst.exists():
        return "same" if same_bytes(src, dst) else "kept"
    tmp = part_path(dst)
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return "copied"


def free_name(dst: Path) -> Path:
    """dst, or dst with " (2)", " (3)"... before the extension when taken."""
    n = 2
    out = dst
    while out.exists():
        out = dst.with_name(f"{dst.stem} ({n}){dst.suffix}")
        n += 1
    return out


def move_keep(src: Path, dst: Path) -> Path:
    """Move src to dst. When dst holds the same bytes already, src is deleted;
    when it holds other bytes, src goes to a free name next to it."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and same_bytes(src, dst):
        make_writable(src)
        src.unlink()
        return dst
    dst = free_name(dst)
    try:
        os.replace(src, dst)
    except OSError:                               # another drive
        shutil.copy2(src, dst)
        make_writable(src)
        src.unlink()
    return dst


def backup_of(root: Path, path: Path) -> Path:
    return root / BACKUP_DIRNAME / path.relative_to(root)


def crop_job(job: dict) -> dict:
    """One crop: backup copies of the image, its sidecars and its mask, the
    write, and for a lossy format the move of the original. Runs in a worker
    process (jpeglib). -> the log record of what was done."""
    root, src = Path(job["root"]), Path(job["path"])
    rec = {"op": "crop", "rel": job["rel"], "write": job["plan"]["write"], "box": job["plan"]["box"]}
    try:
        lossy = job["plan"]["write"] == "png"
        if not lossy:
            rec["image"] = copy_keep(src, backup_of(root, src))
        rec["sidecars"] = {Path(sc).name: copy_keep(Path(sc), backup_of(root, Path(sc))) for sc in job["sidecars"]}
        if job["mask"]:
            rec["mask"] = copy_keep(Path(job["mask"]), backup_of(root, Path(job["mask"])))
        out = write_crop(job["path"], job["plan"], job["head"], job["mask"])
        if "error" in out:
            rec["error"] = out["error"]
            return rec
        rec["out"] = Path(out["out"]).relative_to(root).as_posix()
        if out.get("mask"):
            rec["mask_note"] = out["mask"]
        if lossy:
            rec["moved"] = move_keep(src, backup_of(root, src)).relative_to(root).as_posix()
    except Exception as e:  # noqa: BLE001 - one bad file is reported, not fatal
        rec["error"] = f"{type(e).__name__}: {e}".strip()
    return rec


def move_out(root: Path, it) -> dict:
    """A too-small image: the image, its sidecars and its mask move to _backup.
    A sidecar that another image shares (a.jpg and a.png with one a.txt) is
    copied instead, so the other image keeps it."""
    rec = {"op": "move", "rel": it.rel, "moved": {}}
    try:
        files = [(it.path, False)] + [(sc, it.shared_stem) for sc in it.sidecars]
        if it.mask:
            files.append((it.mask, False))
        for f, shared in files:
            if shared:
                rec["moved"][f.relative_to(root).as_posix()] = "copy:" + copy_keep(f, backup_of(root, f))
            else:
                rec["moved"][f.relative_to(root).as_posix()] = move_keep(f, backup_of(root, f)).relative_to(root).as_posix()
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}".strip()
    return rec


class RunLog:
    """log.jsonl in the run folder: one JSON object per line, flushed as written."""

    def __init__(self, root: Path):
        self.path = root / BACKUP_DIRNAME / RUN_DIRNAME / LOG_NAME
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a", encoding="utf-8")

    def write(self, obj: dict) -> None:
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def real_run(root: Path, items, threads: int) -> dict:
    """Crop and move what the plan says. -> counts by outcome."""
    counts = {"cropped": 0, "moved": 0, "failed": 0, "skipped": 0}
    crops = [it for it in items if it.action == "crop"]
    moves = [it for it in items if it.action == "too small"]
    for it in crops:                              # a lossy image whose PNG name is taken stays as it is
        if it.result["plan"]["write"] == "png" and png_target(it.path).exists():
            it.result["done"] = f"skipped: {png_target(it.path).name} exists"
            counts["skipped"] += 1
    crops = [it for it in crops if not it.result.get("done")]
    if not crops and not moves:
        return counts
    log = RunLog(root)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log.write({"run": run_id, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "images": len(crops) + len(moves)})
    try:
        for it in moves:
            log.write({"op": "intent", "rel": it.rel, "action": "move"})
            rec = move_out(root, it)
            log.write(rec)
            it.result["done"] = "error: " + rec["error"] if "error" in rec else "moved to " + BACKUP_DIRNAME
            counts["failed" if "error" in rec else "moved"] += 1
        for it in crops:
            log.write({"op": "intent", "rel": it.rel, "action": "crop", "write": it.result["plan"]["write"],
                       "sidecars": [sc.relative_to(root).as_posix() for sc in it.sidecars],
                       "mask": it.mask.relative_to(root).as_posix() if it.mask else ""})
        jobs = [{"root": str(root), "path": str(it.path), "rel": it.rel, "plan": it.result["plan"], "head": it.head,
                 "sidecars": [str(sc) for sc in it.sidecars], "mask": str(it.mask) if it.mask else None}
                for it in crops]
        progress = Progress(root.name, "cropped", len(crops))
        for n, (it, rec) in enumerate(zip(crops, run_pool(crop_job, [jobs], threads)), 1):
            log.write(rec)
            if "error" in rec:
                it.result["done"] = "error: " + rec["error"]
                counts["failed"] += 1
            else:
                it.result["done"] = "cropped" + (" into " + Path(rec["out"]).name if rec["write"] == "png" else "")
                counts["cropped"] += 1
            progress.step(n)
        log.write({"end": run_id, **counts})
    finally:
        log.close()
    return counts


# --- undo -------------------------------------------------------------------------

def read_log(root: Path) -> list[dict]:
    path = root / BACKUP_DIRNAME / RUN_DIRNAME / LOG_NAME
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:                        # a line cut short by a crash
            continue
    return out


def last_run(entries: list[dict]) -> tuple[str | None, list[dict]]:
    """The id and records of the last run that was not undone."""
    undone = {e["undo"] for e in entries if "undo" in e}
    runs = [i for i, e in enumerate(entries) if "run" in e and e["run"] not in undone]
    if not runs:
        return None, []
    i = runs[-1]
    j = next((k for k in range(i + 1, len(entries)) if "run" in entries[k]), len(entries))
    return entries[i]["run"], entries[i + 1:j]


def restore_file(root: Path, rel: str, keep_backup: bool) -> str:
    """Put _backup/rel back at rel. keep_backup: copy instead of move (the
    backup is an earlier run's original). -> what happened."""
    src, dst = root / BACKUP_DIRNAME / rel, root / rel
    if not src.exists():
        return "no backup"
    make_writable(dst)
    tmp = part_path(dst)
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    if not keep_backup:
        make_writable(src)
        src.unlink()
    return "restored"


def undo_root(root: Path) -> dict:
    """Undo the last run of root. -> counts."""
    entries = read_log(root)
    run_id, recs = last_run(entries)
    counts = {"restored": 0, "moved back": 0, "conflicts": 0}
    if run_id is None:
        return counts
    done = {r["rel"]: r for r in recs if r.get("op") in ("crop", "move")}
    intents = [r for r in recs if r.get("op") == "intent"]
    bdir = root / BACKUP_DIRNAME
    for intent in reversed(intents):
        rel = intent["rel"]
        rec = done.get(rel, {})
        if intent["action"] == "move":
            for orig, where in (rec.get("moved") or {}).items():
                if where.startswith("copy:"):
                    if where == "copy:copied":
                        (bdir / orig).unlink(missing_ok=True)
                    continue
                if (root / orig).exists():
                    counts["conflicts"] += 1
                    print(f"  not moved back, {orig} exists again")
                    continue
                (root / orig).parent.mkdir(parents=True, exist_ok=True)
                os.replace(root / where, root / orig)
            counts["moved back"] += 1
            continue
        # a crop: the image (or, for a lossy format, the original moved out and the PNG written)
        if intent.get("write") == "png":
            # Without a record (the run died) the steps are inferred: the PNG name was
            # free when the run started, and the original goes to _backup under its own name.
            moved = root / (rec.get("moved") or f"{BACKUP_DIRNAME}/{rel}")
            if not (root / rel).exists() and moved.exists():
                os.replace(moved, root / rel)
                counts["restored"] += 1
            out = root / (rec.get("out") or png_target(Path(rel)).as_posix())
            if (root / rel).exists() and out.exists():
                make_writable(out)
                out.unlink()
        else:
            keep = rec.get("image") == "kept"
            if restore_file(root, rel, keep_backup=keep) == "restored":
                counts["restored"] += 1
        if intent.get("mask"):
            restore_file(root, intent["mask"], keep_backup=rec.get("mask") == "kept")
        for sc in intent.get("sidecars", []):
            status = (rec.get("sidecars") or {}).get(Path(sc).name)
            b = bdir / sc
            if not (root / sc).exists() and b.exists():
                os.replace(b, root / sc)           # the caption went missing: put the copy back
            elif status in ("copied", None) and b.exists() and same_bytes(b, root / sc):
                make_writable(b)
                b.unlink()                        # the caption in place is untouched; drop the copy
    log = RunLog(root)
    log.write({"undo": run_id, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **counts})
    log.close()
    remove_empty_dirs(bdir)
    return counts


def remove_empty_dirs(top: Path) -> None:
    """Remove empty folders under top, deepest first; the run folder stays."""
    if not top.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(top, topdown=False):
        d = Path(dirpath)
        if d == top or d.name == RUN_DIRNAME:
            continue
        try:
            d.rmdir()
        except OSError:
            pass


# --- command line -------------------------------------------------------------

def ignore_ctrl_c() -> None:
    """Worker initializer: Ctrl+C stops the main process only, so a worker
    finishes the image it is on (a crop is never left half written) and the
    main process cancels the rest."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_pool(func, arg_lists, threads: int):
    """func over the zipped argument lists, in order: in worker processes, or
    in this process for one thread or one item. When the caller stops early
    (Ctrl+C, an error), the images not yet started are cancelled at once."""
    n = len(arg_lists[0])
    if threads == 1 or n <= 1:
        yield from (func(*args) for args in zip(*arg_lists))
        return
    pool = ProcessPoolExecutor(max_workers=threads, initializer=ignore_ctrl_c)
    finished = False
    try:
        yield from pool.map(func, *arg_lists, chunksize=4)
        finished = True
    finally:
        pool.shutdown(wait=finished, cancel_futures=not finished)


class Progress:
    """A progress line every PROGRESS_EVERY images, with the time left."""

    def __init__(self, label: str, verb: str, total: int):
        self.label, self.verb, self.total = label, verb, total
        self.t0 = time.perf_counter()

    def step(self, n: int) -> None:
        if n % PROGRESS_EVERY and n != self.total or self.total < PROGRESS_EVERY:
            return
        spent = time.perf_counter() - self.t0
        left = spent / n * (self.total - n)
        print(f"  {self.label}: {self.verb} {n:,} of {self.total:,}, {clock(spent)} so far"
              + (f", about {clock(left)} left" if n < self.total else ""), flush=True)


def clock(seconds: float) -> str:
    m, s_ = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s_:02d}" if h else f"{m}:{s_:02d}"


# --- analysis cache ---------------------------------------------------------------
# A real run after a dry run reuses the dry run's analysis. A file counts as
# unchanged while its size, modification time and file ID stay: a crop keeps
# the modification time, but the replaced file gets a new ID.

def file_key(path: Path) -> list[int]:
    st = os.stat(path)
    return [st.st_size, st.st_mtime_ns, st.st_ino]


def load_cache(root: Path, dark: bool) -> dict:
    path = root / BACKUP_DIRNAME / RUN_DIRNAME / CACHE_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if data.get("detector") != DETECTOR_VERSION or data.get("dark") != dark:
        return {}
    return data.get("files", {})


def save_cache(root: Path, items, dark: bool) -> None:
    files = {}
    for it in items:
        r = it.result
        if r and not r.get("error") and it.key:
            files[it.rel] = {"key": it.key, "result": {k: v for k, v in r.items() if k not in ("plan", "done")}}
    path = root / BACKUP_DIRNAME / RUN_DIRNAME / CACHE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps({"detector": DETECTOR_VERSION, "dark": dark, "files": files}, ensure_ascii=False),
                   encoding="utf-8")
    os.replace(tmp, path)


def analyse_items(items, dark: bool, threads: int, label: str = "", cache: dict | None = None) -> int:
    """Fill item.result for every image that is not skipped, from the cache
    when the file is unchanged, else in worker processes (numpy work on whole
    decoded images). -> how many came from the cache."""
    todo, reused = [], 0
    for it in items:
        if skip_reason(it.head):
            continue
        try:
            it.key = file_key(it.path)
        except OSError:
            it.key = None
        hit = (cache or {}).get(it.rel)
        if hit and it.key and hit["key"] == it.key:
            it.result = hit["result"]
            reused += 1
        else:
            todo.append(it)
    progress = Progress(label, "analysed", len(todo))
    results = run_pool(analyse_file, [[str(it.path) for it in todo], [dark] * len(todo)], threads)
    for n, (it, res) in enumerate(zip(todo, results), 1):
        it.result = res
        progress.step(n)
    return reused


def threads_arg(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--threads must be an integer, got {value!r}")
    if not 1 <= n <= 64:
        raise argparse.ArgumentTypeError(f"--threads must be 1..64, got {n}")
    return n


def check_roots(paths) -> list[Path]:
    roots = []
    for p in paths:
        r = Path(p).resolve()
        if not r.is_dir():
            raise SystemExit(f"Not a folder: {p}")
        roots.append(r)
    for a in roots:
        for b in roots:
            if a != b and a.is_relative_to(b):
                raise SystemExit(f"{a} is inside {b}; give each folder once, without its parent")
    if len(set(roots)) != len(roots):
        raise SystemExit("A folder is given twice")
    return roots


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cut borders off the images of a dataset, in place.")
    ap.add_argument("folders", nargs="+", help="dataset folders, scanned at any depth")
    ap.add_argument("--dry-run", action="store_true",
                    help="analyse and write the plan, report and contact sheets; change nothing")
    ap.add_argument("--no-sheets", action="store_true", help="with --dry-run: skip the contact sheets")
    ap.add_argument("--sheets", action="store_true", help="with a real run: write the contact sheets too")
    ap.add_argument("--undo", action="store_true", help="put back everything the last run of each folder changed")
    ap.add_argument("--exclude", action="append", default=[], metavar="NAME",
                    help="another folder name to skip at any depth; may repeat. Always skipped: "
                         + ", ".join(DEFAULT_EXCLUDES))
    ap.add_argument("--sidecars", default=DEFAULT_SIDECARS, metavar="LIST",
                    help=f"sidecar extensions, comma-separated (default {DEFAULT_SIDECARS})")
    ap.add_argument("--min-area", type=int, default=DEFAULT_MIN_AREA, metavar="N",
                    help=f"an image smaller than N pixels after its crop moves to {BACKUP_DIRNAME} instead "
                         f"(default {DEFAULT_MIN_AREA})")
    ap.add_argument("--no-dark", action="store_true", help="leave dark frames and borders alone")
    ap.add_argument("--reanalyse", action="store_true", help="ignore the analysis cache of the last run")
    ap.add_argument("--threads", type=threads_arg, default=DEFAULT_THREADS, metavar="N",
                    help=f"worker processes (default {DEFAULT_THREADS})")
    args = ap.parse_args(argv)

    if args.undo and args.dry_run:
        ap.error("--undo and --dry-run do not go together")
    sidecar_exts = [e if e.startswith(".") else "." + e for e in (s.strip() for s in args.sidecars.split(",")) if e]
    roots = check_roots(args.folders)
    excludes = DEFAULT_EXCLUDES + args.exclude

    if args.undo:
        for root in roots:
            c = undo_root(root)
            if not any(c.values()):
                print(f"{root}: no run to undo")
            else:
                print(f"{root}: {c['restored']} restored, {c['moved back']} moved back"
                      + (f", {c['conflicts']} not moved back (the name is taken again)" if c["conflicts"] else ""))
        return 0

    failed = 0
    for root in roots:
        items = scan(root, excludes, sidecar_exts)
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            for n, (it, head) in enumerate(zip(items, pool.map(read_header, (it.path for it in items))), 1):
                it.head = head
                if n % 5000 == 0:
                    print(f"  {root.name}: read {n:,} of {len(items):,} headers", flush=True)
        dark = not args.no_dark
        cache = {} if args.reanalyse else load_cache(root, dark)
        reused = analyse_items(items, dark, args.threads, root.name, cache)
        if reused:
            print(f"  {root.name}: {reused:,} images unchanged since the last analysis, taken from the cache")
        save_cache(root, items, dark)
        for it in items:
            if it.result.get("cuts"):
                it.result["plan"] = crop_plan(it.head, it.result, args.min_area)
        for line in summary(root, items, args.min_area):
            print(line)
        if (args.dry_run and not args.no_sheets) or (not args.dry_run and args.sheets):
            out, n = write_sheets(root, items, args.threads)
            print(f"  sheets:  {n} in {out}")
        if not args.dry_run:
            c = real_run(root, items, args.threads)
            failed += c["failed"]
            print(f"  done:    {c['cropped']} cropped, {c['moved']} moved to {BACKUP_DIRNAME} (too small), "
                  f"{c['failed']} failed" + (f", {c['skipped']} skipped" if c["skipped"] else ""))
        write_plan(root, items, not args.no_dark)
        print(f"  report:  {write_report(root, items)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
