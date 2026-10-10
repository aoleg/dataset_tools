#!/usr/bin/env python3
"""
Find heavily compressed images in a dataset with the FBCNN quality predictor,
and restore them with FBCNN.

--extract: every image of every folder given is
scanned at any depth, its JPEG quality factor (QF) is predicted from the stored
pixels, and each image under the highest band limit is copied with its sidecars
into one folder per band, outside the dataset, so a threshold can be chosen by
eye. The dataset itself is never changed; the only file written into it is the
measurement cache in <folder>/_backup/_jpeg_cleanup/cache.json.

An extract folder given instead of a dataset folder is sorted again in place,
from its extract.csv and its manifest, without the dataset: new band limits
move the copies between the band folders.

The fix: every image with a QF under --threshold is restored in memory, at
its QF plus --qf-offset, encoded as it would be written, and judged: only a
restoration that removes enough of the artifacts is saved. Its original and
captions go to <folder>/_backup/<same relative path> first; the restoration
then replaces the image under its own name. --dry-run does all of it but the
writing and draws contact sheets (before and after, at 100% and magnified);
--undo puts back what the last run changed; --review draws the sheets of the
last run from the originals in _backup and the files in place. The report, the log and the sheets
are in <folder>/_backup/_jpeg_cleanup.

Usage:    python jpeg_cleanup.py --extract <folder> [<folder> ...] [--out DIR]
          [--bands LIST] [--max-pixels N] [--exclude NAME] [--sidecars LIST]
          [--reanalyse] [--threads N]
          python jpeg_cleanup.py <folder> [<folder> ...] [--dry-run] [--threshold QF]
          [--qf-offset N] [--min-block-drop X] [--min-qf-gain N] [--sheet-offsets LIST]
          [--sheets | --no-sheets] [--quality Q] [--max-pixels N] [--exclude NAME]
          [--sidecars LIST] [--reanalyse] [--threads N]
          python jpeg_cleanup.py <folder> [<folder> ...] --undo | --review
          python jpeg_cleanup.py --fetch-models
Install:  install.bat (torch from the PyTorch CUDA index, Pillow, numpy)
"""
import argparse
import csv
import filecmp
import hashlib
import io
import json
import math
import os
import shutil
import stat
import sys
import time
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None     # large scans are normal input, not an attack

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))                 # network_fbcnn.py lives next to this file
MODELS_DIR = HERE / "models"
MODEL_URL = "https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/"
# name: (file, size in bytes, sha256). Both from the FBCNN v1.0 release, Apache 2.0.
MODELS = {
    "color": ("fbcnn_color.pth", 287755111, "8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8"),
    "gray": ("fbcnn_gray_double.pth", 287745895, "4444b8b2393649a4acd5d8730410ec3106d15b6e262a49081e4e87b44cfa5cea"),
}

BACKUP_DIRNAME = "_backup"
RUN_DIRNAME = "_jpeg_cleanup"     # inside _backup: the measurement cache
CACHE_NAME = "cache.json"         # measurements by file, reused while a file is unchanged
EXTRACT_SUFFIX = "_jpeg_extract"  # default output: <dataset folder>_jpeg_extract next to it
EXTRACT_CSV = "extract.csv"
SUMMARY_NAME = "summary.txt"
REPORT_NAME = "report.csv"        # inside _backup/_jpeg_cleanup: the fix, one row per image
SHEETS_DIRNAME = "sheets"         # inside _backup/_jpeg_cleanup: the contact sheets of the last fix run
LOG_NAME = "log.jsonl"            # inside _backup/_jpeg_cleanup: every real run and undo; --undo reads it
ORIGINALS_DIRNAME = "originals"   # inside _backup/_jpeg_cleanup: originals whose place in _backup was taken
MANIFEST_NAME = "manifest.json"   # the files the last extract copied, removed by the next one
MOVES_NAME = "moves.json"         # the moves of a re-sort in progress; the next run finishes them
PROGRESS_EVERY = 500              # images between progress lines

IMAGE_EXTS = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff",
              ".gif", ".avif", ".heic", ".heif"}
# Folders never scanned, at any depth: the output folders of the dataset tools.
# _backup is remove_borders' and this tool's, _duplicates is deduplicate's, _prep
# is k2prep's, _classify and _embeddings are classify's, masks and faces are
# face_masks'. Any other folder is scanned; --exclude adds names.
DEFAULT_EXCLUDES = [BACKUP_DIRNAME, "_duplicates", "_prep", "_classify", "_embeddings", "masks", "faces"]
DEFAULT_SIDECARS = ".txt"
DEFAULT_BANDS = "60,70,80,85"
DEFAULT_MAX_PIXELS = 2048 * 2048  # larger images are skipped: the downscale to 1024^2 hides their artifacts
DEFAULT_THREADS = 8
DEFAULT_THRESHOLD = 80            # the fix restores images with a QF under this
DEFAULT_QF_OFFSET = 10            # added to the predicted QF given to the restoration: the gentle side
DEFAULT_QUALITY = 97              # JPEG quality of a restored JPEG
DEFAULT_MIN_BLOCK_DROP = 0.10     # a restoration is worth saving when the blockiness drops this much...
DEFAULT_MIN_QF_GAIN = 25          # ...or the QF rises this much on an image with a JPEG grid
GRID_MIN = 1.05                   # blockiness from which the 8 x 8 grid shows; dithered prints sit at 1.0

MEASURE_VERSION = 1               # bump when a measurement changes meaning; the cache is then dropped
GRAY_SPREAD = 2                   # an RGB image whose channels differ by at most this everywhere is gray
# Formats whose pixels can carry JPEG artifacts the model knows: JPEG itself, and
# lossless containers that may hold the pixels of a decoded JPEG. Lossy WebP,
# AVIF, HEIF and GIF have artifacts of their own, which FBCNN was not trained on.
MEASURED_FORMATS = {"JPEG", "MPO", "PNG", "BMP", "TIFF", "WEBP"}

CSV_FIELDS = ["path", "format", "mode", "width", "height", "megapixels", "gray", "header_q",
              "qf_color", "qf_gray", "qf", "band", "copy", "status"]

# The IJG (libjpeg) luminance table at quality 50, for the header quality estimate.
STD_LUMA = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]


@dataclass
class Item:
    root: Path
    path: Path
    rel: str                                  # relative to root, forward slashes
    sidecars: list[Path] = field(default_factory=list)
    key: list | None = None                   # file_key when measured, for the cache
    m: dict = field(default_factory=dict)     # the measurement (see measure_one)
    band: int | None = None
    status: str = ""
    copy: str = ""                            # the copy, relative to the output folder
    fix: dict = field(default_factory=dict)   # the fix of the image (see fix_one)


# --- scan ---------------------------------------------------------------------

def scan(root: Path, excludes, sidecar_exts) -> list[Item]:
    """The images under root at any depth, with their sidecars. Folders are
    skipped by exact name (case-insensitive), at any depth."""
    excl = {e.casefold() for e in excludes}
    exts = {e.casefold() for e in sidecar_exts}
    items = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if d.casefold() not in excl)
        side_by_stem: dict[str, list[str]] = {}
        for fn in filenames:
            stem, ext = os.path.splitext(fn)
            if ext.casefold() in exts:
                side_by_stem.setdefault(stem.casefold(), []).append(fn)
        for fn in sorted(filenames):
            stem, ext = os.path.splitext(fn)
            if ext.casefold() not in IMAGE_EXTS:
                continue
            p = here / fn
            items.append(Item(root=root, path=p, rel=p.relative_to(root).as_posix(),
                              sidecars=[here / s for s in sorted(side_by_stem.get(stem.casefold(), []))]))
    return items


# --- decoding -------------------------------------------------------------------

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


def header_quality(im: Image.Image) -> int | None:
    """The IJG quality that would give this luminance table. Exact for libjpeg
    tables; an approximation for other encoders (Photoshop, cameras)."""
    tables = getattr(im, "quantization", None)
    if not tables or 0 not in tables or len(tables[0]) != 64:
        return None
    scale = 100.0 * sum(tables[0]) / sum(STD_LUMA)
    q = (200.0 - scale) / 2.0 if scale <= 100.0 else 5000.0 / scale
    return int(min(100, max(1, round(q))))


def facts_of(im: Image.Image, path: Path | None = None) -> dict:
    """The facts of an open image that decide and describe its measurement:
    format, mode, stored size, EXIF orientation, EXIF and ICC blocks, the
    header quality of a JPEG, the frame count and whether a WebP is lossless
    (path needed for that; without it a WebP counts as lossy)."""
    fmt = im.format or ""
    d = {"format": fmt, "mode": im.mode, "width": im.size[0], "height": im.size[1], "gray": None,
         "header_q": None, "skip": "", "orientation": 1, "frames": 1, "lossless_webp": False}
    try:
        o = im.getexif().get(274, 1) or 1
        d["orientation"] = o if o in range(1, 9) else 1
    except Exception:  # noqa: BLE001 - a damaged EXIF block is not fatal
        pass
    d["exif"], d["icc"] = im.info.get("exif"), im.info.get("icc_profile")
    if fmt in ("JPEG", "MPO"):
        d["header_q"] = header_quality(im)
    if fmt != "MPO":
        d["frames"] = getattr(im, "n_frames", 1) or 1
    if fmt == "WEBP" and path is not None:
        d["lossless_webp"] = webp_is_lossless(path)
    return d


def skip_of(d: dict, max_pixels: int) -> str:
    """Why an image with these facts is not measured, or ""."""
    fmt = d["format"]
    if fmt not in MEASURED_FORMATS:
        return f"format {fmt or '?'}"
    if fmt == "WEBP" and not d.get("lossless_webp"):
        return "format lossy WEBP"
    if d.get("frames", 1) > 1:
        return "animated"
    if d["mode"] in ("1", "CMYK", "I", "F") or d["mode"].startswith("I;16"):
        return f"mode {d['mode']}"
    if d["width"] * d["height"] > max_pixels:
        return "large"
    if d.get("transparent"):
        return "transparent"
    return ""


def decode(path: Path, max_pixels: int) -> dict:
    """Open one image and return its facts and, when it is to be measured, its
    stored pixels (no EXIF rotation, so the JPEG block grid stays aligned):
    "rgb" uint8 HxWx3 and, for a gray image, "luma" uint8 HxW. A skipped image
    gets "skip" with the reason. Runs in the decoder threads."""
    d = {"format": "", "mode": "", "width": 0, "height": 0, "gray": None, "header_q": None, "skip": "",
         "orientation": 1}
    try:
        with Image.open(path) as im:
            d = facts_of(im, path)
            d["skip"] = skip_of(d, max_pixels)
            if d["skip"]:
                return d
            im.load()
            if "A" in im.mode or (im.mode == "P" and "transparency" in im.info):
                rgba = im.convert("RGBA")
                if rgba.getchannel("A").getextrema()[0] < 255:
                    d["skip"] = "transparent"
                    return d
                im = rgba.convert("RGB")
            if im.mode in ("L", "LA"):
                d["luma"] = np.array(im.convert("L"))
                d["rgb"] = np.repeat(d["luma"][..., None], 3, axis=2)
                d["gray"] = True
                return d
            rgb = np.array(im.convert("RGB"))
    except Exception as e:  # noqa: BLE001 - any unreadable file is reported, not fatal
        d["skip"] = f"unreadable: {type(e).__name__}: {e}".strip()
        return d
    return with_pixels(d, rgb)


def with_pixels(d: dict, rgb: np.ndarray) -> dict:
    """Add the stored RGB pixels to the facts, with the gray test and the luma."""
    d["rgb"] = rgb
    d["gray"] = is_gray(rgb)
    if d["gray"]:
        d["luma"] = np.array(Image.fromarray(rgb).convert("L"))
    return d


def decode_array(array: np.ndarray, head: dict, max_pixels: int) -> dict:
    """decode() for pixels decoded elsewhere: array holds the upright RGB uint8
    pixels, head the facts (facts_of, or a dict with the same keys; an alpha
    channel that was composited away is reported as "transparent": True). The
    pixels are turned back to the stored orientation, where the JPEG grid is."""
    d = {k: head.get(k) for k in ("format", "mode", "width", "height", "header_q", "orientation", "exif", "icc",
                                   "frames", "lossless_webp", "transparent")}
    d.update(format=d["format"] or "", mode=d["mode"] or "", orientation=d["orientation"] or 1,
             frames=d["frames"] or 1, gray=None, skip="")
    d["skip"] = skip_of(d, max_pixels)
    if d["skip"]:
        return d
    return with_pixels(d, np.ascontiguousarray(to_stored(array, d["orientation"])))


def to_stored(a: np.ndarray, orientation: int) -> np.ndarray:
    """The upright pixels as the file stores them: the inverse of the EXIF
    transpose (ImageOps.exif_transpose)."""
    if orientation == 2:
        return a[:, ::-1]
    if orientation == 3:
        return a[::-1, ::-1]
    if orientation == 4:
        return a[::-1]
    if orientation == 5:
        return a.transpose(1, 0, 2) if a.ndim == 3 else a.T
    if orientation == 6:
        return np.rot90(a, 1)
    if orientation == 7:
        return (a.transpose(1, 0, 2) if a.ndim == 3 else a.T)[::-1, ::-1]
    if orientation == 8:
        return np.rot90(a, -1)
    return a


def to_upright(a: np.ndarray, orientation: int) -> np.ndarray:
    """The stored pixels turned the way the EXIF orientation shows them."""
    if orientation == 6:
        return np.rot90(a, -1)
    if orientation == 8:
        return np.rot90(a, 1)
    return to_stored(a, orientation)                 # the flips and transposes are their own inverse


def is_gray(rgb: np.ndarray) -> bool:
    """True when the three channels differ by at most GRAY_SPREAD everywhere.
    Checked in strips, so a large image needs no full-size temporary."""
    for y in range(0, rgb.shape[0], 256):
        s = rgb[y:y + 256].astype(np.int16)
        if (np.abs(s[..., 0] - s[..., 1]).max() > GRAY_SPREAD
                or np.abs(s[..., 1] - s[..., 2]).max() > GRAY_SPREAD):
            return False
    return True


# --- models ---------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_models() -> None:
    """Download the model files into models/ and check their size and sha256.
    A file that is there and correct is kept."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for fname, size, sha in MODELS.values():
        dst = MODELS_DIR / fname
        if dst.exists() and dst.stat().st_size == size and sha256_of(dst) == sha:
            print(f"  {fname}: present, checksum correct")
            continue
        tmp = dst.with_name(fname + ".part")
        print(f"  {fname}: downloading {size / 1e6:.0f} MB from {MODEL_URL}", flush=True)
        with urllib.request.urlopen(MODEL_URL + fname, timeout=60) as r, open(tmp, "wb") as f:
            done, shown = 0, 0
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if done - shown >= 50 << 20:
                    print(f"    {done / 1e6:.0f} of {size / 1e6:.0f} MB", flush=True)
                    shown = done
        got = sha256_of(tmp)
        if tmp.stat().st_size != size or got != sha:
            tmp.unlink()
            raise SystemExit(f"{fname}: the download does not match (size {done}, sha256 {got}); try again")
        os.replace(tmp, dst)
        print(f"  {fname}: downloaded, checksum correct")


class QualityModel:
    """The FBCNN quality predictors, loaded on first use from models/ only.
    The QF needs the encoder half of the network (head, three downsamplings,
    body encoder, qf_pred); the restoration runs the whole network."""

    def __init__(self):
        import torch                              # imported here: a cached run needs no torch
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        torch.backends.cudnn.benchmark = False    # every image has its own size; tuning would repeat
        torch.backends.cudnn.deterministic = True  # a real run decides exactly as its dry run did
        self.nets = {}
        if self.device.type != "cuda":
            print("  [warning] no CUDA GPU: measuring on the CPU, which is much slower", flush=True)

    def net(self, name: str):
        if name not in self.nets:
            from network_fbcnn import FBCNN
            fname, size, _ = MODELS[name]
            path = MODELS_DIR / fname
            if not path.exists() or path.stat().st_size != size:
                raise SystemExit(f"{path} is missing or incomplete; run install.bat or jpeg_cleanup.py --fetch-models")
            nc = 3 if name == "color" else 1
            m = FBCNN(in_nc=nc, out_nc=nc, nc=[64, 128, 256, 512], nb=4, act_mode="R")
            m.load_state_dict(self.torch.load(path, map_location="cpu", weights_only=True), strict=True)
            self.nets[name] = m.eval().to(self.device, self.dtype)
        return self.nets[name]

    def restore(self, name: str, a: np.ndarray, qf: float) -> np.ndarray:
        """The FBCNN restoration of an HxWx3 (color) or HxW (gray) uint8 array,
        told quality qf (0..100), as a uint8 array of the same shape."""
        torch = self.torch
        m = self.net(name)
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(a)).to(self.device)
            x = (x[None, None] if x.ndim == 2 else x.permute(2, 0, 1)[None]).to(self.dtype) / 255.0
            q = torch.tensor([[1.0 - qf / 100.0]], device=self.device, dtype=self.dtype)
            y, _ = m(x, q)
            y = (y.float().clamp(0, 1) * 255.0).round().to(torch.uint8)[0]
            return (y[0] if a.ndim == 2 else y.permute(1, 2, 0)).cpu().numpy()

    def qf(self, name: str, a: np.ndarray) -> float:
        """Predicted QF (0..100) of an HxWx3 (color) or HxW (gray) uint8 array."""
        torch = self.torch
        m = self.net(name)
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(a)).to(self.device)
            x = (x[None, None] if x.ndim == 2 else x.permute(2, 0, 1)[None]).to(self.dtype) / 255.0
            h, w = x.shape[-2:]
            x = torch.nn.functional.pad(x, (0, -w % 8, 0, -h % 8), mode="replicate")
            x = m.m_body_encoder(m.m_down3(m.m_down2(m.m_down1(m.m_head(x)))))
            return float(1.0 - m.qf_pred(x).float().item()) * 100.0


def measure_one(model: QualityModel, d: dict) -> dict:
    """The measurement of a decoded image: its facts, qf_color, and for a gray
    image qf_gray; qf is the value that decides (the gray model's on a gray
    image: the thresholds were set by eye on these values; the fix itself
    restores every image with the colour model, at qf_color)."""
    m = {k: d[k] for k in ("format", "mode", "width", "height", "gray", "header_q", "skip")}
    if d["skip"]:
        return m
    m["qf_color"] = round(model.qf("color", d["rgb"]), 1)
    if d["gray"]:
        m["qf_gray"] = round(model.qf("gray", d["luma"]), 1)
    m["qf"] = m.get("qf_gray", m["qf_color"])
    return m


# --- cache ------------------------------------------------------------------------
# A file counts as unchanged while its size, modification time and file ID stay.

def file_key(path: Path) -> list[int]:
    st = os.stat(path)
    return [st.st_size, st.st_mtime_ns, st.st_ino]


def cache_path(root: Path) -> Path:
    return root / BACKUP_DIRNAME / RUN_DIRNAME / CACHE_NAME


def load_cache(root: Path) -> dict:
    try:
        data = json.loads(cache_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if data.get("measure") != MEASURE_VERSION:
        return {}
    return data.get("files", {})


def save_cache(root: Path, items, old: dict) -> None:
    """Write what is known about the files scanned: the new measurements, and
    the old entries of files not reached this time (an interrupted run). Files
    gone from the dataset drop out."""
    files = {}
    for it in items:
        if it.fix.get("result") == "written":
            continue                              # a restored file is measured again next time
        if it.m and it.key and not it.m.get("skip", "").startswith("unreadable"):
            files[it.rel] = {"key": it.key, "m": it.m}
        elif it.rel in old:
            files[it.rel] = old[it.rel]
    path = cache_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps({"measure": MEASURE_VERSION, "files": files}, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def needs_measure(hit: dict | None, key, max_pixels: int) -> bool:
    """False when the cached measurement still answers: same file, and either
    measured, or skipped for a reason that has not changed (a "large" skip
    under a raised --max-pixels must be measured now)."""
    if not hit or not key or hit["key"] != key:
        return True
    m = hit["m"]
    if m.get("skip") == "large":
        return m["width"] * m["height"] <= max_pixels
    return False


# --- measuring ----------------------------------------------------------------------

def bounded_map(pool, fn, args, window: int):
    """pool.map with at most `window` results waiting, so decoded images do not
    pile up in memory while the GPU works through them."""
    pending = deque()
    it = iter(args)
    for a in it:
        pending.append(pool.submit(fn, a))
        if len(pending) >= window:
            break
    while pending:
        yield pending.popleft().result()
        nxt = next(it, None)
        if nxt is not None:
            pending.append(pool.submit(fn, nxt))


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


def measure_items(root: Path, items, max_pixels: int, threads: int, reanalyse: bool,
                  model_box: list) -> int:
    """Fill item.m for every image: from the cache when the file is unchanged,
    else decoded in threads and measured on the GPU. The cache is saved at the
    end and on Ctrl+C. model_box holds the QualityModel, made on first need.
    -> how many came from the cache."""
    old = {} if reanalyse else load_cache(root)
    todo, reused = [], 0
    for it in items:
        try:
            it.key = file_key(it.path)
        except OSError:
            it.key = None
        hit = old.get(it.rel)
        if needs_measure(hit, it.key, max_pixels):
            todo.append(it)
        else:
            it.m = dict(hit["m"])
            reused += 1
    try:
        if todo:
            if not model_box:
                model_box.append(QualityModel())
            model = model_box[0]
            progress = Progress(root.name, "measured", len(todo))
            with ThreadPoolExecutor(max_workers=threads) as pool:
                decoded = bounded_map(pool, lambda it: decode(it.path, max_pixels), todo, threads * 2)
                for n, (it, d) in enumerate(zip(todo, decoded), 1):
                    it.m = measure_one(model, d)
                    progress.step(n)
    finally:
        save_cache(root, items, old)
    return reused


# --- bands and copies -----------------------------------------------------------------

def parse_bands(text: str) -> list[int]:
    try:
        bands = sorted({int(x) for x in text.split(",") if x.strip()})
    except ValueError:
        raise argparse.ArgumentTypeError(f"--bands must be integers separated by commas, got {text!r}")
    if not bands or bands[0] < 1 or bands[-1] > 100:
        raise argparse.ArgumentTypeError(f"--bands must be between 1 and 100, got {text!r}")
    return bands


def assign(items, bands: list[int], max_pixels: int) -> None:
    """Set band and status of every item. A band is named by its upper limit:
    band 70 holds 60 <= qf < 70 when the limit below is 60."""
    for it in items:
        m = it.m
        it.band = None
        if m.get("skip"):
            it.status = m["skip"]
        elif m["width"] * m["height"] > max_pixels:
            it.status = "large"                   # measured under a higher --max-pixels before
        else:
            it.band = next((b for b in bands if m["qf"] < b), None)
            it.status = "above" if it.band is None else ""


def make_writable(path: Path) -> None:
    if path.exists() and not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)


def remove_empty_dirs(out: Path) -> None:
    """Remove empty folders under out, deepest first."""
    for dirpath, dirnames, filenames in os.walk(out, topdown=False):
        if Path(dirpath) != out:
            try:
                os.rmdir(dirpath)
            except OSError:
                pass


def read_manifest(out: Path) -> dict:
    try:
        return json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_manifest(out: Path, data: dict) -> None:
    tmp = out / (MANIFEST_NAME + ".part")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, out / MANIFEST_NAME)


def clear_previous(out: Path) -> int:
    """Delete the files the last extract into out listed in its manifest (also
    the planned copies of a run that was stopped), and the folders that are
    empty then. Files someone else put there stay. -> files deleted."""
    n = 0
    for rel in read_manifest(out).get("files", []):
        p = out / rel
        if p.is_file():
            make_writable(p)
            p.unlink()
            n += 1
    remove_empty_dirs(out)
    return n


def copy_file(src: Path, dst: Path) -> None:
    """Copy the bytes and the modification time, not the read-only flag, so the
    next extract can delete the copy."""
    shutil.copyfile(src, dst)
    st = os.stat(src)
    os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns))


def plan_copies(out: Path, items) -> list[tuple]:
    """The copies of every item that has a band: out/<band>/q<QF>__<name>,
    with its sidecars under the same stem; a taken name gets ~2, ~3. Sets
    it.copy. -> (item, [(src, dst relative to out)])."""
    plan, taken = [], set()
    for it in sorted((it for it in items if it.band is not None), key=lambda it: (it.m["qf"], it.rel)):
        folder = str(it.band)
        base = f"q{int(math.floor(it.m['qf'])):03d}__{it.path.stem}"
        stem, n = base, 2
        while f"{folder}/{stem}".casefold() in taken or (out / folder / (stem + it.path.suffix)).exists():
            stem, n = f"{base}~{n}", n + 1
        taken.add(f"{folder}/{stem}".casefold())
        it.copy = f"{folder}/{stem}{it.path.suffix}"
        plan.append((it, [(it.path, it.copy)] + [(sc, f"{folder}/{stem}{sc.suffix}") for sc in it.sidecars]))
    return plan


def copy_items(out: Path, plan) -> None:
    for it, files in plan:
        try:
            for src, rel in files:
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                copy_file(src, out / rel)
            it.status = "copied"
        except OSError as e:
            it.copy, it.status = "", f"copy failed: {e}"


# --- re-sorting an extract folder -----------------------------------------------------
# An extract folder holds everything needed to sort its copies again without
# the dataset: extract.csv has the measurements, manifest.json the files. New
# band limits move copies between bands and drop them; images that were above
# the old highest limit were never copied and need a run on the dataset.

def is_extract_folder(p: Path) -> bool:
    return (p / MANIFEST_NAME).is_file() and (p / EXTRACT_CSV).is_file()


def finish_moves(out: Path) -> int:
    """Carry out the moves of a re-sort that was stopped. extract.csv and the
    manifest were written before the first move, so they already describe the
    result. -> files moved now."""
    path = out / MOVES_NAME
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    n = 0
    for src, dst in plan.get("moves", []):
        s, d = out / src, out / dst
        if s.is_file() and not d.exists():
            d.parent.mkdir(parents=True, exist_ok=True)
            os.replace(s, d)
            n += 1
    for rel in plan.get("deletes", []):
        p = out / rel
        if p.is_file():
            make_writable(p)
            p.unlink()
            n += 1
    remove_empty_dirs(out)
    path.unlink()
    return n


def row_measure(r: dict) -> dict:
    """The measurement of an image from its extract.csv row."""
    def num(k, cast=float):
        return cast(r[k]) if r.get(k) else None
    m = {"format": r["format"], "mode": r["mode"], "width": int(r["width"] or 0), "height": int(r["height"] or 0),
         "gray": None if r["gray"] == "" else r["gray"] == "1", "header_q": num("header_q", int), "skip": ""}
    if r.get("qf"):
        m["qf_color"], m["qf"] = num("qf_color"), num("qf")
        if r.get("qf_gray"):
            m["qf_gray"] = num("qf_gray")
    else:
        m["skip"] = r["status"] or "not measured"
    return m


def resort(out: Path, args) -> tuple[list, list[int], str]:
    """Sort the copies of an extract folder into the bands asked for, by
    moving them; copies in subfolders of a band (from an older version) move
    up into the band folder. -> items, bands, dataset root."""
    done = finish_moves(out)
    if done:
        print(f"  finished {done:,} moves of the last re-sort, which was stopped")
    manifest = read_manifest(out)
    root = manifest.get("root", "")
    bands = args.bands or manifest.get("bands") or parse_bands(DEFAULT_BANDS)
    with open(out / EXTRACT_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    items = []
    for r in rows:
        it = Item(root=Path(root), path=Path(root) / r["path"], rel=r["path"], m=row_measure(r))
        it.copy = r.get("copy", "")
        items.append(it)
    old_copy = {it.rel: it.copy for it in items}
    assign(items, bands, args.max_pixels)
    missing = 0
    for it in items:
        if it.band is not None and not old_copy[it.rel]:
            it.band, it.status = None, "not copied"
            missing += 1
    if missing:
        print(f"  [note] {missing:,} images fall under the new band limits but were above the old ones and "
              f"were never copied; run extract on the dataset folder to add them")

    # The sidecar copies of an image copy: files of the manifest in its folder with its stem.
    by_stem: dict[str, list[str]] = {}
    for rel in manifest.get("files", []):
        p = Path(rel)
        by_stem.setdefault(f"{p.parent.as_posix()}/{p.stem}".casefold(), []).append(rel)
    moves, deletes, files = [], [], []
    for it in items:
        oc = old_copy[it.rel]
        if not oc:
            continue
        p = Path(oc)
        group = by_stem.get(f"{p.parent.as_posix()}/{p.stem}".casefold(), [oc])
        if it.band is None:
            deletes += group
            it.copy = ""
            continue
        folder = str(it.band)
        it.copy, it.status = f"{folder}/{p.name}", "copied"
        for rel in group:
            new = f"{folder}/{Path(rel).name}"
            files.append(new)
            if new != rel:
                moves.append([rel, new])
    sources = {s.casefold() for s, _ in moves}
    for _, dst in moves:
        if (out / dst).exists() and dst.casefold() not in sources:
            raise SystemExit(f"{out / dst} is in the way of a move; move it out of the extract folder first")
    # extract.csv and the manifest describe the result before the first move;
    # moves.json lets the next run finish the moves when this one is stopped.
    write_manifest(out, {"root": root, "bands": bands, "files": files})
    write_csv(out, items)
    if moves or deletes:
        tmp = out / (MOVES_NAME + ".part")
        tmp.write_text(json.dumps({"moves": moves, "deletes": deletes}, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, out / MOVES_NAME)
        n = finish_moves(out)
        print(f"  {n:,} files moved or removed")
    return items, bands, root


# --- report ---------------------------------------------------------------------------

def fmt_num(v) -> str:
    return "" if v is None else str(v)


def write_csv(out: Path, items) -> Path:
    path = out / EXTRACT_CSV
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_FIELDS)
        for it in items:
            m = it.m
            wd, ht = m.get("width", 0), m.get("height", 0)
            w.writerow([it.rel, m.get("format", ""), m.get("mode", ""), wd, ht, f"{wd * ht / 1e6:.2f}",
                        "" if m.get("gray") is None else int(m["gray"]), fmt_num(m.get("header_q")),
                        fmt_num(m.get("qf_color")), fmt_num(m.get("qf_gray")), fmt_num(m.get("qf")),
                        fmt_num(it.band), it.copy, it.status])
    os.replace(tmp, path)
    return path


def summary(root, out: Path, items, bands: list[int], max_pixels: int, seconds: float) -> list[str]:
    measured = [it for it in items if "qf" in it.m and it.status != "large"]
    lines = [f"{root}: {len(items):,} images, {len(measured):,} measured "
             f"({sum(1 for it in measured if it.m.get('gray')):,} gray), {clock(seconds)}"]
    lines.append(f"  output:  {out}")
    lines.append("  bands (qf below the limit; each image in one band):")
    lo, total = 0, 0
    for b in bands:
        n = sum(1 for it in measured if it.band == b)
        total += n
        lines.append(f"    {b:>3}  ({lo:>3} <= qf < {b:>3}): {n:>7,}   a threshold of {b} would take {total:,}")
        lo = b
    lines.append(f"    above {bands[-1]}: {sum(1 for it in measured if it.band is None):,} (not copied)")
    if measured:
        lines.append("  histogram of qf, steps of 5:")
        hist: dict[int, int] = {}
        for it in measured:
            k = min(95, int(it.m["qf"] // 5) * 5)
            hist[k] = hist.get(k, 0) + 1
        top = max(hist.values())
        for k in sorted(hist):
            lines.append(f"    {k:>3}-{k + 4:<3} {hist[k]:>7,}  " + "#" * max(1, round(40 * hist[k] / top)))
    reasons: dict[str, int] = {}
    for it in items:
        if it.status and it.status not in ("copied", "above"):
            r = it.status.split(":")[0] if it.status.startswith(("unreadable", "copy failed")) else it.status
            reasons[r] = reasons.get(r, 0) + 1
    if reasons:
        lines.append("  skipped: " + ", ".join(f"{k} {v:,}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]))
                     + (f"   (large = over {max_pixels:,} pixels)" if "large" in reasons else ""))
    return lines


def write_summary(out: Path, lines) -> None:
    (out / SUMMARY_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)


# --- fix: restoration, report and contact sheets ------------------------------------
# A fix restores every image with a QF under --threshold with FBCNN, told the
# image's own QF plus --qf-offset: the higher the QF it is given, the less it
# smooths, so a positive offset keeps more grain and fine texture. A gray image
# goes through the colour model too, and only the luma of the result is kept.
# A restoration is saved only when it removes enough (see benefit). JPEG sources
# are saved as JPEG at --quality with full-resolution colour (4:4:4); lossless
# sources keep their format. The dry run does all of this in memory and writes
# the report and contact sheets only.

def fix_action(it, threshold: int, max_pixels: int) -> str:
    """fix, keep, or the skip reason."""
    return fix_action_of(it.m, threshold, max_pixels)


def encode_jpeg(a: np.ndarray, quality: int) -> bytes:
    """The bytes a JPEG source is written as: 4:4:4, no chroma subsampling, so
    the re-save adds no colour bleeding of its own."""
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG", quality=quality, subsampling=0, optimize=True)
    return buf.getvalue()


def encode_output(y: np.ndarray, d: dict, quality: int) -> bytes:
    """The bytes a restoration is written as, in the format of its source: JPEG
    (also for MPO, whose extra images are dropped) at quality with 4:4:4
    colour; PNG, lossless WebP, TIFF and BMP losslessly. The EXIF block (with
    the orientation) and the ICC profile stay; an RGB profile is left out of a
    gray result, which it does not fit."""
    im = Image.fromarray(y)
    fmt = d["format"]
    kw = {}
    if d.get("icc") and (y.ndim == 3 or d.get("mode") in ("L", "LA")):
        kw["icc_profile"] = d["icc"]
    if d.get("exif") and fmt in ("JPEG", "MPO", "PNG", "WEBP"):
        kw["exif"] = d["exif"]
    buf = io.BytesIO()
    if fmt in ("JPEG", "MPO"):
        im.save(buf, "JPEG", quality=quality, subsampling=0, optimize=True, **kw)
    elif fmt == "WEBP":
        im.save(buf, "WEBP", lossless=True, **kw)
    elif fmt == "TIFF":
        im.save(buf, "TIFF", compression="tiff_lzw", **kw)
    else:
        im.save(buf, fmt, **kw)
    return buf.getvalue()


def best_window(diff: np.ndarray, size: int, step: int = 8) -> tuple[int, int]:
    """Top-left corner of the size x size window with the largest sum of diff
    (an integral image, so a large image costs one pass)."""
    h, w = diff.shape
    sh, sw = min(size, h), min(size, w)
    ii = np.zeros((h + 1, w + 1), np.float64)
    ii[1:, 1:] = diff.cumsum(0).cumsum(1)
    ys = np.arange(0, h - sh + 1, step)
    xs = np.arange(0, w - sw + 1, step)
    sums = ii[ys[:, None] + sh, xs[None, :] + sw] - ii[ys[:, None], xs[None, :] + sw] \
        - ii[ys[:, None] + sh, xs[None, :]] + ii[ys[:, None], xs[None, :]]
    i, j = np.unravel_index(int(np.argmax(sums)), sums.shape)
    return int(ys[i]), int(xs[j])


THUMB = 256                       # px, the longest side of the thumbnail
CROP = 256                        # px, the crop shown at 100%
ZOOM_SRC = 96                     # px, the part of the crop shown magnified
ZOOM = 3
TILES_PER_SHEET = 8
LABEL_H = 20
SHEET_QUALITY = 95                # the sheets are JPEG 4:4:4 at this quality: their own artifacts stay far below the ones judged
ORIENT = {2: [Image.Transpose.FLIP_LEFT_RIGHT], 3: [Image.Transpose.ROTATE_180], 4: [Image.Transpose.FLIP_TOP_BOTTOM],
          5: [Image.Transpose.TRANSPOSE], 6: [Image.Transpose.ROTATE_270], 7: [Image.Transpose.TRANSVERSE],
          8: [Image.Transpose.ROTATE_90]}


def tile_font(size: int = 14):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def as_rgb(a: np.ndarray) -> np.ndarray:
    return np.repeat(a[..., None], 3, axis=2) if a.ndim == 2 else a


def make_tile(rel: str, label: str, orig: np.ndarray, outs: list[np.ndarray], orientation: int) -> Image.Image:
    """One row of a sheet: the thumbnail (EXIF-rotated, the crop marked in red),
    the crop at 100% before and after each restoration, and its most changed
    part magnified before and after. The crop sits where the first restoration
    changed the image most, which is where the artifacts (or the grain it took
    away) are."""
    o = as_rgb(orig)
    diff = np.abs(o.astype(np.int16) - as_rgb(outs[0])).mean(axis=2, dtype=np.float32)
    y, x = best_window(diff, CROP)
    ch, cw = min(CROP, o.shape[0]), min(CROP, o.shape[1])
    zy, zx = best_window(diff[y:y + ch, x:x + cw], ZOOM_SRC, step=4)
    zh, zw = min(ZOOM_SRC, ch), min(ZOOM_SRC, cw)
    thumb = Image.fromarray(o)
    scale = THUMB / max(thumb.size)
    thumb = thumb.resize((max(1, round(thumb.width * scale)), max(1, round(thumb.height * scale))), Image.LANCZOS)
    ImageDraw.Draw(thumb).rectangle([x * scale, y * scale, (x + cw) * scale - 1, (y + ch) * scale - 1],
                                    outline=(255, 0, 0), width=2)
    for t in ORIENT.get(orientation, []):
        thumb = thumb.transpose(t)
    panels = [o] + [as_rgb(a) for a in outs]
    width = THUMB + len(panels) * (CROP + ZOOM_SRC * ZOOM) + 2 * 8
    tile = Image.new("RGB", (width, LABEL_H + max(THUMB, CROP, ZOOM_SRC * ZOOM)), "white")
    ImageDraw.Draw(tile).text((4, 2), label + "   " + rel, fill="black", font=tile_font())
    tile.paste(thumb, (0, LABEL_H))
    cx = THUMB + 8
    for a in panels:
        tile.paste(Image.fromarray(a[y:y + ch, x:x + cw]), (cx, LABEL_H))
        cx += CROP
    cx += 8
    for a in panels:
        z = Image.fromarray(a[y + zy:y + zy + zh, x + zx:x + zx + zw]).resize((zw * ZOOM, zh * ZOOM), Image.NEAREST)
        tile.paste(z, (cx, LABEL_H))
        cx += ZOOM_SRC * ZOOM
    return tile


def clear_sheets(folder: Path) -> None:
    """Delete the sheets of the last run, so the folder never shows sheets that
    do not belong to the report next to it."""
    if folder.exists():
        for p in [*folder.glob("fix_*.jpg"), *folder.glob("nofix_*.jpg"), *folder.glob("sheet_*.jpg")]:
            p.unlink()


class SheetWriter:
    """Collects tiles in QF order and writes a sheet of TILES_PER_SHEET each,
    named by its number and the QF range it shows, with a header row naming the
    columns."""

    def __init__(self, folder: Path, offsets: list[int], prefix: str, heads: list[str] | None = None):
        self.folder, self.offsets, self.prefix = folder, offsets, prefix
        self.heads = heads or ["original"] + [f"QF {o:+d}" for o in offsets]
        self.tiles: list[tuple[float, Image.Image]] = []
        self.count = 0
        folder.mkdir(parents=True, exist_ok=True)

    def add(self, qf: float, tile: Image.Image) -> None:
        self.tiles.append((qf, tile))
        if len(self.tiles) == TILES_PER_SHEET:
            self.flush()

    def flush(self) -> None:
        if not self.tiles:
            return
        width = max(t.width for _, t in self.tiles)
        heads = self.heads
        sheet = Image.new("RGB", (width, LABEL_H + sum(t.height + 6 for _, t in self.tiles)), "white")
        d = ImageDraw.Draw(sheet)
        font = tile_font()
        cx = THUMB + 8
        for h in heads:
            d.text((cx + 4, 2), h + ", 100%", fill="black", font=font)
            cx += CROP
        cx += 8
        for h in heads:
            d.text((cx + 4, 2), h + f", {ZOOM}x", fill="black", font=font)
            cx += ZOOM_SRC * ZOOM
        y = LABEL_H
        for _, t in self.tiles:
            sheet.paste(t, (0, y))
            y += t.height + 6
        self.count += 1
        lo, hi = self.tiles[0][0], self.tiles[-1][0]
        sheet.save(self.folder / f"{self.prefix}_{self.count:04d}_q{int(lo):02d}-{int(hi):02d}.jpg", "JPEG",
                   quality=SHEET_QUALITY, subsampling=0)
        self.tiles = []


def blockiness(a: np.ndarray) -> float:
    """How much the 8 x 8 JPEG block grid shows: the mean luma step across
    block edges divided by the mean step between other neighbouring pixels,
    averaged over both directions. About 1.0 without blocking; film grain and
    fine texture raise both steps alike and leave it near 1.0."""
    g = (a.astype(np.float32) if a.ndim == 2 else a.astype(np.float32) @ np.float32([0.299, 0.587, 0.114]))
    ratios = []
    for dx in (np.abs(np.diff(g, axis=1)), np.abs(np.diff(g, axis=0)).T):
        if dx.shape[1] < 16:
            continue
        edge = (np.arange(dx.shape[1]) % 8) == 7          # the step between pixel 8k-1 and 8k
        inside = dx[:, ~edge].mean()
        ratios.append(float(dx[:, edge].mean() / inside) if inside > 0 else 1.0)
    return round(sum(ratios) / len(ratios), 3) if ratios else 1.0


def benefit(m: dict, r: dict, min_block_drop: float, min_qf_gain: float) -> str:
    """Why the restoration is worth saving, or "" when it is not. Two signals,
    neither of which film grain can fake: the drop in blockiness (the 8 x 8
    grid fading), and the rise of FBCNN's own QF, which also sees ringing and
    mosquito noise but misreads dithered and screened prints; so the QF counts
    only on an image whose grid shows. The mean change of the pixels is not a
    signal: on scanned prints it mostly measures the grain taken away."""
    drop = r["block_before"] - r["block_after"]
    gain = r["qf_after"] - m["qf"]
    if drop >= min_block_drop:
        return f"blockiness -{drop:.2f}"
    if gain >= min_qf_gain and r["block_before"] >= GRID_MIN:
        return f"QF +{gain:.0f}"
    return ""


def fix_one(model: QualityModel, d: dict, m: dict, offsets: list[int], quality: int, sheet: bool) -> dict:
    """Restore one decoded image with the colour model at its own predicted QF
    plus offsets[0] (and the other offsets when a tile is wanted). A gray image
    goes in as three equal channels and only the luma of the result is kept:
    the gray model erased film grain and barely reacted to the offset, the
    colour model keeps the grain, and the luma cannot carry colour noise.
    -> qf_used, write, qf_after, change, block_before, block_after, "data"
    (the bytes to write) and, when sheet is set, "outs" (the restorations as
    they would be read back from the file)."""
    orig = d["luma"] if d["gray"] else d["rgb"]
    qf = m.get("qf_color", m["qf"])
    jpeg = d["format"] in ("JPEG", "MPO")
    outs, data, restored = [], b"", None
    for i, o in enumerate(offsets if sheet else offsets[:1]):
        y = model.restore("color", d["rgb"], min(100.0, qf + o))
        if d["gray"]:
            y = np.asarray(Image.fromarray(y).convert("L"))
        if i == 0:
            restored = y                                                   # the restoration itself
        if i == 0 or jpeg:
            enc = encode_output(y, d, quality)
            data = data or enc
            if jpeg:
                y = np.array(Image.open(io.BytesIO(enc)))                  # what the file will hold
        outs.append(y)
    r = {"data": data, "restored": restored, "qf_used": round(min(100.0, qf + offsets[0]), 1),
         "write": "jpeg" if d["format"] in ("JPEG", "MPO") else d["format"].lower(),
         "qf_after": round(model.qf("color", as_rgb(outs[0])), 1),
         "change": round(float(np.abs(orig.astype(np.int16) - outs[0]).mean()), 2),
         "block_before": blockiness(orig), "block_after": blockiness(outs[0])}
    if sheet:
        r["outs"] = outs
    return r


def fix_action_of(m: dict, threshold: int, max_pixels: int) -> str:
    """fix, keep, or the skip reason, from a measurement."""
    if m.get("skip"):
        return m["skip"]
    if m["width"] * m["height"] > max_pixels:
        return "large"
    return "fix" if m["qf"] < threshold else "keep"


# --- per-image API ---------------------------------------------------------------
# What the pipeline tool calls, one image at a time, on pixels it decoded
# itself: plan() measures and restores, apply() gives the restored pixels. The
# command line goes through the same plan_decoded(), on files it decodes.

_model: QualityModel | None = None


def model_lazy() -> QualityModel:
    """The FBCNN model, loaded on first use and kept."""
    global _model
    if _model is None:
        _model = QualityModel()
    return _model


def unload() -> None:
    """Drop the model loaded by model_lazy(), so its memory can go."""
    global _model
    _model = None


def plan_options(options: dict | None) -> dict:
    return {"threshold": DEFAULT_THRESHOLD, "qf_offset": DEFAULT_QF_OFFSET, "max_pixels": DEFAULT_MAX_PIXELS,
            "quality": DEFAULT_QUALITY, "min_block_drop": DEFAULT_MIN_BLOCK_DROP, "min_qf_gain": DEFAULT_MIN_QF_GAIN,
            "sheet_offsets": [], "sheet": False, "model": None, "m": None, **(options or {})}


def plan_decoded(d: dict, opts: dict) -> dict:
    """The fix of one decoded image (decode or decode_array's dict), with the
    options of plan(): measure it (or take opts["m"], the cached measurement),
    decide, and restore it when its QF is under the threshold.
    -> {"action": fix (worth saving), little benefit, keep or the skip reason,
        "m": the measurement, "fix": fix_one's numbers with "benefit" and the
        "data" to write, "restored": the restoration as stored pixels (None
        unless worth saving), "outs" and "orig": for a contact-sheet tile when
        opts["sheet"] is set}"""
    if d["skip"]:
        return {"action": d["skip"], "m": {k: d[k] for k in ("format", "mode", "width", "height", "gray",
                                                             "header_q", "skip")}, "fix": {}, "restored": None}
    model = opts["model"] or model_lazy()
    m = opts["m"] or measure_one(model, d)
    action = fix_action_of(m, opts["threshold"], opts["max_pixels"])
    out = {"action": action, "m": m, "fix": {}, "restored": None}
    if action != "fix":
        return out
    offsets = [opts["qf_offset"]] + [o for o in opts["sheet_offsets"] if o != opts["qf_offset"]]
    r = fix_one(model, d, m, offsets, opts["quality"], opts["sheet"])
    r["benefit"] = benefit(m, r, opts["min_block_drop"], opts["min_qf_gain"])
    restored, outs = r.pop("restored"), r.pop("outs", None)
    out["fix"] = r
    if r["benefit"]:
        out["restored"] = restored
    else:
        out["action"] = "little benefit"
    if opts["sheet"]:
        out["outs"], out["orig"] = outs, d["luma"] if d["gray"] else d["rgb"]
    return out


def plan(array: np.ndarray, head: dict, options: dict | None = None) -> dict:
    """The fix of one image. array: the upright RGB uint8 pixels; head: the
    facts of the file (facts_of, or a dict with its keys); options: threshold,
    qf_offset, max_pixels, quality, min_block_drop, min_qf_gain (the command
    line's defaults), model (a QualityModel, default model_lazy()), m (a cached
    measurement of this file, else it is measured), sheet and sheet_offsets
    (keep the restorations for a contact-sheet tile).
    -> plan_decoded's dict, with "restored" turned upright and in RGB, so that
    apply() can hand it back in place of array; the "data" bytes are left out."""
    opts = plan_options(options)
    d = decode_array(array, head, opts["max_pixels"])
    out = plan_decoded(d, opts)
    out["fix"].pop("data", None)
    if out["restored"] is not None:
        out["restored"] = np.ascontiguousarray(to_upright(as_rgb(out["restored"]), d["orientation"]))
    return out


def apply(array: np.ndarray, plan: dict) -> np.ndarray:
    """The restored pixels when the plan found the restoration worth saving,
    else array itself."""
    if plan.get("action") == "fix" and plan.get("restored") is not None:
        return plan["restored"]
    return array


FIX_FIELDS = ["path", "format", "mode", "width", "height", "gray", "header_q", "qf", "action", "benefit",
              "qf_used", "write", "qf_after", "change", "block_before", "block_after", "result"]


def write_report(root: Path, items) -> Path:
    path = root / BACKUP_DIRNAME / RUN_DIRNAME / REPORT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIX_FIELDS)
        for it in items:
            m, r = it.m, it.fix
            w.writerow([it.rel, m.get("format", ""), m.get("mode", ""), m.get("width", 0), m.get("height", 0),
                        "" if m.get("gray") is None else int(m["gray"]), fmt_num(m.get("header_q")),
                        fmt_num(m.get("qf")), it.status, r.get("benefit", ""), fmt_num(r.get("qf_used")),
                        r.get("write", ""), fmt_num(r.get("qf_after")), fmt_num(r.get("change")),
                        fmt_num(r.get("block_before")), fmt_num(r.get("block_after")), r.get("result", "")])
    os.replace(tmp, path)
    return path


# --- writing, backup and undo ------------------------------------------------------
# A real run logs every image before it touches it, so --undo can put back an
# interrupted run too. The original goes to _backup/<relative path>, unless an
# earlier original of that path is there already (from remove_borders, say);
# then it goes to _backup/_jpeg_cleanup/originals/<run>/<relative path>, so the
# undo returns exactly what this run started from and the earlier one stays.

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


def part_path(dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    return dst.with_name(dst.name + ".part")


def same_bytes(a: Path, b: Path) -> bool:
    try:
        return a.stat().st_size == b.stat().st_size and filecmp.cmp(a, b, shallow=False)
    except OSError:
        return False


def copy_keep(src: Path, dst: Path) -> str:
    """Copy src to dst unless dst exists. -> copied, same (dst has these bytes
    already) or kept (dst holds other bytes, left alone)."""
    if dst.exists():
        return "same" if same_bytes(src, dst) else "kept"
    tmp = part_path(dst)
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return "copied"


def finish_write(tmp: Path, dst: Path, src_stat: os.stat_result) -> None:
    """Move tmp over dst; dst keeps the original's times and read-only flag."""
    make_writable(dst)
    os.replace(tmp, dst)
    os.utime(dst, ns=(src_stat.st_atime_ns, src_stat.st_mtime_ns))
    if not src_stat.st_mode & stat.S_IWRITE:
        os.chmod(dst, stat.S_IREAD)


def backup_place(root: Path, it, run_id: str) -> Path:
    bk = root / BACKUP_DIRNAME / it.rel
    if bk.exists() and not same_bytes(it.path, bk):
        bk = root / BACKUP_DIRNAME / RUN_DIRNAME / ORIGINALS_DIRNAME / run_id / it.rel
    return bk


def write_fix(root: Path, it, data: bytes, bk: Path) -> dict:
    """Back up the original and its sidecars, then put the restoration in its
    place under the same name. -> the log record."""
    rec = {"op": "fix", "rel": it.rel}
    tmp = None
    try:
        st = os.stat(it.path)
        rec["image"] = copy_keep(it.path, bk)
        rec["sidecars"] = {sc.relative_to(root).as_posix(): copy_keep(sc, root / BACKUP_DIRNAME / sc.relative_to(root))
                           for sc in it.sidecars}
        tmp = part_path(it.path)
        tmp.write_bytes(data)
        finish_write(tmp, it.path, st)
    except Exception as e:  # noqa: BLE001 - one bad file is reported, not fatal
        rec["error"] = f"{type(e).__name__}: {e}".strip()
        if tmp is not None and tmp.exists():
            tmp.unlink()
    return rec


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


def undo_root(root: Path) -> dict:
    """Put back what the last run of root changed, newest first. An image gets
    its original back from where the run kept it; a backup this run made is
    removed, one that was there before stays. A caption backup this run made
    is removed when the caption in place has the same bytes. -> counts."""
    run_id, recs = last_run(read_log(root))
    counts = {"restored": 0, "no backup": 0}
    if run_id is None:
        return counts
    done = {r["rel"]: r for r in recs if r.get("op") == "fix"}
    bdir = root / BACKUP_DIRNAME
    for intent in reversed([r for r in recs if r.get("op") == "intent"]):
        rel, rec = intent["rel"], done.get(intent["rel"], {})
        bk, dst = root / intent["backup"], root / rel
        if not bk.exists():
            counts["no backup"] += 1
            print(f"  not restored, no backup: {rel}")
            continue
        make_writable(dst)
        tmp = part_path(dst)
        shutil.copy2(bk, tmp)
        os.replace(tmp, dst)
        if intent["fresh"]:
            make_writable(bk)
            bk.unlink()
        counts["restored"] += 1
        for sc in intent.get("sidecars", []):
            b = bdir / sc
            if not (root / sc).exists() and b.exists():
                os.replace(b, root / sc)           # the caption went missing: put the copy back
            elif (rec.get("sidecars") or {}).get(sc) == "copied" and b.exists() and same_bytes(b, root / sc):
                make_writable(b)
                b.unlink()
    log = RunLog(root)
    log.write({"undo": run_id, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **counts})
    log.close()
    for dirpath, dirnames, filenames in os.walk(bdir, topdown=False):
        d = Path(dirpath)
        if d != bdir and d != bdir / RUN_DIRNAME:
            try:
                d.rmdir()
            except OSError:
                pass
    return counts


def review_root(root: Path, threads: int) -> tuple[int, int, int] | None:
    """Contact sheets of the last run that is not undone: every image it wrote,
    the original from where the run kept it next to the file now in place, in
    QF order. Reads files only; no model. -> (sheets, images drawn, images
    written by the run), or None when there is no run."""
    run_id, recs = last_run(read_log(root))
    if run_id is None:
        return None
    intents = {r["rel"]: r for r in recs if r.get("op") == "intent"}
    done = sorted((r for r in recs if r.get("op") == "fix" and "error" not in r and r["rel"] in intents),
                  key=lambda r: (r.get("qf", 0), r["rel"]))
    sheet_dir = root / BACKUP_DIRNAME / RUN_DIRNAME / SHEETS_DIRNAME
    clear_sheets(sheet_dir)
    sheets = SheetWriter(sheet_dir, [], "fix", heads=["original", "written"])

    def tile(rec):
        a = decode(root / intents[rec["rel"]]["backup"], 1 << 62)
        b = decode(root / rec["rel"], 1 << 62)
        if a["skip"] or b["skip"]:
            return None, "original: " + a["skip"] if a["skip"] else "written file: " + b["skip"]
        orig = a["luma"] if a["gray"] else a["rgb"]
        now = b["luma"] if b["gray"] else b["rgb"]
        if orig.shape[:2] != now.shape[:2]:
            return None, "the file in place has another size than its original"
        if orig.ndim != now.ndim:
            orig, now = as_rgb(orig), as_rgb(now)
        change = float(np.abs(orig.astype(np.int16) - now).mean())
        label = (f"SAVED: {rec.get('benefit', '')}  |  QF {rec.get('qf', '-')}, blockiness {blockiness(orig):.2f} > "
                 f"{blockiness(now):.2f}, change {change:.2f}, {a['width']}x{a['height']}{', gray' if a['gray'] else ''}")
        return make_tile(rec["rel"], label, orig, [now], a["orientation"]), ""

    drawn = 0
    progress = Progress(root.name, "drawn", len(done))
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for n, (rec, (tl, why)) in enumerate(zip(done, bounded_map(pool, tile, done, threads * 2)), 1):
            if tl is None:
                print(f"  not drawn, {why}: {rec['rel']}")
            else:
                sheets.add(rec.get("qf", 0), tl)
                drawn += 1
            progress.step(n)
    sheets.flush()
    return sheets.count, drawn, len(done)


# --- the fix of one folder --------------------------------------------------------------

def fix_folder(root: Path, args, model_box: list) -> int:
    """Measure, restore, judge, and in a real run write. -> failed writes."""
    t0 = time.perf_counter()
    dry = args.dry_run
    sidecar_exts = [e if e.startswith(".") else "." + e for e in (s.strip() for s in args.sidecars.split(",")) if e]
    items = scan(root, DEFAULT_EXCLUDES + args.exclude, sidecar_exts)
    print(f"{root}: {len(items):,} images found", flush=True)
    reused = measure_items(root, items, args.max_pixels, args.threads, args.reanalyse, model_box)
    if reused:
        print(f"  {root.name}: {reused:,} images unchanged since the last run, taken from the cache")
    for it in items:
        it.status, it.fix = fix_action(it, args.threshold, args.max_pixels), {}
    todo = sorted((it for it in items if it.status == "fix"), key=lambda it: (it.m["qf"], it.rel))
    offsets = [args.qf_offset] + [o for o in args.sheet_offsets if o != args.qf_offset]
    sheet_dir = root / BACKUP_DIRNAME / RUN_DIRNAME / SHEETS_DIRNAME
    clear_sheets(sheet_dir)
    want_sheets = not args.no_sheets if dry else args.sheets
    # one set of sheets for the restorations worth saving, one for the rest
    sheets = {True: SheetWriter(sheet_dir, offsets, "fix"),
              False: SheetWriter(sheet_dir, offsets, "nofix")} if want_sheets else None
    log, run_id, written, failed = None, time.strftime("%Y%m%d-%H%M%S"), 0, 0
    try:
        if todo:
            if not model_box:
                model_box.append(QualityModel())
            model = model_box[0]
            opts = plan_options({"threshold": args.threshold, "qf_offset": args.qf_offset, "max_pixels": args.max_pixels,
                                 "quality": args.quality, "min_block_drop": args.min_block_drop,
                                 "min_qf_gain": args.min_qf_gain, "sheet_offsets": args.sheet_offsets,
                                 "sheet": sheets is not None, "model": model})
            progress = Progress(root.name, "restored", len(todo))
            with ThreadPoolExecutor(max_workers=args.threads) as pool, ThreadPoolExecutor(max_workers=2) as tiler:
                decoded = bounded_map(pool, lambda it: decode(it.path, args.max_pixels), todo, args.threads * 2)
                pending = deque()
                for n, (it, d) in enumerate(zip(todo, decoded), 1):
                    if d["skip"]:                     # changed since it was measured
                        it.status = d["skip"]
                        continue
                    p = plan_decoded(d, dict(opts, m=it.m))
                    r, outs = p["fix"], p.get("outs")
                    data = r.pop("data")
                    it.fix = r
                    if not r["benefit"]:
                        it.status = "little benefit"
                    elif not dry:
                        if log is None:
                            log = RunLog(root)
                            log.write({"run": run_id, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                       "threshold": args.threshold, "qf_offset": args.qf_offset})
                        bk = backup_place(root, it, run_id)
                        log.write({"op": "intent", "rel": it.rel, "backup": bk.relative_to(root).as_posix(),
                                   "fresh": not bk.exists(),
                                   "sidecars": [sc.relative_to(root).as_posix() for sc in it.sidecars]})
                        rec = write_fix(root, it, data, bk)
                        rec.update(qf=it.m["qf"], benefit=r["benefit"])
                        log.write(rec)
                        r["result"] = "error: " + rec["error"] if "error" in rec else "written"
                        written += "error" not in rec
                        failed += "error" in rec
                    if sheets is not None:
                        label = (f"{'SAVE: ' + r['benefit'] if r['benefit'] else 'LEAVE: little benefit'}  |  "
                                 f"QF {it.m['qf']:.1f} > {r['qf_after']:.1f}, blockiness {r['block_before']:.2f} > "
                                 f"{r['block_after']:.2f}, change {r['change']:.2f}, header {it.m.get('header_q') or '-'}, "
                                 f"{it.m['width']}x{it.m['height']}{', gray' if d['gray'] else ''}")
                        orig = d["luma"] if d["gray"] else d["rgb"]
                        pending.append((bool(r["benefit"]), it.m["qf"],
                                        tiler.submit(make_tile, it.rel, label, orig, outs, d["orientation"])))
                        while pending and (pending[0][2].done() or len(pending) > 8):
                            keep, q, fut = pending.popleft()
                            sheets[keep].add(q, fut.result())
                    progress.step(n)
                for keep, q, fut in pending:
                    sheets[keep].add(q, fut.result())
    finally:
        if log is not None:
            log.write({"end": run_id, "written": written, "failed": failed})
            log.close()
        if written:
            save_cache(root, items, {})           # the written files get new keys; drop their old QF
    if sheets is not None:
        for s in sheets.values():
            s.flush()
    report = write_report(root, items)
    count = {}
    for it in items:
        k = it.status if it.status in ("fix", "little benefit", "keep") else "skipped"
        count[k] = count.get(k, 0) + 1
    print(f"{root}: {len(items):,} images, {clock(time.perf_counter() - t0)}")
    print(f"  threshold QF {args.threshold}, offset {args.qf_offset:+d}: "
          f"{count.get('fix', 0) + count.get('little benefit', 0):,} restored in memory; "
          f"{count.get('fix', 0):,} worth saving (blockiness -{args.min_block_drop:.2f} or QF +{args.min_qf_gain:g}), "
          f"{count.get('little benefit', 0):,} left as they are (little benefit)")
    print(f"  {count.get('keep', 0):,} at or above the threshold, {count.get('skipped', 0):,} skipped")
    fixed = [it for it in items if it.fix]
    if fixed:
        ch = sorted(it.fix["change"] for it in fixed)
        qa = sorted(it.fix["qf_after"] for it in fixed)
        print(f"  change (mean, 8-bit levels): median {ch[len(ch) // 2]:.2f}, max {ch[-1]:.2f}; "
              f"QF after the fix: median {qa[len(qa) // 2]:.1f}, min {qa[0]:.1f}")
    if sheets is not None:
        print(f"  sheets:  {sheets[True].count} fix_*, {sheets[False].count} nofix_* in {sheet_dir}")
    print(f"  report:  {report}")
    if dry:
        print("  dry run: no image was changed")
    else:
        print(f"  done:    {written:,} images restored in place, originals in {root / BACKUP_DIRNAME}"
              + (f"; {failed:,} failed (see the report)" if failed else ""))
        if written and sheets is None:
            print("  sheets:  none in a real run without --sheets; --review draws them from the backups")
    return failed


# --- command line -------------------------------------------------------------------

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


def output_dirs(roots: list[Path], out_arg: str | None) -> list[Path]:
    """The output folder of each root: --out itself for one folder, --out/<name>
    for several, else <root>_jpeg_extract next to the root. Never inside a root,
    which the next scan would read as part of the dataset."""
    if out_arg:
        base = Path(out_arg).resolve()
        outs = [base] if len(roots) == 1 else [base / r.name for r in roots]
    else:
        outs = [r.parent / (r.name + EXTRACT_SUFFIX) for r in roots]
    if len({o.as_posix().casefold() for o in outs}) != len(outs):
        raise SystemExit("Two folders have the same name; give --out for each run separately")
    for o in outs:
        for r in roots:
            if o == r or o.is_relative_to(r):
                raise SystemExit(f"The output folder {o} is inside the dataset folder {r}; give --out outside it")
    return outs


def extract_dataset(root: Path, out: Path, args, model_box: list) -> None:
    t0 = time.perf_counter()
    bands = args.bands or parse_bands(DEFAULT_BANDS)
    sidecar_exts = [e if e.startswith(".") else "." + e for e in (s.strip() for s in args.sidecars.split(",")) if e]
    items = scan(root, DEFAULT_EXCLUDES + args.exclude, sidecar_exts)
    print(f"{root}: {len(items):,} images found", flush=True)
    reused = measure_items(root, items, args.max_pixels, args.threads, args.reanalyse, model_box)
    if reused:
        print(f"  {root.name}: {reused:,} images unchanged since the last run, taken from the cache")
    assign(items, bands, args.max_pixels)
    out.mkdir(parents=True, exist_ok=True)
    finish_moves(out)
    removed = clear_previous(out)
    if removed:
        print(f"  {removed:,} files of the previous extract removed from {out}")
    plan = plan_copies(out, items)
    # The manifest lists the planned copies before the first one is made, so a
    # stopped run leaves nothing the next run cannot clear.
    write_manifest(out, {"root": str(root), "bands": bands, "files": [rel for _, fs in plan for _, rel in fs]})
    copy_items(out, plan)
    write_manifest(out, {"root": str(root), "bands": bands,
                         "files": [rel for it, fs in plan if it.copy for _, rel in fs]})
    write_csv(out, items)
    write_summary(out, summary(root, out, items, bands, args.max_pixels, time.perf_counter() - t0))


def extract(args) -> int:
    folders = check_roots(args.folders)
    resorts = [f for f in folders if is_extract_folder(f)]
    roots = [f for f in folders if f not in resorts]
    if resorts and args.out:
        raise SystemExit("--out does not go with an extract folder, which is sorted in place")
    if resorts and args.reanalyse:
        raise SystemExit("--reanalyse needs the dataset folder; an extract folder has only the copies")
    outs = output_dirs(roots, args.out) if roots else []
    model_box: list = []
    for folder in folders:
        if folder in resorts:
            t0 = time.perf_counter()
            print(f"{folder}: an extract folder, sorted again from its extract.csv", flush=True)
            items, bands, root = resort(folder, args)
            write_summary(folder, summary(root, folder, items, bands, args.max_pixels, time.perf_counter() - t0))
        else:
            extract_dataset(folder, outs[roots.index(folder)], args, model_box)
    return 0


def qf_arg(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"a quality must be an integer, got {value!r}")
    if not 1 <= n <= 100:
        raise argparse.ArgumentTypeError(f"a quality must be 1..100, got {n}")
    return n


def parse_offsets(text: str) -> list[int]:
    try:
        return [int(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"--sheet-offsets must be integers separated by commas, got {text!r}")


def fix(args) -> int:
    """The fix of every folder. A dry run takes an extract folder too: its
    copies have the bytes of their sources, so the report and the sheets show
    what a run on the dataset would do, and the dry run writes only into its
    _backup folder, which a re-sort of the extract leaves alone. A real run
    does not: it would turn the copies into something their extract.csv no
    longer describes."""
    roots = check_roots(args.folders)
    if not args.dry_run:
        for r in roots:
            if is_extract_folder(r):
                raise SystemExit(f"{r} is an extract folder; run the fix on the dataset folder, or use --dry-run")
    model_box: list = []
    failed = sum(fix_folder(root, args, model_box) for root in roots)
    return 1 if failed else 0


def undo(args) -> int:
    for root in check_roots(args.folders):
        c = undo_root(root)
        if not any(c.values()):
            print(f"{root}: no run to undo")
        else:
            print(f"{root}: {c['restored']:,} images restored"
                  + (f", {c['no backup']:,} without a backup (see above)" if c["no backup"] else ""))
    return 0


def review(args) -> int:
    for root in check_roots(args.folders):
        res = review_root(root, args.threads)
        if res is None:
            print(f"{root}: no run to review")
        else:
            n, drawn, total = res
            print(f"{root}: {drawn:,} of {total:,} restored images drawn on {n} sheets in "
                  f"{root / BACKUP_DIRNAME / RUN_DIRNAME / SHEETS_DIRNAME}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Find heavily compressed images in a dataset with FBCNN, and restore them.")
    ap.add_argument("folders", nargs="*",
                    help="dataset folders, scanned at any depth; with --extract also extract folders, sorted again")
    ap.add_argument("--extract", action="store_true",
                    help="copy the images under each band limit into one folder per band, outside the dataset")
    ap.add_argument("--dry-run", action="store_true",
                    help="the fix in memory: write the report and the contact sheets, change no image")
    ap.add_argument("--undo", action="store_true", help="put back what the last fix of each folder changed")
    ap.add_argument("--review", action="store_true",
                    help="contact sheets of the last fix: each original in _backup next to the file now in place")
    ap.add_argument("--threshold", type=qf_arg, default=DEFAULT_THRESHOLD, metavar="QF",
                    help=f"fix images with a QF under this (default {DEFAULT_THRESHOLD})")
    ap.add_argument("--qf-offset", type=int, default=DEFAULT_QF_OFFSET, metavar="N",
                    help=f"added to the predicted QF given to the restoration; higher keeps more grain "
                         f"(default {DEFAULT_QF_OFFSET})")
    ap.add_argument("--sheet-offsets", type=parse_offsets, default=[], metavar="LIST",
                    help="more offsets shown side by side on the contact sheets, comma-separated, e.g. 10,20")
    ap.add_argument("--no-sheets", action="store_true", help="with --dry-run: no contact sheets")
    ap.add_argument("--sheets", action="store_true", help="with a real run: write the contact sheets too")
    ap.add_argument("--min-block-drop", type=float, default=DEFAULT_MIN_BLOCK_DROP, metavar="X",
                    help=f"save a restoration whose blockiness drops this much (default {DEFAULT_MIN_BLOCK_DROP})")
    ap.add_argument("--min-qf-gain", type=float, default=DEFAULT_MIN_QF_GAIN, metavar="N",
                    help=f"...or whose QF rises this much, when the JPEG grid shows (default {DEFAULT_MIN_QF_GAIN})")
    ap.add_argument("--quality", type=qf_arg, default=DEFAULT_QUALITY, metavar="Q",
                    help=f"JPEG quality of a restored JPEG (default {DEFAULT_QUALITY})")
    ap.add_argument("--fetch-models", action="store_true", help="download the FBCNN models into models/ and check them")
    ap.add_argument("--out", metavar="DIR",
                    help=f"output folder (default <folder>{EXTRACT_SUFFIX} next to each dataset folder)")
    ap.add_argument("--bands", type=parse_bands, default=None, metavar="LIST",
                    help=f"band limits, comma-separated (default {DEFAULT_BANDS}; for an extract folder, its own)")
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS, metavar="N",
                    help=f"skip images larger than N pixels (default {DEFAULT_MAX_PIXELS}, which is 2048 x 2048)")
    ap.add_argument("--exclude", action="append", default=[], metavar="NAME",
                    help="another folder name to skip at any depth; may repeat. Always skipped: "
                         + ", ".join(DEFAULT_EXCLUDES))
    ap.add_argument("--sidecars", default=DEFAULT_SIDECARS, metavar="LIST",
                    help=f"sidecar extensions, comma-separated (default {DEFAULT_SIDECARS})")
    ap.add_argument("--reanalyse", action="store_true", help="ignore the measurement cache of the last run")
    ap.add_argument("--threads", type=threads_arg, default=DEFAULT_THREADS, metavar="N",
                    help=f"decoder threads (default {DEFAULT_THREADS})")
    args = ap.parse_args(argv)

    if args.fetch_models:
        fetch_models()
        return 0
    if not args.folders:
        ap.error("give at least one dataset folder")
    if sum((args.extract, args.dry_run, args.undo, args.review)) > 1:
        ap.error("--extract, --dry-run, --undo and --review do not go together")
    if args.undo:
        return undo(args)
    if args.review:
        return review(args)
    try:
        return extract(args) if args.extract else fix(args)
    except KeyboardInterrupt:
        print("\nStopped. The measurements so far are in the cache; the next run continues from there."
              + ("" if args.extract or args.dry_run else
                 " The images restored so far are logged; --undo puts them back."))
        return 130


if __name__ == "__main__":
    sys.exit(main())
