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

--dry-run (the fix; its real run comes next): every image with a QF under
--threshold is restored in memory, at its QF plus --qf-offset, and encoded as
it would be written. The report and the contact sheets (before and after,
at 100% and magnified) go to <folder>/_backup/_jpeg_cleanup; no image changes.

Usage:    python jpeg_cleanup.py --extract <folder> [<folder> ...] [--out DIR]
          [--bands LIST] [--max-pixels N] [--exclude NAME] [--sidecars LIST]
          [--reanalyse] [--threads N]
          python jpeg_cleanup.py <folder> [<folder> ...] --dry-run [--threshold QF]
          [--qf-offset N] [--sheet-offsets LIST] [--no-sheets] [--quality Q]
          [--max-pixels N] [--exclude NAME] [--reanalyse] [--threads N]
          python jpeg_cleanup.py --fetch-models
Install:  install.bat (torch from the PyTorch CUDA index, Pillow, numpy)
"""
import argparse
import csv
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


def decode(path: Path, max_pixels: int) -> dict:
    """Open one image and return its facts and, when it is to be measured, its
    stored pixels (no EXIF rotation, so the JPEG block grid stays aligned):
    "rgb" uint8 HxWx3 and, for a gray image, "luma" uint8 HxW. A skipped image
    gets "skip" with the reason. Runs in the decoder threads."""
    d = {"format": "", "mode": "", "width": 0, "height": 0, "gray": None, "header_q": None, "skip": "",
         "orientation": 1}
    try:
        with Image.open(path) as im:
            fmt = im.format or ""
            d.update(format=fmt, mode=im.mode, width=im.size[0], height=im.size[1])
            try:
                o = im.getexif().get(274, 1) or 1
                d["orientation"] = o if o in range(1, 9) else 1
            except Exception:  # noqa: BLE001 - a damaged EXIF block is not fatal
                pass
            if fmt in ("JPEG", "MPO"):
                d["header_q"] = header_quality(im)
            if fmt not in MEASURED_FORMATS:
                d["skip"] = f"format {fmt or '?'}"
            elif fmt == "WEBP" and not webp_is_lossless(path):
                d["skip"] = "format lossy WEBP"
            elif fmt != "MPO" and (getattr(im, "n_frames", 1) or 1) > 1:
                d["skip"] = "animated"
            elif im.mode in ("1", "CMYK", "I", "F") or im.mode.startswith("I;16"):
                d["skip"] = f"mode {im.mode}"
            elif im.size[0] * im.size[1] > max_pixels:
                d["skip"] = "large"
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
    d["rgb"] = rgb
    d["gray"] = is_gray(rgb)
    if d["gray"]:
        d["luma"] = np.array(Image.fromarray(rgb).convert("L"))
    return d


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
    image, since the fix will use that model)."""
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
# goes through the gray model. JPEG sources are saved as JPEG at --quality with
# full-resolution colour (4:4:4); lossless sources keep their format. The dry
# run does all of this in memory and writes the report and contact sheets only.

def fix_action(it, threshold: int, max_pixels: int) -> str:
    """fix, keep, or the skip reason."""
    m = it.m
    if m.get("skip"):
        return m["skip"]
    if m["width"] * m["height"] > max_pixels:
        return "large"
    return "fix" if m["qf"] < threshold else "keep"


def encode_jpeg(a: np.ndarray, quality: int) -> bytes:
    """The bytes a JPEG source is written as: 4:4:4, no chroma subsampling, so
    the re-save adds no colour bleeding of its own."""
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG", quality=quality, subsampling=0, optimize=True)
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
        for p in folder.glob("sheet_*.jpg"):
            p.unlink()


class SheetWriter:
    """Collects tiles in QF order and writes a sheet of TILES_PER_SHEET each,
    named by its number and the QF range it shows, with a header row naming the
    columns."""

    def __init__(self, folder: Path, offsets: list[int]):
        self.folder, self.offsets = folder, offsets
        self.tiles: list[tuple[float, Image.Image]] = []
        self.count = 0
        clear_sheets(folder)
        folder.mkdir(parents=True, exist_ok=True)

    def add(self, qf: float, tile: Image.Image) -> None:
        self.tiles.append((qf, tile))
        if len(self.tiles) == TILES_PER_SHEET:
            self.flush()

    def flush(self) -> None:
        if not self.tiles:
            return
        width = max(t.width for _, t in self.tiles)
        heads = ["original"] + [f"QF {o:+d}" for o in self.offsets]
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
        sheet.save(self.folder / f"sheet_{self.count:04d}_q{int(lo):02d}-{int(hi):02d}.jpg", "JPEG",
                   quality=SHEET_QUALITY, subsampling=0)
        self.tiles = []


def fix_one(model: QualityModel, d: dict, qf: float, offsets: list[int], quality: int, sheet: bool) -> dict:
    """Restore one decoded image at qf + offsets[0] (and the other offsets when
    a tile is wanted). -> qf_used, write, qf_after, change, and "outs" (the
    restorations as they would be written) when sheet is set."""
    name, a = ("gray", d["luma"]) if d["gray"] else ("color", d["rgb"])
    outs = []
    for o in (offsets if sheet else offsets[:1]):
        y = model.restore(name, a, min(100.0, qf + o))
        if d["format"] in ("JPEG", "MPO"):
            y = np.array(Image.open(io.BytesIO(encode_jpeg(y, quality))))   # what the file will hold
        outs.append(y)
    r = {"model": name, "qf_used": round(min(100.0, qf + offsets[0]), 1),
         "write": "jpeg" if d["format"] in ("JPEG", "MPO") else d["format"].lower(),
         "qf_after": round(model.qf(name, outs[0]), 1),
         "change": round(float(np.abs(a.astype(np.int16) - outs[0]).mean()), 2)}
    if sheet:
        r["outs"] = outs
    return r


FIX_FIELDS = ["path", "format", "mode", "width", "height", "gray", "header_q", "qf", "action", "model", "qf_used",
              "write", "qf_after", "change"]


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
                        fmt_num(m.get("qf")), it.status, r.get("model", ""), fmt_num(r.get("qf_used")),
                        r.get("write", ""), fmt_num(r.get("qf_after")), fmt_num(r.get("change"))])
    os.replace(tmp, path)
    return path


def fix_dry_run(root: Path, args, model_box: list) -> None:
    t0 = time.perf_counter()
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
    sheets = None if args.no_sheets else SheetWriter(sheet_dir, offsets)
    if sheets is None:
        clear_sheets(sheet_dir)
    if todo:
        if not model_box:
            model_box.append(QualityModel())
        model = model_box[0]
        progress = Progress(root.name, "restored", len(todo))
        with ThreadPoolExecutor(max_workers=args.threads) as pool, ThreadPoolExecutor(max_workers=2) as tiler:
            decoded = bounded_map(pool, lambda it: decode(it.path, args.max_pixels), todo, args.threads * 2)
            pending = deque()
            for n, (it, d) in enumerate(zip(todo, decoded), 1):
                if d["skip"]:                     # changed since it was measured
                    it.status = d["skip"]
                    continue
                r = fix_one(model, d, it.m["qf"], offsets, args.quality, sheets is not None)
                outs = r.pop("outs", None)
                it.fix = r
                if sheets is not None:
                    label = (f"QF {it.m['qf']:.1f}, header {it.m.get('header_q') or '-'}, "
                             f"{it.m['width']}x{it.m['height']}, {r['model']} model, change {r['change']:.2f}")
                    orig = d["luma"] if d["gray"] else d["rgb"]
                    pending.append((it.m["qf"], tiler.submit(make_tile, it.rel, label, orig, outs, d["orientation"])))
                    while pending and (pending[0][1].done() or len(pending) > 8):
                        q, fut = pending.popleft()
                        sheets.add(q, fut.result())
                progress.step(n)
            for q, fut in pending:
                sheets.add(q, fut.result())
    if sheets is not None:
        sheets.flush()
    report = write_report(root, items)
    count = {}
    for it in items:
        k = it.status if it.status in ("fix", "keep") else "skipped"
        count[k] = count.get(k, 0) + 1
    print(f"{root}: {len(items):,} images, {clock(time.perf_counter() - t0)}")
    print(f"  threshold QF {args.threshold}, offset {args.qf_offset:+d}: {count.get('fix', 0):,} to fix, "
          f"{count.get('keep', 0):,} kept, {count.get('skipped', 0):,} skipped")
    fixed = [it for it in items if it.fix]
    if fixed:
        ch = sorted(it.fix["change"] for it in fixed)
        qa = sorted(it.fix["qf_after"] for it in fixed)
        print(f"  change (mean, 8-bit levels): median {ch[len(ch) // 2]:.2f}, max {ch[-1]:.2f}; "
              f"QF after the fix: median {qa[len(qa) // 2]:.1f}, min {qa[0]:.1f}")
    if sheets is not None:
        print(f"  sheets:  {sheets.count} in {sheets.folder}")
    print(f"  report:  {report}")
    print("  dry run: no image was changed")


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
    """The dry run of the fix. An extract folder is a valid input too: its
    copies have the bytes of their sources, so the report and the sheets show
    what a run on the dataset would do, and the dry run writes only into its
    _backup folder, which a re-sort of the extract leaves alone."""
    roots = check_roots(args.folders)
    model_box: list = []
    for root in roots:
        fix_dry_run(root, args, model_box)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Find heavily compressed images in a dataset with FBCNN, and restore them.")
    ap.add_argument("folders", nargs="*",
                    help="dataset folders, scanned at any depth; with --extract also extract folders, sorted again")
    ap.add_argument("--extract", action="store_true",
                    help="copy the images under each band limit into one folder per band, outside the dataset")
    ap.add_argument("--dry-run", action="store_true",
                    help="the fix in memory: write the report and the contact sheets, change no image")
    ap.add_argument("--threshold", type=qf_arg, default=DEFAULT_THRESHOLD, metavar="QF",
                    help=f"fix images with a QF under this (default {DEFAULT_THRESHOLD})")
    ap.add_argument("--qf-offset", type=int, default=DEFAULT_QF_OFFSET, metavar="N",
                    help=f"added to the predicted QF given to the restoration; higher keeps more grain "
                         f"(default {DEFAULT_QF_OFFSET})")
    ap.add_argument("--sheet-offsets", type=parse_offsets, default=[], metavar="LIST",
                    help="more offsets shown side by side on the contact sheets, comma-separated, e.g. 10,20")
    ap.add_argument("--no-sheets", action="store_true", help="no contact sheets")
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
    if args.extract and args.dry_run:
        ap.error("--extract and --dry-run do not go together")
    if not args.extract and not args.dry_run:
        ap.error("the fix has only its --dry-run so far")
    try:
        return extract(args) if args.extract else fix(args)
    except KeyboardInterrupt:
        print("\nStopped. The measurements so far are in the cache; the next run continues from there.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
