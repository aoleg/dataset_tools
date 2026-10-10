#!/usr/bin/env python3
"""
Find watermarks in the images of a dataset and paint them out, or trim them off.

Every folder given is scanned at any depth. A YOLO detector finds the
watermarks; the boxes, grown by --dilate, are painted out with LaMa in a
window of context around each one, so only the watermark pixels change. With
--trim the largest watermark-free rectangle is cut out instead, losslessly for
JPEG (whole DCT blocks), when it keeps at least --trim-min-keep of the image;
otherwise the image is painted. Output goes to <folder>_watermark_removed next
to the folder (or --out), with the same relative paths: images without a
watermark are copied unchanged with their captions, painted JPEGs are written
as JPEG at quality 97 without chroma subsampling, painted images of other
formats as PNG (--png writes every painted image as PNG). --dry-run writes a
report of the detections and, with --previews, the masks drawn over the images
into <folder>/_backup/_watermark; the detections are cached there, so a real
run after a dry run detects nothing again.

Detector: corzent/yolov12x-dino3-watermark-detection (YOLOv12 with a DINOv3
backbone, loaded through dino3_compat.py), fetched from Hugging Face on first
use; when that checkpoint cannot be loaded, the YOLOv11 detector of
fancyfeast's joycaption-watermark-detection space is used. Inpainting: the
big-lama TorchScript model of simple-lama-inpainting, fetched on first use.

Usage:    python watermark.py <folder> [<folder> ...] [--out DIR] [--conf 0.1] [--dilate 15]
          [--trim [--trim-min-keep 0.5]] [--max-size 2048] [--png] [--skip-clean]
          [--dry-run [--previews]] [--overwrite] [--detector dino3|yolo11] [--threads N]
          python watermark.py --fetch-models
Install:  install.bat (torch from the PyTorch CUDA index, ultralytics, transformers, Pillow, numpy, OpenCV, jpeglib)
"""
import argparse
import csv
import json
import math
import os
import shutil
import signal
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

Image.MAX_IMAGE_PIXELS = None

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))                 # dino3_compat.py lives next to this file
MODELS_DIR = HERE / "models"                      # a checkpoint put here is used before the Hugging Face cache
# name: (Hugging Face repo, repo type, file)
DETECTORS = {
    "dino3": ("corzent/yolov12x-dino3-watermark-detection", "model", "yolov12x-dino3-watermark-detection.pt"),
    "yolo11": ("fancyfeast/joycaption-watermark-detection", "space", "yolo11x-train28-best.pt"),
}
DETECTOR_ORDER = ["dino3", "yolo11"]              # the first that loads is used
LAMA_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
LAMA_FILE = "big-lama.pt"                         # in the torch hub checkpoints folder, where simple-lama keeps it

BACKUP_DIRNAME = "_backup"
RUN_DIRNAME = "_watermark"                        # inside _backup: cache, report, previews
CACHE_NAME = "cache.json"
REPORT_NAME = "report.csv"
PREVIEW_DIRNAME = "previews"
LOG_NAME = "log.jsonl"
OUT_SUFFIX = "_watermark_removed"
IMAGE_EXTS = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff"}
# Folders never scanned, at any depth: the output folders of the dataset tools.
DEFAULT_EXCLUDES = [BACKUP_DIRNAME, "_duplicates", "_prep", "_classify", "_embeddings", "masks", "faces"]
DEFAULT_CONF = 0.1
DEFAULT_DILATE = 15
DEFAULT_MAX_SIZE = 2048                           # longest side of a LaMa window; larger windows are scaled down
DEFAULT_TRIM_MIN_KEEP = 0.5                       # a trim must keep this share of the image, else the image is painted
MAX_TRIM_BOXES = 20                               # more boxes than this is a tiled watermark: no useful rectangle
CONTEXT_MIN = 128                                 # px of context around a watermark for LaMa, at least
JPEG_QUALITY = 97
PNG_LEVEL = 6
DETECTOR_VERSION = 1                              # with the detector name and conf: the cache key
DEFAULT_THREADS = 4
REPORT_FIELDS = ["path", "width", "height", "boxes", "action", "trim_box", "result"]


# --- models ---------------------------------------------------------------------

def detector_path(name: str, download: bool = True) -> Path:
    """Where the checkpoint of a detector is: models/ next to this file, else
    the Hugging Face cache; with download, fetched into the cache when it is
    in neither. A plain hf_hub_download asks the Hub on every call, so the
    cache is tried first with the network off."""
    repo, repo_type, fname = DETECTORS[name]
    local = MODELS_DIR / fname
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import LocalEntryNotFoundError
    try:
        return Path(hf_hub_download(repo, fname, repo_type=repo_type, local_files_only=True))
    except LocalEntryNotFoundError:
        if not download:
            raise
        print(f"downloading {fname} from {repo}", flush=True)
        return Path(hf_hub_download(repo, fname, repo_type=repo_type))


def lama_path(download: bool = True) -> Path:
    """The LaMa TorchScript file in the torch hub checkpoints folder (the
    place simple-lama-inpainting keeps it, so a copy it fetched is reused)."""
    import torch.hub
    local = MODELS_DIR / LAMA_FILE
    if local.exists():
        return local
    path = Path(torch.hub.get_dir()) / "checkpoints" / LAMA_FILE
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} is missing; run install.bat or watermark.py --fetch-models")
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {LAMA_FILE} from {LAMA_URL}", flush=True)
        torch.hub.download_url_to_file(LAMA_URL, str(path), progress=False)
    return path


def load_detector(names=None, download: bool = True):
    """The first detector of names (default DETECTOR_ORDER) that loads and
    runs: the DINOv3 checkpoint needs the dino3_compat shim on stock
    ultralytics, and a change of ultralytics or transformers can break it, so
    it is tried on a small image before it is trusted. -> (name, model)"""
    os.environ.setdefault("YOLO_OFFLINE", "1")   # no online check, no analytics, no automatic downloads
    from ultralytics import YOLO
    import dino3_compat
    dino3_compat.register()
    errors = []
    for name in names or DETECTOR_ORDER:
        try:
            path = detector_path(name, download)
            model = YOLO(str(path))
            model.predict(np.zeros((64, 64, 3), np.uint8), conf=0.5, verbose=False)
            return name, model
        except Exception as e:  # noqa: BLE001 - the next detector is tried
            errors.append(f"{name}: {type(e).__name__}: {e}")
            print(f"  [warning] detector {name} could not be loaded ({type(e).__name__}: {e})", flush=True)
    raise RuntimeError("no watermark detector could be loaded:\n  " + "\n  ".join(errors))


class Lama:
    """The big-lama TorchScript model, as simple-lama-inpainting runs it: the
    image and the mask padded to a multiple of 8 (symmetric), the mask as a
    0/1 tensor, the result cut back to the image size."""

    def __init__(self, device=None, download: bool = True):
        import torch
        self.torch = torch
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(str(lama_path(download)), map_location=self.device)
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """rgb uint8 HxWx3, mask uint8 HxW (any non-zero is masked) -> uint8 HxWx3."""
        torch = self.torch
        h, w = mask.shape
        ph, pw = -h % 8, -w % 8
        img = np.pad(rgb.astype(np.float32) / 255.0, ((0, ph), (0, pw), (0, 0)), mode="symmetric")
        m = np.pad(mask.astype(np.float32) / 255.0, ((0, ph), (0, pw)), mode="symmetric")
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None].to(self.device)
        y = (torch.from_numpy(m)[None, None].to(self.device) > 0) * 1
        with torch.inference_mode():
            out = self.model(x, y)[0].permute(1, 2, 0).cpu().numpy()
        return np.clip(out * 255, 0, 255).astype(np.uint8)[:h, :w]


_detector: tuple | None = None
_lama: Lama | None = None


def detector_lazy(names=None) -> tuple:
    """(name, model) of the detector, loaded on first use and kept."""
    global _detector
    if _detector is None:
        _detector = load_detector(names)
    return _detector


def lama_lazy() -> Lama:
    global _lama
    if _lama is None:
        _lama = Lama()
    return _lama


# --- detection and painting ---------------------------------------------------------

def detect(model, array: np.ndarray, conf: float) -> list:
    """The watermark boxes of an upright RGB uint8 array: [[x0, y0, x1, y1, conf], ...]
    with integer edges, as the detector reports them (no dilation)."""
    bgr = np.ascontiguousarray(array[:, :, ::-1])        # ultralytics takes a numpy array as BGR
    r = model.predict(bgr, conf=conf, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    out = []
    for box, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
        x0, y0, x1, y1 = (int(v) for v in box)
        out.append([x0, y0, x1, y1, round(float(c), 3)])
    return out


def boxes_mask(shape, boxes, dilate: int) -> np.ndarray:
    """A uint8 mask (255 inside) of the boxes, grown by an ellipse of dilate px."""
    import cv2
    mask = np.zeros(shape[:2], np.uint8)
    for b in boxes:
        x0, y0, x1, y1 = (int(v) for v in b[:4])
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)
    if dilate > 0 and boxes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def largest_clear_rect(boxes, width: int, height: int, max_boxes: int = MAX_TRIM_BOXES):
    """The largest [x0, y0, x1, y1] of the image that overlaps no box, or None.
    Each side of the best rectangle lies on the image border or on a box edge,
    so only those candidates are tested."""
    boxes = np.asarray([b[:4] for b in boxes], dtype=np.int64).reshape(-1, 4)
    if len(boxes) > max_boxes:
        return None                                       # a tiled watermark; no useful rectangle is left
    lefts, rights = np.r_[0, boxes[:, 2]], np.r_[width, boxes[:, 0]]
    tops, bottoms = np.r_[0, boxes[:, 3]], np.r_[height, boxes[:, 1]]
    L, R, T, B = np.meshgrid(lefts, rights, tops, bottoms, indexing="ij")
    L, R, T, B = L.ravel(), R.ravel(), T.ravel(), B.ravel()
    area = np.clip(R - L, 0, None) * np.clip(B - T, 0, None)
    hits = ((boxes[:, 0][None] < R[:, None]) & (L[:, None] < boxes[:, 2][None]) &
            (boxes[:, 1][None] < B[:, None]) & (T[:, None] < boxes[:, 3][None])).any(axis=1)
    area[hits] = 0
    best = int(area.argmax())
    return [int(L[best]), int(T[best]), int(R[best]), int(B[best])] if area[best] > 0 else None


def mask_rects(mask: np.ndarray) -> list:
    """The windows LaMa paints: each connected area of the mask with context
    around it (CONTEXT_MIN px or half its size), overlapping windows merged,
    so no window sees an unfilled part of another watermark as context."""
    import cv2
    H, W = mask.shape
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    rects = []
    for x, y, w, h, _ in stats[1:]:
        margin = max(CONTEXT_MIN, max(w, h) // 2)
        rects.append([max(0, x - margin), max(0, y - margin), min(W, x + w + margin), min(H, y + h + margin)])
    merged = True
    while merged:
        merged = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, b = rects[i], rects[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    rects[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del rects[j]
                    merged = True
                    break
            if merged:
                break
    return rects


def inpaint(array: np.ndarray, boxes, dilate: int = DEFAULT_DILATE, max_size: int = DEFAULT_MAX_SIZE,
            lama: Lama | None = None) -> np.ndarray:
    """Paint the boxes out of an upright RGB uint8 array with LaMa, each in a
    window of context (mask_rects); a window whose longest side exceeds
    max_size is scaled down for LaMa and its result scaled back. Only the
    masked pixels change. -> a new array (array itself when there is no box)."""
    import cv2
    if not boxes:
        return array
    lama = lama or lama_lazy()
    mask = boxes_mask(array.shape, boxes, dilate)
    result = array.copy()
    for x1, y1, x2, y2 in mask_rects(mask):
        crop, crop_mask = array[y1:y2, x1:x2], mask[y1:y2, x1:x2]
        h, w = crop_mask.shape
        scale = min(1.0, max_size / max(h, w)) if max_size > 0 else 1.0
        if scale < 1.0:
            size = (max(8, round(w * scale)), max(8, round(h * scale)))
            small = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
            small_mask = (cv2.resize(crop_mask, size, interpolation=cv2.INTER_AREA) > 0).astype(np.uint8) * 255
            filled = cv2.resize(lama(small, small_mask), (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            filled = lama(np.ascontiguousarray(crop), crop_mask)
        selected = crop_mask > 0
        result[y1:y2, x1:x2][selected] = filled[selected]
    return result


# --- per-image API ---------------------------------------------------------------
# What the pipeline tool calls, one image at a time: plan() finds the
# watermarks of the upright RGB pixels and decides between painting and a
# trim, apply() paints or cuts. The command line goes through the same two.

def plan(array: np.ndarray, head: dict | None = None, options: dict | None = None) -> dict:
    """The watermarks of one image and what to do about them. array: the
    upright RGB uint8 pixels (head is not needed; it is accepted for the
    uniform call); options: conf (default DEFAULT_CONF), dilate, mode
    ("inpaint", or "trim" to cut the largest clear rectangle when it keeps
    trim_min_keep of the image, else paint), max_size (LaMa window), detector
    (the model, default detector_lazy()), boxes (cached detections of this
    image, else the detector runs).
    -> {"boxes": [[x0, y0, x1, y1, conf], ...], "action": "" (no watermark),
        inpaint or trim, "trim_box": [x0, y0, x1, y1] for a trim, "size":
        [w, h], and the dilate, max_size and mode used}"""
    opts = {"conf": DEFAULT_CONF, "dilate": DEFAULT_DILATE, "mode": "inpaint", "trim_min_keep": DEFAULT_TRIM_MIN_KEEP,
            "max_size": DEFAULT_MAX_SIZE, "detector": None, "boxes": None, **(options or {})}
    h, w = array.shape[:2]
    boxes = opts["boxes"]
    if boxes is None:
        model = opts["detector"] or detector_lazy()[1]
        boxes = detect(model, array, opts["conf"])
    boxes = [list(b) for b in boxes]
    out = {"boxes": boxes, "action": "", "trim_box": None, "size": [w, h], "dilate": opts["dilate"],
           "max_size": opts["max_size"], "mode": opts["mode"]}
    if not boxes:
        return out
    out["action"] = "inpaint"
    if opts["mode"] == "trim":
        mask = boxes_mask(array.shape, boxes, opts["dilate"])
        rect = largest_clear_rect(mask_boxes(mask), w, h)
        if rect and (rect[2] - rect[0]) * (rect[3] - rect[1]) >= opts["trim_min_keep"] * w * h:
            out.update(action="trim", trim_box=rect)
    return out


def mask_boxes(mask: np.ndarray) -> list:
    """The bounding boxes of the connected areas of a mask."""
    import cv2
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [[x, y, x + w, y + h] for x, y, w, h, _ in stats[1:]]


def apply(array: np.ndarray, plan: dict, lama: Lama | None = None) -> np.ndarray:
    """The image with its watermarks painted out, or cut to the trim box; the
    array itself when there is no watermark."""
    if plan["action"] == "trim":
        x0, y0, x1, y1 = plan["trim_box"]
        return array[y0:y1, x0:x1]
    if plan["action"] == "inpaint":
        return inpaint(array, plan["boxes"], plan.get("dilate", DEFAULT_DILATE), plan.get("max_size", DEFAULT_MAX_SIZE),
                       lama)
    return array


# --- files ----------------------------------------------------------------------------

def scan(root: Path, excludes) -> list[Path]:
    excl = {e.casefold() for e in excludes}
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d.casefold() not in excl)
        for fn in sorted(filenames):
            if os.path.splitext(fn)[1].casefold() in IMAGE_EXTS:
                out.append(Path(dirpath) / fn)
    return out


def image_header(path: Path) -> dict:
    """Format, EXIF orientation, stored size and JPEG MCU size from the header."""
    with Image.open(path) as im:
        try:
            o = im.getexif().get(274, 1) or 1
        except Exception:  # noqa: BLE001
            o = 1
        fmt = "JPEG" if im.format == "MPO" else (im.format or "")
        head = {"format": fmt, "orientation": o if o in range(1, 9) else 1, "stored": list(im.size), "mcu": None}
        layer = getattr(im, "layer", None)
        if fmt == "JPEG" and layer:
            head["mcu"] = [8 * max(c[1] for c in layer), 8 * max(c[2] for c in layer)]
    return head


def load_image(path: Path) -> np.ndarray:
    """The upright RGB pixels, transparency composited over white."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return np.asarray(Image.alpha_composite(bg, rgba).convert("RGB"))
        return np.asarray(im.convert("RGB"))


def display_to_stored(box, orientation, sw, sh):
    """A box in upright (display) pixels -> the same box in stored pixels."""
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
        return [y0, sh - x1, y1, sh - x0]
    if orientation == 7:
        return [sw - y1, sh - x1, sw - y0, sh - x0]
    if orientation == 8:
        return [sw - y1, x0, sw - y0, x1]
    return [x0, y0, x1, y1]


def exif_for_output(exif_bytes: bytes | None, width: int, height: int, keep_orientation: bool) -> bytes | None:
    """The source EXIF without its thumbnail, with the new size; the
    orientation reset when the pixels written are already upright."""
    if not exif_bytes:
        return None
    try:
        ex = Image.Exif()
        ex.load(exif_bytes)
        sub = ex.get_ifd(0x8769)
        if sub:
            sub[0xA002], sub[0xA003] = width, height
        if not keep_orientation and 274 in ex:
            ex[274] = 1
        return ex.tobytes()
    except Exception:  # noqa: BLE001
        return None


def rewrite_jpeg_header(data: bytes, exif: bytes | None) -> bytes:
    """Replace the EXIF segment of a JPEG file and drop the MPF index, on the
    marker segments before the image data."""
    if data[:2] != b"\xff\xd8":
        return data
    out, pos, done_exif = [data[:2]], 2, False
    while pos + 4 <= len(data) and data[pos] == 0xFF:
        marker = data[pos + 1]
        if marker == 0xDA:
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
            seg = b""
        out.append(seg)
        pos += 2 + length
    out.append(data[pos:])
    return b"".join(out)


def jpeglib_dir() -> Path:
    """A new ASCII-only folder for jpeglib's files; libjpeg opens paths as
    narrow strings, so a Cyrillic path fails."""
    base = Path(tempfile.gettempdir())
    if not str(base).isascii():
        base = Path(Path.cwd().anchor or "C:/")
    return Path(tempfile.mkdtemp(prefix="watermark_", dir=base))


def crop_dct(im, box) -> None:
    """Cut the DCT coefficient arrays of a jpeglib image to the box (stored pixels, origin on the MCU grid)."""
    sf = np.asarray(im.samp_factor)
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


def lossless_trim_box(display_box, head: dict) -> list | None:
    """The stored box of a lossless JPEG trim: the display box in stored
    pixels with its left and top moved inward to the MCU grid, so the
    watermark goes completely. None when nothing is left."""
    if head.get("format") != "JPEG" or not head.get("mcu"):
        return None
    sw, sh = head["stored"]
    x0, y0, x1, y1 = display_to_stored(display_box, head["orientation"], sw, sh)
    mw, mh = head["mcu"]
    x0, y0 = -(-x0 // mw) * mw, -(-y0 // mh) * mh
    return [x0, y0, x1, y1] if x1 > x0 and y1 > y0 else None


def write_lossless(src: Path, dst: Path, stored_box) -> None:
    """Crop whole DCT blocks with jpeglib, no decoding, no re-encoding. Run in
    worker processes only: jpeglib corrupts the heap when Pillow runs in other threads."""
    import jpeglib
    work = jpeglib_dir()
    try:
        shutil.copyfile(src, work / "in.jpg")
        im = jpeglib.read_dct(str(work / "in.jpg"))
        crop_dct(im, stored_box)
        im.write_dct(str(work / "out.jpg"))
        data = (work / "out.jpg").read_bytes()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    x0, y0, x1, y1 = stored_box
    with Image.open(src) as orig:
        exif = exif_for_output(orig.info.get("exif"), x1 - x0, y1 - y0, keep_orientation=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    tmp.write_bytes(rewrite_jpeg_header(data, exif))
    os.replace(tmp, dst)


def write_array(array: np.ndarray, dst: Path, fmt: str, exif: bytes | None, icc: bytes | None) -> None:
    """Write upright pixels as JPEG (quality 97, no chroma subsampling) or PNG."""
    im = Image.fromarray(np.ascontiguousarray(array))
    kw = {k: v for k, v in (("exif", exif_for_output(exif, im.width, im.height, keep_orientation=False)),
                            ("icc_profile", icc)) if v}
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    if fmt == "JPEG":
        im.save(tmp, "JPEG", quality=JPEG_QUALITY, subsampling=0, optimize=True, **kw)
    else:
        im.save(tmp, "PNG", compress_level=PNG_LEVEL, **kw)
    os.replace(tmp, dst)


def write_job(job: dict) -> dict:
    """One output file; runs in a worker process. -> {"out": path} or {"error": ...}"""
    try:
        src, dst = Path(job["src"]), Path(job["dst"])
        kind = job["kind"]
        if kind == "copy":
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
        elif kind == "lossless":
            write_lossless(src, dst, job["stored_box"])
        else:
            with Image.open(src) as im:
                exif, icc = im.info.get("exif"), im.info.get("icc_profile")
            write_array(job["array"], dst, job["fmt"], exif, icc)
        for cap_src, cap_dst in job.get("captions", []):
            Path(cap_dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cap_src, cap_dst)
        return {"out": str(dst)}
    except Exception as e:  # noqa: BLE001 - one bad file is reported, not fatal
        return {"error": f"{type(e).__name__}: {e}".strip()}


def ignore_ctrl_c() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# --- cache, report, previews ----------------------------------------------------------

def file_key(path: Path) -> list[int]:
    st = os.stat(path)
    return [st.st_size, st.st_mtime_ns, st.st_ino]


def run_dir(root: Path) -> Path:
    return root / BACKUP_DIRNAME / RUN_DIRNAME


def cache_signature(detector: str, conf: float) -> dict:
    return {"version": DETECTOR_VERSION, "detector": detector, "conf": conf}


def load_cache(root: Path, signature: dict) -> dict:
    try:
        data = json.loads((run_dir(root) / CACHE_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data.get("files", {}) if data.get("signature") == signature else {}


def save_cache(root: Path, signature: dict, files: dict) -> None:
    path = run_dir(root) / CACHE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps({"signature": signature, "files": files}, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def draw_preview(array: np.ndarray, p: dict, dest: Path) -> None:
    """The image with the mask tinted red and the trim box in yellow, at most 1600 px."""
    im = Image.fromarray(array).convert("RGBA")
    over = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    mask = boxes_mask(array.shape, p["boxes"], p["dilate"])
    red = Image.new("RGBA", im.size, (255, 0, 0, 110))
    over.paste(red, (0, 0), Image.fromarray(mask))
    for b in p["boxes"]:
        d.rectangle(b[:4], outline=(255, 60, 60, 255), width=3)
    if p["trim_box"]:
        d.rectangle(p["trim_box"], outline=(255, 230, 0, 255), width=4)
    im = Image.alpha_composite(im, over).convert("RGB")
    s = 1600 / max(im.size)
    if s < 1:
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest, quality=85)


def write_report(root: Path, rows: list) -> Path:
    path = run_dir(root) / REPORT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(REPORT_FIELDS)
        w.writerows(rows)
    os.replace(tmp, path)
    return path


# --- one folder --------------------------------------------------------------------------

def output_name(rel: str, fmt: str, png: bool) -> str:
    """Where a painted image goes: JPEG stays .jpg, everything else becomes .png."""
    p = Path(rel)
    if fmt == "JPEG" and not png:
        return rel
    return p.with_suffix(".png").as_posix()


def process_folder(root: Path, out: Path, args) -> int:
    """Detect, paint or trim, and write one folder. -> failures."""
    t0 = time.perf_counter()
    files = scan(root, DEFAULT_EXCLUDES + args.exclude)
    print(f"{root}: {len(files):,} images found", flush=True)
    if not files:
        return 0
    signature = None
    cache, fresh = {}, {}
    names = [args.detector] if args.detector else None
    todo_detect = []
    keys = {}
    for p in files:
        rel = p.relative_to(root).as_posix()
        try:
            keys[rel] = file_key(p)
        except OSError:
            keys[rel] = None
    # the cache is read once the detector is known (its name is part of the key);
    # with --reanalyse or no cache, every image is detected
    name_hint = args.detector or DETECTOR_ORDER[0]
    if not args.reanalyse:
        cache = load_cache(root, cache_signature(name_hint, args.conf))
    for p in files:
        rel = p.relative_to(root).as_posix()
        hit = cache.get(rel)
        if not (hit and keys[rel] and hit["key"] == keys[rel]):
            todo_detect.append(p)
    detector_name = name_hint
    if todo_detect:
        detector_name, _ = detector_lazy(names)
        if detector_name != name_hint:
            cache = {} if args.reanalyse else load_cache(root, cache_signature(detector_name, args.conf))
    signature = cache_signature(detector_name, args.conf)
    mode = "trim" if args.trim else "inpaint"
    options = {"conf": args.conf, "dilate": args.dilate, "mode": mode, "trim_min_keep": args.trim_min_keep,
               "max_size": args.max_size}
    rows, jobs, new_cache = [], [], {}
    counts = {"painted": 0, "trimmed": 0, "clean": 0, "skipped": 0, "failed": 0}
    preview_dir = run_dir(root) / PREVIEW_DIRNAME
    if args.dry_run and args.previews and preview_dir.is_dir():
        shutil.rmtree(preview_dir)

    def decoded(p):
        try:
            return p, load_image(p), image_header(p), ""
        except Exception as e:  # noqa: BLE001
            return p, None, None, f"{type(e).__name__}: {e}"

    pool = ProcessPoolExecutor(max_workers=args.threads, initializer=ignore_ctrl_c) if not args.dry_run else None
    pending = []
    try:
        with ThreadPoolExecutor(max_workers=args.threads) as tp:
            for n, (p, array, head, err) in enumerate(tp.map(decoded, files), 1):
                rel = p.relative_to(root).as_posix()
                if err:
                    rows.append([rel, "", "", "", "unreadable", "", err])
                    counts["failed"] += 1
                    print(f"[{n}/{len(files)}] {rel}: unreadable: {err}", flush=True)
                    continue
                hit = cache.get(rel)
                boxes = hit["boxes"] if hit and keys[rel] and hit["key"] == keys[rel] else None
                plan_ = plan(array, head, dict(options, boxes=boxes))
                if keys[rel]:
                    new_cache[rel] = {"key": keys[rel], "boxes": plan_["boxes"]}
                h, w = array.shape[:2]
                what = plan_["action"] or "clean"
                if args.dry_run and args.previews and plan_["boxes"]:
                    draw_preview(array, plan_, preview_dir / (rel + ".jpg"))
                result = ""
                row = [rel, w, h, ";".join(",".join(str(v) for v in b) for b in plan_["boxes"]), what,
                       ",".join(str(v) for v in plan_["trim_box"]) if plan_["trim_box"] else "", result]
                rows.append(row)
                if not args.dry_run:
                    cap = p.with_suffix(".txt")
                    captions = [(str(cap), str(out / Path(rel).with_suffix(".txt")))] if cap.is_file() else []
                    job = None
                    if plan_["action"] == "":
                        if not args.skip_clean:
                            job = {"kind": "copy", "src": str(p), "dst": str(out / rel), "captions": captions}
                    elif plan_["action"] == "trim":
                        stored = lossless_trim_box(plan_["trim_box"], head) if not args.png else None
                        if stored:
                            job = {"kind": "lossless", "src": str(p), "dst": str(out / rel), "stored_box": stored,
                                   "captions": captions}
                        else:
                            job = {"kind": "array", "src": str(p), "dst": str(out / output_name(rel, head["format"], args.png)),
                                   "array": apply(array, plan_), "fmt": "JPEG" if head["format"] == "JPEG" and not args.png else "PNG",
                                   "captions": captions}
                    else:
                        job = {"kind": "array", "src": str(p), "dst": str(out / output_name(rel, head["format"], args.png)),
                               "array": apply(array, plan_), "fmt": "JPEG" if head["format"] == "JPEG" and not args.png else "PNG",
                               "captions": captions}
                    if job is not None:
                        if Path(job["dst"]).exists() and not args.overwrite:
                            result = "exists, skipped"
                            counts["skipped"] += 1
                        else:
                            pending.append((row, what, pool.submit(write_job, job)))
                            while len(pending) > args.threads * 2:
                                row0, what0, fut = pending.pop(0)
                                finish(fut.result(), what0, row0, counts)
                    else:
                        result = "clean, not written"
                row[-1] = result
                if args.dry_run or result:
                    counts["painted" if what == "inpaint" else "trimmed" if what == "trim" else "clean"] += \
                        1 if result != "exists, skipped" else 0
                boxes_txt = f"{len(plan_['boxes'])} watermark(s)" if plan_["boxes"] else "no watermark"
                print(f"[{n}/{len(files)}] {rel}: {boxes_txt}, {what}" + (f" ({result})" if result else ""), flush=True)
        for row0, what0, fut in pending:
            finish(fut.result(), what0, row0, counts)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        save_cache(root, signature, {**{k: v for k, v in cache.items() if k not in new_cache}, **new_cache})
    report = write_report(root, rows)
    print(f"{root}: {len(files):,} images, {time.perf_counter() - t0:.0f}s; "
          f"{counts['painted']:,} painted, {counts['trimmed']:,} trimmed, {counts['clean']:,} without a watermark"
          + (f", {counts['skipped']:,} skipped (output exists)" if counts["skipped"] else "")
          + (f", {counts['failed']:,} failed" if counts["failed"] else ""))
    print(f"  report:  {report}")
    if args.dry_run:
        print("  dry run: nothing was written" + (f"; previews in {preview_dir}" if args.previews else ""))
    else:
        print(f"  output:  {out}")
    return counts["failed"]


def finish(res: dict, what: str, row: list, counts: dict) -> None:
    """Fill the result of a written image into its report row."""
    if "error" in res:
        counts["failed"] += 1
        row[-1] = "error: " + res["error"]
        print(f"  could not write {row[0]}: {res['error']}", flush=True)
    else:
        counts["painted" if what == "inpaint" else "trimmed" if what == "trim" else "clean"] += 1
        row[-1] = "written"


# --- command line ---------------------------------------------------------------------

def threads_arg(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--threads must be an integer, got {value!r}")
    if not 1 <= n <= 32:
        raise argparse.ArgumentTypeError(f"--threads must be 1..32, got {n}")
    return n


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass
    ap = argparse.ArgumentParser(description="Find watermarks in the images of a dataset and paint them out, or trim them off.")
    ap.add_argument("folders", nargs="*", help="dataset folders, scanned at any depth")
    ap.add_argument("--out", metavar="DIR", help=f"output folder (default <folder>{OUT_SUFFIX} next to the folder; one folder only)")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF, help=f"detector confidence threshold (default {DEFAULT_CONF})")
    ap.add_argument("--dilate", type=int, default=DEFAULT_DILATE, metavar="N",
                    help=f"grow the detected boxes by N pixels before painting (default {DEFAULT_DILATE})")
    ap.add_argument("--trim", action="store_true",
                    help="cut the largest watermark-free rectangle out instead of painting, when it keeps enough")
    ap.add_argument("--trim-min-keep", type=float, default=DEFAULT_TRIM_MIN_KEEP, metavar="X",
                    help=f"with --trim: the share of the image a trim must keep, else the image is painted (default {DEFAULT_TRIM_MIN_KEEP})")
    ap.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE, metavar="N",
                    help=f"longest side of a LaMa window; larger windows are scaled down for LaMa (default {DEFAULT_MAX_SIZE})")
    ap.add_argument("--png", action="store_true", help="write every painted or trimmed image as PNG")
    ap.add_argument("--skip-clean", action="store_true", help="do not copy the images without a watermark")
    ap.add_argument("--dry-run", action="store_true", help="detect only: write the report, and with --previews the masks")
    ap.add_argument("--previews", action="store_true", help="with --dry-run: draw the masks over the images")
    ap.add_argument("--overwrite", action="store_true", help="write outputs that exist already")
    ap.add_argument("--detector", choices=list(DETECTORS), help="use this detector only (default: dino3, else yolo11)")
    ap.add_argument("--exclude", action="append", default=[], metavar="NAME",
                    help="another folder name to skip at any depth; may repeat. Always skipped: " + ", ".join(DEFAULT_EXCLUDES))
    ap.add_argument("--reanalyse", action="store_true", help="ignore the detection cache")
    ap.add_argument("--threads", type=threads_arg, default=DEFAULT_THREADS, metavar="N",
                    help=f"decoder threads and writer processes (default {DEFAULT_THREADS}); detection runs one image at a time")
    ap.add_argument("--fetch-models", action="store_true", help="download the detector and LaMa and load them once")
    args = ap.parse_args(argv)

    if args.fetch_models:
        name, _ = load_detector([args.detector] if args.detector else None)
        print(f"detector: {name} ({detector_path(name, False)})")
        Lama()
        print(f"lama: {lama_path(False)}")
        print("models load OK")
        if not args.folders:
            return 0
    if not args.folders:
        ap.error("give at least one folder")
    roots = [Path(f).resolve() for f in args.folders]
    for r in roots:
        if not r.is_dir():
            ap.error(f"not a folder: {r}")
    if args.out and len(roots) > 1:
        ap.error("--out works with one folder only")
    failed = 0
    for root in roots:
        out = Path(args.out).resolve() if args.out else root.parent / (root.name + OUT_SUFFIX)
        if out == root or out.is_relative_to(root):
            ap.error(f"the output folder {out} is inside {root}; give --out outside it")
        failed += process_folder(root, out, args)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
