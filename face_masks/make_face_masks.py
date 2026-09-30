#!/usr/bin/env python3
"""
Generate ai-toolkit loss masks that suppress faces.

For every image in IMG_DIR, writes MASK_DIR/<same basename>.png
(MASK_DIR defaults to IMG_DIR/masks):
  white (255) everywhere  -> full loss
  black (0) over faces    -> loss scaled to dataset mask_min_value
Faces are covered by a soft-edged ellipse grown by --grow around the detected box.
With --invert the colours swap: only the faces are trained, the rest is masked.

Detector: YOLOv8 face model from the adetailer repo (Bingsu/adetailer on HF).
Install:  pip install ultralytics huggingface_hub pillow numpy
Usage:    python make_face_masks.py <img_dir> [<mask_dir>] [--grow 1.35] [--feather 12]
          [--conf 0.3] [--model face_yolov8m.pt] [--include-hair] [--invert] [--preview <dir>]
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MODEL_REPO = "Bingsu/adetailer"
DEFAULT_MODEL = "face_yolov8m.pt"


def fetch_model(name: str) -> str:
    """Local path of the model, downloaded into the Hugging Face cache the first time.

    A plain hf_hub_download asks the Hub for the latest revision on every call,
    even when the file is cached, so the cache is tried first with the network off.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    try:
        return hf_hub_download(MODEL_REPO, name, local_files_only=True)
    except LocalEntryNotFoundError:
        print(f"downloading {name} from {MODEL_REPO}", flush=True)
        return hf_hub_download(MODEL_REPO, name)


def load_model(name: str):
    path = fetch_model(name)
    # Must be set before ultralytics is imported: it skips the DNS online check
    # at import time, and with it the usage analytics that ultralytics sends
    # when it thinks it is online. Inference needs no network.
    os.environ.setdefault("YOLO_OFFLINE", "1")
    from ultralytics import YOLO

    return YOLO(path)


def face_boxes(model, img: Image.Image, conf: float):
    res = model.predict(img, conf=conf, verbose=False)[0]
    if res.boxes is None:
        return []
    return [tuple(map(float, b)) for b in res.boxes.xyxy.cpu().numpy()]


def face_region(box, grow: float, include_hair: bool):
    """Detected face box -> (x0, y0, x1, y1) of the grown face area, not clipped to the image."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = (x1 - x0) * grow, (y1 - y0) * grow
    if include_hair:
        # extend upward by ~60% of face height and widen a little
        cy -= bh * 0.25
        bh *= 1.5
        bw *= 1.15
    return cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2


def make_mask(size, boxes, grow: float, feather: int, include_hair: bool,
              invert: bool = False) -> Image.Image:
    """White (trained) everywhere and black over faces; with invert, the opposite."""
    face, rest = (255, 0) if invert else (0, 255)
    w, h = size
    mask = Image.new("L", (w, h), rest)
    if not boxes:
        return mask
    draw = ImageDraw.Draw(mask)
    for box in boxes:
        draw.ellipse(list(face_region(box, grow, include_hair)), fill=face)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    return mask


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("img_dir")
    ap.add_argument("mask_dir", nargs="?", default=None, help="output dir for masks (default: <img_dir>/masks)")
    ap.add_argument("--grow", type=float, default=1.35, help="scale factor on the detected face box")
    ap.add_argument("--feather", type=int, default=12, help="Gaussian blur radius in px for the mask edge")
    ap.add_argument("--conf", type=float, default=0.3, help="detector confidence threshold")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="adetailer face model file name")
    ap.add_argument("--include-hair", action="store_true", help="extend the ellipse upward to cover hair")
    ap.add_argument("--invert", action="store_true",
                    help="train only the faces: white ellipses on black, surroundings masked")
    ap.add_argument("--preview", default=None, help="optional dir for overlay previews")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.img_dir):
        sys.exit(f"not a directory: {args.img_dir}")
    if args.mask_dir is None:
        args.mask_dir = os.path.join(args.img_dir, "masks")
        print(f"mask_dir not given, writing masks to {args.mask_dir}")

    os.makedirs(args.mask_dir, exist_ok=True)
    if args.preview:
        os.makedirs(args.preview, exist_ok=True)

    files = sorted(
        f for f in os.listdir(args.img_dir) if os.path.splitext(f)[1].lower() in IMG_EXTS
    )
    if not files:
        sys.exit(f"no images in {args.img_dir}")

    model = load_model(args.model)
    no_face = []
    for i, f in enumerate(files, 1):
        stem = os.path.splitext(f)[0]
        out = os.path.join(args.mask_dir, stem + ".png")
        if os.path.exists(out) and not args.overwrite:
            continue
        img = ImageOps.exif_transpose(Image.open(os.path.join(args.img_dir, f))).convert("RGB")
        boxes = face_boxes(model, img, args.conf)
        if not boxes:
            no_face.append(f)
        mask = make_mask(img.size, boxes, args.grow, args.feather, args.include_hair, args.invert)
        mask.save(out)
        if args.preview:
            red = Image.new("RGB", img.size, (255, 0, 0))
            alpha = ImageOps.invert(mask).point(lambda v: int(v * 0.55))
            prev = img.copy()
            prev.paste(red, (0, 0), alpha)
            prev.save(os.path.join(args.preview, stem + ".jpg"), quality=85)
        print(f"[{i}/{len(files)}] {f}: {len(boxes)} face(s)", flush=True)

    if no_face:
        colour = "black: nothing in it is trained" if args.invert else "white"
        print(f"\n{len(no_face)} image(s) with no detected face (mask is all {colour}):")
        for f in no_face:
            print("  ", f)


if __name__ == "__main__":
    main()
