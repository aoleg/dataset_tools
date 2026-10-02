#!/usr/bin/env python3
"""
Reframe photos of people: trim wasted space around the subject (one person or a
group), leave passers-by out, and crop to one of the k2prep aspect ratios.

Phases 1-2 (this version): detection and per-person measurements. Every
image is analysed with three models; the detections are merged into one list
of people, each person is measured, and everything is cached. --previews draws
the detections, --people draws the merged people with their measurements.
  person outlines : yolo26x-seg.pt   (ultralytics assets, COCO class "person")
  faces           : face_yolov8m.pt  (Bingsu/adetailer, Hugging Face)
  pose keypoints  : yolo26x-pose.pt  (ultralytics assets)

Usage:    python reframe.py <folder> [<folder> ...] [--out DIR] [--previews]
          python reframe.py --fetch-models
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

Image.MAX_IMAGE_PIXELS = None

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
OUT_DIRNAME = "_reframed"
DETECTIONS_NAME = "detections.json"
PREVIEW_DIRNAME = "_preview"
CACHE_VERSION = 1
PEOPLE_VERSION = 3           # bump when build_people or measure_people changes
PEOPLE_PREVIEW_DIRNAME = "_people"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# name -> (source, file). "hf" files come from Bingsu/adetailer on Hugging Face,
# "ultralytics" files from the ultralytics GitHub assets.
HF_REPO = "Bingsu/adetailer"
# The COCO segmentation model replaced adetailer's person_yolov8m-seg.pt: on the
# calibration photos both outlined the clearly detected people equally (219 of
# 223), but of 47 weakly detected ones (small, distant, partly hidden) COCO
# outlined 40 and adetailer 28. Neither outlines people in unusual clothing
# reliably, while pose and face may still find them, so the person list cannot
# rely on outlines alone.
MODELS = {
    "person": ("ultralytics", "yolo26x-seg.pt"),
    "face": ("hf", "face_yolov8m.pt"),
    "pose": ("ultralytics", "yolo26x-pose.pt"),
}
PERSON_CLASSES = [0]          # COCO "person"
# Detection runs at this long side. Phone photos are 3000-5000 px; at 640 a
# person 5% of the frame high is 32 px and easily missed, at 1280 it is 64 px.
IMGSZ = 1280
CONF = {"person": 0.25, "face": 0.30, "pose": 0.25}

# COCO keypoint order used by the pose model
KP_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
            "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
            "l_knee", "r_knee", "l_ankle", "r_ankle"]
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def model_path(name: str, download: bool) -> Path:
    """Local path of a model; with download, fetch it if it is missing.

    Hugging Face files are looked up in the cache first, so a run with the
    models present makes no network request.
    """
    source, fname = MODELS[name]
    if source == "hf":
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import LocalEntryNotFoundError
        try:
            return Path(hf_hub_download(HF_REPO, fname, local_files_only=True))
        except LocalEntryNotFoundError:
            if not download:
                raise
            print(f"downloading {fname} from {HF_REPO}", flush=True)
            return Path(hf_hub_download(HF_REPO, fname))
    path = MODELS_DIR / fname
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} is missing; run install.bat or reframe.py --fetch-models")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"downloading {fname} from the ultralytics assets", flush=True)
        from ultralytics.utils.downloads import attempt_download_asset
        attempt_download_asset(str(path))
        if not path.exists():
            raise RuntimeError(f"download of {fname} failed")
    return path


def load_models(download: bool = False) -> dict:
    paths = {name: model_path(name, download) for name in MODELS}
    # Set before ultralytics is imported: no DNS online check, no usage
    # analytics, no automatic downloads. Inference needs no network.
    os.environ.setdefault("YOLO_OFFLINE", "1")
    from ultralytics import YOLO
    return {name: YOLO(str(p)) for name, p in paths.items()}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def load_image(path: Path) -> Image.Image:
    with Image.open(path) as im:
        img = ImageOps.exif_transpose(im)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.load()
    return img


def simplify(poly: np.ndarray, tol: float) -> list:
    import cv2
    if len(poly) < 3:
        return []
    approx = cv2.approxPolyDP(poly.astype(np.float32).reshape(-1, 1, 2), tol, True).reshape(-1, 2)
    return [[int(round(x)), int(round(y))] for x, y in approx]


def detect(models: dict, img: Image.Image) -> dict:
    """Persons with outlines, faces, and pose skeletons, in pixel coordinates of
    the upright image."""
    w, h = img.size
    tol = 0.002 * max(w, h)
    out = {"w": w, "h": h}

    r = models["person"].predict(img, imgsz=IMGSZ, conf=CONF["person"], classes=PERSON_CLASSES,
                                 verbose=False, retina_masks=False)[0]
    persons = []
    if r.boxes is not None:
        polys = r.masks.xy if r.masks is not None else [np.zeros((0, 2))] * len(r.boxes)
        for box, conf, poly in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), polys):
            persons.append({"box": [round(float(v), 1) for v in box], "conf": round(float(conf), 3),
                            "poly": simplify(np.asarray(poly), tol)})
    out["persons"] = persons

    r = models["face"].predict(img, imgsz=IMGSZ, conf=CONF["face"], verbose=False)[0]
    out["faces"] = [] if r.boxes is None else [
        {"box": [round(float(v), 1) for v in box], "conf": round(float(conf), 3)}
        for box, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]

    r = models["pose"].predict(img, imgsz=IMGSZ, conf=CONF["pose"], verbose=False)[0]
    poses = []
    if r.boxes is not None and r.keypoints is not None:
        kps = r.keypoints.data.cpu().numpy()                     # n x 17 x 3 (x, y, conf)
        for box, conf, kp in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), kps):
            poses.append({"box": [round(float(v), 1) for v in box], "conf": round(float(conf), 3),
                          "kp": [[round(float(x), 1), round(float(y), 1), round(float(c), 3)] for x, y, c in kp]})
    out["poses"] = poses
    return out


# ---------------------------------------------------------------------------
# People: one list from three detectors, and per-person measurements
# ---------------------------------------------------------------------------

KP = {n: i for i, n in enumerate(KP_NAMES)}
KP_OK = 0.5                   # keypoint confidence that counts as visible
MATCH_MIN = 0.35              # outline-to-skeleton match score needed to merge
FACE_MATCH_MAX = 0.8          # face centre to head keypoints, in face sizes
EDGE_MARGIN = 0.004           # a person within this share of the long side touches an edge
FOCUS_MIN_SIDE = 32           # smaller head regions give no focus value
FOCUS_SIDE = 64               # head regions are scaled to this before measuring
STANDING_MIN = 0.7            # hip-to-ankle height over leg length for "standing"

# Head size (sqrt of the face box area) from keypoint distances when no face
# was detected. Medians over calibration people with both a face box (>= 40 px)
# and the keypoints: eyes 2.50 (IQR 2.39-2.62, frontal only), ears 1.02
# (0.97-1.09, frontal only), shoulders 0.59 (0.52-0.81; a turned body looks
# narrower, so shoulders are the last resort).
HEAD_PER_EYE_DIST = 2.5
HEAD_PER_EAR_DIST = 1.0
HEAD_PER_SHOULDER = 0.6


def box_iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def poly_bbox(poly):
    a = np.asarray(poly, dtype=np.float32)
    return [float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())]


def poly_area(poly) -> float:
    a = np.asarray(poly, dtype=np.float64)
    if len(a) < 3:
        return 0.0
    x, y = a[:, 0], a[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2)


def kp_inside(kp, poly) -> float:
    """Share of the visible keypoints that lie inside the outline."""
    import cv2
    pts = [(x, y) for x, y, c in kp if c >= KP_OK]
    if not pts or len(poly) < 3:
        return 0.0
    contour = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    return sum(cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0 for x, y in pts) / len(pts)


def head_point(kp):
    """Mean of the visible head keypoints (nose, eyes, ears), or None."""
    pts = [kp[i] for i in range(5) if kp[i][2] >= KP_OK]
    if not pts:
        return None
    return float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))


def union_box(boxes):
    boxes = [b for b in boxes if b]
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def build_people(det: dict) -> list:
    """Merge outlines, skeletons and faces into one list of people.

    A skeleton anchors a person: an outline joins it if their boxes overlap and
    the skeleton's keypoints lie inside the outline, a face joins it if it sits
    on the skeleton's head keypoints. Outlines and faces that join no skeleton
    become people of their own: no single detector finds everybody (people in
    unusual clothing can have a skeleton and a face but no outline; small
    distant people often have an outline but no skeleton).
    """
    poses, outlines, faces = det["poses"], [o for o in det["persons"] if len(o["poly"]) >= 3], det["faces"]
    people = [{"kp": p["kp"], "pose_box": p["box"], "pose_conf": p["conf"]} for p in poses]

    # outlines to skeletons, best matches first
    cand = []
    for i, p in enumerate(poses):
        for j, o in enumerate(outlines):
            iou = box_iou(p["box"], o["box"])
            if iou > 0.1:
                cand.append((0.5 * iou + 0.5 * kp_inside(p["kp"], o["poly"]), i, j))
    used_p, used_o = set(), set()
    for score, i, j in sorted(cand, reverse=True):
        if score >= MATCH_MIN and i not in used_p and j not in used_o:
            used_p.add(i)
            used_o.add(j)
            people[i].update(poly=outlines[j]["poly"], seg_box=outlines[j]["box"], seg_conf=outlines[j]["conf"])
    for j, o in enumerate(outlines):
        if j not in used_o:
            people.append({"poly": o["poly"], "seg_box": o["box"], "seg_conf": o["conf"]})

    # faces to people: by head keypoints, else inside the upper part of the outline
    cand = []
    for f_idx, f in enumerate(faces):
        fx, fy = (f["box"][0] + f["box"][2]) / 2, (f["box"][1] + f["box"][3]) / 2
        fs = math.sqrt(max(1.0, (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1])))
        for i, pe in enumerate(people):
            hp = head_point(pe["kp"]) if "kp" in pe else None
            if hp is not None:
                d = math.hypot(fx - hp[0], fy - hp[1]) / fs
                if d <= FACE_MATCH_MAX:
                    cand.append((d, f_idx, i))
            elif "poly" in pe:
                b = pe["seg_box"]
                if b[0] <= fx <= b[2] and b[1] <= fy <= b[1] + 0.4 * (b[3] - b[1]) and \
                        kp_inside([[fx, fy, 1.0]], pe["poly"]) > 0:
                    cand.append((1.0, f_idx, i))      # weaker than any keypoint match
    used_f, used_pe = set(), set()
    for d, f_idx, i in sorted(cand):
        if f_idx not in used_f and i not in used_pe:
            used_f.add(f_idx)
            used_pe.add(i)
            people[i].update(face=faces[f_idx]["box"], face_conf=faces[f_idx]["conf"])
    for f_idx, f in enumerate(faces):
        if f_idx not in used_f:
            people.append({"face": f["box"], "face_conf": f["conf"]})

    for pe in people:
        pe["sources"] = [k for k, key in (("pose", "kp"), ("seg", "poly"), ("face", "face")) if key in pe]
        pe["conf"] = max(pe.get("pose_conf", 0), pe.get("seg_conf", 0), pe.get("face_conf", 0))
        pe["box"] = union_box([pe.get("seg_box"), pe.get("pose_box"), pe.get("face")])
    return people


def detail_ratio(gray: np.ndarray) -> float:
    """High-frequency energy over contrast, as in k2prep's D metric."""
    h, w = gray.shape
    small = Image.fromarray(gray).resize((max(1, w // 2), max(1, h // 2)), Image.BOX)
    back = np.asarray(small.resize((w, h), Image.BILINEAR), dtype=np.float32)
    a = gray.astype(np.float32)
    return float(np.abs(a - back).mean() / max(float(a.std()), 1e-6))


def measure_people(people: list, w: int, h: int, gray: np.ndarray | None) -> None:
    """Add the measurements that decide subject or passer-by, in place.

    head      head size in px: sqrt of the face box area, else from eye, ear or
              shoulder distance (head_src says which), else box height / 7
    top/foot  y of the top of the person and of the feet (ankles if visible)
    feet      the ankles are visible and not at the bottom edge
    edges     image edges the person touches: any of l, t, r, b
    visible   which body parts have visible keypoints
    facing    1 frontal, 0.8 three-quarter, 0.4 profile, 0 back; None unknown
    yaw       -1 looks towards the image left .. +1 towards the right; None unknown
    focus     detail ratio of the face/head region scaled to FOCUS_SIDE px, so
              it measures blur relative to the head size (what a viewer sees);
              None if the region is too small or no pixels were given
    posture   "standing", "bent" (sitting, kneeling, crouching) or None
    stride    horizontal ankle distance over leg length, standing people only
    centre    x and y of the box centre as a share of the image size
    area      outline area (or box area) as a share of the image area
    """
    margin = EDGE_MARGIN * max(w, h)
    for pe in people:
        kp = pe.get("kp")
        vis = (lambda name: kp is not None and kp[KP[name]][2] >= KP_OK)
        b = pe["box"]
        # head size
        head, src = None, None
        if "face" in pe:
            f = pe["face"]
            head, src = math.sqrt((f[2] - f[0]) * (f[3] - f[1])), "face"
        elif kp is not None:
            def dist(a, c):
                return math.hypot(kp[KP[a]][0] - kp[KP[c]][0], kp[KP[a]][1] - kp[KP[c]][1])
            if vis("l_eye") and vis("r_eye"):
                head, src = dist("l_eye", "r_eye") * HEAD_PER_EYE_DIST, "eyes"
            elif vis("l_ear") and vis("r_ear"):
                head, src = dist("l_ear", "r_ear") * HEAD_PER_EAR_DIST, "ears"
            elif vis("l_shoulder") and vis("r_shoulder"):
                head, src = dist("l_shoulder", "r_shoulder") * HEAD_PER_SHOULDER, "shoulders"
        if head is None:
            head, src = (b[3] - b[1]) / 7.0, "box"
        pe["head"], pe["head_src"] = round(head, 1), src

        ankles = [kp[KP[a]][1] for a in ("l_ankle", "r_ankle") if vis(a)] if kp is not None else []
        pe["top"] = round(b[1], 1)
        pe["foot"] = round(max(ankles) if ankles else b[3], 1)
        pe["feet"] = bool(ankles) and b[3] < h - margin
        pe["edges"] = "".join(e for e, hit in (("l", b[0] <= margin), ("t", b[1] <= margin),
                                               ("r", b[2] >= w - margin), ("b", b[3] >= h - margin)) if hit)
        pe["visible"] = "".join(c for c, parts in (("h", ("nose", "l_eye", "r_eye", "l_ear", "r_ear")),
                                                   ("s", ("l_shoulder", "r_shoulder")), ("p", ("l_hip", "r_hip")),
                                                   ("k", ("l_knee", "r_knee")), ("a", ("l_ankle", "r_ankle")))
                                if any(vis(n) for n in parts))

        # facing and looking direction
        facing, yaw = None, None
        if kp is not None:
            n, le, re = vis("nose"), vis("l_eye"), vis("r_eye")
            la, ra = vis("l_ear"), vis("r_ear")
            if n and le and re:
                facing = 1.0 if la == ra else 0.8
            elif n and (le or re):
                facing = 0.4
            elif not n and not le and not re and (la or ra or vis("l_shoulder") or vis("r_shoulder")):
                facing = 0.0
            if n and le and re:
                mid = (kp[KP["l_eye"]][0] + kp[KP["r_eye"]][0]) / 2
                ed = abs(kp[KP["l_eye"]][0] - kp[KP["r_eye"]][0])
                yaw = max(-1.0, min(1.0, (kp[KP["nose"]][0] - mid) / max(ed, 1.0)))
            elif n and (le or re):
                eye = kp[KP["l_eye"]] if le else kp[KP["r_eye"]]
                yaw = 1.0 if kp[KP["nose"]][0] > eye[0] else -1.0
        elif "face" in pe:
            facing = 0.8
        pe["facing"], pe["yaw"] = facing, None if yaw is None else round(yaw, 2)

        # focus in the face or head region, at full resolution
        region = None
        if "face" in pe:
            region = pe["face"]
        elif kp is not None and head_point(kp) is not None:
            hx, hy = head_point(kp)
            region = [hx - head / 2, hy - head / 2, hx + head / 2, hy + head / 2]
        else:
            region = [b[0], b[1], b[2], b[1] + (b[3] - b[1]) / 7]
        focus = None
        if gray is not None:
            x0, y0 = max(0, int(region[0])), max(0, int(region[1]))
            x1, y1 = min(w, int(region[2])), min(h, int(region[3]))
            if x1 - x0 >= FOCUS_MIN_SIDE and y1 - y0 >= FOCUS_MIN_SIDE:
                crop = Image.fromarray(gray[y0:y1, x0:x1])
                if max(crop.size) > FOCUS_SIDE:
                    crop = crop.resize((FOCUS_SIDE, FOCUS_SIDE), Image.BOX)
                focus = round(detail_ratio(np.asarray(crop)), 4)
        pe["focus"] = focus

        # posture from the leg keypoints; stride only means walking when standing
        posture, stride = None, None
        sides = [sd for sd in ("l", "r") if vis(f"{sd}_hip") and vis(f"{sd}_knee") and vis(f"{sd}_ankle")]
        if sides:
            ratios = []
            for sd in sides:
                hip, knee, ank = kp[KP[f"{sd}_hip"]], kp[KP[f"{sd}_knee"]], kp[KP[f"{sd}_ankle"]]
                leg = math.hypot(hip[0] - knee[0], hip[1] - knee[1]) + math.hypot(knee[0] - ank[0], knee[1] - ank[1])
                ratios.append((ank[1] - hip[1]) / max(leg, 1.0))
            posture = "standing" if min(ratios) >= STANDING_MIN else "bent"
        if posture == "standing" and len(sides) == 2:
            legs = [math.hypot(kp[KP[f"{sd}_hip"]][0] - kp[KP[f"{sd}_ankle"]][0],
                               kp[KP[f"{sd}_hip"]][1] - kp[KP[f"{sd}_ankle"]][1]) for sd in ("l", "r")]
            stride = round(abs(kp[KP["l_ankle"]][0] - kp[KP["r_ankle"]][0]) / max(1.0, sum(legs) / 2), 3)
        pe["posture"], pe["stride"] = posture, stride

        pe["centre"] = [round((b[0] + b[2]) / 2 / w, 4), round((b[1] + b[3]) / 2 / h, 4)]
        area = poly_area(pe["poly"]) if "poly" in pe else (b[2] - b[0]) * (b[3] - b[1])
        pe["area"] = round(area / (w * h), 5)


def people_for(det: dict, img: Image.Image | None) -> list:
    people = build_people(det)
    gray = np.asarray(img.convert("L")) if img is not None else None
    measure_people(people, det["w"], det["h"], gray)
    # keep the cache small: the outline stays, the merged inputs go
    for pe in people:
        for k in ("pose_box", "seg_box"):
            pe.pop(k, None)
    return people


# ---------------------------------------------------------------------------
# Previews
# ---------------------------------------------------------------------------

PREVIEW_SIDE = 1600


def draw_preview(img: Image.Image, det: dict, dest: Path) -> None:
    s = PREVIEW_SIDE / max(img.size)
    prev = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
    over = Image.new("RGBA", prev.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    lw = max(2, round(PREVIEW_SIDE / 500))
    for k, p in enumerate(det["persons"]):
        if len(p["poly"]) >= 3:
            d.polygon([(x * s, y * s) for x, y in p["poly"]], fill=(0, 200, 255, 60), outline=(0, 200, 255, 255),
                      width=lw)
        x0, y0 = p["box"][0] * s, p["box"][1] * s
        d.rectangle((x0, y0, x0 + 64, y0 + 22), fill=(0, 120, 200, 220))
        d.text((x0 + 4, y0 + 4), f"P{k} {p['conf']:.2f}", fill=(255, 255, 255, 255))
    for f in det["faces"]:
        b = [v * s for v in f["box"]]
        d.rectangle(b, outline=(255, 220, 0, 255), width=lw)
    for p in det["poses"]:
        kp = p["kp"]
        for a, b in SKELETON:
            if kp[a][2] > 0.3 and kp[b][2] > 0.3:
                d.line((kp[a][0] * s, kp[a][1] * s, kp[b][0] * s, kp[b][1] * s), fill=(255, 60, 160, 255), width=lw)
        for x, y, c in kp:
            if c > 0.3:
                d.ellipse((x * s - lw * 1.5, y * s - lw * 1.5, x * s + lw * 1.5, y * s + lw * 1.5),
                          fill=(255, 60, 160, 255))
    prev = Image.alpha_composite(prev.convert("RGBA"), over).convert("RGB")
    dest.parent.mkdir(parents=True, exist_ok=True)
    prev.save(dest, quality=85)


def draw_people(img: Image.Image, people: list, dest: Path) -> None:
    """Merged people with their numbers and measurements, for checking phase 2."""
    s = PREVIEW_SIDE / max(img.size)
    prev = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
    over = Image.new("RGBA", prev.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    lw = max(2, round(PREVIEW_SIDE / 500))
    colours = {3: (0, 220, 120), 2: (0, 170, 255), 1: (255, 140, 0)}   # by number of sources
    for k, pe in enumerate(people):
        col = colours.get(len(pe["sources"]), (255, 255, 255))
        if "poly" in pe:
            d.polygon([(x * s, y * s) for x, y in pe["poly"]], fill=col + (45,), outline=col + (255,), width=lw)
        b = [v * s for v in pe["box"]]
        d.rectangle(b, outline=col + (255,), width=1)
        if "face" in pe:
            d.rectangle([v * s for v in pe["face"]], outline=(255, 220, 0, 255), width=lw)
        lines = [f"#{k} {'+'.join(pe['sources'])} {pe['conf']:.2f}",
                 f"head {pe['head']:.0f} {pe['head_src']}  vis {pe['visible'] or '-'}",
                 f"face {pe['facing'] if pe['facing'] is not None else '-'} yaw {pe['yaw'] if pe['yaw'] is not None else '-'}",
                 f"focus {pe['focus'] if pe['focus'] is not None else '-'} {pe['posture'] or '-'} "
                 f"stride {pe['stride'] if pe['stride'] is not None else '-'}",
                 f"edges {pe['edges'] or '-'} feet {'y' if pe['feet'] else 'n'}"]
        tx, ty = b[0] + 2, b[1] + 2
        d.rectangle((tx - 2, ty - 2, tx + 170, ty + 12 * len(lines) + 2), fill=(0, 0, 0, 170))
        for i, line in enumerate(lines):
            d.text((tx, ty + 12 * i), line, fill=col + (255,))
    prev = Image.alpha_composite(prev.convert("RGBA"), over).convert("RGB")
    dest.parent.mkdir(parents=True, exist_ok=True)
    prev.save(dest, quality=85)


# ---------------------------------------------------------------------------
# Scanning and cache
# ---------------------------------------------------------------------------

def scan(root: Path):
    """Images under root; folders whose names start with "_" are skipped."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("_"))
        for fn in sorted(filenames):
            if os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
                out.append(Path(dirpath) / fn)
    return out


def models_signature() -> dict:
    return {name: f"{src}:{fname}" for name, (src, fname) in MODELS.items()} | {"imgsz": IMGSZ, "conf": CONF}


def load_cache(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") == CACHE_VERSION and data.get("models") == models_signature():
            return data.get("files", {})
    except (OSError, ValueError):
        pass
    return {}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


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
    ap.add_argument("folders", nargs="*", help="folders to process recursively")
    ap.add_argument("--out", help="output folder instead of <folder>/_reframed (one folder only)")
    ap.add_argument("--previews", action="store_true", help="draw the detections into <out>/_preview")
    ap.add_argument("--people", action="store_true",
                    help="draw the merged people and their measurements into <out>/_people")
    ap.add_argument("--redetect", action="store_true", help="ignore the detection cache")
    ap.add_argument("--fetch-models", action="store_true", help="download the models and load them once")
    args = ap.parse_args(argv)

    if args.fetch_models:
        models = load_models(download=True)
        for name in MODELS:
            print(f"{name}: {model_path(name, False)}")
        print(f"{len(models)} models load OK")
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

    models = None
    for root in roots:
        out_dir = Path(args.out).resolve() if args.out else root / OUT_DIRNAME
        cache_path = out_dir / DETECTIONS_NAME
        cache = {} if args.redetect else load_cache(cache_path)
        files = scan(root)
        print(f"{root}: {len(files)} image(s)")
        t0, n_new = time.time(), 0
        results = {}
        for k, path in enumerate(files, 1):
            rel = path.relative_to(root).as_posix()
            st = path.stat()
            c = cache.get(rel)
            fresh = c is None or c.get("size") != st.st_size or c.get("mtime_ns") != st.st_mtime_ns
            img = None
            if fresh:
                if models is None:
                    models = load_models()
                img = load_image(path)
                det = detect(models, img)
                det.update(size=st.st_size, mtime_ns=st.st_mtime_ns)
                n_new += 1
            else:
                det = c
            if det.get("people_version") != PEOPLE_VERSION:
                img = img or load_image(path)                # focus needs the pixels
                det["people"] = people_for(det, img)
                det["people_version"] = PEOPLE_VERSION
            results[rel] = det
            if args.previews:
                img = img or load_image(path)
                draw_preview(img, det, out_dir / PREVIEW_DIRNAME / (rel + ".jpg"))
            if args.people:
                img = img or load_image(path)
                draw_people(img, det["people"], out_dir / PEOPLE_PREVIEW_DIRNAME / (rel + ".jpg"))
            srcs = {}
            for pe in det["people"]:
                key = "+".join(pe["sources"])
                srcs[key] = srcs.get(key, 0) + 1
            print(f"[{k}/{len(files)}] {rel}: {len(det['people'])} people ("
                  + ", ".join(f"{v} {k2}" for k2, v in sorted(srcs.items())) + ")"
                  + ("" if fresh else " (cached detections)"), flush=True)
        write_json(cache_path, {"version": CACHE_VERSION, "models": models_signature(),
                                "written": datetime.now().isoformat(timespec="seconds"), "files": results})
        print(f"{n_new} detected, {len(files) - n_new} from cache, {time.time() - t0:.0f}s -> {cache_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
