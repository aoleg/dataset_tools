#!/usr/bin/env python3
"""
Reframe photos of people: trim wasted space around the subject (one person or a
group), leave passers-by out, and crop to one of the k2prep aspect ratios.

Phases 1-5 (this version): detection, per-person measurements, subject
selection and crop planning. Every image is analysed with three models; the detections are
merged into one list of people, each person is measured, and the subject group
is chosen, or the photo is marked to stay unchanged (crowd, no clear subject).
The crop holds the subject with margins, in one of the k2prep aspect ratios.
JPEG crops are planned for a lossless crop (whole DCT blocks, no re-encoding)
unless --reencode is given. Results go to <out>/plan.json. --previews draws the
detections, --people the measurements, --verdicts the subject choice and the
planned crop.
Per-photo corrections: <folder>/overrides.txt, see OVERRIDES_NAME.
  person outlines : yolo26x-seg.pt   (ultralytics assets, COCO class "person")
  faces           : face_yolov8m.pt  (Bingsu/adetailer, Hugging Face)
  pose keypoints  : yolo26x-pose.pt  (ultralytics assets)

Usage:    python reframe.py <folder> [<folder> ...] [--out DIR] [--previews]
          python reframe.py --fetch-models
"""
import argparse
import json
import math
from functools import lru_cache
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
PEOPLE_VERSION = 4           # bump when build_people or measure_people changes
PEOPLE_PREVIEW_DIRNAME = "_people"
VERDICT_PREVIEW_DIRNAME = "_verdicts"
PLAN_NAME = "plan.json"
# One line per photo, path relative to the folder, then an action:
#   photo.jpg keep          leave the photo unchanged
#   photo.jpg crop          crop to the chosen group even if the rules say no
#   photo.jpg people=0,3,5  the subject is exactly these people (numbers from
#                           the --verdicts previews)
# Lines starting with # are comments. A path with spaces goes in quotes.
OVERRIDES_NAME = "overrides.txt"

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
# An outline joins a skeleton if their boxes overlap by MATCH_IOU or most
# keypoints (MATCH_KP) lie inside the outline. Not both: the outline model cuts
# lower legs and feet off, so a correct outline can miss the ankle keypoints.
MATCH_IOU = 0.45
MATCH_KP = 0.6
DUPLICATE_IOU = 0.7           # a leftover outline this close to a person is the same person
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
                inside = kp_inside(p["kp"], o["poly"])
                if iou >= MATCH_IOU or inside >= MATCH_KP:
                    cand.append((iou + inside, i, j))
    used_p, used_o = set(), set()
    for score, i, j in sorted(cand, reverse=True):
        if i not in used_p and j not in used_o:
            used_p.add(i)
            used_o.add(j)
            people[i].update(poly=outlines[j]["poly"], seg_box=outlines[j]["box"], seg_conf=outlines[j]["conf"])
    for j, o in enumerate(outlines):
        if j in used_o:
            continue
        if any(box_iou(o["box"], pe.get("seg_box") or pe["pose_box"]) >= DUPLICATE_IOU for pe in people
               if "pose_box" in pe or "seg_box" in pe):
            continue                                  # the same person, detected twice
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
# Subject selection
# ---------------------------------------------------------------------------
#
# Calibrated on 29 personal photos labelled by eye (27 decidable): all 27
# agree. Every threshold below can move by 20% with at most two photos
# changing, except MIN_SCORE upward and MIN_HEAD_SHARE.

CAND_MIN_REL = 0.40           # candidates: head at least this share of the largest head
FACE_BONUS = {1.0: 0.30, 0.8: 0.25, 0.4: 0.05, 0.0: -0.35}   # by facing
SIDE_EDGE_PENALTY = 0.35      # touching the left or right image edge
SIDE_EDGE_KEEP_REL = 0.75     # a side-edge person stays a candidate only if this large ...
SIDE_EDGE_KEEP_FACING = 0.8   # ... and facing the camera at least this much
CENTRE_FREE = 0.22            # no position penalty within this distance of the middle
CENTRE_WEIGHT = 1.0
GROUP_HEAD_RATIO = 1.8        # same depth: heads within this factor
GROUP_FOOT_TOL = 0.12         # same depth: feet within this share of the image height
GROUP_GAP_HEADS = 3.0         # close together: box gap at most this many head sizes
ABSORB_OVERLAP = 0.5          # an outline-only person this much inside a member joins
MIN_SCORE = 1.0               # the best group must reach this score
MIN_HEAD_SHARE = 0.034        # its largest head over the image's short side
CROWD_PEOPLE = 10             # this many candidates ...
CROWD_DOMINANCE = 2.0         # ... and no head this much larger than the rest = crowd
SINGLE_CENTRE = 0.2           # a lone subject further off-centre is a passer-by
TIE_MARGIN = 0.1              # a second group this close in score is kept too, and flagged


def box_gap(a, b) -> float:
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy)


def share_inside(a, b) -> float:
    """Share of box a that lies inside box b."""
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / max(1.0, (a[2] - a[0]) * (a[3] - a[1]))


def touches_side(pe) -> bool:
    return "l" in pe["edges"] or "r" in pe["edges"]


def person_score(pe, maxhead) -> float:
    """Head size relative to the largest head, plus facing, minus edge and
    off-centre penalties. Size alone does not decide: a passer-by close to the
    camera has the largest head but is turned away or cut by the frame."""
    s = pe["head"] / maxhead + FACE_BONUS.get(pe["facing"], 0.0)
    if touches_side(pe):
        s -= SIDE_EDGE_PENALTY
    s -= CENTRE_WEIGHT * max(0.0, abs(pe["centre"][0] - 0.5) - CENTRE_FREE)
    return s


def is_candidate(pe, maxhead) -> bool:
    rel = pe["head"] / maxhead
    if rel < CAND_MIN_REL:
        return False
    if pe["facing"] == 0.0 and pe["edges"]:
        return False             # turned away and cut by the frame: a passer-by, or a shadow
    if touches_side(pe) and (rel < SIDE_EDGE_KEEP_REL or (pe["facing"] or 0.0) < SIDE_EDGE_KEEP_FACING):
        return False
    return True


def same_group(a, b, h) -> bool:
    """Same depth (head size, foot line) and close together, or overlapping."""
    if max(a["head"], b["head"]) / min(a["head"], b["head"]) > GROUP_HEAD_RATIO:
        return False
    g = box_gap(a["box"], b["box"])
    if g == 0:
        return True
    if a["feet"] and b["feet"] and abs(a["foot"] - b["foot"]) > GROUP_FOOT_TOL * h:
        return False
    return g <= GROUP_GAP_HEADS * (a["head"] + b["head"]) / 2


def absorb(people, members) -> list:
    """Add outline-only people that lie mostly inside a member: duplicates of a
    member, or someone partly hidden behind one. Cropping through them would
    cut a person."""
    out = set(members)
    for k, pe in enumerate(people):
        if k not in out and "kp" not in pe and "face" not in pe and \
                any(share_inside(pe["box"], people[m]["box"]) >= ABSORB_OVERLAP for m in members):
            out.add(k)
    return sorted(out)


def select_subject(det: dict, override: tuple | None = None) -> dict:
    """-> {"verdict": "crop" | "keep", "members": [...], "reason": str, "flags": [...]}

    "keep" leaves the photo unchanged. members are indexes into det["people"];
    for "keep" they are the best group, for the previews.
    """
    people, w, h = det["people"], det["w"], det["h"]
    flags = []
    if override and override[0] == "people":
        ids = [k for k in override[1] if 0 <= k < len(people)]
        if len(ids) != len(override[1]):
            flags.append("override names people that do not exist")
        if not ids:
            return {"verdict": "keep", "members": [], "reason": "override: no valid people", "flags": flags}
        return {"verdict": "crop", "members": absorb(people, ids), "reason": "override: people", "flags": flags}

    def result(verdict, members, reason):
        if override and override[0] == "keep":
            verdict, reason = "keep", f"override: keep ({reason})"
        elif override and override[0] == "crop" and members:
            verdict, reason = "crop", f"override: crop ({reason})"
        return {"verdict": verdict, "members": members, "reason": reason, "flags": flags}

    real = [k for k, pe in enumerate(people) if "kp" in pe or "face" in pe]   # people with a head
    if not real:
        return result("keep", [], "no people")
    maxhead = max(people[k]["head"] for k in real)
    score = {k: person_score(people[k], maxhead) for k in real}
    cand = [k for k in real if is_candidate(people[k], maxhead)]
    if not cand:
        return result("keep", [], "no subject candidates")

    parent = {k: k for k in cand}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i in cand:
        for j in cand:
            if i < j and same_group(people[i], people[j], h):
                parent[find(i)] = find(j)
    groups = {}
    for k in cand:
        groups.setdefault(find(k), []).append(k)
    ranked = sorted(groups.values(), key=lambda g: -max(score[k] for k in g))
    best = list(ranked[0])
    best_score = max(score[k] for k in best)
    tied = [g for g in ranked[1:] if max(score[k] for k in g) >= best_score - TIE_MARGIN]
    for g in tied:
        best += g
    if tied:
        flags.append(f"tie: {len(tied) + 1} groups score the same, all kept")
    head_in = max(people[k]["head"] for k in best)
    head_out = max((people[k]["head"] for k in real if k not in best), default=0.0)
    members = absorb(people, best)
    n_cand = sum(1 for k in real if people[k]["head"] / maxhead >= CAND_MIN_REL)

    if best_score < MIN_SCORE:
        return result("keep", members, f"no clear subject (score {best_score:.2f})")
    if head_in / min(w, h) < MIN_HEAD_SHARE:
        return result("keep", members, f"people too small (head {head_in / min(w, h):.3f} of the short side)")
    if n_cand >= CROWD_PEOPLE and (not head_out or head_in / head_out < CROWD_DOMINANCE):
        return result("keep", members, f"crowd ({n_cand} similar people)")
    if len(best) == 1 and abs(people[best[0]]["centre"][0] - 0.5) > SINGLE_CENTRE:
        return result("keep", members, "lone person off-centre")
    return result("crop", members, f"subject: {len(best)} person(s), score {best_score:.2f}")


def read_overrides(root: Path) -> dict:
    """{relative path (lower case, forward slashes): (action, [people])}"""
    import shlex
    path = root / OVERRIDES_NAME
    out = {}
    if not path.is_file():
        return out
    for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            parts = shlex.split(line, posix=True)
        except ValueError:
            parts = []
        if len(parts) != 2:
            print(f"  {OVERRIDES_NAME} line {n} ignored: {line}")
            continue
        rel, action = parts[0].replace("\\", "/").lower(), parts[1].lower()
        if action in ("keep", "crop"):
            out[rel] = (action, [])
        elif action.startswith("people="):
            try:
                out[rel] = ("people", [int(x) for x in action[7:].split(",") if x.strip()])
            except ValueError:
                print(f"  {OVERRIDES_NAME} line {n} ignored: {line}")
        else:
            print(f"  {OVERRIDES_NAME} line {n} ignored: {line}")
    return out


# ---------------------------------------------------------------------------
# Crop planning
# ---------------------------------------------------------------------------

# Aspect-ratio families and buckets, copied from k2prep.py
# (T:/claude/github2/k2prep), which ports musubi-tuner's BucketSelector.
AR_FAMILIES = ["9:16", "2:3", "4:5", "1:1", "5:4", "3:2", "16:9"]
AR_NOMINAL = [0.5647, 0.6667, 0.8028, 1.0000, 1.2456, 1.5000, 1.7708]
UPSCALE_TOLERANCE = 1.15
RESO_STEPS = 16

# Margins around the subject group, in head sizes of its largest head.
HEADROOM = 0.7                # above the top of the highest member
SIDE_MARGIN = 0.5             # left and right
FOOT_MARGIN = 0.3             # below the feet, when the feet are visible
LOOK_ROOM = 0.6               # extra room on the side a lone subject looks to (times |yaw|)
HEAD_LINE = 1 / 3             # where the highest head goes, as a share of the crop height
MIN_SAVING = 0.10             # leave the photo unchanged if the crop keeps more than 90%
FIT_MIN_COVER = 0.98          # without margins, a ratio must keep this share of the group
CUT_MIN, CUT_MAX = 0.03, 0.97  # a person with this share of the box inside the crop is cut


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


def image_header(path: Path) -> dict:
    """Format, EXIF orientation, stored size and JPEG MCU size, from the header only."""
    with Image.open(path) as im:
        try:
            orientation = im.getexif().get(274, 1) or 1
        except Exception:  # noqa: BLE001
            orientation = 1
        head = {"format": im.format or "", "orientation": orientation if orientation in range(1, 9) else 1,
                "stored": list(im.size)}
        layer = getattr(im, "layer", None)
        if im.format == "JPEG" and layer:
            head["mcu"] = [8 * max(c[1] for c in layer), 8 * max(c[2] for c in layer)]
            head["components"] = len(layer)
    return head


def display_to_stored(box, orientation, sw, sh):
    """A box in upright (display) pixels -> the same box in stored pixels.

    Edges are continuous coordinates: a box [x0, x1) of display pixels maps to
    the stored pixels that EXIF orientation puts there.
    """
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


def stored_to_display(box, orientation, sw, sh):
    """Inverse of display_to_stored."""
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


def align_lossless(box, head):
    """Move a display crop so that its stored origin sits on the MCU grid.

    The origin moves up and left in stored pixels (the crop only grows, so the
    subject stays inside); the far edge of the shorter-grown side then extends
    to restore the aspect ratio, clipped to the image. -> (display box, stored box)
    """
    o = head["orientation"]
    sw, sh = head["stored"]
    mw, mh = head["mcu"]
    rot = o in (5, 6, 7, 8)
    ratio = (box[2] - box[0]) / (box[3] - box[1])
    sratio = 1 / ratio if rot else ratio                 # width / height in stored pixels
    x0, y0, x1, y1 = display_to_stored(box, o, sw, sh)
    x0, y0 = (x0 // mw) * mw, (y0 // mh) * mh
    w, h = x1 - x0, y1 - y0
    if w / h < sratio:
        w = min(sw - x0, round(h * sratio))
        h = min(sh - y0, round(w / sratio))
    else:
        h = min(sh - y0, round(w / sratio))
        w = min(sw - x0, round(h * sratio))
    stored = [int(x0), int(y0), int(x0 + w), int(y0 + h)]
    return [int(v) for v in stored_to_display(stored, o, sw, sh)], stored


def subject_area(people, members, w, h):
    """The group's box plus margins, and the box without margins, both clipped."""
    heads = [people[k]["head"] for k in members if "kp" in people[k] or "face" in people[k]] or \
        [people[k]["head"] for k in members]
    hs = max(heads)
    bare = union_box([people[k]["box"] for k in members])
    x0, y0, x1, y1 = bare[0] - SIDE_MARGIN * hs, bare[1] - HEADROOM * hs, bare[2] + SIDE_MARGIN * hs, bare[3]
    bottom = 0.0
    for k in members:
        pe = people[k]
        if pe["feet"]:
            bottom = max(bottom, pe["foot"] + FOOT_MARGIN * hs, pe["box"][3])
        elif "kp" in pe and not pe["box"][3] >= h - EDGE_MARGIN * max(w, h):
            # a skeleton without visible feet that stops above the bottom edge:
            # the body goes on below the detection (hidden legs, long clothing)
            bottom = float(h)
        else:
            bottom = max(bottom, pe["box"][3] + FOOT_MARGIN * hs)
    y1 = max(y1, bottom)
    real = [k for k in members if "kp" in people[k] or "face" in people[k]]
    if len(real) == 1 and people[real[0]]["yaw"] is not None:
        yaw = people[real[0]]["yaw"]
        if yaw > 0:
            x1 += LOOK_ROOM * hs * yaw
        else:
            x0 += LOOK_ROOM * hs * yaw
    clip = (lambda b: [max(0.0, b[0]), max(0.0, b[1]), min(float(w), b[2]), min(float(h), b[3])])
    return clip([x0, y0, x1, y1]), clip(bare), hs


def head_line_y(people, members):
    ys = []
    for k in members:
        pe = people[k]
        if "face" in pe:
            ys.append((pe["face"][1] + pe["face"][3]) / 2)
        elif "kp" in pe and head_point(pe["kp"]) is not None:
            ys.append(head_point(pe["kp"])[1])
    return min(ys) if ys else None


def cut_cost(crop, others):
    cost = 0.0
    for pe in others:
        share = share_inside(pe["box"], crop)
        if CUT_MIN < share < CUT_MAX:
            cost += pe["area"]
    return cost


def place_axis(pref, lo, hi, size, others, crop_of):
    """Best origin on one axis: fewest cut passers-by (by area), then nearest
    the preferred origin. crop_of(origin) builds the crop box."""
    cands = {min(max(pref, lo), hi), lo, hi}
    for pe in others:
        b = pe["box"]
        a0, a1 = (b[0], b[2]) if crop_of(0)[1] == crop_of(1)[1] else (b[1], b[3])
        for c in (a0 - size, a1, a0, a1 - size):
            cands.add(min(max(c, lo), hi))
    return min(cands, key=lambda c: (round(cut_cost(crop_of(c), others), 6), abs(c - pref)))


def plan_crop(det: dict, sel: dict, head: dict, families, lossless: bool) -> dict:
    """-> {"verdict": "crop", "box": display box, "family", "flags": [...], ...}
    or {"verdict": "keep", "reason": ...}"""
    people, w, h = det["people"], det["w"], det["h"]
    members = sel["members"]
    area, bare, hs = subject_area(people, members, w, h)
    flags = []
    aw, ah = area[2] - area[0], area[3] - area[1]

    # 1. the tightest ratio that holds the group with its margins
    best = None
    for fam in families:
        r = AR_NOMINAL[AR_FAMILIES.index(fam)]
        cw, ch = (ah * r, ah) if aw / ah < r else (aw, aw / r)
        if cw <= w + 0.5 and ch <= h + 0.5:
            key = (cw * ch, abs(math.log(r) - math.log(aw / ah)))
            if best is None or key < best[0]:
                best = (key, fam, min(cw, w), min(ch, h))
    if best is None:
        # 2. no ratio holds the margins: the largest crop of each ratio, placed
        # over the group; keep it if it holds almost all of the group itself
        opts = []
        for fam in families:
            r = AR_NOMINAL[AR_FAMILIES.index(fam)]
            cw, ch = (h * r, h) if w / h > r else (w, w / r)
            x0 = min(max((bare[0] + bare[2]) / 2 - cw / 2, 0), w - cw)
            y0 = min(max((bare[1] + bare[3]) / 2 - ch / 2, 0), h - ch)
            cover = share_inside(bare, [x0, y0, x0 + cw, y0 + ch])
            opts.append((cover, -cw * ch, fam, cw, ch))
        cover, _, fam, cw, ch = max(opts)
        if cover < FIT_MIN_COVER:
            return {"verdict": "keep", "reason": f"the group does not fit any ratio (best {fam} keeps {cover:.0%})"}
        flags.append("tight: no room for margins" if cover >= 0.999 else f"cuts {1 - cover:.1%} of the group")
        best = (None, fam, cw, ch)
    _, fam, cw, ch = best
    r = AR_NOMINAL[AR_FAMILIES.index(fam)]

    # 3. size floor: never smaller than k2prep needs for its 512 bucket
    bkt = bucket_for(512, fam)
    need = bkt[0] * bkt[1] / UPSCALE_TOLERANCE ** 2
    if cw * ch < need:
        g = math.sqrt(need / (cw * ch))
        cw, ch = cw * g, ch * g
        if cw > w + 0.5 or ch > h + 0.5:
            return {"verdict": "keep", "reason": f"too small for a 512 crop at {fam}"}
        flags.append("grown to the 512 size floor")

    # 4. placement: the group inside, the top head near the upper third,
    # centred sideways; then move to avoid cutting passers-by in half
    lo_x, hi_x = max(0.0, area[2] - cw), min(area[0], w - cw)
    lo_y, hi_y = max(0.0, area[3] - ch), min(area[1], h - ch)
    if lo_x > hi_x:                                  # crop narrower than the margins (case 2)
        lo_x, hi_x = max(0.0, bare[2] - cw), min(bare[0], w - cw)
        if lo_x > hi_x:
            lo_x = hi_x = min(max((bare[0] + bare[2]) / 2 - cw / 2, 0.0), w - cw)
    if lo_y > hi_y:
        lo_y, hi_y = max(0.0, bare[3] - ch), min(bare[1], h - ch)
        if lo_y > hi_y:
            lo_y = hi_y = min(max((bare[1] + bare[3]) / 2 - ch / 2, 0.0), h - ch)
    pref_x = (area[0] + area[2]) / 2 - cw / 2
    hy = head_line_y(people, members)
    pref_y = hy - HEAD_LINE * ch if hy is not None else (area[1] + area[3]) / 2 - ch / 2
    others = [pe for k, pe in enumerate(people) if k not in set(members) and pe["area"] >= 0.0008]
    x0 = min(max(pref_x, lo_x), hi_x)
    y0 = min(max(pref_y, lo_y), hi_y)
    for _ in range(2):
        x0 = place_axis(pref_x, lo_x, hi_x, cw, others, lambda c: [c, y0, c + cw, y0 + ch])
        y0 = place_axis(pref_y, lo_y, hi_y, ch, others, lambda c: [x0, c, x0 + cw, c + ch])
    box = [x0, y0, x0 + cw, y0 + ch]
    if cut_cost(box, others) > 0:
        flags.append("cuts a passer-by")

    # 5. integer pixels; lossless JPEG alignment
    box = [int(math.floor(box[0])), int(math.floor(box[1])), int(math.ceil(box[2])), int(math.ceil(box[3]))]
    box = [max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])]
    out = {"verdict": "crop", "family": fam, "flags": flags}
    if lossless and head.get("mcu"):
        box, stored = align_lossless(box, head)
        out["stored_box"] = stored
        out["lossless"] = True
    else:
        out["lossless"] = False
    out["box"] = box
    bw, bh = box[2] - box[0], box[3] - box[1]
    out["size"] = [bw, bh]
    out["ratio_error"] = round(bw / bh / r - 1, 4)
    out["keeps"] = round(bw * bh / (w * h), 3)
    if bw * bh > (1 - MIN_SAVING) * w * h:
        return {"verdict": "keep", "reason": f"the crop keeps {bw * bh / (w * h):.0%} of the photo"}
    return out


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


def draw_verdict(img: Image.Image, det: dict, sel: dict, dest: Path, plan: dict | None = None) -> None:
    """The subject choice and the planned crop: members green, passers-by red,
    the area outside the crop darkened, every person numbered for overrides.txt."""
    from PIL import ImageFont
    people = det["people"]
    s = PREVIEW_SIDE / max(img.size)
    prev = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS).convert("RGBA")
    try:
        font = ImageFont.truetype("arialbd.ttf", 20)
        big = ImageFont.truetype("arialbd.ttf", 30)
    except OSError:
        font = big = ImageFont.load_default()
    cropping = plan is not None and plan.get("verdict") == "crop"
    if cropping:
        c = [v * s for v in plan["box"]]
        shade = Image.new("RGBA", prev.size, (0, 0, 0, 130))
        ImageDraw.Draw(shade).rectangle(c, fill=(0, 0, 0, 0))
        prev = Image.alpha_composite(prev, shade)
    over = Image.new("RGBA", prev.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    members = set(sel["members"])
    for k, pe in enumerate(people):
        b = [v * s for v in pe["box"]]
        col = (0, 230, 90) if k in members and cropping else (150, 150, 150) if k in members else (255, 50, 50)
        d.rectangle(b, outline=col + (230,), width=3 if k in members else 2)
        label = str(k)
        tw = d.textlength(label, font=font)
        d.rectangle((b[0], b[1], b[0] + tw + 8, b[1] + 24), fill=col + (220,))
        d.text((b[0] + 4, b[1] + 1), label, font=font, fill=(0, 0, 0, 255))
    if cropping:
        d.rectangle(c, outline=(255, 255, 0, 255), width=4)
        d.text((c[0] + 8, c[3] - 30), f"{plan['family']}  {plan['size'][0]}x{plan['size'][1]}"
               + ("  lossless" if plan.get("lossless") else ""), font=font, fill=(255, 255, 0, 255))
        title = "CROP  " + sel["reason"]
    else:
        title = "UNCHANGED  " + (plan["reason"] if plan is not None else sel["reason"])
    flags = sel["flags"] + ((plan or {}).get("flags") or [])
    if flags:
        title += "  [" + "; ".join(flags) + "]"
    d.rectangle((0, 0, prev.width, 42), fill=(0, 0, 0, 180))
    d.text((8, 5), title, font=big, fill=(0, 230, 90, 255) if cropping else (255, 200, 0, 255))
    out = Image.alpha_composite(prev, over).convert("RGB")
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.save(dest, quality=85)


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
    ap.add_argument("--verdicts", action="store_true",
                    help="draw the subject choice, with numbered people, into <out>/_verdicts")
    ap.add_argument("--ratios", default=",".join(AR_FAMILIES),
                    help="aspect ratios to choose from, comma-separated (default: all seven k2prep ratios)")
    ap.add_argument("--reencode", action="store_true",
                    help="plan JPEG crops for re-encoding instead of a lossless DCT crop")
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
    families = [r.strip() for r in args.ratios.split(",") if r.strip()]
    bad = [r for r in families if r not in AR_FAMILIES]
    if bad or not families:
        ap.error(f"unknown ratio(s) {', '.join(bad) or '(none)'}; choose from {', '.join(AR_FAMILIES)}")

    models = None
    for root in roots:
        out_dir = Path(args.out).resolve() if args.out else root / OUT_DIRNAME
        cache_path = out_dir / DETECTIONS_NAME
        cache = {} if args.redetect else load_cache(cache_path)
        files = scan(root)
        overrides = read_overrides(root)
        known = {path.relative_to(root).as_posix().lower() for path in files}
        for rel in sorted(set(overrides) - known):
            print(f"  {OVERRIDES_NAME}: no such photo: {rel}")
        print(f"{root}: {len(files)} image(s)" + (f", {len(overrides)} override(s)" if overrides else ""))
        t0, n_new = time.time(), 0
        results, plan = {}, {}
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
            sel = select_subject(det, overrides.get(rel.lower()))
            crop = None
            if sel["verdict"] == "crop":
                crop = plan_crop(det, sel, image_header(path), families, lossless=not args.reencode)
            plan[rel] = dict(sel, crop=crop)
            if args.verdicts:
                img = img or load_image(path)
                draw_verdict(img, det, sel, out_dir / VERDICT_PREVIEW_DIRNAME / (rel + ".jpg"), crop)
            if crop and crop["verdict"] == "crop":
                what = (f"CROP {crop['family']} {crop['size'][0]}x{crop['size'][1]} "
                        f"(keeps {crop['keeps']:.0%}{', lossless' if crop['lossless'] else ''})")
            elif crop:
                what = f"unchanged, {crop['reason']}"
            else:
                what = f"unchanged, {sel['reason']}"
            print(f"[{k}/{len(files)}] {rel}: {len(det['people'])} people; {what}"
                  + "".join(f" [{f}]" for f in sel["flags"] + (crop or {}).get("flags", []))
                  + ("" if fresh else " (cached)"), flush=True)
        write_json(cache_path, {"version": CACHE_VERSION, "models": models_signature(),
                                "written": datetime.now().isoformat(timespec="seconds"), "files": results})
        write_json(out_dir / PLAN_NAME, {"written": datetime.now().isoformat(timespec="seconds"),
                                         "people_version": PEOPLE_VERSION, "photos": plan})
        n_crop = sum(1 for v in plan.values() if v["crop"] and v["crop"]["verdict"] == "crop")
        n_flag = sum(1 for v in plan.values() if v["flags"] or (v["crop"] or {}).get("flags"))
        print(f"{n_new} detected, {len(files) - n_new} from cache, {time.time() - t0:.0f}s; "
              f"{n_crop} to crop, {len(plan) - n_crop} unchanged" + (f", {n_flag} flagged" if n_flag else "")
              + f" -> {out_dir / PLAN_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
