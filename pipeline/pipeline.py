#!/usr/bin/env python3
"""
Run the repair tools of this repository over a dataset image by image: each
image is decoded once, passes through the chosen stages in memory, and is
written once. No temporary files.

Stages, in this order, each taken from its own tool folder (../remove_borders,
../watermark, ../reframe, ../jpeg_cleanup, ../face_masks):
  borders        frames, lines and text banners -> a crop box
  watermark      watermark boxes -> painted out, or a trim box
  reframe        the subject of a photo of people -> a crop box
  (compose)      one crop box from the boxes above, on the JPEG block grid
  (inpaint)      the watermark boxes still inside the crop are painted out
  jpeg_cleanup   heavily compressed JPEGs restored with FBCNN
  (write)        no pixel stage touched the image: one lossless crop of the
                 file (whole DCT blocks for JPEG, exact for lossless formats,
                 a lossy format into PNG); otherwise the pixels, once, as
                 JPEG quality 97 without chroma subsampling, or PNG
  face_masks     a loss mask of the final image into <folder>/masks/<stem>.png

The job comes from a JSON file (--job), see JOB_EXAMPLE and the README. Two
output modes: "in_place" changes the dataset folder, with every original and
its captions copied to <folder>/_backup/<same relative path> first (never
over an earlier backup) and the run logged in _backup/_pipeline/log.jsonl, so
that --undo puts the last run back; "parallel" writes the whole tree into a
new folder and copies the unchanged images and the captions along. A dry run
changes nothing: it writes the report, the previews (reframe verdicts, border
and cleanup contact sheets, watermark masks) and the detection caches in
_backup/_pipeline, so the real run that follows does no new GPU work for the
detections. Captions are never deleted or changed; the log says which images
changed, so they can be captioned again.

Progress goes to stdout as one JSON line per image (index, total, path, what
changed, what was written) and a final summary line.

Usage:    python pipeline.py --job job.json [--dry-run] [--undo] [--threads N]
Install:  ..\\install.bat (the install.bat of every tool above, in sequence)
"""
import argparse
import filecmp
import json
import os
import shutil
import signal
import stat
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _tool in ("remove_borders", "watermark", "reframe", "jpeg_cleanup", "face_masks"):
    if str(REPO / _tool) not in sys.path:
        sys.path.insert(0, str(REPO / _tool))
import remove_borders as rb  # noqa: E402
import watermark as wm  # noqa: E402
import reframe as rf  # noqa: E402
import jpeg_cleanup as jc  # noqa: E402
import make_face_masks as fm  # noqa: E402

BACKUP_DIRNAME = "_backup"
RUN_DIRNAME = "_pipeline"                         # inside _backup: log, cache, report, previews, originals
LOG_NAME = "log.jsonl"
CACHE_NAME = "cache.json"
REPORT_NAME = "report.jsonl"
PREVIEWS_DIRNAME = "previews"
ORIGINALS_DIRNAME = "originals"                   # originals whose place in _backup was taken by an earlier tool's
MASKS_DIRNAME = "masks"
CACHE_VERSION = 1
STAGES = ["borders", "watermark", "reframe", "jpeg_cleanup", "face_masks"]
IMAGE_EXTS = rb.IMAGE_EXTS
DEFAULT_EXCLUDES = rb.DEFAULT_EXCLUDES
DEFAULT_THREADS = 4
JPEG_QUALITY = 97
PNG_LEVEL = 6
TRIM_MIN_OF_REFRAME = 0.5         # a trim that keeps less of the reframe crop is dropped for painting

JOB_EXAMPLE = {
    "folder": "D:/datasets/set",
    "files": ["optional/relative/path.jpg"],
    "stages": {
        "borders": {"dark": True, "min_area": rb.DEFAULT_MIN_AREA},
        "watermark": {"conf": wm.DEFAULT_CONF, "dilate": wm.DEFAULT_DILATE, "mode": "inpaint",
                      "trim_min_keep": wm.DEFAULT_TRIM_MIN_KEEP, "max_size": wm.DEFAULT_MAX_SIZE},
        "reframe": {"ratios": rf.AR_FAMILIES, "overrides": True},
        "jpeg_cleanup": {"threshold": jc.DEFAULT_THRESHOLD, "qf_offset": jc.DEFAULT_QF_OFFSET,
                         "min_block_drop": jc.DEFAULT_MIN_BLOCK_DROP, "min_qf_gain": jc.DEFAULT_MIN_QF_GAIN,
                         "max_pixels": jc.DEFAULT_MAX_PIXELS},
        "face_masks": {"conf": 0.3, "grow": 1.35, "feather": 12, "include_hair": False, "invert": False},
    },
    "output": {"mode": "in_place"},
    "save_png": False,
    "dry_run": False,
    "undo": False,
    "threads": DEFAULT_THREADS,
}


@dataclass
class Item:
    path: Path
    rel: str
    sidecars: list = field(default_factory=list)
    key: list | None = None


@dataclass
class Job:
    root: Path
    files: list | None
    stages: dict
    mode: str                                     # in_place | parallel
    target: Path | None
    save_png: bool
    dry_run: bool
    undo: bool
    threads: int
    excludes: list


def read_job(path: Path, args) -> Job:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    root = Path(data["folder"]).resolve()
    if not root.is_dir():
        raise SystemExit(f"not a folder: {root}")
    stages = data.get("stages") or {}
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s): {', '.join(unknown)}; the stages are {', '.join(STAGES)}")
    out = data.get("output") or {"mode": "in_place"}
    mode = out.get("mode", "in_place")
    if mode not in ("in_place", "parallel"):
        raise SystemExit(f"output mode must be in_place or parallel, got {mode!r}")
    target = None
    if mode == "parallel":
        if not out.get("folder"):
            raise SystemExit("parallel output needs output.folder")
        target = Path(out["folder"]).resolve()
        if target == root or target.is_relative_to(root) or root.is_relative_to(target):
            raise SystemExit(f"the output folder {target} must be outside the dataset folder {root}")
    threads = args.threads or int(data.get("threads") or DEFAULT_THREADS)
    return Job(root=root, files=data.get("files"), stages=stages, mode=mode, target=target,
               save_png=bool(data.get("save_png")), dry_run=bool(args.dry_run or data.get("dry_run")),
               undo=bool(args.undo or data.get("undo")), threads=max(1, min(32, threads)),
               excludes=DEFAULT_EXCLUDES + list(data.get("exclude") or []))


# --- scan ---------------------------------------------------------------------------

def scan(job: Job) -> list[Item]:
    """The images to process, with their .txt captions: the job's file list,
    or the whole folder at any depth without the tools' output folders."""
    root = job.root
    items = []
    if job.files:
        for rel in job.files:
            p = root / rel
            if not p.is_file():
                print(json.dumps({"warning": f"not a file: {rel}"}), flush=True)
                continue
            items.append(Item(path=p, rel=p.relative_to(root).as_posix()))
    else:
        excl = {e.casefold() for e in job.excludes}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d.casefold() not in excl)
            for fn in sorted(filenames):
                if os.path.splitext(fn)[1].casefold() in IMAGE_EXTS:
                    p = Path(dirpath) / fn
                    items.append(Item(path=p, rel=p.relative_to(root).as_posix()))
    for it in items:
        cap = it.path.with_suffix(".txt")
        it.sidecars = [cap] if cap.is_file() else []
        try:
            it.key = rb.file_key(it.path)
        except OSError:
            it.key = None
    return items


# --- decoding -------------------------------------------------------------------------

def decode_image(path: Path) -> tuple[dict, np.ndarray | None]:
    """The header of one image (remove_borders' read_header plus jpeg_cleanup's
    facts: exif, icc, header_q, frames, transparency) and its upright RGB uint8
    pixels, transparency composited over white. Pixels None when skipped."""
    head = rb.read_header(path)
    if head["error"] or head["frames"] > 1:
        return head, None
    with Image.open(path) as im:
        head.update({k: v for k, v in jc.facts_of(im, path).items()
                     if k in ("exif", "icc", "header_q", "lossless_webp")})
        head["transparent"] = False
        if "A" in im.mode or (im.mode == "P" and "transparency" in im.info) or "transparency" in im.info:
            try:
                head["transparent"] = im.convert("RGBA").getchannel("A").getextrema()[0] < 255
            except Exception:  # noqa: BLE001
                head["transparent"] = True
    head["stored"] = [head["width"], head["height"]]
    stored = rb.load_pixels(path)
    return head, np.ascontiguousarray(rb.to_upright(stored, head["orientation"]))


# --- boxes ----------------------------------------------------------------------------

def intersect(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return [x0, y0, x1, y1] if x1 > x0 and y1 > y0 else None


def shift(box, dx, dy):
    return [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]


def area(box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def align_box(display_box, head: dict, inward_display_sides: set) -> tuple[list, list | None]:
    """The composed crop on the JPEG block grid, in stored pixels: the stored
    origin moves to the MCU grid, inward (so that a border or a watermark goes
    completely) on a side that a border cut or a trim set, outward on the
    others, as remove_borders and reframe do on their own; the far edges stay.
    inward_display_sides holds "L" and "T" (and "R", "B", which only matter
    for an image stored rotated) in upright terms.
    -> (display box, stored box); a non-JPEG gets its exact box in stored pixels."""
    o, (sw, sh) = head["orientation"], head["stored"]
    stored = rf.display_to_stored([int(v) for v in display_box], o, sw, sh)
    if not head.get("mcu"):
        return [int(v) for v in display_box], [int(v) for v in stored]
    mw, mh = head["mcu"]
    # which upright side became the stored left and top
    left_src = {1: "L", 2: "R", 3: "R", 4: "L", 5: "T", 6: "T", 7: "B", 8: "B"}[o]
    top_src = {1: "T", 2: "T", 3: "B", 4: "B", 5: "L", 6: "R", 7: "R", 8: "L"}[o]
    x0, y0, x1, y1 = stored
    nx0 = -(-x0 // mw) * mw if left_src in inward_display_sides else (x0 // mw) * mw
    ny0 = -(-y0 // mh) * mh if top_src in inward_display_sides else (y0 // mh) * mh
    if nx0 >= x1:
        nx0 = (x0 // mw) * mw
    if ny0 >= y1:
        ny0 = (y0 // mh) * mh
    stored = [int(nx0), int(ny0), int(x1), int(y1)]
    return [int(v) for v in rb.stored_to_display(stored, o, sw, sh)], stored


# --- the stages of one image ------------------------------------------------------------

class Caches:
    """Per-file detection caches in _backup/_pipeline/cache.json, keyed by
    the file (size, time, ID) and a signature of the stage's inputs."""

    def __init__(self, root: Path):
        self.path = root / BACKUP_DIRNAME / RUN_DIRNAME / CACHE_NAME
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.files = data.get("files", {}) if data.get("version") == CACHE_VERSION else {}
        except (OSError, ValueError):
            self.files = {}
        self.dirty = False

    def get(self, it: Item, stage: str, sig: str):
        e = self.files.get(it.rel)
        if not e or not it.key or e.get("key") != it.key:
            return None
        return e.get(stage, {}).get(sig)

    def put(self, it: Item, stage: str, sig: str, value) -> None:
        if not it.key:
            return
        e = self.files.get(it.rel)
        if not e or e.get("key") != it.key:
            e = self.files[it.rel] = {"key": it.key}
        e.setdefault(stage, {})[sig] = value
        self.dirty = True

    def save(self) -> None:
        if not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".part")
        tmp.write_text(json.dumps({"version": CACHE_VERSION, "files": self.files}, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, self.path)
        self.dirty = False


def sig(*parts) -> str:
    return json.dumps(parts, sort_keys=True, separators=(",", ":"))


class Stages:
    """The stage functions with their options, models loaded on first use."""

    def __init__(self, job: Job, caches: Caches):
        self.job, self.caches = job, caches
        s = job.stages
        self.borders = {"dark": True, "min_area": rb.DEFAULT_MIN_AREA, **(s.get("borders") or {})} if "borders" in s else None
        self.watermark = ({"conf": wm.DEFAULT_CONF, "dilate": wm.DEFAULT_DILATE, "mode": "inpaint",
                           "trim_min_keep": wm.DEFAULT_TRIM_MIN_KEEP, "max_size": wm.DEFAULT_MAX_SIZE,
                           **(s.get("watermark") or {})} if "watermark" in s else None)
        self.reframe = {"ratios": list(rf.AR_FAMILIES), "overrides": True, **(s.get("reframe") or {})} if "reframe" in s else None
        self.cleanup = ({"threshold": jc.DEFAULT_THRESHOLD, "qf_offset": jc.DEFAULT_QF_OFFSET,
                         "min_block_drop": jc.DEFAULT_MIN_BLOCK_DROP, "min_qf_gain": jc.DEFAULT_MIN_QF_GAIN,
                         "max_pixels": jc.DEFAULT_MAX_PIXELS, **(s.get("jpeg_cleanup") or {})}
                        if "jpeg_cleanup" in s else None)
        self.faces = ({"conf": 0.3, "grow": 1.35, "feather": 12, "include_hair": False, "invert": False,
                       **(s.get("face_masks") or {})} if "face_masks" in s else None)
        self.overrides = rf.read_overrides(job.root) if self.reframe and self.reframe.get("overrides", True) else {}
        if self.reframe:
            bad = [r for r in self.reframe["ratios"] if r not in rf.AR_FAMILIES]
            if bad:
                raise SystemExit(f"reframe: unknown ratio(s) {', '.join(bad)}; choose from {', '.join(rf.AR_FAMILIES)}")

    # borders runs on the CPU, in the decoder threads
    def plan_borders(self, it: Item, head: dict, array: np.ndarray) -> dict | None:
        if not self.borders:
            return None
        key = sig(rb.DETECTOR_VERSION, self.borders)
        cached = self.caches.get(it, "borders", key)
        if cached is not None:
            result = dict(cached)
            if result.get("cuts"):
                result["plan"] = rb.crop_plan(head, result, self.borders["min_area"])
            action = rb.action_of(head, result)
            p = {"action": action, "result": result, "box": None, "display_box": None, "write": ""}
            if action in ("crop", "too small"):
                box = result["plan"]["box"]
                W, H = result["size"]
                p.update(box=list(box), display_box=rb.stored_to_display(box, head["orientation"], W, H),
                         write=result["plan"]["write"])
            return p
        p = rb.plan(array, head, self.borders)
        self.caches.put(it, "borders", key, {k: v for k, v in p["result"].items() if k != "plan"})
        return p

    def detect_watermark(self, it: Item, sub: np.ndarray, sub_box) -> list:
        key = sig(wm.DETECTOR_VERSION, self.watermark["conf"], sub_box)
        cached = self.caches.get(it, "watermark", key)
        if cached is not None:
            return [list(b) for b in cached]
        boxes = wm.detect(wm.detector_lazy()[1], sub, self.watermark["conf"])
        self.caches.put(it, "watermark", key, boxes)
        return boxes

    def plan_reframe(self, it: Item, sub: np.ndarray, sub_box, head: dict) -> dict:
        key = sig(rf.CACHE_VERSION, rf.models_signature(), sub_box)
        det = self.caches.get(it, "reframe", key)
        fresh = det is None
        p = rf.plan(sub, {"format": head["format"], "orientation": 1, "stored": [sub.shape[1], sub.shape[0]]},
                    {"det": None if fresh else dict(det), "families": self.reframe["ratios"], "lossless": False,
                     "override": self.overrides.get(it.rel.lower())})
        if fresh or det.get("people_version") != rf.PEOPLE_VERSION:
            self.caches.put(it, "reframe", key, p["det"])
        return p

    def plan_cleanup(self, it: Item, array: np.ndarray, head: dict, inputs, sheet: bool) -> dict:
        """inputs: what made the array (the crop box and the painted boxes), so
        a cached measurement belongs to these pixels."""
        key = sig(jc.MEASURE_VERSION, inputs, self.cleanup["max_pixels"])
        m = self.caches.get(it, "jpeg_cleanup", key)
        opts = {k: v for k, v in self.cleanup.items()} | {"m": m, "sheet": sheet, "quality": JPEG_QUALITY}
        p = jc.plan(array, head, opts)
        if m is None and p["m"] and not p["m"].get("skip", "").startswith("unreadable"):
            self.caches.put(it, "jpeg_cleanup", key, p["m"])
        return p

    def plan_faces(self, it: Item, array: np.ndarray, inputs) -> dict:
        key = sig(self.faces["conf"], inputs)
        boxes = self.caches.get(it, "face_masks", key)
        opts = dict(self.faces)
        if boxes is None:
            p = fm.plan(array, None, opts)
            self.caches.put(it, "face_masks", key, p["boxes"])
            return p
        h, w = array.shape[:2]
        return {"boxes": [list(b) for b in boxes], "size": [w, h], **{k: v for k, v in opts.items() if k != "model"}}


def process(it: Item, head: dict, array: np.ndarray, bp: dict | None, stages: Stages, previews) -> dict:
    """All stages of one decoded image, in order. -> what to write:
    {"changed": [stage names], "box": the final crop in upright pixels or
     None, "stored_box": the same in stored pixels for a lossless JPEG crop,
     "touched": a pixel stage changed the pixels, "array": the final pixels
     (None when untouched), "mask": the face mask array or None, "notes": {...}}"""
    H, W = array.shape[:2]
    full = [0, 0, W, H]
    changed, notes = [], {}
    inward = set()
    # 1. borders
    box = full
    if bp is not None:
        notes["borders"] = {"action": bp["action"], "cuts": rb.describe_cuts(bp["result"].get("cuts", [])),
                            "note": bp["result"].get("review") or bp["result"].get("plan", {}).get("note") or ""}
        if bp["action"] == "crop":
            box = rb.apply(array, bp)
            inward |= {s for s, cut in zip("LTRB", (box[0] > 0, box[1] > 0, box[2] < W, box[3] < H)) if cut}
            changed.append("borders")
        if previews is not None and bp["action"] in ("crop", "review", "too small"):
            previews.border_tile(it, head, array, bp)
    sub = array[box[1]:box[3], box[0]:box[2]]
    sh, sw = sub.shape[:2]
    # 2. watermark detection, on the picture inside the borders
    wm_boxes, trim_box = [], None
    if stages.watermark:
        wm_boxes = stages.detect_watermark(it, sub, box)
        if wm_boxes:
            notes["watermark"] = {"boxes": len(wm_boxes), "conf": max(b[4] for b in wm_boxes)}
            if stages.watermark["mode"] == "trim":
                mask = wm.boxes_mask(sub.shape, wm_boxes, stages.watermark["dilate"])
                rect = wm.largest_clear_rect(wm.mask_boxes(mask), sw, sh)
                if rect and area(rect) >= stages.watermark["trim_min_keep"] * sw * sh:
                    trim_box = rect
        if previews is not None and wm_boxes:
            previews.watermark(it, sub, wm_boxes, trim_box, stages.watermark)
    # 3. reframe, on the same picture
    rf_box = None
    if stages.reframe:
        p = stages.plan_reframe(it, sub, box, head)
        notes["reframe"] = {"verdict": p["verdict"], "reason": p["reason"], "people": len(p["det"]["people"]),
                            "flags": p["flags"]}
        if p["verdict"] == "crop":
            rf_box = rf.apply(sub, p)
            notes["reframe"].update(family=p["crop"]["family"], size=p["crop"]["size"])
        if previews is not None:
            previews.verdict(it, sub, p)
    # 4. compose: inside the borders, then the reframe crop, then the trim
    crop = [0, 0, sw, sh]
    if rf_box:
        crop = rf_box
        changed.append("reframe")
    if trim_box:
        both = intersect(crop, trim_box)
        if both and area(both) >= TRIM_MIN_OF_REFRAME * area(crop):
            if both != crop:
                inward |= {s for s, cut in zip("LTRB", (both[0] > crop[0], both[1] > crop[1],
                                                        both[2] < crop[2], both[3] < crop[3])) if cut}
                crop = both
                changed.append("watermark")
            notes["watermark"]["trim"] = True
        else:
            trim_box = None                               # the trim would cut the subject: paint instead
    if rf_box and crop != full:
        # a reframe origin lies inside the picture: it may move outward to the grid
        pass
    final = shift(crop, box[0], box[1])
    final, stored_box = align_box(final, head, inward)
    final = intersect(final, full) or full
    if final != full and "borders" not in changed and "reframe" not in changed and "watermark" not in changed:
        changed.append("borders" if bp is not None else "crop")
    x0, y0, x1, y1 = final
    out = np.ascontiguousarray(array[y0:y1, x0:x1])
    touched = False
    # 5. paint the watermarks still inside the crop
    if wm_boxes and not trim_box:
        inside = []
        for b in wm_boxes:
            fb = shift(b[:4], box[0], box[1])
            cut = intersect(fb, final)
            if cut:
                inside.append(shift(cut, -x0, -y0) + [b[4]])
        if inside:
            out = wm.inpaint(out, inside, stages.watermark["dilate"], stages.watermark["max_size"])
            touched = True
            changed.append("watermark")
            notes.setdefault("watermark", {})["painted"] = len(inside)
    # 6. jpeg_cleanup on the pixels as they are now
    if stages.cleanup:
        inputs = [final, [b[:4] for b in wm_boxes] if touched else []]
        p = stages.plan_cleanup(it, out, head, inputs, sheet=previews is not None)
        notes["jpeg_cleanup"] = {"action": p["action"], "qf": p["m"].get("qf"), **{k: v for k, v in p["fix"].items()
                                                                                      if k in ("benefit", "qf_after", "block_before", "block_after")}}
        if p["action"] == "fix":
            out = jc.apply(out, p)
            touched = True
            changed.append("jpeg_cleanup")
        if previews is not None and p.get("outs") is not None:
            previews.cleanup_tile(it, head, p)
    # 7. the face mask of the final image
    mask = None
    if stages.faces:
        p = stages.plan_faces(it, out, [final, touched])
        mask = fm.apply(out, p)
        notes["face_masks"] = {"faces": len(p["boxes"])}
    return {"changed": changed, "box": None if final == full else final, "stored_box": stored_box,
            "touched": touched, "array": out if touched else None, "mask": mask, "notes": notes}


# --- previews (dry run) ------------------------------------------------------------------

class Previews:
    """The pictures of a dry run in _backup/_pipeline/previews: the reframe
    verdicts, the watermark masks, and contact sheets of the border crops and
    the cleanup restorations, drawn the way the tools draw them."""

    def __init__(self, root: Path, threads: int):
        self.dir = root / BACKUP_DIRNAME / RUN_DIRNAME / PREVIEWS_DIRNAME
        if self.dir.is_dir():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.border_tiles = []
        self.sheets = None
        self.pool = ThreadPoolExecutor(max_workers=max(1, threads))
        self.pending = []

    def _submit(self, fn, *args):
        self.pending.append(self.pool.submit(fn, *args))
        while len(self.pending) > 8:
            self.pending.pop(0).result()

    def verdict(self, it: Item, sub: np.ndarray, p: dict) -> None:
        dest = self.dir / "reframe" / (it.rel + ".jpg")
        self._submit(rf.draw_verdict, Image.fromarray(sub), p["det"], p["sel"], dest, p["crop"])

    def watermark(self, it: Item, sub: np.ndarray, boxes, trim_box, opts) -> None:
        plan_ = {"boxes": boxes, "dilate": opts["dilate"], "trim_box": trim_box}
        self._submit(wm.draw_preview, sub, plan_, self.dir / "watermark" / (it.rel + ".jpg"))

    def border_tile(self, it: Item, head: dict, array: np.ndarray, bp: dict) -> None:
        stored = Image.fromarray(rb.to_stored(array, head["orientation"]))
        label = f"{it.rel}  {head['width']}x{head['height']}"
        self.border_tiles.append((rb.sheet_group(_ItemLike(bp)), self.pool.submit(rb.tile_of, stored, bp["result"], label)))

    def cleanup_tile(self, it: Item, head: dict, p: dict) -> None:
        if self.sheets is None:
            heads = ["original", "restored"]
            self.sheets = {True: jc.SheetWriter(self.dir / "cleanup", [], "fix", heads),
                           False: jc.SheetWriter(self.dir / "cleanup", [], "nofix", heads)}
        r, m = p["fix"], p["m"]
        label = (f"{'SAVE: ' + r['benefit'] if r['benefit'] else 'LEAVE: little benefit'}  |  "
                 f"QF {m['qf']:.1f} > {r['qf_after']:.1f}, blockiness {r['block_before']:.2f} > {r['block_after']:.2f}, "
                 f"change {r['change']:.2f}, {m['width']}x{m['height']}{', gray' if m.get('gray') else ''}")
        tile = jc.make_tile(it.rel, label, p["orig"], p["outs"], head["orientation"])
        self.sheets[bool(r["benefit"])].add(m["qf"], tile)

    def finish(self) -> dict:
        for f in self.pending:
            f.result()
        counts = {}
        if self.border_tiles:
            out = self.dir / "borders"
            out.mkdir(parents=True, exist_ok=True)
            per = rb.SHEET_COLS * rb.SHEET_ROWS
            for g in rb.SHEET_GROUPS:
                members = [f for grp, f in self.border_tiles if grp == g]
                for start in range(0, len(members), per):
                    chunk = members[start:start + per]
                    rows = (len(chunk) + rb.SHEET_COLS - 1) // rb.SHEET_COLS
                    sheet = Image.new("RGB", (rb.SHEET_COLS * (rb.TILE_W + 8), rows * (rb.TILE_H + 8)), (20, 20, 24))
                    for k, f in enumerate(chunk):
                        w, h, data = f.result()
                        sheet.paste(Image.frombytes("RGB", (w, h), data),
                                    ((k % rb.SHEET_COLS) * (rb.TILE_W + 8) + 4, (k // rb.SHEET_COLS) * (rb.TILE_H + 8) + 4))
                    sheet.save(out / f"{g}_{start // per + 1:03d}.jpg", quality=88)
                    counts["border sheets"] = counts.get("border sheets", 0) + 1
        if self.sheets is not None:
            for s in self.sheets.values():
                s.flush()
            counts["cleanup sheets"] = sum(s.count for s in self.sheets.values())
        self.pool.shutdown()
        return counts


class _ItemLike:
    """What remove_borders.sheet_group reads of an Item: action and result."""

    def __init__(self, bp: dict):
        self.action, self.result = bp["action"], bp["result"]


# --- writing --------------------------------------------------------------------------------

def same_bytes(a: Path, b: Path) -> bool:
    try:
        return a.stat().st_size == b.stat().st_size and filecmp.cmp(a, b, shallow=False)
    except OSError:
        return False


def make_writable(path: Path) -> None:
    if path.exists() and not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)


def copy_keep(src: Path, dst: Path) -> str:
    """Copy src to dst unless dst exists. -> copied, same or kept."""
    if dst.exists():
        return "same" if same_bytes(src, dst) else "kept"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return "copied"


def backup_place(root: Path, rel: str, src: Path, run_id: str) -> str:
    """Where the original of rel goes: _backup/rel, unless an earlier original
    of another run is there; then _backup/_pipeline/originals/<run>/rel."""
    bk = root / BACKUP_DIRNAME / rel
    if bk.exists() and not same_bytes(src, bk):
        bk = root / BACKUP_DIRNAME / RUN_DIRNAME / ORIGINALS_DIRNAME / run_id / rel
    return bk.relative_to(root).as_posix()


def output_name(rel: str, head: dict, write: str, save_png: bool) -> str:
    """The relative path of the image written: the same, or .png."""
    if write == "array":
        fmt = "JPEG" if head["format"] in ("JPEG", "MPO") and not save_png else "PNG"
        return rel if fmt == "JPEG" else Path(rel).with_suffix(".png").as_posix()
    if write == "lossless" and head["kind"] == rb.KIND_LOSSY:
        return Path(rel).with_suffix(".png").as_posix()
    return rel


def write_array(array: np.ndarray, dst: Path, fmt: str, head: dict, src_stat) -> None:
    im = Image.fromarray(np.ascontiguousarray(array))
    exif = rf.exif_for_output(head.get("exif"), im.width, im.height, keep_orientation=False)
    kw = {k: v for k, v in (("exif", exif), ("icc_profile", head.get("icc"))) if v}
    tmp = rb.part_path(dst)
    if fmt == "JPEG":
        im.save(tmp, "JPEG", quality=JPEG_QUALITY, subsampling=0, optimize=True, **kw)
    else:
        im.save(tmp, "PNG", compress_level=PNG_LEVEL, **kw)
    rb.finish_write(tmp, dst, src_stat)


def write_job(job: dict) -> dict:
    """One image's writes, in a worker process: the backups (in place), the
    image, the mask. -> the log record."""
    rec = {"op": "done", "rel": job["rel"], "changed": job["changed"], "write": job["write"]}
    try:
        root, src, dst = Path(job["root"]), Path(job["src"]), Path(job["dst"])
        head = job["head"]
        in_place = job["mode"] == "in_place"
        st = os.stat(src)
        if in_place and job["write"] != "none":
            rec["backup"] = copy_keep(src, root / job["backup"])
            rec["sidecars"] = {Path(sc).relative_to(root).as_posix(): copy_keep(Path(sc), root / BACKUP_DIRNAME / Path(sc).relative_to(root))
                               for sc in job["sidecars"]}
        if job["write"] == "none":
            if not in_place:
                copy_keep(src, dst)
                for sc in job["sidecars"]:
                    copy_keep(Path(sc), dst.parent / Path(sc).name)
        elif job["write"] == "lossless":
            kind = head["kind"]
            if kind == rb.KIND_JPEG:
                rb.write_jpeg(src, dst, job["stored_box"])
            elif kind == rb.KIND_LOSSLESS:
                rb.write_exact(src, dst, job["stored_box"], head.get("bits", 8))
            else:
                rb.write_png(src, dst, job["stored_box"])
        else:
            write_array(job["array"], dst, job["fmt"], head, st)
        if job["write"] != "none":
            if not in_place:
                for sc in job["sidecars"]:
                    copy_keep(Path(sc), dst.parent / Path(sc).name)
            elif dst != src:
                make_writable(src)                        # the image changed its name: the original is in _backup
                src.unlink()
                rec["removed"] = job["rel"]
        rec["out"] = dst.relative_to(root if in_place else Path(job["target"])).as_posix()
        if job.get("mask") is not None:
            mpath = Path(job["mask_path"])
            if in_place and mpath.exists():
                rec["mask_backup"] = copy_keep(mpath, root / BACKUP_DIRNAME / mpath.relative_to(root))
            mpath.parent.mkdir(parents=True, exist_ok=True)
            # two images with one stem (a.jpg, a.png) share one mask: each
            # worker writes its own part file, the last one to finish wins
            tmp = mpath.with_name(f"{mpath.name}.part{os.getpid()}")
            Image.fromarray(job["mask"]).save(tmp, "PNG", compress_level=PNG_LEVEL)
            make_writable(mpath)
            os.replace(tmp, mpath)
            rec["mask"] = mpath.relative_to(root if in_place else Path(job["target"])).as_posix()
    except Exception as e:  # noqa: BLE001 - one bad file is reported, not fatal
        rec["error"] = f"{type(e).__name__}: {e}".strip()
        # the writes are atomic, so the image in place is intact: the backups
        # this job copied for it are not needed and go again
        try:
            root, src = Path(job["root"]), Path(job["src"])
            if rec.get("backup") == "copied" and src.is_file() and same_bytes(src, root / job["backup"]):
                (root / job["backup"]).unlink()
                rec["backup"] = "removed"
            for sc, status in list((rec.get("sidecars") or {}).items()):
                b = root / BACKUP_DIRNAME / sc
                if status == "copied" and b.is_file() and same_bytes(b, root / sc):
                    b.unlink()
                    rec["sidecars"][sc] = "removed"
        except OSError:
            pass
    return rec


def ignore_ctrl_c() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class RunLog:
    def __init__(self, root: Path):
        self.path = root / BACKUP_DIRNAME / RUN_DIRNAME / LOG_NAME
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a", encoding="utf-8")

    def write(self, obj: dict) -> None:
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def read_log(root: Path) -> list[dict]:
    path = root / BACKUP_DIRNAME / RUN_DIRNAME / LOG_NAME
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def last_run(entries: list[dict]) -> tuple[dict | None, list[dict]]:
    undone = {e["undo"] for e in entries if "undo" in e}
    runs = [i for i, e in enumerate(entries) if "run" in e and e["run"] not in undone]
    if not runs:
        return None, []
    i = runs[-1]
    j = next((k for k in range(i + 1, len(entries)) if "run" in entries[k]), len(entries))
    return entries[i], entries[i + 1:j]


# --- undo ------------------------------------------------------------------------------------

def undo(root: Path) -> dict:
    """Put back what the last run changed: in place, every image from its
    backup, a renamed output removed, the masks restored or removed; in
    parallel mode, the files written into the target removed. -> counts."""
    head, recs = last_run(read_log(root))
    counts = {"restored": 0, "removed": 0, "no backup": 0}
    if head is None:
        return counts
    mode, target = head["mode"], Path(head.get("target") or "")
    done = {r["rel"]: r for r in recs if r.get("op") == "done" and "error" not in r}
    intents = [r for r in recs if r.get("op") == "intent"]
    for intent in reversed(intents):
        rec = done.get(intent["rel"])
        if rec is None:
            continue
        if mode == "parallel":
            for key in ("out", "mask"):
                f = target / rec[key] if rec.get(key) else None
                if f and f.is_file():
                    make_writable(f)
                    f.unlink()
                    counts["removed"] += 1
                    try:
                        f.parent.rmdir()
                    except OSError:
                        pass
            if rec.get("out"):
                cap = (target / rec["out"]).with_suffix(".txt")
                if cap.is_file() and intent.get("sidecars"):
                    cap.unlink()
            continue
        if intent["write"] != "none":
            bk = root / intent["backup"]
            if not bk.exists():
                counts["no backup"] += 1
                print(json.dumps({"warning": f"no backup for {intent['rel']}"}), flush=True)
            else:
                out = root / rec["out"]
                if rec["out"] != intent["rel"] and out.is_file():
                    make_writable(out)
                    out.unlink()
                    counts["removed"] += 1
                dst = root / intent["rel"]
                make_writable(dst)
                tmp = rb.part_path(dst)
                shutil.copy2(bk, tmp)
                os.replace(tmp, dst)
                if rec.get("backup") == "copied":
                    make_writable(bk)
                    bk.unlink()
                counts["restored"] += 1
            for sc, status in (rec.get("sidecars") or {}).items():
                b = root / BACKUP_DIRNAME / sc
                if not (root / sc).exists() and b.exists():
                    os.replace(b, root / sc)
                elif status == "copied" and b.exists() and same_bytes(b, root / sc):
                    make_writable(b)
                    b.unlink()
        if rec.get("mask"):
            m = root / rec["mask"]
            mb = root / BACKUP_DIRNAME / rec["mask"]
            if rec.get("mask_backup") in ("copied", "same") and mb.exists():
                make_writable(m)
                os.replace(mb, m)
                counts["restored"] += 1
            elif rec.get("mask_backup") == "kept":
                pass                                       # an earlier tool's backup stays; the mask stays too
            elif m.is_file():
                make_writable(m)
                m.unlink()
                counts["removed"] += 1
            try:
                m.parent.rmdir()                      # a masks folder this run made and emptied
            except OSError:
                pass
    log = RunLog(root)
    log.write({"undo": head["run"], "time": time.strftime("%Y-%m-%d %H:%M:%S"), **counts})
    log.close()
    rb.remove_empty_dirs(root / BACKUP_DIRNAME)
    return counts


# --- the run ---------------------------------------------------------------------------------

def bounded_map(pool, fn, args, window: int):
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


def emit(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def run(job: Job) -> int:
    root = job.root
    t0 = time.perf_counter()
    items = scan(job)
    total = len(items)
    caches = Caches(root)
    stages = Stages(job, caches)
    previews = Previews(root, job.threads) if job.dry_run else None
    run_id = time.strftime("%Y%m%d-%H%M%S")
    taken = {e["run"] for e in read_log(root) if "run" in e}
    while run_id in taken:                        # two runs within one second
        run_id += "x"
    log = None
    if not job.dry_run:
        log = RunLog(root)
        log.write({"run": run_id, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": job.mode,
                   "target": str(job.target) if job.target else "", "stages": job.stages, "save_png": job.save_png,
                   "images": total})
    report_path = root / BACKUP_DIRNAME / RUN_DIRNAME / REPORT_NAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = open(report_path, "w", encoding="utf-8")
    counts = {"changed": 0, "unchanged": 0, "skipped": 0, "failed": 0, "masks": 0}
    index = {it.rel: n for n, it in enumerate(items, 1)}

    def decoded(it: Item):
        try:
            head, array = decode_image(it.path)
            bp = stages.plan_borders(it, head, array) if array is not None else None
            return it, head, array, bp, ""
        except Exception as e:  # noqa: BLE001
            return it, {}, None, None, f"{type(e).__name__}: {e}"

    def finish(rec: dict, line: dict) -> None:
        if log is not None:
            log.write(rec)
        if "error" in rec:
            line["error"] = rec["error"]
            counts["failed"] += 1
        else:
            line["written"] = rec.get("out", "")
            if rec.get("mask"):
                line["mask"] = rec["mask"]
                counts["masks"] += 1
            if rec.get("removed"):
                line["removed"] = rec["removed"]
            counts["changed" if line["changed"] else "unchanged"] += 1
        report.write(json.dumps(line, ensure_ascii=False) + "\n")
        emit(line)

    pool = None if job.dry_run else ProcessPoolExecutor(max_workers=job.threads, initializer=ignore_ctrl_c)
    pending = []
    try:
        with ThreadPoolExecutor(max_workers=job.threads) as tp:
            for it, head, array, bp, err in bounded_map(tp, decoded, items, job.threads * 2):
                line = {"index": index[it.rel], "total": total, "path": it.rel, "changed": []}
                if err or array is None:
                    line["skipped"] = err or rb.skip_reason(head)
                    counts["skipped"] += 1
                    report.write(json.dumps(line, ensure_ascii=False) + "\n")
                    emit(line)
                    continue
                try:
                    r = process(it, head, array, bp, stages, previews)
                except Exception as e:  # noqa: BLE001 - one bad image is reported, not fatal
                    line["error"] = f"{type(e).__name__}: {e}"
                    counts["failed"] += 1
                    report.write(json.dumps(line, ensure_ascii=False) + "\n")
                    emit(line)
                    continue
                line["changed"] = r["changed"]
                line["notes"] = r["notes"]
                write = "array" if r["touched"] else "lossless" if r["box"] else "none"
                if r["box"]:
                    line["box"] = r["box"]
                line["write"] = write
                if job.dry_run:
                    line["mask"] = bool(r["mask"] is not None)
                    counts["changed" if r["changed"] else "unchanged"] += 1
                    report.write(json.dumps(line, ensure_ascii=False) + "\n")
                    emit(line)
                    continue
                out_rel = output_name(it.rel, head, write, job.save_png)
                base = root if job.mode == "in_place" else job.target
                dst = base / out_rel
                if write != "none" and job.mode == "in_place" and dst != it.path and dst.exists():
                    line["skipped"] = f"{out_rel} exists already; the image is left as it is"
                    line["changed"] = []
                    counts["skipped"] += 1
                    report.write(json.dumps(line, ensure_ascii=False) + "\n")
                    emit(line)
                    continue
                mask_path = None
                if r["mask"] is not None:
                    mask_path = base / Path(out_rel).parent / MASKS_DIRNAME / (Path(out_rel).stem + ".png")
                intent = {"op": "intent", "rel": it.rel, "write": write, "out": out_rel,
                          "sidecars": [sc.relative_to(root).as_posix() for sc in it.sidecars],
                          "mask": mask_path.relative_to(base).as_posix() if mask_path else "",
                          "backup": backup_place(root, it.rel, it.path, run_id) if job.mode == "in_place" and write != "none" else ""}
                log.write(intent)
                wjob = {"root": str(root), "target": str(job.target) if job.target else "", "rel": it.rel,
                        "src": str(it.path), "dst": str(dst), "head": {k: v for k, v in head.items()},
                        "mode": job.mode, "write": write, "changed": r["changed"], "stored_box": r["stored_box"],
                        "sidecars": [str(sc) for sc in it.sidecars], "backup": intent["backup"],
                        "fmt": "JPEG" if out_rel.lower().endswith((".jpg", ".jpeg", ".jpe", ".jfif")) else "PNG",
                        "array": r["array"], "mask": r["mask"], "mask_path": str(mask_path) if mask_path else ""}
                if write == "lossless" and head["kind"] == rb.KIND_JPEG and not r["stored_box"]:
                    # a JPEG without a known MCU size: crop the pixels instead
                    wjob.update(write="array", array=np.ascontiguousarray(array[r["box"][1]:r["box"][3], r["box"][0]:r["box"][2]]))
                    line["write"] = "array"
                pending.append((line, pool.submit(write_job, wjob)))
                while len(pending) > job.threads * 2:
                    line0, fut = pending.pop(0)
                    finish(fut.result(), line0)
        for line0, fut in pending:
            finish(fut.result(), line0)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        caches.save()
        if log is not None:
            log.write({"end": run_id, **counts})
            log.close()
        report.close()
    summary = {"summary": True, "images": total, **counts, "seconds": round(time.perf_counter() - t0, 1),
               "dry_run": job.dry_run, "mode": job.mode, "report": str(report_path)}
    if previews is not None:
        summary["previews"] = str(previews.dir)
        summary.update(previews.finish())
    try:
        import torch
        if torch.cuda.is_available():
            summary["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
    except Exception:  # noqa: BLE001
        pass
    emit(summary)
    return 1 if counts["failed"] else 0


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace", encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Run the repair tools over a dataset image by image, from a job file.")
    ap.add_argument("--job", required=False, metavar="FILE", help="the job as JSON (see --example)")
    ap.add_argument("--dry-run", action="store_true", help="plan, report and preview; change nothing")
    ap.add_argument("--undo", action="store_true", help="put back what the last run of the job's folder changed")
    ap.add_argument("--threads", type=int, default=0, metavar="N", help="decoder threads and writer processes")
    ap.add_argument("--example", action="store_true", help="print an example job and exit")
    args = ap.parse_args(argv)
    if args.example:
        print(json.dumps(JOB_EXAMPLE, indent=1))
        return 0
    if not args.job:
        ap.error("--job is required")
    job = read_job(Path(args.job), args)
    if job.undo:
        c = undo(job.root)
        emit({"summary": True, "undo": True, **c})
        return 0
    return run(job)


if __name__ == "__main__":
    sys.exit(main())
