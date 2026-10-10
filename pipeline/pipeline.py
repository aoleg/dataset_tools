#!/usr/bin/env python3
"""
Run the repair tools of this repository over a dataset from a job file, one
model at a time: detection passes over all images first, each loading one
model and caching what it finds, then the plan, then one apply pass that
decodes each image once, runs the in-memory chain, and writes it once.

Passes, in this order, each over all the images of the job (or its file list):
  borders    remove_borders analysis (CPU)                  -> border cuts
  watermark  the watermark detector, on the full original   -> boxes
  reframe    the three reframe detectors, on the original   -> people
  faces      the face detector, on the full original        -> face boxes
  quality    the FBCNN quality predictor, on the original   -> quality factor
  compose    no model: one crop box from the border cuts, the watermark trim
             and the subject crop, on the JPEG block grid; the watermark boxes
             left inside it; the face boxes moved into it; whether the image
             is eligible for the cleanup (QF under the threshold, crop within
             the size limit)
  benefit    FBCNN restores the composed crop of each eligible image in
             memory and judges, as jpeg_cleanup does, whether the restoration
             is worth saving (the apply pass restores again: FBCNN runs twice)
  apply      decode once upright, crop, paint the watermark boxes left inside
             the crop (LaMa), restore (FBCNN) when the verdict says so, write
             once (a lossless crop of the file when no pixel stage touched the
             image), then the face mask of the final image into masks/
Every detection is cached per image in <folder>/_backup/_pipeline/cache.json,
keyed by the file's size and time and the model's signature, so a pass whose
results are cached loads no model, and a real run after a dry run detects
nothing again. A dry run is every pass but the last; it writes plan.json and
the previews (reframe verdicts, border and cleanup contact sheets, watermark
masks). The detection passes only read the images; the apply pass changes
files and must run alone.

Two output modes: "in_place" changes the dataset folder, with the original of
every changed image and its caption copied to <folder>/_backup/<same
relative path> before the write, never over an original that is there
already (one backup per image, the untouched original, shared with the other
tools), and the run logged in _backup/_pipeline/log.jsonl, so that --undo
returns the images of the last run to their originals; "parallel" writes the
whole tree into a new folder with the unchanged images and the captions
copied along. Captions are never deleted or changed; the log says which images
changed, so they can be captioned again.

Progress goes to stdout as JSON lines: one per image per pass (pass, index,
total, path), a summary per pass, and a final summary.

Usage:    python pipeline.py --job job.json [--dry-run] [--undo] [--threads N]
Install:  ..\\install.bat (the install.bat of every tool above, in sequence)
"""
import argparse
import filecmp
import gc
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
RUN_DIRNAME = "_pipeline"                         # inside _backup: log, cache, plan, report, previews
LOG_NAME = "log.jsonl"
CACHE_NAME = "cache.json"
PLAN_NAME = "plan.json"
REPORT_NAME = "report.jsonl"
PREVIEWS_DIRNAME = "previews"
MASKS_DIRNAME = "masks"
CACHE_VERSION = 2
STAGES = ["borders", "watermark", "reframe", "jpeg_cleanup", "face_masks"]
IMAGE_EXTS = rb.IMAGE_EXTS
DEFAULT_EXCLUDES = rb.DEFAULT_EXCLUDES
DEFAULT_THREADS = 4
JPEG_QUALITY = 97
PNG_LEVEL = 6
TRIM_MIN_OF_REFRAME = 0.5         # a trim that keeps less of the reframe crop is dropped for painting
HEAD_KEYS = ("format", "kind", "mode", "bits", "width", "height", "mcu", "orientation", "frames", "error",
             "header_q", "lossless_webp", "transparent", "stored")

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


# --- scan and decode ----------------------------------------------------------------

def file_key(path: Path) -> list[int]:
    """A file counts as the same while its size and modification time stay."""
    st = os.stat(path)
    return [st.st_size, st.st_mtime_ns]


def scan(job: Job) -> list[Item]:
    """The images to process, with their .txt captions: the job's file list,
    or the whole folder at any depth without the tools' output folders."""
    root = job.root
    items = []
    if job.files:
        for rel in job.files:
            p = root / rel
            if not p.is_file():
                emit({"warning": f"not a file: {rel}"})
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
            it.key = file_key(it.path)
        except OSError:
            it.key = None
    return items


def read_head(path: Path) -> dict:
    """The header of one image (remove_borders' read_header plus jpeg_cleanup's
    facts), without the EXIF and ICC bytes, so it can be cached as JSON."""
    head = rb.read_header(path)
    head["stored"] = [head["width"], head["height"]]
    head.setdefault("header_q", None)
    head.setdefault("lossless_webp", False)
    head.setdefault("transparent", False)
    if head["error"] or head["frames"] > 1:
        return head
    with Image.open(path) as im:
        facts = jc.facts_of(im, path)
        head["header_q"], head["lossless_webp"] = facts["header_q"], facts["lossless_webp"]
        if "A" in im.mode or (im.mode == "P" and "transparency" in im.info) or "transparency" in im.info:
            try:
                head["transparent"] = im.convert("RGBA").getchannel("A").getextrema()[0] < 255
            except Exception:  # noqa: BLE001
                head["transparent"] = True
    return head


def decode_image(path: Path, head: dict | None = None) -> tuple[dict, np.ndarray | None]:
    """The header and the upright RGB uint8 pixels of one image, transparency
    composited over white. Pixels None when the image is skipped."""
    head = head or read_head(path)
    if head["error"] or head["frames"] > 1:
        return head, None
    stored = rb.load_pixels(path)
    return head, np.ascontiguousarray(rb.to_upright(stored, head["orientation"]))


def upright_size(head: dict) -> tuple[int, int]:
    w, h = head["stored"]
    return (h, w) if head["orientation"] in (5, 6, 7, 8) else (w, h)


def exif_and_icc(path: Path) -> tuple:
    with Image.open(path) as im:
        return im.info.get("exif"), im.info.get("icc_profile")


# --- boxes ----------------------------------------------------------------------------

def intersect(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return [x0, y0, x1, y1] if x1 > x0 and y1 > y0 else None


def shift(box, dx, dy):
    return [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]


def area(box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def align_box(display_box, head: dict, inward_display_sides: set) -> tuple[list, list]:
    """The composed crop on the JPEG block grid, in stored pixels: the stored
    origin moves to the MCU grid, inward (so that a border or a watermark goes
    completely) on a side that a border cut or a trim set, outward on the
    others, as remove_borders and reframe do on their own; the far edges stay.
    inward_display_sides holds "L", "T", "R", "B" in upright terms.
    -> (display box, stored box); a non-JPEG gets its exact box in stored pixels."""
    o, (sw, sh) = head["orientation"], head["stored"]
    stored = rf.display_to_stored([int(v) for v in display_box], o, sw, sh)
    if not head.get("mcu"):
        return [int(v) for v in display_box], [int(v) for v in stored]
    mw, mh = head["mcu"]
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


# --- caches -----------------------------------------------------------------------------

class Caches:
    """Per-file caches in _backup/_pipeline/cache.json: the header and, per
    pass, the result under a signature of the pass's inputs. An entry holds
    while the file's size and time stay."""

    def __init__(self, root: Path):
        self.path = root / BACKUP_DIRNAME / RUN_DIRNAME / CACHE_NAME
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.files = data.get("files", {}) if data.get("version") == CACHE_VERSION else {}
        except (OSError, ValueError):
            self.files = {}
        self.dirty = False

    def entry(self, it: Item) -> dict | None:
        e = self.files.get(it.rel)
        return e if e and it.key and e.get("key") == it.key else None

    def get(self, it: Item, pass_name: str, sig: str):
        e = self.entry(it)
        return None if e is None else e.get(pass_name, {}).get(sig)

    def put(self, it: Item, pass_name: str, sig: str, value) -> None:
        if not it.key:
            return
        e = self.entry(it)
        if e is None:
            e = self.files[it.rel] = {"key": it.key}
        e.setdefault(pass_name, {})[sig] = value
        self.dirty = True

    def head(self, it: Item) -> dict:
        e = self.entry(it)
        if e is not None and e.get("head"):
            return dict(e["head"])
        head = {k: v for k, v in read_head(it.path).items() if k in HEAD_KEYS}
        if it.key:
            if e is None:
                e = self.files[it.rel] = {"key": it.key}
            e["head"] = head
            self.dirty = True
        return dict(head)

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


def emit(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


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


def free_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# --- the detection passes -------------------------------------------------------------

class Stages:
    """The options of the stages in the job, with their tools' defaults."""

    def __init__(self, job: Job):
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


def detection_pass(name: str, items: list, caches: Caches, signature, compute, threads: int,
                   in_threads: bool = False, unload=None, only=None) -> dict:
    """One pass over the images: the result of compute(it, head, array) cached
    under the pass name and signature (a string, or a function of the item).
    Images whose result is cached are not decoded; a pass with nothing to
    compute loads no model. compute runs in the main thread (a GPU model), or
    in the decoder threads with in_threads (CPU work). only: the items the
    pass applies to (default all). unload() is called at the end to drop the
    model. -> the pass summary."""
    t0 = time.perf_counter()
    chosen = items if only is None else only
    total = len(chosen)
    index = {it.rel: n for n, it in enumerate(chosen, 1)}
    sig_of = signature if callable(signature) else (lambda it: signature)
    todo, counts = [], {"computed": 0, "cached": 0, "skipped": 0, "failed": 0}
    for it in chosen:
        if it.key and caches.get(it, name, sig_of(it)) is not None:
            counts["cached"] += 1
            emit({"pass": name, "index": index[it.rel], "total": total, "path": it.rel, "cached": True})
        else:
            todo.append(it)

    def work(it: Item):
        try:
            head, array = decode_image(it.path, caches.head(it))
            if array is None:
                return it, head, None, None, ""
            value = compute(it, head, array) if in_threads else None
            return it, head, array, value, ""
        except Exception as e:  # noqa: BLE001
            return it, None, None, None, f"{type(e).__name__}: {e}"

    try:
        if todo:
            with ThreadPoolExecutor(max_workers=threads) as tp:
                for it, head, array, value, err in bounded_map(tp, work, todo, threads * 2):
                    line = {"pass": name, "index": index[it.rel], "total": total, "path": it.rel}
                    if err:
                        line["error"] = err
                        counts["failed"] += 1
                    elif array is None:
                        line["skipped"] = rb.skip_reason(head)
                        counts["skipped"] += 1
                    else:
                        try:
                            if not in_threads:
                                value = compute(it, head, array)
                            caches.put(it, name, sig_of(it), value)
                            counts["computed"] += 1
                        except Exception as e:  # noqa: BLE001 - one bad image is reported, not fatal
                            line["error"] = f"{type(e).__name__}: {e}"
                            counts["failed"] += 1
                    emit(line)
    finally:
        caches.save()
        if unload is not None:
            unload()
        free_gpu()
    summary = {"pass": name, "summary": True, "images": total, **counts, "seconds": round(time.perf_counter() - t0, 1)}
    emit(summary)
    return summary


def run_detection_passes(items: list, caches: Caches, stages: Stages, threads: int) -> list:
    summaries = []
    if stages.borders:
        dark = stages.borders["dark"]
        summaries.append(detection_pass(
            "borders", items, caches, sig(rb.DETECTOR_VERSION, dark),
            lambda it, head, array: {k: v for k, v in rb.plan(array, head, {"dark": dark})["result"].items() if k != "plan"},
            threads, in_threads=True))
    if stages.watermark:
        conf = stages.watermark["conf"]
        summaries.append(detection_pass(
            "watermark", items, caches, sig(wm.DETECTOR_VERSION, conf),
            lambda it, head, array: wm.detect(wm.detector_lazy()[1], array, conf),
            threads, unload=lambda: wm.unload(detector=True, lama=False)))
    if stages.reframe:
        def reframe_compute(it, head, array):
            p = rf.plan(array, {"format": head["format"], "orientation": 1, "stored": list(array.shape[1::-1])},
                        {"lossless": False})
            return p["det"]
        summaries.append(detection_pass(
            "reframe", items, caches, sig(rf.CACHE_VERSION, rf.models_signature(), rf.PEOPLE_VERSION),
            reframe_compute, threads, unload=rf.unload))
    if stages.faces:
        conf = stages.faces["conf"]
        summaries.append(detection_pass(
            "faces", items, caches, sig(conf),
            lambda it, head, array: fm.plan(array, None, {"conf": conf})["boxes"],
            threads, unload=fm.unload))
    if stages.cleanup:
        max_pixels = stages.cleanup["max_pixels"]

        def quality_compute(it, head, array):
            d = jc.decode_array(array, head, max_pixels)
            if d["skip"]:
                return {k: d.get(k) for k in ("format", "mode", "width", "height", "gray", "header_q", "skip")}
            return jc.measure_one(jc.model_lazy(), d)
        summaries.append(detection_pass(
            "quality", items, caches, sig(jc.MEASURE_VERSION, max_pixels), quality_compute, threads, unload=jc.unload))
    return summaries


# --- compose -----------------------------------------------------------------------------

def compose(it: Item, caches: Caches, stages: Stages) -> dict:
    """The plan of one image from the cached detections, no model: the final
    crop (upright and stored), the watermark boxes left inside it, the face
    boxes moved into it, the cleanup eligibility, and notes per stage."""
    head = caches.head(it)
    entry = {"path": it.rel, "changed": [], "notes": {}, "box": None, "stored_box": None, "wm_inside": [],
             "faces": None, "cleanup": None, "write": "none", "head": {k: head.get(k) for k in HEAD_KEYS}}
    if head["error"] or head["frames"] > 1:
        entry["skipped"] = rb.skip_reason(head)
        return entry
    W, H = upright_size(head)
    full = [0, 0, W, H]
    changed, notes = entry["changed"], entry["notes"]
    inward = set()
    # borders
    box = full
    if stages.borders:
        result = caches.get(it, "borders", sig(rb.DETECTOR_VERSION, stages.borders["dark"]))
        if result is None:
            notes["borders"] = {"action": "skip", "note": "not analysed"}
        else:
            result = dict(result)
            if result.get("cuts"):
                result["plan"] = rb.crop_plan(head, result, stages.borders["min_area"])
            action = rb.action_of(head, result)
            notes["borders"] = {"action": action, "cuts": rb.describe_cuts(result.get("cuts", [])),
                                "note": result.get("review") or result.get("plan", {}).get("note") or ""}
            entry["borders_result"] = result
            if action == "crop":
                bx = rb.stored_to_display(result["plan"]["box"], head["orientation"], *head["stored"])
                box = [max(0, bx[0]), max(0, bx[1]), min(W, bx[2]), min(H, bx[3])]
                inward |= {s for s, cut in zip("LTRB", (box[0] > 0, box[1] > 0, box[2] < W, box[3] < H)) if cut}
                changed.append("borders")
    # watermark boxes, in original coordinates
    wm_boxes, trim_box = [], None
    if stages.watermark:
        wm_boxes = caches.get(it, "watermark", sig(wm.DETECTOR_VERSION, stages.watermark["conf"])) or []
        wm_boxes = [list(b) for b in wm_boxes]
        if wm_boxes:
            notes["watermark"] = {"boxes": len(wm_boxes), "conf": max(b[4] for b in wm_boxes)}
            if stages.watermark["mode"] == "trim":
                mask = wm.boxes_mask((H, W), wm_boxes, stages.watermark["dilate"])
                rect = wm.largest_clear_rect(wm.mask_boxes(mask), W, H)
                rect = intersect(rect, box) if rect else None
                if rect and area(rect) >= stages.watermark["trim_min_keep"] * area(box):
                    trim_box = rect
        entry["wm_boxes"] = wm_boxes
    # the subject crop, planned on the original
    rf_box = None
    if stages.reframe:
        det = caches.get(it, "reframe", sig(rf.CACHE_VERSION, rf.models_signature(), rf.PEOPLE_VERSION))
        if det is not None and det.get("people_version") == rf.PEOPLE_VERSION:
            sel = rf.select_subject(det, stages.overrides.get(it.rel.lower()))
            crop = None
            if sel["verdict"] == "crop":
                crop = rf.plan_crop(det, sel, {"orientation": 1, "stored": [W, H]}, stages.reframe["ratios"], False)
            cropping = bool(crop) and crop["verdict"] == "crop"
            notes["reframe"] = {"verdict": "crop" if cropping else "keep",
                                "reason": sel["reason"] if cropping or not crop else crop["reason"],
                                "people": len(det["people"]), "flags": sel["flags"] + ((crop or {}).get("flags") or [])}
            entry["reframe"] = {"sel": sel, "crop": crop}
            if cropping:
                rf_box = intersect(crop["box"], box) or box
                notes["reframe"].update(family=crop["family"], size=crop["size"])
        else:
            notes["reframe"] = {"verdict": "keep", "reason": "not detected", "people": 0, "flags": []}
    # compose: inside the borders, then the subject crop, then the trim
    crop_box = box
    if rf_box:
        crop_box = rf_box
        changed.append("reframe")
    if trim_box:
        both = intersect(crop_box, trim_box)
        if both and area(both) >= TRIM_MIN_OF_REFRAME * area(crop_box):
            if both != crop_box:
                inward |= {s for s, cut in zip("LTRB", (both[0] > crop_box[0], both[1] > crop_box[1],
                                                        both[2] < crop_box[2], both[3] < crop_box[3])) if cut}
                crop_box = both
                changed.append("watermark")
            notes["watermark"]["trim"] = True
        else:
            trim_box = None                               # the trim would cut the subject: paint instead
    final, stored_box = align_box(crop_box, head, inward)
    final = intersect(final, full) or full
    if final == full:
        entry["box"], entry["stored_box"] = None, None
        for s_ in ("borders", "reframe"):
            if s_ in changed:
                changed.remove(s_)
    else:
        entry["box"], entry["stored_box"] = final, stored_box
    x0, y0, x1, y1 = final
    cw, ch = x1 - x0, y1 - y0
    # the watermark boxes still inside the crop, in crop coordinates
    if wm_boxes and not trim_box:
        inside = []
        for b in wm_boxes:
            cut = intersect(b[:4], final)
            if cut:
                inside.append(shift(cut, -x0, -y0) + [b[4]])
        if inside:
            entry["wm_inside"] = inside
            changed.append("watermark")
            notes.setdefault("watermark", {})["painted"] = len(inside)
    # the face boxes, moved into the crop
    if stages.faces:
        boxes = caches.get(it, "faces", sig(stages.faces["conf"]))
        moved = []
        for b in boxes or []:
            cut = intersect(b[:4], final)
            if cut:
                moved.append(shift(cut, -x0, -y0))
        entry["faces"] = {"boxes": moved, "size": [cw, ch],
                          **{k: v for k, v in stages.faces.items() if k != "conf"}}
        notes["face_masks"] = {"faces": len(moved)}
    # cleanup eligibility: the quality under the threshold, the crop within the limit
    if stages.cleanup:
        m = caches.get(it, "quality", sig(jc.MEASURE_VERSION, stages.cleanup["max_pixels"]))
        fits = cw * ch <= stages.cleanup["max_pixels"]
        if m is None:
            notes["jpeg_cleanup"] = {"action": "not measured"}
        elif m.get("skip") and not (m["skip"] == "large" and fits):
            notes["jpeg_cleanup"] = {"action": m["skip"], "qf": None}
        elif not fits:
            notes["jpeg_cleanup"] = {"action": "large", "qf": m.get("qf")}
        elif m.get("qf") is not None and m["qf"] >= stages.cleanup["threshold"]:
            notes["jpeg_cleanup"] = {"action": "keep", "qf": m["qf"]}
        else:
            entry["cleanup"] = {"m": None if m.get("skip") else m, "verdict": None}
            notes["jpeg_cleanup"] = {"action": "eligible", "qf": m.get("qf")}
    return entry


def cleanup_signature(entry: dict, stages: Stages) -> str:
    opts = {k: stages.cleanup[k] for k in ("threshold", "qf_offset", "min_block_drop", "min_qf_gain", "max_pixels")}
    return sig(jc.MEASURE_VERSION, entry["box"], opts, JPEG_QUALITY)


def finish_plan(entry: dict) -> None:
    """The write kind of an image once its cleanup verdict is known."""
    if entry.get("skipped"):
        return
    fix = bool(entry.get("cleanup") and (entry["cleanup"].get("verdict") or {}).get("action") == "fix")
    if fix and "jpeg_cleanup" not in entry["changed"]:
        entry["changed"].append("jpeg_cleanup")
    entry["write"] = "array" if entry["wm_inside"] or fix else "lossless" if entry["box"] else "none"


def write_plan(root: Path, job: Job, entries: dict) -> Path:
    path = root / BACKUP_DIRNAME / RUN_DIRNAME / PLAN_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    slim = {}
    for rel, e in entries.items():
        slim[rel] = {k: v for k, v in e.items() if k not in ("borders_result", "reframe", "wm_boxes", "head")}
        if e.get("cleanup"):
            slim[rel]["cleanup"] = {"qf": (e["cleanup"].get("m") or {}).get("qf"), "verdict": e["cleanup"].get("verdict")}
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps({"written": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": job.stages, "mode": job.mode,
                               "images": slim}, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return path


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

    def draw(self, it: Item, head: dict, array: np.ndarray, entry: dict, caches: Caches, stages: Stages) -> None:
        if entry.get("reframe"):
            det = caches.get(it, "reframe", sig(rf.CACHE_VERSION, rf.models_signature(), rf.PEOPLE_VERSION))
            if det is not None:
                self._submit(rf.draw_verdict, Image.fromarray(array), det, entry["reframe"]["sel"],
                             self.dir / "reframe" / (it.rel + ".jpg"), entry["reframe"]["crop"])
        if entry.get("wm_boxes"):
            trim = entry["box"] if entry["notes"].get("watermark", {}).get("trim") else None
            plan_ = {"boxes": entry["wm_boxes"], "dilate": stages.watermark["dilate"], "trim_box": trim}
            self._submit(wm.draw_preview, array, plan_, self.dir / "watermark" / (it.rel + ".jpg"))
        if entry.get("borders_result") and entry["notes"]["borders"]["action"] in ("crop", "review", "too small"):
            stored = Image.fromarray(rb.to_stored(array, head["orientation"]))
            label = f"{it.rel}  {head['width']}x{head['height']}"
            bp = _ItemLike(entry["notes"]["borders"]["action"], entry["borders_result"])
            self.border_tiles.append((rb.sheet_group(bp), self.pool.submit(rb.tile_of, stored, entry["borders_result"], label)))

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

    def __init__(self, action: str, result: dict):
        self.action, self.result = action, result


# --- writing ---------------------------------------------------------------------------------

def same_bytes(a: Path, b: Path) -> bool:
    try:
        return a.stat().st_size == b.stat().st_size and filecmp.cmp(a, b, shallow=False)
    except OSError:
        return False


def make_writable(path: Path) -> None:
    if path.exists() and not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)


def copy_keep(src: Path, dst: Path) -> str:
    """Copy src to dst unless dst exists: a backup is the one untouched
    original and is never overwritten. -> copied, same or kept."""
    if dst.exists():
        return "same" if same_bytes(src, dst) else "kept"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return "copied"


def output_name(rel: str, head: dict, write: str, save_png: bool) -> str:
    """The relative path of the image written: the same, or .png."""
    if write == "array":
        fmt = "JPEG" if head["format"] in ("JPEG", "MPO") and not save_png else "PNG"
        return rel if fmt == "JPEG" else Path(rel).with_suffix(".png").as_posix()
    if write == "lossless" and head["kind"] == rb.KIND_LOSSY:
        return Path(rel).with_suffix(".png").as_posix()
    return rel


def write_array(array: np.ndarray, dst: Path, fmt: str, exif: bytes | None, icc: bytes | None, src_stat) -> None:
    im = Image.fromarray(np.ascontiguousarray(array))
    exif = rf.exif_for_output(exif, im.width, im.height, keep_orientation=False)
    kw = {k: v for k, v in (("exif", exif), ("icc_profile", icc)) if v}
    tmp = rb.part_path(dst)
    if fmt == "JPEG":
        im.save(tmp, "JPEG", quality=JPEG_QUALITY, subsampling=0, optimize=True, **kw)
    else:
        im.save(tmp, "PNG", compress_level=PNG_LEVEL, **kw)
    rb.finish_write(tmp, dst, src_stat)


def write_job(job: dict) -> dict:
    """One image's writes, in a worker process: the backup (in place, once,
    before the first byte changes), the image, the mask. -> the log record."""
    rec = {"op": "done", "rel": job["rel"], "changed": job["changed"], "write": job["write"]}
    root, src, dst = Path(job["root"]), Path(job["src"]), Path(job["dst"])
    in_place = job["mode"] == "in_place"
    try:
        head = job["head"]
        st = os.stat(src)
        if in_place and job["write"] != "none":
            rec["backup"] = copy_keep(src, root / BACKUP_DIRNAME / job["rel"])
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
            exif, icc = exif_and_icc(src)
            write_array(job["array"], dst, job["fmt"], exif, icc, st)
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
        # the writes are atomic, so the image in place is intact: a backup this
        # job copied for it is not needed and goes again
        try:
            bk = root / BACKUP_DIRNAME / job["rel"]
            if rec.get("backup") == "copied" and src.is_file() and same_bytes(src, bk):
                bk.unlink()
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
    """Return the images of the last run to their originals: in place, every
    image written from _backup (the one untouched original, so an earlier
    run's change goes too), a renamed output removed, the masks restored or
    removed; in parallel mode, the files written into the target removed.
    A backup this run made is removed; one that was there before stays. -> counts."""
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
            bk = root / BACKUP_DIRNAME / intent["rel"]
            if not bk.exists():
                counts["no backup"] += 1
                emit({"warning": f"no backup for {intent['rel']}"})
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
            if rec.get("mask_backup") in ("copied", "same", "kept") and mb.exists():
                make_writable(m)
                tmp = rb.part_path(m)
                shutil.copy2(mb, tmp)
                os.replace(tmp, m)
                if rec["mask_backup"] == "copied":
                    mb.unlink()
                counts["restored"] += 1
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


# --- the apply pass ---------------------------------------------------------------------------

def restore_array(array: np.ndarray, head: dict, m: dict, opts: dict) -> np.ndarray:
    """The FBCNN restoration of upright pixels, as jpeg_cleanup's fix_one
    makes it: the colour model told the image's own QF plus the offset, on
    the stored pixels; a gray image keeps the luma only."""
    d = jc.decode_array(array, head, 1 << 62)
    model = jc.model_lazy()
    qf = m.get("qf_color", m["qf"])
    y = model.restore("color", d["rgb"], min(100.0, qf + opts["qf_offset"]))
    if d["gray"]:
        y = np.asarray(Image.fromarray(y).convert("L"))
    return np.ascontiguousarray(rb.to_upright(jc.as_rgb(y), d["orientation"]))


def apply_pass(job: Job, items: list, entries: dict, caches: Caches, stages: Stages, run_id: str, log: RunLog,
               report) -> dict:
    """Decode once, crop, paint, restore, write once, mask: the images whose
    plan changes them, and in parallel mode every image. -> counts"""
    t0 = time.perf_counter()
    root = job.root
    base = root if job.mode == "in_place" else job.target
    chosen = []
    for it in items:
        e = entries[it.rel]
        if e.get("skipped"):
            continue
        if job.mode == "parallel" or e["write"] != "none" or e["faces"] is not None:
            chosen.append(it)
    total = len(chosen)
    index = {it.rel: n for n, it in enumerate(chosen, 1)}
    counts = {"changed": 0, "unchanged": 0, "skipped": 0, "failed": 0, "masks": 0}

    def finish(rec: dict, line: dict) -> None:
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

    def decoded(it: Item):
        try:
            head, array = decode_image(it.path, caches.head(it))
            return it, head, array, ""
        except Exception as e:  # noqa: BLE001
            return it, None, None, f"{type(e).__name__}: {e}"

    pool = ProcessPoolExecutor(max_workers=job.threads, initializer=ignore_ctrl_c)
    pending = []
    try:
        with ThreadPoolExecutor(max_workers=job.threads) as tp:
            for it, head, array, err in bounded_map(tp, decoded, chosen, job.threads * 2):
                e = entries[it.rel]
                line = {"pass": "apply", "index": index[it.rel], "total": total, "path": it.rel,
                        "changed": list(e["changed"]), "notes": e["notes"], "write": e["write"]}
                if e["box"]:
                    line["box"] = e["box"]
                if err or array is None:
                    line["error"] = err or "could not be decoded"
                    counts["failed"] += 1
                    report.write(json.dumps(line, ensure_ascii=False) + "\n")
                    emit(line)
                    continue
                try:
                    x0, y0, x1, y1 = e["box"] or [0, 0, array.shape[1], array.shape[0]]
                    out = np.ascontiguousarray(array[y0:y1, x0:x1])
                    if e["wm_inside"]:
                        out = wm.inpaint(out, e["wm_inside"], stages.watermark["dilate"], stages.watermark["max_size"])
                    verdict = (e.get("cleanup") or {}).get("verdict") or {}
                    if verdict.get("action") == "fix":
                        out = restore_array(out, head, e["cleanup"]["m"], stages.cleanup)
                    mask = fm.apply(out, e["faces"]) if e["faces"] is not None else None
                except Exception as ex:  # noqa: BLE001 - one bad image is reported, not fatal
                    line["error"] = f"{type(ex).__name__}: {ex}"
                    counts["failed"] += 1
                    report.write(json.dumps(line, ensure_ascii=False) + "\n")
                    emit(line)
                    continue
                write = e["write"]
                out_rel = output_name(it.rel, head, write, job.save_png)
                dst = base / out_rel
                if write != "none" and job.mode == "in_place" and dst != it.path and dst.exists():
                    line["skipped"] = f"{out_rel} exists already; the image is left as it is"
                    line["changed"] = []
                    counts["skipped"] += 1
                    report.write(json.dumps(line, ensure_ascii=False) + "\n")
                    emit(line)
                    continue
                mask_path = None
                if mask is not None:
                    mask_path = base / Path(out_rel).parent / MASKS_DIRNAME / (Path(out_rel).stem + ".png")
                log.write({"op": "intent", "rel": it.rel, "write": write, "out": out_rel,
                           "sidecars": [sc.relative_to(root).as_posix() for sc in it.sidecars],
                           "mask": mask_path.relative_to(base).as_posix() if mask_path else ""})
                wjob = {"root": str(root), "target": str(job.target) if job.target else "", "rel": it.rel,
                        "src": str(it.path), "dst": str(dst), "head": {k: v for k, v in head.items() if k in HEAD_KEYS},
                        "mode": job.mode, "write": write, "changed": list(e["changed"]), "stored_box": e["stored_box"],
                        "sidecars": [str(sc) for sc in it.sidecars],
                        "fmt": "JPEG" if out_rel.lower().endswith((".jpg", ".jpeg", ".jpe", ".jfif")) else "PNG",
                        "array": out if write == "array" else None, "mask": mask,
                        "mask_path": str(mask_path) if mask_path else ""}
                if write == "lossless" and head["kind"] == rb.KIND_JPEG and not head.get("mcu"):
                    wjob.update(write="array", array=out)        # a JPEG without a known block size
                    line["write"] = "array"
                pending.append((line, pool.submit(write_job, wjob)))
                while len(pending) > job.threads * 2:
                    line0, fut = pending.pop(0)
                    finish(fut.result(), line0)
        for line0, fut in pending:
            finish(fut.result(), line0)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        wm.unload()
        jc.unload()
        free_gpu()
    summary = {"pass": "apply", "summary": True, "images": total, **counts, "seconds": round(time.perf_counter() - t0, 1)}
    emit(summary)
    return counts


# --- the run ---------------------------------------------------------------------------------

def run(job: Job) -> int:
    root = job.root
    t0 = time.perf_counter()
    items = scan(job)
    caches = Caches(root)
    stages = Stages(job)
    passes = run_detection_passes(items, caches, stages, job.threads)

    # compose, no model
    t1 = time.perf_counter()
    entries, counts = {}, {"planned": 0, "unchanged": 0, "skipped": 0}
    for n, it in enumerate(items, 1):
        e = compose(it, caches, stages)
        entries[it.rel] = e
        line = {"pass": "compose", "index": n, "total": len(items), "path": it.rel}
        if e.get("skipped"):
            line["skipped"] = e["skipped"]
            counts["skipped"] += 1
        else:
            line.update(changed=e["changed"], notes=e["notes"])
            if e["box"]:
                line["box"] = e["box"]
            counts["planned" if e["changed"] else "unchanged"] += 1
        emit(line)
    caches.save()
    emit({"pass": "compose", "summary": True, "images": len(items), **counts, "seconds": round(time.perf_counter() - t1, 1)})

    # the cleanup benefit, FBCNN on the composed crop of the eligible images
    previews = Previews(root, job.threads) if job.dry_run else None
    if stages.cleanup:
        eligible = [it for it in items if entries[it.rel].get("cleanup")]
        opts = {k: stages.cleanup[k] for k in ("threshold", "qf_offset", "min_block_drop", "min_qf_gain", "max_pixels")}

        def benefit_compute(it, head, array):
            e = entries[it.rel]
            x0, y0, x1, y1 = e["box"] or [0, 0, array.shape[1], array.shape[0]]
            crop = np.ascontiguousarray(array[y0:y1, x0:x1])
            # the crop's own stored size decides the size limit, not the original's
            cw, ch = x1 - x0, y1 - y0
            head_crop = dict(head, width=ch if head["orientation"] in (5, 6, 7, 8) else cw,
                             height=cw if head["orientation"] in (5, 6, 7, 8) else ch)
            p = jc.plan(crop, head_crop, {**opts, "quality": JPEG_QUALITY, "m": e["cleanup"]["m"],
                                          "sheet": previews is not None})
            if previews is not None and p.get("outs") is not None:
                previews.cleanup_tile(it, head, p)
            numbers = {k: v for k, v in p["fix"].items() if k in ("benefit", "qf_used", "qf_after", "change",
                                                                   "block_before", "block_after")}
            return {"action": p["action"], "fix": numbers, "m": p["m"]}
        for it in eligible:
            s_ = cleanup_signature(entries[it.rel], stages)
            # the benefit of a cached verdict is still drawn on the sheets of a dry run
            if previews is not None and caches.get(it, "benefit", s_) is not None:
                caches.files[it.rel]["benefit"].pop(s_, None)
        passes.append(detection_pass(
            "benefit", items, caches, lambda it: cleanup_signature(entries[it.rel], stages), benefit_compute,
            job.threads, unload=jc.unload, only=eligible) if eligible else {"pass": "benefit", "summary": True, "images": 0})
        for it in eligible:
            e = entries[it.rel]
            e["cleanup"]["verdict"] = caches.get(it, "benefit", cleanup_signature(e, stages))
            if e["cleanup"]["verdict"]:
                e["cleanup"]["m"] = e["cleanup"]["verdict"].get("m") or e["cleanup"]["m"]
                e["notes"]["jpeg_cleanup"] = {"action": e["cleanup"]["verdict"]["action"], "qf": (e["cleanup"]["m"] or {}).get("qf"),
                                              **{k: v for k, v in e["cleanup"]["verdict"]["fix"].items()
                                                 if k in ("benefit", "qf_after", "block_before", "block_after")}}
    for e in entries.values():
        finish_plan(e)
    plan_path = write_plan(root, job, entries)
    planned = sum(1 for e in entries.values() if e.get("changed"))

    if job.dry_run:
        # the previews need the pixels once more
        def draw(it: Item):
            e = entries[it.rel]
            if e.get("skipped") or not (e.get("reframe") or e.get("wm_boxes") or e.get("borders_result")):
                return it, ""
            try:
                head, array = decode_image(it.path, caches.head(it))
                if array is not None:
                    previews.draw(it, head, array, e, caches, stages)
                return it, ""
            except Exception as ex:  # noqa: BLE001
                return it, f"{type(ex).__name__}: {ex}"
        with ThreadPoolExecutor(max_workers=job.threads) as tp:
            for it, err in bounded_map(tp, draw, items, job.threads * 2):
                if err:
                    emit({"pass": "previews", "path": it.rel, "error": err})
        drawn = previews.finish()
        summary = {"summary": True, "dry_run": True, "images": len(items), "changed": planned,
                   "unchanged": len(items) - planned - counts["skipped"], "skipped": counts["skipped"],
                   "plan": str(plan_path), "previews": str(previews.dir), **drawn,
                   "seconds": round(time.perf_counter() - t0, 1), "mode": job.mode}
        summary.update(peak_vram())
        emit(summary)
        return 0

    # the apply pass
    run_id = time.strftime("%Y%m%d-%H%M%S")
    taken = {e["run"] for e in read_log(root) if "run" in e}
    while run_id in taken:                        # two runs within one second
        run_id += "x"
    log = RunLog(root)
    log.write({"run": run_id, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": job.mode,
               "target": str(job.target) if job.target else "", "stages": job.stages, "save_png": job.save_png,
               "images": len(items)})
    report_path = root / BACKUP_DIRNAME / RUN_DIRNAME / REPORT_NAME
    report = open(report_path, "w", encoding="utf-8")
    done = {"changed": 0, "unchanged": 0, "skipped": 0, "failed": 0, "masks": 0}
    try:
        done = apply_pass(job, items, entries, caches, stages, run_id, log, report)
    finally:
        log.write({"end": run_id, **done})
        log.close()
        report.close()
    skipped = done["skipped"] + counts["skipped"]
    summary = {"summary": True, "dry_run": False, "images": len(items), **done, "skipped": skipped,
               "unchanged": len(items) - done["changed"] - done["failed"] - skipped,
               "seconds": round(time.perf_counter() - t0, 1), "mode": job.mode, "report": str(report_path),
               "plan": str(plan_path)}
    summary.update(peak_vram())
    emit(summary)
    return 1 if done["failed"] else 0


def peak_vram() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {"peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20)}
    except Exception:  # noqa: BLE001
        pass
    return {}


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace", encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Run the repair tools over a dataset from a job file, one model at a time.")
    ap.add_argument("--job", required=False, metavar="FILE", help="the job as JSON (see --example)")
    ap.add_argument("--dry-run", action="store_true", help="the detection passes, the plan and the previews; change nothing")
    ap.add_argument("--undo", action="store_true", help="return the images of the last run of the job's folder to their originals")
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
