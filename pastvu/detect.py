"""What is in the picture: box statistics for the saved photos (the detect command).

YOLOE-26l with text prompts (a frozen vocabulary in four families) and the
adetailer face model run over the saved file of every photo of a target, in
batches, and one row per photo goes into the store's boxes table: the counted
persons, vehicles, objects and things with the size of the largest, the faces
inside persons, and every stored box as JSON. cherrypick --interest reads the
derived columns and never the names: the detector's fine names are nets, not
labels (every tram came back as bus, a missile as submarine), so a rule reads
the family of a box and its size.

A row is marked with the hash of the vocabulary. A changed vocabulary makes
the old rows stale, and the next run lists them again. The prompt embeddings of
the vocabulary are made once (the text encoder, 254 MB, is fetched for that)
and saved in models/ as yoloe-26l-<vocab>.npz; every later run loads them
without the encoder.

Coordinates are shares of the saved image, EXIF-upright: x1, y1, x2, y2 in 0..1.
A counted box has confidence >= COUNT_CONF and covers less than MAX_AREA of the
frame (rare words produce full-frame false positives); boxes from STORE_CONF up
are kept in the JSON, so a threshold between the two is a query later. A face
counts only when at least FACE_INSIDE of its box lies inside a counted person
(the face model boxed a window ornament once).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
REFRAME_MODELS = HERE.parent / "reframe" / "models"       # the face model is shared with reframe
DETECTOR = "yoloe-26l-seg.pt"                              # ultralytics assets
DETECTOR_ID = "yoloe-26l"                                  # what ultralytics binds prompt embeddings to
TEXT_ENCODER = "mobileclip2_b.ts"                          # ultralytics assets, needed once per vocabulary
FACE = "face_yolov8m.pt"
FACE_REPO = "Bingsu/adetailer"
MODEL_NAME = "yoloe-26l+face_yolov8m"                      # the model column of the rows

# The vocabulary, frozen: a change makes a new hash and the rows of the old one stale. Superordinate
# words (vehicle, transport, furniture, machine, railway, statue) almost never fire on these photos,
# concrete basic nouns do; the rules read the family and the box size, never the name.
VOCAB = {
    "people": ["person"],
    "vehicle": ["car", "bus", "truck", "train", "boat", "aircraft", "helicopter", "rocket", "submarine"],
    "object": ["chair", "counter", "chandelier", "bed", "display case", "sign", "poster", "flag", "monument",
               "satellite dish", "robot"],
    "context": ["building"],
}
NAMES = [n for v in VOCAB.values() for n in v]
FAMILY = {n: fam for fam, v in VOCAB.items() for n in v}
THING_FAMILIES = ("vehicle", "object")                     # max_thing_area: any family but people and context

STORE_CONF = 0.15        # boxes kept in the JSON
COUNT_CONF = 0.25        # boxes that count in the derived columns
FACE_CONF = 0.30
MAX_AREA = 0.90          # a counted box covers less of the frame than this
FACE_INSIDE = 0.80       # share of a face box inside a counted person for the face to count
IMGSZ = 640
BATCH = 16
DECODERS = 4


def vocab_hash(vocab: dict | None = None) -> str:
    """The first 8 hex digits of the sha1 of the lines family:name, sorted."""
    vocab = VOCAB if vocab is None else vocab
    lines = sorted(f"{fam}:{name}" for fam, names in vocab.items() for name in names)
    return hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest()[:8]


# --- models --------------------------------------------------------------------------------

def model_path(name: str, download: bool = True) -> Path:
    """Where a weights file is: models/ next to this file. The face model is copied from
    reframe's models/ or the Hugging Face cache when it is there, else downloaded into the
    cache first; the detector comes from the ultralytics assets. With the files present a run
    makes no network request."""
    path = MODELS_DIR / name
    if path.exists():
        return path
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    if name == FACE:
        shared = REFRAME_MODELS / FACE
        if shared.exists():
            shutil.copyfile(shared, tmp)
            tmp.replace(path)
            return path
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import LocalEntryNotFoundError
        try:
            cached = hf_hub_download(FACE_REPO, name, local_files_only=True)
        except LocalEntryNotFoundError:
            if not download:
                raise FileNotFoundError(f"{path} is missing; run install.bat or detect.py --fetch-models")
            print(f"downloading {name} from {FACE_REPO}", flush=True)
            cached = hf_hub_download(FACE_REPO, name)
        shutil.copyfile(cached, tmp)
        tmp.replace(path)
        return path
    if not download:
        raise FileNotFoundError(f"{path} is missing; run install.bat or detect.py --fetch-models")
    print(f"downloading {name} from the ultralytics assets", flush=True)
    from ultralytics.utils.downloads import attempt_download_asset
    attempt_download_asset(str(path))
    if not path.exists():
        raise RuntimeError(f"download of {name} failed")
    return path


def prompt_file(vocab: str | None = None) -> Path:
    return MODELS_DIR / f"{DETECTOR_ID}-{vocab or vocab_hash()}.npz"


@dataclass
class Models:
    det: object
    face: object
    vocab: str
    imgsz: int


def load_models(imgsz: int = IMGSZ, download: bool = True, log=print) -> Models:
    """The detector with the vocabulary's prompt embeddings, and the face model. The
    embeddings are loaded from models/ when they were saved before; else the text encoder
    is fetched into models/ (ultralytics downloads it into the working directory, so the
    working directory is models/ for that call), the embeddings are made and saved."""
    from ultralytics import YOLO, YOLOE
    det = YOLOE(str(model_path(DETECTOR, download)))
    vocab = vocab_hash()
    npz = prompt_file(vocab)
    if npz.exists():
        det.load_prompt_embeddings(npz)
    else:
        if not download and not (MODELS_DIR / TEXT_ENCODER).exists():
            raise FileNotFoundError(f"{npz} is missing and so is the text encoder; run install.bat or "
                                    f"detect.py --fetch-models")
        log(f"making the prompt embeddings of vocabulary {vocab} ({len(NAMES)} names) -> {npz.name}")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cwd = os.getcwd()
        os.chdir(MODELS_DIR)
        try:
            pe = det.get_text_pe(NAMES)
        finally:
            os.chdir(cwd)
        det.set_classes(NAMES, pe)
        det.save_prompt_embeddings(npz)
    if list(det.names.values()) != NAMES:
        raise RuntimeError(f"{npz.name} names the classes {list(det.names.values())}, not the vocabulary")
    face = YOLO(str(model_path(FACE, download)))
    return Models(det, face, vocab, imgsz)


# --- one image -----------------------------------------------------------------------------

def load_image(path: Path, imgsz: int = IMGSZ) -> Image.Image | None:
    """The photo EXIF-upright, RGB, reduced to imgsz on the long side (never enlarged: the
    detector's own letterbox scales a small image up, as the probes had it). None when the
    file is gone (a reclassify moved it between the listing and the read)."""
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
    except (FileNotFoundError, OSError):
        return None
    if max(im.size) > imgsz:
        im.thumbnail((imgsz, imgsz))
    return im


def _inside(face, person) -> float:
    """The share of the face box inside the person box."""
    fx1, fy1, fx2, fy2 = face[1:5]
    px1, py1, px2, py2 = person[2:6]
    iw = max(0.0, min(fx2, px2) - max(fx1, px1))
    ih = max(0.0, min(fy2, py2) - max(fy1, py1))
    area = (fx2 - fx1) * (fy2 - fy1)
    return iw * ih / area if area > 0 else 0.0


def stats_of(boxes: list, faces: list, count_conf: float = COUNT_CONF, max_area: float = MAX_AREA) -> dict:
    """The derived columns from a photo's boxes ([name, conf, x1, y1, x2, y2] as shares) and faces
    ([conf, x1, y1, x2, y2]): the counted persons and the tallest, the faces inside persons and the
    tallest, the counted vehicles and the largest with its width, the largest object, the largest
    thing of any family but people and context."""
    def area(b):
        return (b[4] - b[2]) * (b[5] - b[3])

    counted = [b for b in boxes if b[1] >= count_conf and area(b) < max_area]
    persons = [b for b in counted if FAMILY.get(b[0]) == "people"]
    vehicles = [b for b in counted if FAMILY.get(b[0]) == "vehicle"]
    objects = [b for b in counted if FAMILY.get(b[0]) == "object"]
    things = [b for b in counted if FAMILY.get(b[0]) in THING_FAMILIES]
    inside = [f for f in faces if any(_inside(f, p) >= FACE_INSIDE for p in persons)]
    big_v = max(vehicles, key=area, default=None)
    return {
        "n_person": len(persons),
        "max_person_h": round(max((b[5] - b[3] for b in persons), default=0.0), 4),
        "n_face": len(inside),
        "max_face_h": round(max((f[4] - f[2] for f in inside), default=0.0), 4),
        "n_vehicle": len(vehicles),
        "max_vehicle_area": round(area(big_v), 4) if big_v else 0.0,
        "max_vehicle_w": round(big_v[4] - big_v[2], 4) if big_v else 0.0,
        "max_object_area": round(max((area(b) for b in objects), default=0.0), 4),
        "max_thing_area": round(max((area(b) for b in things), default=0.0), 4),
    }


def boxes_json(boxes: list, faces: list) -> str:
    return json.dumps({"b": boxes, "f": faces}, separators=(",", ":"))


def detect_batch(models: Models, images: list[Image.Image]) -> list[tuple[list, list]]:
    """(boxes, faces) per image: the detector's boxes from STORE_CONF up and the face model's
    from FACE_CONF up, both as shares of the image, four decimals."""
    if not images:
        return []
    out = []
    det = models.det.predict(images, imgsz=models.imgsz, conf=STORE_CONF, quantize=16, verbose=False)
    fac = models.face.predict(images, imgsz=models.imgsz, conf=FACE_CONF, quantize=16, verbose=False)
    for r, fr in zip(det, fac):
        boxes = []
        for xyxy, conf, cls in zip(r.boxes.xyxyn.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            boxes.append([r.names[int(cls)], round(conf, 3)] + [round(v, 4) for v in xyxy])
        faces = [[round(conf, 3)] + [round(v, 4) for v in xyxy]
                 for xyxy, conf in zip(fr.boxes.xyxyn.tolist(), fr.boxes.conf.tolist())]
        out.append((boxes, faces))
    return out


def row_of(cid: int, models: Models, boxes: list, faces: list, now: float | None = None) -> dict:
    """One boxes row (store.BOX_COLS) of a photo."""
    return {"cid": cid, "model": MODEL_NAME, "vocab": models.vocab, "imgsz": models.imgsz,
            "time": time.time() if now is None else now, **stats_of(boxes, faces), "json": boxes_json(boxes, faces)}


# --- the run over saved photos -------------------------------------------------------------

def select(store, scope: str, prefix: str, vocab: str, redo: bool = False) -> tuple[list[tuple[int, str]], dict]:
    """(cid, path) of the photos of a scope under a folder prefix (relative to the base) that need
    a boxes row of this vocabulary, and the counts for the listing line."""
    rows = store.boxes_pending(scope, prefix, vocab, redo)
    counts = store.boxes_count(scope, vocab)
    return rows, {"saved": counts["saved"], "boxed before": counts["boxed"], "to box": len(rows)}


def report(stats: dict, total: int, elapsed: float, vram=None, final: bool = False) -> str:
    n = stats["boxed"]
    rate = n / elapsed if elapsed > 0 else 0.0
    lines = [f"{'Finished: ' if final else ''}{n:,} of {total:,} photos boxed in {_hms(elapsed)} ({rate:.1f} per second)"]
    if n:
        lines.append(f"  with a person {stats['person'] / n:.0%}, a face {stats['face'] / n:.0%}, "
                     f"a vehicle {stats['vehicle'] / n:.0%}, a thing {stats['thing'] / n:.0%}")
    if stats["missing"]:
        lines.append(f"  image missing: {stats['missing']:,} (listed again by the next run)")
    if vram is not None and getattr(vram, "ok", False):
        lines.append("  " + vram.report())
    return "\n".join(lines)


def _hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min {s:02d} s" if m else f"{s} s"


FOLLOW_EVERY = 30        # --follow: seconds between two looks for new photos


def run(store, base: Path, targets: list[tuple[int, str]], models: Models, display, report_every: int = 500,
        vram=None, batch: int = BATCH, decoders: int = DECODERS, follow=None) -> dict:
    """Box the targets ((cid, path), in order) in batches: a pool decodes the next batch while the
    detector runs this one, and a batch's rows go into the store in one transaction, so a stopped
    run continues with the rest. The path of each photo is asked from the store at decode time (a
    reclassify may have moved it); a photo whose file is gone is counted and skipped, and no row is
    written for it, so the next run lists it again. follow: a function giving the targets to add
    when the list is done, asked every FOLLOW_EVERY s until Ctrl+C (None: stop when the list is
    done); a photo tried once in this run is not tried again. The first Ctrl+C finishes the batch
    in hand, the second stops at once. -> the counts, and "stopped" ("Ctrl+C" or None)."""
    from collections import deque
    from autocaption import _Interrupt
    stats = {"boxed": 0, "missing": 0, "person": 0, "face": 0, "vehicle": 0, "thing": 0}
    tried = {cid for cid, _ in targets}
    queue_ = deque(targets)
    total = len(targets)
    t_start = time.time()
    stopped = None

    def decode(item):
        cid, path = item
        st = store.status(cid)
        rel = st["path"] if st and st.get("path") else path
        return cid, rel, load_image(base / rel, models.imgsz)

    def submit():
        return [pool.submit(decode, queue_.popleft()) for _ in range(min(batch, len(queue_)))]

    pool = ThreadPoolExecutor(max(1, decoders))
    try:
        with _Interrupt() as intr:
            ahead = submit()
            while True:
                if intr.stop:
                    stopped = "Ctrl+C"
                    break
                if not ahead:
                    if follow is None:
                        break
                    display.set(f"{stats['boxed']:,} boxed; caught up with the download, waiting for new photos")
                    deadline = time.time() + FOLLOW_EVERY
                    while time.time() < deadline and not intr.stop:
                        time.sleep(0.5)
                    if intr.stop:
                        stopped = "Ctrl+C"
                        break
                    new = [(c, p) for c, p in follow() if c not in tried]
                    tried.update(c for c, _ in new)
                    queue_.extend(new)
                    total += len(new)
                    if new:
                        display.add_total(len(new))
                    ahead = submit()
                    continue
                decoded = [f.result() for f in ahead]
                ahead = submit()
                loaded = [(cid, rel, im) for cid, rel, im in decoded if im is not None]
                for cid, rel, im in decoded:
                    if im is None:
                        stats["missing"] += 1
                        display.line({"cid": cid, "state": "missing", "path": rel})
                if loaded:
                    display.set(f"{loaded[0][0]}: boxing {loaded[0][1]} (+{len(loaded) - 1})")
                    results = detect_batch(models, [im for _, _, im in loaded])
                    now = time.time()
                    rows = [row_of(cid, models, boxes, faces, now)
                            for (cid, _, _), (boxes, faces) in zip(loaded, results)]
                    store.save_boxes(rows)
                    for row in rows:
                        stats["boxed"] += 1
                        stats["person"] += row["n_person"] > 0
                        stats["face"] += row["n_face"] > 0
                        stats["vehicle"] += row["n_vehicle"] > 0
                        stats["thing"] += row["max_thing_area"] > 0
                        display.line({"cid": row["cid"], "state": "boxed", **{k: row[k] for k in (
                            "n_person", "max_person_h", "n_face", "max_face_h", "n_vehicle", "max_vehicle_area",
                            "max_thing_area")}})
                for cid, _, im in decoded:
                    display.advance(saved=im is not None)
                if report_every and loaded and \
                        stats["boxed"] // report_every != (stats["boxed"] - len(loaded)) // report_every:
                    display.print(report(stats, total, time.time() - t_start, vram))
    except KeyboardInterrupt:
        stopped = "Ctrl+C"
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    display.print(report(stats, total, time.time() - t_start, vram, final=True))
    if stopped:
        display.print(f"Stopped: {stopped}. Run the same command again to continue.")
    stats["stopped"] = stopped
    return stats


# --- install ---------------------------------------------------------------------------------

def fetch_models(log=print) -> None:
    """Download the weights into models/ and make the prompt embeddings of the vocabulary
    (install.bat; the text encoder is fetched for that once)."""
    for name in (DETECTOR, FACE):
        log(f"{name}: {model_path(name, download=True)}")
    m = load_models(download=True, log=log)
    log(f"prompt embeddings of vocabulary {m.vocab}: {prompt_file(m.vocab)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="the models of the detect command")
    ap.add_argument("--fetch-models", action="store_true",
                    help="download the detector and the face model into models/ and make the prompt embeddings")
    a = ap.parse_args()
    if a.fetch_models:
        fetch_models()
        print("models load OK")
    else:
        ap.print_help()
        sys.exit(2)
