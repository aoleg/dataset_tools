"""A replica base: a pick shared by `share` (share.py), rebuilt from PastVu by `run`.

The base holds only pastvu.db: the store's own tables for the picked photos (their
records, boxes, NIQE, hashes) and the `replica` table, which names each photo's
path in the pick, the sidecars it had there (captions), the sha256 of the original
download and of our saved file, and the outcome. meta: kind "replica" and
replica_format.

`run` on such a base downloads each pending photo's original and replays what the
first run did, from the record alone: the band rows, the downscale (from, to, and
whether a later reprocess made it), the crop box on the JPEG grid, and the painted
watermark boxes with their dilate. So no detector, no border analysis and no
classifier is loaded; LaMa only for the photos that had a watermark painted.

- The original is fetched by its recorded file name. Its sha256 equal to ours: the
  replay. Otherwise PastVu is asked for the record: a photo gone or no longer
  public is skipped with that reason; a new file with the same picture size
  (orientation 1) is replayed with its own band rows (outcome "original changed");
  anything else is skipped as changed on PastVu.
- The CPU work (decode, crop, downscale, encode) runs in worker processes; a
  painted photo is replayed in its download thread, with LaMa behind a lock.
- A photo is written like a download: its path is claimed (state writing), the
  image and its sidecars are written, then the status and the outcome are
  committed in one transaction. A photo left in state writing by a killed run is
  deleted and done again.
- The outcome says how close the file is to ours: identical (the same sha256: the
  lossless crops), rebuilt (encoded anew: a downscale or a paint), or original
  changed.
"""

from __future__ import annotations

import hashlib
import io
import json
import signal
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

import encode as enc
import process as pr
from client import PastvuError, Stopped, TransportError, page_url
from store import DONE, FAILED, PENDING, SKIPPED, WRITING

KIND = "replica"
FORMAT = 1                   # the layout of the replica table and meta; run refuses a newer one
SCOPE = "pick"               # the scope name of every status row of a replica base
SCHEMA = """
CREATE TABLE IF NOT EXISTS replica (
    cid INTEGER PRIMARY KEY, path TEXT, sidecars TEXT, download_sha TEXT, output_sha TEXT, outcome TEXT,
    updated REAL);
"""

IDENTICAL, REBUILT, CHANGED = "identical", "rebuilt", "original changed"
GONE, NOT_PUBLIC, CHANGED_ON_PASTVU = "gone from PastVu", "not public on PastVu", "changed on PastVu"
MISSING_STATUS = (403, 404, 410)      # an image URL that answers these: the file is not there under that name
REPORT_NAME = "replica-missing.tsv"


def is_replica(store) -> bool:
    return store.get_meta("kind") == KIND


def check_format(store) -> None:
    f = store.get_meta("replica_format")
    if not isinstance(f, int) or f > FORMAT:
        raise SystemExit(f"This pastvu.db was made by a newer version of the tool (replica format {f}, this one "
                         f"reads {FORMAT}). Download the repository again and run once more.")


def ensure_schema(db) -> None:
    db.executescript(SCHEMA)


# --- the replay: bytes and a record in, the saved file out -----------------------------------

def painted_boxes(rec: dict) -> list:
    return ((rec.get("processing") or {}).get("watermarks") or {}).get("painted") or []


def needs_lama(rec: dict) -> bool:
    return bool(painted_boxes(rec))


def _resize(array: np.ndarray, size) -> np.ndarray:
    """The downscale to the recorded size: process.downscale's area averaging, with the size the
    record gives instead of the one the limit gives (the same for the same limit). A photo our
    base downscaled before 2026-11-05 (Lanczos) comes out a little different; close is enough."""
    import cv2
    w, h = int(size[0]), int(size[1])
    if (array.shape[1], array.shape[0]) == (w, h):
        return array
    return cv2.resize(np.ascontiguousarray(array), (w, h), interpolation=cv2.INTER_AREA)


def _paint(array: np.ndarray, p: dict, lama) -> np.ndarray:
    import watermark as wm
    w = p.get("watermarks") or {}
    return wm.inpaint(array, w.get("painted") or [], int(w.get("dilate") or wm.DEFAULT_DILATE), wm.DEFAULT_MAX_SIZE,
                      lama)


def _first_write(data: bytes, head: dict, p: dict, ds: dict | None, rows: int | None, png: bool, quality: int,
                 lama) -> bytes:
    """What the download wrote: ds is the stream's downscale (the band cut as rows, the picture
    downscaled, then the crop box) or None (the crop on the original, lossless where it can be)."""
    painted = bool((p.get("watermarks") or {}).get("painted"))
    if ds:
        band = int(ds.get("band_rows") or 0) if rows is None else rows
        array = pr.decode(data, head)
        small = _resize(pr.crop_array(array, pr.picture_box(head, band)), ds["to"])
        final = pr.crop_array(small, p.get("box"))
        if painted:
            final = _paint(final, p, lama)
        h, w = small.shape[:2]
        out, _ = enc.array_bytes(final, dict(head, stored=[w, h], orientation=1, mcu=None), png, quality)
        return out
    plan = {"box": p.get("box"), "stored_box": p.get("stored_box")}
    painted_array = None
    if painted:
        painted_array = _paint(pr.crop_array(pr.decode(data, head), plan["box"]), p, lama)
    out, _, _ = enc.encode(data, head, plan, painted_array, png, quality=quality)
    return out


def _shrink(data: bytes, size, png: bool, quality: int) -> bytes:
    """reprocess --downscale of a saved file (reprocess.shrink_file): a PNG stays PNG."""
    head = pr.head_of(data)
    small = _resize(pr.decode(data, head), size)
    h, w = small.shape[:2]
    out, _ = enc.array_bytes(small, dict(head, stored=[w, h], orientation=1, mcu=None),
                             png or head["format"] == "PNG", quality)
    return out


def replay(data: bytes, rec: dict, rows: int | None = None, lama=None) -> tuple[bytes, str]:
    """The saved file of a photo, from its original bytes and its record. rows: the band rows
    of a changed original (None: the record's). lama: the inpainter for a painted photo (None:
    watermark's own, loaded on first use). -> (bytes, extension)"""
    p = rec.get("processing") or {}
    out = p.get("output") or {}
    ext = Path(out.get("path") or "x.jpg").suffix.lower() or ".jpg"
    png = ext == ".png"
    ds = p.get("downscale")
    head = pr.head_of(data)
    if ds and "reprocessed" in ds:
        # the download's write (lossless, exact or a painted array at the quality of the time), then the shrink
        first = _first_write(data, head, p, None, rows, png, enc.JPEG_QUALITY, lama)
        result = _shrink(first, ds["to"], png, int(out.get("quality") or enc.JPEG_QUALITY_DOWNSCALED))
    else:
        quality = int(out.get("quality") or (enc.JPEG_QUALITY_DOWNSCALED if ds else enc.JPEG_QUALITY))
        result = _first_write(data, head, p, ds, rows, png, quality, lama)
    with Image.open(io.BytesIO(result)) as im:
        fmt = im.format
    return result, enc.EXTENSIONS.get(fmt, ext)


def replay_task(data: bytes, rec: dict, rows: int | None) -> tuple[bytes, str]:
    """In a worker process: a photo without a painted watermark."""
    return replay(data, rec, rows)


# --- LaMa ------------------------------------------------------------------------------------

class SharedLama:
    """One LaMa for every download thread: loaded on the first painted photo, on the GPU when
    CUDA works (a short test inference), else on the CPU; one call at a time."""

    def __init__(self, log=print):
        self.lock = threading.Lock()
        self.model = None
        self.log = log
        self.device = ""

    def _load(self):
        import torch
        import watermark as wm
        if torch.cuda.is_available():
            try:
                m = wm.Lama(torch.device("cuda"))
                m(np.zeros((64, 64, 3), np.uint8), np.full((64, 64), 255, np.uint8))
                self.device = f"GPU ({torch.cuda.get_device_name(0)})"
                return m
            except Exception as e:  # noqa: BLE001 - an old GPU the CUDA build has no kernels for
                self.log(f"  LaMa does not run on the GPU ({type(e).__name__}: {e}); it runs on the CPU instead")
        self.device = "CPU"
        return wm.Lama(torch.device("cpu"))

    def __call__(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        with self.lock:
            if self.model is None:
                self.model = self._load()
                self.log(f"  LaMa loaded on the {self.device}")
            out = self.model(rgb, mask)
        if self.device != "CPU":
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        return out


# --- the store side ------------------------------------------------------------------------------

def counts(store) -> dict:
    """State and outcome counts of the replica: {"total", "done", "pending", "skipped", "failed", outcome: n}."""
    with store.lock:
        rows = store.db.execute("SELECT s.state, r.outcome, COUNT(*) FROM status s JOIN replica r ON r.cid=s.cid "
                                "GROUP BY s.state, r.outcome").fetchall()
    c = Counter()
    for state, outcome, n in rows:
        c["total"] += n
        c[state] += n
        if outcome:
            c[outcome] += n
    return dict(c)


def pending(store, retries: int) -> list[int]:
    with store.lock:
        return [r[0] for r in store.db.execute(
            "SELECT cid FROM status WHERE scope=? AND (state=? OR (state=? AND attempts<?)) ORDER BY cid",
            (SCOPE, PENDING, FAILED, retries))]


def cleanup_interrupted(store, base: Path) -> int:
    """Photos left in state writing by a killed run: their files deleted, back to pending."""
    import layout
    with store.lock:
        rows = store.db.execute("SELECT cid, path FROM status WHERE state=? AND scope=?", (WRITING, SCOPE)).fetchall()
    for cid, path in rows:
        if path:
            img = base / path
            for f in (img, *layout.sidecars(img)):
                f.unlink(missing_ok=True)
        with store.lock:
            store.db.execute("UPDATE status SET state=?, path=NULL, updated=? WHERE cid=?", (PENDING, time.time(), cid))
    return len(rows)


def row(store, cid: int) -> dict:
    with store.lock:
        r = store.db.execute("SELECT path, sidecars, download_sha, output_sha FROM replica WHERE cid=?",
                             (cid,)).fetchone()
    if r is None:
        raise pr.Rejected("not in the replica table", failed=True)
    return {"path": r[0], "sidecars": json.loads(r[1] or "{}"), "download_sha": r[2], "output_sha": r[3]}


def missing_rows(store, retries: int) -> list[tuple]:
    """(cid, state, reason, path) of the photos skipped, or failed as often as --retries allows."""
    with store.lock:
        return store.db.execute(
            "SELECT s.cid, s.state, s.reason, r.path FROM status s JOIN replica r ON r.cid=s.cid "
            "WHERE s.state=? OR (s.state=? AND s.attempts>=?) ORDER BY s.cid", (SKIPPED, FAILED, retries)).fetchall()


def write_report(store, base: Path, retries: int) -> Path | None:
    rows = missing_rows(store, retries)
    path = base / "_reports" / REPORT_NAME
    if not rows:
        path.unlink(missing_ok=True)
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["cid\tstate\treason\tpage\tpath"]
    lines += [f"{cid}\t{state}\t{reason}\t{page_url(cid)}\t{p}" for cid, state, reason, p in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- the run ---------------------------------------------------------------------------------

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Run:
    def __init__(self, store, base: Path, client, fetchers: int = 4, workers: int = 4, retries: int = 3,
                 limit: int | None = None, report_every: int = 500, display: str = "live"):
        self.store, self.base, self.client = store, Path(base), client
        self.fetchers, self.workers, self.retries = max(1, fetchers), max(0, workers), retries
        self.limit, self.report_every, self.display_mode = limit, max(1, report_every), display
        self.stop = threading.Event()
        self.client.stop = self.stop
        # twice as many threads as downloads: while one thread rebuilds and writes, another downloads,
        # and PastVu never sees more than `fetchers` requests at once
        self.net = threading.Semaphore(self.fetchers)
        self.lock = threading.Lock()
        self.stats = Counter()
        self.seen = 0
        self.d = None
        self.lama = SharedLama(self.log)
        self.ex = None

    def log(self, text: str) -> None:
        (self.d.print if self.d is not None else print)(text)

    # --- one photo ---

    def fetch(self, cid: int, rec: dict, rep: dict) -> tuple[bytes, int | None, str | None]:
        """The original bytes, the band rows of a changed original (else None) and its outcome."""
        old = rec.get("record") or {}
        dl = rec.get("download") or {}
        want = rep.get("download_sha") or dl.get("sha256")
        file = old.get("file")
        data = None
        if file:
            try:
                data, _ = self.client.image(file, "a")
            except Stopped:
                raise
            except TransportError as e:
                if not any(f"HTTP {c}" in str(e) for c in MISSING_STATUS):
                    raise pr.Rejected(f"download: {e}", failed=True)
        if data is not None and sha256(data) == want:
            return data, None, None
        try:
            result = self.client.give_for_page(cid)
        except PastvuError as e:
            if e.kind == "NotFoundError":
                raise pr.Rejected(GONE)
            raise pr.Rejected(f"api: {e.code or e.kind}", failed=True)
        photo = result.get("photo") or {}
        if int(photo.get("s") or 5) != 5:
            raise pr.Rejected(NOT_PUBLIC)
        new_file = photo.get("file")
        if not new_file:
            raise pr.Rejected(GONE)
        if data is None or new_file != file:
            try:
                data, _ = self.client.image(new_file, "a")
            except Stopped:
                raise
            except TransportError as e:
                if any(f"HTTP {c}" in str(e) for c in MISSING_STATUS):
                    raise pr.Rejected(GONE)
                raise pr.Rejected(f"download: {e}", failed=True)
        if sha256(data) == want:
            return data, None, None
        try:
            head = pr.head_of(data)
        except Exception as e:  # noqa: BLE001
            raise pr.Rejected(f"download: unreadable image ({type(e).__name__})", failed=True)
        same_picture = (int(photo.get("w") or 0), int(photo.get("h") or 0)) == (int(old.get("w") or 0),
                                                                                 int(old.get("h") or 0))
        if not same_picture or head["orientation"] != 1 or int(dl.get("orientation") or 1) != 1:
            raise pr.Rejected(CHANGED_ON_PASTVU)
        rows = pr.band_rows(head, photo)
        return data, rows, CHANGED

    def rebuild(self, data: bytes, rec: dict, rows: int | None) -> tuple[bytes, str]:
        if needs_lama(rec):
            return replay(data, rec, rows, self.lama)
        if self.ex is None:
            return replay(data, rec, rows)
        from concurrent.futures.process import BrokenProcessPool
        for attempt in (1, 2):
            ex = self.ex
            try:
                return ex.submit(replay_task, data, rec, rows).result()
            except BrokenProcessPool:
                with self.lock:
                    if self.ex is ex:
                        ex.shutdown(wait=False, cancel_futures=True)
                        self.ex = self._pool()
                if attempt == 2:
                    raise pr.Rejected("replay: the worker process crashed", failed=True)
        raise AssertionError("unreachable")

    def write(self, cid: int, rep: dict, out: bytes, ext: str, changed: str | None) -> str:
        import layout
        rel = Path(rep["path"])
        if rel.suffix.lower() != ext:
            rel = rel.with_suffix(ext)
        rel_s = rel.as_posix()
        path = self.base / rel
        with self.store.lock:
            self.store.db.execute("UPDATE status SET state=?, path=?, updated=? WHERE cid=?",
                                  (WRITING, rel_s, time.time(), cid))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(out)
            for suffix, text in (rep.get("sidecars") or {}).items():
                (path.parent / (path.stem + suffix)).write_bytes(text.encode("utf-8"))
            sha = sha256(out)
            w, h = pr.upright_size(pr.head_of(out))
            outcome = changed or (IDENTICAL if sha == rep.get("output_sha") else REBUILT)
            now = time.time()

            def work(db):
                db.execute("UPDATE status SET state=?, reason='', path=?, sha256=?, bytes=?, width=?, height=?, "
                           "updated=? WHERE cid=?", (DONE, rel_s, sha, len(out), w, h, now, cid))
                db.execute("UPDATE replica SET outcome=?, updated=? WHERE cid=?", (outcome, now, cid))
            self.store.tx(work)
        except BaseException:
            for f in (path, *layout.sidecars(path)):
                f.unlink(missing_ok=True)
            with self.store.lock:
                self.store.db.execute("UPDATE status SET state=?, path=NULL, updated=? WHERE cid=?",
                                      (PENDING, time.time(), cid))
            raise
        return outcome

    def reject(self, cid: int, r: pr.Rejected) -> None:
        state = FAILED if r.failed else SKIPPED
        now = time.time()

        def work(db):
            db.execute("UPDATE status SET state=?, reason=?, path=NULL, attempts=attempts+?, updated=? WHERE cid=?",
                       (state, r.reason, 1 if r.failed else 0, now, cid))
            db.execute("UPDATE replica SET outcome=?, updated=? WHERE cid=?", (None if r.failed else r.reason, now, cid))
        self.store.tx(work)

    def one(self, cid: int) -> None:
        d = self.d
        try:
            rec = self.store.record(cid)
            if not rec:
                raise pr.Rejected("no record in the store", failed=True)
            rep = row(self.store, cid)
            with self.net:
                if self.stop.is_set():
                    return                           # stopping: the photo stays pending
                d.set(f"{cid}: downloading")
                data, rows, changed = self.fetch(cid, rec, rep)
            d.set(f"{cid}: {'painting' if needs_lama(rec) else 'rebuilding'}")
            out, ext = self.rebuild(data, rec, rows)
            outcome = self.write(cid, rep, out, ext, changed)
        except Stopped:
            return                                   # the run is stopping: the photo stays pending
        except pr.Rejected as r:
            self.reject(cid, r)
            key = "failed" if r.failed else r.reason
            with self.lock:
                self.stats[key] += 1
            if r.failed and self.stats["failed"] <= 20:
                self.log(f"  {cid}: {r.reason}")
            d.advance(saved=False)
            d.line({"cid": cid, "state": FAILED if r.failed else SKIPPED, "reason": r.reason})
            self.tick()
            return
        except Exception as e:  # noqa: BLE001 - one bad photo never stops the run
            self.reject(cid, pr.Rejected(f"{type(e).__name__}: {e}", failed=True))
            with self.lock:
                self.stats["failed"] += 1
            if self.stats["failed"] <= 20:
                self.log(f"  {cid}: {type(e).__name__}: {e}")
            d.advance(saved=False)
            self.tick()
            return
        with self.lock:
            self.stats["saved"] += 1
            self.stats[outcome] += 1
        d.advance(saved=True)
        d.line({"cid": cid, "state": DONE, "path": rep["path"], "outcome": outcome})
        self.tick()

    def tick(self) -> None:
        with self.lock:
            self.seen += 1
            due = self.seen % self.report_every == 0
            s = dict(self.stats)
        if due:
            self.log(f"[{self.seen:,} this run] " + describe(s))

    # --- the loop ---

    def _pool(self):
        from concurrent.futures import ProcessPoolExecutor
        import prepare
        return ProcessPoolExecutor(self.workers, initializer=prepare._init_worker)

    def run(self) -> dict:
        from progress import Display
        import cv2  # noqa: F401 - imported once here: the threads' first photos would import it at once,
        import watermark  # noqa: F401 - and OpenCV's import is not safe from several threads
        n = cleanup_interrupted(self.store, self.base)
        if n:
            print(f"  {n} photos were being written when the last run stopped; they are done again")
        todo = pending(self.store, self.retries)
        c = counts(self.store)
        total = c.get("total", 0)
        if self.limit is not None:
            todo = todo[:max(0, self.limit)]
        t0 = time.time()
        if not todo:
            return {"seconds": 0.0, "stats": {}, "taken": 0}
        it = iter(todo)
        it_lock = threading.Lock()
        if self.workers > 0:
            self.ex = self._pool()

        def worker():
            while not self.stop.is_set():
                with it_lock:
                    cid = next(it, None)
                if cid is None:
                    return
                self.one(cid)

        def on_sigint(signum, frame):
            if self.stop.is_set():
                raise KeyboardInterrupt
            self.stop.set()
            self.log("Ctrl+C: finishing the photos in hand, then stopping. Press Ctrl+C again to stop at once.")

        try:
            old = signal.signal(signal.SIGINT, on_sigint)
        except ValueError:
            old = None
        done_before = total - len(pending(self.store, self.retries))
        try:
            with Display(total, done_before, self.display_mode) as d:
                self.d = d
                threads = [threading.Thread(target=worker, daemon=True, name=f"replica-{i}")
                           for i in range(self.fetchers * 2)]
                for t in threads:
                    t.start()
                while any(t.is_alive() for t in threads):
                    for t in threads:
                        t.join(0.25)
                d.set("finished" if not self.stop.is_set() else "stopped")
        finally:
            self.d = None
            if old is not None:
                signal.signal(signal.SIGINT, old)
            if self.ex is not None:
                self.ex.shutdown(wait=False, cancel_futures=True)
        return {"seconds": time.time() - t0, "stats": dict(self.stats), "taken": len(todo),
                "stopped": self.stop.is_set(), "lama": self.lama.device}


# --- snapshot: what the partner kept ---------------------------------------------------------

SNAPSHOT_KIND = "pastvu-curation"
SNAPSHOT_FORMAT = 1


def scan_images(base: Path) -> dict[int, list[str]]:
    """Every image under the base whose name is a photo number -> its paths relative to the base
    (a photo can be there twice). _reports and other folders starting with _ are left out."""
    import os
    import layout
    found: dict[int, list[str]] = {}
    other = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if not d.startswith("_")]
        for name in filenames:
            stem, ext = os.path.splitext(name)
            if ext.lower() not in layout.IMAGE_EXTS:
                continue
            rel = Path(dirpath, name).relative_to(base).as_posix()
            if stem.isdigit():
                found.setdefault(int(stem), []).append(rel)
            else:
                other.append(rel)
    found[-1] = other                                  # the images whose name is no photo number
    return found


def snapshot(store, base: Path) -> dict:
    """The curation of a rebuilt pick: kept (saved by run and still in the folder, wherever it
    is now), deleted (saved by run, no image of that number left), never received (skipped or
    failed by run, or not done yet), moved (kept under another path), and images that are no
    photo of the pick."""
    found = scan_images(base)
    other = found.pop(-1)
    with store.lock:
        rows = store.db.execute("SELECT s.cid, s.state, s.reason, s.path, r.path FROM status s "
                                "JOIN replica r ON r.cid=s.cid ORDER BY s.cid").fetchall()
    kept, deleted, never, moved = [], [], {}, {}
    known = set()
    for cid, state, reason, saved, target in rows:
        known.add(cid)
        here = found.get(cid) or []
        if state != DONE:
            never[str(cid)] = reason or state
            continue
        if not here:
            deleted.append(cid)
            continue
        kept.append(cid)
        if saved not in here:
            moved[str(cid)] = here[0]
    unknown = sorted(rel for cid, rels in found.items() if cid not in known for rel in rels) + sorted(other)
    return {"kind": SNAPSHOT_KIND, "format": SNAPSHOT_FORMAT, "name": store.get_meta("replica_name"),
            "replica_made": store.get_meta("replica_made"), "made": time.time(),
            "made_text": time.strftime("%Y-%m-%d %H:%M"), "photos": len(rows),
            "kept": kept, "deleted": deleted, "never": never, "moved": moved,
            "unknown": len(unknown), "unknown_sample": unknown[:50]}


def write_snapshot(snap: dict, base: Path) -> Path:
    import gzip
    path = base / f"curation-{time.strftime('%Y%m%d-%H%M', time.localtime(snap['made']))}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False)
    return path


def read_snapshot(path: Path) -> dict:
    import gzip
    try:
        opener = gzip.open if Path(path).suffix.lower() == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as f:
            snap = json.load(f)
    except (OSError, ValueError) as e:
        raise SystemExit(f"{path}: not a curation snapshot ({type(e).__name__}: {e})")
    if not isinstance(snap, dict) or snap.get("kind") != SNAPSHOT_KIND:
        raise SystemExit(f"{path}: not a curation snapshot (made by run.bat snapshot on a shared pick)")
    if int(snap.get("format") or 0) > SNAPSHOT_FORMAT:
        raise SystemExit(f"{path} was made by a newer version of the tool; update the repository")
    return snap


def describe(s: dict) -> str:
    parts = [f"{s.get('saved', 0):,} saved ({s.get(IDENTICAL, 0):,} identical, {s.get(REBUILT, 0):,} rebuilt, "
             f"{s.get(CHANGED, 0):,} from a changed original)"]
    for key in (GONE, NOT_PUBLIC, CHANGED_ON_PASTVU):
        if s.get(key):
            parts.append(f"{s[key]:,} {key}")
    parts.append(f"{s.get('failed', 0):,} failed")
    return ", ".join(parts)
