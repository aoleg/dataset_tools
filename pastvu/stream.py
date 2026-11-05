"""The stream: every photo of a scope that is not done yet goes through
fetch, band and borders (fetch threads), watermark detection, painting and
classification (the main thread, models resident on the GPU), and encoding
and writing (writer threads). Images, plans and output bytes stay in memory.

Writing one photo: its path is claimed in the store (state writing), the
image and its caption are written, then the status, the record, the
embedding and the counters are committed in one transaction. A photo found
in state writing at the start of a run was interrupted: its files are
deleted and it is done again.

An import (importer.py) reads each photo from its source folder instead of
PastVu (no band), writes it under its source name with its cleaned caption,
and skips the copies of a base's photos after the embedding.

A raw run (run --raw, Options.raw) stops after the fetch's gates: the original
bytes go into <base>/_storage with the record as <cid>.json and the facts caption
as <cid>.txt, the store marks the photo stored, and no model is loaded. The
processing of the storage is a second source of the same stream: storage
threads (as many as the fetch threads by default: a 3 MB scan takes 0.36 s to
read, decode and plan) check stored photos against the store and prepare them
(decode, band, borders) a few ahead, and the GPU stage takes one whenever no download
is ready (run --pull: downloads first, the storage fills the gaps; pull: the
storage alone). After a stored photo is committed, or rejected, its raw files
are deleted (--keep-rejects keeps a reject's). --limit counts the photos taken
from both sources together.

A photo whose long side is more than max_aspect (21:9) times its short side is
skipped before the download (by the record's size) or after the crop.
A photo the quality probe scores at or above its threshold is skipped after
the embedding, before the category decision and the writing, and never
fetched again (like a photo under --min-side).

Ctrl+C finishes the images already fetched and stops; a second Ctrl+C stops
at once. Either way the store says what is done, and the next run of the same
command continues with the rest.
"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import colour
import encode as enc
import folders
import layout
import process as pr
import sidecars
from client import IMG, Client, PastvuError, Stopped, TransportError, page_url
from progress import Display, Stats, interval_block, raw_report_block, report_block
from store import DONE, FAILED, SKIPPED, STORED, Store

TOOL_VERSION = "0.2"


@dataclass
class Options:
    base: Path
    scope: dict                       # the primary scope (its name titles the run)
    scopes: dict | None = None        # every scope the run writes into (run --by), name -> scope
    min_side: int = 512
    max_aspect: float = 21 / 9        # long side over short side; 0: no limit
    png: bool = False
    wm_conf: float = 0.1
    wm_paint_conf: float = 0.4
    wm_mode: str = "inpaint"
    wm_dilate: int = 15
    wm_max_size: int = 2048
    borders: bool = True
    dark_borders: bool = True
    fetchers: int = 4
    depth: int = 16
    writers: int = 2
    report_every: int = 500
    limit: int | None = None
    retries: int = 3
    txt_fields: list = field(default_factory=lambda: list(sidecars.TXT_FIELDS))
    json_files: bool = False
    json_sidecar: bool = False
    comments: bool = False
    colour: bool = True               # tell black-and-white from colour (an import with --top none does not)
    display: str = "live"
    raw: bool = False                 # run --raw: download into the storage, no processing
    keep_rejects: bool = False        # a stored photo rejected by the processing keeps its raw files
    storage_threads: int = 4          # threads that read and prepare stored photos
    storage_depth: int = 4            # stored photos prepared ahead of the GPU (at least the threads)

    @property
    def scope_dir(self) -> Path:
        return self.base / self.scope["folder"]


@dataclass
class Job:
    cid: int
    scope: str = ""
    result: dict = None
    photo: dict = None
    data: bytes = b""
    ctype: str = ""
    head: dict = None
    array: np.ndarray = None
    rows: int = 0
    band: dict = None
    pic_box: list = None
    borders: dict = None
    comments: dict = None
    caption: str | None = None        # an import: the source's caption (raw)
    sidecars: dict = None             # an import: the source's other sidecars, copied as they are
    written: list = None              # an import: the sidecars written next to the image
    raw: str | None = None            # from the storage: the raw image's path relative to the base
    t_fetch: float = 0.0


def cleanup_interrupted(store: Store, base: Path, scope: str) -> int:
    """Photos left in state writing by a killed run: delete their files, make them pending."""
    rows = store.writing_rows(scope)
    for cid, path in rows:
        if path:
            img = base / path
            for f in (img, *layout.sidecars(img)):
                f.unlink(missing_ok=True)
        store.unclaim(cid)
    return len(rows)


class Stream:
    def __init__(self, client: Client, store: Store, opts: Options, classifier=None, detector=None, lama=None,
                 caption_client=None, vram=None, mode_probe=None, exporter=None, source: Path | None = None,
                 against=None):
        self.client, self.store, self.opts = client, store, opts
        self.classifier = classifier
        self.det_name, self.det = detector or (None, None)
        self.lama = lama
        self.scope = opts.scope["name"]
        self.scopes = opts.scopes or {self.scope: opts.scope}
        self.chunkers = {n: folders.ScopeChunks(store, opts.base, sc, log=self.log) for n, sc in self.scopes.items()}
        self.stats = Stats()
        self.stop = threading.Event()
        self.in_flight = 0
        self.lock = threading.Lock()
        self.display: Display | None = None
        self.caption_client, self.vram = caption_client, vram
        self.mode_probe = mode_probe       # decides the borderline black-and-white or colour photos
        self.exporter = exporter           # run --export: links each saved photo into the export
        self.source = Path(source) if source else None      # an import: the folder the photos are read from
        self.against = against             # an import: importer.Against, the base whose copies are skipped
        self.captioner = None
        self.budget: int | None = None     # photos still to take under --limit, over both sources
        self.storage_source = False        # this run reads the storage too
        self.records_refetched = 0         # stored photos whose JSON was missing
        self.in_request: dict[int, float] = {}   # fetch thread -> when its current fetch began (the report)
        self.storage_threads_n = 0
        self.retries_seen = 0              # the client's retry count at the last report block
        enc_model = getattr(getattr(classifier, "encoder", None), "model_id", None)
        self.emb_key = (enc_model, getattr(classifier, "patches", None))
        self.tool = {"name": "dataset_tools/pastvu", "version": TOOL_VERSION, "detector": self.det_name,
                     "detector_conf": opts.wm_conf, "paint_conf": opts.wm_paint_conf, "encoder": enc_model,
                     "priority_margin": getattr(classifier, "priority_margin", None),
                     "quality_threshold": getattr(getattr(classifier, "quality", None), "threshold", None),
                     "patches": getattr(classifier, "patches", None)}

    def log(self, text: str) -> None:
        d = getattr(self, "display", None)
        (d.print if d is not None else print)(text)

    # --- fetch threads -------------------------------------------------------------

    def fetch_local(self, cid: int, scope: str) -> Job:
        """An import: the photo from its source file; no PastVu record, so no band to cut."""
        import importer
        o = self.opts
        t0 = time.perf_counter()
        job = Job(cid, scope)
        rel = self.store.source_of(cid)
        if rel is None:
            raise pr.Rejected("not in the source list", failed=True)
        path = self.source / rel
        try:
            job.data = path.read_bytes()
        except OSError as e:
            raise pr.Rejected(f"read: {e.strerror or e}", failed=True)
        try:
            job.head = pr.head_of(job.data)
        except Exception as e:  # noqa: BLE001
            raise pr.Rejected(f"unreadable image ({type(e).__name__})", failed=True)
        if job.head["frames"] > 1:
            raise pr.Rejected("animated")
        job.rows = 0
        job.array = pr.decode(job.data, job.head)
        h, w = job.array.shape[:2]
        if min(w, h) < o.min_side:
            raise pr.Rejected("too small")
        wide = pr.aspect_reason(w, h, o.max_aspect)
        if wide:
            raise pr.Rejected(wide)
        job.photo = {"name": layout.stem_of(cid), "src": rel, "w": w, "h": h}
        job.result = {}
        job.ctype = "image/" + str(job.head.get("format") or "").lower()
        job.pic_box = pr.picture_box(job.head, 0)
        if o.borders:
            job.borders = pr.borders_plan(job.array, {"dark": o.dark_borders})
        job.caption = importer.read_caption(path)
        job.sidecars = importer.read_sidecars(path)
        job.t_fetch = time.perf_counter() - t0
        return job

    def fetch_remote(self, cid: int, scope: str) -> Job:
        """The record and the bytes from PastVu, with every gate that needs no
        pixels: what a raw run does for a photo. A full fetch goes on with prepare."""
        o = self.opts
        t0 = time.perf_counter()
        job = Job(cid, scope)
        try:
            job.result = self.client.give_for_page(cid)
        except PastvuError as e:
            raise pr.Rejected("gone" if e.kind in ("NotFoundError",) else f"api: {e.code or e.kind}")
        p = job.photo = job.result.get("photo") or {}
        if int(p.get("type") or 1) != 1:
            raise pr.Rejected("painting")
        if int(p.get("s") or 5) != 5:
            raise pr.Rejected("not public")
        w, h = int(p.get("w") or 0), int(p.get("h") or 0)
        if min(w, h) < o.min_side:
            raise pr.Rejected("too small")
        wide = pr.aspect_reason(w, h, o.max_aspect)
        if wide:
            raise pr.Rejected(wide)
        if not p.get("file"):
            raise pr.Rejected("no file in the record", failed=True)
        t_dl = time.perf_counter()
        job.data, job.ctype = self.client.image(p["file"], "a")
        self.stats.latency(time.perf_counter() - t_dl)
        if not job.ctype.startswith("image/"):
            raise pr.Rejected(f"download: not an image ({job.ctype or 'no type'})", failed=True)
        try:
            job.head = pr.head_of(job.data)
        except Exception as e:  # noqa: BLE001
            raise pr.Rejected(f"download: unreadable image ({type(e).__name__})", failed=True)
        if job.head["frames"] > 1:
            raise pr.Rejected("animated")
        job.rows = pr.band_rows(job.head, p)
        if o.comments:
            try:
                job.comments = self.client.comments(cid)
            except (PastvuError, TransportError) as e:
                job.comments = {"error": str(e)}
        job.t_fetch = time.perf_counter() - t0
        return job

    def prepare(self, job: Job) -> Job:
        """The pixels: decode upright, verify the band, the picture box, the border plan."""
        o = self.opts
        t0 = time.perf_counter()
        job.array = pr.decode(job.data, job.head)
        job.band = pr.verify_band(job.array, job.head, job.rows)
        job.pic_box = pr.picture_box(job.head, job.rows)
        if o.borders:
            picture = pr.crop_array(job.array, job.pic_box)
            job.borders = pr.borders_plan(picture, {"dark": o.dark_borders})
        job.t_fetch += time.perf_counter() - t0
        return job

    def fetch(self, cid: int, scope: str) -> Job:
        if self.source is not None:
            return self.fetch_local(cid, scope)
        return self.prepare(self.fetch_remote(cid, scope))

    def fetch_stored(self, cid: int, scope: str) -> Job:
        """A photo of the storage: its bytes and its record from <base>/_storage, checked
        against the store (a missing or changed file is failed, its raw path cleared, so a
        later run downloads it again); the record is asked from PastVu again when the JSON
        is missing or unreadable."""
        o = self.opts
        t0 = time.perf_counter()
        job = Job(cid, scope)
        t_row = time.perf_counter()
        row = self.store.status(cid) or {}
        self.stats.time("store_read", time.perf_counter() - t_row)      # waits for the store's lock too
        rel = row.get("raw") or row.get("path")
        if not rel:
            raise pr.Rejected("stored file unknown", failed=True)
        job.raw = rel
        img = o.base / rel
        try:
            job.data = img.read_bytes()
        except OSError as e:
            raise pr.Rejected(f"stored file missing: {e.strerror or e}", failed=True)
        if row.get("sha256") and hashlib.sha256(job.data).hexdigest() != row["sha256"]:
            raise pr.Rejected("stored file changed", failed=True)
        doc = None
        try:
            doc = json.loads(img.with_suffix(".json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        if isinstance(doc, dict) and isinstance(doc.get("result"), dict) and doc["result"].get("photo"):
            job.result = doc["result"]
            job.ctype = str((doc.get("download") or {}).get("content_type") or "")
        else:
            with self.lock:
                self.records_refetched += 1
                first = self.records_refetched == 1
            if first:
                self.log(f"  {cid}: no record in the storage; asking PastVu for it (said once)")
            try:
                job.result = self.client.give_for_page(cid)
            except PastvuError as e:
                raise pr.Rejected(f"stored record: {e.kind} {e.code}", failed=True)
        p = job.photo = job.result.get("photo") or {}
        try:
            job.head = pr.head_of(job.data)
        except Exception as e:  # noqa: BLE001
            raise pr.Rejected(f"stored file unreadable ({type(e).__name__})", failed=True)
        if not job.ctype:
            job.ctype = "image/" + str(job.head.get("format") or "").lower()
        job.rows = pr.band_rows(job.head, p)
        job.t_fetch = time.perf_counter() - t0
        return self.prepare(job)

    def take(self, it, lock) -> tuple[int, str] | None:
        """The next (cid, scope) of a source, within --limit counted over both sources."""
        with lock:
            if self.budget is not None and self.budget <= 0:
                return None
            pair = next(it, None)
            if pair is not None and self.budget is not None:
                self.budget -= 1
            return pair

    def fetch_worker(self, cids, cid_lock, out_q: queue.Queue) -> None:
        while not self.stop.is_set():
            pair = self.take(cids, cid_lock)
            if pair is None:
                break
            cid, scope = pair
            me = threading.get_ident()
            with self.lock:
                self.in_flight += 1
                self.in_request[me] = time.time()
            try:
                item = self.fetch_remote(cid, scope) if self.opts.raw else self.fetch(cid, scope)
            except Stopped:
                item = None                      # the run is stopping: the photo stays pending for the next run
            except pr.Rejected as r:
                item = r
            except TransportError as e:
                item = pr.Rejected(f"download: {e}", failed=True)
            except Exception as e:  # noqa: BLE001 - one bad photo never stops the stream
                item = pr.Rejected(f"fetch: {type(e).__name__}: {e}", failed=True)
            finally:
                with self.lock:
                    self.in_flight -= 1
                    self.in_request.pop(me, None)
            if item is None:
                continue
            if isinstance(item, pr.Rejected):
                item.cid, item.scope = cid, scope
            out_q.put(item)
        out_q.put(None)

    def storage_worker(self, pairs, lock, sq: queue.Queue) -> None:
        """The storage source: stored photos read and prepared a few ahead of the GPU."""
        while not self.stop.is_set():
            pair = self.take(pairs, lock)
            if pair is None:
                break
            cid, scope = pair
            try:
                item = self.fetch_stored(cid, scope)
            except Stopped:
                continue                         # the run is stopping: the photo stays in the storage
            except pr.Rejected as r:
                item = r
            except TransportError as e:
                item = pr.Rejected(f"stored record: {e}", failed=True)
            except Exception as e:  # noqa: BLE001
                item = pr.Rejected(f"stored: {type(e).__name__}: {e}", failed=True)
            if isinstance(item, pr.Rejected):
                item.cid, item.scope = cid, scope
            t_put = time.perf_counter()
            sq.put(item)
            self.stats.time("storage_wait_put", time.perf_counter() - t_put)   # the GPU had no slot for it
        sq.put(None)

    # --- outcomes ------------------------------------------------------------------

    def remove_raw(self, rel: str) -> None:
        """Delete a photo's raw files in the storage: the image and every file of its name."""
        img = self.opts.base / rel
        for f in (img, *layout.sidecars(img)):
            f.unlink(missing_ok=True)

    def reject(self, cid: int, r: pr.Rejected, scope: str) -> None:
        key = r.reason.split(":")[0].split(" (")[0]
        state = FAILED if r.failed else SKIPPED
        raw = (self.store.status(cid) or {}).get("raw")
        self.store.reject(cid, scope, state, r.reason, {f"c.{state}.{key}": 1})
        if raw and (r.failed or not self.opts.keep_rejects):
            self.remove_raw(raw)             # nothing fetches a reject again; a failed one is downloaded afresh
        self.stats.add(**{f"{state}.{key}": 1})
        self.display.advance(saved=False)
        self.display.line({"cid": cid, "state": state, "reason": r.reason})

    # --- the GPU stage -------------------------------------------------------------

    def release_cache(self) -> None:
        """Give the allocator's cache back after a paint. A LaMa window of 2048 px needs
        4.6 GB at its peak and leaves 7 to 14 GB reserved, which the allocator would keep
        for the rest of the run; giving it back costs about 40 ms, and the next big window
        is re-allocated in the same time it took cached (cuDNN keeps its tuning)."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - never let a cache release stop the stream
            pass

    def process(self, job: Job) -> tuple:
        import watermark as wm
        o, d, s = self.opts, self.display, self.stats
        picture = pr.crop_array(job.array, job.pic_box)
        boxes = []
        if self.det is not None:
            d.set(f"{job.cid}: detecting watermarks")
            t = time.perf_counter()
            boxes = wm.detect(self.det, picture, o.wm_conf)
            s.time("detect", time.perf_counter() - t)
        plan = pr.compose(job.head, job.pic_box, job.rows > 0, job.borders, boxes,
                          {"mode": o.wm_mode, "dilate": o.wm_dilate, "paint_conf": o.wm_paint_conf})
        cw, ch = plan["size"]
        if min(cw, ch) < o.min_side:
            raise pr.Rejected("too small after crop")
        wide = pr.aspect_reason(cw, ch, o.max_aspect)
        if wide:
            raise pr.Rejected(wide + " after crop")
        crop = pr.crop_array(job.array, plan["box"])
        painted = None
        if plan["wm_inside"]:
            d.set(f"{job.cid}: painting {len(plan['wm_inside'])} watermark box(es)")
            t = time.perf_counter()
            painted = wm.inpaint(crop, plan["wm_inside"], o.wm_dilate, o.wm_max_size, self.lama)
            self.release_cache()
            s.time("paint", time.perf_counter() - t)
        final = painted if painted is not None else crop
        t = time.perf_counter()
        tone = colour.scores(final, plan["wm_inside"], o.wm_dilate) if o.colour else None  # painted boxes left out
        s.time("colour", time.perf_counter() - t)
        vec = verdict = None
        if self.classifier is not None:
            d.set(f"{job.cid}: classifying")
            t = time.perf_counter()
            vec = self.classifier.embed(final)
            bad = float(self.classifier.low_quality(vec)[0])
            if not np.isnan(bad):          # at or above the threshold
                s.time("classify", time.perf_counter() - t)
                raise pr.Rejected(self.classifier.quality.reason(bad))
            if self.against is not None:
                d.set(f"{job.cid}: looking for a copy in the base")
                hit = self.against.match(vec, final)
                if hit:
                    s.time("classify", time.perf_counter() - t)
                    raise pr.Rejected(self.against.reason(hit))
            verdict = self.classifier.decide(vec)
            s.time("classify", time.perf_counter() - t)
        if tone is not None:
            tone = colour.decide(tone, self.mode_probe, vec)
        return plan, boxes, painted, vec, verdict, final.shape[1], final.shape[0], final, tone

    # --- writer threads ------------------------------------------------------------

    def chunker_for(self, scope: str):
        """The scope's chunker; a scope the run did not open (a stored photo of another
        scope) gets one on first use."""
        with self.lock:
            c = self.chunkers.get(scope)
            if c is None:
                sc = self.store.scope(scope)
                if sc is None:
                    raise pr.Rejected(f"scope {scope} is gone", failed=True)
                c = self.chunkers[scope] = folders.ScopeChunks(self.store, self.opts.base, sc, log=self.log)
                self.scopes[scope] = sc
            return c

    def write_raw(self, job: Job) -> None:
        """A raw run: the original bytes, the record and the facts caption into the storage,
        then the store row, in that order (a crash before the row leaves files the next
        raw write overwrites under the same names)."""
        o, s, d = self.opts, self.stats, self.display
        ext = enc.EXTENSIONS.get(job.head["format"], ".bin")
        rel = layout.storage_rel(job.cid, ext)
        path = o.base / rel
        d.set(f"{job.cid}: storing {rel}")
        t = time.perf_counter()
        sha = hashlib.sha256(job.data).hexdigest()
        p = job.photo
        result = dict(job.result)
        if job.comments is not None:
            result["comments"] = job.comments
        download = {"url": f"{IMG}/a/{p['file']}", "bytes": len(job.data), "sha256": sha, "content_type": job.ctype,
                    "format": job.head["format"], "stored_size": job.head["stored"],
                    "orientation": job.head["orientation"], "mcu": job.head["mcu"]}
        doc = sidecars.raw_json(job.cid, page_url(job.cid), result, download, job.scope,
                                {"name": self.tool["name"], "version": self.tool["version"]})
        files = (path, path.with_suffix(".json"), path.with_suffix(".txt"))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(job.data)
            files[1].write_text(doc, encoding="utf-8")
            files[2].write_text(sidecars.caption(p, o.txt_fields), encoding="utf-8")
            self.store.store_raw(job.cid, job.scope, rel, sha, len(job.data))
        except BaseException:
            for f in files:
                f.unlink(missing_ok=True)
            raise
        s.time("write", time.perf_counter() - t)
        s.add(stored=1, raw_bytes=len(job.data))
        d.advance(saved=True)
        d.line({"cid": job.cid, "state": STORED, "path": rel, "bytes": len(job.data)})

    def write(self, job: Job, plan: dict, boxes: list, painted, vec, verdict, fw: int, fh: int, final=None,
              tone=None) -> None:
        o, s, d = self.opts, self.stats, self.display
        d.set(f"{job.cid}: encoding")
        t = time.perf_counter()
        data, ext, mode = enc.encode(job.data, job.head, plan, painted, o.png)
        s.time("encode", time.perf_counter() - t)
        if self.classifier is not None:
            verdict = self.classifier.example_override(data, verdict)
        folder = verdict.folder if verdict else ""
        scope = job.scope
        chunker = self.chunker_for(scope)
        if o.colour:
            tone = tone or colour.decide({"off_hue": 0.0, "strong": 0.0, "tone": 0.0})
        tone_mode = tone["mode"] if tone else None
        rel_scope = chunker.place(layout.with_mode(chunker.sc, tone_mode, folder), f"{layout.stem_of(job.cid)}{ext}")
        rel = chunker.rel(rel_scope)
        path = o.base / rel
        d.set(f"{job.cid}: saving {rel_scope}")
        t = time.perf_counter()
        self.store.claim(job.cid, scope, rel)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            if self.source is not None:
                import importer
                facts = importer.clean_caption(job.caption) if job.caption else ""   # none left: no .txt
                job.written = importer.write_sidecars(path, facts, job.sidecars or {}, bool(plan["changed"]))
            else:
                facts = sidecars.caption(job.photo, o.txt_fields)
                path.with_suffix(".txt").write_text(facts, encoding="utf-8")
            sha = hashlib.sha256(data).hexdigest()
            doc = self.record_json(job, plan, boxes, mode, rel, data, sha, fw, fh, verdict, tone)
            if o.json_files:
                rf = layout.record_file(o.base, job.cid)
                rf.parent.mkdir(parents=True, exist_ok=True)
                rf.write_text(doc, encoding="utf-8")
            if o.json_sidecar:
                path.with_suffix(".json").write_text(doc, encoding="utf-8")
            emb = (self.emb_key[0], self.emb_key[1], vec) if vec is not None else None
            counters = self.counters(job, plan, mode, folder, boxes, tone)
            self.store.finish(job.cid, scope, {"category": folder, "path": rel, "sha256": sha, "bytes": len(data),
                                                    "width": fw, "height": fh, "changed": plan["changed"],
                                                    "mode": tone_mode},
                              doc, emb, counters)
        except BaseException:
            chunker.abandon(rel_scope)
            for f in (path, *layout.sidecars(path)):
                f.unlink(missing_ok=True)
            self.store.unclaim(job.cid)
            raise
        t_move = time.perf_counter()
        chunker.done(rel_scope)          # may move the category's images into 001 now that it passed the split
        moved = time.perf_counter() - t_move
        if moved > 0.5:                  # a chunk move: thousands of files, the writer queue fills behind it
            s.time("chunk_move", moved)
            s.add(chunk_moves=1)
        if job.raw:
            self.remove_raw(job.raw)     # committed: the storage copy has served
            s.add(pulled=1)
        if self.exporter is not None:
            if self.exporter.add(job.cid, self.store.status(job.cid)["path"], tone_mode, folder):
                s.add(exported=1)
        s.time("write", time.perf_counter() - t)
        if self.captioner is not None:
            # after the photo is committed: a wait here (the VLM is behind) slows the stream, nothing else
            import autocaption as ac
            self.captioner.put(ac.Target(job.cid, path, image=final, tags=facts, scope=scope, base=o.base))
        s.add(done=1)
        d.advance(saved=True)
        d.line({"cid": job.cid, "state": DONE, "path": rel, "write": mode, "changed": plan["changed"],
                "colour": tone_mode,
                "class": verdict.cls if verdict else None})

    def record_json(self, job: Job, plan, boxes, mode, rel, data, sha, fw, fh, verdict, tone=None) -> str:
        o = self.opts
        p = job.photo
        if self.source is not None:
            import importer
            processing = {"borders": pr.describe_borders(job.borders) if job.borders else None,
                          "watermarks": {"detected": boxes, "painted": plan["wm_inside"], "trim": plan["trim"],
                                         "mode": o.wm_mode, "conf": o.wm_conf, "paint_conf": o.wm_paint_conf,
                                         "dilate": o.wm_dilate},
                          "box": plan["box"], "stored_box": plan["stored_box"], "changed": plan["changed"],
                          "write": mode, "colour": tone,
                          "output": {"path": rel, "bytes": len(data), "sha256": sha, "size": [fw, fh],
                                     "stored_size": list(enc.size_of(data)),
                                     "orientation": job.head["orientation"] if mode != "array" else 1}}
            return importer.record_json(job.cid, p["name"], p["src"], job.data, job.caption, processing,
                                        verdict.as_dict() if verdict else None, {**self.tool, "scope": job.scope},
                                        job.written)
        download = {"url": f"{IMG}/a/{p['file']}", "bytes": len(job.data), "sha256": hashlib.sha256(job.data).hexdigest(),
                    "content_type": job.ctype, "format": job.head["format"], "stored_size": job.head["stored"],
                    "orientation": job.head["orientation"], "mcu": job.head["mcu"]}
        processing = {"band": {"rows": job.rows, **(job.band or {})},
                      "borders": pr.describe_borders(job.borders) if job.borders else None,
                      "watermarks": {"detected": boxes, "painted": plan["wm_inside"], "trim": plan["trim"],
                                     "mode": o.wm_mode, "conf": o.wm_conf, "paint_conf": o.wm_paint_conf,
                                     "dilate": o.wm_dilate},
                      "box": plan["box"], "stored_box": plan["stored_box"], "changed": plan["changed"],
                      "write": mode,
                      "colour": tone,
                      "output": {"path": rel, "bytes": len(data), "sha256": sha, "size": [fw, fh],
                                 "stored_size": list(enc.size_of(data)),
                                 "orientation": job.head["orientation"] if mode != "array" else 1}}
        classification = verdict.as_dict() if verdict else None
        result = dict(job.result)
        if job.comments is not None:
            result["comments"] = job.comments
        return sidecars.record_json(job.cid, page_url(job.cid), result, download, processing, classification,
                                    {**self.tool, "scope": job.scope})

    def counters(self, job: Job, plan: dict, mode: str, folder: str, boxes: list, tone: dict | None = None) -> dict:
        kw = {"c.done": 1, f"c.write.{mode}": 1}
        if "band" in plan["changed"]:
            kw["c.band"] = 1
        if job.rows and not (job.band or {}).get("ok"):
            kw["c.band_unverified"] = 1
        if "borders" in plan["changed"]:
            kw["c.borders"] = 1
            for kind in {c["kind"] for c in ((job.borders or {}).get("result") or {}).get("cuts") or []}:
                kw[f"c.cut.{kind}"] = 1
        if (job.borders or {}).get("action") == "review":
            kw["c.borders_review"] = 1
        if plan["wm_inside"]:
            kw["c.wm_images"] = 1
            kw["c.wm_boxes"] = len(plan["wm_inside"])
        if plan["trim"]:
            kw["c.wm_trim"] = 1
        if boxes and not plan["wm_inside"] and not plan["trim"]:
            kw["c.wm_below"] = 1
        if folder:
            kw[f"cat.{folder}"] = 1
        if tone:
            kw[f"c.mode.{tone['mode']}"] = 1
            if tone.get("by") == "probe":
                kw["c.mode.probe"] = 1
        return kw

    def writer_worker(self, wq: queue.Queue) -> None:
        while True:
            item = wq.get()
            if item is None:
                break
            job = item[0]
            try:
                if self.opts.raw:
                    self.write_raw(job)
                else:
                    self.write(*item)
            except Exception as e:  # noqa: BLE001
                self.reject(job.cid, pr.Rejected(f"write: {type(e).__name__}: {e}", failed=True), job.scope)
                self.display.print(f"[write error] {job.cid}: {''.join(traceback.format_exception_only(e)).strip()}")

    # --- the run -------------------------------------------------------------------

    def stop_requested_by_captioner(self) -> bool:
        """The captioner stopped the run because the endpoint failed."""
        import autocaption as ac
        return self.captioner is not None and self.captioner.fails_in_row >= ac.MAX_CONSECUTIVE_FAILS

    def totals(self) -> tuple[dict, dict]:
        """The counters and the state counts of every scope of the run, summed."""
        from collections import Counter
        counters, counts = Counter(), Counter()
        for n in self.scopes:
            counters.update(self.store.counters(n))
            counts.update(self.store.scope_counts(n))
        return dict(counters), dict(counts)

    def report(self, t_start: float, final: bool = False) -> str:
        counters, counts = self.totals()
        if self.opts.raw:
            text = raw_report_block(counters, self.stats, counts, time.time() - t_start, final)
        else:
            text = report_block(counters, self.stats, counts, time.time() - t_start, final)
        if self.storage_source:
            text += (f"\n  from the storage: {int(self.stats.c['pulled']):,} processed this run, "
                     f"{self.store.stored_count():,} left in it")
        text += "\n" + self.interval_report()
        if self.caption_client is not None:
            text += f"\n  captioned by the VLM in this run: {int(self.stats.c['captioned']):,}"
        if self.vram is not None and self.vram.ok:
            text += "\n  " + self.vram.report()
        if final and len(self.scopes) > 1:
            for n in sorted(self.scopes):
                c = self.store.scope_counts(n)
                k = self.store.counters(n)
                stored = f"  {c['stored']:>6,} stored" if self.opts.raw or c["stored"] else ""
                modes = "" if self.opts.raw else \
                    f"  bw {int(k.get('c.mode.bw', 0)):,} / colour {int(k.get('c.mode.color', 0)):,}"
                text += (f"\n    {n:32} {c['done']:>7,} saved{stored}  {c['skipped']:>6,} skipped  "
                         f"{c['failed']:>4,} failed  {c['elsewhere']:>6,} in other scopes  {c['pending']:>7,} left"
                         f"{modes}")
        return text

    def interval_report(self) -> str:
        """What happened since the last report block: the rate, where the main thread's
        time went, the downloads' latencies and the threads in a request, the storage
        side, the writers. Also one JSON line (--json-progress) with the same numbers."""
        snap = self.stats.interval()
        stats = getattr(self.client, "stats", None)
        if isinstance(stats, dict):
            r = int(stats.get("retries", 0))
            snap["c"]["retries"] = r - self.retries_seen
            self.retries_seen = r
        now = time.time()
        with self.lock:
            ages = [now - t0 for t0 in self.in_request.values()]
        live = {"writers": self.opts.writers, "in_request": len(ages), "oldest_request": max(ages) if ages else 0.0,
                "storage_threads": self.storage_threads_n}
        text, numbers = interval_block(snap, live)
        if self.display is not None:
            self.display.line({"interval": numbers})
        return text

    def put_to_writers(self, wq: queue.Queue, item: tuple) -> None:
        t = time.perf_counter()
        wq.put(item)
        self.stats.time("wait_writers", time.perf_counter() - t)

    def run(self, pairs: list[tuple[int, str]], scope_total: int, completed: int,
            stored: list[tuple[int, str]] = ()) -> dict:
        """pairs: (cid, scope) to download; stored: (cid, scope) to take from the storage
        whenever no download is ready. scope_total and completed size the progress bar."""
        o = self.opts
        stored = list(stored)
        if o.raw and stored:
            raise ValueError("a raw run does not read the storage")
        self.budget = max(0, o.limit) if o.limit is not None else None
        self.storage_source = bool(stored)
        if self.client is not None and hasattr(self.client, "stop"):
            self.client.stop = self.stop         # Ctrl+C: no request is retried, photos in flight stay pending
        it, sit = iter(pairs), iter(stored)
        cid_lock = threading.Lock()
        out_q: queue.Queue = queue.Queue(maxsize=max(1, o.depth))
        sq: queue.Queue = queue.Queue(maxsize=max(1, o.storage_depth, o.storage_threads))
        wq: queue.Queue = queue.Queue(maxsize=max(2, o.writers * 2))
        run_id = self.store.start_run(self.scope, {"cids": len(pairs), "stored": len(stored),
                                                   "scopes": sorted(self.scopes),
                                                   **{k: str(v) for k, v in vars(o).items()
                                                      if k not in ("scope", "scopes")}})
        t_start = time.time()
        processed = 0
        with Display(scope_total, completed, o.display, vram=self.vram) as d:
            self.display = d
            if self.caption_client is not None:
                import autocaption as ac
                self.captioner = ac.StreamCaptioner(self.store, self.caption_client, self.stats,
                                                    lambda: self.display, self.stop)
            where = self.scope if len(self.scopes) == 1 else f"{self.scope} ({len(self.scopes)} scopes)"
            what = ("to download into the storage" if o.raw else "to download") if pairs or not stored else ""
            text = f"{where}: {len(pairs):,} photos {what}" if what else f"{where}:"
            if stored:
                text += f"{' and' if what else ''} {len(stored):,} photos in the storage to process"
            if o.limit is not None:
                text += f", at most {o.limit:,} in this run"
            d.print(f"{text}; {completed:,} of {scope_total:,} done, skipped or saved under another scope before.")
            fetchers = [threading.Thread(target=self.fetch_worker, args=(it, cid_lock, out_q), daemon=True,
                                         name=f"fetch-{i}") for i in range(o.fetchers)] if pairs else []
            storages = [threading.Thread(target=self.storage_worker, args=(sit, cid_lock, sq), daemon=True,
                                         name=f"storage-{i}") for i in range(max(1, o.storage_threads))] \
                if stored else []
            writers = [threading.Thread(target=self.writer_worker, args=(wq,), daemon=True, name=f"write-{i}")
                       for i in range(o.writers)]
            for t in fetchers + writers + storages:
                t.start()
            self.storage_threads_n = len(storages)
            ended, sended, interrupts = 0, 0, 0
            remote_ended, storage_ended = not fetchers, not storages
            try:
                while not (remote_ended and storage_ended):
                    try:
                        # downloads first; a stored photo only when no download is ready
                        t_loop = time.perf_counter()
                        item, source = None, None
                        if not remote_ended:
                            try:
                                item = out_q.get_nowait() if not storage_ended else out_q.get(timeout=0.5)
                                source = "download"
                            except queue.Empty:
                                pass
                        if source is None and not storage_ended:
                            try:
                                item = sq.get(timeout=0.5)
                                source = "storage"
                            except queue.Empty:
                                pass
                        self.stats.time("wait_input", time.perf_counter() - t_loop)
                        if source is None:
                            with self.lock:
                                n = self.in_flight
                            d.set(f"waiting for downloads ({n} in flight)" if not self.stop.is_set()
                                  else "stopping: finishing the images in flight")
                            continue
                        if item is None:
                            if source == "download":
                                ended += 1
                                remote_ended = ended >= len(fetchers)
                            else:
                                sended += 1
                                storage_ended = sended >= len(storages)
                            continue
                        if isinstance(item, pr.Rejected):
                            self.reject(item.cid, item, item.scope)
                        elif o.raw:
                            self.stats.time("fetch", item.t_fetch)
                            self.put_to_writers(wq, (item,))
                        else:
                            self.stats.time("pull" if source == "storage" else "fetch", item.t_fetch)
                            t_stage = time.perf_counter()
                            res = None
                            try:
                                res = self.process(item)
                                item.array = None
                            except pr.Rejected as r:
                                self.reject(item.cid, r, item.scope)
                            except Exception as e:  # noqa: BLE001
                                self.reject(item.cid, pr.Rejected(f"processing: {type(e).__name__}: {e}", failed=True),
                                            item.scope)
                                d.print(f"[error] {item.cid}: {type(e).__name__}: {e}")
                                if "out of memory" in str(e).lower():
                                    import torch
                                    torch.cuda.empty_cache()
                            self.stats.time("stage", time.perf_counter() - t_stage)
                            if res is not None:
                                self.put_to_writers(wq, (item,) + res)
                        processed += 1
                        if o.report_every and processed % o.report_every == 0:
                            d.print(self.report(t_start))
                    except KeyboardInterrupt:
                        interrupts += 1
                        if interrupts == 1:
                            self.stop.set()
                            d.print("Ctrl+C: finishing the images already fetched, then stopping. "
                                    "Press Ctrl+C again to stop at once.")
                        else:
                            raise
            except KeyboardInterrupt:
                d.print("Stopped. The images being written are finished; everything else is left for the next run.")
            finally:
                self.stop.set()
                d.set("finishing the writes")
                for _ in writers:
                    wq.put(None)
                for t in writers:
                    t.join()
                if self.captioner is not None:
                    d.set("finishing the captions")
                    self.captioner.close()
                if storages:
                    folders.prune_empty(o.base / layout.STORAGE_DIR)
                d.set("done")
                d.print(self.report(t_start, final=True))
        c = self.stats.c
        failed = sum(v for k, v in c.items() if k.startswith("failed."))
        skipped = sum(v for k, v in c.items() if k.startswith("skipped."))
        self.store.end_run(run_id, int(c["done"] + c["stored"]), int(skipped), int(failed))
        return {"done": int(c["done"]), "stored": int(c["stored"]), "pulled": int(c["pulled"]),
                "skipped": int(skipped), "failed": int(failed), "seconds": time.time() - t_start}
