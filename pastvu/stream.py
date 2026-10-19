"""The stream: every photo of a scope that is not done yet goes through
fetch, band and borders (fetch threads), watermark detection, painting and
classification (the main thread, models resident on the GPU), and encoding
and writing (writer threads). Images, plans and output bytes stay in memory.

Writing one photo: its path is claimed in the store (state writing), the
image and its caption are written, then the status, the record, the
embedding and the counters are committed in one transaction. A photo found
in state writing at the start of a run was interrupted: its files are
deleted and it is done again.

Ctrl+C finishes the images already fetched and stops; a second Ctrl+C stops
at once. Either way the store says what is done, and the next run of the same
command continues with the rest.
"""

from __future__ import annotations

import hashlib
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import encode as enc
import layout
import process as pr
import sidecars
from client import IMG, Client, PastvuError, TransportError, page_url
from progress import Display, Stats, report_block
from store import DONE, FAILED, SKIPPED, Store

TOOL_VERSION = "0.2"


@dataclass
class Options:
    base: Path
    scope: dict
    min_side: int = 512
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
    display: str = "live"

    @property
    def scope_dir(self) -> Path:
        return self.base / self.scope["name"]


@dataclass
class Job:
    cid: int
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
    t_fetch: float = 0.0


def cleanup_interrupted(store: Store, base: Path, scope: str) -> int:
    """Photos left in state writing by a killed run: delete their files, make them pending."""
    rows = store.writing_rows(scope)
    for cid, path in rows:
        if path:
            img = base / path
            for f in (img, img.with_suffix(".txt"), img.with_suffix(".json")):
                f.unlink(missing_ok=True)
        store.unclaim(cid)
    return len(rows)


class Stream:
    def __init__(self, client: Client, store: Store, opts: Options, classifier=None, detector=None, lama=None):
        self.client, self.store, self.opts = client, store, opts
        self.classifier = classifier
        self.det_name, self.det = detector or (None, None)
        self.lama = lama
        self.scope = opts.scope["name"]
        self.chunker = layout.Chunker(opts.scope["split"], opts.scope["width"],
                                      [self.scope_rel(p) for p in store.scope_paths(self.scope)])
        self.stats = Stats()
        self.stop = threading.Event()
        self.in_flight = 0
        self.lock = threading.Lock()
        self.display: Display | None = None
        enc_model = getattr(getattr(classifier, "encoder", None), "model_id", None)
        self.emb_key = (enc_model, getattr(classifier, "patches", None))
        self.tool = {"name": "dataset_tools/pastvu", "version": TOOL_VERSION, "detector": self.det_name,
                     "detector_conf": opts.wm_conf, "paint_conf": opts.wm_paint_conf, "encoder": enc_model,
                     "patches": getattr(classifier, "patches", None)}

    def scope_rel(self, base_rel: str) -> str:
        """A path relative to the base -> relative to the scope folder."""
        prefix = self.scope + "/"
        return base_rel[len(prefix):] if base_rel.startswith(prefix) else base_rel

    # --- fetch threads -------------------------------------------------------------

    def fetch(self, cid: int) -> Job:
        o = self.opts
        t0 = time.perf_counter()
        job = Job(cid)
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
        if not p.get("file"):
            raise pr.Rejected("no file in the record", failed=True)
        job.data, job.ctype = self.client.image(p["file"], "a")
        if not job.ctype.startswith("image/"):
            raise pr.Rejected(f"download: not an image ({job.ctype or 'no type'})", failed=True)
        try:
            job.head = pr.head_of(job.data)
        except Exception as e:  # noqa: BLE001
            raise pr.Rejected(f"download: unreadable image ({type(e).__name__})", failed=True)
        if job.head["frames"] > 1:
            raise pr.Rejected("animated")
        job.rows = pr.band_rows(job.head, p)
        job.array = pr.decode(job.data, job.head)
        job.band = pr.verify_band(job.array, job.head, job.rows)
        job.pic_box = pr.picture_box(job.head, job.rows)
        if o.borders:
            picture = pr.crop_array(job.array, job.pic_box)
            job.borders = pr.borders_plan(picture, {"dark": o.dark_borders})
        if o.comments:
            try:
                job.comments = self.client.comments(cid)
            except (PastvuError, TransportError) as e:
                job.comments = {"error": str(e)}
        job.t_fetch = time.perf_counter() - t0
        return job

    def fetch_worker(self, cids, cid_lock, out_q: queue.Queue) -> None:
        while not self.stop.is_set():
            with cid_lock:
                cid = next(cids, None)
            if cid is None:
                break
            with self.lock:
                self.in_flight += 1
            try:
                item = self.fetch(cid)
            except pr.Rejected as r:
                item = r
            except TransportError as e:
                item = pr.Rejected(f"download: {e}", failed=True)
            except Exception as e:  # noqa: BLE001 - one bad photo never stops the stream
                item = pr.Rejected(f"fetch: {type(e).__name__}: {e}", failed=True)
            finally:
                with self.lock:
                    self.in_flight -= 1
            if isinstance(item, pr.Rejected):
                item.cid = cid
            out_q.put(item)
        out_q.put(None)

    # --- outcomes ------------------------------------------------------------------

    def reject(self, cid: int, r: pr.Rejected) -> None:
        key = r.reason.split(":")[0].split(" (")[0]
        state = FAILED if r.failed else SKIPPED
        self.store.reject(cid, self.scope, state, r.reason, {f"c.{state}.{key}": 1})
        self.stats.add(**{f"{state}.{key}": 1})
        self.display.advance(saved=False)
        self.display.line({"cid": cid, "state": state, "reason": r.reason})

    # --- the GPU stage -------------------------------------------------------------

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
        crop = pr.crop_array(job.array, plan["box"])
        painted = None
        if plan["wm_inside"]:
            d.set(f"{job.cid}: painting {len(plan['wm_inside'])} watermark box(es)")
            t = time.perf_counter()
            painted = wm.inpaint(crop, plan["wm_inside"], o.wm_dilate, o.wm_max_size, self.lama)
            s.time("paint", time.perf_counter() - t)
        final = painted if painted is not None else crop
        vec = verdict = None
        if self.classifier is not None:
            d.set(f"{job.cid}: classifying")
            t = time.perf_counter()
            vec = self.classifier.embed(final)
            verdict = self.classifier.decide(vec)
            s.time("classify", time.perf_counter() - t)
        return plan, boxes, painted, vec, verdict, final.shape[1], final.shape[0]

    # --- writer threads ------------------------------------------------------------

    def write(self, job: Job, plan: dict, boxes: list, painted, vec, verdict, fw: int, fh: int) -> None:
        o, s, d = self.opts, self.stats, self.display
        d.set(f"{job.cid}: encoding")
        t = time.perf_counter()
        data, ext, mode = enc.encode(job.data, job.head, plan, painted, o.png)
        s.time("encode", time.perf_counter() - t)
        if self.classifier is not None:
            verdict = self.classifier.example_override(data, verdict)
        folder = verdict.folder if verdict else ""
        rel_scope = self.chunker.place(folder, f"{job.cid}{ext}")
        rel = f"{self.scope}/{rel_scope}"
        path = o.base / rel
        d.set(f"{job.cid}: saving {rel_scope}")
        t = time.perf_counter()
        self.store.claim(job.cid, self.scope, rel)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.with_suffix(".txt").write_text(sidecars.caption(job.photo, o.txt_fields), encoding="utf-8")
            sha = hashlib.sha256(data).hexdigest()
            doc = self.record_json(job, plan, boxes, mode, rel, data, sha, fw, fh, verdict)
            if o.json_files:
                rf = layout.record_file(o.base, job.cid)
                rf.parent.mkdir(parents=True, exist_ok=True)
                rf.write_text(doc, encoding="utf-8")
            if o.json_sidecar:
                path.with_suffix(".json").write_text(doc, encoding="utf-8")
            emb = (self.emb_key[0], self.emb_key[1], vec) if vec is not None else None
            counters = self.counters(job, plan, mode, folder, boxes)
            self.store.finish(job.cid, self.scope, {"category": folder, "path": rel, "sha256": sha, "bytes": len(data),
                                                    "width": fw, "height": fh, "changed": plan["changed"]},
                              doc, emb, counters)
        except BaseException:
            self.chunker.release(rel_scope)
            for f in (path, path.with_suffix(".txt"), path.with_suffix(".json")):
                f.unlink(missing_ok=True)
            self.store.unclaim(job.cid)
            raise
        s.time("write", time.perf_counter() - t)
        s.add(done=1)
        d.advance(saved=True)
        d.line({"cid": job.cid, "state": DONE, "path": rel, "write": mode, "changed": plan["changed"],
                "class": verdict.cls if verdict else None})

    def record_json(self, job: Job, plan, boxes, mode, rel, data, sha, fw, fh, verdict) -> str:
        o = self.opts
        p = job.photo
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
                      "output": {"path": rel, "bytes": len(data), "sha256": sha, "size": [fw, fh],
                                 "stored_size": list(enc.size_of(data)),
                                 "orientation": job.head["orientation"] if mode != "array" else 1}}
        classification = None
        if verdict:
            classification = {"folder": verdict.folder, "class": verdict.cls, "subclass": verdict.sub,
                              "confidence": round(verdict.confidence, 4), "share": round(verdict.share, 2),
                              "vote": verdict.vote, "reason": verdict.reason}
        result = dict(job.result)
        if job.comments is not None:
            result["comments"] = job.comments
        return sidecars.record_json(job.cid, page_url(job.cid), result, download, processing, classification,
                                    {**self.tool, "scope": self.scope})

    def counters(self, job: Job, plan: dict, mode: str, folder: str, boxes: list) -> dict:
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
        return kw

    def writer_worker(self, wq: queue.Queue) -> None:
        while True:
            item = wq.get()
            if item is None:
                break
            job = item[0]
            try:
                self.write(*item)
            except Exception as e:  # noqa: BLE001
                self.reject(job.cid, pr.Rejected(f"write: {type(e).__name__}: {e}", failed=True))
                self.display.print(f"[write error] {job.cid}: {''.join(traceback.format_exception_only(e)).strip()}")

    # --- the run -------------------------------------------------------------------

    def report(self, t_start: float, final: bool = False) -> str:
        counts = self.store.scope_counts(self.scope)
        return report_block(self.store.counters(self.scope), self.stats, counts, time.time() - t_start, final)

    def run(self, cids: list[int], scope_total: int, completed: int) -> dict:
        o = self.opts
        if o.limit:
            cids = cids[:o.limit]
        it = iter(cids)
        cid_lock = threading.Lock()
        out_q: queue.Queue = queue.Queue(maxsize=max(1, o.depth))
        wq: queue.Queue = queue.Queue(maxsize=max(2, o.writers * 2))
        run_id = self.store.start_run(self.scope, {"cids": len(cids), **{k: str(v) for k, v in vars(o).items()
                                                                          if k != "scope"}})
        t_start = time.time()
        processed = 0
        with Display(scope_total, completed, o.display) as d:
            self.display = d
            d.print(f"{self.scope}: {len(cids):,} photos to do in this run; {completed:,} of {scope_total:,} done, "
                    f"skipped or saved under another scope before.")
            fetchers = [threading.Thread(target=self.fetch_worker, args=(it, cid_lock, out_q), daemon=True,
                                         name=f"fetch-{i}") for i in range(o.fetchers)]
            writers = [threading.Thread(target=self.writer_worker, args=(wq,), daemon=True, name=f"write-{i}")
                       for i in range(o.writers)]
            for t in fetchers + writers:
                t.start()
            ended, interrupts = 0, 0
            try:
                while ended < len(fetchers):
                    try:
                        try:
                            item = out_q.get(timeout=0.5)
                        except queue.Empty:
                            with self.lock:
                                n = self.in_flight
                            d.set(f"waiting for downloads ({n} in flight)" if not self.stop.is_set()
                                  else "stopping: finishing the images in flight")
                            continue
                        if item is None:
                            ended += 1
                            continue
                        if isinstance(item, pr.Rejected):
                            self.reject(item.cid, item)
                        else:
                            self.stats.time("fetch", item.t_fetch)
                            try:
                                res = self.process(item)
                                item.array = None
                                wq.put((item,) + res)
                            except pr.Rejected as r:
                                self.reject(item.cid, r)
                            except Exception as e:  # noqa: BLE001
                                self.reject(item.cid, pr.Rejected(f"processing: {type(e).__name__}: {e}", failed=True))
                                d.print(f"[error] {item.cid}: {type(e).__name__}: {e}")
                                if "out of memory" in str(e).lower():
                                    import torch
                                    torch.cuda.empty_cache()
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
                d.set("done")
                d.print(self.report(t_start, final=True))
        c = self.stats.c
        failed = sum(v for k, v in c.items() if k.startswith("failed."))
        skipped = sum(v for k, v in c.items() if k.startswith("skipped."))
        self.store.end_run(run_id, int(c["done"]), int(skipped), int(failed))
        return {"done": int(c["done"]), "skipped": int(skipped), "failed": int(failed),
                "seconds": time.time() - t_start}
