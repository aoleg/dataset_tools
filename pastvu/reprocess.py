"""reprocess --downscale: the saved photos of a base with more pixels than the limit
are rewritten at the limit's area, in place, by the stream's own rule
(process.downscale_size, Lanczos) and its own encoder (encode.array_bytes: JPEG at
jpeg_quality_downscaled without chroma subsampling, or PNG; EXIF with the
orientation reset and the new size, ICC kept). A saved PNG stays PNG, a JPEG stays
JPEG; --png makes every one PNG.

- The photos are chosen from the store's width and height; no file is opened to decide.
- The file is replaced first (a temporary ~<hex> name, then os.replace), then the
  row (path, size, bytes, sha256, changed) and the record (processing.downscale,
  write "downscaled", output with the quality) in one transaction; the scope's
  write counter moves.
- A photo whose file is already within the limit while its row is not was
  rewritten by a run that stopped before the row: its row, record, caches and
  export link are brought up to date from the file as it is ("repaired").
  Temporary files such a run left in the photos' folders are deleted first.
- The latent and text-encoder caches of the photo (.npz, .safetensors) are deleted,
  since they were made from the old pixels; captions stay.
- The live export shares the file by a hard link, which os.replace breaks: the
  export's link is made again from the new file.
- The classifier's embedding stays (from a 576-patch resize of the pixels: the
  downscaled ones give the same to the fourth decimal).
- Reading, resizing and encoding run in worker processes that ignore Ctrl+C; at
  most four photos per worker are handed out ahead. The first Ctrl+C lets the
  photos in hand finish and be recorded, then stops; a second stops at once.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import signal
import sys
import threading
import time
from pathlib import Path

import encode as enc
import layout

MODE = "downscaled"
TEMP_NAME = re.compile(r"^~[0-9a-f]{8}\.(jpg|png)$")
AHEAD = 4                       # photos handed to each worker ahead of its results


def candidates(store, scopes: list[str] | None, side: int) -> list[dict]:
    """The saved photos over side^2 pixels, from the store."""
    return [{"cid": cid, "scope": scope, "path": path, "width": w, "height": h, "bytes": size or 0}
            for cid, scope, path, w, h, size in store.oversized_rows(scopes, side * side)]


def sweep_temp(base: Path, items: list[dict]) -> int:
    """Delete the temporary files (~<8 hex>.jpg/.png) a stopped run left in the folders of
    these photos. Only reprocess makes such names. -> deleted"""
    n = 0
    for folder in sorted({(base / it["path"]).parent for it in items}):
        names = layout._find_files(os.path.join(str(folder), "~*")) if sys.platform == "win32" else None
        if names is None:                            # not Windows, or no name index: list the folder
            try:
                names = [e.name for e in os.scandir(folder) if e.is_file()]
            except OSError:
                names = []
        for name in names:
            if TEMP_NAME.match(name):
                (folder / name).unlink(missing_ok=True)
                n += 1
    return n


def _ignore_sigint() -> None:
    """A worker process leaves Ctrl+C to the main process, which finishes the photos in hand."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def shrink_file(task: dict) -> dict:
    """In a worker process: read the photo, downscale it to the area, write it under the
    same stem (the extension follows the format), replacing the old file. A file already
    within the area is reported as it is ("already": an earlier run rewrote it).
    task: {"cid", "path" (absolute), "area", "png", "quality"} -> {"cid", "path", "width",
    "height", "bytes", "sha256", "from": [w, h] (when downscaled now), "quality", "already",
    "error"}"""
    import process as pr
    src = Path(task["path"])
    out = {"cid": task["cid"], "path": str(src), "error": "", "already": False, "quality": None}
    try:
        if not src.is_file():          # rewritten under the other extension by a run that stopped
            alt = next((p for p in (src.with_suffix(".png"), src.with_suffix(".jpg")) if p != src and p.is_file()), None)
            if alt is None:
                out["error"] = "file missing"
                return out
            src = alt
            out["path"] = str(src)
        data = src.read_bytes()
        head = pr.head_of(data)
        w, h = pr.upright_size(head)
        if w * h <= task["area"]:
            out.update(already=True, width=w, height=h, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            return out
        array = pr.decode(data, head)
        out["from"] = [array.shape[1], array.shape[0]]
        small = pr.downscale(array, task["area"])
        png = task["png"] or head["format"] == "PNG"
        head = dict(head, stored=[small.shape[1], small.shape[0]], orientation=1, mcu=None)
        bytes_, fmt = enc.array_bytes(small, head, png, task["quality"])
        dst = src.with_suffix(enc.EXTENSIONS[fmt])
        tmp = src.with_name(f"~{secrets.token_hex(4)}{dst.suffix}")
        tmp.write_bytes(bytes_)
        os.replace(tmp, dst)
        if dst != src:
            src.unlink(missing_ok=True)
        out.update(path=str(dst), width=small.shape[1], height=small.shape[0], bytes=len(bytes_),
                   sha256=hashlib.sha256(bytes_).hexdigest(), quality=None if fmt == "PNG" else task["quality"])
    except Exception as e:  # noqa: BLE001 - one unreadable photo never stops the rest
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def record_after(rec: dict | None, res: dict, rel: str, side: int) -> tuple[str, list]:
    """The record and the status' changed list after the downscale."""
    rec = dict(rec or {})
    p = dict(rec.get("processing") or {})
    old_write = p.get("write", "")
    changed = [c for c in (p.get("changed") or []) if c != "downscale"] + ["downscale"]
    p["downscale"] = {"from": res["from"], "to": [res["width"], res["height"]], "limit": side, "band_rows": 0,
                      "reprocessed": time.time(), "was": old_write}
    if res.get("already"):
        p["downscale"]["repaired"] = True        # the file was rewritten by a run that stopped before this record
    p["write"] = MODE
    p["changed"] = changed
    output = dict(p.get("output") or {})
    output.update(path=rel, bytes=res["bytes"], sha256=res["sha256"], size=[res["width"], res["height"]],
                  stored_size=[res["width"], res["height"]], orientation=1, quality=res.get("quality"))
    p["output"] = output
    rec["processing"] = p
    return json.dumps(rec, ensure_ascii=False, indent=1, default=str), changed


def apply(store, base: Path, items: list[dict], side: int, png: bool = False, workers: int = 0, exporter=None,
          log=print, progress=None, quality: int = enc.JPEG_QUALITY_DOWNSCALED,
          stop: threading.Event | None = None) -> dict:
    """Downscale the photos of `items` (candidates()). workers 0: in this process.
    progress: called after each photo with (done, total). stop: set to stop after the
    photos in hand (Ctrl+C sets it). -> counts, with "stopped" """
    area = side * side
    stop = stop or threading.Event()
    tasks = [{"cid": it["cid"], "path": str(base / it["path"]), "area": area, "png": png, "quality": quality}
             for it in items]
    by_cid = {it["cid"]: it for it in items}
    counts = {"done": 0, "failed": 0, "repaired": 0, "before": 0, "after": 0, "caches": 0, "relinked": 0,
              "stopped": False}
    seen = 0

    def record(res: dict) -> None:
        nonlocal seen
        seen += 1
        it = by_cid[res["cid"]]
        if res["error"]:
            counts["failed"] += 1
            log(f"  {res['cid']}: {res['error']}")
        else:
            res.setdefault("from", [it["width"], it["height"]])
            rel = Path(res["path"]).relative_to(base).as_posix()
            old = store.record(res["cid"]) or {}
            old_mode = (old.get("processing") or {}).get("write") or "lossless"
            record_json, changed = record_after(old, res, rel, side)
            store.resize(res["cid"], rel, res["width"], res["height"], res["bytes"], res["sha256"], changed, record_json)
            if old_mode != MODE:
                store.shift_counters(it["scope"], {f"c.write.{old_mode}": -1, f"c.write.{MODE}": 1})
            img = base / rel
            for s in layout.sidecars(img):
                if s.suffix.lower() in layout.CACHE_EXTS:
                    s.unlink(missing_ok=True)
                    counts["caches"] += 1
            if exporter is not None and exporter.relink(res["cid"], rel):
                counts["relinked"] += 1
            counts["repaired" if res["already"] else "done"] += 1
            counts["before"] += it["bytes"]
            counts["after"] += res["bytes"]
        if progress is not None:
            progress(seen, len(tasks))

    def on_sigint(signum, frame):
        if stop.is_set():
            raise KeyboardInterrupt
        stop.set()
        log("Ctrl+C: finishing the photos in hand, then stopping. Press Ctrl+C again to stop at once.")

    try:
        old_handler = signal.signal(signal.SIGINT, on_sigint)
    except ValueError:                       # not the main thread: no handler, stop still works
        old_handler = None
    try:
        if workers <= 0:
            for task in tasks:
                if stop.is_set():
                    break
                record(shrink_file(task))
        else:
            from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
            pending = iter(tasks)
            with ProcessPoolExecutor(max_workers=workers, initializer=_ignore_sigint) as pool:
                futures = set()

                def top_up() -> None:
                    while not stop.is_set() and len(futures) < workers * AHEAD:
                        task = next(pending, None)
                        if task is None:
                            return
                        futures.add(pool.submit(shrink_file, task))

                top_up()
                while futures:
                    done, _ = wait(futures, timeout=1.0, return_when=FIRST_COMPLETED)
                    for f in done:
                        futures.discard(f)
                        record(f.result())
                    top_up()
    finally:
        if old_handler is not None:
            signal.signal(signal.SIGINT, old_handler)
    counts["stopped"] = stop.is_set() and seen < len(tasks)
    return counts
