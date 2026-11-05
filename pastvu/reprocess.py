"""reprocess --downscale: the saved photos of a base with more pixels than the limit
are rewritten at the limit's area, in place, by the stream's own rule
(process.downscale_size, Lanczos) and its own encoder (encode.array_bytes: JPEG q97
without chroma subsampling, or PNG; EXIF with the orientation reset and the new
size, ICC kept). A saved PNG stays PNG, a JPEG stays JPEG; --png makes every one
PNG.

- The photos are chosen from the store's width and height; no file is opened to decide.
- The file is replaced first (a temporary name, then os.replace), then the row
  (path, size, bytes, sha256, changed) and the record (processing.downscale,
  write "downscaled", output) in one transaction; the scope's write counter moves.
  A killed run leaves every photo either untouched or fully updated, and the size
  rule skips the rewritten ones next time.
- The latent and text-encoder caches of the photo (.npz, .safetensors) are deleted,
  since they were made from the old pixels; captions stay.
- The live export shares the file by a hard link, which os.replace breaks: the
  export's link is made again from the new file.
- The classifier's embedding stays (from a 576-patch resize of the pixels: the
  downscaled ones give the same to the fourth decimal).
- Reading, resizing and encoding run in worker processes; the store and the
  export are updated in the main process as results come in.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import layout

MODE = "downscaled"


def candidates(store, scopes: list[str] | None, side: int) -> list[dict]:
    """The saved photos over side^2 pixels, from the store."""
    return [{"cid": cid, "scope": scope, "path": path, "width": w, "height": h, "bytes": size or 0}
            for cid, scope, path, w, h, size in store.oversized_rows(scopes, side * side)]


def shrink_file(task: dict) -> dict:
    """In a worker process: read the photo, downscale it to the area, write it under the
    same stem (the extension follows the format), replacing the old file.
    task: {"cid", "path" (absolute), "area", "png"} -> {"cid", "path", "width", "height",
    "bytes", "sha256", "from": [w, h], "error"}"""
    import encode as enc
    import process as pr
    src = Path(task["path"])
    out = {"cid": task["cid"], "path": str(src), "error": ""}
    try:
        data = src.read_bytes()
        head = pr.head_of(data)
        array = pr.decode(data, head)
        h, w = array.shape[:2]
        out["from"] = [w, h]
        small = pr.downscale(array, task["area"])
        if small is array:
            out["error"] = "fits already"
            return out
        png = task["png"] or head["format"] == "PNG"
        head = dict(head, stored=[small.shape[1], small.shape[0]], orientation=1, mcu=None)
        bytes_, fmt = enc.array_bytes(small, head, png)
        dst = src.with_suffix(enc.EXTENSIONS[fmt])
        tmp = src.with_name(f"~{secrets.token_hex(4)}{dst.suffix}")
        tmp.write_bytes(bytes_)
        os.replace(tmp, dst)
        if dst != src:
            src.unlink(missing_ok=True)
        out.update(path=str(dst), width=small.shape[1], height=small.shape[0], bytes=len(bytes_),
                   sha256=hashlib.sha256(bytes_).hexdigest())
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
    p["write"] = MODE
    p["changed"] = changed
    output = dict(p.get("output") or {})
    output.update(path=rel, bytes=res["bytes"], sha256=res["sha256"], size=[res["width"], res["height"]],
                  stored_size=[res["width"], res["height"]], orientation=1)
    p["output"] = output
    rec["processing"] = p
    return json.dumps(rec, ensure_ascii=False, indent=1, default=str), changed


def apply(store, base: Path, items: list[dict], side: int, png: bool = False, workers: int = 0, exporter=None,
          log=print, progress=None) -> dict:
    """Downscale the photos of `items` (candidates()). workers 0: in this process.
    progress: called after each photo with (done, total). -> counts"""
    area = side * side
    tasks = [{"cid": it["cid"], "path": str(base / it["path"]), "area": area, "png": png} for it in items]
    by_cid = {it["cid"]: it for it in items}
    counts = {"done": 0, "failed": 0, "fits": 0, "before": 0, "after": 0, "caches": 0, "relinked": 0}
    if workers > 0:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(shrink_file, tasks, chunksize=4)
    else:
        pool = None
        results = map(shrink_file, tasks)
    try:
        for n, res in enumerate(results, 1):
            it = by_cid[res["cid"]]
            if res["error"] == "fits already":
                counts["fits"] += 1
            elif res["error"]:
                counts["failed"] += 1
                log(f"  {res['cid']}: {res['error']}")
            else:
                rel = Path(res["path"]).relative_to(base).as_posix()
                record, changed = record_after(store.record(res["cid"]), res, rel, side)
                old_mode = ((store.record(res["cid"]) or {}).get("processing") or {}).get("write") or "lossless"
                store.resize(res["cid"], rel, res["width"], res["height"], res["bytes"], res["sha256"], changed, record)
                if old_mode != MODE:
                    store.shift_counters(it["scope"], {f"c.write.{old_mode}": -1, f"c.write.{MODE}": 1})
                img = base / rel
                for s in layout.sidecars(img):
                    if s.suffix.lower() in layout.CACHE_EXTS:
                        s.unlink(missing_ok=True)
                        counts["caches"] += 1
                if exporter is not None and exporter.relink(res["cid"], rel):
                    counts["relinked"] += 1
                counts["done"] += 1
                counts["before"] += it["bytes"]
                counts["after"] += res["bytes"]
            if progress is not None:
                progress(n, len(tasks))
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    return counts
