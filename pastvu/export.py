"""export: the records of a scope as JSON Lines (one full record per line) or
Parquet (one row per photo: the main fields as columns, the full record as a
JSON string), for publishing a dataset or for analysis."""

from __future__ import annotations

import json
from pathlib import Path

import sidecars
from store import Store


def rows(store: Store, scope: str):
    for cid, path, category, sha in store.done_rows(scope):
        rec = store.record(cid)
        if rec is None:
            continue
        cap = store.caption(cid)
        if cap:
            rec = {**rec, "autocaption": {"caption": cap["caption"], "original": cap["original"], "model": cap["model"],
                                          "prompt_sha": cap["prompt_sha"]}}
        yield cid, path, category, rec


def flat(cid: int, path: str, category: str, rec: dict) -> dict:
    p = rec.get("record") or {}
    cls = rec.get("classification") or {}
    geo = p.get("geo") or [None, None]
    regions = p.get("regions") or []
    out = rec.get("processing", {}).get("output", {})
    return {
        "cid": cid,
        "path": path,
        "category": category,
        "confidence": cls.get("confidence"),
        "page": rec.get("page"),
        "title": p.get("title"),
        "year": p.get("year"),
        "year2": p.get("year2"),
        "lat": geo[0] if len(geo) > 0 else None,
        "lon": geo[1] if len(geo) > 1 else None,
        "direction": p.get("dir"),
        "regions_en": " > ".join(r.get("title_en", "") for r in regions),
        "regions_local": " > ".join(r.get("title_local", "") for r in regions),
        "address": p.get("address"),
        "author": p.get("author"),
        "source": sidecars.clean(p.get("source")),
        "uploader": (p.get("user") or {}).get("login"),
        "width": (out.get("size") or [None, None])[0],
        "height": (out.get("size") or [None, None])[1],
        "sha256": out.get("sha256"),
        "caption": sidecars.caption(p),
        "vlm_caption": (rec.get("autocaption") or {}).get("caption"),
        "record": json.dumps(rec, ensure_ascii=False, default=str),
    }


def export(store: Store, scope: str, dst: Path, fmt: str) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    if fmt == "jsonl":
        with open(dst, "w", encoding="utf-8", newline="\n") as f:
            for cid, path, category, rec in rows(store, scope):
                f.write(json.dumps({"path": path, "category": category, **rec}, ensure_ascii=False, default=str) + "\n")
                n += 1
        return n
    import pyarrow as pa
    import pyarrow.parquet as pq
    S, I, F = pa.string(), pa.int64(), pa.float64()
    schema = pa.schema([("cid", I), ("path", S), ("category", S), ("confidence", F), ("page", S), ("title", S),
                        ("year", I), ("year2", I), ("lat", F), ("lon", F), ("direction", S), ("regions_en", S),
                        ("regions_local", S), ("address", S), ("author", S), ("source", S), ("uploader", S),
                        ("width", I), ("height", I), ("sha256", S), ("caption", S), ("vlm_caption", S),
                        ("record", S)])
    batch, writer = [], None
    try:
        for cid, path, category, rec in rows(store, scope):
            batch.append(flat(cid, path, category, rec))
            if len(batch) >= 5000:
                table = pa.Table.from_pylist(batch, schema=schema)
                writer = writer or pq.ParquetWriter(str(dst), schema, compression="zstd")
                writer.write_table(table)
                n += len(batch)
                batch = []
        if batch:
            table = pa.Table.from_pylist(batch, schema=schema)
            writer = writer or pq.ParquetWriter(str(dst), schema, compression="zstd")
            writer.write_table(table)
            n += len(batch)
    finally:
        if writer:
            writer.close()
    return n
