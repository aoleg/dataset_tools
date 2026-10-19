"""reclassify: sort the saved photos of a scope again with the current
samples folder, moving each image and its caption in place between category
folders. The PastVu number stays the file name, so records and captions stay
joined.

- Embeddings come from the store; an image without one is embedded from its
  file (the encoder loads only then).
- A photo whose file is not where the store says was moved by hand. Its new
  place is kept and recorded (--override-manual sorts it like the rest).
- A photo whose file is gone is reported and left alone.
- Every move goes into the store, so --undo puts the last reclassify back.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import folders
import layout
from folders import move_pair, prune_empty, scan_scope  # noqa: F401
from store import Store


def update_record(store: Store, cid: int, rel: str, verdict_dict: dict | None, note: dict) -> str | None:
    rec = store.record(cid)
    if rec is None:
        return None
    rec.setdefault("processing", {}).setdefault("output", {})["path"] = rel
    if verdict_dict is not None:
        note["previous"] = rec.get("classification")
        rec["classification"] = verdict_dict
    rec.setdefault("reclassified", []).append(note)
    return json.dumps(rec, ensure_ascii=False, indent=1, default=str)


def plan(store: Store, base: Path, scope: dict, classifier, override_manual: bool, log=print) -> dict:
    """What a reclassify would do. -> {"moves": [...], "manual": [...], "missing": [...], "kept": n, "embedded": n}"""
    name, width = scope["name"], layout.scope_width(scope)
    scope_dir = base / scope["folder"]
    on_disk = scan_scope(scope_dir)
    rows = store.done_rows(name)
    prefix = scope["folder"] + "/"
    manual, missing, todo = [], [], []
    for cid, path, category, sha in rows:
        stored = path[len(prefix):] if path.startswith(prefix) else path
        actual = on_disk.get(cid)
        if actual is None:
            missing.append(cid)
            continue
        if actual != stored and not override_manual:
            if layout.category_of(actual, width) != layout.category_of(stored, width):
                manual.append((cid, stored, actual))
                continue
        todo.append((cid, actual, layout.category_of(actual, width), sha))
    # embeddings
    model, patches = classifier.model_id, classifier.patches
    vecs, embedded = [], 0
    t0 = time.time()
    for i, (cid, actual, _, _) in enumerate(todo):
        v = store.embedding(cid, model, patches)
        if v is None:
            v = classifier.embed_file(scope_dir / actual)
            store.set_embedding(cid, model, patches, v)
            embedded += 1
            if embedded % 200 == 0:
                log(f"  embedded {embedded:,} images without a stored embedding ({time.time() - t0:.0f} s)")
        vecs.append(v)
    verdicts = classifier.decide_many(vecs) if vecs else []
    moves, kept = [], 0
    for (cid, actual, cat_now, sha), v in zip(todo, verdicts):
        ex = classifier.example_class(sha)
        if ex is not None:
            v.folder, v.cls, v.sub, v.confidence, v.reason = ex, ex, ex, 1.0, "example"
        if v.folder == cat_now:
            kept += 1
        else:
            moves.append({"cid": cid, "src": actual, "from": cat_now, "to": v.folder, "verdict": v.as_dict()})
    return {"moves": moves, "manual": manual, "missing": missing, "kept": kept, "embedded": embedded,
            "checked": len(rows)}


def apply(store: Store, base: Path, scope: dict, p: dict, samples: str, log=print) -> dict:
    name, width = scope["name"], layout.scope_width(scope)
    folder = scope["folder"]
    scope_dir = base / folder
    # hand-sorted photos: record where they are now
    for cid, stored, actual in p["manual"]:
        cat = layout.category_of(actual, width)
        old = layout.category_of(stored, width)
        rec = update_record(store, cid, f"{folder}/{actual}", None, {"time": time.time(), "manual": True, "from": old,
                                                                      "to": cat})
        store.move(cid, f"{folder}/{actual}", cat, rec)
        store.move_counter(name, old, cat)
    chunks = folders.ScopeChunks(store, base, scope, log, defer=True)
    run = store.start_reclass(name, samples)
    moved = failed = 0
    scans: dict = {}
    for seq, m in enumerate(p["moves"]):
        src = m["src"]
        if not (scope_dir / src).is_file():
            here = folders.locate(base, folder, m["cid"], None, scans)
            if here is None:
                log(f"  {m['cid']}: not found; not moved")
                failed += 1
                continue
            src = here[len(folder) + 1:]
        fname = src.rsplit("/", 1)[-1]
        dst = chunks.place(m["to"], fname)
        try:
            move_pair(scope_dir / src, scope_dir / dst)
        except OSError as e:
            chunks.abandon(dst)
            log(f"  {m['cid']}: not moved ({e})")
            failed += 1
            continue
        chunks.remove(src)
        note = {"time": time.time(), "run": run, "from": m["from"], "to": m["to"], "samples": samples}
        rec = update_record(store, m["cid"], chunks.rel(dst), m["verdict"], note)
        store.move(m["cid"], chunks.rel(dst), m["to"], rec)
        chunks.done(dst)
        store.move_counter(name, m["from"], m["to"])
        store.add_reclass_move(run, seq, m["cid"], src, dst, m["from"], m["to"])
        moved += 1
    chunks.settle()
    pruned = prune_empty(scope_dir)
    return {"run": run, "moved": moved, "failed": failed, "pruned": pruned, "manual": len(p["manual"])}


def undo(store: Store, base: Path, scope: dict, log=print) -> dict:
    """Put the photos of the last reclassify back into the categories they came
    from (into a numbered folder with room when the category was chunked since)."""
    name, folder = scope["name"], scope["folder"]
    scope_dir = base / folder
    last = store.last_reclass(name)
    if not last:
        raise SystemExit(f"{name}: no reclassify to undo")
    chunks = folders.ScopeChunks(store, base, scope, log, defer=True)
    back = skipped = 0
    scans: dict = {}
    for seq, cid, src, dst, cat_from, cat_to in store.reclass_moves(last["id"]):
        st = store.status(cid)
        if not st or st["scope"] != name or st["category"] != cat_to:
            log(f"  {cid}: moved again since; left where it is")
            skipped += 1
            continue
        here = folders.locate(base, folder, cid, st["path"], scans)
        if here is None:
            log(f"  {cid}: the file is gone; left as it is")
            skipped += 1
            continue
        here = here[len(folder) + 1:]
        back_to = chunks.place(cat_from, here.rsplit("/", 1)[-1])
        try:
            move_pair(scope_dir / here, scope_dir / back_to)
        except OSError as e:
            chunks.abandon(back_to)
            log(f"  {cid}: not moved back ({e})")
            skipped += 1
            continue
        chunks.remove(here)
        rec = store.record(cid)
        if rec is not None:
            rec.setdefault("processing", {}).setdefault("output", {})["path"] = chunks.rel(back_to)
            note = next((h for h in reversed(rec.get("reclassified") or []) if h.get("run") == last["id"]), None)
            if note and "previous" in note:
                rec["classification"] = note["previous"]
            rec.setdefault("reclassified", []).append({"time": time.time(), "undo": last["id"], "to": cat_from})
            rec = json.dumps(rec, ensure_ascii=False, indent=1, default=str)
        store.move(cid, chunks.rel(back_to), cat_from, rec)
        chunks.done(back_to)
        store.move_counter(name, cat_to, cat_from)
        back += 1
    chunks.settle()
    store.undo_reclass(last["id"])
    pruned = prune_empty(scope_dir)
    return {"run": last["id"], "back": back, "skipped": skipped, "pruned": pruned}


def summary(p: dict) -> str:
    flows = Counter((m["from"], m["to"]) for m in p["moves"])
    lines = [f"{p['checked']:,} photos: {len(p['moves']):,} to move, {p['kept']:,} stay, "
             f"{len(p['manual']):,} moved by hand (kept), {len(p['missing']):,} files missing; "
             f"{p['embedded']:,} embedded from their files"]
    for (a, b), n in flows.most_common(25):
        lines.append(f"  {n:6,}  {a or '(none)'} -> {b}")
    if len(flows) > 25:
        lines.append(f"  ... and {len(flows) - 25} more")
    return "\n".join(lines)
