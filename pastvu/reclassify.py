"""reclassify: sort the saved photos of a scope again with the current
samples folder, moving each image and its caption in place between category
folders. The PastVu number stays the file name, so records and captions stay
joined.

- Embeddings come from the store; an image without one is embedded from its
  file (the encoder loads only then).
- A photo whose file is not where the store says was moved by hand. Its new
  place is kept and recorded (--override-manual sorts it like the rest).
- A photo whose file is gone is reported and left alone.
- A photo the quality probe scores at or above its threshold, or whose long
  side is more than max_aspect times its short side, is deleted (image,
  caption, JSON sidecar, export links) and marked skipped, so no run fetches
  it again. Hand-sorted photos are kept unless --override-manual. Deleting
  cannot be undone.
- Every move goes into the store, so --undo puts the last reclassify back.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

import folders
import layout
import process as pr
from folders import move_pair, prune_empty, scan_scope  # noqa: F401
import quality
from store import SKIPPED, Store


def update_record(store: Store, cid: int, rel: str, verdict_dict: dict | None, note: dict,
                  mode: str | None = None) -> str | None:
    rec = store.record(cid)
    if rec is None:
        return None
    rec.setdefault("processing", {}).setdefault("output", {})["path"] = rel
    if verdict_dict is not None:
        note["previous"] = rec.get("classification")
        rec["classification"] = verdict_dict
    if mode:
        col = rec["processing"].setdefault("colour", {})
        if col.get("mode") != mode:
            note["mode_from"] = col.get("mode")
            col.update(mode=mode, by="hand")
    rec.setdefault("reclassified", []).append(note)
    return json.dumps(rec, ensure_ascii=False, indent=1, default=str)


def plan(store: Store, base: Path, scope: dict, classifier, override_manual: bool, log=print,
         max_aspect: float = 0.0) -> dict:
    """What a reclassify would do. -> {"moves": [...], "low_quality": [...], "aspect": [...], "manual": [...],
    "missing": [...], "kept": n, "embedded": n}; low_quality and aspect are the photos to delete.

    Paths and categories are relative to the scope; in a base split by black-and-white
    and colour they start with the mode (bw/50_transport), so a photo moved by hand
    from bw to color is a hand-sorted photo like one moved to another category."""
    name, width = scope["name"], layout.scope_width(scope)
    on_disk = folders.scan_rels(base, scope)
    rows = store.done_rows(name)
    sizes = store.sizes(name) if max_aspect > 0 else {}
    manual, missing, todo, aspect = [], [], [], []
    for cid, path, category, sha in rows:
        stored = layout.to_scope(scope, path) or path
        actual = on_disk.get(cid)
        if actual is None:
            missing.append(cid)
            continue
        if actual != stored and not override_manual:
            if layout.category_of(actual, width) != layout.category_of(stored, width):
                manual.append((cid, stored, actual))
                continue
        wide = pr.aspect_reason(*sizes.get(cid, (0, 0)), max_aspect)
        if wide:
            aspect.append({"cid": cid, "src": actual, "category": layout.category_of(actual, width), "reason": wide})
            continue
        todo.append((cid, actual, layout.category_of(actual, width), sha))
    # embeddings
    model, patches = classifier.model_id, classifier.patches
    vecs, embedded = [], 0
    t0 = time.time()
    for i, (cid, actual, _, _) in enumerate(todo):
        v = store.embedding(cid, model, patches)
        if v is None:
            v = classifier.embed_file(base / layout.to_base(scope, actual))
            store.set_embedding(cid, model, patches, v)
            embedded += 1
            if embedded % 200 == 0:
                log(f"  embedded {embedded:,} images without a stored embedding ({time.time() - t0:.0f} s)")
        vecs.append(v)
    bad = classifier.low_quality(vecs) if vecs else np.zeros(0)
    low = [{"cid": cid, "src": actual, "category": cat_now, "p_bad": round(float(b), 4),
            "reason": f"{quality.REASON} (P(bad) {b:.2f})"}
           for (cid, actual, cat_now, _), b in zip(todo, bad) if not np.isnan(b)]
    keep = [i for i, b in enumerate(bad) if np.isnan(b)]
    todo, vecs = [todo[i] for i in keep], [vecs[i] for i in keep]
    verdicts = classifier.decide_many(vecs) if vecs else []
    moves, kept = [], 0
    for (cid, actual, cat_now, sha), v in zip(todo, verdicts):
        ex = classifier.example_class(sha)
        if ex is not None:
            v.folder, v.cls, v.sub, v.confidence, v.reason = ex, ex, ex, 1.0, "example"
        mode, _ = layout.mode_category(scope, cat_now)
        to = layout.with_mode(scope, mode, v.folder)
        if to == cat_now:
            kept += 1
        else:
            moves.append({"cid": cid, "src": actual, "from": cat_now, "to": to, "verdict": v.as_dict()})
    return {"moves": moves, "low_quality": low, "aspect": aspect, "manual": manual, "missing": missing,
            "kept": kept, "embedded": embedded, "checked": len(rows)}


UNSURE = "_unsure"


def plan_unsure(store: Store, base: Path, scope: dict) -> dict:
    """The photos in _unsure/<category> (from versions that had the folder) into
    <category>, their best guess; no classifier. Same shape as plan()."""
    name, width = scope["name"], layout.scope_width(scope)
    on_disk = folders.scan_rels(base, scope)
    rows = store.done_rows(name)
    moves, missing, kept = [], [], 0
    for cid, path, category, sha in rows:
        actual = on_disk.get(cid)
        if actual is None:
            missing.append(cid)
            continue
        cat = layout.category_of(actual, width)
        mode, plain = layout.mode_category(scope, cat)
        if plain != UNSURE and not plain.startswith(UNSURE + "/"):
            kept += 1
            continue
        best = plain[len(UNSURE) + 1:]
        verdict = dict((store.record(cid) or {}).get("classification") or {})
        verdict["folder"] = best
        moves.append({"cid": cid, "src": actual, "from": cat, "to": layout.with_mode(scope, mode, best),
                      "verdict": verdict})
    return {"moves": moves, "low_quality": [], "aspect": [], "manual": [], "missing": missing, "kept": kept,
            "embedded": 0, "checked": len(rows)}


def _follow(store: Store, exporter, cid: int) -> None:
    if exporter is not None:
        st = store.status(cid)
        exporter.move(cid, st.get("mode"), st.get("category") or "")


def delete_photos(store: Store, base: Path, scope: dict, items: list, chunks, log=print, exporter=None) -> int:
    """Delete the photos (low quality, aspect ratio) and mark them skipped with their reason:
    no run fetches them again."""
    name = scope["name"]
    deleted = 0
    scans: dict = {}
    for m in items:
        cid, src = m["cid"], m["src"]
        if not (base / layout.to_base(scope, src)).is_file():
            here = folders.locate(base, scope, cid, None, scans)
            if here is None:
                log(f"  {cid}: not found; marked skipped ({m['reason']}) all the same")
            else:
                src = layout.to_scope(scope, here)
        img = base / layout.to_base(scope, src)
        for f in (img, img.with_suffix(".txt"), img.with_suffix(".json"), layout.record_file(base, cid)):
            f.unlink(missing_ok=True)
        chunks.remove(src)
        if exporter is not None:
            exporter.remove(cid)
        store.reject(cid, name, SKIPPED, m["reason"], {})
        deleted += 1
    return deleted


def apply(store: Store, base: Path, scope: dict, p: dict, samples: str, log=print, exporter=None) -> dict:
    name, width = scope["name"], layout.scope_width(scope)
    # hand-sorted photos: record where they are now (their category, and their mode in a split base)
    for cid, stored, actual in p["manual"]:
        cat = layout.category_of(actual, width)
        old = layout.category_of(stored, width)
        mode, plain = layout.mode_category(scope, cat)
        rel = layout.to_base(scope, actual)
        rec = update_record(store, cid, rel, None, {"time": time.time(), "manual": True, "from": old, "to": cat}, mode)
        store.move(cid, rel, plain, rec, mode)
        _follow(store, exporter, cid)
    chunks = folders.ScopeChunks(store, base, scope, log, defer=True)
    deleted = delete_photos(store, base, scope, (p.get("low_quality") or []) + (p.get("aspect") or []), chunks,
                            log, exporter)
    run = store.start_reclass(name, samples)
    moved = failed = 0
    scans: dict = {}
    for seq, m in enumerate(p["moves"]):
        src = m["src"]
        if not (base / layout.to_base(scope, src)).is_file():
            here = folders.locate(base, scope, m["cid"], None, scans)
            if here is None:
                log(f"  {m['cid']}: not found; not moved")
                failed += 1
                continue
            src = layout.to_scope(scope, here)
        fname = src.rsplit("/", 1)[-1]
        dst = chunks.place(m["to"], fname)
        try:
            move_pair(base / layout.to_base(scope, src), base / chunks.rel(dst))
        except OSError as e:
            chunks.abandon(dst)
            log(f"  {m['cid']}: not moved ({e})")
            failed += 1
            continue
        chunks.remove(src)
        note = {"time": time.time(), "run": run, "from": m["from"], "to": m["to"], "samples": samples}
        rec = update_record(store, m["cid"], chunks.rel(dst), m["verdict"], note)
        store.move(m["cid"], chunks.rel(dst), layout.mode_category(scope, m["to"])[1], rec)
        chunks.done(dst)
        _follow(store, exporter, m["cid"])
        store.add_reclass_move(run, seq, m["cid"], src, dst, m["from"], m["to"])
        moved += 1
    chunks.settle()
    pruned = folders.prune_scope(base, scope)
    store.recount(name)
    return {"run": run, "moved": moved, "failed": failed, "pruned": pruned, "manual": len(p["manual"]),
            "deleted": deleted}


def undo(store: Store, base: Path, scope: dict, log=print, exporter=None) -> dict:
    """Put the photos of the last reclassify back into the categories they came
    from (into a numbered folder with room when the category was chunked since)."""
    name = scope["name"]
    last = store.last_reclass(name)
    if not last:
        raise SystemExit(f"{name}: no reclassify to undo")
    chunks = folders.ScopeChunks(store, base, scope, log, defer=True)
    back = skipped = 0
    scans: dict = {}
    for seq, cid, src, dst, cat_from, cat_to in store.reclass_moves(last["id"]):
        st = store.status(cid)
        mode_to, plain_to = layout.mode_category(scope, cat_to)
        if not st or st["scope"] != name or st["category"] != plain_to or (mode_to and st.get("mode") != mode_to):
            log(f"  {cid}: moved again since; left where it is")
            skipped += 1
            continue
        here = folders.locate(base, scope, cid, st["path"], scans)
        if here is None:
            log(f"  {cid}: the file is gone; left as it is")
            skipped += 1
            continue
        here = layout.to_scope(scope, here)
        back_to = chunks.place(cat_from, here.rsplit("/", 1)[-1])
        try:
            move_pair(base / layout.to_base(scope, here), base / chunks.rel(back_to))
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
        store.move(cid, chunks.rel(back_to), layout.mode_category(scope, cat_from)[1], rec)
        chunks.done(back_to)
        _follow(store, exporter, cid)
        back += 1
    chunks.settle()
    store.undo_reclass(last["id"])
    pruned = folders.prune_scope(base, scope)
    store.recount(name)
    return {"run": last["id"], "back": back, "skipped": skipped, "pruned": pruned}


def summary(p: dict) -> str:
    flows = Counter((m["from"], m["to"]) for m in p["moves"])
    lines = [f"{p['checked']:,} photos: {len(p['moves']):,} to move, {p['kept']:,} stay, "
             f"{len(p.get('low_quality') or []):,} low quality and {len(p.get('aspect') or []):,} over the aspect "
             f"ratio (to delete), "
             f"{len(p['manual']):,} moved by hand (kept), {len(p['missing']):,} files missing; "
             f"{p['embedded']:,} embedded from their files"]
    for (a, b), n in flows.most_common(25):
        lines.append(f"  {n:6,}  {a or '(none)'} -> {b}")
    if len(flows) > 25:
        lines.append(f"  ... and {len(flows) - 25} more")
    return "\n".join(lines)
