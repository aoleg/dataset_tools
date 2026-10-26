"""cleanup: a review of the saved photos by their quality, without deleting
anything yourself would not.

- `cleanup 0.8` scores every saved photo of the scopes with the quality probe
  (the threshold given replaces quality_threshold of the conf for this
  command only; the category classifier is not used) and moves each one at
  or above the threshold, with its caption, into one flat folder next to the
  base, <base>_review. Its name there starts with its score (0.934_12345.jpg),
  so a sort by name puts the worst first. A second cleanup adds to the open
  review.
- You delete the images that are bad. Then:
  - `cleanup --done` puts the rest back into their categories and marks them
    reviewed-good: no reclassify or cleanup sends them away again for their
    quality;
  - `cleanup --undo` puts the rest back as they were, not marked.
  Either way a photo whose image you deleted is skipped as "low quality
  (reviewed)" and never downloaded again, its caption left behind is deleted,
  and the review folder goes when nothing else is left in it.
- A photo in review is in the state review: not pending (no run downloads it
  again), not done (reclassify, extract, export and autocaption leave it
  alone). regroup and drop wait until the review is settled.
- Hand-sorted photos are left out, as in reclassify (--override-manual takes
  them too), and so are the photos you kept in an earlier review.
- The live export keeps its links during the review (a hard link stays valid
  when the base's file moves); the links of the photos you deleted go when
  the review is settled.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path

import folders
import layout
import numpy as np
import quality
from store import DONE, SKIPPED, Store

META = "review"
SUFFIX = "_review"
SIDECARS = (".txt", ".json")


def review_dir(store: Store, base: Path) -> Path:
    """The open review's folder, or the one a new review gets: <base>_review next to the base."""
    meta = store.get_meta(META) or {}
    return Path(meta["folder"]) if meta.get("folder") else base.parent / f"{base.name}{SUFFIX}"


def plan(store: Store, base: Path, scope: dict, classifier, threshold: float, override_manual: bool = False,
         log=print) -> dict:
    """The saved photos of a scope that go to review.
    -> {"picks": [{cid, src (scope-relative), p_bad}], "checked", "manual", "missing", "kept_good", "embedded"}"""
    name, width = scope["name"], layout.scope_width(scope)
    on_disk = folders.scan_rels(base, scope)
    good = store.reviewed_good()
    todo, manual, missing, kept_good = [], 0, 0, 0
    rows = store.done_rows(name)
    for cid, path, category, sha in rows:
        if cid in good:
            kept_good += 1
            continue
        stored = layout.to_scope(scope, path) or path
        actual = on_disk.get(cid)
        if actual is None:
            missing += 1
            continue
        if actual != stored and not override_manual and \
                layout.category_of(actual, width) != layout.category_of(stored, width):
            manual += 1
            continue
        todo.append((cid, actual))
    model, patches = classifier.model_id, classifier.patches
    vecs, embedded = [], 0
    t0 = time.time()
    for cid, actual in todo:
        v = store.embedding(cid, model, patches)
        if v is None:
            v = classifier.embed_file(base / layout.to_base(scope, actual))
            store.set_embedding(cid, model, patches, v)
            embedded += 1
            if embedded % 200 == 0:
                log(f"  embedded {embedded:,} images without a stored embedding ({time.time() - t0:.0f} s)")
        vecs.append(v)
    p = classifier.quality.p_bad(np.stack(vecs)) if vecs else np.zeros(0)
    picks = [{"cid": cid, "src": actual, "p_bad": round(float(b), 4)}
             for (cid, actual), b in zip(todo, p) if b >= threshold]
    picks.sort(key=lambda m: -m["p_bad"])
    return {"picks": picks, "checked": len(rows), "manual": manual, "missing": missing, "kept_good": kept_good,
            "embedded": embedded}


def summary(name: str, p: dict, threshold: float) -> str:
    return (f"{name}: {p['checked']:,} saved photos, {len(p['picks']):,} at P(bad) >= {threshold} to review; "
            f"{p['kept_good']:,} kept in an earlier review, {p['manual']:,} sorted by hand and {p['missing']:,} "
            f"with their file gone are left out; {p['embedded']:,} embedded from their files")


def move_to_review(store: Store, base: Path, scope: dict, picks: list, threshold: float, log=print) -> int:
    """Move the picked images and their sidecars into the review folder. -> moved."""
    folder = review_dir(store, base)
    folder.mkdir(parents=True, exist_ok=True)
    meta = store.get_meta(META) or {"folder": str(folder), "started": time.time(), "thresholds": []}
    if threshold not in meta["thresholds"]:
        meta["thresholds"].append(threshold)
    store.set_meta(META, meta)
    chunks = folders.ScopeChunks(store, base, scope, log, defer=True)
    moved = 0
    for m in picks:
        src_rel = layout.to_base(scope, m["src"])
        src = base / src_rel
        name = f"{m['p_bad']:.3f}_{src.name}"
        try:
            folders.move_pair(src, folder / name)
        except OSError as e:
            log(f"  {m['cid']}: not moved ({e})")
            continue
        chunks.remove(m["src"])
        store.add_review(m["cid"], scope["name"], src_rel, name, m["p_bad"])
        moved += 1
    chunks.settle()
    folders.prune_scope(base, scope)
    store.recount(scope["name"])
    return moved


def settle(store: Store, base: Path, keep_good: bool, log=print, exporter=None) -> dict:
    """End the review: the images still in the folder go back (marked reviewed-good with
    keep_good), the deleted ones are skipped for good. -> counts."""
    items = store.review_items()
    if not items:
        raise SystemExit(f"{base}: no cleanup review is open")
    folder = review_dir(store, base)
    by_scope = defaultdict(list)
    for it in items:
        by_scope[it[1]].append(it)
    back = gone = stuck = 0
    for name, rows in by_scope.items():
        sc = store.scope(name)
        if sc is None:
            log(f"  scope {name} is gone; its {len(rows):,} photos stay in {folder}")
            stuck += len(rows)
            continue
        chunks = folders.ScopeChunks(store, base, sc, log, defer=True)
        for cid, _, src, fname, p_bad in rows:
            f = folder / fname
            if f.is_file():
                st = store.status(cid) or {}
                cat = layout.with_mode(sc, st.get("mode"), st.get("category") or "")
                dst_rel = chunks.place(cat, Path(src).name)
                dst = base / chunks.rel(dst_rel)
                try:
                    folders.move_pair(f, dst)
                except OSError as e:
                    chunks.abandon(dst_rel)
                    log(f"  {cid}: not moved back ({e}); it stays in {folder}")
                    stuck += 1
                    continue
                note = ("reviewed", {"time": time.time(), "p_bad": p_bad, "verdict": "good" if keep_good else "undo"})
                rec = folders.record_with_path(store, base, cid, chunks.rel(dst_rel), note)
                store.end_review(cid, DONE, path=chunks.rel(dst_rel), record=rec, good=keep_good, p_bad=p_bad)
                chunks.done(dst_rel)
                back += 1
            else:
                for ext in SIDECARS:                        # the caption you left behind
                    f.with_suffix(ext).unlink(missing_ok=True)
                layout.record_file(base, cid).unlink(missing_ok=True)
                if exporter is not None:
                    exporter.remove(cid)
                store.end_review(cid, SKIPPED, reason=f"{quality.REASON} (reviewed, P(bad) {p_bad:.2f})")
                gone += 1
        chunks.settle()
        store.recount(name)
    left = []
    if folder.is_dir():
        folders.prune_empty(folder)
        left = [p for p in folder.rglob("*") if p.is_file()]
        if not left:
            os.rmdir(folder)
    if not store.review_count():
        store.set_meta(META, None)
    return {"back": back, "gone": gone, "stuck": stuck, "left": len(left), "folder": str(folder)}
