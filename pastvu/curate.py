"""curate: a partner's curation of a shared pick, applied to our pick folder.

The partner rebuilt the pick from a `share` zip, deleted the photos they found of no
value, and ran `snapshot` (replica.snapshot), which compares their folder with their
pastvu.db: kept, deleted, never received (skipped or failed by their rebuild). The
snapshot names photos by number, so a photo the partner moved to another folder is
kept, and a file whose bytes differ from ours (a rebuilt downscale) still matches.

`curate <pick> --from <snapshot>` moves every photo the partner deleted, the image and
every file of its name, out of the pick into <pick>_deleted, under the same relative
path. The photos the partner never received stay, as do the photos the partner moved
(their move is not repeated here). The moved photos are written into the pick's
cherrypick.json as excluded: a later cherrypick into the folder does not copy them
again. Each run is logged in <pick>_deleted/curate-log.json, and `curate <pick>
--undo` moves the last run back and takes its photos off the excluded list.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cherrypick as cp
import folders

LOG = "curate-log.json"
MIN_OVERLAP = 0.95          # the share of the snapshot's photos that must be in the pick


def deleted_dir(pick: Path) -> Path:
    return pick.parent / f"{pick.name}_deleted"


def _entry_key(manifest: dict, base: Path) -> str:
    want = str(Path(base).resolve()).lower()
    for key in manifest.get("bases") or {}:
        if str(Path(key).resolve()).lower() == want:
            return key
    raise SystemExit(f"the pick holds no photos of {base}; give the base it was picked from with --base")


def _write_manifest(pick: Path, manifest: dict) -> None:
    f = pick / cp.MANIFEST
    tmp = f.with_name(f.name + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, f)


def read_log(dd: Path) -> dict:
    f = dd / LOG
    if f.is_file():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except ValueError as e:
            raise SystemExit(f"{f}: {e}")
    return {"runs": []}


def write_log(dd: Path, log: dict) -> None:
    dd.mkdir(parents=True, exist_ok=True)
    f = dd / LOG
    tmp = f.with_name(f.name + ".tmp")
    tmp.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, f)


def plan(pick: Path, base: Path, snap: dict) -> dict:
    """What a snapshot means for the pick. -> {"moves": [(cid, rel)], "missing": [cid] (deleted
    by the partner, gone here already), counts}"""
    pick = Path(pick).resolve()
    manifest = cp._read_manifest(pick)
    if not manifest["bases"]:
        raise SystemExit(f"{pick} has no {cp.MANIFEST}: give the output folder of a cherrypick")
    entry = manifest["bases"][_entry_key(manifest, base)]
    files = {int(k): v for k, v in (entry.get("files") or {}).items()}
    excluded = {int(c) for c in entry.get("excluded") or []}       # moved out by an earlier run
    kept = [int(c) for c in snap.get("kept") or []]
    deleted = [int(c) for c in snap.get("deleted") or []]
    never = [int(c) for c in (snap.get("never") or {})]
    theirs = set(kept) | set(deleted) | set(never)
    if not theirs:
        raise SystemExit("the snapshot lists no photos")
    inside = theirs & (set(files) | excluded)
    if len(inside) < MIN_OVERLAP * len(theirs):
        raise SystemExit(f"the snapshot is of another pick: {len(inside):,} of its {len(theirs):,} photos are in {pick}")
    moves, missing, before = [], [], 0
    for cid in deleted:
        rel = files.get(cid)
        if rel is None:
            before += cid in excluded
            continue
        if (pick / rel).is_file():
            moves.append((cid, rel))
        else:
            missing.append(cid)
    in_pick = set(files)
    return {"pick": pick, "moves": moves, "missing": missing, "before": before,
            "kept": len(set(kept) & in_pick), "never": len(set(never) & in_pick),
            "moved_there": sum(1 for c in (snap.get("moved") or {}) if int(c) in in_pick),
            "not_in_snapshot": len(in_pick - theirs), "foreign": len(theirs - in_pick),
            "unknown": int(snap.get("unknown") or 0), "photos": len(files),
            "left": len(files) - len(moves) - len(missing)}


def apply(pick: Path, base: Path, p: dict, source: str, log=print) -> dict:
    """Move the deleted photos out and mark them excluded. -> counts"""
    pick = Path(pick).resolve()
    dd = deleted_dir(pick)
    if dd.exists() and any(dd.iterdir()) and not (dd / LOG).is_file():
        raise SystemExit(f"{dd} exists and was not made by curate; move it away first")
    book = read_log(dd)
    run = {"id": len(book["runs"]) + 1, "time": time.time(), "snapshot": source, "moves": [],
           "excluded": [], "undone": None}
    book["runs"].append(run)
    stats = {"moved": 0, "conflict": 0, "failed": 0}
    t0 = time.time()
    try:
        for i, (cid, rel) in enumerate(p["moves"], 1):
            src, dst = pick / rel, dd / rel
            if dst.exists():
                stats["conflict"] += 1
                if stats["conflict"] <= 5:
                    log(f"  {cid}: {dst} exists; left in the pick")
                continue
            try:
                folders.move_pair(src, dst)
            except OSError as e:
                stats["failed"] += 1
                if stats["failed"] <= 5:
                    log(f"  {cid}: not moved ({e})")
                continue
            run["moves"].append([cid, rel])
            stats["moved"] += 1
            if i % 10000 == 0:
                log(f"  {i:,} of {len(p['moves']):,} ({time.time() - t0:.0f} s)")
    finally:
        moved = {c for c, _ in run["moves"]}
        run["excluded"] = sorted(moved | set(p["missing"]))
        write_log(dd, book)
        manifest = cp._read_manifest(pick)
        key = _entry_key(manifest, base)
        entry = manifest["bases"][key]
        entry["excluded"] = sorted(set(int(c) for c in entry.get("excluded") or []) | set(run["excluded"]))
        entry["files"] = {k: v for k, v in (entry.get("files") or {}).items() if int(k) not in moved}
        _write_manifest(pick, manifest)
    folders.prune_empty(pick)
    stats["excluded"] = len(run["excluded"])
    stats["run"] = run["id"]
    return stats


def undo(pick: Path, base: Path, log=print) -> dict:
    """Move the last run's photos back into the pick and take them off the excluded list."""
    pick = Path(pick).resolve()
    dd = deleted_dir(pick)
    book = read_log(dd)
    run = next((r for r in reversed(book["runs"]) if not r.get("undone")), None)
    if run is None:
        raise SystemExit(f"nothing to undo in {dd}")
    back, gone, blocked = {}, 0, 0
    for cid, rel in reversed(run["moves"]):
        src, dst = dd / rel, pick / rel
        if not src.is_file():
            gone += 1
            continue
        if dst.exists():
            blocked += 1
            if blocked <= 5:
                log(f"  {cid}: {dst} exists; left in {dd}")
            continue
        folders.move_pair(src, dst)
        back[cid] = rel
    run["undone"] = time.time()
    write_log(dd, book)
    manifest = cp._read_manifest(pick)
    entry = manifest["bases"][_entry_key(manifest, base)]
    free = set(back) | (set(run["excluded"]) - {c for c, _ in run["moves"]})
    entry["excluded"] = sorted(set(int(c) for c in entry.get("excluded") or []) - free)
    files = entry.get("files") or {}
    files.update({str(c): rel for c, rel in back.items()})
    entry["files"] = dict(sorted(files.items(), key=lambda kv: int(kv[0])))
    _write_manifest(pick, manifest)
    folders.prune_empty(dd)
    return {"run": run["id"], "back": len(back), "gone": gone, "blocked": blocked}


def summary(p: dict, snap: dict) -> list[str]:
    lines = [f"Snapshot of {snap.get('name')} made {snap.get('made_text')}: {snap.get('photos', 0):,} photos, "
             f"{len(snap.get('kept') or []):,} kept, {len(snap.get('deleted') or []):,} deleted, "
             f"{len(snap.get('never') or {}):,} never received.",
             f"{p['pick']}: {p['photos']:,} photos. To move out: {len(p['moves']):,}; deleted by the partner and "
             f"already gone here: {len(p['missing']):,}; moved out by an earlier curate: {p['before']:,}. "
             f"Stay: {p['left']:,} ({p['kept']:,} kept by the partner, "
             f"{p['never']:,} they never received, {p['not_in_snapshot']:,} not in the snapshot)."]
    if p["moved_there"]:
        lines.append(f"  {p['moved_there']:,} kept photos are in another folder at the partner's; their move is not "
                     f"repeated here.")
    if p["foreign"]:
        lines.append(f"  {p['foreign']:,} photos of the snapshot are not in this pick (ignored).")
    if p["unknown"]:
        lines.append(f"  The partner's folder holds {p['unknown']:,} images that are no photo of the pick (ignored).")
    return lines

