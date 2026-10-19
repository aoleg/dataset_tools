"""regroup: move saved photos between scopes of a base folder, with no
download. One move engine behind two commands:

- `run --regroup` pulls every saved photo the run's scopes list into them,
  from whatever scope holds it now (`pull_plan`).
- `regroup --by RULE [TARGET...]` splits the target scopes, or groups the
  whole base, by decade, country or region (`rule_plan`); the photos' scope
  memberships move with them, so a later run resumes the new scopes.

A photo keeps the category folder it has on disk (hand sorting survives) and
goes into the lowest numbered folder with room in its new scope. Image and
caption are renamed; records, embeddings and VLM captions are keyed by the
photo's number and follow. A saved photo whose file is gone is marked not
downloaded, so the next run fetches it. A source scope left owning nothing,
whose photos are all listed by the new scopes, is removed with its empty
folders. Counters are recomputed for every scope involved, and the move log
lets `regroup --undo` put the last regroup back.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import folders
import layout
import partition as pt
from folders import move_pair, prune_empty, remove_folder
from store import DONE, FAILED, SKIPPED, Store


KEYS = ("file", "owner", "missing", "failed", "members", "removed", "created", "skipped")


@dataclass
class Move:
    cid: int
    src: str
    dst: str
    kind: str                 # "file" (a saved photo), "owner" (skipped or failed), "missing" (saved, file gone)
    path: str | None = None   # the file, relative to the base
    category: str | None = None


# --- what goes where ---------------------------------------------------------------------

def pull_plan(store: Store, scopes: list[str]) -> dict[int, str]:
    """run --regroup: every member of the run's scopes belongs in that scope (the first one listing it)."""
    wanted: dict[int, str] = {}
    for name in scopes:
        for cid in store.members(name):
            wanted.setdefault(cid, name)
    return wanted


def photo_facts(store: Store, cid: int) -> dict:
    """What the partition rule needs: the start year and the region chain."""
    row = store.photo(cid) or {}
    rec = store.record(cid) or {}
    return {"year": row.get("year", rec.get("year")), "rs": row.get("rs") or [], "regions": rec.get("regions")}


def rule_plan(store: Store, aliases: dict, groups: dict, rule: pt.Rule, sources: list[dict], whole_base: bool,
              log=print) -> dict:
    """regroup --by: the scope each member of the sources belongs in.
    -> {"wanted": {cid: name}, "members": [(cid, src, dst)], "defs": {name: definition}}"""
    label, regions = None, None
    if whole_base and not rule.place:
        same = all(sorted(s["regions"]) == sorted(sources[0]["regions"]) for s in sources)
        label = sources[0]["label"] if same else "all"
        regions = sorted({c for s in sources for c in s["regions"]})
    keyed: dict[int, tuple] = {}
    pairs: list[tuple[int, str, tuple]] = []
    contrib: dict[tuple, list[dict]] = defaultdict(list)
    t0, n = time.time(), 0
    for s in sources:
        part = pt.Partitioner(store, aliases, groups, rule, label or s["label"],
                              regions if regions is not None else s["regions"])
        for cid in store.members(s["name"]):
            if cid not in keyed:
                t = part.target(photo_facts(store, cid))
                keyed[cid] = (t.label, t.regions, t.decade)
            key = keyed[cid]
            pairs.append((cid, s["name"], key))
            if s not in contrib[key]:
                contrib[key].append(s)
            n += 1
            if n % 20000 == 0:
                log(f"  {n:,} photos placed ({time.time() - t0:.0f} s)")
    names: dict[tuple, str] = {}
    defs: dict[str, dict] = {}
    for key, srcs in contrib.items():
        lab, regs, decade = key
        if lab == "unplaced":
            regs = tuple(sorted({c for s in srcs for c in s["regions"]}))
        if decade is not None:
            years, dec = (decade, decade + 9), decade
        else:
            decs = {s["decade"] for s in srcs}
            if len(decs) == 1 and None not in decs:
                dec = decs.pop()
                years = (dec, dec + 9)
            else:
                years, dec = (min(s["years"][0] for s in srcs), max(s["years"][1] for s in srcs)), None
        name = layout.check_scope_name(layout.scope_name([lab], years, dec))
        d = {"name": name, "label": lab, "regions": sorted(regs), "years": tuple(years), "decade": dec,
             "split": srcs[0]["split"]}
        if name in defs and (defs[name]["regions"], defs[name]["years"], defs[name]["decade"]) != \
                (d["regions"], d["years"], d["decade"]):
            raise SystemExit(f"the rule makes two different scopes named {name}; regroup fewer scopes at a time")
        defs[name] = d
        names[key] = name
        sc = store.scope(name)
        if sc and (sorted(sc["regions"]) != d["regions"] or tuple(sc["years"]) != d["years"] or sc["decade"] != dec):
            raise SystemExit(f"scope {name} exists with regions {sc['regions']}, years {list(sc['years'])}, decade "
                             f"{sc['decade']}; the rule wants regions {d['regions']}, years {list(d['years'])}, "
                             f"decade {dec}. Rename or drop the existing scope first.")
    wanted: dict[int, str] = {}
    members = []
    for cid, src, key in pairs:
        dst = names[key]
        wanted.setdefault(cid, dst)
        if src != dst:
            members.append((cid, src, dst))
    return {"wanted": wanted, "members": members, "defs": defs}


# --- the plan ----------------------------------------------------------------------------

def plan(store: Store, base: Path, wanted: dict[int, str], members: list | None = None, defs: dict | None = None,
         pull_from: set | None = None) -> dict:
    """The moves that bring every photo of `wanted` into its scope. pull_from:
    only photos owned by these scopes move (None: from any scope)."""
    members, defs = members or [], defs or {}
    scopes: dict[str, dict] = {}
    scans: dict[str, dict] = {}
    moves: list[Move] = []
    for cid, dst in wanted.items():
        st = store.status(cid)
        if not st or st["state"] not in (DONE, SKIPPED, FAILED):
            continue
        owner = st["scope"]
        if not owner or owner == dst or (pull_from is not None and owner not in pull_from):
            continue
        if st["state"] != DONE:
            moves.append(Move(cid, owner, dst, "owner"))
            continue
        if owner not in scopes:
            scopes[owner] = store.scope(owner) or {"width": 0, "split": 0, "folder": owner}
        osc = scopes[owner]
        actual = folders.locate(base, osc["folder"], cid, st["path"], scans)
        if actual is None:
            moves.append(Move(cid, owner, dst, "missing", st["path"]))
            continue
        cat = layout.category_of(actual[len(osc["folder"]) + 1:], layout.scope_width(osc))
        moves.append(Move(cid, owner, dst, "file", actual, cat))
    dsts = set(wanted.values())
    covered = set(wanted)
    leaving = defaultdict(set)
    for m in moves:
        leaving[m.src].add(m.cid)
    unlisted = defaultdict(set)
    for cid, src, dst in members:
        unlisted[src].add(cid)
    remove = []
    for src in sorted((set(leaving) | set(unlisted)) - dsts):
        if not store.scope(src):
            continue
        owned = {r[0] for r in store.owned_rows(src)} - leaving[src]
        left = set(store.members(src)) - unlisted[src]
        if not owned and left <= covered:
            remove.append(src)
    create = {n: d for n, d in defs.items() if not store.scope(n) and n in dsts}
    return {"moves": moves, "members": members, "create": create, "remove": remove, "wanted": len(wanted)}


def summary(p: dict) -> list[str]:
    rows = Counter((m.src, m.dst, m.kind) for m in p["moves"])
    mem = Counter((src, dst) for _, src, dst in p["members"])
    pairs = sorted({(s, d) for s, d, _ in rows} | set(mem))
    out = []
    for s, d in pairs:
        parts = []
        if rows[(s, d, "file")]:
            parts.append(f"{rows[(s, d, 'file')]:,} saved photos")
        if rows[(s, d, "owner")]:
            parts.append(f"{rows[(s, d, 'owner')]:,} skipped or failed")
        if rows[(s, d, "missing")]:
            parts.append(f"{rows[(s, d, 'missing')]:,} files gone (to download again)")
        if mem[(s, d)]:
            parts.append(f"{mem[(s, d)]:,} listed photos")
        out.append(f"  {s} -> {d}: " + ", ".join(parts))
    for n, d in sorted(p["create"].items()):
        out.append(f"  new scope {n}: regions {d['regions']}, {layout.years_label(d['years'], d['decade'])}")
    for n in p["remove"]:
        out.append(f"  scope {n} is left empty and is removed")
    if not out:
        out.append("  nothing to move")
    return out


# --- moving ------------------------------------------------------------------------------

def repath_record(store: Store, base: Path, cid: int, rel: str, note: dict | None, undo_run: int | None = None):
    rec = store.record(cid)
    if rec is None:
        return None
    rec.setdefault("processing", {}).setdefault("output", {})["path"] = rel
    if note is not None:
        rec.setdefault("regrouped", []).append(note)
    elif undo_run is not None:
        rec["regrouped"] = [n for n in rec.get("regrouped") or [] if n.get("run") != undo_run]
        if not rec["regrouped"]:
            rec.pop("regrouped")
    text = json.dumps(rec, ensure_ascii=False, indent=1, default=str)
    rf = layout.record_file(base, cid)
    if rf.is_file():
        rf.write_text(text, encoding="utf-8")
    return text


def apply(store: Store, base: Path, p: dict, rule: str, target: str, log=print) -> dict:
    run = store.start_regroup(rule, target)
    for n, d in p["create"].items():          # the numbered folder width is set below, from what it lists
        store.add_scope(n, d["regions"], d["years"], d["decade"], d["split"], d["label"], 0)
    seq = store.move_members(run, 0, p["members"]) if p["members"] else 0
    # numbered folder widths for scopes that have none yet
    dsts = {m.dst for m in p["moves"]} | set(p["create"])
    for n in dsts:
        sc = store.scope(n)
        if sc and sc["split"] and not sc["width"]:
            store.set_scope_width(n, layout.chunk_width(max(store.scope_counts(n)["total"], 1), sc["split"]))
    chunkers = {n: folders.ScopeChunks(store, base, store.scope(n), log, defer=True) for n in dsts}
    stats = Counter()
    lost_files = set()
    t0 = time.time()
    for i, m in enumerate(p["moves"], 1):
        if m.kind == "file":
            fname = m.path.rsplit("/", 1)[-1]
            rel_scope = chunkers[m.dst].place(m.category, fname)
            rel = chunkers[m.dst].rel(rel_scope)
            try:
                move_pair(base / m.path, base / rel)
            except OSError as e:
                chunkers[m.dst].abandon(rel_scope)
                log(f"  {m.cid}: not moved ({e})")
                stats["failed"] += 1
                continue
            store.add_regroup_move(run, seq, m.cid, m.src, m.path, m.dst, rel, "file", False)
            rec = repath_record(store, base, m.cid, rel, {"time": time.time(), "run": run, "from": m.src,
                                                          "to": m.dst})
            store.set_owner(m.cid, m.dst, rel, m.category, rec)
            chunkers[m.dst].done(rel_scope)
            lost_files.add(m.src)
            stats["file"] += 1
        elif m.kind == "owner":
            st = store.status(m.cid)
            store.add_regroup_move(run, seq, m.cid, m.src, st["path"], m.dst, st["path"], "owner", False)
            store.set_owner(m.cid, m.dst, st["path"], st["category"])
            stats["owner"] += 1
        else:
            store.add_regroup_move(run, seq, m.cid, m.src, m.path, m.dst, None, "missing", False)
            store.forget_photos([m.cid])
            stats["missing"] += 1
        seq += 1
        if i % 1000 == 0:
            log(f"  {i:,} of {len(p['moves']):,} moved ({time.time() - t0:.0f} s)")
    for c in chunkers.values():
        c.settle()
    removed = []
    for n in p["remove"]:
        sc = store.scope(n)
        if not sc or store.owned_rows(n):
            continue
        removed.append({**{k: sc[k] for k in ("name", "regions", "years", "decade", "split", "width", "label",
                                              "folder")}, "members": store.members(n)})
        store.remove_scope(n)
        if not remove_folder(base, sc["folder"]):
            log(f"  {base / sc['folder']} still holds files that are not photos of this tool; the folder stays")
    for n in lost_files:
        sc = store.scope(n)
        if sc:
            prune_empty(base / sc["folder"])
        store.close_reclass_runs(n)
    touched = ({m.src for m in p["moves"]} | dsts | {s for _, s, _ in p["members"]}) - {r["name"] for r in removed}
    for n in touched:
        if store.scope(n):
            store.recount(n)
    store.end_regroup(run, stats["file"], removed, sorted(p["create"]))
    stats["members"] = len(p["members"])
    stats["removed"] = len(removed)
    stats["created"] = len(p["create"])
    return {"run": run, **{k: stats[k] for k in KEYS}}


def drop_scope(store: Store, base: Path, sc: dict, log=print) -> dict:
    """Delete the scope's files and forget the scope. Its photos are marked not
    downloaded, with no record, embedding or VLM caption, so a later run
    fetches and processes them from scratch. Cannot be undone."""
    name = sc["name"]
    rows = store.owned_rows(name)
    deleted = 0
    for cid, state, path, category in rows:
        if path:
            img = base / path
            for f in (img, img.with_suffix(".txt"), img.with_suffix(".json")):
                if f.is_file():
                    f.unlink()
                    deleted += f is img
        rf = layout.record_file(base, cid)
        if rf.is_file():
            rf.unlink()
    store.forget_photos([r[0] for r in rows])
    store.close_reclass_runs(name)
    store.remove_scope(name)
    gone = remove_folder(base, sc["folder"])
    if not gone:
        log(f"  {base / sc['folder']} still holds files that are not photos of this tool; the folder stays")
    return {"photos": len(rows), "files": deleted, "folder_removed": gone}


def undo(store: Store, base: Path, log=print) -> dict:
    last = store.last_regroup()
    if not last:
        raise SystemExit("no regroup to undo in this base folder")
    run = last["id"]
    for r in last["removed"]:
        if not store.scope(r["name"]):
            store.add_scope(r["name"], r["regions"], tuple(r["years"]), r["decade"], r["split"], r.get("label"),
                            r["width"], r.get("folder"))
        store.add_members(r["name"], r["members"])
    stats = Counter()
    touched = set()
    chunkers: dict[str, folders.ScopeChunks] = {}
    for seq, cid, src, src_path, dst, dst_path, kind, new_member in store.regroup_moves(run):
        touched |= {src, dst}
        if kind == "member":
            if new_member:
                store.remove_member(dst, cid)
            store.add_member(src, cid)
            stats["members"] += 1
        elif kind == "file":
            st = store.status(cid)
            if not st or st["scope"] != dst:
                log(f"  {cid}: no longer in {dst}; left where it is")
                stats["skipped"] += 1
                continue
            dsc, ssc = store.scope(dst), store.scope(src)
            here = folders.locate(base, dsc["folder"] if dsc else dst, cid, st["path"], {})
            if here is None or ssc is None:
                log(f"  {cid}: the file is gone; left as it is")
                stats["skipped"] += 1
                continue
            if src not in chunkers:
                chunkers[src] = folders.ScopeChunks(store, base, ssc, log, defer=True)
            ch = chunkers[src]
            old_rel = src_path[len(ssc["folder"]) + 1:] if src_path.startswith(ssc["folder"] + "/") else src_path
            cat = layout.category_of(old_rel, layout.scope_width(ssc))
            back_rel = ch.place(cat, here.rsplit("/", 1)[-1])
            try:
                move_pair(base / here, base / ch.rel(back_rel))
            except OSError as e:
                ch.abandon(back_rel)
                log(f"  {cid}: not moved back ({e})")
                stats["skipped"] += 1
                continue
            rec = repath_record(store, base, cid, ch.rel(back_rel), None, undo_run=run)
            store.set_owner(cid, src, ch.rel(back_rel), cat, rec)
            ch.done(back_rel)
            stats["file"] += 1
        elif kind == "owner":
            st = store.status(cid)
            if st and st["scope"] == dst:
                store.set_owner(cid, src, st["path"], st["category"])
                stats["owner"] += 1
        else:
            stats["missing"] += 1
    for c in chunkers.values():
        c.settle()
    for n in last["created"]:
        sc = store.scope(n)
        if sc and not store.members(n) and not store.owned_rows(n):
            store.remove_scope(n)
            remove_folder(base, sc["folder"])
            touched.discard(n)
            stats["removed"] += 1
    for n in touched:
        sc = store.scope(n)
        if sc:
            prune_empty(base / sc["folder"])
            store.recount(n)
    store.undo_regroup(run)
    return {"run": run, "rule": last["rule"], "target": last["target"], **{k: stats[k] for k in KEYS}}
