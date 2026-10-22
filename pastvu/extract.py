"""extract: copy the photos of one category out of a base folder into a flat
folder for training, keeping only the bw/ and color/ level.

    <base parent>/050-transport_1970s/bw/123456.jpg (+ .txt, .short)
                                     color/...

- The category is named by its number (050 or 50), its name (transport) or
  its full name (050-transport). Scopes sorted with different samples can
  hold different folders under one number: a name that matches two different
  categories is refused, and the full name settles it.
- The photos are found where they are on disk (the category folders of every
  scope), so a photo moved by hand to another category or between bw and
  color is extracted from where it sits now.
- Filters narrow the selection and add to the default folder name, never to
  the tree: --region (the photo's own region chain), --years (dating
  overlaps), --decade (start year), --scope (names or patterns), --bw/--color.
- Running it again tops the folder up: what is new is copied, files already
  there are left as they are (captions edited for training stay), and files
  that no longer match are reported; --prune deletes those. The folder's
  extract.json holds the criteria and the files it copied.
- Images are copied; --link hard-links them instead (captions are always
  copied).
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import folders
import layout

SIDECARS = (".txt", ".short")
MANIFEST = "extract.json"
LINK_TIP = ("Tip: --link hard-links the images instead of copying them, so they take no extra disk space on the "
            "same drive (captions are still copied). A tool that edits images in place would then change the "
            "base's files too.")
_NUM = re.compile(r"^(\d+)[-_ ]*(.*)$")


def norm(text: str) -> str:
    return re.sub(r"[-_\s]+", " ", text.strip().lower()).strip()


def split_name(folder: str) -> tuple[int | None, str]:
    """'050-transport' -> (50, 'transport'); 'outliers' -> (None, 'outliers')."""
    m = _NUM.match(folder)
    if m:
        return int(m.group(1)), norm(m.group(2))
    return None, norm(folder)


def matches(arg: str, folder: str) -> bool:
    num, name = split_name(folder)
    if arg.strip().isdigit():
        return num is not None and num == int(arg)
    a_num, a_name = split_name(arg)
    if a_num is not None:                            # a full name: number and name both
        return num == a_num and name == a_name
    return name == norm(arg)


def category_folders(base: Path, sc: dict) -> dict[str, list[tuple[str | None, Path]]]:
    """Category folder name -> [(mode, folder on disk)] of a scope, from disk."""
    out = defaultdict(list)
    for prefix, root in layout.scope_roots(sc):
        d = base / root
        if not d.is_dir():
            continue
        mode = prefix.rstrip("/") or None
        for e in os.scandir(d):
            if e.is_dir() and not e.name.startswith((".", "_")) and not e.name.isdigit():
                out[e.name].append((mode, Path(e.path)))
    return out


def resolve_category(arg: str, names: Counter) -> list[str]:
    """The folder names of all scopes that one argument means. Names that only
    differ in how they are written (50_transport, 050-transport) are one category."""
    hits = [n for n in names if matches(arg, n)]
    if not hits:
        known = ", ".join(sorted(names)) or "none"
        raise SystemExit(f"no category matches {arg!r}; the categories on disk are: {known}")
    kinds = {split_name(n) for n in hits}
    if len(kinds) > 1:
        listing = ", ".join(f"{n} ({names[n]:,} folders)" for n in sorted(hits))
        raise SystemExit(f"{arg!r} matches different categories: {listing}. Give the full name.")
    return hits


def label_of(hits: list[str], names: Counter) -> str:
    """The output folder's category part: the most used way of writing it."""
    return max(hits, key=lambda n: (names[n], n))


def _scan(folder: Path) -> dict[int, Path]:
    out = {}
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "_"))]
        for fn in filenames:
            stem, ext = os.path.splitext(fn)
            if stem.isdigit() and ext.lower() in layout.IMAGE_EXTS:
                out[int(stem)] = Path(dirpath) / fn
    return out


def select(store, base: Path, scopes: list[dict], args_categories: list[str], filters: dict, log=print) -> dict:
    """-> {label: {"hits": [...], "photos": {cid: (mode, source path)}}} per category argument."""
    per_scope = {sc["name"]: category_folders(base, sc) for sc in scopes}
    names = Counter(n for cats in per_scope.values() for n in cats)
    region_set = set(filters.get("regions") or [])
    years, decade = filters.get("years"), filters.get("decade")
    want_mode = filters.get("mode")
    out = {}
    for arg in args_categories:
        hits = resolve_category(arg, names)
        photos: dict[int, tuple[str | None, Path]] = {}
        for sc in scopes:
            found = {}
            for n in hits:
                for mode, d in per_scope[sc["name"]].get(n, []):
                    for cid, path in _scan(d).items():
                        found[cid] = (mode, path)
            if not found:
                continue
            cids = list(found)
            facts = _facts(store, cids, need_regions=bool(region_set))
            for cid in cids:
                f = facts.get(cid)
                if not f or f["state"] != "done" or f["scope"] != sc["name"]:
                    continue                          # not finished, or another scope's
                mode = found[cid][0] or f["mode"]
                if want_mode and mode != want_mode:
                    continue
                y0 = f["year"]
                y1 = f["year2"] if f["year2"] is not None else y0
                if decade is not None and (y0 is None or not decade <= y0 <= decade + 9):
                    continue
                if years and (y0 is None or y0 > years[1] or y1 < years[0]):
                    continue
                if region_set and not (region_set & set(f["chain"])):
                    continue
                photos[cid] = (mode, found[cid][1])
        out[label_of(hits, names)] = {"arg": arg, "hits": hits, "photos": photos}
    return out


def _facts(store, cids: list[int], need_regions: bool) -> dict[int, dict]:
    out = {}
    with store.lock:
        for i in range(0, len(cids), 900):
            part = cids[i:i + 900]
            q = ",".join("?" * len(part))
            for cid, state, scope, mode, year, year2, rs in store.db.execute(
                    f"SELECT s.cid, s.state, s.scope, s.mode, p.year, p.year2, p.rs FROM status s "
                    f"LEFT JOIN photos p ON p.cid=s.cid WHERE s.cid IN ({q})", part):
                out[cid] = {"state": state, "scope": scope, "mode": mode, "year": year, "year2": year2,
                            "rs": json.loads(rs or "[]"), "chain": []}
    if need_regions:
        for cid, f in out.items():
            rec = store.record(cid) or {}
            chain = [int(r["cid"]) for r in rec.get("regions") or [] if r.get("cid")]
            if not chain:
                for c in f["rs"]:
                    r = store.region(int(c)) if c else None
                    if r:
                        chain = list(r["parents"]) + [int(c)]
                        break
            f["chain"] = chain
    return out


def folder_name(label: str, filters: dict) -> str:
    parts = [label]
    if filters.get("region_labels"):
        parts.append("+".join(filters["region_labels"]))
    if filters.get("decade") is not None:
        parts.append(f"{filters['decade']}s")
    elif filters.get("years"):
        parts.append(f"{filters['years'][0]}-{filters['years'][1]}")
    if filters.get("scopes"):
        parts.append("+".join(layout.slug(s.replace("*", "")) or "all" for s in filters["scopes"]))
    if filters.get("mode"):
        parts.append(filters["mode"])
    return "_".join(parts)


class Numbered:
    """--split N: numbered folders 001, 002, ... of at most N images under each
    of bw/ and color/, from the first image on; the lowest one with room fills first."""

    def __init__(self, split: int, width: int, rels):
        self.split, self.width = split, width
        self.counts: dict[str, Counter] = defaultdict(Counter)
        for rel in rels:
            parts = rel.split("/")
            if len(parts) >= 2 and parts[-2].isdigit():
                self.counts["/".join(parts[:-2])][int(parts[-2])] += 1

    def place(self, folder: str, name: str) -> str:
        c = self.counts[folder]
        chunk = 1
        while c[chunk] >= self.split:
            chunk += 1
        c[chunk] += 1
        return "/".join(p for p in (folder, f"{chunk:0{self.width}d}", name) if p)


def _place(out: Path, mode: str | None, name: str, numbered) -> Path:
    rel = numbered.place(mode or "", name) if numbered else (f"{mode}/{name}" if mode else name)
    return out / rel


def extract(photos: dict[int, tuple[str | None, Path]], out: Path, criteria: dict, split: int = 0,
            link: bool = False, prune: bool = False, dry_run: bool = False, log=print) -> dict:
    """Copy (or link) the photos into out, topping up what is there. -> counts."""
    manifest_file = out / MANIFEST
    old = {}
    if manifest_file.is_file():
        try:
            old = {int(k): v for k, v in json.loads(manifest_file.read_text(encoding="utf-8")).get("files", {}).items()}
        except (OSError, ValueError):
            old = {}
    present = folders.scan_scope(out) if out.is_dir() else {}           # cid -> rel, as it is now
    stats = Counter({k: 0 for k in ("copied", "linked", "there", "new", "failed", "gone", "stale", "pruned")})
    chunker = None
    if split > 0:
        chunker = Numbered(split, layout.chunk_width(len(photos) + len(present), split), present.values())
    files = dict(old)
    t0 = time.time()
    for i, (cid, (mode, src)) in enumerate(sorted(photos.items()), 1):
        if cid in present:
            stats["there"] += 1
            files[cid] = present[cid]
            continue
        if dry_run:
            stats["new"] += 1
            continue
        dst = _place(out, mode, src.name, chunker)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.is_file():
            stats["gone"] += 1                         # moved since the scan (a category passing the split)
            continue
        try:
            if link:
                try:
                    os.link(src, dst)
                    stats["linked"] += 1
                except OSError:
                    shutil.copy2(src, dst)
                    stats["copied"] += 1
            else:
                shutil.copy2(src, dst)
                stats["copied"] += 1
            for ext in SIDECARS:
                s2 = src.with_suffix(ext)
                if s2.is_file():
                    shutil.copy2(s2, dst.with_suffix(ext))
        except OSError as e:
            log(f"  {cid}: not copied ({e})")
            stats["failed"] += 1
            continue
        files[cid] = dst.relative_to(out).as_posix()
        if i % 2000 == 0:
            log(f"  {i:,} of {len(photos):,} ({time.time() - t0:.0f} s)")
    # files the folder holds from earlier extracts that no longer match
    stale = {cid: rel for cid, rel in files.items() if cid not in photos}
    stats["stale"] = len(stale)
    if prune and not dry_run:
        for cid, rel in stale.items():
            img = out / rel
            for f in (img, *(img.with_suffix(e) for e in SIDECARS)):
                if f.is_file():
                    f.unlink()
            files.pop(cid, None)
        stats["pruned"] = len(stale)
        folders.prune_empty(out)
    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text(json.dumps({"version": 1, "time": time.time(), "criteria": criteria,
                                             "files": {str(k): v for k, v in sorted(files.items())}},
                                            ensure_ascii=False, indent=1), encoding="utf-8")
    stats["selected"] = len(photos)
    stats["modes"] = dict(Counter(m or "-" for m, _ in photos.values()))
    return dict(stats)


def scope_filter(scopes: list[dict], patterns: list[str] | None) -> list[dict]:
    if not patterns:
        return scopes
    out = [sc for sc in scopes if any(fnmatch.fnmatchcase(sc["name"], p) for p in patterns)]
    if not out:
        raise SystemExit(f"no scope matches {', '.join(patterns)}")
    return out
