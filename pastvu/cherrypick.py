"""cherrypick: a training set picked by quality from a base folder, balanced
between colour and black-and-white.

    run.bat cherrypick 0.85 30 5000 -o D:\\training

The unit is one category in one mode (bw, color): a bucket. Each bucket is
ranked by P(good) = 1 - P(bad) of the quality probe, on the embeddings the
store holds, and cut by whichever of the ceilings given is the lowest:

- a threshold (a value with a dot, 0.85): P(good) at or above it;
- a percentile (1 to 100): the top X % of the bucket as it is in the base;
- a number per category (over 100): the category's photos, split between the
  modes by their pools' ratio (see balance).

Balance (--balance, always on with a number per category): black-and-white
never exceeds colour. After the threshold and the percentile, with pools c
and b, b' = min(b, c) and c' = c; with a number N, total = min(N, c' + b')
is split in the ratio c' : b'. More colour than black-and-white is fine; the
other way round is not. Without balance each mode is cut on its own.

The decades are all, or --decades (from the photo's start year; not on an
import, which has no years). 200-outliers is never included. Photos come
from the store (category, mode, path), so a hand move counts once reclassify
has recorded it. A photo without an embedding is left out and counted.

The output holds <category>/<bw|color>/[NNN/]<name> (color_first in the
conf's export_layout swaps the first two). Images are copied, or hard-linked
with --link; .txt and .short captions are copied. cherrypick.json in the
output holds, per base, the criteria and the files, so an import and the
PastVu base can go into one folder. A rerun copies what is selected and
missing, keeps what is there, reports the files of this base that are no
longer selected, and deletes them with --prune.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import extract as xt
import folders
import layout
import numpy as np
from store import DONE

MANIFEST = "cherrypick.json"
OUTLIERS = "outliers"
SIDECARS = xt.SIDECARS


# --- the parameters -----------------------------------------------------------------------------


def parse_values(values: list[str], quality=None, top=None, per_category=None) -> dict:
    """The ceilings from the positional values (by their shape) and the named forms.
    -> {"quality": float | None, "top": int | None, "per_category": int | None}"""
    out = {"quality": quality, "top": top, "per_category": per_category}
    for v in values:
        t = v.strip()
        if "." in t:
            kind, val = "quality", _float(t)
            if not 0 < val <= 1:
                raise SystemExit(f"{v}: a quality threshold is a P(good) above 0 and at most 1, e.g. 0.85")
        elif t.isdigit():
            n = int(t)
            if 1 <= n <= 100:
                kind, val = "top", n
            elif n > 100:
                kind, val = "per_category", n
            else:
                raise SystemExit(f"{v}: give a percentile from 1 to 100, a number per category over 100, "
                                 f"or a threshold with a dot (0.85)")
        else:
            raise SystemExit(f"{v}: not a threshold (0.85), a percentile (30) or a number per category (5000)")
        if out[kind] is not None:
            raise SystemExit(f"{v}: the {kind.replace('_', ' ')} is given twice")
        out[kind] = val
    if out["top"] is not None and not 1 <= out["top"] <= 100:
        raise SystemExit("--top takes a percentile from 1 to 100")
    if out["per_category"] is not None and out["per_category"] < 1:
        raise SystemExit("--per-category takes a number of photos, 1 or more")
    if out["quality"] is not None and not 0 < out["quality"] <= 1:
        raise SystemExit("--quality takes a P(good) above 0 and at most 1")
    if all(v is None for v in out.values()):
        raise SystemExit("give at least one of: a quality threshold (0.85), a percentile (30), a number per "
                         "category (5000)")
    return out


def _float(t: str) -> float:
    try:
        return float(t)
    except ValueError:
        raise SystemExit(f"{t}: not a number")


def describe(c: dict, balance: bool, decades: list[int] | None) -> str:
    """The reading of the parameters, in words."""
    parts = []
    if c["quality"] is not None:
        parts.append(f"P(good) at least {c['quality']:g}")
    if c["top"] is not None:
        parts.append(f"the top {c['top']} % of each bucket")
    if c["per_category"] is not None:
        parts.append(f"at most {c['per_category']:,} photos per category")
    text = "For each category, in black-and-white and in colour separately: " + "; ".join(parts) + "."
    if balance:
        text += (" Balance: black-and-white is cut down to the colour count"
                 + (", and the number per category is split in the ratio of the two pools" if c["per_category"] else "")
                 + ".")
    else:
        text += " No balance: each mode is cut on its own."
    text += " Decades: " + (", ".join(f"{d}s" for d in decades) if decades else "all") + "."
    return text


def parse_decades(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
    out = []
    for v in values:
        t = v.strip().lower().rstrip("s")
        if not (t.isdigit() and len(t) == 4 and int(t) % 10 == 0):
            raise SystemExit(f"--decades {v}: give decades as 1960s or 1960")
        out.append(int(t))
    return sorted(set(out))


# --- the photos and their scores ------------------------------------------------------------------


def load_photos(store, decades: list[int] | None, categories: list[str] | None, log=print) -> dict:
    """Every saved photo of the base, outliers left out: cid -> {"cat", "mode", "path", "decade"}.
    categories: extract-style arguments (050, transport, 050-transport) that narrow the set."""
    with store.lock:
        rows = store.db.execute("SELECT s.cid, s.category, s.mode, s.path, p.year FROM status s LEFT JOIN photos p "
                                "ON p.cid=s.cid WHERE s.state=? AND s.path IS NOT NULL", (DONE,)).fetchall()
    names = Counter(r[1] for r in rows if r[1])
    keep = None
    if categories:
        keep = set()
        for arg in categories:
            keep.update(xt.resolve_category(arg, names))
    out = {}
    for cid, cat, mode, path, year in rows:
        if not cat or xt.split_name(cat)[1] == OUTLIERS or (keep is not None and cat not in keep):
            continue
        dec = (year // 10 * 10) if year is not None else None
        if decades and dec not in decades:
            continue
        out[cid] = {"cat": cat, "mode": mode or "bw", "path": path, "decade": dec}
    return out


def score(store, quality, model: str, patches: int, cids: set[int], log=print) -> dict[int, float]:
    """P(good) of each photo from its stored embedding, in one pass over the embeddings table."""
    out = {}
    t0 = time.time()
    with store.lock:
        cur = store.db.execute("SELECT cid, vec FROM embeddings WHERE model=? AND patches=?", (model, patches))
        while True:
            rows = cur.fetchmany(20000)
            if not rows:
                break
            rows = [r for r in rows if r[0] in cids]
            if not rows:
                continue
            V = np.stack([np.frombuffer(r[1], np.float16).astype(np.float32) for r in rows])
            p = 1.0 - quality.p_bad(V)
            for (cid, _), q in zip(rows, p):
                out[cid] = float(q)
    log(f"  {len(out):,} photos scored in {time.time() - t0:.0f} s")
    return out


# --- the selection --------------------------------------------------------------------------------


def _cut(ranked: list[int], scores: dict[int, float], c: dict) -> list[int]:
    """One bucket, ranked best first, under the threshold and the percentile."""
    n = len(ranked)
    if c["quality"] is not None:
        n = min(n, sum(1 for cid in ranked if scores[cid] >= c["quality"]))
    if c["top"] is not None:
        n = min(n, len(ranked) * c["top"] // 100)
    return ranked[:n]


def select(photos: dict, scores: dict[int, float], c: dict, balance: bool) -> tuple[dict[int, dict], list[dict]]:
    """The picked photos and a table row per category.
    -> ({cid: photo}, [{"cat", "color_pool", "bw_pool", "color", "bw", "note"}])"""
    buckets: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"color": [], "bw": []})
    unscored = 0
    for cid, ph in photos.items():
        if cid not in scores:
            unscored += 1
            continue
        buckets[ph["cat"]][ph["mode"]].append(cid)
    picked: dict[int, dict] = {}
    table = []
    for cat in sorted(buckets):
        ranked = {m: sorted(cids, key=lambda cid: (-scores[cid], cid)) for m, cids in buckets[cat].items()}
        pools = {m: len(v) for m, v in ranked.items()}
        cut = {m: _cut(v, scores, c) for m, v in ranked.items()}
        want = {m: len(v) for m, v in cut.items()}
        note = ""
        if balance:
            want["bw"] = min(want["bw"], want["color"])
            if c["per_category"] is not None:
                total = min(c["per_category"], want["color"] + want["bw"])
                if total:
                    colour = round(total * want["color"] / (want["color"] + want["bw"]))
                    want = {"color": colour, "bw": total - colour}
            if not want["color"] and pools["bw"]:
                note = "no colour photos pass: nothing picked under balance"
        for m in ("color", "bw"):
            for cid in cut[m][:want[m]]:
                picked[cid] = photos[cid]
        table.append({"cat": cat, "color_pool": pools["color"], "bw_pool": pools["bw"], "color": want["color"],
                      "bw": want["bw"], "note": note})
    if unscored:
        table.append({"cat": "(no embedding, left out)", "color_pool": unscored, "bw_pool": 0, "color": 0, "bw": 0,
                      "note": ""})
    return picked, table


def table_text(table: list[dict]) -> str:
    lines = [f"  {'category':30} {'colour':>16}   {'black-and-white':>16}"]
    for r in table:
        lines.append(f"  {r['cat']:30} {r['color_pool']:7,} -> {r['color']:6,}   {r['bw_pool']:7,} -> {r['bw']:6,}"
                     + (f"   {r['note']}" if r["note"] else ""))
    tc, tb = sum(r["color"] for r in table), sum(r["bw"] for r in table)
    lines.append(f"  {'total':30} {'':7} -> {tc:6,}   {'':7} -> {tb:6,}   ({tc + tb:,} photos)")
    return "\n".join(lines)


# --- the output -----------------------------------------------------------------------------------


def _read_manifest(out: Path) -> dict:
    f = out / MANIFEST
    if f.is_file():
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(m.get("bases"), dict):
                return m
        except (OSError, ValueError):
            pass
    return {"version": 1, "bases": {}}


def write(picked: dict[int, dict], base: Path, out: Path, criteria: dict, order: str, split: int = 0,
          link: bool = False, prune: bool = False, dry_run: bool = False, log=print) -> dict:
    """Copy (or link) the picked photos into out, keeping what is there. -> counts."""
    manifest = _read_manifest(out)
    mine = manifest["bases"].get(str(base), {})
    old = {int(k): v for k, v in (mine.get("files") or {}).items()}
    present = {cid: rel for cid, rel in old.items() if (out / rel).is_file()}
    stats = Counter({k: 0 for k in ("copied", "linked", "there", "new", "failed", "gone", "stale", "pruned")})
    numbered = None
    if split > 0:
        numbered = xt.Numbered(split, layout.chunk_width(len(picked) + len(present), split), present.values())
    files = dict(present)
    t0 = time.time()
    for i, (cid, ph) in enumerate(sorted(picked.items()), 1):
        if cid in present:
            stats["there"] += 1
            continue
        if dry_run:
            stats["new"] += 1
            continue
        src = base / ph["path"]
        if not src.is_file():
            stats["gone"] += 1
            continue
        folder = f"{ph['mode']}/{ph['cat']}" if order == layout.COLOR_FIRST else f"{ph['cat']}/{ph['mode']}"
        rel = numbered.place(folder, src.name) if numbered else f"{folder}/{src.name}"
        dst = out / rel
        if dst.exists():                                   # another base's photo of the same name
            log(f"  {cid}: {rel} is there already from another base; not copied")
            stats["failed"] += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
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
        files[cid] = rel
        if i % 2000 == 0:
            log(f"  {i:,} of {len(picked):,} ({time.time() - t0:.0f} s)")
    stale = {cid: rel for cid, rel in files.items() if cid not in picked}
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
        manifest["bases"][str(base)] = {"criteria": criteria, "time": time.time(),
                                        "files": {str(k): v for k, v in sorted(files.items())}}
        (out / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    stats["selected"] = len(picked)
    return dict(stats)
