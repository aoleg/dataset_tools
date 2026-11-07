"""deduplicate: copies of one photo inside the base (PLAN.md section 24).

The hashes. Every saved photo gets a row in the store's hashes
table: pHash and dHash of the picture and of its mirror image, the luma
spread and the colourfulness, by the deduplicate tool's recipe
(dedup.image_hashes: Pillow draft decode at 512, upright, 256 px LANCZOS
thumbnail, luma), so the two tools agree on every hash. The checksum is not
computed again: the store has each file's sha256.

The hashes are 64-bit; SQLite integers are signed, so a hash is stored as
the signed integer of the same bits (to_int64) and read back as unsigned
(Store.hash_rows).

The photos are read and hashed in a pool of threads (Pillow releases the
GIL while it decodes), at most threads x AHEAD ahead of the writer; the
rows go BATCH per transaction. The path is asked from the store at read
time, as niqe does (a reclassify may have moved the photo). A photo that is
missing, unreadable or no longer saved gets no row, so the next run lists
it again.

The copies (find). Inside PastVu the SigLIP cosine cannot decide on its
own: two thirds of the photos have a neighbour at 0.90 or more, and most
neighbours at 0.96 to 0.98 are other frames of the same roll. So:

1. Candidates: the NEIGHBOURS nearest photos of every saved photo by the
   cosine of the stored embeddings, at min_cos or more (GPU, in blocks of
   queries against the whole matrix).
2. Every pair of byte-identical files (same sha256) is a copy (exact),
   whatever the embeddings say.
3. A candidate pair is no copy when either picture is near-uniform
   (dedup.FLAT_LUMA_STD) or when one is clearly in colour and the other is
   not (dedup's colour veto).
4. dedup's strict hash rule (pHash <= 8 and dHash <= 10) on the pictures
   as they are: a copy (hash); against the mirror image of one of them: a
   copy (mirror).
5. Otherwise within dedup's candidate window (both distances <= 22, plain
   or mirrored): the feature test of the importer (ORB, RANSAC homography,
   warped-centre NCC) decides (features). These pairs are also written to
   a file of their own with their numbers, for calibration.
6. Anything else: no copy.

Groups are the connected components of the copy pairs. The keeper is the
deduplicate tool's choice (rank_members on score_file: the largest training
bucket the copy reaches, then the rendered score, with the guard), after a
member with painted watermarks steps aside for one without; a tie goes to
the larger picture, then to the lower number (the earlier upload).
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent / "deduplicate") not in sys.path:
    sys.path.insert(0, str(HERE.parent / "deduplicate"))
import dedup  # noqa: E402

HASH_VERSION = "dh1-256"      # dedup's recipe at 256 px, plain and mirrored; a change makes stored rows stale
THREADS = 12
AHEAD = 8                     # photos per thread read ahead of the writer
BATCH = 512                   # rows per transaction


def to_int64(hex_hash: str) -> int:
    """A 64-bit hash in hex as the signed integer of the same bits (SQLite's INTEGER)."""
    v = int(hex_hash, 16)
    return v - (1 << 64) if v >= 1 << 63 else v


def hash_row(cid: int, path: Path) -> tuple:
    """The hashes row of one file: (cid, HASH_VERSION, phash, dhash, phash_m, dhash_m, luma_std, colour)."""
    from PIL import Image
    with Image.open(path) as im:
        h = dedup.image_hashes(im, mirror=True)
    return (cid, HASH_VERSION, to_int64(h["phash"]), to_int64(h["dhash"]), to_int64(h["phash_m"]),
            to_int64(h["dhash_m"]), float(h["luma_std"]), float(h["colour"]))


def read_photo(store, base: Path, cid: int, path: str) -> tuple:
    """(cid, rel, row or None, why None): the photo's path from the store now, its file hashed."""
    from store import DONE
    st = store.status(cid)
    if not st or st.get("state") != DONE or not st.get("path"):
        return cid, path, None, "gone"
    rel = st["path"]
    f = base / rel
    if not f.is_file():
        return cid, rel, None, "missing"
    try:
        return cid, rel, hash_row(cid, f), None
    except Exception:  # noqa: BLE001 - a damaged file is counted, not fatal
        return cid, rel, None, "unreadable"


def save_rows(store, rows: list[tuple], tries: int = 10) -> None:
    """save_hashes, waiting out a writer that holds the database longer than the busy timeout."""
    for attempt in range(tries):
        try:
            store.save_hashes(rows)
            return
        except sqlite3.OperationalError as e:
            if "locked" not in str(e) or attempt == tries - 1:
                raise
            time.sleep(2)


def report(stats: Counter, total: int, seconds: float, final: bool = False) -> str:
    done = stats["hashed"]
    rate = done / seconds if seconds > 0 else 0.0
    left = max(0, total - sum(stats[k] for k in ("hashed", "missing", "unreadable", "gone")))
    eta = f", about {left / rate / 60:,.0f} min left" if rate > 0 and left and not final else ""
    text = (f"{'Finished: ' if final else ''}{done:,} of {total:,} photos hashed in {seconds / 60:.1f} min "
            f"({rate:.1f}/s{eta})")
    rest = [f"{stats[k]:,} {k}" for k in ("missing", "unreadable", "gone") if stats[k]]
    return text + ("; " + ", ".join(rest) if rest else "")


def hash_photos(store, base: Path, targets: list[tuple[int, str]], display, report_every: int = 50000,
                threads: int = THREADS, stop=None) -> dict:
    """Hash the targets ((cid, path), in order) and write their rows. -> the counts, "stopped"
    ("Ctrl+C" or None) and "total"."""
    stats: Counter = Counter()
    rows: list[tuple] = []
    queue_ = deque(targets)
    window: deque = deque()
    total = len(targets)
    t0 = time.time()
    stopped = None

    def flush():
        if rows:
            save_rows(store, list(rows))
            rows.clear()

    pool = ThreadPoolExecutor(max(1, threads))
    try:
        while True:
            while queue_ and len(window) < max(1, threads) * AHEAD:
                window.append(pool.submit(read_photo, store, base, *queue_.popleft()))
            if not window:
                break
            cid, rel, row, why = window.popleft().result()
            if row is None:
                stats[why] += 1
                display.line({"cid": cid, "state": why, "path": rel})
            else:
                rows.append(row)
                stats["hashed"] += 1
                display.set(f"{cid}: {rel}")
                display.line({"cid": cid, "state": "hashed"})
                if len(rows) >= BATCH:
                    flush()
            display.advance(saved=row is not None)
            n = sum(stats.values())
            if report_every and n % report_every == 0:
                display.print(report(stats, total, time.time() - t0))
            if stop is not None and stop.is_set():
                stopped = "Ctrl+C"
                break
    except KeyboardInterrupt:
        stopped = "Ctrl+C"
    finally:
        for f in window:
            f.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        flush()
    display.print(report(stats, total, time.time() - t0, final=True))
    if stopped:
        display.print(f"Stopped: {stopped}. Run the same command again to continue.")
    out = dict(stats)
    out.update(stopped=stopped, total=total)
    return out


# --- the copies ------------------------------------------------------------------------------

MIN_COS = 0.90                # candidates: no hash-rule copy was found below 0.93 in a sample of the base
NEIGHBOURS = 10               # nearest photos per photo
QUERY_BLOCK = 1024            # queries per GPU block: 1024 x 1M float16 similarities is 2 GB
STRICT = dedup.MATCH_RULES["strict"][0]          # (8, 10)
WINDOW = dedup.CAND_MAX_PHASH                    # 22: the feature test's candidate window
FEATURE_THREADS = 12
KINDS = ("exact", "hash", "mirror", "features")  # copy pairs, most certain first
SURE_KINDS = KINDS[:3]                           # the kinds that make groups by default (not features)
DUP_SUFFIX = "_duplicates"                       # <base>_duplicates: the copies moved out, by their base paths
# Candidate pairs, by what the hashes say: far, the feature window, a strict hash or mirror match, or
# a veto (a near-uniform picture, colour against black-and-white), or a picture without a hashes row.
FAR, WIN, HASH, MIRROR, FLAT, COLOUR, UNHASHED = range(7)
CODE_NAMES = ("far", "window", "hash", "mirror", "flat", "colour", "unhashed")
REVIEW_SUFFIX = "_duplicates_review"


def load_embeddings(store, model: str, patches: int) -> tuple[np.ndarray, np.ndarray]:
    """(cids int64, embeddings float16 L2-normalised [n, d]) of every saved photo with an embedding of
    this model and patch budget, in cid order. One pass over the table in its own order, sorted
    afterwards: an ORDER BY or a LENGTH() over the blobs made SQLite sort 2.3 GB through temp files
    (160 GB read in 15 minutes on the live base)."""
    from store import DONE
    with store.lock:
        cur = store.db.execute("SELECT e.cid, e.vec FROM embeddings e JOIN status s ON s.cid=e.cid "
                               "WHERE s.state=? AND e.model=? AND e.patches=?", (DONE, model, patches))
        ids, parts = [], []
        while rows := cur.fetchmany(20000):
            x = np.stack([np.frombuffer(r[1], np.float16) for r in rows]).astype(np.float32)
            x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
            ids.append(np.fromiter((r[0] for r in rows), np.int64, len(rows)))
            parts.append(x.astype(np.float16))
    if not ids:
        return np.zeros(0, np.int64), np.zeros((0, 0), np.float16)
    cids = np.concatenate(ids)
    order = np.argsort(cids, kind="stable")
    V = np.concatenate(parts)
    del parts
    return cids[order], V[order]


def neighbours(V: np.ndarray, k: int = NEIGHBOURS, min_cos: float = MIN_COS, block: int = QUERY_BLOCK,
               device: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The pairs (a, b, cosine) with a < b (rows of V) where b is among the k nearest of a or a among
    the k nearest of b, at min_cos or more; each pair once. GPU when there is one, in float16."""
    n = len(V)
    k = min(k, n - 1)
    if k < 1:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    qa, qb, qc = [], [], []
    with torch.no_grad():
        M = torch.from_numpy(V).to(dev)
        if dev == "cpu":
            M = M.float()
        for s in range(0, n, block):
            e = min(n, s + block)
            S = M[s:e] @ M.T
            r = torch.arange(e - s, device=dev)
            S[r, r + s] = -2.0                               # not its own neighbour
            vals, idx = S.topk(k, dim=1)
            vals = vals.float()                              # the floor in float32: float16 0.9 is 0.8999
            keep = vals >= min_cos
            rows = keep.nonzero(as_tuple=True)[0]
            qa.append((rows + s).cpu().numpy())
            qb.append(idx[keep].cpu().numpy())
            qc.append(vals[keep].float().cpu().numpy())
            del S
        del M
        if dev == "cuda":
            torch.cuda.empty_cache()
    a, b, c = np.concatenate(qa), np.concatenate(qb).astype(np.int64), np.concatenate(qc)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    _, first = np.unique(lo * n + hi, return_index=True)
    return lo[first], hi[first], c[first]


def classify_pairs(ca: np.ndarray, cb: np.ndarray, H: dict) -> dict[str, np.ndarray]:
    """What the hashes say of each pair of photos (cids ca[i], cb[i]): "code" (FAR ... UNHASHED) and the
    Hamming distances "p", "d" (as they are) and "pm", "dm" (b mirrored), -1 without hashes."""
    pos_a = np.searchsorted(H["cid"], ca)
    pos_b = np.searchsorted(H["cid"], cb)
    n_h = len(H["cid"])
    have = (pos_a < n_h) & (pos_b < n_h)
    have[have] &= (H["cid"][pos_a[have]] == ca[have]) & (H["cid"][pos_b[have]] == cb[have])
    pa, pb = np.where(have, pos_a, 0), np.where(have, pos_b, 0)
    if n_h == 0:
        z = np.full(len(ca), -1, np.int16)
        return {"code": np.full(len(ca), UNHASHED, np.int8), "p": z, "d": z, "pm": z, "dm": z}

    def dist(x, y):
        return np.bitwise_count(x ^ y).astype(np.int16)
    p, d = dist(H["phash"][pa], H["phash"][pb]), dist(H["dhash"][pa], H["dhash"][pb])
    pm, dm = dist(H["phash"][pa], H["phash_m"][pb]), dist(H["dhash"][pa], H["dhash_m"][pb])
    flat = (H["luma_std"][pa] < dedup.FLAT_LUMA_STD) | (H["luma_std"][pb] < dedup.FLAT_LUMA_STD)
    col_a, col_b = H["colour"][pa], H["colour"][pb]
    hi_c, lo_c = np.maximum(col_a, col_b), np.minimum(col_a, col_b)
    colour = (hi_c >= dedup.COLOUR_MIN) & (lo_c < dedup.COLOUR_RATIO * hi_c)
    strict = (p <= STRICT[0]) & (d <= STRICT[1])
    strict_m = (pm <= STRICT[0]) & (dm <= STRICT[1])
    window = ((p <= WINDOW) & (d <= WINDOW)) | ((pm <= WINDOW) & (dm <= WINDOW))
    code = np.full(len(ca), FAR, np.int8)
    code[window] = WIN
    code[strict_m] = MIRROR
    code[strict] = HASH
    code[colour] = COLOUR
    code[flat] = FLAT
    code[~have] = UNHASHED
    for x in (p, d, pm, dm):
        x[~have] = -1
    return {"code": code, "p": p, "d": d, "pm": pm, "dm": dm}


def exact_pairs(store) -> list[tuple[int, int]]:
    """(a, b) of the saved photos whose files have the same sha256: the lowest number of each such set
    with every other one (a star, enough to join them into one group)."""
    from store import DONE
    by: dict[str, list[int]] = {}
    with store.lock:
        cur = store.db.execute("SELECT cid, sha256 FROM status WHERE state=? AND sha256 IS NOT NULL AND sha256<>''",
                               (DONE,))
        while rows := cur.fetchmany(50000):
            for cid, sha in rows:
                by.setdefault(sha, []).append(cid)
    out = []
    for group in by.values():
        if len(group) > 1:
            group.sort()
            out += [(group[0], b) for b in group[1:]]
    return out


def feature_pairs(store, base: Path, pairs: list[tuple[int, int]], threads: int = FEATURE_THREADS,
                  log=print) -> list[tuple | None]:
    """The importer's feature test, (inliers, cover, centre ncc), for each pair of photos; None when a
    file is missing or cannot be read. The paths are asked from the store at test time."""
    import importer

    def one(pair):
        files = []
        for cid in pair:
            st = store.status(cid)
            if not st or not st.get("path"):
                return None
            files.append(base / st["path"])
        try:
            return importer.feature_test(*files)
        except Exception:  # noqa: BLE001 - an unreadable file: no verdict
            return None

    out: list = []
    t0 = time.time()
    with ThreadPoolExecutor(max(1, threads)) as pool:
        for i, r in enumerate(pool.map(one, pairs), 1):
            out.append(r)
            if i % 5000 == 0:
                log(f"  feature test: {i:,} of {len(pairs):,} pairs ({i / (time.time() - t0):.0f}/s)")
    return out


def components(edges) -> list[list[int]]:
    """The connected components (cids, sorted) of the pairs, by union-find; the largest first."""
    parent: dict[int, int] = {}

    def root(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[int]] = {}
    for x in list(parent):
        groups.setdefault(root(x), []).append(x)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: (-len(g), g[0]))


def member_info(store, cid: int) -> dict:
    """What the report and the keeper need of one photo: path, scope, category, mode, size, bytes,
    title, year and whether watermarks were painted out of it."""
    st = store.status(cid) or {}
    rec = store.record(cid) or {}
    pv = rec.get("record") or {}
    wm = (rec.get("processing") or {}).get("watermarks") or {}
    return {"cid": cid, "path": st.get("path"), "scope": st.get("scope"), "category": st.get("category"),
            "mode": st.get("mode"), "width": st.get("width") or 0, "height": st.get("height") or 0,
            "bytes": st.get("bytes"), "title": pv.get("title"), "year": pv.get("y") or pv.get("year"),
            "painted": bool(wm.get("painted"))}


def choose_keeper(members: list[int], info: dict, scores: dict) -> tuple[int | None, str]:
    """The copy to keep: dedup.rank_members (training bucket, rendered score, guard) over the members
    that could be scored; a member with painted watermarks steps aside when one without exists; a tie
    in score goes to the larger picture, then the lower number. (None, why) when fewer than two
    members could be scored."""
    ok = [c for c in members if "error" not in scores[c]]
    if len(ok) < 2:
        return None, "fewer than two copies could be scored"
    clean = [c for c in ok if not info[c]["painted"]]
    pool = clean if clean and len(clean) < len(ok) else ok

    def pixels(c):
        return info[c]["width"] * info[c]["height"]
    items = {c: SimpleNamespace(data={"score_info": scores[c]}, role=1, captioned=True, mtime=(-pixels(c), c),
                                root_idx=0, rel=info[c]["path"] or "") for c in pool}
    top, reason = dedup.rank_members(pool, items)
    if reason.startswith("equal quality"):
        ties = [c for c in pool if scores[c]["tier"] == scores[top]["tier"]
                and abs(scores[c]["score"] - scores[top]["score"]) < dedup.TIE and c != top]
        reason = ("equal quality; more pixels" if all(pixels(top) > pixels(c) for c in ties)
                  else "equal quality; lower number")
    if pool is clean:
        reason = f"no painted watermarks; {reason}" if len(pool) > 1 else "no painted watermarks"
    return top, reason


def build_groups(edges: dict, info: dict, scores: dict, targets: set[int] | None = None) -> list[dict]:
    """The groups of copies: the connected components of the copy pairs ({(a, b): {"kind", ...}}), each
    with its keeper (choose_keeper on info and scores) and its members; a member is a "copy" (it would
    move) when it is not the keeper, could be scored and is in the targets (None: all)."""
    comps = components(edges)
    if targets is not None:
        comps = [g for g in comps if any(c in targets for c in g)]
    adj: dict[int, list] = {}
    for (a, b), e in edges.items():
        adj.setdefault(a, []).append((a, b, e))
    groups = []
    for n, g in enumerate(comps, 1):
        keeper, reason = choose_keeper(g, info, scores)
        g_edges = [{**e, "a": a, "b": b} for c in g for (a, b, e) in adj.get(c, [])]
        rule = max((e["kind"] for e in g_edges), key=KINDS.index)
        ms = []
        for c in g:
            m = {k_: v for k_, v in info[c].items() if k_ not in ("copy", "stays", "in_targets")}
            m.update({k_: scores[c].get(k_) for k_ in ("tier", "score", "error") if k_ in scores[c]})
            m["copy"] = keeper is not None and c != keeper and "error" not in scores[c]
            if targets is not None:
                m["in_targets"] = c in targets
                if m["copy"] and c not in targets:
                    m["copy"], m["stays"] = False, "outside the targets"
            ms.append(m)
        groups.append({"group": n, "rule": rule, "keeper": keeper, "reason": reason, "size": len(g),
                       "scopes": len({m["scope"] for m in ms}), "categories": len({m["category"] for m in ms}),
                       "modes": len({m["mode"] for m in ms}), "members": ms, "edges": g_edges})
    return groups


def groups_from_report(path: Path, copy_kinds: tuple[str, ...] = SURE_KINDS) -> list[dict]:
    """The groups of a dry run's groups file, built again from its pairs of the kinds given (the members'
    sizes, scores and paths as the report has them), so the keepers are the ones reviewed."""
    info, scores, edges = {}, {}, {}
    targeted = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            g = json.loads(line)
            for m in g["members"]:
                info[m["cid"]] = m
                scores[m["cid"]] = {"error": m["error"]} if m.get("error") else {"tier": m.get("tier"),
                                                                                   "score": m.get("score")}
                targeted = targeted or "in_targets" in m
            for e in g["edges"]:
                if e["kind"] in copy_kinds:
                    edges[(e["a"], e["b"])] = {k_: v for k_, v in e.items() if k_ not in ("a", "b")}
    targets = {c for c, m in info.items() if m.get("in_targets")} if targeted else None
    return build_groups(edges, info, scores, targets)


def _band(c: float) -> str:
    """The 0.01 cosine band of a pair ("0.95": 0.95 to 0.96; "0.99" includes 1.0)."""
    return f"{min(99, int(np.floor(c * 100 + 1e-6))) / 100:.2f}"


def band_counts(cos: np.ndarray, code: np.ndarray) -> dict[str, Counter]:
    """{band: Counter of code names} of the candidate pairs."""
    b = np.minimum(99, np.floor(cos.astype(np.float64) * 100 + 1e-6).astype(np.int64))
    keys, counts = np.unique(b * 16 + code.astype(np.int64), return_counts=True)
    out: dict[str, Counter] = {}
    for key, n in zip(keys.tolist(), counts.tolist()):
        out.setdefault(f"{key // 16 / 100:.2f}", Counter())[CODE_NAMES[key % 16]] += n
    return out


def write_review(base: Path, groups: list[dict], window: list[dict], per_rule: int | None, seed: int = 0,
                 log=print) -> Path:
    """A flat folder next to the base of hard links (copies when a link fails) for a look by eye:
    g<group>_<rule>_<K|D>_<cid> for the groups (K the keeper, D a copy that would go), at most per_rule
    random groups of each rule (None: all), and w<n>_<pass|fail|none>_<inliers>_<ncc>_<cid> for as many
    random pairs of the feature window. The folder of an earlier run is replaced; a folder of that
    name that the dataset tools did not make is refused."""
    import random
    folder = base.parent / f"{base.name}{REVIEW_SUFFIX}"
    if folder.exists():
        if not (folder / dedup.BACKUP_MARK).is_file():
            raise SystemExit(f"{folder} exists and was not made by deduplicate; rename it or move it away")
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    (folder / dedup.BACKUP_MARK).write_text(json.dumps({"tool": "pastvu deduplicate --review",
                                                        "made": time.strftime("%Y-%m-%d %H:%M:%S")}), "utf-8")
    rng = random.Random(seed)

    def link(rel, name):
        src = base / rel
        dst = folder / (name + src.suffix.lower())
        try:
            os.link(src, dst)
        except OSError:
            try:
                shutil.copy2(src, dst)
            except OSError:
                return 0
        return 1

    def sample(rows):
        return rows if per_rule is None or len(rows) <= per_rule else rng.sample(rows, per_rule)

    n = 0
    for rule in KINDS:
        for g in sorted(sample([g for g in groups if g["rule"] == rule]), key=lambda g: g["group"]):
            for m in g["members"]:
                if m["path"]:
                    role = "K" if m["cid"] == g["keeper"] else "D" if m.get("copy") else "U"
                    n += link(m["path"], f"g{g['group']:05d}_{rule}_{role}_{m['cid']}")
    for i, w in enumerate(sample(window)):
        verdict = "none" if w["inliers"] is None else "pass" if w["passed"] else "fail"
        ncc = "x" if w["ncc"] is None else f"{w['ncc']:.2f}"
        for cid, rel in ((w["a"], w["path_a"]), (w["b"], w["path_b"])):
            if rel:
                n += link(rel, f"w{i:05d}_{verdict}_{w['inliers'] or 0}_{ncc}_{cid}")
    log(f"  review: {n:,} files in {folder}")
    return folder


def find(store, base: Path, model: str, patches: int, targets: set[int] | None = None, min_cos: float = MIN_COS,
         k: int = NEIGHBOURS, features: bool = True, threads: int = THREADS, review: int | None | bool = False,
         copy_kinds: tuple[str, ...] = SURE_KINDS, log=print) -> dict:
    """Find the copies (the module text), write the report files to <base>/_reports, move nothing.
    targets: the cids whose copies count (None: the whole base); every saved photo is a candidate.
    features: run the feature test of the window pairs (their file is written either way).
    copy_kinds: the kinds of pairs that join photos into groups; "features" is not among them by
    default, because the test passes other frames of one scene (PLAN.md section 24, stage 2).
    review: False for none, None for every group, N for N random groups per rule.
    -> the summary (also written as JSON; "reports"["groups"] is what apply_report moves)."""
    t0 = time.time()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    reports = base / "_reports"
    reports.mkdir(parents=True, exist_ok=True)

    cids, V = load_embeddings(store, model, patches)
    log(f"  {len(cids):,} embeddings ({model}, {patches} patches) read in {time.time() - t0:.0f} s")
    t1 = time.time()
    ia, ib, cos = neighbours(V, k, min_cos)
    ca, cb = cids[ia], cids[ib]
    log(f"  {len(ca):,} candidate pairs at cosine {min_cos:g} or more ({k} nearest each) in {time.time() - t1:.0f} s")
    H = store.hash_rows(HASH_VERSION)
    cls = classify_pairs(ca, cb, H)
    code = cls["code"]

    # byte-identical files, with or without an embedding nearby
    pos = {int(c): i for i, c in enumerate(cids)}
    exact = exact_pairs(store)
    exact_cos = {}
    for a, b in exact:
        if a in pos and b in pos:
            exact_cos[(a, b)] = float(V[pos[a]].astype(np.float32) @ V[pos[b]].astype(np.float32))
    del V

    # the feature window
    win_idx = np.nonzero(code == WIN)[0]
    exact_set = set(exact)
    win_idx = np.array([i for i in win_idx if (int(ca[i]), int(cb[i])) not in exact_set], np.int64)
    win_pairs = [(int(ca[i]), int(cb[i])) for i in win_idx]
    if features and win_pairs:
        log(f"  feature test of {len(win_pairs):,} pairs in the hash window ({threads} threads)")
        t2 = time.time()
        tests = feature_pairs(store, base, win_pairs, threads, log)
        log(f"  feature test done in {(time.time() - t2) / 60:.1f} min")
    else:
        tests = [None] * len(win_pairs)

    # copy pairs
    edges: dict[tuple[int, int], dict] = {}
    for a, b in exact:
        edges[(a, b)] = {"kind": "exact", "cos": exact_cos.get((a, b))}
    for kind_code, kind in ((HASH, "hash"), (MIRROR, "mirror")):
        for i in np.nonzero(code == kind_code)[0]:
            key = (int(ca[i]), int(cb[i]))
            if key not in edges:
                edges[key] = {"kind": kind, "cos": round(float(cos[i]), 4), "p": int(cls["p"][i]),
                              "d": int(cls["d"][i]), "pm": int(cls["pm"][i]), "dm": int(cls["dm"][i])}
    import importer
    window_rows = []
    for i, (a, b), t in zip(win_idx, win_pairs, tests):
        passed = t is not None and importer.passes(t)
        row = {"a": a, "b": b, "cos": round(float(cos[i]), 4), "band": _band(float(cos[i])),
               "p": int(cls["p"][i]), "d": int(cls["d"][i]),
               "pm": int(cls["pm"][i]), "dm": int(cls["dm"][i]),
               "inliers": None if t is None else int(t[0]), "cover": None if t is None else round(float(t[1]), 3),
               "ncc": None if t is None or t[2] is None else round(float(t[2]), 3), "passed": passed}
        window_rows.append(row)
        if passed and "features" in copy_kinds:
            edges[(a, b)] = {"kind": "features", **{k_: row[k_] for k_ in
                                                   ("cos", "p", "d", "pm", "dm", "inliers", "cover", "ncc")}}

    # groups and keepers
    edges = {key: e for key, e in edges.items() if e["kind"] in copy_kinds}
    comps = components(edges)
    if targets is not None:
        comps = [g for g in comps if any(c in targets for c in g)]
    members = sorted({c for g in comps for c in g})
    log(f"  {len(edges):,} copy pairs ({', '.join(copy_kinds)}) in {len(comps):,} groups of {len(members):,} photos; "
        "scoring them")
    t3 = time.time()
    with ThreadPoolExecutor(max(1, threads)) as pool:
        info = dict(zip(members, pool.map(lambda c: member_info(store, c), members)))
        scores = dict(zip(members, pool.map(
            lambda c: dedup.score_file(str(base / info[c]["path"])) if info[c]["path"] else {"error": "no path"},
            members)))
    log(f"  scored in {(time.time() - t3) / 60:.1f} min")
    groups = build_groups(edges, info, scores, targets)

    # the summary
    bands = band_counts(cos, code)
    for w in window_rows:
        bands.setdefault(w["band"], Counter())["features pass" if w["passed"] else "features fail"] += 1
    moving = [m for g in groups for m in g["members"] if m["copy"]]
    summary = {
        "time": stamp, "model": model, "patches": patches, "min_cos": min_cos, "neighbours": k,
        "copy_kinds": list(copy_kinds),
        "photos": len(cids), "hashed": len(H["cid"]), "candidates": len(ca),
        "pairs": dict(Counter(CODE_NAMES[x] for x in code)), "exact_pairs": len(exact),
        "window_tested": sum(1 for t in tests if t is not None), "window_passed": sum(w["passed"] for w in window_rows),
        "copy_pairs": dict(Counter(e["kind"] for e in edges.values())),
        "groups": len(groups), "groups_by_rule": dict(Counter(g["rule"] for g in groups)),
        "largest_group": max((g["size"] for g in groups), default=0),
        "groups_unresolved": sum(1 for g in groups if g["keeper"] is None),
        "groups_across_scopes": sum(1 for g in groups if g["scopes"] > 1),
        "groups_across_categories": sum(1 for g in groups if g["categories"] > 1),
        "groups_across_modes": sum(1 for g in groups if g["modes"] > 1),
        "would_move": len(moving),
        "would_move_by_rule": dict(Counter(g["rule"] for g in groups for m in g["members"] if m["copy"])),
        "would_move_by_category": dict(Counter(m["category"] for m in moving).most_common()),
        "bands": {b: dict(c) for b, c in sorted(bands.items())},
        "minutes": round((time.time() - t0) / 60, 1),
    }
    paths = {"groups": reports / f"deduplicate-{stamp}-groups.jsonl",
             "window": reports / f"deduplicate-{stamp}-window.jsonl",
             "summary": reports / f"deduplicate-{stamp}-summary.json"}
    with open(paths["groups"], "w", encoding="utf-8") as f:
        for g in groups:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    paths_of = {}
    for w in window_rows:
        for side in ("a", "b"):
            if w[side] not in paths_of:
                st = store.status(w[side]) or {}
                paths_of[w[side]] = st.get("path")
        w["path_a"], w["path_b"] = paths_of[w["a"]], paths_of[w["b"]]
    with open(paths["window"], "w", encoding="utf-8") as f:
        for w in window_rows:
            f.write(json.dumps(w, ensure_ascii=False) + "\n")
    if review is not False:
        summary["review"] = str(write_review(base, groups, window_rows, review, log=log))
    summary["reports"] = {k_: str(p) for k_, p in paths.items()}
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=1), "utf-8")
    return summary


def summary_text(s: dict) -> str:
    """The summary of find() for the console."""
    lines = [f"{s['candidates']:,} candidate pairs among {s['photos']:,} photos ({s['hashed']:,} hashed); "
             f"{s['exact_pairs']:,} byte-identical pairs",
             "  cosine  " + "".join(f"{k:>10}" for k in ("hash", "mirror", "window", "far", "flat", "colour")) +
             f"{'feat pass':>11}"]
    cols = ("hash", "mirror", "window", "far", "flat", "colour")
    for b, c in s["bands"].items():
        lines.append(f"  {b:>6}  " + "".join(f"{c.get(k, 0):>10,}" for k in cols) + f"{c.get('features pass', 0):>11,}")
    lines.append(f"Feature window: {s['window_tested']:,} pairs tested, {s['window_passed']:,} passed")
    lines.append("Copy pairs: " + ", ".join(f"{v:,} {k}" for k, v in s["copy_pairs"].items()))
    lines.append(f"{s['groups']:,} groups (" + ", ".join(f"{v:,} {k}" for k, v in s["groups_by_rule"].items())
                 + f"), largest {s['largest_group']}; {s['groups_unresolved']:,} without a keeper; "
                 f"{s['groups_across_scopes']:,} across scopes, {s['groups_across_categories']:,} across categories, "
                 f"{s['groups_across_modes']:,} across colour modes")
    lines.append(f"Would move {s['would_move']:,} photos (" + ", ".join(
        f"{v:,} {k}" for k, v in s["would_move_by_rule"].items()) + ")")
    lines.append("Reports: " + ", ".join(s["reports"].values()))
    if s.get("review"):
        lines.append(f"Review folder: {s['review']}")
    return "\n".join(lines)


# --- moving the copies out, and back -----------------------------------------------------------

def dup_dir(base: Path) -> Path:
    return base.parent / f"{base.name}{DUP_SUFFIX}"


def claim_dup_dir(base: Path) -> Path:
    """<base>_duplicates, made with the dataset tools' mark (the other tools never scan it); a folder of
    that name without the mark is refused."""
    folder = dup_dir(base)
    if folder.exists() and not (folder / dedup.BACKUP_MARK).is_file():
        raise SystemExit(f"{folder} exists and was not made by deduplicate; rename it or move it away")
    folder.mkdir(parents=True, exist_ok=True)
    mark = folder / dedup.BACKUP_MARK
    if not mark.is_file():
        mark.write_text(json.dumps({"tool": "pastvu deduplicate", "made": time.strftime("%Y-%m-%d %H:%M:%S")}),
                        "utf-8")
    return folder


def apply(store, base: Path, groups: list[dict], args: dict, exporter=None, log=print) -> dict:
    """Move every copy of the groups (members with "copy") out of the base into <base>_duplicates, at
    its path in the base, with every file of its name; skip it in the store with the reason "copy of
    <keeper>"; delete its export links. One logged run (dedupe_runs, dedupe_moves) for --undo.
    A copy whose image is not where the store says (nor anywhere in its scope) is counted, not moved.
    -> counts and the run id."""
    import folders
    import layout
    folder = claim_dup_dir(base)
    todo = [(m, g) for g in groups if g["keeper"] is not None for m in g["members"] if m.get("copy")]
    run = store.dedupe_start({**args, "copies": len(todo)})
    chunks: dict[str, object] = {}
    scans: dict = {}
    touched: set[str] = set()
    moved = missing = failed = 0
    t0 = time.time()
    for seq, (m, g) in enumerate(todo, 1):
        cid = m["cid"]
        st = store.status(cid) or {}
        sc = store.scope(st.get("scope") or "") if st.get("state") == "done" else None
        if sc is None:
            missing += 1
            continue
        rel = folders.locate(base, sc, cid, st.get("path"), scans)
        if rel is None:
            missing += 1
            continue
        dst = folder / rel
        try:
            folders.move_pair(base / rel, dst)
        except OSError as e:
            log(f"  {cid}: not moved ({e})")
            failed += 1
            continue
        if sc["name"] not in chunks:
            chunks[sc["name"]] = folders.ScopeChunks(store, base, sc, log, defer=True)
        scope_rel = layout.to_scope(sc, rel)
        if scope_rel is not None:
            chunks[sc["name"]].remove(scope_rel)
        if exporter is not None:
            exporter.remove(cid)
        store.dedupe_move(run, seq, cid, g["keeper"], g["rule"], sc["name"], rel, rel, f"copy of {g['keeper']}")
        touched.add(sc["name"])
        moved += 1
        if moved % 1000 == 0:
            log(f"  moved {moved:,} of {len(todo):,} ({time.time() - t0:.0f} s)")
    for name, ch in chunks.items():
        ch.settle()
    for name in touched:
        sc = store.scope(name)
        folders.prune_scope(base, sc)
        store.recount(name)
    return {"run": run, "moved": moved, "missing": missing, "failed": failed, "copies": len(todo),
            "folder": str(folder)}


def undo(store, base: Path, exporter=None, log=print) -> dict:
    """Put back the copies of the newest deduplicate run not undone: each image still in
    <base>_duplicates goes back into its category (numbered folders as the scope has them), done
    again, linked into the export again; one deleted from the folder by hand stays skipped.
    -> counts."""
    import folders
    import layout
    last = store.dedupe_last()
    if last is None:
        raise SystemExit(f"{base}: no deduplicate run to undo")
    folder = dup_dir(base)
    rows = store.dedupe_moves(last["id"])
    by_scope: dict[str, list] = {}
    for r in rows:
        by_scope.setdefault(r[4], []).append(r)
    back = gone = stuck = 0
    for name, items in by_scope.items():
        sc = store.scope(name)
        if sc is None:
            log(f"  scope {name} is gone; its {len(items):,} copies stay in {folder}")
            stuck += len(items)
            continue
        chunks = folders.ScopeChunks(store, base, sc, log, defer=True)
        for _seq, cid, keeper, _rule, _scope, src, dst_name in items:
            f = folder / dst_name
            if not f.is_file():
                gone += 1
                continue
            st = store.status(cid) or {}
            cat = layout.with_mode(sc, st.get("mode"), st.get("category") or "")
            place = chunks.place(cat, Path(src).name)
            rel = chunks.rel(place)
            try:
                folders.move_pair(f, base / rel)
            except OSError as e:
                chunks.abandon(place)
                log(f"  {cid}: not moved back ({e}); it stays in {folder}")
                stuck += 1
                continue
            rec = folders.record_with_path(store, base, cid, rel, ("deduplicate", {"time": time.time(),
                                                                                   "undo": last["id"],
                                                                                   "copy_of": keeper}))
            store.dedupe_restore(cid, rel, rec)
            chunks.done(place)
            if exporter is not None:
                exporter.add(cid, rel, st.get("mode"), st.get("category") or "")
            back += 1
        chunks.settle()
        store.recount(name)
    store.dedupe_undone(last["id"])
    if folder.is_dir():
        folders.prune_empty(folder)
    return {"run": last["id"], "back": back, "gone": gone, "stuck": stuck, "folder": str(folder)}
