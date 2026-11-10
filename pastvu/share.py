"""share: a pick (cherrypick's output folder) as one zip that a partner rebuilds from
PastVu with `run` (replica.py), instead of the images themselves.

The zip holds:
- <name>/pastvu.db: a base store with the picked photos only. Their PastVu
  records (without the reclassify and review history, which name local
  folders), boxes, NIQE, hashes and the region tree; a pending status row each,
  with the category and mode; and the replica table: the photo's path in the
  pick, the text of its sidecars there (its captions), the sha256 of the
  original download and of our saved file. --embeddings adds the classifier's
  embeddings (about 2.3 KB a photo; nothing in the replica run reads them).
- pastvu.conf: [paths] base and the download threads, to put next to run.bat.
- README_RU.txt: the instructions, in Russian.

The photos are the ones the pick folder holds now: a picked image deleted from
the folder by hand is left out, and so is a photo of another base.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import subprocess
import time
import zipfile
from pathlib import Path

import layout
import replica as rp
from store import PENDING, Store

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "share_templates"
CONF_TEMPLATE = TEMPLATES / "conf.template"      # pastvu.conf in the zip (that name is git-ignored)
README_TEMPLATE = TEMPLATES / "README_RU.txt"
REPO_URL = "https://github.com/aoleg/dataset_tools"
DROP_KEYS = ("reclassified", "reviewed")          # record keys that hold local folders and our own history
LOCAL_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")
COPY_TABLES = ("photos", "boxes", "niqe", "hashes")


def _drop_paths(obj):
    """obj without the values that are absolute local paths. -> (obj, dropped)"""
    if isinstance(obj, dict):
        out, n = {}, 0
        for k, v in obj.items():
            if isinstance(v, str) and LOCAL_PATH.match(v):
                n += 1
                continue
            v, m = _drop_paths(v)
            out[k], n = v, n + m
        return out, n
    if isinstance(obj, list):
        out, n = [], 0
        for v in obj:
            if isinstance(v, str) and LOCAL_PATH.match(v):
                n += 1
                continue
            v, m = _drop_paths(v)
            out.append(v)
            n += m
        return out, n
    return obj, 0


def clean_record(rec: dict) -> tuple[dict, int]:
    """The record as shared: without DROP_KEYS and without local paths. -> (record, values dropped)"""
    rec = {k: v for k, v in rec.items() if k not in DROP_KEYS}
    return _drop_paths(rec)


def clean_criteria(criteria: dict) -> dict:
    """cherrypick's criteria without the base and samples folders."""
    c = {k: v for k, v in (criteria or {}).items() if k not in ("base", "samples")}
    return _drop_paths(c)[0]


def manifest_files(pick: Path, base: Path) -> tuple[dict[int, str], dict]:
    """The pick's files of this base (cid -> path in the pick) and its criteria."""
    import cherrypick as cp
    mf = pick / cp.MANIFEST
    if not mf.is_file():
        raise SystemExit(f"{pick} has no {cp.MANIFEST}: give the output folder of a cherrypick")
    try:
        manifest = json.loads(mf.read_text(encoding="utf-8"))
    except ValueError as e:
        raise SystemExit(f"{mf}: {e}")
    want = str(Path(base).resolve()).lower()
    for key, entry in (manifest.get("bases") or {}).items():
        if str(Path(key).resolve()).lower() == want:
            return {int(k): v for k, v in (entry.get("files") or {}).items()}, entry.get("criteria") or {}
    raise SystemExit(f"{pick} holds no pick of {base}; give the base it was picked from with --base")


def repo_commit() -> str:
    """The repository's commit (short), or "main" when git cannot tell; and whether the
    pastvu folder has changes not committed."""
    try:
        r = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                           timeout=20)
        return r.stdout.strip() or "main"
    except (OSError, subprocess.SubprocessError):
        return "main"


def uncommitted() -> bool:
    try:
        r = subprocess.run(["git", "-C", str(HERE), "status", "--porcelain", "--", "."], capture_output=True,
                           text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False
    return any(line[3:].endswith(".py") or line[3:].endswith(".bat") for line in r.stdout.splitlines())


def _columns(db, table: str, schema: str = "main") -> list[str]:
    return [r[1] for r in db.execute(f"PRAGMA {schema}.table_info({table})")]


def _copy(db, table: str, where: str = "cid IN (SELECT cid FROM temp.pick)") -> int:
    cols = [c for c in _columns(db, table) if c in set(_columns(db, table, "src"))]
    names = ", ".join(cols)
    cur = db.execute(f"INSERT INTO main.{table} ({names}) SELECT {names} FROM src.{table} WHERE {where}")
    return cur.rowcount


def build_db(src_db: Path, pick: Path, files: dict[int, str], criteria: dict, path: Path, name: str,
             embeddings: bool = False, log=print) -> dict:
    """pastvu.db of the replica at path. -> counts"""
    path.unlink(missing_ok=True)
    Store(path).close()                                  # the store's own schema
    db = sqlite3.connect(f"file:{path.as_posix()}", uri=True, isolation_level=None)
    try:
        rp.ensure_schema(db)
        db.execute("ATTACH DATABASE ? AS src", (f"file:{Path(src_db).as_posix()}?mode=ro",))
        db.execute("CREATE TEMP TABLE pick (cid INTEGER PRIMARY KEY, path TEXT)")
        db.execute("BEGIN")
        db.executemany("INSERT INTO temp.pick VALUES (?, ?)", sorted(files.items()))
        missing = db.execute("DELETE FROM temp.pick WHERE cid NOT IN (SELECT cid FROM src.records) "
                             "OR cid NOT IN (SELECT cid FROM src.status)").rowcount
        files = {cid: rel for cid, rel in db.execute("SELECT cid, path FROM temp.pick")}
        res = {"photos": len(files)}
        for table in COPY_TABLES + (("embeddings",) if embeddings else ()):
            res[table] = _copy(db, table)
        res["regions"] = _copy(db, "regions", "1")
        now = time.time()
        res["status"] = db.execute(
            "INSERT INTO main.status (cid, state, reason, attempts, scope, category, mode, updated) "
            "SELECT p.cid, ?, '', 0, ?, s.category, s.mode, ? FROM temp.pick p JOIN src.status s ON s.cid=p.cid",
            (PENDING, rp.SCOPE, now)).rowcount
        dropped = shared = 0
        src_sha = dict(db.execute("SELECT s.cid, s.sha256 FROM src.status s JOIN temp.pick p ON p.cid=s.cid"))
        t0 = time.time()
        rows = db.execute("SELECT r.cid, r.json FROM src.records r JOIN temp.pick p ON p.cid=r.cid").fetchall()
        recs, reps = [], []
        sidecar_files = sidecar_bytes = unreadable = 0
        for i, (cid, text) in enumerate(rows, 1):
            rec, n = clean_record(json.loads(text))
            dropped += n
            recs.append((cid, json.dumps(rec, ensure_ascii=False, separators=(",", ":"))))
            img = pick / files[cid]
            cars = {}
            for s in layout.export_sidecars(img):
                try:
                    cars[s.name[len(img.stem):]] = s.read_bytes().decode("utf-8")
                    sidecar_files += 1
                    sidecar_bytes += s.stat().st_size
                except (OSError, UnicodeDecodeError) as e:
                    unreadable += 1
                    if unreadable <= 5:
                        log(f"  {s}: not shared ({type(e).__name__})")
            reps.append((cid, files[cid], json.dumps(cars, ensure_ascii=False) if cars else None,
                         (rec.get("download") or {}).get("sha256"), src_sha.get(cid), None, now))
            shared += 1
            if i % 20000 == 0:
                log(f"  {i:,} of {len(rows):,} records and sidecars ({time.time() - t0:.0f} s)")
        db.executemany("INSERT INTO main.records (cid, json) VALUES (?, ?)", recs)
        db.executemany("INSERT INTO main.replica VALUES (?, ?, ?, ?, ?, ?, ?)", reps)
        meta = {"kind": rp.KIND, "replica_format": rp.FORMAT, "replica_name": name, "replica_made": now,
                "replica_photos": shared, "replica_criteria": clean_criteria(criteria)}
        db.executemany("INSERT OR REPLACE INTO main.meta VALUES (?, ?)",
                       [(k, json.dumps(v, ensure_ascii=False)) for k, v in meta.items()])
        db.execute("COMMIT")
        db.execute("DETACH DATABASE src")
        db.execute("VACUUM")
        res.update(records=len(recs), dropped_values=dropped, sidecars=sidecar_files, sidecar_bytes=sidecar_bytes,
                   unreadable=unreadable, missing_records=missing)
        return res
    finally:
        db.close()


def fill(template: Path, values: dict) -> str:
    text = template.read_text(encoding="utf-8")
    for k, v in values.items():
        text = text.replace("{" + k + "}", str(v))
    return text


def build(store, base: Path, pick: Path, out: Path | None = None, name: str | None = None,
          embeddings: bool = False, log=print) -> dict:
    """The share zip of a pick. -> counts, with "zip" and "bytes" """
    pick = Path(pick).resolve()
    files, criteria = manifest_files(pick, base)
    name = name or pick.name
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$", name):
        raise SystemExit(f"{name!r}: give --name in Latin letters, digits, _ . and - (the partner's folder name)")
    out = Path(out).resolve() if out else pick.parent / f"{name}_share.zip"
    if out.suffix.lower() != ".zip":
        out = out.with_suffix(".zip")
    if out.is_relative_to(pick) or out.is_relative_to(Path(base).resolve()):
        raise SystemExit(f"{out}: put the zip outside the pick and the base")
    present = {cid: rel for cid, rel in files.items() if (pick / rel).is_file()}
    gone = len(files) - len(present)
    log(f"{pick}: {len(present):,} photos of {base}" + (f" ({gone:,} listed by the pick but no longer in the "
                                                       f"folder: left out)" if gone else ""))
    if not present:
        raise SystemExit("nothing to share")
    work = out.parent / f".{out.stem}.tmp"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        dbp = work / "pastvu.db"
        t0 = time.time()
        res = build_db(store.path, pick, present, criteria, dbp, name, embeddings, log)
        log(f"  database: {dbp.stat().st_size / 2**20:,.0f} MB in {time.time() - t0:.0f} s")
        size =sum((pick / rel).stat().st_size for rel in present.values())
        commit = repo_commit()
        values = {"name": name, "photos": f"{res['records']:,}".replace(",", " "),
                  "gb": f"{size / 1e9:.0f}", "commit": commit, "repo": REPO_URL, "format": rp.FORMAT,
                  "date": time.strftime("%Y-%m-%d")}
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp_zip = work / out.name
        with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            z.write(dbp, f"{name}/pastvu.db")
            z.writestr("pastvu.conf", fill(CONF_TEMPLATE, values).encode("utf-8"))
            z.writestr("README_RU.txt", fill(README_TEMPLATE, values).encode("utf-8"))
        shutil.move(str(tmp_zip), out)
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        res.update(zip=str(out), bytes=out.stat().st_size, pick_bytes=size, commit=commit, sha256=digest,
                   left_out=gone, dirty=uncommitted())
        return res
    finally:
        shutil.rmtree(work, ignore_errors=True)
