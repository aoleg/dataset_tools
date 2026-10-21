"""The SQLite store <base>/pastvu.db: one per base folder, shared by every
scope in it.

- regions: PastVu's region tree.
- scopes: a named download (regions, years or a decade, split size).
- shards, photos, members: the index; photos are global, members says which
  scopes list a photo.
- status: one row per photo, global, so a photo saved under one scope is not
  downloaded again for another. A photo is pending, writing (its path is
  claimed and its files are being written), done, skipped or failed.
- records, embeddings: the PastVu record with what was done, and the
  classifier's embedding of the saved pixels, per photo.
- counters: per-scope totals for the reports.
- reclass_runs, reclass_moves: what reclassify moved, for --undo.
- captions, caption_log, caption_fails, caption_runs: autocaption. captions
  holds each photo's current VLM caption and the caption it had before the
  first one (the original); caption_log holds, per run, the text a file had
  before and after, which --undo writes back.

A photo becomes done in one transaction with its record, its embedding and
its counters, so a killed process loses nothing that was finished.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

DB_NAME = "pastvu.db"
SCHEMA_VERSION = 2
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS regions (
    cid INTEGER PRIMARY KEY, parent INTEGER, parents TEXT, title_en TEXT, title_local TEXT,
    phc INTEGER, pac INTEGER, child_len INTEGER, pulled REAL);
CREATE INDEX IF NOT EXISTS regions_parent ON regions(parent);
CREATE TABLE IF NOT EXISTS scopes (
    name TEXT PRIMARY KEY, regions TEXT, years TEXT, decade INTEGER, split INTEGER, width INTEGER,
    created REAL, indexed REAL);
CREATE TABLE IF NOT EXISTS shards (
    key TEXT PRIMARY KEY, scope TEXT, region INTEGER, filter TEXT, count INTEGER, pages_done INTEGER DEFAULT 0,
    rows INTEGER DEFAULT 0, done INTEGER DEFAULT 0, updated REAL);
CREATE TABLE IF NOT EXISTS photos (
    cid INTEGER PRIMARY KEY, file TEXT, title TEXT, year INTEGER, year2 INTEGER, dir TEXT, rs TEXT,
    first_seen REAL, last_seen REAL);
CREATE TABLE IF NOT EXISTS members (scope TEXT, cid INTEGER, PRIMARY KEY (scope, cid)) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS members_cid ON members(cid);
CREATE TABLE IF NOT EXISTS status (
    cid INTEGER PRIMARY KEY, state TEXT, reason TEXT, attempts INTEGER DEFAULT 0, scope TEXT, category TEXT,
    path TEXT, sha256 TEXT, bytes INTEGER, width INTEGER, height INTEGER, changed TEXT, updated REAL);
CREATE INDEX IF NOT EXISTS status_state ON status(state);
CREATE INDEX IF NOT EXISTS status_scope ON status(scope);
CREATE TABLE IF NOT EXISTS records (cid INTEGER PRIMARY KEY, json TEXT);
CREATE TABLE IF NOT EXISTS embeddings (cid INTEGER PRIMARY KEY, model TEXT, patches INTEGER, vec BLOB);
CREATE TABLE IF NOT EXISTS counters (scope TEXT, name TEXT, value REAL DEFAULT 0, PRIMARY KEY (scope, name));
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT, started REAL, ended REAL, args TEXT, done INTEGER,
    skipped INTEGER, failed INTEGER);
CREATE TABLE IF NOT EXISTS reclass_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT, started REAL, samples TEXT, moved INTEGER, undone REAL);
CREATE TABLE IF NOT EXISTS reclass_moves (
    run INTEGER, seq INTEGER, cid INTEGER, src TEXT, dst TEXT, cat_from TEXT, cat_to TEXT, PRIMARY KEY (run, seq));
CREATE TABLE IF NOT EXISTS captions (
    cid INTEGER PRIMARY KEY, original TEXT, caption TEXT, run INTEGER, model TEXT, prompt_sha TEXT, time REAL);
CREATE TABLE IF NOT EXISTS caption_log (
    run INTEGER, cid INTEGER, before TEXT, after TEXT, PRIMARY KEY (run, cid)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS caption_fails (cid INTEGER PRIMARY KEY, attempts INTEGER, error TEXT, time REAL);
CREATE TABLE IF NOT EXISTS regroup_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, rule TEXT, target TEXT, started REAL, moved INTEGER DEFAULT 0,
    removed TEXT, created TEXT, undone REAL);
CREATE TABLE IF NOT EXISTS regroup_moves (
    run INTEGER, seq INTEGER, cid INTEGER, src_scope TEXT, src_path TEXT, dst_scope TEXT, dst_path TEXT, kind TEXT,
    new_member INTEGER, PRIMARY KEY (run, seq)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS caption_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT, target TEXT, model TEXT, prompt_sha TEXT, started REAL,
    ended REAL, captioned INTEGER DEFAULT 0, failed INTEGER DEFAULT 0, undone REAL);
"""

PENDING, WRITING, DONE, SKIPPED, FAILED = "pending", "writing", "done", "skipped", "failed"
REGION_COLS = "cid, parent, parents, title_en, title_local, phc, pac, child_len"
STATUS_COLS = ("cid", "state", "reason", "attempts", "scope", "category", "path", "sha256", "bytes", "width", "height",
               "changed")


class Store:
    def __init__(self, path: Path | str):
        self.path = Path(path) if str(path) != ":memory:" else path
        if isinstance(self.path, Path):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        old = self.db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='status'").fetchone()
        if old:
            cols = {r[1] for r in self.db.execute("PRAGMA table_info(status)")}
            if "scope" not in cols:
                raise SystemExit(f"{self.path} is a store of the first version of this tool (one store per output "
                                 f"folder); it cannot be opened as a base folder store.")
        self.db.executescript(SCHEMA)
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(scopes)")}
        if "label" not in cols:
            self.db.execute("ALTER TABLE scopes ADD COLUMN label TEXT")
        if "folder" not in cols:
            self.db.execute("ALTER TABLE scopes ADD COLUMN folder TEXT")
        self.set_meta("schema", SCHEMA_VERSION)

    def close(self) -> None:
        with self.lock:
            self.db.close()

    def tx(self, fn):
        """Run fn(db) in one transaction."""
        with self.lock:
            self.db.execute("BEGIN")
            try:
                out = fn(self.db)
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
            self.db.execute("COMMIT")
            return out

    # --- meta ----------------------------------------------------------------------

    def get_meta(self, key: str, default=None):
        with self.lock:
            row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_meta(self, key: str, value) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value, ensure_ascii=False)))

    # --- regions -------------------------------------------------------------------

    def put_regions(self, regions: list[dict]) -> None:
        now = time.time()
        rows = [(r["cid"], (r.get("parents") or [None])[-1], json.dumps(r.get("parents") or []), r.get("title_en", ""),
                 r.get("title_local", ""), r.get("phc", 0), r.get("pac", 0), r.get("childLen", 0), now) for r in regions]
        self.tx(lambda db: db.executemany("INSERT OR REPLACE INTO regions VALUES (?,?,?,?,?,?,?,?,?)", rows))

    @staticmethod
    def _region(row) -> dict:
        return {"cid": row[0], "parent": row[1], "parents": json.loads(row[2] or "[]"), "title_en": row[3],
                "title_local": row[4], "phc": row[5], "pac": row[6], "child_len": row[7]}

    def region(self, cid: int) -> dict | None:
        with self.lock:
            row = self.db.execute(f"SELECT {REGION_COLS} FROM regions WHERE cid=?", (cid,)).fetchone()
        return self._region(row) if row else None

    def children(self, cid: int) -> list[dict]:
        with self.lock:
            rows = self.db.execute(f"SELECT {REGION_COLS} FROM regions WHERE parent=? ORDER BY phc DESC", (cid,)).fetchall()
        return [self._region(r) for r in rows]

    def all_regions(self) -> list[dict]:
        with self.lock:
            rows = self.db.execute(f"SELECT {REGION_COLS} FROM regions").fetchall()
        return [self._region(r) for r in rows]

    def regions_count(self) -> int:
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM regions").fetchone()[0]

    def find_regions(self, text: str) -> list[dict]:
        like = f"%{text}%"
        with self.lock:
            rows = self.db.execute(f"SELECT {REGION_COLS} FROM regions WHERE title_en LIKE ? OR title_local LIKE ? "
                                   "ORDER BY phc DESC LIMIT 40", (like, like)).fetchall()
        return [self._region(r) for r in rows]

    # --- scopes --------------------------------------------------------------------

    def scope(self, name: str) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT name, regions, years, decade, split, width, created, indexed, label, folder "
                                  "FROM scopes WHERE name=?", (name,)).fetchone()
        if not row:
            return None
        return {"name": row[0], "regions": json.loads(row[1]), "years": tuple(json.loads(row[2])), "decade": row[3],
                "split": row[4], "width": row[5], "created": row[6], "indexed": row[7],
                "label": row[8] or row[0].rsplit("_", 1)[0], "folder": row[9] or row[0]}

    def layout(self) -> str:
        """The folder tree of the base: set when its first scope is made; bases
        from before the option are flat."""
        return self.get_meta("layout") or "flat"

    def add_scope(self, name: str, regions: list[int], years: tuple[int, int], decade: int | None, split: int,
                  label: str | None = None, width: int = 0, folder: str | None = None) -> dict:
        """A new scope. Its folder follows the base's layout unless given; two
        scopes never share a folder or nest one inside the other."""
        import layout as lay
        if label is None:
            tail = "_" + lay.years_label(years, decade)
            label = name[:-len(tail)] if name.endswith(tail) and len(name) > len(tail) else name
        if folder is None:
            folder = lay.scope_folder(self.layout(), name, label, years, decade)
        if split > 0 and not width:
            width = 3
        for other in self.scopes():
            a, b = other["folder"] + "/", folder + "/"
            if a.startswith(b) or b.startswith(a):
                raise SystemExit(f"scope {name} would go into {folder}, which is the folder of scope {other['name']}"
                                 f" ({other['folder']}); give it another --name")
        with self.lock:
            self.db.execute("INSERT INTO scopes (name, regions, years, decade, split, width, created, label, folder) "
                            "VALUES (?,?,?,?,?,?,?,?,?)",
                            (name, json.dumps(sorted(regions)), json.dumps(list(years)), decade, split, width,
                             time.time(), label, folder))
        return self.scope(name)

    def ensure_scope(self, name: str, regions, years: tuple[int, int], decade: int | None, split: int, label: str,
                     width: int) -> dict:
        """The scope of that name, made when missing; an existing one must have
        the same regions, years and decade."""
        sc = self.scope(name)
        if sc is None:
            return self.add_scope(name, list(regions), years, decade, split, label, width)
        if sorted(sc["regions"]) != sorted(regions) or tuple(sc["years"]) != tuple(years) or sc["decade"] != decade:
            raise SystemExit(f"scope {name} exists with regions {sc['regions']}, years {list(sc['years'])}, decade "
                             f"{sc['decade']}; the rule wants regions {sorted(regions)}, years {list(years)}, decade "
                             f"{decade}. Rename or drop the existing scope.")
        return sc

    def remove_scope(self, name: str) -> None:
        """Forget a scope: its definition, members, shards and counters (not its
        photos' status). The shards of a `run --by` family that made it are
        forgotten too, so the next run of the family pages again and makes it again."""
        def work(db):
            for t, col in (("members", "scope"), ("shards", "scope"), ("counters", "scope"), ("scopes", "name")):
                db.execute(f"DELETE FROM {t} WHERE {col}=?", (name,))
            for key, value in db.execute("SELECT key, value FROM meta WHERE key LIKE 'family:%'").fetchall():
                names = json.loads(value)
                if name in names:
                    db.execute("DELETE FROM shards WHERE scope=?", (key[len("family:"):],))
                    db.execute("UPDATE meta SET value=? WHERE key=?", (json.dumps([n for n in names if n != name]), key))
        self.tx(work)

    def add_family(self, family: str, names) -> None:
        """Remember the scopes a `run --by` family made (its shards are keyed by the family)."""
        cur = set(self.get_meta(f"family:{family}", []))
        self.set_meta(f"family:{family}", sorted(cur | set(names)))

    def family(self, family: str) -> list[str]:
        return self.get_meta(f"family:{family}", [])

    def members(self, scope: str) -> list[int]:
        with self.lock:
            return [r[0] for r in self.db.execute("SELECT cid FROM members WHERE scope=? ORDER BY cid", (scope,))]

    def add_member(self, scope: str, cid: int) -> bool:
        with self.lock:
            return self.db.execute("INSERT OR IGNORE INTO members (scope, cid) VALUES (?, ?)", (scope, cid)).rowcount > 0

    def remove_member(self, scope: str, cid: int) -> None:
        with self.lock:
            self.db.execute("DELETE FROM members WHERE scope=? AND cid=?", (scope, cid))

    def move_members(self, run: int, seq0: int, rows: list[tuple[int, str, str]]) -> int:
        """Membership moves of a regroup in one transaction: (cid, src, dst);
        logged from seq0 on, with whether dst gained the member. -> next seq."""
        def work(db):
            seq = seq0
            for cid, src, dst in rows:
                new = db.execute("INSERT OR IGNORE INTO members (scope, cid) VALUES (?, ?)", (dst, cid)).rowcount > 0
                db.execute("DELETE FROM members WHERE scope=? AND cid=?", (src, cid))
                db.execute("INSERT INTO regroup_moves VALUES (?,?,?,?,?,?,?,?,?)",
                           (run, seq, cid, src, None, dst, None, "member", int(new)))
                seq += 1
            return seq
        return self.tx(work)

    def add_members(self, scope: str, cids) -> None:
        def work(db):
            db.executemany("INSERT OR IGNORE INTO members (scope, cid) VALUES (?, ?)", [(scope, c) for c in cids])
        self.tx(work)

    def set_scope_width(self, name: str, width: int) -> None:
        with self.lock:
            self.db.execute("UPDATE scopes SET width=?, indexed=? WHERE name=?", (width, time.time(), name))

    def scopes(self) -> list[dict]:
        with self.lock:
            names = [r[0] for r in self.db.execute("SELECT name FROM scopes ORDER BY created")]
        return [self.scope(n) for n in names]

    # --- index ---------------------------------------------------------------------

    def shard(self, key: str) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT key, region, filter, count, pages_done, rows, done FROM shards WHERE key=?",
                                  (key,)).fetchone()
        return dict(zip(("key", "region", "filter", "count", "pages_done", "rows", "done"), row)) if row else None

    def start_shard(self, key: str, scope: str, region: int, filter_: str, count: int, restart: bool) -> None:
        with self.lock:
            prev = self.shard(key)
            pages = 0 if restart or not prev else prev["pages_done"]
            self.db.execute("INSERT OR REPLACE INTO shards (key, scope, region, filter, count, pages_done, rows, done, "
                            "updated) VALUES (?,?,?,?,?,?,?,0,?)",
                            (key, scope, region, filter_, count, pages, prev["rows"] if prev and not restart else 0,
                             time.time()))

    def add_page(self, key: str, scope: str, page: int, photos: list[dict], member) -> int:
        """Store one gallery page; member(photo) says whether the photo belongs
        to the scope (True), to another scope (its name, as `run --by` splits a
        download), or to none (False or None). -> new members."""
        now = time.time()

        def work(db):
            new = 0
            for p in photos:
                cid = int(p["cid"])
                db.execute("INSERT INTO photos (cid, file, title, year, year2, dir, rs, first_seen, last_seen) "
                           "VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(cid) DO UPDATE SET file=excluded.file, "
                           "title=excluded.title, year=excluded.year, dir=excluded.dir, rs=excluded.rs, "
                           "last_seen=excluded.last_seen",
                           (cid, p.get("file", ""), p.get("title", ""), p.get("year"), p.get("year2"), p.get("dir"),
                            json.dumps(p.get("rs") or []), now, now))
                target = member(p)
                if target:
                    cur = db.execute("INSERT OR IGNORE INTO members (scope, cid) VALUES (?, ?)",
                                     (scope if target is True else target, cid))
                    new += cur.rowcount
                    db.execute("INSERT OR IGNORE INTO status (cid, state, updated) VALUES (?, ?, ?)", (cid, PENDING, now))
            db.execute("UPDATE shards SET pages_done=?, rows=rows+?, updated=? WHERE key=?",
                       (page + 1, len(photos), now, key))
            return new

        return self.tx(work)

    def finish_shard(self, key: str) -> None:
        with self.lock:
            self.db.execute("UPDATE shards SET done=1, updated=? WHERE key=?", (time.time(), key))

    def photo(self, cid: int) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT cid, file, title, year, year2, dir, rs FROM photos WHERE cid=?", (cid,)).fetchone()
        if not row:
            return None
        d = dict(zip(("cid", "file", "title", "year", "year2", "dir", "rs"), row))
        d["rs"] = json.loads(d["rs"] or "[]")
        return d

    # --- status --------------------------------------------------------------------

    def scope_counts(self, scope: str) -> dict:
        """States of the scope's members; 'elsewhere' counts members saved under another scope."""
        with self.lock:
            total = self.db.execute("SELECT COUNT(*) FROM members WHERE scope=?", (scope,)).fetchone()[0]
            rows = self.db.execute("SELECT s.state, s.scope = ?, COUNT(*) FROM members m JOIN status s ON s.cid=m.cid "
                                   "WHERE m.scope=? GROUP BY s.state, s.scope = ?", (scope, scope, scope)).fetchall()
        out = {"total": total, PENDING: 0, WRITING: 0, DONE: 0, SKIPPED: 0, FAILED: 0, "elsewhere": 0}
        for state, mine, n in rows:
            if state == DONE and not mine:
                out["elsewhere"] += n
            else:
                out[state] = out.get(state, 0) + n
        return out

    def pending_cids(self, scope: str, retries: int) -> list[int]:
        with self.lock:
            rows = self.db.execute(
                "SELECT m.cid FROM members m JOIN status s ON s.cid=m.cid WHERE m.scope=? AND "
                "(s.state=? OR (s.state=? AND s.attempts<?)) ORDER BY m.cid", (scope, PENDING, FAILED, retries)).fetchall()
        return [r[0] for r in rows]

    def member_cids(self, scopes: list[str]) -> list[int]:
        """Every photo the scopes list, each once, in PastVu number order."""
        if not scopes:
            return []
        q = ",".join("?" * len(scopes))
        with self.lock:
            return [r[0] for r in self.db.execute(f"SELECT DISTINCT cid FROM members WHERE scope IN ({q}) ORDER BY cid",
                                                  scopes)]

    def pending_pairs(self, scopes: list[str], retries: int) -> list[tuple[int, str]]:
        """(cid, scope) to do in any of the scopes, each photo once, in cid order."""
        if not scopes:
            return []
        q = ",".join("?" * len(scopes))
        with self.lock:
            rows = self.db.execute(
                f"SELECT m.cid, MIN(m.scope) FROM members m JOIN status s ON s.cid=m.cid WHERE m.scope IN ({q}) AND "
                f"(s.state=? OR (s.state=? AND s.attempts<?)) GROUP BY m.cid ORDER BY m.cid",
                (*scopes, PENDING, FAILED, retries)).fetchall()
        return [(r[0], r[1]) for r in rows]

    def owned_rows(self, scope: str) -> list[tuple]:
        """(cid, state, path, category) of every photo the scope owns (done, skipped, failed)."""
        with self.lock:
            return self.db.execute("SELECT cid, state, path, category FROM status WHERE scope=? ORDER BY cid",
                                   (scope,)).fetchall()

    def set_owner(self, cid: int, scope: str | None, path: str | None, category: str | None,
                  record: str | None = None, state: str | None = None) -> None:
        def work(db):
            if state:
                db.execute("UPDATE status SET state=?, scope=?, path=?, category=?, updated=? WHERE cid=?",
                           (state, scope, path, category, time.time(), cid))
            else:
                db.execute("UPDATE status SET scope=?, path=?, category=?, updated=? WHERE cid=?",
                           (scope, path, category, time.time(), cid))
            if record is not None:
                db.execute("UPDATE records SET json=? WHERE cid=?", (record, cid))
        self.tx(work)

    def forget_photos(self, cids: list[int]) -> None:
        """Back to not downloaded: status pending, no record, embedding or VLM caption."""
        def work(db):
            for cid in cids:
                db.execute("UPDATE status SET state=?, reason='', attempts=0, scope=NULL, category=NULL, path=NULL, "
                           "sha256=NULL, bytes=NULL, width=NULL, height=NULL, changed=NULL, updated=? WHERE cid=?",
                           (PENDING, time.time(), cid))
                for t in ("records", "embeddings", "captions", "caption_fails", "caption_log"):
                    db.execute(f"DELETE FROM {t} WHERE cid=?", (cid,))
        self.tx(work)

    def recount(self, scope: str) -> dict:
        """The scope's counters again, from the status and records of the photos it owns."""
        from collections import Counter
        total = Counter()
        with self.lock:
            rows = self.db.execute("SELECT s.cid, s.state, s.reason, s.category, r.json FROM status s LEFT JOIN records r "
                                   "ON r.cid=s.cid WHERE s.scope=?", (scope,)).fetchall()
        for cid, state, reason, category, rec in rows:
            if state == DONE:
                total.update(record_counters(json.loads(rec) if rec else {}, category or ""))
            elif state in (SKIPPED, FAILED):
                key = (reason or "").split(":")[0].split(" (")[0]
                total[f"c.{state}.{key}"] += 1

        def work(db):
            db.execute("DELETE FROM counters WHERE scope=?", (scope,))
            self._add_counters(db, scope, dict(total))
        self.tx(work)
        return dict(total)

    def writing_rows(self, scope: str) -> list[tuple[int, str]]:
        with self.lock:
            return self.db.execute("SELECT cid, path FROM status WHERE state=? AND scope=?", (WRITING, scope)).fetchall()

    def claim(self, cid: int, scope: str, path: str) -> None:
        with self.lock:
            self.db.execute("UPDATE status SET state=?, scope=?, path=?, updated=? WHERE cid=?",
                            (WRITING, scope, path, time.time(), cid))

    def unclaim(self, cid: int) -> None:
        with self.lock:
            self.db.execute("UPDATE status SET state=?, path=NULL, updated=? WHERE cid=?", (PENDING, time.time(), cid))

    def finish(self, cid: int, scope: str, facts: dict, record: str, emb: tuple | None, counters: dict) -> None:
        """Done: status, record, embedding and counters in one transaction.
        emb: (model, patches, float16 vector) or None."""
        now = time.time()

        def work(db):
            db.execute("UPDATE status SET state=?, reason='', scope=?, category=?, path=?, sha256=?, bytes=?, width=?, "
                       "height=?, changed=?, updated=? WHERE cid=?",
                       (DONE, scope, facts.get("category", ""), facts["path"], facts.get("sha256"), facts.get("bytes"),
                        facts.get("width"), facts.get("height"), json.dumps(facts.get("changed") or []), now, cid))
            db.execute("INSERT OR REPLACE INTO records (cid, json) VALUES (?, ?)", (cid, record))
            if emb is not None:
                model, patches, vec = emb
                db.execute("INSERT OR REPLACE INTO embeddings (cid, model, patches, vec) VALUES (?,?,?,?)",
                           (cid, model, patches, np.asarray(vec, np.float16).tobytes()))
            self._add_counters(db, scope, counters)

        self.tx(work)

    def reject(self, cid: int, scope: str, state: str, reason: str, counters: dict) -> None:
        now = time.time()

        def work(db):
            db.execute("UPDATE status SET state=?, reason=?, scope=?, path=NULL, attempts=attempts+?, updated=? "
                       "WHERE cid=?", (state, reason, scope, 1 if state == FAILED else 0, now, cid))
            self._add_counters(db, scope, counters)

        self.tx(work)

    def status(self, cid: int) -> dict | None:
        with self.lock:
            row = self.db.execute(f"SELECT {', '.join(STATUS_COLS)} FROM status WHERE cid=?", (cid,)).fetchone()
        return dict(zip(STATUS_COLS, row)) if row else None

    def done_rows(self, scope: str) -> list[tuple]:
        """(cid, path, category, sha256) of the photos saved under the scope; paths relative to the base."""
        with self.lock:
            return self.db.execute("SELECT cid, path, category, sha256 FROM status WHERE state=? AND scope=? ORDER BY cid",
                                   (DONE, scope)).fetchall()

    def scope_paths(self, scope: str) -> list[str]:
        with self.lock:
            return [r[0] for r in self.db.execute("SELECT path FROM status WHERE scope=? AND state IN (?, ?) AND path "
                                                  "IS NOT NULL", (scope, DONE, WRITING))]

    def move_paths(self, rows: list[tuple[int, str, str | None]]) -> None:
        """New paths of saved photos in one transaction: (cid, path, record JSON or None)."""
        now = time.time()

        def work(db):
            for cid, path, record in rows:
                db.execute("UPDATE status SET path=?, updated=? WHERE cid=?", (path, now, cid))
                if record is not None:
                    db.execute("UPDATE records SET json=? WHERE cid=?", (record, cid))
        self.tx(work)

    def move(self, cid: int, path: str, category: str, record: str | None = None) -> None:
        def work(db):
            db.execute("UPDATE status SET path=?, category=?, updated=? WHERE cid=?", (path, category, time.time(), cid))
            if record is not None:
                db.execute("UPDATE records SET json=? WHERE cid=?", (record, cid))

        self.tx(work)

    # --- records and embeddings ------------------------------------------------------

    def record(self, cid: int) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT json FROM records WHERE cid=?", (cid,)).fetchone()
        return json.loads(row[0]) if row else None

    def embedding(self, cid: int, model: str, patches: int) -> np.ndarray | None:
        with self.lock:
            row = self.db.execute("SELECT vec FROM embeddings WHERE cid=? AND model=? AND patches=?",
                                  (cid, model, patches)).fetchone()
        return np.frombuffer(row[0], np.float16).astype(np.float32) if row else None

    def set_embedding(self, cid: int, model: str, patches: int, vec) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO embeddings (cid, model, patches, vec) VALUES (?,?,?,?)",
                            (cid, model, patches, np.asarray(vec, np.float16).tobytes()))

    # --- counters and runs -------------------------------------------------------------

    @staticmethod
    def _add_counters(db, scope: str, delta: dict) -> None:
        for k, v in (delta or {}).items():
            db.execute("INSERT INTO counters (scope, name, value) VALUES (?,?,?) ON CONFLICT(scope, name) DO UPDATE SET "
                       "value=value+excluded.value", (scope, k, float(v)))

    def counters(self, scope: str) -> dict:
        with self.lock:
            return {k: v for k, v in self.db.execute("SELECT name, value FROM counters WHERE scope=?", (scope,))}

    def move_counter(self, scope: str, old: str, new: str) -> None:
        """A photo changed category: move one count between cat.<old> and cat.<new>."""
        self.tx(lambda db: self._add_counters(db, scope, {f"cat.{old}": -1, f"cat.{new}": 1}))

    def start_run(self, scope: str, args: dict) -> int:
        with self.lock:
            cur = self.db.execute("INSERT INTO runs (scope, started, args) VALUES (?,?,?)",
                                  (scope, time.time(), json.dumps(args, ensure_ascii=False, default=str)))
            return cur.lastrowid

    def end_run(self, run_id: int, done: int, skipped: int, failed: int) -> None:
        with self.lock:
            self.db.execute("UPDATE runs SET ended=?, done=?, skipped=?, failed=? WHERE id=?",
                            (time.time(), done, skipped, failed, run_id))

    # --- reclassify ----------------------------------------------------------------------

    def start_reclass(self, scope: str, samples: str) -> int:
        with self.lock:
            cur = self.db.execute("INSERT INTO reclass_runs (scope, started, samples, moved) VALUES (?,?,?,0)",
                                  (scope, time.time(), samples))
            return cur.lastrowid

    def add_reclass_move(self, run: int, seq: int, cid: int, src: str, dst: str, cat_from: str, cat_to: str) -> None:
        with self.lock:
            self.db.execute("INSERT INTO reclass_moves VALUES (?,?,?,?,?,?,?)", (run, seq, cid, src, dst, cat_from, cat_to))
            self.db.execute("UPDATE reclass_runs SET moved=moved+1 WHERE id=?", (run,))

    def last_reclass(self, scope: str) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT id, started, samples, moved FROM reclass_runs WHERE scope=? AND undone IS NULL "
                                  "ORDER BY id DESC LIMIT 1", (scope,)).fetchone()
        return dict(zip(("id", "started", "samples", "moved"), row)) if row else None

    def reclass_moves(self, run: int) -> list[tuple]:
        with self.lock:
            return self.db.execute("SELECT seq, cid, src, dst, cat_from, cat_to FROM reclass_moves WHERE run=? "
                                   "ORDER BY seq DESC", (run,)).fetchall()

    def undo_reclass(self, run: int) -> None:
        with self.lock:
            self.db.execute("UPDATE reclass_runs SET undone=? WHERE id=?", (time.time(), run))

    # --- autocaption ---------------------------------------------------------------------

    def start_caption_run(self, scope: str, target: str, model: str, prompt_sha: str) -> int:
        with self.lock:
            cur = self.db.execute("INSERT INTO caption_runs (scope, target, model, prompt_sha, started) VALUES "
                                  "(?,?,?,?,?)", (scope, target, model, prompt_sha, time.time()))
            return cur.lastrowid

    def end_caption_run(self, run: int, captioned: int, failed: int) -> None:
        with self.lock:
            self.db.execute("UPDATE caption_runs SET ended=?, captioned=?, failed=? WHERE id=?",
                            (time.time(), captioned, failed, run))

    def caption(self, cid: int) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT cid, original, caption, run, model, prompt_sha FROM captions WHERE cid=?",
                                  (cid,)).fetchone()
        return dict(zip(("cid", "original", "caption", "run", "model", "prompt_sha"), row)) if row else None

    def captioned_cids(self, scope: str) -> set[int]:
        with self.lock:
            return {r[0] for r in self.db.execute("SELECT c.cid FROM captions c JOIN status s ON s.cid=c.cid "
                                                  "WHERE s.scope=?", (scope,))}

    def caption_attempts(self) -> dict[int, int]:
        with self.lock:
            return dict(self.db.execute("SELECT cid, attempts FROM caption_fails"))

    def save_caption(self, run: int, cid: int, before: str, after: str, model: str, prompt_sha: str) -> None:
        """A new VLM caption: the backup of the file's previous text in the run's
        log, the caption itself, and the original kept from the first time."""
        now = time.time()

        def work(db):
            db.execute("INSERT OR REPLACE INTO caption_log (run, cid, before, after) VALUES (?,?,?,?)",
                       (run, cid, before, after))
            db.execute("INSERT INTO captions (cid, original, caption, run, model, prompt_sha, time) VALUES "
                       "(?,?,?,?,?,?,?) ON CONFLICT(cid) DO UPDATE SET caption=excluded.caption, run=excluded.run, "
                       "model=excluded.model, prompt_sha=excluded.prompt_sha, time=excluded.time",
                       (cid, before, after, run, model, prompt_sha, now))
            db.execute("DELETE FROM caption_fails WHERE cid=?", (cid,))

        self.tx(work)

    def caption_failed(self, cid: int, error: str) -> None:
        with self.lock:
            self.db.execute("INSERT INTO caption_fails (cid, attempts, error, time) VALUES (?,1,?,?) "
                            "ON CONFLICT(cid) DO UPDATE SET attempts=attempts+1, error=excluded.error, "
                            "time=excluded.time", (cid, error[:500], time.time()))

    def last_caption_run(self, scope: str) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT r.id, r.target, r.started, (SELECT COUNT(*) FROM caption_log l WHERE "
                                  "l.run=r.id) FROM caption_runs r WHERE r.scope=? AND r.undone IS NULL AND EXISTS "
                                  "(SELECT 1 FROM caption_log l WHERE l.run=r.id) ORDER BY r.id DESC LIMIT 1",
                                  (scope,)).fetchone()
        return dict(zip(("id", "target", "started", "captioned"), row)) if row else None

    def caption_log_rows(self, run: int) -> list[tuple]:
        with self.lock:
            return self.db.execute("SELECT cid, before, after FROM caption_log WHERE run=?", (run,)).fetchall()

    def unsave_caption(self, run: int, cid: int) -> None:
        """Undo one photo's caption of a run: the photo's caption becomes the one
        of its latest earlier run that is not undone, or none."""

        def work(db):
            prev = db.execute("SELECT l.run, l.after FROM caption_log l JOIN caption_runs r ON r.id=l.run WHERE "
                              "l.cid=? AND l.run<? AND r.undone IS NULL ORDER BY l.run DESC LIMIT 1",
                              (cid, run)).fetchone()
            if prev:
                db.execute("UPDATE captions SET caption=?, run=? WHERE cid=?", (prev[1], prev[0], cid))
            else:
                db.execute("DELETE FROM captions WHERE cid=?", (cid,))

        self.tx(work)

    def undo_caption_run(self, run: int) -> None:
        with self.lock:
            self.db.execute("UPDATE caption_runs SET undone=? WHERE id=?", (time.time(), run))

    # --- regroup --------------------------------------------------------------------------

    def start_regroup(self, rule: str, target: str) -> int:
        with self.lock:
            return self.db.execute("INSERT INTO regroup_runs (rule, target, started) VALUES (?,?,?)",
                                   (rule, target, time.time())).lastrowid

    def add_regroup_move(self, run: int, seq: int, cid: int, src_scope, src_path, dst_scope, dst_path, kind: str,
                         new_member: bool) -> None:
        with self.lock:
            self.db.execute("INSERT INTO regroup_moves VALUES (?,?,?,?,?,?,?,?,?)",
                            (run, seq, cid, src_scope, src_path, dst_scope, dst_path, kind, int(new_member)))

    def end_regroup(self, run: int, moved: int, removed: list[dict], created: list[str]) -> None:
        with self.lock:
            self.db.execute("UPDATE regroup_runs SET moved=?, removed=?, created=? WHERE id=?",
                            (moved, json.dumps(removed, ensure_ascii=False), json.dumps(created), run))

    def last_regroup(self) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT id, rule, target, moved, removed, created FROM regroup_runs WHERE undone IS "
                                  "NULL ORDER BY id DESC LIMIT 1").fetchone()
        if not row:
            return None
        return {"id": row[0], "rule": row[1], "target": row[2], "moved": row[3],
                "removed": json.loads(row[4] or "[]"), "created": json.loads(row[5] or "[]")}

    def regroup_moves(self, run: int) -> list[tuple]:
        with self.lock:
            return self.db.execute("SELECT seq, cid, src_scope, src_path, dst_scope, dst_path, kind, new_member FROM "
                                   "regroup_moves WHERE run=? ORDER BY seq DESC", (run,)).fetchall()

    def undo_regroup(self, run: int) -> None:
        with self.lock:
            self.db.execute("UPDATE regroup_runs SET undone=? WHERE id=?", (time.time(), run))

    def close_reclass_runs(self, scope: str) -> int:
        """Reclassify runs of a scope whose photos moved can no longer be undone: close them."""
        with self.lock:
            return self.db.execute("UPDATE reclass_runs SET undone=? WHERE scope=? AND undone IS NULL",
                                   (time.time(), scope)).rowcount

    # --- the run lock -----------------------------------------------------------------------

    # who may work beside whom: a download beside a caption job, never two downloads,
    # and nothing beside a command that moves or deletes files
    LOCK_CONFLICTS = {"exclusive": {"exclusive", "run", "shared"}, "run": {"exclusive", "run"},
                      "shared": {"exclusive"}}

    def acquire_lock(self, command: str, mode: str = "exclusive") -> None:
        """Take this process's place in the base folder's lock, or stop when a
        running command conflicts with this one. Entries of processes that no
        longer run are dropped."""
        import os
        import runlock
        me = {"pid": os.getpid(), "start": runlock.process_start(os.getpid()), "command": command, "mode": mode,
              "time": time.time()}
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                row = self.db.execute("SELECT value FROM meta WHERE key='lock'").fetchone()
                held = json.loads(row[0]) if row else []
                held = [held] if isinstance(held, dict) else held
                live = [h for h in held if h.get("pid") != me["pid"] and runlock.is_running(h.get("pid"), h.get("start"))]
                for h in live:
                    if h.get("mode", "exclusive") in self.LOCK_CONFLICTS[mode]:
                        raise SystemExit(f"Another pastvu command works on this base folder: {h.get('command')} "
                                         f"(process {h.get('pid')}). Wait for it to finish.")
                self.db.execute("INSERT OR REPLACE INTO meta VALUES ('lock', ?)", (json.dumps(live + [me]),))
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
            self.db.execute("COMMIT")

    def release_lock(self) -> None:
        import os
        with self.lock:
            row = self.db.execute("SELECT value FROM meta WHERE key='lock'").fetchone()
            if not row:
                return
            held = json.loads(row[0])
            held = [held] if isinstance(held, dict) else held
            rest = [h for h in held if h.get("pid") != os.getpid()]
            if rest:
                self.db.execute("UPDATE meta SET value=? WHERE key='lock'", (json.dumps(rest),))
            else:
                self.db.execute("DELETE FROM meta WHERE key='lock'")


def record_counters(rec: dict, folder: str) -> dict:
    """The counters one saved photo adds to its scope, from its record."""
    p = rec.get("processing") or {}
    wm = p.get("watermarks") or {}
    kw = {"c.done": 1, f"c.write.{p.get('write', 'lossless')}": 1}
    changed = p.get("changed") or []
    band = p.get("band") or {}
    if "band" in changed:
        kw["c.band"] = 1
    if band.get("rows") and not band.get("ok"):
        kw["c.band_unverified"] = 1
    borders = p.get("borders") or {}
    if "borders" in changed:
        kw["c.borders"] = 1
        for kind in {c["kind"] for c in borders.get("cuts") or []}:
            kw[f"c.cut.{kind}"] = 1
    if borders.get("action") == "review":
        kw["c.borders_review"] = 1
    if wm.get("painted"):
        kw["c.wm_images"] = 1
        kw["c.wm_boxes"] = len(wm["painted"])
    if wm.get("trim"):
        kw["c.wm_trim"] = 1
    if wm.get("detected") and not wm.get("painted") and not wm.get("trim"):
        kw["c.wm_below"] = 1
    if folder:
        kw[f"cat.{folder}"] = 1
    return kw
