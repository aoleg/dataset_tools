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
            row = self.db.execute("SELECT name, regions, years, decade, split, width, created, indexed FROM scopes "
                                  "WHERE name=?", (name,)).fetchone()
        if not row:
            return None
        return {"name": row[0], "regions": json.loads(row[1]), "years": tuple(json.loads(row[2])), "decade": row[3],
                "split": row[4], "width": row[5], "created": row[6], "indexed": row[7]}

    def add_scope(self, name: str, regions: list[int], years: tuple[int, int], decade: int | None, split: int) -> dict:
        with self.lock:
            self.db.execute("INSERT INTO scopes (name, regions, years, decade, split, width, created) VALUES "
                            "(?,?,?,?,?,0,?)", (name, json.dumps(sorted(regions)), json.dumps(list(years)), decade, split,
                                                time.time()))
        return self.scope(name)

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
        to the scope (decades keep start years only). -> new members."""
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
                if member(p):
                    cur = db.execute("INSERT OR IGNORE INTO members (scope, cid) VALUES (?, ?)", (scope, cid))
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
