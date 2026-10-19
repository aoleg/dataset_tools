"""The SQLite store in <out>_meta/pastvu.db: the region tree, the index of
the scope (one row per photo), the status of every photo (what the stream did
with it) and the run counters. Everything a resume needs is here; the files in
the dataset folder are the output, not the state.

One connection, shared by the threads of the stream under a lock; every
public method is one transaction.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

DB_NAME = "pastvu.db"
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS regions (
    cid INTEGER PRIMARY KEY, parent INTEGER, parents TEXT, title_en TEXT, title_local TEXT,
    phc INTEGER, pac INTEGER, child_len INTEGER, pulled REAL);
CREATE INDEX IF NOT EXISTS regions_parent ON regions(parent);
CREATE TABLE IF NOT EXISTS shards (
    key TEXT PRIMARY KEY, region INTEGER, filter TEXT, count INTEGER, pages_done INTEGER DEFAULT 0,
    rows INTEGER DEFAULT 0, done INTEGER DEFAULT 0, updated REAL);
CREATE TABLE IF NOT EXISTS photos (
    cid INTEGER PRIMARY KEY, file TEXT, title TEXT, year INTEGER, year2 INTEGER, dir TEXT, rs TEXT,
    shard TEXT, first_seen REAL, last_seen REAL);
CREATE TABLE IF NOT EXISTS status (
    cid INTEGER PRIMARY KEY, state TEXT, reason TEXT, attempts INTEGER DEFAULT 0, category TEXT, path TEXT,
    sha256 TEXT, bytes INTEGER, width INTEGER, height INTEGER, changed TEXT, updated REAL);
CREATE INDEX IF NOT EXISTS status_state ON status(state);
CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value REAL DEFAULT 0);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, started REAL, ended REAL, args TEXT, done INTEGER, skipped INTEGER,
    failed INTEGER);
"""

PENDING, DONE, SKIPPED, FAILED = "pending", "done", "skipped", "failed"


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        with self.lock:
            self.db.close()

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
        with self.lock:
            self.db.execute("BEGIN")
            self.db.executemany("INSERT OR REPLACE INTO regions VALUES (?,?,?,?,?,?,?,?,?)", rows)
            self.db.execute("COMMIT")

    def region(self, cid: int) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT cid, parent, parents, title_en, title_local, phc, pac, child_len FROM regions "
                                  "WHERE cid=?", (cid,)).fetchone()
        return self._region_row(row) if row else None

    def children(self, cid: int) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT cid, parent, parents, title_en, title_local, phc, pac, child_len FROM regions "
                                   "WHERE parent=? ORDER BY phc DESC", (cid,)).fetchall()
        return [self._region_row(r) for r in rows]

    def regions_count(self) -> int:
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM regions").fetchone()[0]

    def find_regions(self, text: str) -> list[dict]:
        like = f"%{text}%"
        with self.lock:
            rows = self.db.execute("SELECT cid, parent, parents, title_en, title_local, phc, pac, child_len FROM regions "
                                   "WHERE title_en LIKE ? OR title_local LIKE ? ORDER BY phc DESC LIMIT 40",
                                   (like, like)).fetchall()
        return [self._region_row(r) for r in rows]

    @staticmethod
    def _region_row(row) -> dict:
        return {"cid": row[0], "parent": row[1], "parents": json.loads(row[2] or "[]"), "title_en": row[3],
                "title_local": row[4], "phc": row[5], "pac": row[6], "child_len": row[7]}

    # --- shards and photos ---------------------------------------------------------

    def shard(self, key: str) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT key, region, filter, count, pages_done, rows, done FROM shards WHERE key=?",
                                  (key,)).fetchone()
        if not row:
            return None
        return dict(zip(("key", "region", "filter", "count", "pages_done", "rows", "done"), row))

    def start_shard(self, key: str, region: int, filter_: str, count: int) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO shards (key, region, filter, count, pages_done, rows, done, updated) "
                            "VALUES (?,?,?,?, COALESCE((SELECT pages_done FROM shards WHERE key=?), 0), "
                            "COALESCE((SELECT rows FROM shards WHERE key=?), 0), 0, ?)",
                            (key, region, filter_, count, key, key, time.time()))

    def add_page(self, key: str, page: int, photos: list[dict]) -> int:
        """Store one gallery page; -> how many cids were new."""
        now = time.time()
        new = 0
        with self.lock:
            self.db.execute("BEGIN")
            for p in photos:
                cid = int(p["cid"])
                exists = self.db.execute("SELECT 1 FROM photos WHERE cid=?", (cid,)).fetchone()
                self.db.execute(
                    "INSERT OR REPLACE INTO photos (cid, file, title, year, year2, dir, rs, shard, first_seen, last_seen) "
                    "VALUES (?,?,?,?,?,?,?,?, COALESCE((SELECT first_seen FROM photos WHERE cid=?), ?), ?)",
                    (cid, p.get("file", ""), p.get("title", ""), p.get("year"), p.get("year2"), p.get("dir"),
                     json.dumps(p.get("rs") or []), key, cid, now, now))
                if not exists:
                    new += 1
                    self.db.execute("INSERT OR IGNORE INTO status (cid, state, updated) VALUES (?, ?, ?)",
                                    (cid, PENDING, now))
            self.db.execute("UPDATE shards SET pages_done=?, rows=rows+?, updated=? WHERE key=?",
                            (page + 1, len(photos), now, key))
            self.db.execute("COMMIT")
        return new

    def finish_shard(self, key: str) -> None:
        with self.lock:
            self.db.execute("UPDATE shards SET done=1, updated=? WHERE key=?", (time.time(), key))

    def photo(self, cid: int) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT cid, file, title, year, year2, dir, rs, shard FROM photos WHERE cid=?",
                                  (cid,)).fetchone()
        if not row:
            return None
        d = dict(zip(("cid", "file", "title", "year", "year2", "dir", "rs", "shard"), row))
        d["rs"] = json.loads(d["rs"] or "[]")
        return d

    def counts(self) -> dict:
        with self.lock:
            total = self.db.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
            rows = self.db.execute("SELECT state, COUNT(*) FROM status GROUP BY state").fetchall()
        out = {"total": total, PENDING: 0, DONE: 0, SKIPPED: 0, FAILED: 0}
        out.update({s: n for s, n in rows})
        return out

    # --- status --------------------------------------------------------------------

    def pending_cids(self, retry_failed: int = 0) -> list[int]:
        """The cids the stream still has to do, in cid order: pending, and
        failed ones with fewer than retry_failed attempts."""
        with self.lock:
            rows = self.db.execute(
                "SELECT cid FROM status WHERE state=? OR (state=? AND attempts<?) ORDER BY cid",
                (PENDING, FAILED, retry_failed)).fetchall()
        return [r[0] for r in rows]

    def set_status(self, cid: int, state: str, reason: str = "", **facts) -> None:
        cols = {"category", "path", "sha256", "bytes", "width", "height", "changed"}
        extra = {k: v for k, v in facts.items() if k in cols}
        if "changed" in extra and not isinstance(extra["changed"], str):
            extra["changed"] = json.dumps(extra["changed"])
        sets = ", ".join(f"{k}=?" for k in extra)
        with self.lock:
            self.db.execute(
                f"UPDATE status SET state=?, reason=?, attempts=attempts+?, updated=?{', ' + sets if sets else ''} WHERE cid=?",
                (state, reason, 1 if state == FAILED else 0, time.time(), *extra.values(), cid))

    def status(self, cid: int) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT cid, state, reason, attempts, category, path, sha256, bytes, width, height, "
                                  "changed FROM status WHERE cid=?", (cid,)).fetchone()
        if not row:
            return None
        return dict(zip(("cid", "state", "reason", "attempts", "category", "path", "sha256", "bytes", "width", "height",
                         "changed"), row))

    def done_rows(self):
        with self.lock:
            return self.db.execute("SELECT cid, path, sha256, bytes, width, height, category FROM status WHERE state=?",
                                   (DONE,)).fetchall()

    # --- counters ------------------------------------------------------------------

    def add_counters(self, delta: dict) -> None:
        if not delta:
            return
        with self.lock:
            self.db.execute("BEGIN")
            for k, v in delta.items():
                self.db.execute("INSERT INTO counters (name, value) VALUES (?, ?) "
                                "ON CONFLICT(name) DO UPDATE SET value=value+excluded.value", (k, float(v)))
            self.db.execute("COMMIT")

    def counters(self) -> dict:
        with self.lock:
            return {k: v for k, v in self.db.execute("SELECT name, value FROM counters").fetchall()}

    def start_run(self, args: dict) -> int:
        with self.lock:
            cur = self.db.execute("INSERT INTO runs (started, args) VALUES (?, ?)",
                                  (time.time(), json.dumps(args, ensure_ascii=False, default=str)))
            return cur.lastrowid

    def end_run(self, run_id: int, done: int, skipped: int, failed: int) -> None:
        with self.lock:
            self.db.execute("UPDATE runs SET ended=?, done=?, skipped=?, failed=? WHERE id=?",
                            (time.time(), done, skipped, failed, run_id))
