"""Accounting of VK API calls: the run journal of a dataset and the ledger.

Two files:

- `Budget`: <dataset>/_vkds/budget.json, the per-run detail of one dataset
  (successful calls by method, the last calls, the runs with their attempts),
  plus the hard cap of one run (--budget N), which counts every HTTP attempt.
- `Ledger`: the ledger of this checkout of the tool ([paths] ledger in
  vkds.conf, default _vkds/ledger.json next to the conf). Daily buckets (UTC
  days) of attempts and successes by method, the limit hits and the first
  success after each of them, and an optional reset_day the user sets once
  the reset of the monthly quota (10,000 calls per application) is observed.
  The month view sums the days from the start of the current period.

Every HTTP attempt counts, retried and failed ones included; successes count
separately. Cache hits and CDN downloads are free and count nowhere.
"""
from __future__ import annotations

import calendar
from datetime import UTC, date, datetime
from pathlib import Path

from .fs import read_json, write_json

LOG_CAP = 500    # how many recent calls budget.json keeps
RUNS_CAP = 200   # how many runs budget.json keeps
HITS_CAP = 200   # how many limit hits / clearances the ledger keeps


class BudgetExceeded(RuntimeError):
    """The cap of API attempts of this run (--budget) is reached."""


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def today_utc() -> date:
    return datetime.now(UTC).date()


class Budget:
    """The per-dataset journal of successful calls plus the cap of one run."""

    def __init__(self, path: Path | None = None, run_limit: int | None = None,
                 label: str = ""):
        self.path = Path(path) if path else None
        self.run_limit = run_limit
        self.label = label
        self.run_used = 0      # HTTP attempts of this run (what --budget caps)
        self.run_ok = 0        # successful calls of this run
        data = read_json(self.path, None) if self.path else None
        if not isinstance(data, dict):
            data = {}
        data.setdefault("total_calls", 0)
        data.setdefault("total_attempts", data.get("total_calls", 0))
        data.setdefault("by_method", {})
        data.setdefault("runs", [])
        data.setdefault("log", [])
        self.data = data

    @property
    def total(self) -> int:
        """Successful calls recorded in this journal, all time."""
        return int(self.data.get("total_calls", 0))

    def check(self) -> None:
        """Raise BudgetExceeded when the attempts of this run reached the cap."""
        if self.run_limit is not None and self.run_used >= self.run_limit:
            raise BudgetExceeded(
                "Достигнут лимит {} вызовов API на этот запуск. "
                "Общий расход: run.bat budget".format(self.run_limit))

    def attempt(self, method: str) -> None:
        """Count one HTTP attempt (call check() first)."""
        self.run_used += 1
        self.data["total_attempts"] = int(self.data.get("total_attempts", 0)) + 1

    def spend(self, method: str) -> None:
        """Record one successful call of a method."""
        self.run_ok += 1
        d = self.data
        d["total_calls"] = int(d.get("total_calls", 0)) + 1
        by = d.setdefault("by_method", {})
        by[method] = int(by.get(method, 0)) + 1
        log = d.setdefault("log", [])
        log.append({"ts": now_iso(), "method": method})
        if len(log) > LOG_CAP:
            del log[:len(log) - LOG_CAP]
        # save now and then, so the journal survives a power cut
        if self.path is not None and int(d["total_calls"]) % 25 == 0:
            self.save()

    def finish_run(self, ok: bool = True) -> None:
        runs = self.data.setdefault("runs", [])
        runs.append({"ts": now_iso(), "label": self.label,
                     "calls": self.run_ok, "attempts": self.run_used, "ok": ok})
        if len(runs) > RUNS_CAP:
            del runs[:len(runs) - RUNS_CAP]
        self.save()

    def save(self) -> None:
        if self.path is not None:
            write_json(self.path, self.data)


def _empty_ledger() -> dict:
    return {"version": 1, "days": {}, "limit_hits": [], "limit_cleared": [],
            "reset_day": None}


def _norm_ledger(data) -> dict:
    if not isinstance(data, dict):
        data = {}
    base = _empty_ledger()
    base.update(data)
    if not isinstance(base.get("days"), dict):
        base["days"] = {}
    for key in ("limit_hits", "limit_cleared"):
        if not isinstance(base.get(key), list):
            base[key] = []
    return base


def parse_reset_day(value) -> int | None:
    """reset_day as a day of the month 1..31; an ISO date gives its day."""
    if value is None or value == "":
        return None
    try:
        if isinstance(value, str) and "-" in value:
            return date.fromisoformat(value.strip()[:10]).day
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if 1 <= n <= 31 else None


def period_start(today: date, reset_day: int | None) -> date:
    """The first day of the quota period that holds `today`.

    Without a reset day: the first of the calendar month. With one: the last
    date on or before today whose day is reset_day (clamped to the length of
    the month, so 31 means the last day of a short month).
    """
    if not reset_day:
        return today.replace(day=1)
    y, m = today.year, today.month
    for _ in range(2):
        last = calendar.monthrange(y, m)[1]
        cand = date(y, m, min(reset_day, last))
        if cand <= today:
            return cand
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
    return today.replace(day=1)   # not reached


class Ledger:
    """Daily attempts and successes of this checkout, with the limit hits.

    Every write reads the file again and adds to what is there, so two runs
    of the tool at once lose at most what lands between one read and one
    write. A ledger without a path keeps everything in memory (tests).
    """

    def __init__(self, path: Path | None = None, monthly: int = 10000):
        self.path = Path(path) if path else None
        self.monthly = int(monthly)
        self.data = _norm_ledger(read_json(self.path, None) if self.path else None)

    def _reload(self) -> dict:
        if self.path is not None:
            self.data = _norm_ledger(read_json(self.path, None))
        return self.data

    def _save(self) -> None:
        if self.path is not None:
            write_json(self.path, self.data)

    def record(self, method: str, ok: bool, today: date | None = None) -> None:
        """One HTTP attempt of `method`; ok is True for a successful response.

        The first success after a limit hit is written to limit_cleared.
        """
        d = self._reload()
        key = (today or today_utc()).isoformat()
        day = d["days"].setdefault(key, {"attempts": 0, "ok": 0, "by_method": {}})
        m = day.setdefault("by_method", {}).setdefault(method, {"attempts": 0, "ok": 0})
        day["attempts"] = int(day.get("attempts", 0)) + 1
        m["attempts"] = int(m.get("attempts", 0)) + 1
        if ok:
            day["ok"] = int(day.get("ok", 0)) + 1
            m["ok"] = int(m.get("ok", 0)) + 1
            hits, cleared = d["limit_hits"], d["limit_cleared"]
            if hits and (not cleared or str(cleared[-1]) < str(hits[-1].get("ts", ""))):
                cleared.append(now_iso())
                del cleared[:max(0, len(cleared) - HITS_CAP)]
        self._save()

    def record_limit_hit(self, code: int, message: str, method: str = "") -> dict:
        d = self._reload()
        hit = {"ts": now_iso(), "code": code, "message": message, "method": method}
        d["limit_hits"].append(hit)
        del d["limit_hits"][:max(0, len(d["limit_hits"]) - HITS_CAP)]
        self._save()
        return hit

    @property
    def reset_day(self) -> int | None:
        return parse_reset_day(self.data.get("reset_day"))

    def period_total(self, today: date | None = None) -> dict:
        """Attempts and successes from the start of the current period to today."""
        today = today or today_utc()
        start = period_start(today, self.reset_day)
        tot = {"start": start.isoformat(), "end": today.isoformat(),
               "reset_day": self.reset_day, "attempts": 0, "ok": 0, "by_method": {}}
        for key, day in self.data["days"].items():
            try:
                k = date.fromisoformat(str(key))
            except ValueError:
                continue
            if not (start <= k <= today) or not isinstance(day, dict):
                continue
            tot["attempts"] += int(day.get("attempts", 0))
            tot["ok"] += int(day.get("ok", 0))
            for name, m in (day.get("by_method") or {}).items():
                t = tot["by_method"].setdefault(name, {"attempts": 0, "ok": 0})
                t["attempts"] += int((m or {}).get("attempts", 0))
                t["ok"] += int((m or {}).get("ok", 0))
        return tot
