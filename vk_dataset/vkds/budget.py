"""Учёт вызовов VK API: месячный лимит — 10 000 (без верификации бизнес-профиля).

Каждый вызов журналируется в <датасет>/_vkds/budget.json, поэтому и человек,
и ИИ-агент всегда видят, сколько лимита потрачено и на какие методы.
Опция --budget N задаёт жёсткий потолок на один запуск (страховка от ошибок).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .fs import read_json, write_json

LOG_CAP = 500    # сколько последних записей держать в журнале
RUNS_CAP = 200   # сколько последних запусков помнить


class BudgetExceeded(RuntimeError):
    """Достигнут лимит вызовов API, заданный для текущего запуска (--budget)."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Budget:
    """Журнал расхода вызовов API + лимит на текущий запуск."""

    def __init__(self, path: Optional[Path] = None, run_limit: Optional[int] = None,
                 label: str = ""):
        self.path = Path(path) if path else None
        self.run_limit = run_limit
        self.label = label
        self.run_used = 0
        data = read_json(self.path, None) if self.path else None
        if not isinstance(data, dict):
            data = {}
        data.setdefault("total_calls", 0)
        data.setdefault("by_method", {})
        data.setdefault("runs", [])
        data.setdefault("log", [])
        self.data = data

    @property
    def total(self) -> int:
        """Всего вызовов, записанных в журнал за всё время."""
        return int(self.data.get("total_calls", 0))

    def check(self) -> None:
        """Бросает BudgetExceeded, если лимит запуска исчерпан."""
        if self.run_limit is not None and self.run_used >= self.run_limit:
            raise BudgetExceeded(
                "Достигнут лимит {} вызовов API на этот запуск. "
                "Общий расход: vkds budget".format(self.run_limit))

    def spend(self, method: str) -> None:
        """Засчитать один выполненный вызов метода."""
        self.check()
        self.run_used += 1
        d = self.data
        d["total_calls"] = int(d.get("total_calls", 0)) + 1
        by = d.setdefault("by_method", {})
        by[method] = int(by.get(method, 0)) + 1
        log = d.setdefault("log", [])
        log.append({"ts": now_iso(), "method": method})
        if len(log) > LOG_CAP:
            del log[:len(log) - LOG_CAP]
        # периодически сохраняем — журнал переживёт обрыв питания
        if self.path is not None and int(d["total_calls"]) % 25 == 0:
            self.save()

    def finish_run(self, ok: bool = True) -> None:
        runs = self.data.setdefault("runs", [])
        runs.append({"ts": now_iso(), "label": self.label,
                     "calls": self.run_used, "ok": ok})
        if len(runs) > RUNS_CAP:
            del runs[:len(runs) - RUNS_CAP]
        self.save()

    def save(self) -> None:
        if self.path is not None:
            write_json(self.path, self.data)
