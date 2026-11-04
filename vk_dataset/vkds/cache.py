"""Кэш ответов VK API: повторные прогоны и контрольные сверки бесплатны.

Кэшируются только «тяжёлые» списочные методы (photos.getAlbums, photos.get) —
именно они дают основную массу вызовов при полном скачивании группы.
groups.getById НЕ кэшируется: один вызов на прогон — дешёвая живая проверка,
что группа всё ещё существует и доступна (поэтому whoami остаётся честной
проверкой токена, а повторный прогон замечает переименование группы).

Файлы: <датасет>/_vkds/cache/<метод>/<sha1(метод+параметры)>.json с полями
ts, ts_epoch, params, response. TTL по умолчанию 7 дней (состав альбомов
меняется редко; докачка после обрыва попадает в свежий кэш того же дня).

Кэш-хиты НЕ расходуют месячный лимит и не пишутся в budget.json — в этом
и есть смысл: контрольный повторный прогон той же команды стоит 1 вызов
(groups.getById) вместо полного прохода по альбомам.

Управление: --refresh — читать мимо кэша (записи обновляются),
--no-cache — полностью отключить (файлы не создаются и не читаются).
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Optional

from .budget import now_iso
from .fs import read_json, write_json

CACHEABLE = {"photos.getAlbums", "photos.get"}
DEFAULT_TTL_DAYS = 7.0


class ApiCache:
    """Простейший дисковый кэш ответов списочных методов VK API."""

    def __init__(self, root: Path, ttl_days: float = DEFAULT_TTL_DAYS):
        self.root = Path(root)
        self.ttl = max(0.0, float(ttl_days)) * 86400.0
        self.hits = 0      # сколько ответов отдано из кэша (для отчёта)
        self.writes = 0

    @staticmethod
    def _key(method: str, params: dict) -> str:
        """Ключ = sha1(метод + отсортированные параметры)."""
        norm = "&".join("{}={}".format(k, params[k]) for k in sorted(params))
        return hashlib.sha1("{}|{}".format(method, norm)
                            .encode("utf-8")).hexdigest()

    def _path(self, method: str, key: str) -> Path:
        return self.root / method.replace(".", "_") / (key + ".json")

    def get(self, method: str, params: dict) -> Optional[Any]:
        """Ответ из кэша или None (нет файла / истёк TTL / не кэшируем)."""
        if method not in CACHEABLE:
            return None
        data = read_json(self._path(method, self._key(method, params)), None)
        if not isinstance(data, dict) or "response" not in data:
            return None
        ts = float(data.get("ts_epoch") or 0)
        if self.ttl > 0 and (time.time() - ts) > self.ttl:
            return None
        self.hits += 1
        return data["response"]

    def put(self, method: str, params: dict, response: Any) -> None:
        """Сохранить успешный ответ (только кэшируемые методы)."""
        if method not in CACHEABLE:
            return
        p = self._path(method, self._key(method, params))
        write_json(p, {"ts": now_iso(), "ts_epoch": time.time(),
                       "params": params, "response": response})
        self.writes += 1
