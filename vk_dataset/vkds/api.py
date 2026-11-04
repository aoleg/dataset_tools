"""Клиент VK API v5.199: лимит частоты, повторы ошибок, execute, бюджет вызовов,
кэш списочных методов.

Важно: в месячный лимит ВК (10 000 вызовов без верифицированного бизнес-профиля)
входят ТОЛЬКО запросы к api.vk.com. Скачивание самих файлов с CDN ВК —
бесплатно и не ограничено (см. fs.download_file). Ответы из кэша (cache.py)
тоже бесплатны — это и делает повторные/контрольные прогоны бесплатными.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import requests

from .budget import Budget
from .cache import ApiCache

API_URL = "https://api.vk.com/method/"
API_VERSION = "5.199"
# Повторяемые коды: 1 — неизвестная, 6 — слишком много запросов/сек,
# 9 — flood control, 10 — внутренняя ошибка сервера.
RETRY_CODES = {1, 6, 9, 10}


class VkApiError(RuntimeError):
    """Ошибка VK API с кодом и именем метода."""

    def __init__(self, code: int, message: str, method: str = ""):
        super().__init__("VK API {}: [{}] {}".format(method, code, message))
        self.code = code
        self.method = method


class VkClient:
    """Однопоточный клиент API с троттлингом (~3 запроса/сек) и ретраями.

    Файлы с CDN качаются отдельно (fs.download_file) — они лимит не расходуют,
    поэтому экономить нужно только вызовы call()/execute(). Тяжёлые списочные
    методы (photos.getAlbums, photos.get) при наличии cache отдаются из кэша
    без вызова API и без расхода бюджета.
    """

    def __init__(self, token: str, delay: float = 0.35, timeout: float = 30.0,
                 session=None, budget: Optional[Budget] = None,
                 retry_sleep: float = 1.5,
                 cache: Optional[ApiCache] = None, refresh: bool = False):
        self.token = token
        self.delay = max(0.0, delay)
        self.timeout = timeout
        self.budget = budget
        self.retry_sleep = retry_sleep
        self.cache = cache
        self.refresh = refresh
        self.session = session if session is not None else requests.Session()
        if session is None:
            self.session.headers["User-Agent"] = "vkds/0.4 (dataset harvester)"
        self.api_calls = 0          # живых вызовов за этот запуск (для отчёта)
        self.cache_hits = 0         # ответов отдано из кэша (бесплатно)
        self._last_call = 0.0

    def call(self, method: str, **params: Any) -> dict:
        """Вызов метода API. Возвращает payload["response"].

        Сначала проверяется кэш (если включён и не --refresh): хит бесплатен.
        Повторяет запрос при сетевых сбоях и кодах 1/6/9/10 (до 6 попыток);
        успешный ответ кэшируемого метода попадает в кэш.
        """
        clean = {k: v for k, v in params.items() if v is not None}
        if self.cache is not None and not self.refresh:
            cached = self.cache.get(method, clean)
            if cached is not None:
                self.cache_hits += 1
                return cached
        if self.budget is not None:
            self.budget.check()
        data = dict(clean)
        data["access_token"] = self.token
        data["v"] = API_VERSION
        last_err: Exception | None = None
        for attempt in range(6):
            self._throttle()
            try:
                resp = self.session.post(API_URL + method, data=data,
                                         timeout=self.timeout)
                payload = resp.json()
            except (requests.RequestException, ValueError) as e:
                last_err = e
                time.sleep(self.retry_sleep * (attempt + 1))
                continue
            if isinstance(payload, dict) and "error" in payload:
                err = payload["error"]
                code = int(err.get("error_code", 0) or 0)
                if code in RETRY_CODES and attempt < 5:
                    pause = 8.0 if code == 9 else self.retry_sleep
                    time.sleep(pause * (attempt + 1))
                    continue
                raise VkApiError(code, str(err.get("error_msg", "")), method)
            if self.budget is not None:
                self.budget.spend(method)
            self.api_calls += 1
            if self.cache is not None:
                self.cache.put(method, clean, payload["response"])
            return payload["response"]
        raise VkApiError(-1, "сеть недоступна: {}".format(last_err), method)

    def execute(self, code: str) -> Any:
        """execute: до 25 методов API за один вызов.

        Основной способ экономии месячного лимита на больших объёмах
        (стены, комментарии); пока не используется.
        """
        return self.call("execute", code=code)

    def _throttle(self) -> None:
        now = time.monotonic()
        wait = self._last_call + self.delay - now
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()
