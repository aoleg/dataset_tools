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

from .budget import Budget, Ledger
from .cache import ApiCache

API_URL = "https://api.vk.com/method/"
API_VERSION = "5.199"
# Повторяемые коды: 1 — неизвестная, 6 — слишком много запросов/сек,
# 9 — flood control, 10 — внутренняя ошибка сервера.
RETRY_CODES = {1, 6, 9, 10}
# 29: "Rate limit reached", the limit error (never retried); any other
# non-retryable error whose message says "limit" is treated the same way.
LIMIT_CODE = 29


def is_limit_error(code: int, message: str) -> bool:
    return code == LIMIT_CODE or (code not in RETRY_CODES
                                  and "limit" in (message or "").lower())


class VkApiError(RuntimeError):
    """Ошибка VK API с кодом и именем метода."""

    def __init__(self, code: int, message: str, method: str = "",
                 limit: bool = False):
        super().__init__("VK API {}: [{}] {}".format(method, code, message))
        self.code = code
        self.message = message
        self.method = method
        self.limit = limit      # the quota error: stop the run, never retry


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
                 cache: Optional[ApiCache] = None, refresh: bool = False,
                 ledger: Optional[Ledger] = None):
        self.token = token
        self.delay = max(0.0, delay)
        self.timeout = timeout
        self.budget = budget
        self.ledger = ledger
        self.retry_sleep = retry_sleep
        self.cache = cache
        self.refresh = refresh
        self.session = session if session is not None else requests.Session()
        if session is None:
            self.session.headers["User-Agent"] = "vkds/0.6 (dataset harvester)"
        self.api_calls = 0          # живых вызовов за этот запуск (для отчёта)
        self.attempts = 0           # HTTP-попыток за запуск (с повторами и сбоями)
        self.cache_hits = 0         # ответов отдано из кэша (бесплатно)
        self._last_call = 0.0

    def call(self, method: str, _live: bool = False, **params: Any) -> dict:
        """Call an API method and return payload["response"].

        The cache is read first unless --refresh or `_live` (a real download
        lists live, D7); a hit is free. Network failures and the codes
        1/6/9/10 are retried (up to 6 attempts); every attempt is counted in
        the run budget and the ledger. The limit error (29, or a message
        with "limit") is recorded as a limit hit and raised at once with
        `limit=True`. A successful response of a cacheable method is cached.
        """
        clean = {k: v for k, v in params.items() if v is not None}
        if self.cache is not None and not self.refresh and not _live:
            cached = self.cache.get(method, clean)
            if cached is not None:
                self.cache_hits += 1
                return cached
        data = dict(clean)
        data["access_token"] = self.token
        data["v"] = API_VERSION
        last_err: Exception | None = None
        for attempt in range(6):
            if self.budget is not None:
                self.budget.check()
                self.budget.attempt(method)
            self.attempts += 1
            self._throttle()
            try:
                resp = self.session.post(API_URL + method, data=data,
                                         timeout=self.timeout)
                payload = resp.json()
            except (requests.RequestException, ValueError) as e:
                self._record(method, False)
                last_err = e
                time.sleep(self.retry_sleep * (attempt + 1))
                continue
            if isinstance(payload, dict) and "error" in payload:
                self._record(method, False)
                err = payload["error"]
                code = int(err.get("error_code", 0) or 0)
                msg = str(err.get("error_msg", ""))
                if is_limit_error(code, msg):
                    if self.ledger is not None:
                        self.ledger.record_limit_hit(code, msg, method)
                    raise VkApiError(code, msg, method, limit=True)
                if code in RETRY_CODES and attempt < 5:
                    pause = 8.0 if code == 9 else self.retry_sleep
                    time.sleep(pause * (attempt + 1))
                    continue
                raise VkApiError(code, msg, method)
            self._record(method, True)
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

    def _record(self, method: str, ok: bool) -> None:
        if self.ledger is not None:
            self.ledger.record(method, ok)

    def _throttle(self) -> None:
        now = time.monotonic()
        wait = self._last_call + self.delay - now
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()
