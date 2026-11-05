"""The PastVu api2 client: one HTTP client, retries with back-off, an optional
request cap, and the three things the tool asks the site for: a method call,
an image, and nothing else.

api2 takes ``?method=<name>&params=<JSON>`` over GET or POST and answers
``{"result": ..., "rid": ...}`` or ``{"type": ..., "code": ..., "message":
...}``. No key, no login, CORS open.
"""

from __future__ import annotations

import json
import random
import threading
import time

import httpx

API = "https://api.pastvu.com/api2"
IMG = "https://img.pastvu.com"
PAGE = "https://pastvu.com"
USER_AGENT = "dataset_tools-pastvu/0.1 (+https://github.com/aoleg/dataset_tools)"
TIMEOUT = httpx.Timeout(connect=20.0, read=180.0, write=20.0, pool=60.0)
# A stalled transfer held a fetch thread for the read timeout per read (a trickle: minutes),
# then the retries; one photo could hold a thread for twenty minutes. Now: 30 s without a
# byte, or 60 s for the whole image, and the photo is left for a later attempt.
READ_TIMEOUT = 30.0
IMAGE_DEADLINE = 60.0


class DeadlineExceeded(httpx.ReadTimeout):
    """An image download that took longer than the deadline in all."""
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}
MAX_RETRIES = 6
BACKOFF_BASE = 1.5                        # seconds; doubles each retry, with jitter, capped
BACKOFF_CAP = 90.0


class PastvuError(Exception):
    """An error the site reported for a method call (not a transport failure)."""

    def __init__(self, method: str, kind: str, code: str, message: str):
        super().__init__(f"{method}: {kind} {code}: {message}")
        self.method, self.kind, self.code, self.message = method, kind, code, message


class TransportError(Exception):
    """A request that failed after every retry."""


class Stopped(TransportError):
    """A request given up because the run is stopping (Client.stop is set): not a
    failure of the photo, which stays as it was for the next run."""


class RateCap:
    """A token bucket: at most `per_second` requests per second, off when None."""

    def __init__(self, per_second: float | None):
        self.rate = per_second
        self.lock = threading.Lock()
        self.tokens = float(per_second or 0)
        self.last = time.monotonic()

    def wait(self) -> None:
        if not self.rate:
            return
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.rate, self.tokens + (now - self.last) * self.rate)
                self.last = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                need = (1.0 - self.tokens) / self.rate
            time.sleep(need)


class Client:
    """Thread-safe: httpx.Client pools connections and may be shared by threads."""

    def __init__(self, max_per_second: float | None = None, connections: int = 8, user_agent: str = USER_AGENT,
                 read_timeout: float = READ_TIMEOUT, image_deadline: float = IMAGE_DEADLINE):
        """read_timeout: seconds without a byte after which a request is abandoned (and retried);
        image_deadline: seconds an image download may take in all, 0 for no limit."""
        self.http = httpx.Client(
            headers={"User-Agent": user_agent, "Accept": "application/json, image/*;q=0.9, */*;q=0.1"},
            timeout=httpx.Timeout(connect=TIMEOUT.connect, read=read_timeout, write=TIMEOUT.write, pool=TIMEOUT.pool),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=connections, max_keepalive_connections=connections),
        )
        self.image_deadline = image_deadline
        self.cap = RateCap(max_per_second)
        self.stats = {"requests": 0, "retries": 0, "bytes": 0}
        self.lock = threading.Lock()
        # Set by the stream to its stop event: once set, no request is retried (the back-off
        # doubles up to 90 s, so one refused download could hold a Ctrl+C for minutes).
        self.stop: threading.Event | None = None

    def stopping(self) -> bool:
        return self.stop is not None and self.stop.is_set()

    def close(self) -> None:
        self.http.close()

    # --- transport -------------------------------------------------------------

    def _download(self, method: str, url: str, deadline: float, **kw) -> httpx.Response:
        """A request read in chunks against a clock: DeadlineExceeded once it has taken
        longer than deadline seconds in all. The chunks are joined into a Response."""
        t0 = time.monotonic()
        with self.http.stream(method, url, **kw) as r:
            chunks = []
            for chunk in r.iter_bytes():
                chunks.append(chunk)
                if time.monotonic() - t0 > deadline:
                    raise DeadlineExceeded(f"took more than {deadline:.0f} s", request=r.request)
            return httpx.Response(r.status_code, headers=r.headers, content=b"".join(chunks), request=r.request)

    def _request(self, method: str, url: str, what: str, deadline: float = 0.0, **kw) -> httpx.Response:
        delay = BACKOFF_BASE
        last = ""
        for attempt in range(MAX_RETRIES + 1):
            self.cap.wait()
            with self.lock:
                self.stats["requests"] += 1
            try:
                r = self._download(method, url, deadline, **kw) if deadline > 0 else self.http.request(method, url, **kw)
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last = f"{type(e).__name__}: {e}"
            else:
                with self.lock:
                    self.stats["bytes"] += len(r.content)
                if r.status_code < 400:
                    return r
                last = f"HTTP {r.status_code}"
                if r.status_code not in RETRY_STATUS:
                    raise TransportError(f"{what}: {last}")
                retry_after = r.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = max(delay, float(retry_after))
            if attempt == MAX_RETRIES:
                break
            if self.stopping():
                raise Stopped(f"{what}: not retried, the run is stopping ({last})")
            with self.lock:
                self.stats["retries"] += 1
            wait = min(BACKOFF_CAP, delay) * (0.75 + random.random() * 0.5)
            if self.stop is not None:
                if self.stop.wait(wait):            # woken by the stop: give up at once
                    raise Stopped(f"{what}: not retried, the run is stopping ({last})")
            else:
                time.sleep(wait)
            delay = min(BACKOFF_CAP, delay * 2)
        raise TransportError(f"{what}: gave up after {MAX_RETRIES} retries ({last})")

    # --- the API ---------------------------------------------------------------

    def call(self, method: str, params: dict | None = None):
        """The ``result`` of an api2 method; PastvuError when the site refuses."""
        query = {"method": method, "params": json.dumps(params or {}, ensure_ascii=False, separators=(",", ":"))}
        r = self._request("GET", API, f"api2 {method}", params=query)
        ctype = r.headers.get("Content-Type", "")
        if "json" not in ctype:
            # a Cloudflare challenge or an outage page: treat as a transport failure, with back-off
            raise TransportError(f"api2 {method}: non-JSON answer ({ctype or 'no content type'})")
        d = r.json()
        if "result" in d:
            return d["result"]
        raise PastvuError(method, d.get("type", "Error"), d.get("code", ""), d.get("message", ""))

    def regions(self) -> list[dict]:
        return self.call("region.giveListPublic")["regions"]

    def give_ps(self, filter_: str, skip: int, limit: int = 100) -> dict:
        """One gallery page: {"count": n, "photos": [...], ...}."""
        return self.call("photo.givePS", {"filter": filter_, "skip": skip, "limit": limit})

    def count(self, filter_: str) -> int:
        return int(self.give_ps(filter_, 0, 1)["count"])

    def give_for_page(self, cid: int) -> dict:
        """The full record of a photo ({"photo": {...}, "can": {...}, "forEdit": ...})."""
        return self.call("photo.giveForPage", {"cid": cid})

    def comments(self, cid: int) -> dict:
        return self.call("comment.giveForObj", {"cid": cid})

    def image(self, file: str, variant: str = "a") -> tuple[bytes, str]:
        """The bytes of an image variant (a: original, d: standard, h: thumb) and its content type."""
        r = self._request("GET", f"{IMG}/{variant}/{file}", f"image {variant}/{file}", deadline=self.image_deadline)
        return r.content, r.headers.get("Content-Type", "")


def filter_string(regions: list[int] | None = None, years: tuple[int, int] | None = None, photos_only: bool = True,
                  geo: str | None = None, exclude: list[int] | None = None) -> str:
    """The gallery filter string: parts joined by '_', values by '!'."""
    parts = []
    if regions:
        parts.append("r!" + "!".join(str(int(c)) for c in regions))
    if exclude:
        parts.append("re!" + "!".join(str(int(c)) for c in exclude))
    if years:
        parts.append(f"y!{int(years[0])}!{int(years[1])}")
    if photos_only:
        parts.append("t!1")
    if geo in ("0", "1"):
        parts.append(f"geo!{geo}")
    return "_".join(parts)


def page_url(cid: int) -> str:
    return f"{PAGE}/{int(cid)}"
