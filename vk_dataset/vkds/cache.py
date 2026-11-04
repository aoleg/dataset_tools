"""Cache of VK API responses for scan and --dry-run.

Only the heavy listing methods are cached (photos.getAlbums, photos.get).
groups.getById is never cached: one live call per run checks that the group
still exists (so whoami stays an honest token check and a rerun notices a
renamed group).

Files: <dataset>/_vkds/cache/<method>/<sha1(method + parameters)>.json with
ts, ts_epoch, params and response. The default TTL is 7 days.

Cache hits cost no quota and are not written to budget.json. Only scan and
--dry-run read the cache (D7): a real download lists live (the copy URLs are
signed and may expire) and writes the cache; a control rerun costs
groups.getById plus photos.getAlbums, and an album with nothing missing is
reconciled from the manifest without photos.get.

--refresh reads past the cache (and rewrites its entries), --no-cache turns it
off (no file is read or written).
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from .budget import now_iso
from .fs import read_json, write_json

CACHEABLE = {"photos.getAlbums", "photos.get"}
DEFAULT_TTL_DAYS = 7.0


class ApiCache:
    """A plain disk cache of the responses of the listing methods."""

    def __init__(self, root: Path, ttl_days: float = DEFAULT_TTL_DAYS):
        self.root = Path(root)
        self.ttl = max(0.0, float(ttl_days)) * 86400.0
        self.hits = 0      # responses served from the cache (for the report)
        self.writes = 0

    @staticmethod
    def _key(method: str, params: dict) -> str:
        """The key: sha1 of the method and the sorted parameters."""
        norm = "&".join("{}={}".format(k, params[k]) for k in sorted(params))
        return hashlib.sha1("{}|{}".format(method, norm)
                            .encode("utf-8")).hexdigest()

    def _path(self, method: str, key: str) -> Path:
        return self.root / method.replace(".", "_") / (key + ".json")

    def get(self, method: str, params: dict) -> Any | None:
        """The cached response, or None (no file, TTL passed, not cacheable)."""
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
        """Store a successful response (cacheable methods only)."""
        if method not in CACHEABLE:
            return
        p = self._path(method, self._key(method, params))
        write_json(p, {"ts": now_iso(), "ts_epoch": time.time(),
                       "params": params, "response": response})
        self.writes += 1
