"""File helpers: safe folder names, atomic JSON writes, and downloads from
VK's CDN (which cost no API quota).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

_WIN_RESERVED = ({"CON", "PRN", "AUX", "NUL"}
                 | {"COM{}".format(i) for i in range(1, 10)}
                 | {"LPT{}".format(i) for i in range(1, 10)})


def sanitize_dirname(name: str, max_len: int = 100) -> str:
    """A folder or file name safe on Windows and Linux (Cyrillic is kept)."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name or ""))
    s = re.sub(r"\s+", " ", s).strip(" .")
    if len(s) > max_len:
        s = s[:max_len].rstrip(" .")
    if s.upper() in _WIN_RESERVED:
        s = "_" + s
    return s


def read_json(path: Path, default: Any = None) -> Any:
    """Read JSON; return `default` on any problem."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    """Write JSON atomically (a .tmp first, then a replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    tmp.replace(path)


def download_file(session, url: str, dest: Path, retries: int = 4,
                  timeout: float = 90.0, min_bytes: int = 1024,
                  retry_sleep: float = 1.5) -> dict:
    """Download one file from VK's CDN (no API quota) and classify the outcome.

    The file is written to a *.part and renamed only when complete, so no
    partial file is left in a dataset.

    Returns a dict:
      status    "ok", "dead", "forbidden" or "failed";
      http_code the response code (None on a network error);
      bytes     the size of the file;
      error     the error text (for dead, forbidden and failed).

    A 404 means the file is gone from the CDN while the API still returns the
    photo's metadata, a common VK case. A 404 is not retried; the callers
    (albums.download_photos, recheck-dead) try the other sizes, which is free,
    and a photo counts as dead only when every size answers 404. A 403 is not
    retried either. Only 429, 5xx and network failures are retried, with a
    growing pause.
    """
    dest = Path(dest)
    result = {"status": "failed", "http_code": None, "bytes": 0, "error": None}
    if dest.exists() and dest.stat().st_size >= min_bytes:
        result.update(status="ok", bytes=dest.stat().st_size)
        return result
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(retries):
        try:
            with session.get(url, stream=True, timeout=timeout) as r:
                code = r.status_code
                if code == 404:            # gone from the CDN for good
                    result.update(status="dead", http_code=404,
                                  error="файл удалён с CDN ВК (404)")
                    return result
                if code == 403:            # forbidden: no retry
                    result.update(status="forbidden", http_code=403,
                                  error="доступ запрещён (403)")
                    return result
                if code != 200:
                    raise OSError("HTTP {}".format(code))
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(65536):
                        if chunk:
                            f.write(chunk)
            if tmp.stat().st_size < min_bytes:
                raise OSError("файл подозрительно мал")
            tmp.replace(dest)
            result.update(status="ok", http_code=200, bytes=dest.stat().st_size)
            return result
        except Exception as e:
            try:
                tmp.unlink()
            except OSError:
                pass
            result["error"] = str(e)
            m = re.search(r"HTTP (\d+)", str(e))
            if m:
                code = int(m.group(1))
                result["http_code"] = code
                if code != 429 and code < 500:
                    break              # other 4xx: no point retrying
            if attempt == retries - 1:
                break
            time.sleep(retry_sleep * (attempt + 1))
    return result
