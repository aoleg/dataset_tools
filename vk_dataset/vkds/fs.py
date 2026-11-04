"""Файловые утилиты: безопасные имена папок, атомарная запись JSON,
скачивание файлов с CDN ВК (не входит в лимиты API).
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
    """Безопасное имя папки/файла для Windows и Linux (кириллица сохраняется)."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name or ""))
    s = re.sub(r"\s+", " ", s).strip(" .")
    if len(s) > max_len:
        s = s[:max_len].rstrip(" .")
    if s.upper() in _WIN_RESERVED:
        s = "_" + s
    return s


def read_json(path: Path, default: Any = None) -> Any:
    """Читает JSON; при любой проблеме возвращает default."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    """Атомарная запись JSON (сначала .tmp, затем замена)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    tmp.replace(path)


def download_file(session, url: str, dest: Path, retries: int = 4,
                  timeout: float = 90.0, min_bytes: int = 1024,
                  retry_sleep: float = 1.5) -> dict:
    """Скачивает файл с CDN ВК (в лимиты API НЕ входит) с классификацией ошибок.

    Пишет во временный *.part и переименовывает только после полной загрузки —
    недокачанных файлов в датасете не остаётся.

    Возвращает словарь:
      status    — "ok" | "dead" | "forbidden" | "failed";
      http_code — код ответа (None при сетевой ошибке);
      bytes     — размер скачанного файла;
      error     — текст ошибки (для dead/forbidden/failed).

    ВАЖНО (частая проблема ВК): 404 — «мёртвое» фото, файл физически удалён
    с CDN, при этом API продолжает отдавать его метаданные. Все размеры одной
    фотографии — один физический файл, поэтому 404 НЕ повторяется ни разу и
    не имеет смысла пробовать другие размеры. 403 (доступ запрещён) тоже не
    повторяется — вызывающий код может попробовать копию меньшего размера.
    Повторяются только 429/5xx и сетевые сбои (с нарастающей паузой).
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
                if code == 404:            # файл удалён с CDN — навсегда
                    result.update(status="dead", http_code=404,
                                  error="файл удалён с CDN ВК (404)")
                    return result
                if code == 403:            # доступ запрещён — не повторяем
                    result.update(status="forbidden", http_code=403,
                                  error="доступ запрещён (403)")
                    return result
                if code != 200:
                    raise IOError("HTTP {}".format(code))
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(65536):
                        if chunk:
                            f.write(chunk)
            if tmp.stat().st_size < min_bytes:
                raise IOError("файл подозрительно мал")
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
                    break              # прочие 4xx бессмысленно повторять
            if attempt == retries - 1:
                break
            time.sleep(retry_sleep * (attempt + 1))
    return result
