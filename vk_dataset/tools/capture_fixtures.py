#!/usr/bin/env python3
"""Capture raw VK API responses as fixtures for the offline tests.

Run once by the partner on his machine (no live call is possible on the
maintainers' side), from the tool folder:

    ..\\venv\\Scripts\\python.exe tools\\capture_fixtures.py <group> [--out-dir tests\\fixtures] [--dry-run]

About 7 calls through VkClient, with the ledger of vkds.conf and a built-in
run cap of 8 attempts: groups.getById; the first page of photos.getAlbums;
one photos.get page (50 photos) of the largest album and one of the oldest
album (photos without dimensions); two wall.get pages of 100 posts
(filter=owner); and groups.search with the neutral query SEARCH_QUERY and
count=5. That last call settles whether the key may search groups at all:
the `search groups` verb depends on it, and VK documents groups.search for
user keys, while the partner runs a service key. An error answer (for
example error 28, the method is unavailable with this key) is written like
any other and is exactly the answer wanted. Each response is written as
<out-dir>/<method>_<n>.json as received (an error payload too), plus
_capture.json with the group ref, the date, the parameters of every call and
the call count. A file in which the token string occurs is never written.
What the script prints is Russian, for the partner (D2).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOL_DIR))

from vkds import __version__  # noqa: E402
from vkds import conf as vconf  # noqa: E402
from vkds.albums import parse_group_ref  # noqa: E402
from vkds.api import VkApiError, VkClient  # noqa: E402
from vkds.budget import Budget, BudgetExceeded, Ledger, now_iso  # noqa: E402

RUN_CAP = 8
PHOTOS_PAGE = 50
WALL_PAGE = 100
SEARCH_QUERY = "музей"
SEARCH_COUNT = 5


class TokenLeak(RuntimeError):
    """The token string occurs in what would be written."""


class _Recorder:
    """Wraps the HTTP session and keeps the last API payload as received."""

    def __init__(self, session):
        self._session = session
        self.last = None

    def post(self, url, data=None, timeout=None):
        resp = self._session.post(url, data=data, timeout=timeout)
        try:
            self.last = resp.json()
        except ValueError:
            self.last = None
        return resp

    def __getattr__(self, name):
        return getattr(self._session, name)


def plan_text(group: str) -> list[str]:
    return [
        "1) groups.getById group_id={} fields=description,members_count".format(group),
        "2) photos.getAlbums owner_id=-<id группы> offset=0 count=100 need_system=0",
        "3) photos.get из самого большого альбома: count={} photo_sizes=1 extended=1 rev=0".format(PHOTOS_PAGE),
        "4) photos.get из самого старого альбома (по дате создания), так же",
        "5) wall.get owner_id=-<id группы> filter=owner extended=0 count={} offset=0".format(WALL_PAGE),
        "6) wall.get то же, offset={} (если на стене больше {} постов)".format(WALL_PAGE, WALL_PAGE),
        "7) groups.search q={} count={} (можно ли этим ключом искать сообщества)".format(
            SEARCH_QUERY, SEARCH_COUNT),
    ]


def _dump(path: Path, data, token: str) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=1)
    if token and token in text:
        raise TokenLeak("в ответе для {} найден токен — файл не записан".format(path.name))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def capture(client: VkClient, group: str, out_dir: Path, token: str) -> dict:
    """Make the calls and write the files; return the _capture.json content."""
    out_dir = Path(out_dir)
    rec = _Recorder(client.session)
    client.session = rec
    counters: dict[str, int] = {}
    calls = []
    meta = {"group": group, "date": now_iso(), "vkds": __version__,
            "calls": calls, "call_count": 0}

    def call(method: str, **params):
        rec.last = None
        result, error = None, None
        try:
            result = client.call(method, _live=True, **params)
        except VkApiError as e:
            if e.limit:
                raise
            error = e
        counters[method] = counters.get(method, 0) + 1
        name = "{}_{}.json".format(method.replace(".", "_"), counters[method])
        if rec.last is not None:
            _dump(out_dir / name, rec.last, token)
        calls.append({"n": len(calls) + 1, "method": method, "params": params,
                      "file": name if rec.last is not None else None,
                      "error": str(error) if error else None})
        return result

    try:
        resp = call("groups.getById", group_id=parse_group_ref(group),
                    fields="description,members_count")
        groups = (resp or {}).get("groups") or (resp or {}).get("items") or []
        if not groups:
            raise VkApiError(0, "группа не найдена", "groups.getById")
        owner = -int(groups[0]["id"])
        albums = (call("photos.getAlbums", owner_id=owner, offset=0, count=100,
                       need_system=0, need_covers=0) or {}).get("items") or []
        albums = [a for a in albums if a.get("size")]
        if albums:
            largest = max(albums, key=lambda a: int(a.get("size") or 0))
            others = [a for a in albums if a["id"] != largest["id"]] or []
            picks = [largest]
            if others:
                picks.append(min(others, key=lambda a: int(a.get("created") or 0)))
            for a in picks:
                call("photos.get", owner_id=owner, album_id=str(a["id"]),
                     photo_sizes=1, extended=1, rev=0, offset=0, count=PHOTOS_PAGE)
        wall = call("wall.get", owner_id=owner, filter="owner", extended=0,
                    offset=0, count=WALL_PAGE)
        if wall and int(wall.get("count") or 0) > WALL_PAGE:
            call("wall.get", owner_id=owner, filter="owner", extended=0,
                 offset=WALL_PAGE, count=WALL_PAGE)
        call("groups.search", q=SEARCH_QUERY, count=SEARCH_COUNT)
    finally:
        meta["call_count"] = client.attempts
        _dump(out_dir / "_capture.json", meta, token)
    return meta


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(
        description="Снять сырые ответы VK API для офлайн-тестов (около 7 вызовов)")
    ap.add_argument("group", help="Короткое имя / id / ссылка открытого сообщества")
    ap.add_argument("--out-dir", default=str(TOOL_DIR / "tests" / "fixtures"),
                    help="Куда записать файлы (по умолчанию tests\\fixtures)")
    ap.add_argument("--token", help="Токен (иначе VK_TOKEN или файл token из vkds.conf)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Только показать вызовы, ничего не записывать")
    ns = ap.parse_args(argv)

    if ns.dry_run:
        print("Будут сделаны вызовы (не больше {} попыток):".format(RUN_CAP))
        for line in plan_text(ns.group):
            print("   " + line)
        print("Файлы: {}".format(Path(ns.out_dir).resolve()))
        print("Dry-run: ни одного вызова, ничего не записано.")
        return 0

    try:
        conf = vconf.load()
    except vconf.ConfError as e:
        print("[!] {}".format(e), file=sys.stderr)
        return 2
    token, _ = vconf.read_token(ns.token, conf)
    if not token:
        print("Не задан токен: --token, переменная окружения VK_TOKEN или файл {} "
              "([paths] token в vkds.conf).".format(conf.token_file), file=sys.stderr)
        return 2
    client = VkClient(token, delay=conf.delay, budget=Budget(None, run_limit=RUN_CAP),
                      ledger=Ledger(conf.ledger, monthly=conf.monthly))
    out_dir = Path(ns.out_dir).resolve()
    try:
        meta = capture(client, ns.group, out_dir, token)
    except TokenLeak as e:
        print("[!] Остановлено: {}.".format(e), file=sys.stderr)
        return 2
    except BudgetExceeded as e:
        print("[!] {}".format(e), file=sys.stderr)
        return 2
    except VkApiError as e:
        print("[!] Ошибка VK API: {}".format(e), file=sys.stderr)
        return 1
    errors = [c for c in meta["calls"] if c.get("error")]
    print("Вызовов API: {} (лимит запуска {}). Файлов: {}. Папка: {}".format(
        meta["call_count"], RUN_CAP, sum(1 for c in meta["calls"] if c.get("file")) + 1,
        out_dir))
    for c in errors:
        print("   [i] {}: {} (ответ с ошибкой тоже сохранён)".format(c["method"], c["error"]))
    print("Токена в файлах нет (проверено перед записью каждого файла).")
    print("Отправьте папку {} целиком (архивом zip) вручную, например почтой или "
          "в мессенджере. Больше ничего делать не нужно.".format(out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
