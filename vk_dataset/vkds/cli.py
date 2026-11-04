"""The vkds command line: photo datasets from VK groups.

Verbs (the cost in API calls in brackets):

  scan albums   the album index of a group (1 + 1 per 100 albums)
  scan wall     the wall index _wall.json: posts, texts, photos (1 per 100 posts)
  albums        download albums into <group title>/<album title>/ (full size,
                captions in photo<owner>_<id>.txt; 1 per 1000 photos listed)
  wall          download the photos of selected posts (wall.getById, 1 per 100 posts)
  peek          sample photos of an album for a look (1 or 2)
  whoami        check the token (1)
  budget        the ledger of this checkout and the dataset journal (0)
  recheck-dead  re-try the dead photos from the registry (0, CDN only)
  migrate       rename photo_<id> to photo<owner>_<id> (0)

Run as run.bat <verb> ... from anywhere, for example
  run.bat scan albums examplegroup
  run.bat albums examplegroup --album-id 123 --dry-run
Paths and limits come from vkds.conf next to vkds.py (vkds.conf.example is the
template): the token file, the datasets root base, the pause between calls.

The cache (<out>/_vkds/cache, 7 days) is read only by scan and --dry-run; a
real download lists live and writes it, and an album with nothing missing in
the manifest costs no photos.get. --refresh reads past the cache, --no-cache
turns it off. Everything the tool prints is Russian (D2). See README.md.
"""
from __future__ import annotations

import argparse
import sys

import requests

from . import __version__
from . import conf as vconf
from .albums import AlbumsOptions, run_albums, run_peek, run_scan_albums
from .api import VkApiError, VkClient
from .budget import Budget, BudgetExceeded, Ledger, today_utc
from .cache import ApiCache
from .migrate import MigrateError, run_migrate
from .recheck import RecheckError, run_recheck_dead
from .wall import ScanWallOptions, WallError, WallOptions, run_scan_wall, run_wall


def _read_token(explicit: str, conf: vconf.Conf) -> str:
    """Precedence: --token, then VK_TOKEN, then the file [paths] token of vkds.conf."""
    token, source = vconf.read_token(explicit, conf)
    if token and source not in ("--token", "VK_TOKEN"):
        print("Токен взят из файла {}".format(source))
    return token


def build_parser(conf: vconf.Conf | None = None) -> argparse.ArgumentParser:
    conf = conf or vconf.Conf()
    # The common flags work before and after the verb:
    #   run.bat --out D scan albums grp   or   run.bat scan albums grp --out D
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--token", default=argparse.SUPPRESS,
                        help="Ключ доступа VK API "
                             "(иначе VK_TOKEN или файл token из vkds.conf)")
    common.add_argument("--out", default=argparse.SUPPRESS,
                        help="Корневая папка датасета (относительная — "
                             "внутри base из vkds.conf)")
    common.add_argument("--delay", type=float, default=argparse.SUPPRESS,
                        help="Пауза между вызовами API, сек (~3 запроса/с)")
    common.add_argument("--budget", type=int, default=argparse.SUPPRESS,
                        metavar="N",
                        help="Жёсткий лимит вызовов API на этот запуск")
    common.add_argument("--no-cache", action="store_true",
                        default=argparse.SUPPRESS,
                        help="Отключить кэш ответов API")
    common.add_argument("--refresh", action="store_true",
                        default=argparse.SUPPRESS,
                        help="Игнорировать кэш при чтении (и обновить его)")

    p = argparse.ArgumentParser(
        prog="vkds",
        description="Сборщик фото-датасетов из ВКонтакте. Этап 1: альбомы "
                    "сообществ. Руководство: README.md.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-V", "--version", action="version",
                   version="vkds " + __version__)
    p.add_argument("--token", help="Ключ доступа VK API "
                    "(иначе переменная VK_TOKEN или файл token из vkds.conf)")
    p.add_argument("--out", default=None,
                   help="Корневая папка датасета; без --out — base из vkds.conf, "
                        "относительная — внутри base")
    p.add_argument("--delay", type=float, default=conf.delay,
                   help="Пауза между вызовами API, сек (~3 запроса/с)")
    p.add_argument("--budget", type=int, default=None, metavar="N",
                   help="Жёсткий лимит вызовов API на этот запуск (страховка)")
    p.add_argument("--no-cache", action="store_true", default=False,
                   help="Отключить кэш ответов API")
    p.add_argument("--refresh", action="store_true", default=False,
                   help="Игнорировать кэш при чтении (и обновить его)")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("albums", parents=[common],
                       help="Скачать фотоальбомы сообщества")
    a.add_argument("groups", nargs="+",
                   help="Короткое имя / id / ссылка сообщества (можно несколько)")
    a.add_argument("--album-id", dest="album_ids", action="append",
                   help="Скачать только этот альбом (можно несколько раз; "
                        "также wall/saved/profile)")
    a.add_argument("--title-regex",
                   help="Скачать только альбомы, чьё название/описание "
                        "подходит под регулярное выражение")
    a.add_argument("--include-wall", action="store_true",
                   help="Дополнительно скачать фото со стены сообщества")
    a.add_argument("--min-area", type=int, default=0, metavar="N",
                   help="Пропускать фото мельче N пикселей (307200 = 640x480)")
    a.add_argument("--min-side", type=int, default=0, metavar="N",
                   help="Пропускать фото, у которых любая сторона меньше N пикселей "
                        "(512 — порог импорта pastvu)")
    a.add_argument("--redownload-missing", action="store_true",
                   help="Скачать заново фото, чьи файлы перемещены из датасета "
                        "(по умолчанию они считаются обработанными)")
    a.add_argument("--workers", type=int, default=4,
                   help="Параллельных потоков загрузки файлов")
    a.add_argument("--dry-run", action="store_true",
                   help="Показать план скачивания, не тратя вызовы на фото")
    a.add_argument("--no-meta", action="store_true",
                   help="Не писать metadata.json, album.txt и photo<owner>_<id>.txt")
    a.add_argument("--recheck-dead", action="store_true",
                   help="Перепроверить мёртвые фото (404) из _dead_photos.json "
                        "— ВК иногда восстанавливает файлы")
    a.add_argument("--dir-pattern", default="{group_name}/{album}",
                   help="Шаблон папок: {group} {group_name} {group_id} "
                        "{album} {album_id}; по умолчанию папка группы — "
                        "НАЗВАНИЕ группы, альбома — название альбома")

    s = sub.add_parser("scan", parents=[common],
                       help="Разведка без скачивания (для ИИ-агента)")
    ssub = s.add_subparsers(dest="what", required=True)
    sa = ssub.add_parser("albums", parents=[common],
                         help="Список альбомов группы с размерами")
    sa.add_argument("group", help="Короткое имя / id / ссылка сообщества")
    sw = ssub.add_parser("wall", parents=[common],
                         help="Индекс стены группы в _wall.json (wall.get, 1 вызов на 100 постов)")
    sw.add_argument("group", help="Короткое имя / id / ссылка сообщества")
    sw.add_argument("--since", metavar="ГГГГ-ММ-ДД",
                    help="Листать до постов этой даты включительно (UTC)")
    sw.add_argument("--until", metavar="ГГГГ-ММ-ДД",
                    help="Считать в сводке посты не позже этой даты (UTC)")
    sw.add_argument("--limit", type=int, default=None, metavar="N",
                    help="Остановиться после N постов")
    sw.add_argument("--all", action="store_true",
                    help="Все записи стены (filter=all), а не только от имени группы")

    wl = sub.add_parser("wall", parents=[common],
                        help="Скачать фото отобранных постов стены (по _wall.json; "
                             "wall.getById, 1 вызов на 100 постов)")
    wl.add_argument("group", help="Короткое имя / id сообщества или папка группы "
                                  "(из _groups.json датасета)")
    wl.add_argument("--since", metavar="ГГГГ-ММ-ДД", help="Посты с этой даты (UTC, включительно)")
    wl.add_argument("--until", metavar="ГГГГ-ММ-ДД", help="Посты по эту дату (UTC, включительно)")
    wl.add_argument("--text-regex", default="",
                    help="Только посты, чей текст (или текст оригинала репоста) "
                         "подходит под регулярное выражение")
    wl.add_argument("--photos-min", type=int, default=1, metavar="N",
                    help="Только посты, где не меньше N фото")
    wl.add_argument("--post-id", dest="post_ids", action="append",
                    help="Только этот пост (можно несколько раз)")
    wl.add_argument("--min-area", type=int, default=0, metavar="N",
                    help="Пропускать фото мельче N пикселей (262144 = 512x512)")
    wl.add_argument("--min-side", type=int, default=0, metavar="N",
                    help="Пропускать фото, у которых любая сторона меньше N пикселей "
                         "(512 — порог импорта pastvu)")
    wl.add_argument("--redownload-missing", action="store_true",
                    help="Скачать заново фото, чьи файлы перемещены из датасета "
                         "(по умолчанию они считаются обработанными)")
    wl.add_argument("--workers", type=int, default=4,
                    help="Параллельных потоков загрузки файлов")
    wl.add_argument("--recheck-dead", action="store_true",
                    help="Перепроверить мёртвые фото (404) из _dead_photos.json")
    wl.add_argument("--dry-run", action="store_true",
                    help="Показать план по _wall.json, без вызовов API и без токена")

    k = sub.add_parser("peek", parents=[common],
                       help="Пробные фото альбома (визуальная оценка агентом)")
    k.add_argument("group", help="Короткое имя / id / ссылка сообщества")
    k.add_argument("--album-id", required=True)
    k.add_argument("--limit", type=int, default=6,
                   help="Сколько фото скачать как пробы")
    k.add_argument("--offset", type=int, default=0,
                   help="Смещение внутри альбома (начало/середина/конец)")
    k.add_argument("--min-side", type=int, default=0, metavar="N",
                   help="Не качать пробы, у которых любая сторона меньше N пикселей")

    w = sub.add_parser("whoami", parents=[common],
                       help="Проверить токен (1 вызов API)")
    w.add_argument("--group", default="1",
                   help="Любое сообщество для проверки токена")

    sub.add_parser("budget", parents=[common],
                   help="Журнал вызовов API этой копии и датасета (0 вызовов)")

    r = sub.add_parser("recheck-dead", parents=[common],
                       help="Перепроверить мёртвые фото по _dead_photos.json "
                            "(0 вызовов API, только CDN)")
    r.add_argument("group", help="Папка группы, её id или короткое имя "
                                 "(из _groups.json датасета)")
    r.add_argument("--limit", type=int, default=None, metavar="N",
                   help="Проверить не больше N записей реестра")
    r.add_argument("--sizes", action="store_true",
                   help="Пробовать все сохранённые размеры, а не только лучший")

    m = sub.add_parser("migrate", parents=[common],
                       help="Переименовать photo_<id>.* в photo<owner>_<id>.* "
                            "(0 вызовов API)")
    m.add_argument("dataset", help="Папка датасета или группы (относительная — "
                                   "внутри base из vkds.conf)")
    m.add_argument("--dry-run", action="store_true",
                   help="Только показать, что будет переименовано")
    return p


def _wall_options(ns) -> WallOptions:
    return WallOptions(since=ns.since, until=ns.until, text_regex=ns.text_regex or "",
                       photos_min=ns.photos_min, post_ids=[str(x) for x in (ns.post_ids or [])],
                       min_area=ns.min_area, min_side=ns.min_side,
                       redownload_missing=ns.redownload_missing,
                       workers=ns.workers, dry_run=ns.dry_run,
                       recheck_dead=ns.recheck_dead)


def _fmt(n: int) -> str:
    return "{:,}".format(int(n)).replace(",", " ")


def _print_ledger(ledger: Ledger) -> None:
    tot = ledger.period_total()
    rd = tot["reset_day"]
    print("Журнал вызовов этой копии инструмента: {}{}".format(
        ledger.path, "" if ledger.path and ledger.path.is_file() else " (пока пуст)"))
    print("Период с {} ({}): попыток {}, успешных {} из {} в месяц".format(
        tot["start"], "день сброса {}".format(rd) if rd else "начало месяца",
        _fmt(tot["attempts"]), _fmt(tot["ok"]), _fmt(ledger.monthly)))
    for name, m in sorted(tot["by_method"].items(), key=lambda x: -x[1]["attempts"]):
        print("   {:<28} попыток {}, успешных {}".format(name, m["attempts"], m["ok"]))
    hits = ledger.data.get("limit_hits") or []
    cleared = ledger.data.get("limit_cleared") or []
    if hits:
        print("Отметки лимита (limit_hit):")
        for h in hits[-10:]:
            print("   {} [{}] {}".format(h.get("ts", ""), h.get("code", ""),
                                        h.get("message", "")))
        if cleared:
            print("Первый успех после отметки (limit_cleared): {}".format(
                ", ".join(str(c) for c in cleared[-5:])))
    else:
        print("Отметок лимита нет.")
    if not rd:
        print("День сброса лимита не задан (поле reset_day в журнале; "
              "узнаётся по limit_hit и limit_cleared).")


def _print_totals(rep: dict) -> None:
    groups = rep.get("groups", [])
    if not groups or any(g.get("dry_run") for g in groups):
        return
    t = {"photos": 0, "downloaded": 0, "exists": 0, "small": 0,
         "failed": 0, "no_url": 0, "dead": 0, "forbidden": 0, "moved": 0}
    for g in groups:
        for k in t:
            t[k] += g.get(k, 0)
    print("\n==== ИТОГ ====")
    print("Групп: {} | альбомов: {} | фото: {} | скачано: {} | уже было: {} | "
          "мёртвых (404): {} | отфильтровано: {} | перемещено: {} | ошибок: {}".format(
              len(groups), sum(len(g.get("albums", [])) for g in groups),
              t["photos"], t["downloaded"], t["exists"], t["dead"],
              t["small"], t["moved"], t["failed"] + t["no_url"] + t["forbidden"]))


def main(argv=None, conf: vconf.Conf | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except Exception:
            pass
    try:
        conf = conf or vconf.load()
    except vconf.ConfError as e:
        print("[!] {}".format(e), file=sys.stderr)
        return 2
    ns = build_parser(conf).parse_args(argv)
    ledger = Ledger(conf.ledger, monthly=conf.monthly)

    # budget needs no token and makes no call; without --out and base it
    # prints the ledger of this checkout alone
    if ns.cmd == "budget":
        _print_ledger(ledger)
        try:
            out_root = vconf.resolve_out(ns.out, conf)
        except vconf.ConfError:
            return 0
        b = Budget(out_root / "_vkds" / "budget.json")
        print("\nЖурнал датасета: {}".format(b.path))
        print("Успешных вызовов API записано: {} (попыток: {}; "
              "месячный лимит без верификации — {})".format(
                  b.total, b.data.get("total_attempts", b.total), _fmt(conf.monthly)))
        for m, n in sorted(b.data.get("by_method", {}).items(), key=lambda x: -x[1]):
            print("   {:<28} {}".format(m, n))
        runs = b.data.get("runs", [])[-10:]
        if runs:
            print("Последние запуски:")
            for r in runs:
                print("   {} {:<12} вызовов: {}, попыток: {}".format(
                    r.get("ts", ""), r.get("label", ""), r.get("calls", 0),
                    r.get("attempts", r.get("calls", 0))))
        return 0

    try:
        out_root = vconf.resolve_out(
            ns.dataset if ns.cmd == "migrate" else ns.out, conf)
    except vconf.ConfError as e:
        print("[!] {}".format(e), file=sys.stderr)
        return 2

    # recheck-dead, migrate and wall --dry-run need no token and make no call
    if ns.cmd == "migrate":
        try:
            run_migrate(out_root, dry_run=ns.dry_run)
        except MigrateError as e:
            print("[!] {}".format(e), file=sys.stderr)
            return 2
        return 0
    if ns.cmd == "wall" and ns.dry_run:
        try:
            run_wall(None, ns.group, out_root, _wall_options(ns))
        except WallError as e:
            print("[!] {}".format(e), file=sys.stderr)
            return 2
        return 0
    if ns.cmd == "recheck-dead":
        session = requests.Session()
        session.headers["User-Agent"] = "vkds/{} (dataset harvester)".format(__version__)
        try:
            run_recheck_dead(session, out_root, ns.group, limit=ns.limit,
                             all_sizes=ns.sizes)
        except RecheckError as e:
            print("[!] {}".format(e), file=sys.stderr)
            return 2
        return 0

    token = _read_token(ns.token, conf)
    if not token:
        print("Не задан токен: --token, переменная окружения VK_TOKEN или файл "
              "{} ([paths] token в vkds.conf; настройка — README.md).".format(
                  conf.token_file),
              file=sys.stderr)
        return 2

    budget = Budget(out_root / "_vkds" / "budget.json",
                    run_limit=ns.budget, label=ns.cmd)
    cache = None if ns.no_cache else ApiCache(out_root / "_vkds" / "cache")
    client = VkClient(token, delay=ns.delay, budget=budget, cache=cache,
                      refresh=ns.refresh, ledger=ledger)
    rc = 0
    try:
        if ns.cmd == "albums":
            opts = AlbumsOptions(
                album_ids=[str(x) for x in (ns.album_ids or [])],
                title_regex=ns.title_regex or "",
                include_wall=ns.include_wall,
                min_area=ns.min_area,
                min_side=ns.min_side,
                redownload_missing=ns.redownload_missing,
                workers=ns.workers,
                dry_run=ns.dry_run,
                no_meta=ns.no_meta,
                dir_pattern=ns.dir_pattern,
                recheck_dead=ns.recheck_dead)
            rep = run_albums(client, ns.groups, out_root, opts)
            _print_totals(rep)
        elif ns.cmd == "scan" and getattr(ns, "what", None) == "albums":
            run_scan_albums(client, ns.group, out_root)
        elif ns.cmd == "scan" and getattr(ns, "what", None) == "wall":
            run_scan_wall(client, ns.group, out_root, ScanWallOptions(
                since=ns.since, until=ns.until, limit=ns.limit,
                include_all=ns.all, refresh=ns.refresh))
        elif ns.cmd == "wall":
            run_wall(client, ns.group, out_root, _wall_options(ns))
        elif ns.cmd == "peek":
            run_peek(client, ns.group, ns.album_id, ns.limit, ns.offset, out_root,
                     min_side=ns.min_side)
        elif ns.cmd == "whoami":
            resp = client.call("groups.getById", group_id=ns.group)
            g = (resp.get("groups") or [{}])[0]
            print("Токен работает. Проверочное сообщество: {} (id={})".format(
                g.get("name", "?"), g.get("id", "?")))
        else:
            print("Неизвестная команда", file=sys.stderr)
            rc = 1
    except WallError as e:
        print("\n[!] {}".format(e), file=sys.stderr)
        rc = 2
    except BudgetExceeded as e:
        print("\n[!] {}".format(e))
        print("Выполненное до остановки сохранено; повторный запуск докачает.")
        rc = 2
    except VkApiError as e:
        if e.limit:
            print("\n[!] Достигнут лимит VK API: [{}] {}. Отметка limit_hit "
                  "записана в журнал {} ({} UTC). Повторите после сброса "
                  "лимита; расход — run.bat budget.".format(
                      e.code, e.message, ledger.path, today_utc().isoformat()),
                  file=sys.stderr)
            rc = 2
        else:
            print("\n[!] Ошибка VK API: {}".format(e), file=sys.stderr)
            rc = 1
    except KeyboardInterrupt:
        print("\n[!] Прервано. Скачанное сохранено; повторный запуск докачает.")
        rc = 130
    except Exception as e:  # unexpected: the agent needs a readable refusal
        print("\n[!] Непредвиденная ошибка: {}".format(e), file=sys.stderr)
        rc = 1
    finally:
        budget.finish_run(ok=(rc == 0))
    print("\nВызовов API за запуск: {} (попыток {}); из кэша бесплатно: {}; "
          "всего в журнале датасета: {} (файлы с CDN и кэш — бесплатно, в "
          "лимит не входят)".format(client.api_calls, client.attempts,
                                    client.cache_hits, budget.total))
    return rc


if __name__ == "__main__":
    sys.exit(main())
