"""CLI vkds: сборщик фото-датасетов из ВКонтакте.

Этап 1 (текущий) — альбомы сообществ. Глаголы:

  albums        скачать альбомы группы в <название группы>/<название
                альбома>/ (полный размер, описания, photo_<id>.txt)
  scan albums   разведка: список альбомов без скачивания (для ИИ-агента)
  peek          пробные фото альбома для визуальной оценки агентом
  whoami        проверить токен (1 вызов API)
  budget        показать расход вызовов API (0 вызовов)

Запуск: run.bat <глагол> ... из папки инструмента или откуда угодно, например
  run.bat scan albums examplegroup
  run.bat albums examplegroup --album-id 123 --dry-run
Пути и лимиты — vkds.conf рядом с vkds.py (образец — vkds.conf.example):
токен, папка датасетов base, пауза между вызовами.

Кэш: photos.getAlbums/photos.get кэшируются в <out>/_vkds/cache (TTL 7
дней) — повторные и контрольные прогоны почти бесплатны. --refresh читает
мимо кэша, --no-cache отключает его.
Руководство — README.md.
"""
from __future__ import annotations

import argparse
import sys

from . import __version__
from . import conf as vconf
from .albums import AlbumsOptions, run_albums, run_peek, run_scan_albums
from .api import VkApiError, VkClient
from .budget import Budget, BudgetExceeded
from .cache import ApiCache


def _read_token(explicit: str, conf: vconf.Conf) -> str:
    """Приоритет: --token > переменная VK_TOKEN > файл [paths] token из vkds.conf."""
    token, source = vconf.read_token(explicit, conf)
    if token and source not in ("--token", "VK_TOKEN"):
        print("Токен взят из файла {}".format(source))
    return token


def build_parser(conf: vconf.Conf | None = None) -> argparse.ArgumentParser:
    conf = conf or vconf.Conf()
    # Общие флаги разрешены и ДО, и ПОСЛЕ глагола:
    #   run.bat --out D scan albums grp   или   run.bat scan albums grp --out D
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
    a.add_argument("--workers", type=int, default=4,
                   help="Параллельных потоков загрузки файлов")
    a.add_argument("--dry-run", action="store_true",
                   help="Показать план скачивания, не тратя вызовы на фото")
    a.add_argument("--no-meta", action="store_true",
                   help="Не писать metadata.json, album.txt и photo_<id>.txt")
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

    k = sub.add_parser("peek", parents=[common],
                       help="Пробные фото альбома (визуальная оценка агентом)")
    k.add_argument("group", help="Короткое имя / id / ссылка сообщества")
    k.add_argument("--album-id", required=True)
    k.add_argument("--limit", type=int, default=6,
                   help="Сколько фото скачать как пробы")
    k.add_argument("--offset", type=int, default=0,
                   help="Смещение внутри альбома (начало/середина/конец)")

    w = sub.add_parser("whoami", parents=[common],
                       help="Проверить токен (1 вызов API)")
    w.add_argument("--group", default="1",
                   help="Любое сообщество для проверки токена")

    sub.add_parser("budget", parents=[common],
                   help="Показать журнал расхода вызовов API (0 вызовов)")
    return p


def _print_totals(rep: dict) -> None:
    groups = rep.get("groups", [])
    if not groups or any(g.get("dry_run") for g in groups):
        return
    t = {"photos": 0, "downloaded": 0, "exists": 0, "small": 0,
         "failed": 0, "no_url": 0, "dead": 0, "forbidden": 0}
    for g in groups:
        for k in t:
            t[k] += g.get(k, 0)
    print("\n==== ИТОГ ====")
    print("Групп: {} | альбомов: {} | фото: {} | скачано: {} | уже было: {} | "
          "мёртвых (404): {} | отфильтровано: {} | ошибок: {}".format(
              len(groups), sum(len(g.get("albums", [])) for g in groups),
              t["photos"], t["downloaded"], t["exists"], t["dead"],
              t["small"], t["failed"] + t["no_url"] + t["forbidden"]))


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
    try:
        out_root = vconf.resolve_out(ns.out, conf)
    except vconf.ConfError as e:
        print("[!] {}".format(e), file=sys.stderr)
        return 2

    # budget не требует токена и не делает вызовов
    if ns.cmd == "budget":
        b = Budget(out_root / "_vkds" / "budget.json")
        print("Файл журнала: {}".format(b.path))
        print("Всего вызовов API записано: {} "
              "(месячный лимит без верификации — {})".format(
                  b.total, "{:,}".format(conf.monthly).replace(",", " ")))
        for m, n in sorted(b.data.get("by_method", {}).items(), key=lambda x: -x[1]):
            print("   {:<28} {}".format(m, n))
        runs = b.data.get("runs", [])[-10:]
        if runs:
            print("Последние запуски:")
            for r in runs:
                print("   {} {:<10} вызовов: {}".format(
                    r.get("ts", ""), r.get("label", ""), r.get("calls", 0)))
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
                      refresh=ns.refresh)
    rc = 0
    try:
        if ns.cmd == "albums":
            opts = AlbumsOptions(
                album_ids=[str(x) for x in (ns.album_ids or [])],
                title_regex=ns.title_regex or "",
                include_wall=ns.include_wall,
                min_area=ns.min_area,
                workers=ns.workers,
                dry_run=ns.dry_run,
                no_meta=ns.no_meta,
                dir_pattern=ns.dir_pattern,
                recheck_dead=ns.recheck_dead)
            rep = run_albums(client, ns.groups, out_root, opts)
            _print_totals(rep)
        elif ns.cmd == "scan" and getattr(ns, "what", None) == "albums":
            run_scan_albums(client, ns.group, out_root)
        elif ns.cmd == "peek":
            run_peek(client, ns.group, ns.album_id, ns.limit, ns.offset, out_root)
        elif ns.cmd == "whoami":
            resp = client.call("groups.getById", group_id=ns.group)
            g = (resp.get("groups") or [{}])[0]
            print("Токен работает. Проверочное сообщество: {} (id={})".format(
                g.get("name", "?"), g.get("id", "?")))
        else:
            print("Неизвестная команда", file=sys.stderr)
            rc = 1
    except BudgetExceeded as e:
        print("\n[!] {}".format(e))
        print("Выполненное до остановки сохранено; повторный запуск докачает.")
        rc = 2
    except VkApiError as e:
        print("\n[!] Ошибка VK API: {}".format(e), file=sys.stderr)
        rc = 1
    except KeyboardInterrupt:
        print("\n[!] Прервано. Скачанное сохранено; повторный запуск докачает.")
        rc = 130
    except Exception as e:  # непредвиденное — агенту нужен читаемый отказ
        print("\n[!] Непредвиденная ошибка: {}".format(e), file=sys.stderr)
        rc = 1
    finally:
        budget.finish_run(ok=(rc == 0))
    print("\nВызовов API за запуск: {}; из кэша бесплатно: {}; всего в журнале: "
          "{} (файлы с CDN и кэш — бесплатно, в лимит не входят)".format(
              client.api_calls, client.cache_hits, budget.total))
    return rc


if __name__ == "__main__":
    sys.exit(main())
