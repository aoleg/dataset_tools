"""Этап 1: сканирование и скачивание фотоальбомов сообщества ВК.

Структура датасета (с версии 0.3, имена файлов с 0.5):
  <out>/<папка группы — по НАЗВАНИЮ группы>/<папка альбома — по НАЗВАНИЮ>/photo<owner>_<id>.jpg

Имя файла — форма ВК photo<owner_id>_<photo_id> с владельцем со знаком
(photo-9642436_457239017 у фото группы, photo123_456 у фото пользователя):
уникально между группами. Старые photo_<id>.jpg (до 0.5) при докачке
считаются существующими; переименование — глагол migrate.

Рядом с фото и альбомами:
  photo<owner>_<id>.txt — очищенная подпись фото (без ссылок, хэштегов и
                    упоминаний), имя как у фото — пары «картинка-текст»;
  album.txt       — название, описание, даты и статистика выгрузки альбома;
  metadata.json   — машинные данные всех фото (тексты, даты, гео, лайки).

На уровне папки группы:
  _manifest.csv     — таблица ВСЕХ фото (включая мёртвые) со статусами —
                      основа для сборки датасета;
  _dead_photos.json — реестр мёртвых фото (404 на CDN ВК) с полными
                      метаданными и ссылками на все размеры (для попыток
                      восстановления);
  _albums.json      — индекс альбомов; _group.json — данные группы.

Мёртвые фото (файл удалён с CDN ВК, но API по-прежнему отдаёт метаданные) —
частая проблема ВК: 404 на лучшей копии -> пробуются все остальные размеры
(бесплатно); только если все ответили 404, фото попадает в реестр, и при
следующих запусках CDN вообще не дёргается (TTL 30 дней; --recheck-dead и
глагол recheck-dead перепроверяют — ВК иногда восстанавливает файлы).

Списки при скачивании (D7): настоящий прогон берёт photos.getAlbums и
photos.get только живыми (и пишет кэш); кэш читают лишь scan и --dry-run.
Альбом, у которого по манифесту и диску ничего не недостаёт, пропускается
без единого вызова photos.get.

Совпадающие названия разных альбомов получают суффикс "__<id>"; раннее
назначенные папки при докачке не меняются. Датасеты vkds 0.2 (папка группы
по screen_name) автоматически мигрируются: докачка идёт в прежние папки.

Команды для ИИ-агента (дешёвая разведка -> решение -> скачивание):
  run_scan_albums() — список альбомов с размерами, без скачивания;
  run_peek()        — пробные фото альбома для визуальной оценки;
  run_albums()      — скачивание отобранных альбомов.
"""
from __future__ import annotations

import csv
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .api import VkApiError, VkClient
from .budget import now_iso
from .captions import clean_caption
from .fs import download_file, read_json, sanitize_dirname, write_json
from .models import AlbumInfo, GroupInfo

ALBUMS_PER_PAGE = 100            # максимум photos.getAlbums
PHOTOS_PER_PAGE = 1000           # максимум photos.get
SIZE_TYPE_ORDER = "smxopqryzw"   # возрастание размера копий (w — максимальный)
SPECIAL_ALBUMS = {
    "wall": "Фотографии со стены",
    "saved": "Сохранённые фотографии",
    "profile": "Фотографии профиля",
}
DEFAULT_DIR_PATTERN = "{group_name}/{album}"
DEAD_TTL_DAYS = 30               # сколько дней верим реестру мёртвых фото

MANIFEST_NAME = "_manifest.csv"
MANIFEST_COLS = [
    "photo_id", "owner_id", "group_id", "group_name", "group_screen",
    "album_id", "album_title", "album_dir", "file", "path", "txt",
    "status", "size_type", "width", "height", "bytes", "date",
    "likes", "comments", "text",
    # wall rows (phase 3); album rows leave them empty
    "post_id", "post_date", "repost_from", "reposts",
    # the first date a row's file was seen gone from its path (moved away)
    "moved",
]


@dataclass
class AlbumsOptions:
    """Параметры скачивания альбомов."""

    album_ids: list = field(default_factory=list)  # только эти альбомы (id строками)
    title_regex: str = ""            # фильтр по названию/описанию альбома
    include_wall: bool = False       # системный альбом «фото со стены»
    min_area: int = 0                # пропускать фото мельче N пикселей
    min_side: int = 0                # пропускать фото, у которых сторона меньше N
    redownload_missing: bool = False  # fetch again the photos whose file was moved away
    workers: int = 4                 # параллельных потоков загрузки CDN
    dry_run: bool = False            # только показать план
    no_meta: bool = False            # не писать metadata.json / album.txt / .txt
    dir_pattern: str = DEFAULT_DIR_PATTERN
    dl_retry_sleep: float = 1.5      # пауза между повторами загрузки файла
    recheck_dead: bool = False       # перепроверять известные мёртвые фото
    dead_ttl_days: int = DEAD_TTL_DAYS


# ----------------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------------

def photo_stem(owner_id, photo_id) -> str:
    """VK's own id form: photo-9642436_457239017 (group), photo123_456 (user)."""
    return "photo{}_{}".format(int(owner_id), photo_id)


def old_stem(photo_id) -> str:
    """The file stem before 0.5."""
    return "photo_{}".format(photo_id)


def write_caption(dest_dir: Path, stem: str, text: str) -> str:
    """Write <stem>.txt with the cleaned caption unless one exists; return the
    name of the .txt that exists afterwards, or "".

    An existing .txt is never rewritten or deleted: a sidecar may have been
    edited by hand or replaced by autocaption, and a resume must not clobber
    it (the rule of the pastvu import). An empty cleaned caption writes
    nothing. Callers pass only images that were downloaded in this run or
    exist on disk.
    """
    path = Path(dest_dir) / (stem + ".txt")
    if path.is_file():
        return path.name
    cap = clean_caption(text)
    if not cap:
        return ""
    try:
        path.write_text(cap, encoding="utf-8")
        return path.name
    except OSError:
        return ""


def size_score(s: dict) -> tuple:
    """Sort key of a copy: area when both dimensions are known, else the
    letter of its type (s < m < x < ... < w). Larger is better."""
    w = s.get("width") or 0
    h = s.get("height") or 0
    if w > 0 and h > 0:
        return (2, w * h)
    t = SIZE_TYPE_ORDER.find(str(s.get("type", "")))
    return (1, t if t >= 0 else -1)


def size_candidates(photo: dict, limit: int = 64) -> list:
    """Копии фото, отсортированные по убыванию размера (первая — лучшая).

    У фото, загруженных до 2012 года, width/height бывают нулевыми — тогда
    размер сравнивается по типу копии (s < m < x < ... < w, w до 2560x2048).
    """
    sizes = [s for s in (photo.get("sizes") or []) if s.get("url")]
    return sorted(sizes, key=size_score, reverse=True)[:max(1, limit)]


def too_small(w, h, min_area: int = 0, min_side: int = 0) -> bool:
    """True when a copy of known size is under --min-area or has a side under
    --min-side (the import's gate is a side of 512). Unknown sizes pass."""
    try:
        w, h = int(float(w or 0)), int(float(h or 0))
    except (TypeError, ValueError):
        return False
    if not (w > 0 and h > 0):
        return False
    return bool((min_area and w * h < min_area) or (min_side and min(w, h) < min_side))


def pick_best_size(photo: dict) -> dict | None:
    """Выбирает копию фото максимального размера из массива sizes."""
    cands = size_candidates(photo, 1)
    return cands[0] if cands else None


def iso_date(unix) -> str | None:
    if not unix:
        return None
    return datetime.fromtimestamp(int(unix), tz=timezone.utc).isoformat()


def parse_group_ref(value: str) -> str:
    """Короткое имя / id / ссылка сообщества -> ref для groups.getById."""
    g = str(value or "").strip()
    for dom in ("vk.com/", "vk.ru/", "m.vk.com/", "new.vk.com/"):
        if dom in g:
            g = g.split(dom, 1)[1].split("?", 1)[0].strip("/")
            break
    if "/" in g:  # напр. vk.ru/examplegroup/photos или /album… — берём первый сегмент
        g = g.split("/", 1)[0]
    g = g.lstrip("-")
    for prefix in ("public", "club", "event", "group"):
        if g.startswith(prefix) and g[len(prefix):].isdigit():
            return g[len(prefix):]
    return g


def resolve_groups(client: VkClient, targets) -> list[GroupInfo]:
    """Определяет сообщества по именам/ссылкам. Все группы — одним вызовом API."""
    refs = []
    for t in targets:
        r = parse_group_ref(t)
        if r and r not in refs:
            refs.append(r)
    if not refs:
        return []

    def _to_groups(items):
        out = []
        for g in items or []:
            out.append(GroupInfo(
                id=int(g["id"]),
                name=str(g.get("name", "")),
                screen_name=str(g.get("screen_name") or ""),
                description=str(g.get("description") or ""),
                members_count=g.get("members_count"),
            ))
        return out

    try:
        resp = client.call("groups.getById", group_id=",".join(refs),
                           fields="description,members_count")
        groups = _to_groups(resp.get("groups") or resp.get("items"))
    except VkApiError as e0:
        if e0.limit:
            raise
        # одна из ссылок битая — определяем по одной, плохие пропускаем
        groups, last = [], None
        for r in refs:
            try:
                resp = client.call("groups.getById", group_id=r,
                                   fields="description,members_count")
                groups.extend(_to_groups(resp.get("groups") or resp.get("items")))
            except VkApiError as e:
                if e.limit:
                    raise
                last = e
                print("[!] Сообщество «{}» не найдено: {}".format(r, e))
        if not groups and last is not None:
            raise last
    seen, uniq = set(), []
    for g in groups:
        if g.id not in seen:
            seen.add(g.id)
            uniq.append(g)
    return uniq


def fetch_albums(client: VkClient, group: GroupInfo,
                 live: bool = False) -> list[AlbumInfo]:
    """Полный список обычных альбомов сообщества (постранично, 100/вызов).

    live=True (настоящий прогон, D7) — мимо кэша, ответ пишется в кэш.
    """
    albums, offset = [], 0
    while True:
        resp = client.call("photos.getAlbums", _live=live, owner_id=group.owner_id,
                           offset=offset, count=ALBUMS_PER_PAGE,
                           need_system=0, need_covers=0)
        items = resp.get("items", [])
        albums.extend(items)
        offset += len(items)
        total = resp.get("count")
        if not items or (total is not None and offset >= int(total)):
            break
    out = []
    for a in albums:
        out.append(AlbumInfo(
            id=str(a["id"]),
            title=str(a.get("title") or ""),
            description=str(a.get("description") or ""),
            size=int(a.get("size") or 0),
            created=a.get("created"),
            updated=a.get("updated"),
        ))
    return out


def fetch_photos(client: VkClient, group: GroupInfo, album_id: str,
                 live: bool = True) -> list[dict]:
    """Все фотографии одного альбома (постранично, до 1000/вызов).

    Для скачивания список всегда живой (D7): URL копий подписаны и могут
    устаревать.
    """
    photos, offset = [], 0
    while True:
        resp = client.call("photos.get", _live=live, owner_id=group.owner_id,
                           album_id=str(album_id), photo_sizes=1, extended=1,
                           rev=0, offset=offset, count=PHOTOS_PER_PAGE)
        items = resp.get("items", [])
        photos.extend(items)
        offset += len(items)
        total = resp.get("count")
        if not items or (total is not None and offset >= int(total)):
            break
    return photos


def make_record(photo: dict) -> dict:
    """Фото из ответа API -> плоская запись для metadata.json / пайплайна."""
    cands = size_candidates(photo)
    best = cands[0] if cands else None
    return {
        "id": photo.get("id"),
        "owner_id": photo.get("owner_id"),
        "date_unix": photo.get("date"),
        "date": iso_date(photo.get("date")),
        "text": str(photo.get("text") or ""),
        "lat": photo.get("lat"),
        "long": photo.get("long"),
        "likes": (photo.get("likes") or {}).get("count"),
        "comments": (photo.get("comments") or {}).get("count"),
        "url": best.get("url") if best else None,
        "alt_urls": [s["url"] for s in cands[1:]],   # все копии меньшего размера
        "alt_sizes": [{k: s.get(k) for k in ("type", "width", "height")}
                      for s in cands[1:]],
        "width": best.get("width") if best else None,
        "height": best.get("height") if best else None,
        "size_type": best.get("type") if best else None,
    }


def select_albums(albums: list[AlbumInfo], opts: AlbumsOptions) -> list[AlbumInfo]:
    """Фильтрует альбомы по --album-id / --title-regex, добавляет системные."""
    wanted = {str(a) for a in (opts.album_ids or [])}
    rx = re.compile(opts.title_regex, re.IGNORECASE) if opts.title_regex else None
    sel = []
    for a in albums:
        if wanted and a.id not in wanted:
            continue
        if rx and not (rx.search(a.title) or rx.search(a.description)):
            continue
        sel.append(a)
    # Системные альбомы (wall и др.) отсутствуют в обычном списке
    for key, title in SPECIAL_ALBUMS.items():
        if (wanted and key in wanted) or \
           (key == "wall" and opts.include_wall and not wanted):
            sel.append(AlbumInfo(id=key, title=title, size=None, special=True))
    return sel


# ----------------------------------------------------------------------------
# Реестр мёртвых фото (404 на CDN ВК)
# ----------------------------------------------------------------------------

def load_dead(gdir: Path) -> dict:
    """Читает _dead_photos.json группы: {"<owner>_<photo_id>": entry, ...}."""
    data = read_json(Path(gdir) / "_dead_photos.json", None)
    return data if isinstance(data, dict) else {}


def save_dead(gdir: Path, dead: dict) -> None:
    write_json(Path(gdir) / "_dead_photos.json", dead)


def make_dead_entry(group: GroupInfo, album: AlbumInfo, photo: dict,
                    http_code: int = 404, urls_live: bool = True) -> dict:
    """Полная карточка мёртвого фото: всё, что осталось кроме самого файла.

    Сохраняются подпись, дата, лайки, ссылки на ВСЕ размеры (пригодятся для
    попыток восстановления через web.archive.org или повторной проверки).
    """
    best = pick_best_size(photo)
    owner = photo.get("owner_id", group.owner_id)
    pid = photo.get("id")
    return {
        "photo_id": pid,
        "owner_id": owner,
        "group_id": group.id,
        "album_id": str(album.id),
        "album_title": album.title,
        "date": iso_date(photo.get("date")),
        "text": str(photo.get("text") or ""),
        "likes": (photo.get("likes") or {}).get("count"),
        "comments": (photo.get("comments") or {}).get("count"),
        "best_size": ({k: best.get(k) for k in ("type", "width", "height", "url")}
                      if best else None),
        "sizes": [{k: s.get(k) for k in ("type", "width", "height", "url")}
                  for s in (photo.get("sizes") or []) if s.get("url")],
        "link": "https://vk.com/photo{}_{}".format(owner, pid),
        "http_status": http_code,
        "urls_live": urls_live,   # every size answered 404 on URLs of a live listing
        "first_seen_dead": now_iso(),
        "last_checked": now_iso(),
    }


def register_dead(dead: dict, entry: dict) -> None:
    """Добавляет/обновляет запись в реестре (ключ — owner_photo_id)."""
    key = "{}_{}".format(entry.get("owner_id"), entry.get("photo_id"))
    old = dead.get(key)
    if old:
        old.update({k: v for k, v in entry.items() if k != "first_seen_dead"})
        old["last_checked"] = now_iso()
    else:
        dead[key] = entry


def _dead_is_fresh(entry: dict, ttl_days: int) -> bool:
    """Свежесть проверки: верим реестру TTL дней (по умолчанию 30).

    Отметка, которая не разбирается (или без часового пояса), — устаревшая.
    """
    try:
        t = datetime.fromisoformat(str(entry.get("last_checked")))
        return (datetime.now(timezone.utc) - t).days < max(1, ttl_days)
    except Exception:
        return False


# ----------------------------------------------------------------------------
# Индекс альбомов и назначение папок
# ----------------------------------------------------------------------------

def load_index(gdir: Path) -> dict:
    idx = read_json(Path(gdir) / "_albums.json", None)
    if not isinstance(idx, dict):
        idx = {}
    if not isinstance(idx.get("albums"), list):
        idx["albums"] = []
    return idx


def merge_index(idx: dict, group: GroupInfo, entries: list[dict]) -> dict:
    """Вливает записи альбомов в индекс группы, сохраняя порядок и историю."""
    by_id = {str(e.get("id")): e for e in idx["albums"]}
    order = [str(e.get("id")) for e in idx["albums"]]
    for e in entries:
        k = str(e["id"])
        if k in by_id:
            by_id[k].update(e)
        else:
            by_id[k] = dict(e)
            order.append(k)
    idx["albums"] = [by_id[k] for k in order if k in by_id]
    idx["group"] = group.as_dict()
    idx["updated"] = now_iso()
    return idx


def group_dir_segment(pattern: str, group: GroupInfo) -> str:
    """Уровень папки группы из шаблона (первый сегмент пути)."""
    head = pattern.split("/", 1)[0]
    cand = head.format(group=group.dir_key, group_id=group.id,
                       group_name=group.name_key, album="", album_id="")
    if "{album" in cand or not cand.strip():
        cand = group.name_key       # шаблон без уровня группы
    return sanitize_dirname(cand) or group.dir_key


def assign_group_dir(out_root: Path, group: GroupInfo, pattern: str,
                     register: bool = True) -> str:
    """Папка группы в датасете — по НАЗВАНИЮ группы (по умолчанию с 0.3).

    Порядок разрешения:
      1. прежняя папка из реестра <out>/_groups.json (докачка не переезжает);
      2. миграция датасетов vkds 0.2: если есть <out>/<screen_name>/_group.json;
      3. название группы (name_key), при совпадении названий разных групп —
         суффикс "__<screen_name|id>".
    """
    out_root = Path(out_root)
    reg_path = out_root / "_groups.json"
    reg = read_json(reg_path, None)
    reg = (reg if isinstance(reg, dict) and isinstance(reg.get("groups"), dict)
           else {"groups": {}})
    key = str(group.id)
    prev_dir = (reg["groups"].get(key) or {}).get("dir")
    if prev_dir:
        return str(prev_dir)
    if (out_root / group.dir_key / "_group.json").is_file():
        cand = group.dir_key                       # миграция датасета 0.2.x
    else:
        cand = group_dir_segment(pattern, group)
    used = {str((v or {}).get("dir")) for v in reg["groups"].values()}
    used.discard(None)
    if cand in used:
        sn = group.screen_name or ""
        if not sn or sn in ("club{}".format(group.id), "public{}".format(group.id)):
            sn = str(group.id)
        cand = "{}__{}".format(cand, sn)
    if register:
        reg["groups"][key] = {"id": group.id, "name": group.name,
                              "screen_name": group.screen_name, "dir": cand}
        reg["updated"] = now_iso()
        write_json(reg_path, reg)
    return cand


def assign_dirs(group: GroupInfo, albums: list[AlbumInfo], pattern: str,
                prev: dict, group_dir: str | None = None) -> dict:
    """id альбома -> относительный путь папки (от корня датасета).

    Ранее назначенные пути (prev, из _albums.json) не меняются — докачка
    всегда попадает в те же папки. При совпадении названий разных альбомов
    к имени второго добавляется "__<id>" (если в шаблоне нет {album_id}).

    group_dir — уже разрешённая папка группы (с учётом коллизий названий
    и миграции датасетов 0.2): первый сегмент шаблона заменяется ею, чтобы
    альбомы легли в ту же папку, что и служебные файлы группы.
    """
    if group_dir and "/" in pattern:
        head, tail = pattern.split("/", 1)
        if "{album" in tail:          # шаблон вида <группа>/<альбом...>
            pattern = "{}/{}".format(group_dir, tail)
    has_id = "{album_id}" in pattern
    dirs, used = {}, set(prev.values())
    for al in sorted(albums, key=lambda a: str(a.id)):
        key = str(al.id)
        if key in prev and prev.get(key):
            dirs[key] = prev[key]
            continue
        base = sanitize_dirname(al.title) or "album_{}".format(key)
        cand = pattern.format(group=group.dir_key, group_id=group.id,
                              group_name=group.name_key,
                              album=base, album_id=key)
        if not has_id and cand in used:
            base = "{}__{}".format(base, key)
            cand = pattern.format(group=group.dir_key, group_id=group.id,
                                  group_name=group.name_key,
                                  album=base, album_id=key)
        dirs[key] = cand
        used.add(cand)
    return dirs


def photo_key(owner_id, photo_id) -> tuple | None:
    """(owner, photo id) as (int, str), the identity of a photo in a group."""
    try:
        return int(float(str(owner_id))), str(photo_id)
    except (TypeError, ValueError):
        return None


def note_moved(gdir: Path, out_root: Path) -> dict:
    """Photos whose file is gone from the path its manifest row names.

    A row with status ok, exists or dup whose file is no longer at its path
    was moved away (by deduplicate, unrelated.py, another tool or by hand):
    it is settled, never listed or downloaded again (the rule the pastvu
    import applies to a file gone from an import). Its row keeps its status
    and gets the first date it was seen gone in the column `moved`; the
    manifest is written when a date is added. Returns (owner, id) -> row.
    """
    rows = read_manifest(gdir)
    moved, changed = {}, False
    for r in rows.values():
        if r.get("status") in ("ok", "exists", "dup") and r.get("path") \
                and not (Path(out_root) / r["path"]).is_file():
            if not r.get("moved"):
                r["moved"] = now_iso()[:10]
                changed = True
            key = photo_key(r.get("owner_id"), r.get("photo_id"))
            if key:
                moved.setdefault(key, r)
    if changed:
        write_manifest(gdir, rows)
    return moved


def count_statuses(records: list[dict]) -> dict:
    st = {"ok": 0, "exists": 0, "small": 0, "failed": 0, "no_url": 0,
          "dead": 0, "forbidden": 0, "moved": 0}
    for r in records:
        s = r.get("status", "no_url")
        st[s] = st.get(s, 0) + 1
    return st


def download_photos(session, records: list[dict], dest_dir: Path,
                    workers: int = 4, min_area: int = 0, min_side: int = 0,
                    retry_sleep: float = 1.5, dead: dict | None = None,
                    recheck_dead: bool = False,
                    dead_ttl_days: int = DEAD_TTL_DAYS,
                    write_txt: bool = True, moved: dict | None = None) -> list[dict]:
    """Скачивает фото параллельно с CDN; в записи добавляются file/status.

    Статусы: ok (скачано), exists (уже было), small (отфильтровано по площади),
    dead (404 — файл удалён с CDN ВК), forbidden (403), failed (не скачалось),
    no_url (нет копий в ответе API).

    Мёртвые фото: известные по реестру _dead_photos.json (проверка свежее
    dead_ttl_days) вообще не дёргают CDN — повторные запуски бесплатны.
    При неудаче лучшей копии (404, 403, сбой) пробуются ВСЕ меньшие копии
    (alt_urls; меньше min_area или со стороной меньше min_side — пропускаются),
    без вызовов API. dead — только
    если каждая опробованная копия ответила 404; использованная меньшая копия
    отмечается в записи (fallback_from, url_used, её размеры).

    Имя файла — photo<owner>_<id>.jpg; файл старой формы photo_<id>.jpg
    считается существующим. Для скачанных/существующих фото с подписью
    пишется <имя файла>.txt с очищенной подписью (clean_caption).
    """
    dead = dead if dead is not None else {}
    dest_dir = Path(dest_dir)
    jobs = []
    for rec in records:
        if not rec.get("url"):
            rec["status"] = "no_url"
            continue
        if too_small(rec.get("width"), rec.get("height"), min_area, min_side):
            rec["status"] = "small"
            continue
        fname = photo_stem(rec["owner_id"], rec["id"]) + ".jpg"
        dest = dest_dir / fname
        old = dest_dir / (old_stem(rec["id"]) + ".jpg")
        rec["file"] = fname
        if dest.exists() and dest.stat().st_size > 0:
            rec["status"] = "exists"
            continue
        if old.exists() and old.stat().st_size > 0:
            rec["file"] = old.name        # датасет до 0.5, ещё не migrate
            rec["status"] = "exists"
            continue
        gone = (moved or {}).get(photo_key(rec.get("owner_id"), rec["id"]))
        if gone is not None:              # its file was moved away: settled
            rec["status"] = "moved"
            rec["moved_row"] = gone
            continue
        ent = dead.get("{}_{}".format(rec.get("owner_id"), rec["id"]))
        if ent and not recheck_dead and _dead_is_fresh(ent, dead_ttl_days):
            rec["status"] = "dead"
            rec["dead_cached"] = True
            continue
        jobs.append((rec, dest))

    def _fetch(rec, dest):
        """Лучшая копия, затем все меньшие; dead — только если все 404."""
        best = {"type": rec.get("size_type"), "width": rec.get("width"),
                "height": rec.get("height")}
        alts = list(rec.get("alt_urls") or [])
        alt_sizes = list(rec.get("alt_sizes") or [])
        alt_sizes += [{}] * (len(alts) - len(alt_sizes))
        tries = [(rec["url"], best)] + [(u, s or {}) for u, s in zip(alts, alt_sizes) if u]
        statuses = []
        for i, (u, sz) in enumerate(tries):
            if i and too_small(sz.get("width"), sz.get("height"), min_area, min_side):
                continue                 # копия мельче фильтра — не годится
            res = download_file(session, u, dest, retry_sleep=retry_sleep)
            statuses.append(res["status"])
            if res["status"] == "ok":
                rec["url_used"] = u
                if i:
                    rec["fallback_from"] = rec.get("size_type")
                    rec.update(size_type=sz.get("type"), width=sz.get("width"),
                               height=sz.get("height"))
                return res
        if statuses and all(x == "dead" for x in statuses):
            return {"status": "dead"}
        if "forbidden" in statuses:
            return {"status": "forbidden"}
        return {"status": "failed"}

    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futs = {pool.submit(_fetch, rec, dest): rec for rec, dest in jobs}
            for fut in as_completed(futs):
                rec = futs[fut]
                try:
                    res = fut.result() or {"status": "failed"}
                except Exception:
                    res = {"status": "failed"}
                st = res.get("status")
                rec["status"] = st if st in ("ok", "dead", "forbidden") else "failed"

    for rec in records:
        rec.setdefault("status", "no_url")
        if rec.get("file") and rec.get("status") in ("ok", "exists"):
            try:
                rec["bytes"] = (dest_dir / rec["file"]).stat().st_size
            except OSError:
                rec["bytes"] = 0
        if write_txt and rec.get("file") and rec.get("status") in ("ok", "exists"):
            rec["txt"] = write_caption(dest_dir, Path(rec["file"]).stem,
                                       rec.get("text") or "")
    return records


# ----------------------------------------------------------------------------
# Описание альбома и манифест датасета
# ----------------------------------------------------------------------------

def write_album_txt(album_dir: Path, group: GroupInfo, album: AlbumInfo,
                    stats: dict) -> None:
    """Человекочитаемое описание альбома — album.txt рядом с фото."""
    link = ("https://vk.com/album{}_{}".format(group.owner_id, album.id)
            if str(album.id).isdigit() else "-")
    lines = [
        "Альбом: {}".format(album.title or "(без названия)"),
        "ID альбома: {}".format(album.id),
        "Группа: {} (vk.com/{}, id {})".format(
            group.name, group.screen_name or "club{}".format(group.id), group.id),
        "Фото по данным API: {}".format(
            album.size if album.size is not None else "?"),
        "Создан: {}".format(iso_date(album.created) or "?"),
        "Обновлён: {}".format(iso_date(album.updated) or "?"),
        "Ссылка: {}".format(link),
        "",
        "Описание альбома:",
        album.description.strip() or "(без описания)",
        "",
        "Последняя выгрузка ({}): фото {}, скачано {}, уже было {}, "
        "мёртвых (404) {}, отфильтровано {}, ошибок {}".format(
            now_iso(), stats.get("photos", 0), stats.get("ok", 0),
            stats.get("exists", 0), stats.get("dead", 0),
            stats.get("small", 0),
            stats.get("failed", 0) + stats.get("forbidden", 0)
            + stats.get("no_url", 0)),
    ]
    try:
        (Path(album_dir) / "album.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        pass


def read_manifest(gdir: Path) -> dict:
    """_manifest.csv группы: {(album_id, photo_id): row} в порядке файла."""
    path = Path(gdir) / MANIFEST_NAME
    rows = {}
    if path.is_file():
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    rows[(str(r.get("album_id")), str(r.get("photo_id")))] = r
        except Exception:
            rows = {}
    return rows


def write_manifest(gdir: Path, rows: dict) -> Path:
    path = Path(gdir) / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows.values():
            w.writerow(r)
    return path


def update_manifest(gdir: Path, out_root: Path, group: GroupInfo,
                    album: AlbumInfo, records: list[dict],
                    rel_dir: str, extra: dict | None = None) -> Path:
    """Обновляет _manifest.csv группы: по строке на КАЖДОЕ фото альбома.

    Манифест — основа для сборки датасета: пути к файлам, имя .txt с
    очищенной подписью, статусы, размеры; колонка text — подпись как есть.
    Мёртвые фото тоже входят (status=dead) — их подписи и метаданные
    сохраняются для восстановления. Строки других альбомов не трогаются;
    строки обрабатываемого альбома заменяются.

    extra — поля, общие для всех строк (у постов стены: post_id, post_date,
    repost_from, reposts, likes и comments поста). Запись со статусом dup
    (фото уже есть в другом месте группы) получает путь существующего файла
    из dup_path.
    """
    gdir, out_root = Path(gdir), Path(out_root)
    rows = read_manifest(gdir)
    for rec in records:
        key = (str(album.id), str(rec["id"]))
        if rec.get("status") == "moved":
            own = rows.get(key)
            if own is not None and own.get("path"):
                own["moved"] = own.get("moved") or now_iso()[:10]
                if extra:
                    own.update({k: v for k, v in extra.items() if not own.get(k)})
                continue
            gone = rec.get("moved_row") or {}
            rec = dict(rec, status="dup", dup_path=gone.get("path") or "",
                       txt=gone.get("txt") or "", moved=gone.get("moved") or now_iso()[:10])
        fname = rec.get("file") or ""
        if rec.get("status") not in ("ok", "exists"):
            fname = ""                # файла нет (dead/failed/small) — не указываем
        rel = (Path(rel_dir) / fname).as_posix() if fname else ""
        if rec.get("status") == "dup" and rec.get("dup_path"):
            rel = rec["dup_path"]
            fname = Path(rel).name
        txt = (rec.get("txt") or "") if fname else ""
        rows[(str(album.id), str(rec["id"]))] = {
            "photo_id": rec.get("id"), "owner_id": rec.get("owner_id"),
            "group_id": group.id, "group_name": group.name,
            "group_screen": group.screen_name,
            "album_id": album.id, "album_title": album.title,
            "album_dir": rel_dir, "file": fname, "path": rel, "txt": txt,
            "status": rec.get("status"), "size_type": rec.get("size_type"),
            "width": rec.get("width"), "height": rec.get("height"),
            "bytes": rec.get("bytes", ""), "date": rec.get("date"),
            "likes": rec.get("likes"), "comments": rec.get("comments"),
            "text": rec.get("text") or "",
            "moved": rec.get("moved") or "",
        }
        if extra:
            rows[(str(album.id), str(rec["id"]))].update(extra)
    return write_manifest(gdir, rows)


def row_done(r: dict, out_root: Path, dead: dict, opts) -> str | None:
    """The reconcile verdict of one manifest row, without any API call.

    "exists" when its file (ok, exists, dup) is on disk, "dead" when the dead
    registry has a fresh entry and no recheck is asked, "small" when it is
    still under the current min_area or min_side; None when the photo is not
    settled. `opts` needs min_area, recheck_dead and dead_ttl_days (min_side
    is optional).
    """
    status = r.get("status")
    if status in ("ok", "exists", "dup"):
        if r.get("path") and (Path(out_root) / r["path"]).is_file():
            return "exists"
        if r.get("path") and not getattr(opts, "redownload_missing", False):
            return "moved"            # the file was moved away: settled
        return None
    if status == "dead":
        ent = dead.get("{}_{}".format(r.get("owner_id"), r.get("photo_id")))
        if opts.recheck_dead or not ent or not _dead_is_fresh(ent, opts.dead_ttl_days):
            return None
        return "dead"
    if status == "small":
        small = too_small(r.get("width"), r.get("height"), opts.min_area,
                          getattr(opts, "min_side", 0))
        return "small" if small else None
    return None


def reconcile_album(rows: list[dict], album: AlbumInfo, entry: dict,
                    out_root: Path, dead: dict, opts: AlbumsOptions) -> dict | None:
    """Сверка альбома с манифестом и диском без вызовов API (D7).

    Возвращает счётчики, если у альбома ничего не недостаёт: число строк
    равно размеру альбома из живого photos.getAlbums (или размер и дата
    обновления те же, что при последнем живом списке), каждый файл ok/exists
    на диске, каждое мёртвое фото свежо в реестре, каждое small мельче
    текущего --min-area. Иначе None — альбом надо листать живым.
    """
    if album.is_special or album.size is None or not rows:
        return None
    same_listing = (entry.get("listed_size") == album.size
                    and entry.get("listed_updated") == album.updated
                    and entry.get("listed_count") == len(rows))
    if len(rows) != album.size and not same_listing:
        return None
    st = {"ok": 0, "exists": 0, "small": 0, "failed": 0, "no_url": 0,
          "dead": 0, "forbidden": 0, "moved": 0}
    for r in rows:
        verdict = row_done(r, out_root, dead, opts)
        if verdict is None:
            return None
        st[verdict] += 1
    return st


# ----------------------------------------------------------------------------
# Основные сценарии
# ----------------------------------------------------------------------------

def run_albums(client: VkClient, targets, out_root, opts: AlbumsOptions) -> dict:
    """Скачивает альбомы сообществ (может быть несколько) в структуру
    <out>/<название группы>/<название альбома>/. Возвращает сводку."""
    out_root = Path(out_root)
    groups = resolve_groups(client, targets)
    report = {"groups": [], "api_calls": client.api_calls}
    for group in groups:
        report["groups"].append(_process_group(client, group, out_root, opts))
    report["api_calls"] = client.api_calls
    return report


def _process_group(client: VkClient, group: GroupInfo, out_root: Path,
                   opts: AlbumsOptions) -> dict:
    print("\n=== {} (id={}) ===".format(group.name, group.id))
    try:
        albums = fetch_albums(client, group, live=not opts.dry_run)
    except VkApiError as e:
        if e.limit:
            raise
        print("  [!] Не удалось получить список альбомов: {}".format(e))
        return {"group": group.as_dict(), "error": str(e), "albums": []}

    selected = select_albums(albums, opts)
    if not selected:
        print("  Альбомы не найдены или не прошли фильтр.")
        return {"group": group.as_dict(), "albums": []}

    gkey = assign_group_dir(out_root, group, opts.dir_pattern,
                            register=not opts.dry_run)
    print("  Папка группы: «{}»".format(gkey))
    gdir = out_root / gkey
    idx = load_index(gdir)
    prev = {str(e.get("id")): e.get("dir") for e in idx["albums"] if e.get("dir")}
    dirs = assign_dirs(group, selected, opts.dir_pattern, prev, group_dir=gkey)

    if opts.dry_run:
        _print_dry_run(group, albums, selected, dirs)
        return {"group": group.as_dict(), "dry_run": True,
                "albums": [dict(id=a.id, title=a.title, size=a.size,
                                dir=dirs[str(a.id)]) for a in selected]}

    # Фиксируем индекс и инфо о группе ДО скачивания: назначение папок
    # сохранится даже при обрыве в середине.
    entries = []
    for al in selected:
        e = al.as_dict()
        e["dir"] = dirs[str(al.id)]
        e["status"] = "planned"
        entries.append(e)
    gdir.mkdir(parents=True, exist_ok=True)
    merge_index(idx, group, entries)
    write_json(gdir / "_albums.json", idx)
    write_json(gdir / "_group.json", group.as_dict())

    dead = load_dead(gdir)
    moved = {} if opts.redownload_missing else note_moved(gdir, out_root)
    manifest = {}
    for (aid, _pid), row in read_manifest(gdir).items():
        manifest.setdefault(aid, []).append(row)
    entries_by_id = {str(e.get("id")): dict(e) for e in idx["albums"]}
    gsum = {"group": group.as_dict(), "albums": [], "photos": 0,
            "downloaded": 0, "exists": 0, "small": 0, "failed": 0,
            "no_url": 0, "dead": 0, "forbidden": 0, "moved": 0}
    for i, al in enumerate(selected, 1):
        arep = _process_album(client, group, al, out_root / dirs[str(al.id)],
                              opts, i, len(selected), rel_dir=dirs[str(al.id)],
                              out_root=out_root, gdir=gdir, dead=dead,
                              rows=manifest.get(str(al.id), []),
                              entry=entries_by_id.get(str(al.id), {}), moved=moved)
        for e in idx["albums"]:
            if str(e.get("id")) == str(al.id):
                e.update(arep.pop("index"))
                break
        write_json(gdir / "_albums.json", idx)
        gsum["albums"].append(arep)
        for k in ("photos", "downloaded", "exists", "small", "failed",
                  "no_url", "dead", "forbidden", "moved"):
            gsum[k] += arep.get(k, 0)
    save_dead(gdir, dead)

    print("  Итого по «{}»: альбомов {}, фото {}, скачано {}, уже было {}, "
          "мёртвых (404) {}, отфильтровано {}, перемещено {}, ошибок {}".format(
              gkey, len(selected), gsum["photos"], gsum["downloaded"],
              gsum["exists"], gsum["dead"], gsum["small"], gsum["moved"],
              gsum["failed"] + gsum["no_url"] + gsum["forbidden"]))
    print("  Манифест: {} | реестр мёртвых: {}".format(
        gdir / MANIFEST_NAME, gdir / "_dead_photos.json"))
    return gsum


def _process_album(client: VkClient, group: GroupInfo, album: AlbumInfo,
                   album_dir: Path, opts: AlbumsOptions,
                   num: int, total: int, rel_dir: str = "",
                   out_root: Path | None = None,
                   gdir: Path | None = None, dead: dict | None = None,
                   rows: list | None = None, entry: dict | None = None,
                   moved: dict | None = None) -> dict:
    print("[{}/{}] Альбом «{}» (id={}), фото по данным API: {}".format(
        num, total, album.title, album.id,
        album.size if album.size is not None else "?"))
    print("      -> {}".format(album_dir))
    arep = {"id": album.id, "title": album.title, "dir": rel_dir,
            "photos": 0, "downloaded": 0,
            "exists": 0, "small": 0, "failed": 0, "no_url": 0,
            "dead": 0, "forbidden": 0, "moved": 0,
            "index": {"status": "started", "last_run": now_iso()}}
    if out_root is not None and rows:
        done = reconcile_album(rows, album, entry or {}, out_root,
                               dead if dead is not None else {}, opts)
        if done is not None:
            arep.update({"photos": len(rows), "exists": done["exists"],
                         "dead": done["dead"], "small": done["small"],
                         "moved": done["moved"]})
            arep["index"].update({"status": "done", "reconciled": now_iso(),
                                  "stats": {"photos": len(rows), "downloaded": 0,
                                            "exists": done["exists"],
                                            "small": done["small"], "failed": 0,
                                            "no_url": 0, "dead": done["dead"],
                                            "forbidden": 0, "moved": done["moved"]}})
            print("      сверка с манифестом: всё на месте (есть {}, мёртвых {}, "
                  "отфильтровано {}, перемещено {}), вызовов API: 0".format(
                      done["exists"], done["dead"], done["small"], done["moved"]))
            return arep
    try:
        photos = fetch_photos(client, group, album.id)
    except VkApiError as e:
        if e.limit:
            raise
        print("      [!] Пропуск альбома: {}".format(e))
        arep["index"]["status"] = "error"
        arep["index"]["error"] = str(e)
        return arep
    if not photos:
        print("      Альбом пуст.")
        arep["index"]["status"] = "empty"
        return arep

    # защита от дублей одного фото в ответе API (сбой пагинации, повторные
    # записи): иначе параллельные потоки пишут один и тот же файл
    uniq, seen = [], set()
    for p in photos:
        pid = p.get("id")
        if pid in seen:
            continue
        seen.add(pid)
        uniq.append(p)
    if len(uniq) != len(photos):
        print("      [i] Дублей фото в ответе API: {} — пропущены".format(
            len(photos) - len(uniq)))
    records = [make_record(p) for p in uniq]
    for rec in records:
        if rec.get("owner_id") in (None, ""):
            rec["owner_id"] = group.owner_id
    album_dir.mkdir(parents=True, exist_ok=True)
    records = download_photos(client.session, records, album_dir,
                              workers=opts.workers, min_area=opts.min_area,
                              min_side=opts.min_side,
                              retry_sleep=opts.dl_retry_sleep,
                              dead=dead, recheck_dead=opts.recheck_dead,
                              dead_ttl_days=opts.dead_ttl_days,
                              write_txt=not opts.no_meta, moved=moved)
    st = count_statuses(records)
    arep.update({"photos": len(records), "downloaded": st["ok"],
                 "exists": st["exists"], "small": st["small"],
                 "failed": st["failed"], "no_url": st["no_url"],
                 "dead": st["dead"], "forbidden": st["forbidden"],
                 "moved": st["moved"]})

    # Новые мёртвые фото -> реестр группы (сохраняем всё, что осталось)
    if dead is not None:
        photos_by_id = {p.get("id"): p for p in photos}
        new_dead = 0
        for rec in records:
            if rec.get("status") == "dead" and not rec.get("dead_cached"):
                register_dead(dead, make_dead_entry(
                    group, album, photos_by_id.get(rec["id"], {}), urls_live=True))
                new_dead += 1
        if new_dead and gdir is not None:
            save_dead(gdir, dead)
            print("      [!] Мёртвых фото (404, файл удалён с CDN ВК): {} — "
                  "метаданные сохранены в _dead_photos.json".format(new_dead))

    if not opts.no_meta:
        write_album_txt(album_dir, group, album, dict(st, photos=len(records)))
        write_json(album_dir / "metadata.json", {
            "group": group.as_dict(),
            "album": {
                "id": album.id, "title": album.title,
                "description": album.description,
                "created": iso_date(album.created),
                "updated": iso_date(album.updated),
                "size_api": album.size, "fetched": len(records),
                "special": album.special,
            },
            "generated": now_iso(),
            "photos": records,
        })

    if gdir is not None and out_root is not None:
        update_manifest(gdir, out_root, group, album, records, rel_dir)

    arep["index"].update({
        "status": "done",
        "listed_size": album.size, "listed_updated": album.updated,
        "listed_count": len(records),
        "stats": {"photos": len(records), "downloaded": st["ok"],
                  "exists": st["exists"], "small": st["small"],
                  "failed": st["failed"], "no_url": st["no_url"],
                  "dead": st["dead"], "forbidden": st["forbidden"],
                  "moved": st["moved"]},
    })
    print("      итог: скачано {}, уже было {}, мёртвых (404) {}, "
          "отфильтровано (мельче {} px^2 или стороной меньше {} px) {}, "
          "перемещено {}, ошибок {}".format(
              st["ok"], st["exists"], st["dead"], opts.min_area or "-",
              opts.min_side or "-", st["small"], st["moved"],
              st["failed"] + st["no_url"] + st["forbidden"]))
    return arep


def _print_dry_run(group, albums, selected, dirs):
    total_photos = 0
    est = 1 + math.ceil(len(albums) / ALBUMS_PER_PAGE)
    print("  Dry-run: будет скачано {} из {} альбомов".format(len(selected),
                                                              len(albums)))
    for i, a in enumerate(selected, 1):
        total_photos += a.size or 0
        est += max(1, math.ceil((a.size or 1) / PHOTOS_PER_PAGE))
        print("   {:>3}. [id={}] «{}» — {} фото -> {}".format(
            i, a.id, a.title, a.size if a.size is not None else "?",
            dirs[str(a.id)]))
    print("  Всего фото: {} | примерно вызовов API: {} "
          "(сами файлы с CDN — бесплатно)".format(total_photos, est))


def run_scan_albums(client: VkClient, target, out_root) -> dict:
    """Разведка для агента: список альбомов группы с размерами и датами.

    Стоимость: 1 вызов groups.getById + 1 на каждые 100 альбомов.
    Результат сохраняется в <out>/<группа>/_albums.json (статус «scanned»).
    """
    out_root = Path(out_root)
    group = resolve_groups(client, [target])[0]
    albums = fetch_albums(client, group)

    gkey = assign_group_dir(out_root, group, DEFAULT_DIR_PATTERN)
    gdir = out_root / gkey
    idx = load_index(gdir)
    prev = {str(e.get("id")): e.get("dir") for e in idx["albums"] if e.get("dir")}
    dirs = assign_dirs(group, albums, DEFAULT_DIR_PATTERN, prev, group_dir=gkey)
    entries = []
    for a in albums:
        e = a.as_dict()
        e["dir"] = dirs[str(a.id)]
        e["status"] = "scanned"
        entries.append(e)
    gdir.mkdir(parents=True, exist_ok=True)
    merge_index(idx, group, entries)
    write_json(gdir / "_albums.json", idx)
    write_json(gdir / "_group.json", group.as_dict())

    print("Сообщество: {} (id={}, папка «{}»)".format(group.name, group.id, gkey))
    print("Найдено альбомов: {}".format(len(albums)))
    for a in albums:
        upd = (iso_date(a.updated) or "")[:10]
        print("  [id={:>6}] {:>6} фото  {:<10} «{}»".format(
            a.id, a.size if a.size is not None else "?", upd, a.title))
    est = 1 + math.ceil(len(albums) / ALBUMS_PER_PAGE) + \
        sum(max(1, math.ceil((a.size or 1) / PHOTOS_PER_PAGE)) for a in albums)
    print("Полное скачивание всех альбомов: ~{} фото, ~{} вызовов API.".format(
        sum(a.size or 0 for a in albums), est))
    print("Фото со стены в scan не входят: при скачивании добавьте --include-wall.")
    print("Индекс: {}".format(gdir / "_albums.json"))
    return {"group": group.as_dict(), "albums": [a.as_dict() for a in albums],
            "est_api_calls": est}


def run_peek(client: VkClient, target, album_id, limit: int, offset: int,
             out_root, min_side: int = 0) -> dict:
    """Пробная выборка фото альбома — «глаза» агента перед решением.

    Стоимость: 1 вызов photos.get (+1 groups.getById, если группа ещё не
    известна). Файлы скачиваются с CDN в полном размере — бесплатно.
    Складываются в <out>/_peek/<группа>/album_<id>/ вместе с _meta.json;
    мёртвые (404) помечаются статусом dead.
    """
    out_root = Path(out_root)
    group = resolve_groups(client, [target])[0]
    resp = client.call("photos.get", _live=True, owner_id=group.owner_id,
                       album_id=str(album_id), photo_sizes=1, extended=1,
                       offset=int(offset), count=int(limit))
    photos = resp.get("items", [])
    dest = out_root / "_peek" / group.dir_key / "album_{}".format(album_id)
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    for p in photos:
        rec = make_record(p)
        if rec.get("owner_id") in (None, ""):
            rec["owner_id"] = group.owner_id
        fname = photo_stem(rec["owner_id"], rec["id"]) + ".jpg"
        if too_small(rec.get("width"), rec.get("height"), 0, min_side):
            rec["status"] = "small"
        elif rec.get("url"):
            res = download_file(client.session, rec["url"], dest / fname)
            rec["http_status"] = res.get("http_code")
            rec["status"] = res["status"]
        else:
            rec["status"] = "no_url"
        ok = rec["status"] == "ok"
        rec["file"] = fname if ok else None
        if ok:
            rec["txt"] = write_caption(dest, Path(fname).stem, rec.get("text") or "")
        records.append(rec)
    write_json(dest / "_meta.json", {
        "group": group.as_dict(), "album_id": str(album_id),
        "offset": int(offset), "count": len(records), "photos": records,
    })
    dead_n = sum(1 for r in records if r["status"] == "dead")
    print("Пробы альбома {} ({}): {} фото -> {}".format(
        album_id, group.name, len(records), dest))
    if dead_n:
        print("   [!] мёртвых (404): {} — файлы удалены с CDN ВК".format(dead_n))
    for r in records:
        print("   {} | {}x{} | {} | {}".format(
            r["file"] or "-", r["width"], r["height"], (r["date"] or "")[:10],
            (r["text"] or "")[:60]))
    return {"dest": str(dest), "count": len(records), "dead": dead_n}
