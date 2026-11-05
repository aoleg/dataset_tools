"""Finding groups and sampling them: the verbs `search groups` and `sample`.

`search groups --q Q [--q ...]` calls groups.search once per query (up to
1000 results each; --type group, page or all), then groups.getById on the ids
found, 500 per call, for the member count, the description and the counters
(groups.search returns only the basic fields). The results of all queries are
merged by group id into <out>/_candidates.json: earlier candidates stay, fresh
data updates them, and each candidate keeps the queries that found it. The
printed table is short on purpose (the partner's agent reads it): id, short
name, title, members, open or closed, the first 60 characters of the
description, the number of this run's queries that found the group and the
estimated calls of a `scan albums` plus a `scan wall` of it.

`sample <group> [--n 12] [--from wall|albums] [--album-id ID] [--sheet]
[--min-side N]` downloads a handful of photos for a look:

- from the wall, one wall.get page (100 posts with their photos, one call);
  the n photos are spread evenly over the posts that have photos, one photo
  per post when there are enough posts, else several per post;
- from the albums, photos.getAlbums plus one photos.get page (up to 1000
  photos) of the largest album or of --album-id (two calls); the n photos are
  spread evenly over the page.

The group is looked up offline in _candidates.json or the dataset's
_groups.json; only a group found in neither costs one groups.getById more.
Files go to <out>/_samples/<group folder>/ named as everywhere,
photo<owner>_<id>.jpg with the cleaned caption as .txt (the post text for a
wall photo), plus _sample.json and, with --sheet, sheet.jpg (vkds.sheet). A
photo whose every copy answers 404 is skipped and replaced by its neighbour,
never registered as dead: a sample is not a dataset. A sample also refines
the candidate's cost estimate (the number of wall posts or albums).

What the tool prints stays Russian (D2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .albums import (
    ALBUMS_PER_PAGE,
    DEFAULT_DIR_PATTERN,
    PHOTOS_PER_PAGE,
    SPECIAL_ALBUMS,
    assign_group_dir,
    fetch_albums,
    make_record,
    parse_group_ref,
    photo_stem,
    resolve_groups,
    too_small,
    write_caption,
)
from .api import VkApiError, VkClient
from .budget import now_iso
from .captions import clean_caption
from .fs import download_file, read_json, write_json
from .models import GroupInfo
from .sheet import SHEET_NAME, SheetError, render_sheet
from .wall import WALL_PAGE, caption_of, normalize_post

CANDIDATES_NAME = "_candidates.json"
SAMPLES_DIR = "_samples"
SAMPLE_NAME = "_sample.json"
SEARCH_TYPES = {"group": "group", "page": "page", "all": None}
SEARCH_MAX = 1000          # the maximum count of groups.search
GETBYID_MAX = 500          # group ids per groups.getById
# 5 authorization failed, 15 access denied, 27 group key, 28 application key:
# the key may not call the method at all
AUTH_CODES = {5, 15, 27, 28}
CLOSED_LABELS = {0: "открыт", 1: "закрыт", 2: "частный"}


class SearchError(Exception):
    """No query, or the key may not search: a readable refusal."""


class SampleError(Exception):
    """An unknown or closed group, an empty wall or album: a readable refusal."""


def _fmt(n) -> str:
    if n is None or n == "":
        return "?"
    return "{:,}".format(int(n)).replace(",", " ")


def _short(text, n: int) -> str:
    return " ".join(str(text or "").split())[:n]


# ----------------------------------------------------------------------------
# _candidates.json
# ----------------------------------------------------------------------------

def load_candidates(out_root: Path) -> dict:
    data = read_json(Path(out_root) / CANDIDATES_NAME, None)
    if not isinstance(data, dict):
        data = {}
    if not isinstance(data.get("groups"), dict):
        data["groups"] = {}
    if not isinstance(data.get("queries"), dict):
        data["queries"] = {}
    return data


def scan_cost(entry: dict) -> tuple[int, bool]:
    """The calls of `scan albums` plus `scan wall` of a candidate, and whether
    both counts are known. Each scan costs one groups.getById plus one call
    per 100 albums or posts (at least one); an unknown count counts as one
    page, so the estimate is then a lower bound."""
    albums = (entry.get("counters") or {}).get("albums")
    posts = entry.get("wall_posts")
    c_albums = 1 + max(1, math.ceil(int(albums) / ALBUMS_PER_PAGE)) if albums is not None else 2
    c_wall = 1 + max(1, math.ceil(int(posts) / WALL_PAGE)) if posts is not None else 2
    return c_albums + c_wall, albums is not None and posts is not None


def _closed(entry: dict) -> str:
    try:
        return CLOSED_LABELS.get(int(entry.get("is_closed") or 0), "?")
    except (TypeError, ValueError):
        return "?"


# ----------------------------------------------------------------------------
# search groups
# ----------------------------------------------------------------------------

def run_search_groups(client: VkClient, queries, out_root, count: int = 20,
                      group_type: str = "all") -> dict:
    """Search groups by every query, merge them into _candidates.json, print the table."""
    out_root = Path(out_root)
    qs = []
    for q in queries or []:
        q = " ".join(str(q or "").split())
        if q and q not in qs:
            qs.append(q)
    if not qs:
        raise SearchError("Нужен хотя бы один запрос: --q \"<слова>\".")
    if group_type not in SEARCH_TYPES:
        raise SearchError("--type: group, page или all.")
    count = max(1, min(SEARCH_MAX, int(count)))

    found: dict[int, dict] = {}
    hits: dict[int, list] = {}
    per_query: dict[str, dict] = {}
    for q in qs:
        try:
            resp = client.call("groups.search", q=q, type=SEARCH_TYPES[group_type], count=count)
        except VkApiError as e:
            if e.limit:
                raise
            if e.code in AUTH_CODES:
                raise SearchError(
                    "Поиск сообществ (groups.search) недоступен с этим ключом: [{}] {}. "
                    "Сервисный ключ, похоже, не может искать; нужен ключ пользователя. "
                    "Без поиска группы называют вручную (короткое имя или ссылка) и сразу "
                    "делают sample.".format(e.code, e.message)) from e
            print("[!] Запрос «{}»: {}".format(q, e))
            continue
        items = (resp or {}).get("items") or []
        per_query[q] = {"total": (resp or {}).get("count"), "returned": len(items)}
        for it in items:
            try:
                gid = int(it["id"])
            except (KeyError, TypeError, ValueError):
                continue
            found.setdefault(gid, dict(it))
            if q not in hits.setdefault(gid, []):
                hits[gid].append(q)

    ids = list(found)
    details: dict[int, dict] = {}
    for i in range(0, len(ids), GETBYID_MAX):
        chunk = ids[i:i + GETBYID_MAX]
        try:
            resp = client.call("groups.getById", group_id=",".join(str(x) for x in chunk),
                               fields="description,members_count,counters,activity")
        except VkApiError as e:
            if e.limit:
                raise
            print("[!] Подробности сообществ (участники, описание) не получены: {}".format(e))
            break
        for g in (resp or {}).get("groups") or (resp or {}).get("items") or []:
            try:
                details[int(g["id"])] = g
            except (KeyError, TypeError, ValueError):
                continue

    cand = load_candidates(out_root)
    now = now_iso()
    new_ids = []
    for gid in ids:
        key = str(gid)
        e = cand["groups"].get(key)
        if not isinstance(e, dict):
            e = {"first_seen": now}
            new_ids.append(gid)
        d = dict(found[gid])
        d.update(details.get(gid) or {})
        e.update({
            "id": gid,
            "screen_name": str(d.get("screen_name") or e.get("screen_name") or ""),
            "name": str(d.get("name") or e.get("name") or ""),
            "type": str(d.get("type") or e.get("type") or ""),
            "is_closed": d.get("is_closed", e.get("is_closed", 0)),
            "members_count": d.get("members_count", e.get("members_count")),
            "description": str(d.get("description") or e.get("description") or ""),
            "activity": str(d.get("activity") or e.get("activity") or ""),
            "last_seen": now,
        })
        counters = d.get("counters")
        if isinstance(counters, dict):
            old = e.get("counters") if isinstance(e.get("counters"), dict) else {}
            old.update({k: counters[k] for k in ("albums", "photos", "topics", "videos")
                        if k in counters})
            e["counters"] = old
        queries_of = e.get("queries") if isinstance(e.get("queries"), list) else []
        for q in hits.get(gid, []):
            if q not in queries_of:
                queries_of.append(q)
        e["queries"] = queries_of
        cand["groups"][key] = e
    for q, info in per_query.items():
        cand["queries"][q] = {"ts": now, "type": group_type, "count": count, **info}
    cand["updated"] = now
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(out_root / CANDIDATES_NAME, cand)

    rows = [cand["groups"][str(g)] for g in ids]
    rows.sort(key=lambda e: -(int(e.get("members_count") or 0)))
    print("Поиск сообществ: запросов {}, найдено {} (новых {}), всего кандидатов в файле {}".format(
        len(qs), len(ids), len(new_ids), len(cand["groups"])))
    for q in qs:
        info = per_query.get(q)
        print("   «{}»: {}".format(q, "ошибка" if info is None else "{} из {}".format(
            info["returned"], _fmt(info["total"]))))
    if rows:
        print("{:>10}  {:<20} {:<40} {:>9}  {:<7} {:>4} {:>6}  {}".format(
            "id", "короткое имя", "название", "участн.", "доступ", "запр", "вызов", "описание"))
    for e in rows:
        cost, known = scan_cost(e)
        closed = _closed(e)
        est = "—" if closed != "открыт" else (str(cost) if known else "от {}".format(cost))
        print("{:>10}  {:<20} {:<40} {:>9}  {:<7} {:>4} {:>6}  {}".format(
            e["id"], _short(e.get("screen_name"), 20), _short(e.get("name"), 40),
            _fmt(e.get("members_count")), closed, len(hits.get(int(e["id"]), [])), est,
            _short(e.get("description"), 60)))
    print("«вызов» — оценка scan albums + scan wall; «от N» — число альбомов или постов "
          "пока неизвестно (посты считает sample); «—» — закрытое сообщество, стену и "
          "альбомы не прочитать.")
    print("Кандидаты: {}".format(out_root / CANDIDATES_NAME))
    return {"queries": qs, "found": ids, "new": new_ids, "per_query": per_query,
            "candidates": len(cand["groups"])}


# ----------------------------------------------------------------------------
# sample
# ----------------------------------------------------------------------------

@dataclass
class SampleOptions:
    n: int = 12
    source: str = "wall"           # wall or albums
    album_id: str | None = None
    sheet: bool = False
    min_side: int = 0


def _group_from(d: dict) -> GroupInfo:
    return GroupInfo(id=int(d["id"]), name=str(d.get("name") or ""),
                     screen_name=str(d.get("screen_name") or ""),
                     description=str(d.get("description") or ""),
                     members_count=d.get("members_count"))


def resolve_offline(out_root: Path, target) -> tuple[GroupInfo | None, dict | None]:
    """The group from _candidates.json or the dataset's _groups.json, no call.
    Returns (group, candidate entry or None)."""
    out_root = Path(out_root)
    ref = parse_group_ref(target)
    if not ref:
        return None, None
    for e in load_candidates(out_root)["groups"].values():
        if isinstance(e, dict) and ref in (str(e.get("id")), str(e.get("screen_name") or "")):
            try:
                return _group_from(e), e
            except (KeyError, TypeError, ValueError):
                break
    reg = read_json(out_root / "_groups.json", None)
    groups = (reg or {}).get("groups") if isinstance(reg, dict) else None
    for gid, g in (groups or {}).items():
        g = g or {}
        if ref in (str(gid), str(g.get("screen_name") or "")):
            gj = read_json(out_root / str(g.get("dir") or "") / "_group.json", None)
            try:
                return _group_from(gj if isinstance(gj, dict) and gj.get("id") else
                                   dict(g, id=gid)), None
            except (KeyError, TypeError, ValueError):
                return None, None
    return None, None


def _slots(n_units: int, n: int) -> list[list[int]]:
    """n contiguous slots over n_units positions, each ordered from its centre
    outwards; one slot per position when there are no more positions than n."""
    if n_units <= n:
        return [[i] for i in range(n_units)]
    out = []
    for i in range(n):
        lo, hi = i * n_units // n, (i + 1) * n_units // n
        mid = (lo + hi - 1) // 2
        out.append(sorted(range(lo, hi), key=lambda k, m=mid: (abs(k - m), k)))
    return out


def wall_plan(posts: list[dict], n: int) -> list[tuple[int, list]]:
    """(quota, candidates) slots over the posts that have photos; a candidate
    is (position, post, photo). One photo per post when there are at least n
    posts (the first photo of the slot's centre post first), else the n
    photos shared out over the posts."""
    with_photos = [p for p in posts if p.get("photos")]
    if not with_photos:
        return []
    if len(with_photos) >= n:
        plan = []
        for slot in _slots(len(with_photos), n):
            first = [((k, 0), with_photos[k], with_photos[k]["photos"][0]) for k in slot]
            rest = [((k, j), with_photos[k], ph) for k in slot
                    for j, ph in enumerate(with_photos[k]["photos"]) if j]
            plan.append((1, first + rest))
        return plan
    base, extra = divmod(n, len(with_photos))
    return [(base + (1 if k < extra else 0),
             [((k, j), p, ph) for j, ph in enumerate(p["photos"])])
            for k, p in enumerate(with_photos)]


def album_plan(photos: list[dict], n: int) -> list[tuple[int, list]]:
    return [(1, [((k, 0), None, photos[k]) for k in slot]) for slot in _slots(len(photos), n)]


def pick(plan: list[tuple[int, list]], n: int, fetch) -> list[tuple]:
    """Fill every slot's quota in its order, then top up from the untried
    candidates of all slots in turn until n photos are in. fetch(candidate)
    returns a record or None (skipped). Returns (position, record) pairs in
    page order."""
    got, rest_of = [], []
    for quota, cands in plan:
        k, rest = 0, []
        for c in cands:
            if k >= quota or len(got) >= n:
                rest.append(c)
                continue
            rec = fetch(c)
            if rec is not None:
                got.append((c[0], rec))
                k += 1
        rest_of.append(rest)
    while len(got) < n and any(rest_of):
        for rest in rest_of:
            if rest and len(got) < n:
                c = rest.pop(0)
                rec = fetch(c)
                if rec is not None:
                    got.append((c[0], rec))
    got.sort(key=lambda x: x[0])
    return got


def _fetcher(session, dest: Path, group: GroupInfo, min_side: int, skipped: dict, seen: set):
    def fetch(cand):
        _pos, post, photo = cand
        rec = make_record(photo)
        if rec.get("owner_id") in (None, ""):
            rec["owner_id"] = group.owner_id
        key = (str(rec["owner_id"]), str(rec.get("id")))
        if key in seen:                       # the same photo in two posts
            return None
        seen.add(key)
        if too_small(rec.get("width"), rec.get("height"), 0, min_side):
            skipped["small"] += 1
            return None
        if not rec.get("url"):
            skipped["no_url"] += 1
            return None
        stem = photo_stem(rec["owner_id"], rec["id"])
        tries = [(rec["url"], {})] + list(zip(rec.get("alt_urls") or [],
                                              (rec.get("alt_sizes") or []) + [{}] * 64,
                                              strict=False))
        statuses = []
        for i, (url, sz) in enumerate(tries):
            if i and too_small(sz.get("width"), sz.get("height"), 0, min_side):
                continue
            res = download_file(session, url, dest / (stem + ".jpg"), retry_sleep=0.5)
            statuses.append(res["status"])
            if res["status"] == "ok":
                if i:
                    rec.update(size_type=sz.get("type"), width=sz.get("width"),
                               height=sz.get("height"))
                rec["url_used"] = url
                break
        else:
            if statuses and all(s == "dead" for s in statuses):
                skipped["dead"] += 1
            else:
                skipped["failed"] += 1
            return None
        caption = caption_of(post) if post is not None else rec.get("text") or ""
        out = {"file": stem + ".jpg", "txt": write_caption(dest, stem, caption),
               "caption": clean_caption(caption), "owner_id": rec["owner_id"], "id": rec["id"],
               "link": "https://vk.com/photo{}_{}".format(rec["owner_id"], rec["id"]),
               "width": rec.get("width"), "height": rec.get("height"),
               "size_type": rec.get("size_type"), "date": rec.get("date"),
               "url_used": rec.get("url_used")}
        if post is not None:
            copy = post.get("copy") or {}
            out.update(post_id=post.get("id"), post_date=post.get("date"),
                       repost_from=("{}_{}".format(copy.get("owner_id"), copy.get("id"))
                                    if copy else ""))
        return out
    return fetch


def run_sample(client: VkClient, target, out_root, opts: SampleOptions) -> dict:
    """Sample a group from its wall or an album; write the files, _sample.json
    and, with --sheet, sheet.jpg. Raises SampleError, SheetError."""
    out_root = Path(out_root)
    n = int(opts.n)
    if n < 1:
        raise SampleError("--n: нужно хотя бы одно фото.")
    if opts.source not in ("wall", "albums"):
        raise SampleError("--from: wall или albums.")
    group, entry = resolve_offline(out_root, target)
    if group is None:
        print("[i] Группы «{}» нет в {} и _groups.json: +1 вызов groups.getById.".format(
            target, CANDIDATES_NAME))
        try:
            found = resolve_groups(client, [target])
        except VkApiError as e:
            if e.limit:
                raise
            raise SampleError("Сообщество «{}» не найдено: {}".format(target, e)) from e
        if not found:
            raise SampleError("Сообщество «{}» не найдено.".format(target))
        group = found[0]
    elif entry is not None and _closed(entry) != "открыт":
        raise SampleError("Сообщество «{}» ({}): {}, стену и альбомы не прочитать; выборка "
                          "не сделана, вызовов 0.".format(group.name, group.id, _closed(entry)))

    gkey = assign_group_dir(out_root, group, DEFAULT_DIR_PATTERN, register=False)
    dest = out_root / SAMPLES_DIR / gkey
    dest.mkdir(parents=True, exist_ok=True)
    skipped = {"dead": 0, "small": 0, "failed": 0, "no_url": 0}
    fetch = _fetcher(client.session, dest, group, opts.min_side, skipped, set())
    meta = {"group": group.as_dict(), "source": opts.source, "date": now_iso(), "n": n,
            "min_side": opts.min_side}
    note = ""

    if opts.source == "wall":
        try:
            resp = client.call("wall.get", owner_id=group.owner_id, offset=0, count=WALL_PAGE,
                               filter="owner", extended=0)
        except VkApiError as e:
            if e.limit:
                raise
            raise SampleError("Стена «{}» недоступна: {}".format(group.name, e)) from e
        items = (resp or {}).get("items") or []
        total = int((resp or {}).get("count") or 0)
        posts = [normalize_post(it) for it in items]
        with_photos = [p for p in posts if p["photos"]]
        if not with_photos:
            raise SampleError("На первой странице стены «{}» ({} постов из {}) нет фото; "
                              "попробуйте --from albums.".format(group.name, len(items), total))
        got = pick(wall_plan(posts, n), n, fetch)
        meta["wall"] = {"count": total, "page_posts": len(items),
                        "posts_with_photos": len(with_photos),
                        "post_ids": [p["id"] for p in with_photos]}
        head = "со стены: страница {} постов, с фото {}, всего постов на стене {}".format(
            len(items), len(with_photos), _fmt(total))
        note = "scan wall этой группы: ~{} вызовов ({} постов по 100 за вызов).".format(
            1 + max(1, math.ceil(total / WALL_PAGE)), _fmt(total))
        refine = {"wall_posts": total}
    else:
        aid = str(opts.album_id) if opts.album_id else None
        albums = []
        if aid is None or aid.isdigit():
            try:
                albums = fetch_albums(client, group)
            except VkApiError as e:
                if e.limit:
                    raise
                raise SampleError("Альбомы «{}» недоступны: {}".format(group.name, e)) from e
        if aid is None:
            sized = [a for a in albums if a.size]
            if not sized:
                raise SampleError("У «{}» нет альбомов с фото; попробуйте --from wall.".format(
                    group.name))
            album = max(sized, key=lambda a: a.size or 0)
        else:
            album = next((a for a in albums if str(a.id) == aid), None)
        title = album.title if album else SPECIAL_ALBUMS.get(aid, "альбом {}".format(aid))
        size = album.size if album else None
        aid = str(album.id) if album else aid
        try:
            resp = client.call("photos.get", _live=True, owner_id=group.owner_id, album_id=aid,
                               photo_sizes=1, extended=1, rev=0, offset=0, count=PHOTOS_PER_PAGE)
        except VkApiError as e:
            if e.limit:
                raise
            raise SampleError("Альбом {} «{}» недоступен: {}".format(aid, title, e)) from e
        photos = (resp or {}).get("items") or []
        if not photos:
            raise SampleError("В альбоме {} «{}» нет фото.".format(aid, title))
        got = pick(album_plan(photos, n), n, fetch)
        meta["album"] = {"id": aid, "title": title, "size": size if size is not None
                         else (resp or {}).get("count"), "page_photos": len(photos),
                         "albums": len(albums)}
        head = "из альбома {} «{}» ({} фото, на странице {}); альбомов у группы {}".format(
            aid, title, _fmt(meta["album"]["size"]), len(photos), len(albums))
        if albums:
            est = 1 + math.ceil(len(albums) / ALBUMS_PER_PAGE) + sum(
                max(1, math.ceil((a.size or 1) / PHOTOS_PER_PAGE)) for a in albums)
            note = ("scan albums: ~{} вызова; все альбомы ({} фото): ~{} вызовов.".format(
                1 + max(1, math.ceil(len(albums) / ALBUMS_PER_PAGE)),
                _fmt(sum(a.size or 0 for a in albums)), est))
        refine = {"albums": len(albums)} if albums else {}

    records = [rec for _pos, rec in got]
    meta["records"] = records
    meta["skipped"] = skipped
    meta["sheet"] = None
    sheet_err = None
    if opts.sheet and records:
        subtitle = "{} участников · {} · {} фото · id {}".format(
            _fmt(group.members_count), "стена" if opts.source == "wall" else "альбом",
            len(records), group.id)
        try:
            render_sheet([{"path": dest / r["file"], "caption": r["caption"]} for r in records],
                         dest / SHEET_NAME, group.name, subtitle)
            meta["sheet"] = SHEET_NAME
        except SheetError as e:
            sheet_err = e
    write_json(dest / SAMPLE_NAME, meta)

    cand = load_candidates(out_root)
    ce = cand["groups"].get(str(group.id))
    if isinstance(ce, dict):
        if "wall_posts" in refine:
            ce["wall_posts"] = refine["wall_posts"]
        if "albums" in refine:
            ce.setdefault("counters", {})
            if isinstance(ce["counters"], dict):
                ce["counters"]["albums"] = refine["albums"]
        ce["sample"] = {"ts": now_iso(), "from": opts.source, "photos": len(records),
                        "folder": "{}/{}".format(SAMPLES_DIR, gkey)}
        write_json(out_root / CANDIDATES_NAME, cand)

    print("Выборка «{}» (id={}, участников {}): {}".format(
        group.name, group.id, _fmt(group.members_count), head))
    print("   фото: {} из {}; пропущено: мёртвых (404) {}, мелких {}, ошибок {}".format(
        len(records), n, skipped["dead"], skipped["small"], skipped["failed"] + skipped["no_url"]))
    for i, r in enumerate(records, 1):
        print("   {:>2}. {} | {}x{} | {} | {}".format(
            i, r["file"], r["width"], r["height"], (r.get("date") or "")[:10],
            r["caption"][:60]))
    if note:
        print(note)
    print("Папка: {}".format(dest))
    if meta["sheet"]:
        print("Лист: {}".format(dest / SHEET_NAME))
    if sheet_err is not None:
        raise sheet_err
    return {"dest": str(dest), "records": records, "skipped": skipped,
            "sheet": str(dest / SHEET_NAME) if meta["sheet"] else None}
