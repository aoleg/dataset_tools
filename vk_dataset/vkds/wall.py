"""The wall of a group: `scan wall` builds _wall.json, `wall` downloads.

Two steps, as for albums:

- `run_scan_wall` pages wall.get newest first (count=100, filter=owner, or
  all with --all, extended=0) and keeps every post in <group>/_wall.json,
  keyed by post id: the full text, the deepest original of a repost
  (copy_history[-1]: owner, id, date, text), is_pinned, date, likes,
  reposts, comments and the photo attachments (owner, id, sizes). The pinned
  post is kept whatever its date and ignored by the stop rule. The scan
  stops when the last non-pinned post of a page is older than --since, at
  the end of the wall, after --limit posts, or at a post already in the
  index (an incremental rescan) unless --refresh. When the index already
  covers the newest posts and --since asks for older ones, the scan jumps
  below the known region (offset of the known post plus the known posts
  under it, minus a small overlap) instead of paging through it.
- `run_wall` selects posts from the index without any call (--since,
  --until, --text-regex over the own and the original text, --photos-min,
  --post-id) and lists the selected posts that still miss photos live with
  wall.getById, 100 per call, so the download URLs are fresh (D7) and
  --until costs nothing extra. It never calls wall.get.

Files go to <group>/wall/<YYYY-MM>/photo<owner>_<id>.jpg, the month of the
group's post (UTC), the owner of the photo itself (the original owner of a
repost). The .txt of every photo of a post is the cleaned own text of the
post, or the original's text when the own text is empty after cleaning;
an existing .txt is never rewritten. A photo whose (owner, id) already has a
file in the group's manifest (an album, the system wall album, an earlier
post) is not downloaded again: its wall row gets status dup and the path of
the existing file. Rows of wall photos carry album_id post_<post_id> and
the post columns of the manifest.

What the tool prints stays Russian (D2).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .albums import (DEAD_TTL_DAYS, DEFAULT_DIR_PATTERN, assign_group_dir,
                     download_photos, load_dead, make_dead_entry, make_record,
                     read_manifest, register_dead, resolve_groups,
                     row_done, save_dead, update_manifest, write_caption)
from .api import VkClient
from .budget import now_iso
from .captions import clean_caption
from .fs import read_json, write_json
from .models import AlbumInfo, GroupInfo
from .recheck import RecheckError, find_group_dir

WALL_PAGE = 100          # maximum of wall.get
GETBYID_PAGE = 100       # posts per wall.getById
JUMP_OVERLAP = 5         # posts of overlap when jumping below the known region
WALL_NAME = "_wall.json"


class WallError(Exception):
    """No index, a bad date or an unknown group: a readable refusal."""


# ----------------------------------------------------------------------------
# Dates and posts
# ----------------------------------------------------------------------------

def parse_day(value: str | None, end: bool = False) -> int | None:
    """YYYY-MM-DD as a Unix time (UTC): the day's first second, or with
    end=True its last second (the bounds are inclusive calendar days)."""
    if not value:
        return None
    try:
        d = date.fromisoformat(str(value).strip())
    except ValueError as e:
        raise WallError("Дата «{}»: нужен формат ГГГГ-ММ-ДД.".format(value)) from e
    t = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    if end:
        t += timedelta(days=1, seconds=-1)
    return int(t.timestamp())


def month_of(unix) -> str:
    return datetime.fromtimestamp(int(unix or 0), tz=timezone.utc).strftime("%Y-%m")


def _photos_of(attachments) -> list[dict]:
    out = []
    for a in attachments or []:
        if (a or {}).get("type") == "photo" and isinstance(a.get("photo"), dict):
            out.append(a["photo"])
    return out


def _slim_photo(ph: dict) -> dict:
    return {"owner_id": ph.get("owner_id"), "id": ph.get("id"),
            "date": ph.get("date"), "text": str(ph.get("text") or ""),
            "sizes": [{k: s.get(k) for k in ("type", "width", "height", "url")}
                      for s in (ph.get("sizes") or []) if s.get("url")]}


def post_photos(item: dict) -> list[dict]:
    """The photo attachments of a post and of every level of its repost
    chain, each (owner, id) once, in order."""
    seen, out = set(), []
    for src in [item] + list(item.get("copy_history") or []):
        for ph in _photos_of(src.get("attachments")):
            key = (ph.get("owner_id"), ph.get("id"))
            if key not in seen:
                seen.add(key)
                out.append(ph)
    return out


def normalize_post(item: dict) -> dict:
    """A wall.get / wall.getById item -> the _wall.json entry."""
    chain = item.get("copy_history") or []
    orig = chain[-1] if chain else None
    return {
        "id": item.get("id"),
        "owner_id": item.get("owner_id"),
        "from_id": item.get("from_id"),
        "date": item.get("date"),
        "text": str(item.get("text") or ""),
        "is_pinned": bool(item.get("is_pinned")),
        "likes": (item.get("likes") or {}).get("count"),
        "reposts": (item.get("reposts") or {}).get("count"),
        "comments": (item.get("comments") or {}).get("count"),
        "copy": ({"owner_id": orig.get("owner_id"), "id": orig.get("id"),
                  "date": orig.get("date"), "text": str(orig.get("text") or "")}
                 if orig else None),
        "photos": [_slim_photo(ph) for ph in post_photos(item)],
    }


def caption_of(post: dict) -> str:
    """The raw text that captions the photos of a post: the own text when
    something is left after cleaning, else the original's of a repost."""
    own = post.get("text") or ""
    if clean_caption(own):
        return own
    return ((post.get("copy") or {}).get("text") or "") if post.get("copy") else own


def load_wall(gdir: Path) -> dict:
    idx = read_json(Path(gdir) / WALL_NAME, None)
    if not isinstance(idx, dict):
        idx = {}
    if not isinstance(idx.get("posts"), dict):
        idx["posts"] = {}
    return idx


def _sorted_posts(posts: dict) -> list[dict]:
    return sorted(posts.values(), key=lambda p: -int(p.get("date") or 0))


# ----------------------------------------------------------------------------
# scan wall
# ----------------------------------------------------------------------------

@dataclass
class ScanWallOptions:
    since: str | None = None
    until: str | None = None
    limit: int | None = None
    include_all: bool = False       # filter=all instead of owner
    refresh: bool = False           # do not stop at known posts


def _known_below(posts: dict, date_unix: int) -> int:
    return sum(1 for p in posts.values()
               if not p.get("is_pinned") and int(p.get("date") or 0) <= int(date_unix or 0))


def run_scan_wall(client: VkClient, target, out_root, opts: ScanWallOptions) -> dict:
    since_ts = parse_day(opts.since)
    until_ts = parse_day(opts.until, end=True)
    out_root = Path(out_root)
    group = resolve_groups(client, [target])[0]
    gkey = assign_group_dir(out_root, group, DEFAULT_DIR_PATTERN)
    gdir = out_root / gkey
    idx = load_wall(gdir)
    posts = idx["posts"]
    old_covered = idx.get("covered_to")
    old_end = bool(idx.get("reached_end"))
    flt = "all" if opts.include_all else "owner"

    offset, paged, wall_calls = 0, 0, 0
    scan_oldest, met_known, jumped, hit_end = None, False, False, False
    new_posts = 0
    while True:
        resp = client.call("wall.get", owner_id=group.owner_id, offset=offset,
                           count=WALL_PAGE, filter=flt, extended=0)
        wall_calls += 1
        items = resp.get("items", []) or []
        total = int(resp.get("count") or 0)
        if not items:
            hit_end = True
            break
        stop, jump_to, last_np = False, None, None
        for pos, it in enumerate(items):
            post = normalize_post(it)
            key = str(post["id"])
            known = key in posts
            if not known:
                new_posts += 1
            posts[key] = post
            if post["is_pinned"]:
                continue
            paged += 1
            last_np = int(post["date"] or 0)
            scan_oldest = last_np if scan_oldest is None else min(scan_oldest, last_np)
            if known and not opts.refresh and not met_known:
                met_known = True
                deeper = (old_covered is not None and not old_end
                          and (since_ts is None or since_ts < int(old_covered)))
                if not deeper:
                    stop = True
                    break
                jumped = True
                jump_to = max(offset + pos + 1,
                              offset + pos + _known_below(posts, last_np) - JUMP_OVERLAP)
                break
            if opts.limit and paged >= opts.limit:
                stop = True
                break
        if stop:
            break
        if jump_to is not None:
            offset = jump_to
            continue
        if since_ts is not None and last_np is not None and last_np < since_ts:
            break
        offset += len(items)
        if offset >= total:
            hit_end = True
            break

    if met_known and not opts.refresh and old_covered is not None:
        covered = min(int(old_covered), scan_oldest) if scan_oldest is not None else old_covered
    else:
        covered = scan_oldest
    idx.update({"group": group.as_dict(), "filter": flt, "updated": now_iso(),
                "covered_to": covered,
                "reached_end": bool(hit_end or (met_known and not jumped and old_end))})
    gdir.mkdir(parents=True, exist_ok=True)
    write_json(gdir / WALL_NAME, idx)
    write_json(gdir / "_group.json", group.as_dict())

    sel = [p for p in posts.values()
           if (since_ts is None or int(p["date"] or 0) >= since_ts or p["is_pinned"])
           and (until_ts is None or int(p["date"] or 0) <= until_ts)]
    by_year_posts, by_year_photos = Counter(), Counter()
    for p in sel:
        y = month_of(p["date"])[:4]
        by_year_posts[y] += 1
        by_year_photos[y] += len(p["photos"])
    with_photos = [p for p in sel if p["photos"]]
    est = math.ceil(len(with_photos) / GETBYID_PAGE)
    print("Стена «{}» (id={}, папка «{}»): вызовов wall.get {}, новых постов {}, "
          "всего в индексе {}".format(group.name, group.id, gkey,
                                      wall_calls, new_posts, len(posts)))
    print("Посты {}{}: {}, с фото: {}, фото: {}".format(
        "с {} ".format(opts.since) if opts.since else "",
        "по {}".format(opts.until) if opts.until else "", len(sel), len(with_photos),
        sum(len(p["photos"]) for p in sel)))
    for y in sorted(by_year_posts):
        print("   {}: постов {:>6}, фото {:>6}".format(y, by_year_posts[y], by_year_photos[y]))
    print("Скачивание этих постов: ~{} вызовов wall.getById (по 100 постов; "
          "файлы с CDN — бесплатно).".format(est))
    print("Индекс: {}".format(gdir / WALL_NAME))
    return {"group": group.as_dict(), "posts": len(posts), "new": new_posts,
            "selected": len(sel), "with_photos": len(with_photos), "est_calls": est,
            "covered_to": covered, "reached_end": idx["reached_end"]}


# ----------------------------------------------------------------------------
# wall (download)
# ----------------------------------------------------------------------------

@dataclass
class WallOptions:
    since: str | None = None
    until: str | None = None
    text_regex: str = ""
    photos_min: int = 1
    post_ids: list = field(default_factory=list)
    min_area: int = 0
    workers: int = 4
    dry_run: bool = False
    recheck_dead: bool = False
    dead_ttl_days: int = DEAD_TTL_DAYS
    dl_retry_sleep: float = 1.5


def select_posts(idx: dict, opts: WallOptions) -> list[dict]:
    """Posts of the index that pass the filters, oldest first (no call)."""
    since_ts = parse_day(opts.since)
    until_ts = parse_day(opts.until, end=True)
    rx = re.compile(opts.text_regex, re.IGNORECASE) if opts.text_regex else None
    wanted = {str(x) for x in (opts.post_ids or [])}
    out = []
    for p in idx["posts"].values():
        d = int(p.get("date") or 0)
        if wanted and str(p.get("id")) not in wanted:
            continue
        if since_ts is not None and d < since_ts:
            continue
        if until_ts is not None and d > until_ts:
            continue
        if len(p.get("photos") or []) < max(1, opts.photos_min):
            continue
        if rx and not (rx.search(p.get("text") or "")
                       or rx.search((p.get("copy") or {}).get("text") or "")):
            continue
        out.append(p)
    return sorted(out, key=lambda p: (int(p.get("date") or 0), int(p.get("id") or 0)))


def _existing_files(rows: dict, out_root: Path) -> dict:
    """(owner, photo_id) -> (album_id, path) of every photo with a file."""
    have = {}
    for (aid, pid), r in rows.items():
        if r.get("status") in ("ok", "exists") and r.get("path") \
                and (out_root / r["path"]).is_file():
            try:
                have.setdefault((int(float(r.get("owner_id"))), str(pid)), (aid, r["path"]))
            except (TypeError, ValueError):
                continue
    return have


def _post_settled(post: dict, rows_by_album: dict, out_root: Path, dead: dict,
                  opts: WallOptions, have: dict) -> bool:
    """True when every photo of the post needs no call: its wall row is
    settled (file, fresh dead entry, filtered), or it is a dup of a file
    elsewhere in the group."""
    aid = "post_{}".format(post["id"])
    rows = {str(r.get("photo_id")): r for r in rows_by_album.get(aid, [])}
    for ph in post.get("photos") or []:
        r = rows.get(str(ph.get("id")))
        if r is not None and row_done(r, out_root, dead, opts):
            continue
        hit = have.get((int(ph.get("owner_id") or 0), str(ph.get("id"))))
        if hit and hit[0] != aid:
            continue
        return False
    return True


def _resolve_offline(out_root: Path, target) -> tuple[Path, dict]:
    try:
        gdir = find_group_dir(out_root, target)
    except RecheckError:
        gdir = None
    if gdir is None or not (gdir / WALL_NAME).is_file():
        raise WallError("Нет индекса стены {} для «{}»: сначала run.bat scan wall {} "
                        "(вызов wall.get — 1 на 100 постов).".format(
                            WALL_NAME, target, target))
    return gdir, load_wall(gdir)


def run_wall(client: VkClient | None, target, out_root, opts: WallOptions) -> dict:
    """Download the photos of the selected posts. client may be None for
    --dry-run, which makes no call at all."""
    out_root = Path(out_root)
    gdir, idx = _resolve_offline(out_root, target)
    out_root = gdir.parent
    gkey = gdir.name
    g = idx.get("group") or {}
    group = GroupInfo(id=int(g.get("id") or 0), name=str(g.get("name") or gkey),
                      screen_name=str(g.get("screen_name") or ""))
    sel = select_posts(idx, opts)
    dead = load_dead(gdir)
    rows = read_manifest(gdir)
    rows_by_album: dict = {}
    for (aid, _pid), r in rows.items():
        rows_by_album.setdefault(aid, []).append(r)
    have = _existing_files(rows, out_root)
    todo = [p for p in sel if not _post_settled(p, rows_by_album, out_root, dead, opts, have)]
    est = math.ceil(len(todo) / GETBYID_PAGE)
    n_photos = sum(len(p["photos"]) for p in sel)
    months = Counter(month_of(p["date"]) for p in sel)
    print("\n=== Стена «{}» (id={}) ===".format(group.name, group.id))
    print("  Отобрано постов: {}, фото в них: {}, месяцев: {}; без недостающих "
          "фото: {}".format(len(sel), n_photos, len(months), len(sel) - len(todo)))

    if opts.dry_run:
        _print_wall_dry_run(sel, todo, months, est, gkey)
        return {"group": group.as_dict(), "dry_run": True, "selected": len(sel),
                "photos": n_photos, "todo": len(todo), "est_calls": est,
                "months": dict(months)}

    live = {}
    for i in range(0, len(todo), GETBYID_PAGE):
        chunk = todo[i:i + GETBYID_PAGE]
        ids = ",".join("{}_{}".format(p.get("owner_id") or group.owner_id, p["id"])
                       for p in chunk)
        resp = client.call("wall.getById", posts=ids, extended=0)
        items = resp if isinstance(resp, list) else (resp or {}).get("items", [])
        for it in items or []:
            live[str(it.get("id"))] = it

    gsum = {"group": group.as_dict(), "posts": len(sel), "photos": 0, "downloaded": 0,
            "exists": 0, "dup": 0, "small": 0, "failed": 0, "no_url": 0, "dead": 0,
            "forbidden": 0, "gone": 0}
    todo_ids = {str(p["id"]) for p in todo}
    for post in sel:
        _wall_post(client, group, gdir, out_root, post, live.get(str(post["id"])),
                   opts, dead, rows_by_album, have, gsum,
                   fetched=str(post["id"]) in todo_ids)
    save_dead(gdir, dead)
    for k, v in live.items():
        new = normalize_post(v)
        new["is_pinned"] = new["is_pinned"] or bool((idx["posts"].get(k) or {}).get("is_pinned"))
        idx["posts"][k] = new
    write_json(gdir / WALL_NAME, idx)
    print("  Итого по стене «{}»: постов {}, фото {}, скачано {}, уже было {}, "
          "дубли {}, мёртвых (404) {}, отфильтровано {}, ошибок {}, постов удалено "
          "{}".format(gkey, len(sel), gsum["photos"], gsum["downloaded"],
                      gsum["exists"], gsum["dup"], gsum["dead"], gsum["small"],
                      gsum["failed"] + gsum["no_url"] + gsum["forbidden"], gsum["gone"]))
    print("  Вызовов wall.getById: {} | манифест: {}".format(est, gdir / "_manifest.csv"))
    return gsum


def _wall_post(client, group: GroupInfo, gdir: Path, out_root: Path, post: dict,
               live_item: dict | None, opts: WallOptions, dead: dict,
               rows_by_album: dict, have: dict, gsum: dict, fetched: bool) -> None:
    aid = "post_{}".format(post["id"])
    rel_dir = "{}/wall/{}".format(gkey_of(gdir), month_of(post["date"]))
    dest_dir = out_root / rel_dir
    if fetched and live_item is None:
        gsum["gone"] += 1           # deleted since the scan: nothing to list
        print("  [!] Пост {} не вернулся из wall.getById (удалён?) — пропуск".format(post["id"]))
        return
    source = normalize_post(live_item) if live_item is not None else post
    if live_item is not None:
        photos = post_photos(live_item)
    else:
        photos = [dict(ph, sizes=ph.get("sizes") or []) for ph in post.get("photos") or []]
    raw = caption_of(source)
    title = clean_caption(raw)[:60]
    album = AlbumInfo(id=aid, title=title, special=True)
    extra = {"post_id": post["id"], "post_date": _iso(post["date"]),
             "repost_from": ("{}_{}".format(source["copy"]["owner_id"], source["copy"]["id"])
                             if source.get("copy") else ""),
             "reposts": source.get("reposts"), "likes": source.get("likes"),
             "comments": source.get("comments")}
    old = {str(r.get("photo_id")): r for r in rows_by_album.get(aid, [])}
    records, to_download = [], []
    for ph in photos:
        rec = make_record(ph)
        if rec.get("owner_id") in (None, ""):
            rec["owner_id"] = group.owner_id
        rec["photo_text"] = rec.get("text") or ""
        rec["text"] = raw
        r = old.get(str(rec["id"]))
        hit = have.get((int(rec["owner_id"]), str(rec["id"])))
        if hit and hit[0] != aid:
            rec["status"] = "dup"
            rec["dup_path"] = hit[1]
            existing = out_root / hit[1]
            rec["txt"] = write_caption(existing.parent, existing.stem, raw)
            rec["file"] = existing.name
        elif not fetched and r is not None and row_done(r, out_root, dead, opts):
            rec.update(status=r.get("status") if r.get("status") != "ok" else "exists",
                       file=r.get("file") or "", txt=r.get("txt") or "",
                       bytes=r.get("bytes"))
            if rec["status"] == "dead":
                rec["dead_cached"] = True
        else:
            to_download.append(rec)
        records.append(rec)
    if to_download:
        dest_dir.mkdir(parents=True, exist_ok=True)
        download_photos(client.session, to_download, dest_dir, workers=opts.workers,
                        min_area=opts.min_area, retry_sleep=opts.dl_retry_sleep,
                        dead=dead, recheck_dead=opts.recheck_dead,
                        dead_ttl_days=opts.dead_ttl_days, write_txt=True)
    by_id = {(p.get("owner_id"), p.get("id")): p for p in photos}
    for rec in records:
        st = rec.get("status")
        if st == "dead" and not rec.get("dead_cached"):
            ph = by_id.get((rec.get("owner_id"), rec.get("id")), {})
            register_dead(dead, make_dead_entry(group, album, ph, urls_live=True))
        if st in ("ok", "exists") and rec.get("file"):
            have.setdefault((int(rec["owner_id"]), str(rec["id"])),
                            (aid, "{}/{}".format(rel_dir, rec["file"])))
        key = {"ok": "downloaded"}.get(st, st)
        if key in gsum:
            gsum[key] += 1
        gsum["photos"] += 1
    update_manifest(gdir, out_root, group, album, records, rel_dir, extra=extra)


def gkey_of(gdir: Path) -> str:
    return Path(gdir).name


def _iso(unix) -> str:
    return datetime.fromtimestamp(int(unix or 0), tz=timezone.utc).isoformat()


def _print_wall_dry_run(sel: list, todo: list, months: Counter, est: int, gkey: str) -> None:
    print("  Dry-run: ничего не скачивается и не вызывается.")
    todo_ids = {str(p["id"]) for p in todo}
    for p in sel[:30]:
        mark = "" if str(p["id"]) in todo_ids else " (всё на месте)"
        print("   пост {:>8} {} фото {:>2} -> {}/wall/{}{} | {}".format(
            p["id"], _iso(p["date"])[:10], len(p["photos"]), gkey, month_of(p["date"]),
            mark, clean_caption(caption_of(p))[:50]))
    if len(sel) > 30:
        print("   ... ещё постов: {}".format(len(sel) - 30))
    for m in sorted(months):
        print("   {}: постов {}".format(m, months[m]))
    print("  Постов к живому списку: {} из {} | примерно вызовов wall.getById: {} "
          "(по 100 постов; файлы с CDN — бесплатно)".format(len(todo), len(sel), est))
