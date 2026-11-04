"""recheck-dead: re-try the dead photos of a group from _dead_photos.json.

No API call at all and no listing: the registry stores every size URL of a
dead photo, so the check touches the CDN only. Without --sizes the best copy
is tried (as a download does); with --sizes every stored size, largest
first. Three counts come out: every tried URL answered 404, some size gave
200 (the photo is recovered: file, .txt, manifest row, metadata.json, and the
entry leaves the registry), other failures (403, 5xx, network).

The group is found offline: a folder path, or the id, short name, folder or
name of a group in <out>/_groups.json.
"""
from __future__ import annotations

from pathlib import Path

from .albums import (load_dead, load_index, parse_group_ref, photo_stem,
                     read_manifest, save_dead, size_score, write_caption,
                     write_manifest)
from .budget import now_iso
from .fs import download_file, read_json, sanitize_dirname, write_json


class RecheckError(Exception):
    """The group folder or its registry cannot be found."""


def find_group_dir(out_root: Path, target: str) -> Path:
    """The group folder for `target`, without any API call."""
    out_root = Path(out_root)
    t = str(target or "").strip()
    for cand in (Path(t), out_root / t):
        if t and cand.is_dir() and (cand / "_dead_photos.json").is_file():
            return cand
    ref = parse_group_ref(t)
    reg = read_json(out_root / "_groups.json", None)
    groups = (reg or {}).get("groups") if isinstance(reg, dict) else None
    for gid, g in (groups or {}).items():
        g = g or {}
        if ref and ref in (str(gid), str(g.get("screen_name") or ""),
                           str(g.get("dir") or ""), str(g.get("name") or "")):
            return out_root / str(g.get("dir"))
    raise RecheckError(
        "Группа «{}» не найдена в {}: укажите папку группы, её id или короткое "
        "имя из _groups.json.".format(target, out_root / "_groups.json"))


def _urls(entry: dict, all_sizes: bool) -> list[dict]:
    best = entry.get("best_size") or {}
    if not all_sizes:
        return [best] if best.get("url") else []
    sizes = [s for s in (entry.get("sizes") or []) if s.get("url")]
    sizes.sort(key=size_score, reverse=True)   # dimensionless (pre-2012): by type
    if best.get("url") and all(s["url"] != best["url"] for s in sizes):
        sizes.insert(0, best)
    return sizes


def _album_dir(gdir: Path, out_root: Path, entry: dict,
               row: dict | None = None) -> tuple[Path, str]:
    """(absolute folder, folder relative to the dataset root) of the album.

    The manifest row's album_dir first (a wall post's month folder), then
    the album index, then a folder named after the album title.
    """
    if row and row.get("album_dir"):
        return out_root / row["album_dir"], str(row["album_dir"])
    aid = str(entry.get("album_id"))
    for e in load_index(gdir)["albums"]:
        if str(e.get("id")) == aid and e.get("dir"):
            return out_root / e["dir"], str(e["dir"])
    rel = "{}/{}".format(gdir.name, sanitize_dirname(entry.get("album_title") or "")
                         or "album_{}".format(aid))
    return out_root / rel, rel


def _update_metadata(album_dir: Path, entry: dict, fname: str, size: dict,
                     nbytes: int, txt: str) -> None:
    path = album_dir / "metadata.json"
    meta = read_json(path, None)
    if not isinstance(meta, dict):
        return
    for rec in meta.get("photos") or []:
        if str(rec.get("id")) == str(entry.get("photo_id")):
            rec.update(status="ok", file=fname, txt=txt, bytes=nbytes,
                       url_used=size.get("url"), size_type=size.get("type"),
                       width=size.get("width"), height=size.get("height"),
                       recovered=now_iso())
    write_json(path, meta)


def run_recheck_dead(session, out_root, target, limit: int | None = None,
                     all_sizes: bool = False, retry_sleep: float = 1.5) -> dict:
    out_root = Path(out_root)
    gdir = find_group_dir(out_root, target)
    out_root = gdir.parent
    dead = load_dead(gdir)
    keys = list(dead)[:limit] if limit else list(dead)
    counts = {"checked": 0, "all_404": 0, "recovered": 0, "other": 0}
    rows = read_manifest(gdir)
    recovered = []
    print("Перепроверка мёртвых фото «{}»: {} из {} записей реестра, копии: {}".format(
        gdir.name, len(keys), len(dead), "все размеры" if all_sizes else "лучшая"))
    for key in keys:
        entry = dead[key]
        counts["checked"] += 1
        rk = (str(entry.get("album_id")), str(entry.get("photo_id")))
        album_dir, rel_dir = _album_dir(gdir, out_root, entry, rows.get(rk))
        owner = entry.get("owner_id") or -int(entry.get("group_id") or 0)
        fname = photo_stem(owner, entry.get("photo_id")) + ".jpg"
        statuses, got = [], None
        for size in _urls(entry, all_sizes):
            album_dir.mkdir(parents=True, exist_ok=True)
            res = download_file(session, size["url"], album_dir / fname,
                                retry_sleep=retry_sleep)
            statuses.append(res["status"])
            if res["status"] == "ok":
                got = (size, res)
                break
        if got:
            size, res = got
            counts["recovered"] += 1
            txt = write_caption(album_dir, Path(fname).stem, entry.get("text") or "")
            nbytes = res.get("bytes") or (album_dir / fname).stat().st_size
            row = rows.get(rk) or {
                "photo_id": entry.get("photo_id"), "owner_id": owner,
                "group_id": entry.get("group_id"), "album_id": entry.get("album_id"),
                "album_title": entry.get("album_title"), "date": entry.get("date"),
                "likes": entry.get("likes"), "comments": entry.get("comments"),
                "text": entry.get("text") or ""}
            row.update({"album_dir": rel_dir, "file": fname,
                        "path": "{}/{}".format(rel_dir, fname), "txt": txt,
                        "status": "ok", "size_type": size.get("type"),
                        "width": size.get("width"), "height": size.get("height"),
                        "bytes": nbytes})
            rows[rk] = row
            _update_metadata(album_dir, entry, fname, size, nbytes, txt)
            recovered.append(key)
            print("   восстановлено: {} ({}x{}, копия {})".format(
                fname, size.get("width"), size.get("height"), size.get("type")))
            continue
        result = "all_404" if statuses and all(x == "dead" for x in statuses) else "other"
        counts[result] += 1
        entry["last_checked"] = now_iso()
        entry["last_recheck"] = {"ts": now_iso(), "result": result,
                                 "sizes": "all" if all_sizes else "best",
                                 "statuses": statuses}
    for key in recovered:
        dead.pop(key, None)
    save_dead(gdir, dead)
    if recovered:
        write_manifest(gdir, rows)
    print("Итог: все копии 404: {} | есть копия 200 (восстановлено): {} | "
          "другие ошибки: {}".format(counts["all_404"], counts["recovered"],
                                     counts["other"]))
    print("Вызовов API: 0 (только CDN). Реестр: {}".format(gdir / "_dead_photos.json"))
    return counts
