"""migrate: rename a dataset of vkds before 0.5 to photo<owner>_<id> names.

For each group folder (one with _manifest.csv) the manifest and every
album's metadata.json give the photo id and its owner; photo_<id>.* becomes
photo<owner>_<id>.*, every sidecar (.txt, .json, ...) with its image, and both
files are rewritten. A target name that already exists is skipped and
reported, the old file stays. --dry-run lists what would move and changes
nothing. No API call. The dead registry keeps its keys (<owner>_<photo_id>).
"""
from __future__ import annotations

import re
from pathlib import Path

from .albums import MANIFEST_NAME, old_stem, photo_stem, read_manifest, write_manifest
from .fs import read_json, write_json

_OLD = re.compile(r"^photo_(\d+)$")


class MigrateError(Exception):
    """No group folder with a manifest under the given path."""


def group_dirs(dataset: Path) -> list[Path]:
    dataset = Path(dataset)
    if (dataset / MANIFEST_NAME).is_file():
        return [dataset]
    found = sorted(d for d in dataset.iterdir()
                   if d.is_dir() and (d / MANIFEST_NAME).is_file()) if dataset.is_dir() else []
    if not found:
        raise MigrateError("В {} нет _manifest.csv ни в самой папке, ни в "
                           "папках групп.".format(dataset))
    return found


def _owner(row_owner, group_id) -> int | None:
    for v in (row_owner, ):
        try:
            if str(v).strip() not in ("", "None"):
                return int(float(str(v)))
        except ValueError:
            pass
    try:
        return -abs(int(float(str(group_id))))
    except ValueError:
        return None


def _sidecars(album_dir: Path, pid) -> list[Path]:
    """Every file of the album folder whose name is photo_<id>.<something>."""
    prefix = old_stem(pid) + "."
    try:
        return sorted(p for p in album_dir.iterdir()
                      if p.is_file() and p.name.startswith(prefix)
                      and not p.name.endswith(".part"))
    except OSError:
        return []


def migrate_group(gdir: Path, dry_run: bool = False) -> dict:
    gdir = Path(gdir)
    out_root = gdir.parent
    rows = read_manifest(gdir)
    counts = {"photos": 0, "files": 0, "skipped": 0, "already": 0}
    moved = {}        # (album_dir, old file name) -> new file name
    plan = []         # (src, dst)
    seen = set()
    for row in rows.values():
        fname = row.get("file") or ""
        stem = Path(fname).stem
        if not fname or not _OLD.match(stem):
            if fname:
                counts["already"] += 1
            continue
        pid = _OLD.match(stem).group(1)
        owner = _owner(row.get("owner_id"), row.get("group_id"))
        if owner is None:
            counts["skipped"] += 1
            print("   [!] нет владельца у фото {}: пропуск".format(pid))
            continue
        album_dir = out_root / (row.get("album_dir") or "")
        if (album_dir, pid) in seen:
            continue
        seen.add((album_dir, pid))
        new_stem = photo_stem(owner, pid)
        files = _sidecars(album_dir, pid)
        if not files:
            counts["skipped"] += 1
            print("   [!] нет файлов {} в {}: пропуск".format(fname, album_dir))
            continue
        clash = [f for f in files
                 if (album_dir / (new_stem + f.name[len(old_stem(pid)):])).exists()]
        if clash:
            counts["skipped"] += 1
            print("   [!] уже есть {}: пропуск {}".format(
                new_stem + clash[0].name[len(old_stem(pid)):], clash[0].name))
            continue
        counts["photos"] += 1
        for f in files:
            dst = album_dir / (new_stem + f.name[len(old_stem(pid)):])
            plan.append((f, dst))
            moved[(str(album_dir), f.name)] = dst.name
    for src, dst in plan:
        print("   {} -> {}".format(src.relative_to(out_root).as_posix(), dst.name))
        counts["files"] += 1
        if not dry_run:
            src.rename(dst)
    if dry_run or not plan:
        return counts
    for row in rows.values():
        album_dir = str(out_root / (row.get("album_dir") or ""))
        for col in ("file", "txt"):
            new = moved.get((album_dir, row.get(col) or ""))
            if new:
                row[col] = new
        if row.get("file") and row.get("album_dir"):
            row["path"] = "{}/{}".format(row["album_dir"], row["file"])
    write_manifest(gdir, rows)
    for album_dir in sorted({a for a, _ in moved}):
        path = Path(album_dir) / "metadata.json"
        meta = read_json(path, None)
        if not isinstance(meta, dict):
            continue
        changed = False
        for rec in meta.get("photos") or []:
            for col in ("file", "txt"):
                new = moved.get((album_dir, rec.get(col) or ""))
                if new:
                    rec[col] = new
                    changed = True
        if changed:
            write_json(path, meta)
    return counts


def run_migrate(dataset, dry_run: bool = False) -> dict:
    total = {"photos": 0, "files": 0, "skipped": 0, "already": 0}
    for gdir in group_dirs(Path(dataset)):
        print("Группа «{}»{}:".format(gdir.name, " (dry-run, ничего не меняется)"
                                      if dry_run else ""))
        c = migrate_group(gdir, dry_run=dry_run)
        for k in total:
            total[k] += c[k]
    print("Итог: фото {} ({} файлов) {}, пропущено {}, уже в новой форме {}".format(
        total["photos"], total["files"],
        "будет переименовано" if dry_run else "переименовано",
        total["skipped"], total["already"]))
    return total
