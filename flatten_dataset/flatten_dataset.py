#!/usr/bin/env python3
"""
Flatten a dataset folder: move every image and its captions out of the
subfolders into the folder itself, under new normalised names.

The new name of a file is made of the folder path and a running number:

    <folder>__<subfolder1>__<subfolder2>-datasetNNNNNN.<ext>

An image and its captions (files with the same name and another extension, .txt
by default) form a group and get the same new name, so they stay a pair. In the
folder names, Cyrillic is transliterated, accents are dropped, and spaces and
every other character that is not a Latin letter, a digit or "-" become "_".
No new file name is longer than --max-length characters (80 by default): the
deepest subfolder names are cut first. Files that belong to no image stay where
they are.

Every rename is planned before the first one and written to
<folder>/_flatten_dataset/manifest.json; --undo reads it and puts every file
back under its old name in its old subfolder.

Usage:    python flatten_dataset.py <folder> [<folder> ...] [--dry-run] [--undo]
          [--max-length N] [--sidecars LIST] [--exclude NAME]
Install:  nothing; Python 3.10 or newer
"""
import argparse
import csv
import json
import os
import re
import secrets
import sys
import time
import unicodedata
from pathlib import Path

TOOL_DIRNAME = "_flatten_dataset"  # in the root: manifest.json, dry_run.csv
MANIFEST_NAME = "manifest.json"    # the plan of the last run; --undo reads it
DRY_RUN_NAME = "dry_run.csv"
MANIFEST_VERSION = 1
DEFAULT_MAX_LENGTH = 80            # characters of a new file name, extension included
DEFAULT_SIDECARS = ".txt"
NUMBER_PREFIX = "-dataset"
MIN_DIGITS = 6
SEPARATOR = "__"                   # between folder names; a name part never holds "__" itself
MIN_PART = 8                       # a subfolder name is cut to this before a shallower one is cut
FALLBACK_PART = "folder"           # a folder name with nothing left after cleaning
MASKS_DIRNAME = "masks"            # face_masks: <dir>/masks/<stem>.png
TMP_PREFIX = ".flatten-"
PROGRESS_EVERY = 50000             # renames between progress lines
LIST_LIMIT = 20                    # lines of any list printed

IMAGE_EXTS = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff",
              ".gif", ".avif", ".heic", ".heif"}
# Folders never flattened, at any depth: this tool's own folder and the output
# folders of the other dataset tools. A masks folder is not flattened as images;
# the mask of an image moves with it to <root>/masks.
DEFAULT_EXCLUDES = [TOOL_DIRNAME, "_backup", "_duplicates", "_prep", "_classify", "_embeddings",
                    MASKS_DIRNAME, "faces"]

CYRILLIC = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
    # Ukrainian and Belarusian letters
    "є": "ye", "і": "i", "ї": "yi", "ґ": "g", "ў": "u",
}


# --- names --------------------------------------------------------------------

def transliterate(text: str) -> str:
    """Cyrillic to Latin; Latin letters lose their accents; any other character
    that is not ASCII becomes "_"."""
    text = unicodedata.normalize("NFC", text)   # a name written as и + breve is й
    out = []
    for i, ch in enumerate(text):
        latin = CYRILLIC.get(ch.lower())
        if latin is None:
            out.append(ch)
            continue
        if ch.isupper() and latin:
            # ЖУК -> ZHUK, Жук -> Zhuk
            near_upper = (i + 1 < len(text) and text[i + 1].isupper()) or (i > 0 and text[i - 1].isupper())
            latin = latin.upper() if near_upper else latin.capitalize()
        out.append(latin)
    out2 = []
    for ch in unicodedata.normalize("NFKD", "".join(out)):
        if ch.isascii():
            out2.append(ch)
        elif not unicodedata.combining(ch):
            out2.append("_")
    return "".join(out2)


def clean_name(name: str) -> str:
    """A folder name as a part of a file name: Latin letters, digits, "-" and
    single "_", never "_" or "-" at either end."""
    name = re.sub(r"[^A-Za-z0-9-]+", "_", transliterate(name))
    return name.strip("_-") or FALLBACK_PART


def cut(part: str, n: int) -> str:
    return part[:n].rstrip("_-") or part[:n]


def name_prefix(parts: list[str], budget: int) -> str:
    """The folder names joined by "__", at most budget characters long. The
    deepest subfolder name is cut first, down to MIN_PART characters, then the
    one above it; when that is not enough, the deepest subfolders are dropped,
    and the root name is cut last."""
    parts = list(parts)

    def length() -> int:
        return sum(map(len, parts)) + len(SEPARATOR) * (len(parts) - 1)

    for i in range(len(parts) - 1, 0, -1):
        excess = length() - budget
        if excess <= 0:
            break
        parts[i] = cut(parts[i], max(MIN_PART, len(parts[i]) - excess))
    while length() > budget and len(parts) > 1:
        parts.pop()
    if length() > budget:
        parts[0] = cut(parts[0], budget)
    return SEPARATOR.join(parts)


def natural_key(name: str):
    """img2 before img10."""
    return [(0, int(t), "") if t.isdigit() else (1, 0, t) for t in re.split(r"(\d+)", name.casefold()) if t]


# --- scan ---------------------------------------------------------------------

class Group:
    """An image (or several images with one stem) and its sidecars in one folder."""
    def __init__(self, dir_parts: tuple, names: list[str], mask: str | None):
        self.dir_parts = dir_parts    # folder names from the root, as on disk
        self.names = names            # file names in that folder
        self.mask = mask              # path of the mask relative to the root, or None


def scan(root: Path, excludes, sidecar_exts):
    """The groups under root in numbering order: the root's own files first, then
    each subfolder (natural order) before its own subfolders. Also returns the
    files that stay: sidecars without an image, other files, and links."""
    excl = {e.casefold() for e in excludes}
    exts = {e.casefold() for e in sidecar_exts}
    groups, orphans, others, links = [], [], [], []

    def walk(here: Path, dir_parts: tuple):
        try:
            entries = list(os.scandir(here))
        except OSError as e:
            print(f"  cannot read {here}: {e}")
            return
        subdirs, files = [], []
        masks_dir = None
        for e in entries:
            if e.is_symlink() or (e.is_dir() and os.path.isjunction(e.path)):
                links.append("/".join(dir_parts + (e.name,)))
            elif e.is_dir():
                if e.name.casefold() == MASKS_DIRNAME:
                    masks_dir = e.name
                if e.name.casefold() not in excl:
                    subdirs.append(e.name)
            elif e.is_file():
                files.append(e.name)
        masks = {}
        if masks_dir:
            try:
                masks = {os.path.splitext(f)[0].casefold(): f for f in os.listdir(here / masks_dir)
                         if f.casefold().endswith(".png")}
            except OSError:
                masks = {}
        by_stem: dict[str, list[str]] = {}
        for fn in files:
            by_stem.setdefault(os.path.splitext(fn)[0].casefold(), []).append(fn)
        for key in sorted(by_stem, key=natural_key):
            names = sorted(by_stem[key], key=natural_key)
            images = [n for n in names if os.path.splitext(n)[1].casefold() in IMAGE_EXTS]
            sides = [n for n in names if os.path.splitext(n)[1].casefold() in exts]
            rest = [n for n in names if n not in images and n not in sides]
            others.extend("/".join(dir_parts + (n,)) for n in rest)
            if images:
                mask = "/".join(dir_parts + (masks_dir, masks[key])) if key in masks else None
                groups.append(Group(dir_parts, images + sides, mask))
            else:
                orphans.extend("/".join(dir_parts + (n,)) for n in sides)
        for d in sorted(subdirs, key=natural_key):
            walk(here / d, dir_parts + (d,))

    walk(root, ())
    return groups, orphans, others, links


# --- plan ---------------------------------------------------------------------
#
# A run renames in two phases. Phase A renames every file whose new name is free
# straight to it, and every file whose new name is the old name of another file
# (a folder flattened before by hand, or an extension that only changes case) to
# a temporary name next to it. Phase B renames the temporary names to the new
# names; by then every old name is empty. Within a phase no rename targets a
# path that another rename of the same phase still has to empty, so a rename of
# the phase in progress is done exactly when its source is gone. The manifest
# records how many phases are complete; --undo relies on both facts.

def plan_root(root: Path, args) -> dict:
    groups, orphans, others, links = scan(root, DEFAULT_EXCLUDES + args.exclude, args.sidecars)
    digits = max(MIN_DIGITS, len(str(len(groups))))
    ext_len = max([len(os.path.splitext(n)[1]) for g in groups for n in g.names]
                  + [len(".png") if any(g.mask for g in groups) else 0], default=0)
    budget = args.max_length - ext_len - len(NUMBER_PREFIX) - digits
    if groups and budget < 1:
        raise SystemExit(f"--max-length {args.max_length} leaves no room for the folder name "
                         f"(the number takes {len(NUMBER_PREFIX) + digits} and the extension {ext_len})")

    root_part = clean_name(root.name) if root.name else FALLBACK_PART
    # A file that stays in the root must not share a new stem: it would become
    # a caption of an image it does not belong to.
    staying = {os.path.splitext(r)[0].casefold(): r for r in orphans + others + links if "/" not in r}
    conflicts = []
    prefixes: dict[tuple, tuple[str, bool]] = {}
    files, cut_groups = [], 0
    for number, g in enumerate(groups, 1):
        if g.dir_parts not in prefixes:
            parts = [root_part] + [clean_name(p) for p in g.dir_parts]
            prefix = name_prefix(parts, budget)
            prefixes[g.dir_parts] = (prefix, prefix != SEPARATOR.join(parts))
        prefix, was_cut = prefixes[g.dir_parts]
        cut_groups += was_cut
        stem = f"{prefix}{NUMBER_PREFIX}{number:0{digits}d}"
        if stem.casefold() in staying:
            conflicts.append(f"{staying[stem.casefold()]} stays and has the new name of an image")
        for n in g.names:
            files.append({"from": "/".join(g.dir_parts + (n,)), "to": stem + os.path.splitext(n)[1].lower()})
        if g.mask:
            files.append({"from": g.mask, "to": f"{MASKS_DIRNAME}/{stem}.png"})

    files = [f for f in files if f["from"] != f["to"]]   # already named right: nothing to do
    sources = {f["from"].casefold() for f in files}
    targets: dict[str, str] = {}
    token = secrets.token_hex(4)
    for i, f in enumerate(files):
        key = f["to"].casefold()
        if key in targets:
            conflicts.append(f"{f['from']} and {targets[key]} both become {f['to']}")
        targets[key] = f["from"]
        if key in sources:
            folder = f["from"].rpartition("/")[0]
            f["tmp"] = (folder + "/" if folder else "") + f"{TMP_PREFIX}{token}-{i}.tmp"
        elif (root / f["to"]).exists():
            conflicts.append(f"{f['to']} exists and is not renamed by this run")
    created = [MASKS_DIRNAME] if any(f["to"].startswith(MASKS_DIRNAME + "/") for f in files) \
        and not (root / MASKS_DIRNAME).is_dir() else []
    return {"tool": "flatten_dataset", "version": MANIFEST_VERSION, "root": str(root),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"), "max_length": args.max_length,
            "sidecars": args.sidecars, "state": "planned", "phases_done": 0,
            "groups": len(groups), "cut_groups": cut_groups,
            "files": files, "created_dirs": created, "removed_dirs": [],
            "_conflicts": conflicts, "_orphans": orphans, "_others": others, "_links": links}


def phases(files: list[dict]) -> list[list[tuple[str, str]]]:
    a = [(f["from"], f.get("tmp", f["to"])) for f in files]
    b = [(f["tmp"], f["to"]) for f in files if "tmp" in f]
    return [a, b]


def write_manifest(root: Path, manifest: dict):
    path = root / TOOL_DIRNAME / MANIFEST_NAME
    path.parent.mkdir(exist_ok=True)
    part = path.with_name(path.name + ".part")
    with open(part, "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in manifest.items() if not k.startswith("_")}, fh, ensure_ascii=False, indent=1)
    os.replace(part, path)


def read_manifest(root: Path) -> dict | None:
    path = root / TOOL_DIRNAME / MANIFEST_NAME
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --- run ----------------------------------------------------------------------

def print_list(title: str, items: list[str]):
    if not items:
        return
    print(f"  {title}: {len(items)}")
    for s in items[:LIST_LIMIT]:
        print(f"    {s}")
    if len(items) > LIST_LIMIT:
        print(f"    ... and {len(items) - LIST_LIMIT} more")


def print_plan(m: dict):
    print(f"  images: {m['groups']}, files to rename with their captions and masks: {len(m['files'])}")
    if m["cut_groups"]:
        print(f"  images with a folder name cut to fit {m['max_length']} characters: {m['cut_groups']}")
    print_list("stay in place: captions without an image", m["_orphans"])
    print_list("stay in place: other files", m["_others"])
    print_list("not followed: links and junctions", m["_links"])


def remove_empty_dirs(root: Path, files: list[dict]) -> list[str]:
    """Remove the subfolders that the run emptied, deepest first."""
    dirs = set()
    for f in files:
        p = Path(f["from"]).parent
        while p != Path("."):
            dirs.add(p)
            p = p.parent
    removed = []
    for d in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
        try:
            os.rmdir(root / d)
            removed.append(d.as_posix())
        except OSError:
            pass
    return removed


def run_root(root: Path, args) -> bool:
    old = read_manifest(root)
    if old is not None:
        print(f"{root}: flattened on {old.get('time')} already; run --undo first")
        return False
    m = plan_root(root, args)
    print(f"{root}:")
    print_plan(m)
    if m["_conflicts"]:
        print_list("conflicts, nothing renamed", m["_conflicts"])
        return False
    if args.dry_run:
        out = root / TOOL_DIRNAME / DRY_RUN_NAME
        out.parent.mkdir(exist_ok=True)
        with open(out, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["from", "to"])
            w.writerows((f["from"], f["to"]) for f in m["files"])
        for f in m["files"][:LIST_LIMIT]:
            print(f"    {f['from']}  ->  {f['to']}")
        print(f"  dry run: nothing renamed; the full list is in {out}")
        return True
    if not m["files"]:
        print("  nothing to rename")
        return True

    for d in m["created_dirs"]:
        (root / d).mkdir()
    write_manifest(root, m)
    done, total = 0, len(m["files"]) + sum("tmp" in f for f in m["files"])
    try:
        for i, phase in enumerate(phases(m["files"])):
            for src, dst in phase:
                os.rename(root / src, root / dst)
                done += 1
                if done % PROGRESS_EVERY == 0:
                    print(f"  {done}/{total} renamed")
            if i == 0 and len(phase) < total:   # a phase B follows
                m["phases_done"] = 1
                write_manifest(root, m)
    except KeyboardInterrupt:
        print(f"  interrupted after {done} of {total} renames; --undo puts them back")
        raise
    except OSError as e:
        print(f"  rename failed after {done} of {total}: {e}")
        print("  --undo puts back what was renamed")
        return False
    m["phases_done"] = 2
    m["removed_dirs"] = remove_empty_dirs(root, m["files"])
    m["state"] = "done"
    write_manifest(root, m)
    print(f"  files renamed: {len(m['files'])}, empty folders removed: {len(m['removed_dirs'])}")
    print(f"  manifest: {root / TOOL_DIRNAME / MANIFEST_NAME}")
    return True


# --- undo ---------------------------------------------------------------------

def undo_root(root: Path) -> bool:
    m = read_manifest(root)
    if m is None:
        print(f"{root}: nothing to undo (no {TOOL_DIRNAME}/{MANIFEST_NAME})")
        return True
    print(f"{root}: undoing the run of {m.get('time')}")
    a, b = phases(m["files"])
    restored, errors = 0, []

    def reverse(moves):
        # A rename of a phase is done exactly when its source is gone (see "plan").
        nonlocal restored
        for src, dst in reversed(moves):
            s, d = root / src, root / dst
            if s.exists() or not d.exists():
                continue
            try:
                s.parent.mkdir(parents=True, exist_ok=True)
                os.rename(d, s)
                restored += 1
            except OSError as e:
                errors.append(f"{dst} -> {src}: {e}")

    for d in m.get("removed_dirs", []):
        (root / d).mkdir(parents=True, exist_ok=True)
    if m.get("phases_done", 0) >= 1 and b:
        reverse(b)
        if errors:
            print_list("could not put back", errors)
            return False
        m["phases_done"] = 0
        write_manifest(root, m)
    reverse(a)

    missing = [f["from"] for f in m["files"] if not (root / f["from"]).exists()]
    print(f"  files put back: {restored}")
    print_list("could not put back", errors)
    if missing:
        print_list("not back at their old names", missing)
        print(f"  the manifest stays; delete {root / TOOL_DIRNAME} to give up on these files")
        return False
    for d in m.get("created_dirs", []):
        try:
            os.rmdir(root / d)
        except OSError:
            pass
    tool_dir = root / TOOL_DIRNAME
    for name in (MANIFEST_NAME, DRY_RUN_NAME):
        (tool_dir / name).unlink(missing_ok=True)
    try:
        tool_dir.rmdir()
    except OSError:
        pass
    print("  every file is back; the manifest is deleted")
    return True


# --- main ---------------------------------------------------------------------

def sidecars_arg(value: str) -> list[str]:
    exts = []
    for e in value.split(","):
        e = e.strip()
        if e:
            exts.append(e if e.startswith(".") else "." + e)
    if any(e.casefold() in IMAGE_EXTS for e in exts):
        raise argparse.ArgumentTypeError("an image extension cannot be a sidecar")
    return exts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Move the images and captions of all subfolders into the "
                                             "folder itself, under normalised names; --undo puts them back.")
    ap.add_argument("folders", nargs="+", help="dataset folders, flattened at any depth")
    ap.add_argument("--dry-run", action="store_true",
                    help=f"show the plan and write it to {TOOL_DIRNAME}\\{DRY_RUN_NAME}; rename nothing")
    ap.add_argument("--undo", action="store_true", help="put every file of the last run back")
    ap.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, metavar="N",
                    help=f"the longest new file name, extension included (default {DEFAULT_MAX_LENGTH})")
    ap.add_argument("--sidecars", type=sidecars_arg, default=sidecars_arg(DEFAULT_SIDECARS), metavar="LIST",
                    help=f"extensions of the files that go with an image, comma-separated "
                         f"(default {DEFAULT_SIDECARS})")
    ap.add_argument("--exclude", action="append", default=[], metavar="NAME",
                    help="another folder name to leave alone, at any depth; may repeat")
    args = ap.parse_args(argv)
    if args.undo and args.dry_run:
        ap.error("--undo and --dry-run do not go together")

    roots = []
    for f in args.folders:
        p = Path(f).resolve()
        if not p.is_dir():
            ap.error(f"not a folder: {f}")
        roots.append(p)
    for p in roots:
        for q in roots:
            if p != q and q.is_relative_to(p):
                ap.error(f"{q} is inside {p}; give one of them")

    ok = True
    try:
        for root in roots:
            ok &= undo_root(root) if args.undo else run_root(root, args)
    except KeyboardInterrupt:
        return 130
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
