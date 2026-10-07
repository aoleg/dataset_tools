#!/usr/bin/env python3
"""
Extract part of a dataset by caption keywords: every image whose caption (.txt)
contains the keywords moves, with its caption, out of the dataset into a new
folder next to it, in the same subfolders.

    python extract_keywords.py 1940s "street scene" --dataset D:\\photos
    python extract_keywords.py soldier AND 1940s --dataset D:\\photos

Keywords given one after another are alternatives (any of them is enough);
keywords joined by AND must all be in the caption. AND binds closer than the
space between keywords: "a AND b c" takes captions with both a and b, or with c.
A keyword matches whole words, in any letter case, and the words of a phrase may
be separated by any white space, so "street scene" also finds "Street\\nscene".
--partial matches inside words too ("car" finds "cart"), which suits watermark
text such as a site name.

The new folder is <dataset>_<first keyword> next to the dataset; a folder that
exists already is an error. Every move is written to
<new folder>/_extract_keywords/manifest.json before the first one, and --undo
reads it and moves every file back.

Usage:    python extract_keywords.py KEYWORD [AND|OR KEYWORD ...] --dataset FOLDER
          [--out FOLDER] [--partial] [--case-sensitive] [--dry-run]
          python extract_keywords.py KEYWORD ... --dataset FOLDER --undo
          python extract_keywords.py --undo EXTRACTED_FOLDER
Install:  nothing; Python 3.10 or newer
"""
import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

TOOL_DIRNAME = "_extract_keywords"  # in the new folder: manifest.json
MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1
CAPTION_EXT = ".txt"
MASKS_DIRNAME = "masks"             # face_masks: <dir>/masks/<stem>.png moves with its image
LIST_LIMIT = 20                     # lines of any list printed
PROGRESS_EVERY = 10000              # moves between progress lines

IMAGE_EXTS = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff",
              ".gif", ".avif", ".heic", ".heif"}
# Folders never searched, at any depth: this tool's own folder and the output
# folders of the other dataset tools.
EXCLUDES = {TOOL_DIRNAME, "_flatten_dataset", "_backup", "_duplicates", "_prep", "_classify",
            "_embeddings", MASKS_DIRNAME, "faces"}
AND, OR = "AND", "OR"
BAD_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f\s]+')


# --- keywords -----------------------------------------------------------------

def parse_query(words: list[str]) -> list[list[str]]:
    """The keywords as alternatives of AND groups: "a AND b c" -> [[a, b], [c]].
    Only AND and OR in capitals are operators; OR is the same as a space."""
    groups, group, expect_term = [], [], True
    for w in words:
        if w in (AND, OR):
            if expect_term:
                raise ValueError(f"{w} needs a keyword before it")
            if w == OR:
                groups.append(group)
                group = []
            expect_term = True
            continue
        term = " ".join(w.split())
        if not term:
            raise ValueError("an empty keyword")
        if not expect_term:         # two keywords without AND: alternatives
            groups.append(group)
            group = []
        group.append(term)
        expect_term = False
    if expect_term:
        raise ValueError("the keywords end with AND or OR" if words else "no keywords")
    groups.append(group)
    return groups


def keyword_regex(term: str, partial: bool, case_sensitive: bool) -> re.Pattern:
    """A phrase whose words may be separated by any white space; a whole-word
    match unless partial."""
    body = r"\s+".join(re.escape(w) for w in term.split())
    if not partial:
        body = rf"(?<!\w){body}(?!\w)"
    return re.compile(body, 0 if case_sensitive else re.IGNORECASE)


def describe(groups: list[list[str]]) -> str:
    return " OR ".join(" AND ".join(f'"{t}"' for t in g) for g in groups)


def folder_name_part(term: str) -> str:
    """A keyword as a part of a folder name: no characters Windows forbids, "_"
    for spaces."""
    return BAD_NAME_CHARS.sub("_", term).strip("_. ") or "keyword"


# --- scan ---------------------------------------------------------------------

def read_caption(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError as e:
        print(f"  cannot read {path}: {e}")
        return None
    return data.decode("utf-8-sig", errors="replace")


def scan(root: Path):
    """Yield (folder parts, caption name, [files of the image], mask or None) for
    every caption that has an image, and collect the captions without one.
    The files of an image are every file with the stem of the caption in its
    folder: the images (a.jpg, a.png), the caption and any other sidecar."""
    orphans, links = [], []

    def walk(here: Path, parts: tuple):
        try:
            entries = list(os.scandir(here))
        except OSError as e:
            print(f"  cannot read {here}: {e}")
            return
        subdirs, files, masks_dir = [], [], None
        for e in entries:
            if e.is_symlink() or (e.is_dir() and os.path.isjunction(e.path)):
                links.append("/".join(parts + (e.name,)))
            elif e.is_dir():
                if e.name.casefold() == MASKS_DIRNAME:
                    masks_dir = e.name
                if e.name.casefold() not in {x.casefold() for x in EXCLUDES}:
                    subdirs.append(e.name)
            elif e.is_file():
                files.append(e.name)
        by_stem: dict[str, list[str]] = {}
        for fn in files:
            by_stem.setdefault(os.path.splitext(fn)[0].casefold(), []).append(fn)
        masks = {}
        if masks_dir:
            try:
                masks = {os.path.splitext(f)[0].casefold(): f for f in os.listdir(here / masks_dir)
                         if f.casefold().endswith(".png")}
            except OSError:
                masks = {}
        for key in sorted(by_stem):
            names = sorted(by_stem[key])
            captions = [n for n in names if os.path.splitext(n)[1].casefold() == CAPTION_EXT]
            if not captions:
                continue
            if not any(os.path.splitext(n)[1].casefold() in IMAGE_EXTS for n in names):
                orphans.extend("/".join(parts + (c,)) for c in captions)
                continue
            mask = "/".join(parts + (masks_dir, masks[key])) if key in masks else None
            yield parts, captions[0], names, mask
        for d in sorted(subdirs):
            yield from walk(here / d, parts + (d,))

    return walk(root, ()), orphans, links


def find_matches(root: Path, groups, partial: bool, case_sensitive: bool):
    regexes = {t: keyword_regex(t, partial, case_sensitive) for g in groups for t in g}
    hits = {t: 0 for t in regexes}
    matches, captions = [], 0
    items, orphans, links = scan(root)
    for parts, caption, names, mask in items:
        captions += 1
        text = read_caption(root.joinpath(*parts, caption))
        if text is None:
            continue
        found = {t for t, rx in regexes.items() if rx.search(text)}
        for t in found:
            hits[t] += 1
        if any(all(t in found for t in g) for g in groups):
            files = ["/".join(parts + (n,)) for n in names]
            if mask:
                files.append(mask)
            matches.append(files)
    return matches, captions, hits, orphans, links


# --- manifest -----------------------------------------------------------------

def manifest_path(out: Path) -> Path:
    return out / TOOL_DIRNAME / MANIFEST_NAME


def write_manifest(out: Path, m: dict):
    path = manifest_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with open(part, "w", encoding="utf-8") as fh:
        json.dump(m, fh, ensure_ascii=False, indent=1)
    os.replace(part, path)


def read_manifest(out: Path) -> dict | None:
    path = manifest_path(out)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --- moves --------------------------------------------------------------------

def move(src: Path, dst: Path):
    """Move a file and never overwrite one. A rename on the same drive, a copy
    and delete across drives."""
    if dst.exists():
        raise FileExistsError(f"{dst} exists")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(src, dst)
    except OSError as e:
        if getattr(e, "winerror", None) != 17 and e.errno != 18:   # not "another drive"
            raise
        shutil.move(src, dst)


def remove_emptied_dirs(root: Path, rel_files: list[str]) -> list[str]:
    """Remove the subfolders of root that the moves emptied, deepest first."""
    dirs = set()
    for f in rel_files:
        p = Path(f).parent
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


def print_list(title: str, items: list[str]):
    if not items:
        return
    print(f"  {title}: {len(items)}")
    for s in items[:LIST_LIMIT]:
        print(f"    {s}")
    if len(items) > LIST_LIMIT:
        print(f"    ... and {len(items) - LIST_LIMIT} more")


# --- run ----------------------------------------------------------------------

def run(dataset: Path, out: Path, groups, args) -> bool:
    print(f"dataset:  {dataset}")
    print(f"keywords: {describe(groups)}"
          f"{' (inside words too)' if args.partial else ''}{' (letter case counts)' if args.case_sensitive else ''}")
    if out.exists() and not args.dry_run:
        print(f"{out} exists already; run --undo first, or give another folder with --out")
        return False
    matches, captions, hits, orphans, links = find_matches(dataset, groups, args.partial, args.case_sensitive)
    print(f"  captions searched: {captions}")
    if len(hits) > 1:
        for t, n in hits.items():
            print(f"    with \"{t}\": {n}")
    print(f"  images to extract: {len(matches)} ({sum(map(len, matches))} files with captions and masks)")
    print_list("captions without an image, not searched", orphans)
    print_list("not followed: links and junctions", links)
    if args.dry_run:
        print_list("would move", [m[0] for m in matches])
        print(f"  dry run: nothing moved; the run would create {out}")
        return True
    if not matches:
        print("  nothing to extract; no folder created")
        return True

    files = [f for m in matches for f in m]
    m = {"tool": "extract_keywords", "version": MANIFEST_VERSION, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
         "dataset": str(dataset), "out": str(out), "keywords": groups, "partial": args.partial,
         "case_sensitive": args.case_sensitive, "state": "planned", "files": files, "removed_dirs": []}
    out.mkdir(parents=True)
    write_manifest(out, m)
    done = 0
    try:
        for f in files:
            move(dataset / f, out / f)
            done += 1
            if done % PROGRESS_EVERY == 0:
                print(f"  {done}/{len(files)} moved")
    except KeyboardInterrupt:
        print(f"  interrupted after {done} of {len(files)} moves; --undo puts them back")
        raise
    except OSError as e:
        print(f"  move failed after {done} of {len(files)}: {e}")
        print("  --undo puts back what was moved")
        return False
    m["removed_dirs"] = remove_emptied_dirs(dataset, files)
    m["state"] = "done"
    write_manifest(out, m)
    print(f"  files moved: {len(files)} into {out}")
    if m["removed_dirs"]:
        print(f"  empty subfolders removed from the dataset: {len(m['removed_dirs'])}")
    return True


# --- undo ---------------------------------------------------------------------

def undo(out: Path) -> bool:
    m = read_manifest(out)
    if m is None:
        print(f"{out}: nothing to undo (no {TOOL_DIRNAME}/{MANIFEST_NAME})")
        return True
    dataset = Path(m["dataset"])
    print(f"{out}: moving the extraction of {m.get('time')} back to {dataset}")
    if not dataset.is_dir():
        print(f"  the dataset folder {dataset} is missing; nothing moved")
        return False
    restored, errors, gone = 0, [], []
    for d in m.get("removed_dirs", []):
        (dataset / d).mkdir(parents=True, exist_ok=True)
    for f in m["files"]:
        src, dst = out / f, dataset / f
        if not src.exists():
            if not dst.exists():
                gone.append(f)          # deleted from the extraction
            continue                    # else never moved (an interrupted run)
        try:
            move(src, dst)
            restored += 1
        except OSError as e:
            errors.append(f"{f}: {e}")
    print(f"  files moved back: {restored}")
    print_list("missing from the extracted folder (deleted there?)", gone)
    if errors:
        print_list("could not move back", errors)
        print("  the manifest stays; run --undo again when these are fixed")
        return False
    manifest_path(out).unlink()
    try:
        (out / TOOL_DIRNAME).rmdir()
    except OSError:
        pass
    remove_emptied_dirs(out, m["files"])
    try:
        out.rmdir()
        print(f"  {out} is empty and deleted")
    except OSError:
        left = sum(1 for p in out.rglob("*") if p.is_file())
        print(f"  {out} stays: {left} files in it were not extracted by this run")
    return True


# --- main ---------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Move the images whose captions contain the keywords, with their captions, out of a "
                    "dataset into <dataset>_<first keyword>; --undo moves them back.",
        epilog='Keywords one after another: any of them. Joined by AND: all of them. '
               'Example: 1940s "street scene" AND night --dataset D:\\photos')
    ap.add_argument("keywords", nargs="*", help="keywords; put a phrase in quotes; AND joins keywords that "
                                                "must all be in the caption")
    ap.add_argument("--dataset", metavar="FOLDER", help="the dataset folder, searched with all its subfolders")
    ap.add_argument("--out", metavar="FOLDER", help="the new folder (default: <dataset>_<first keyword> "
                                                    "next to the dataset)")
    ap.add_argument("--partial", action="store_true", help="match inside words too (\"car\" finds \"cart\")")
    ap.add_argument("--case-sensitive", action="store_true", help="letter case must match")
    ap.add_argument("--dry-run", action="store_true", help="show what would move; move nothing")
    ap.add_argument("--undo", nargs="?", const=True, metavar="FOLDER",
                    help="move the files of an extraction back: give the same keywords and --dataset "
                         "(or --out) as the run, or the extracted folder itself")
    args = ap.parse_args(argv)

    if args.undo is not None:
        if args.dry_run:
            ap.error("--undo and --dry-run do not go together")
        if isinstance(args.undo, str):          # --undo FOLDER
            if args.keywords or args.dataset or args.out:
                ap.error("--undo FOLDER takes no keywords, --dataset or --out")
            out = Path(args.undo).resolve()
        elif args.out:
            out = Path(args.out).resolve()
        elif not args.dataset and len(args.keywords) == 1 and Path(args.keywords[0]).is_dir():
            out = Path(args.keywords[0]).resolve()   # run.bat <folder> --undo
        else:
            if not args.dataset or not args.keywords:
                ap.error("--undo needs the keywords and --dataset of the run, or the extracted folder")
            groups = parse_or_exit(ap, args.keywords)
            dataset = Path(args.dataset).resolve()
            out = dataset.with_name(f"{dataset.name}_{folder_name_part(groups[0][0])}")
        try:
            return 0 if undo(out) else 1
        except KeyboardInterrupt:
            print("  interrupted; run --undo again to finish")
            return 130

    if not args.dataset:
        if len(args.keywords) == 1 and manifest_path(Path(args.keywords[0])).is_file():
            ap.error(f"{args.keywords[0]} is an extracted folder; add --undo to move it back")
        ap.error("--dataset FOLDER is required")
    groups = parse_or_exit(ap, args.keywords)
    dataset = Path(args.dataset).resolve()
    if not dataset.is_dir():
        ap.error(f"not a folder: {args.dataset}")
    if args.out:
        out = Path(args.out).resolve()
    else:
        out = dataset.with_name(f"{dataset.name}_{folder_name_part(groups[0][0])}")
    if out == dataset or out.is_relative_to(dataset) or dataset.is_relative_to(out):
        ap.error(f"the new folder {out} must be outside the dataset")
    try:
        return 0 if run(dataset, out, groups, args) else 1
    except KeyboardInterrupt:
        return 130


def parse_or_exit(ap, words) -> list[list[str]]:
    try:
        return parse_query(words)
    except ValueError as e:
        ap.error(str(e))


if __name__ == "__main__":
    sys.exit(main())
