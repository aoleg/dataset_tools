#!/usr/bin/env python3
"""
Move the images whose captions contain a keyword, with all their files, out of
a dataset into <dataset>_<keyword> next to it, in the same subfolders.

    python extract_media.py D:\\train\\1_main-dataset_1024 "black-and-white"
    python extract_media.py D:\\train\\1_main-dataset_1024 "black-and-white" --apply
    python extract_media.py D:\\train\\1_main-dataset_1024 "black-and-white" --undo

Keywords as in extract_keywords: keywords given one after another are
alternatives, keywords joined by AND must all be in the caption, and AND binds
closer than the space between keywords. A keyword matches whole words in any
letter case; the words of a keyword may be separated by any white space or
hyphens in the caption, so "black-and-white" also finds "Black and white".
--partial matches inside words too.

With an image go all the files of its name in its folder (other images, the
caption, other sidecars), its face masks in _controls (or an old masks
folder), and its files in _latent_cache and _t_e_cache. Nothing is rewritten:
the caption stays the same, so both AI Toolkit caches stay valid and the new
folder trains at once as a dataset of its own.

Without --apply nothing moves: the plan is written to the manifest
(<dataset>_<keyword>_extract.json next to the dataset) and summarised. An
interrupted --apply resumes from the manifest when run again; --undo moves
every file back.

Usage:    python extract_media.py FOLDER KEYWORD [AND|OR KEYWORD ...] [--partial]
          [--case-sensitive] [--dest NAME] [--manifest PATH] [--apply | --undo]
Install:  nothing; Python 3.10 or newer
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

import rebalance_medium as rm

TOOL = "extract_media"
MANIFEST_VERSION = 1
MANIFEST_SUFFIX = "_extract.json"   # <dataset>_<keyword>_extract.json next to the dataset
CAPTION_EXT = ".txt"
IMAGE_EXTS = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff",
              ".gif", ".avif", ".heic", ".heif"}
CACHE_DIRNAMES = (rm.LATENT_DIRNAME, rm.TE_DIRNAME)
MASKS_DIRNAME = "_controls"         # face_masks: <dir>/_controls/<stem>.<ext>
OLD_MASKS_DIRNAME = "masks"         # the same until 2026-10-11
MASK_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
# Folders never searched: the caches and masks (searched with their images),
# the old in-dataset output folders of the dataset tools, and any folder whose
# name starts with "." or that the dataset tools marked as theirs.
SKIP_DIRS = {d.casefold() for d in (*CACHE_DIRNAMES, MASKS_DIRNAME, OLD_MASKS_DIRNAME, rm.BACKUP_DIRNAME,
                                    "_duplicates", "_prep", "_classify", "_embeddings", "_flatten_dataset",
                                    "faces")}
TOOLS_MARK = "_dataset_tools.json"
SETTINGS = ("input", "dest", "keywords", "partial", "case_sensitive")
AND, OR = "AND", "OR"
BAD_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f\s]+')


# --- keywords (as in extract_keywords) ----------------------------------------

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
    """A keyword whose words may be separated by any white space or hyphens:
    "black-and-white" also finds "black and white"; a whole-word match unless
    partial."""
    words = [w for w in re.split(r"[\s-]+", term) if w]
    body = r"[\s-]+".join(re.escape(w) for w in words) if words else re.escape(term)
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

def scan(root: Path):
    """One entry per name in a folder that has a caption and an image: its
    files (images, caption, sidecars), masks and cache files, as paths relative
    to root. Also the captions without an image and the links not followed."""
    groups, orphans, links = [], [], []

    def walk(here: Path, parts: tuple):
        try:
            entries = sorted(os.scandir(here), key=lambda e: e.name)
        except OSError as e:
            print(f"  cannot read {here}: {e}")
            return
        subdirs, files, special = [], [], {}
        for e in entries:
            if rm.is_link(e):
                links.append("/".join(parts + (e.name,)))
            elif e.is_dir(follow_symlinks=False):
                key = e.name.casefold()
                if key in (*(d.casefold() for d in CACHE_DIRNAMES), MASKS_DIRNAME.casefold(), OLD_MASKS_DIRNAME):
                    special[key] = e.name
                if (not e.name.startswith(".") and key not in SKIP_DIRS
                        and not os.path.isfile(os.path.join(e.path, TOOLS_MARK))):
                    subdirs.append(e.name)
            elif e.is_file(follow_symlinks=False):
                files.append(e.name)
        by_stem: dict[str, list[str]] = {}
        for f in files:
            by_stem.setdefault(os.path.splitext(f)[0].casefold(), []).append(f)
        listed = {k: rm.list_files(here / name) for k, name in special.items()}
        rel = lambda *p: "/".join(parts + p)
        for key in sorted(by_stem):
            names = by_stem[key]
            captions = [n for n in names if os.path.splitext(n)[1].casefold() == CAPTION_EXT]
            if not captions:
                continue
            if not any(os.path.splitext(n)[1].casefold() in IMAGE_EXTS for n in names):
                orphans.extend(rel(c) for c in captions)
                continue
            stems = sorted({os.path.splitext(n)[0] for n in names})
            masks, caches = [], []
            for k in (OLD_MASKS_DIRNAME, MASKS_DIRNAME.casefold()):
                if k in special:
                    masks += [rel(special[k], f) for f in listed[k] if os.path.splitext(f)[0].casefold() == key
                              and os.path.splitext(f)[1].casefold() in MASK_EXTS]
            for d in CACHE_DIRNAMES:
                k = d.casefold()
                if k in special:
                    rxs = [rm.cache_rx(s) for s in stems]
                    caches += [rel(special[k], f) for f in listed[k] if any(rx.match(f) for rx in rxs)]
            groups.append({"dir": "/".join(parts), "stems": stems, "caption_file": rel(captions[0]),
                           "files": [rel(n) for n in sorted(names)], "masks": masks, "caches": caches})
        for d in subdirs:
            walk(here / d, parts + (d,))

    walk(root, ())
    return groups, orphans, links


def plan_run(src: Path, settings: dict) -> dict:
    regexes = {t: keyword_regex(t, settings["partial"], settings["case_sensitive"])
               for g in settings["keywords"] for t in g}
    hits = {t: 0 for t in regexes}
    groups, orphans, links = scan(src)
    items, unreadable = [], []
    for g in groups:
        try:
            text = rm.read_text(src / g["caption_file"])
        except (OSError, UnicodeDecodeError) as e:
            unreadable.append(f"{g['caption_file']}: {e}")
            continue
        found = {t for t, rx in regexes.items() if rx.search(text)}
        for t in found:
            hits[t] += 1
        if any(all(t in found for t in grp) for grp in settings["keywords"]):
            items.append({**g, "caption": text, "status": "planned"})
    summary = {"captions": len(groups), "hits": hits, "images": len(items),
               "files": sum(len(it["files"]) for it in items),
               "masks": sum(len(it["masks"]) for it in items),
               "latent_cache_files": sum(1 for it in items for c in it["caches"] if f"/{rm.LATENT_DIRNAME}/" in f"/{c}"),
               "te_cache_files": sum(1 for it in items for c in it["caches"] if f"/{rm.TE_DIRNAME}/" in f"/{c}"),
               "folders_touched": len({it["dir"] for it in items})}
    rm.print_list("captions without an image, not searched", orphans)
    rm.print_list("captions not read as UTF-8, not searched", unreadable)
    rm.print_list("links and junctions, not followed", links)
    return {"tool": TOOL, "version": MANIFEST_VERSION, **settings,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "summary": summary, "items": items}


def print_summary(m: dict):
    s = m["summary"]
    print(f"  keywords: {describe(m['keywords'])}{' (inside words too)' if m['partial'] else ''}"
          f"{' (letter case counts)' if m['case_sensitive'] else ''}")
    print(f"  captions searched: {s['captions']}")
    if len(s["hits"]) > 1:
        for t, n in s["hits"].items():
            print(f"    with \"{t}\": {n}")
    print(f"  images to extract: {s['images']}, in {s['folders_touched']} folders: {s['files']} image, caption "
          f"and sidecar files, {s['masks']} masks, {s['latent_cache_files']} latent cache files, "
          f"{s['te_cache_files']} text-embedding cache files")
    rm.print_list("captions", [f"{it['caption_file']}: {rm.one_line(it['caption'])}" for it in m["items"]])


# --- moves --------------------------------------------------------------------

def all_files(it: dict) -> list[str]:
    return it["files"] + it["masks"] + it["caches"]


def apply_item(it: dict, src: Path, dst: Path, notes: list[str]):
    """Move every file of one image; any of them may have moved before an
    interruption. Nothing moves when a file is in both folders."""
    files = all_files(it)
    both = [f for f in files if (src / f).exists() and (dst / f).exists()]
    if both:
        raise OSError(f"in both folders: {both[0]}")
    for f in it["files"]:
        if not (src / f).exists() and not (dst / f).exists():
            raise OSError(f"missing: {f}")
    for f in files:
        if (src / f).exists():
            rm.move(src / f, dst / f)
        elif not (dst / f).exists():
            notes.append(f"{f}: gone before the move")     # a mask or cache file deleted meanwhile
    it["status"] = "moved"


def undo_item(it: dict, src: Path, dst: Path, notes: list[str]):
    """Move every file of one image back, also the cache files the trainer made
    for it in the new folder."""
    files = list(all_files(it))
    for d in CACHE_DIRNAMES:
        folder = dst / it["dir"] / d
        rxs = [rm.cache_rx(s) for s in it["stems"]]
        for name in rm.list_files(folder):
            rel = "/".join(p for p in (it["dir"], d, name) if p)
            if rel not in files and any(rx.match(name) for rx in rxs):
                files.append(rel)
    taken = [f for f in files if (dst / f).exists() and (src / f).exists() and f not in it["caches"]]
    if taken:
        raise OSError(f"the old place is taken: {taken[0]}")
    for f in files:
        s, d = dst / f, src / f
        if not s.exists():
            if not d.exists() and f in it["files"]:
                notes.append(f"{f}: missing from the new folder")
            continue
        if d.exists():
            s.unlink()          # a cache file of the same name in both: the same key, the same content
        else:
            rm.move(s, d)
    it["status"] = "restored"


# --- runs ---------------------------------------------------------------------

def check_stale(m: dict, src: Path):
    stale = [it["files"][0] for it in m["items"] if it["status"] == "moved"
             and any((src / f).exists() for f in it["files"])]
    if stale:
        raise rm.Refused(f"the manifest is stale: {len(stale)} images it lists as moved are in the dataset "
                         f"(first: {stale[0]})")


def run_items(m: dict, mpath: Path, todo: list[dict], step, label: str) -> tuple[int, list[str], list[str]]:
    done, errors, notes = 0, [], []
    try:
        for it in todo:
            try:
                step(it, notes)
                done += 1
            except OSError as e:
                errors.append(f"{it['caption_file']}: {e}")
            if (done + len(errors)) % rm.SAVE_EVERY == 0:
                rm.write_manifest(mpath, m)
                print(f"  {label} {done + len(errors)}/{len(todo)}")
    finally:
        rm.write_manifest(mpath, m)
    return done, errors, notes


def run_apply(src: Path, dst: Path, mpath: Path, settings: dict) -> bool:
    m = rm.read_manifest(mpath)
    resume = m is not None and any(
        it["status"] == "moved" or (it["status"] == "planned" and dst.exists()) for it in m["items"])
    if resume:
        if not all(m.get(k) == settings[k] for k in SETTINGS):
            raise rm.Refused(f"{mpath} records an applied run with other settings; run --undo first")
        check_stale(m, src)
        print(f"  resuming the run of {m['created']} from {mpath}")
    else:
        if dst.exists():
            raise rm.Refused(f"the destination {dst} exists; remove it or give another --dest")
        m = plan_run(src, settings)
        if not m["items"]:
            rm.write_manifest(mpath, m)
            print_summary(m)
            print("  nothing to extract; no folder created")
            return True
    todo = [it for it in m["items"] if it["status"] == "planned"]
    rm.write_manifest(mpath, m)
    done, errors, notes = run_items(m, mpath, todo, lambda it, n: apply_item(it, src, dst, n), "moved")
    m["applied"] = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "images": done}
    rm.write_manifest(mpath, m)
    print_summary(m)
    print(f"  moved now: {done} images; in all: {sum(1 for it in m['items'] if it['status'] == 'moved')} "
          f"images in {dst}")
    rm.print_list("notes", notes)
    if errors:
        rm.print_list("could not move", errors)
        print(f"{len(errors)} images were not moved; fix the cause and run --apply again")
        return False
    return True


def run_undo(src: Path, mpath: Path) -> bool:
    m = rm.read_manifest(mpath)
    if m is None:
        raise rm.Refused(f"no manifest at {mpath}")
    if Path(m["input"]) != src:
        raise rm.Refused(f"{mpath} is the manifest of {m['input']}, not of {src}")
    dst = Path(m["dest"])
    todo = [it for it in m["items"] if it["status"] == "moved"
            or (it["status"] == "planned" and any((dst / f).exists() for f in all_files(it)))]
    print(f"  undoing the run of {m['created']}: {len(todo)} images back from {dst}")
    done, errors, notes = run_items(m, mpath, todo, lambda it, n: undo_item(it, src, dst, n), "moved back")
    m["undone"] = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "images": done}
    rm.write_manifest(mpath, m)
    dirs = {dst / it["dir"] / d for it in todo for d in (*CACHE_DIRNAMES, MASKS_DIRNAME, OLD_MASKS_DIRNAME)}
    rm.remove_empty_dirs(dirs | {dst / it["dir"] for it in todo}, dst)
    left = rm.clear_folder(dst)
    print_summary(m)
    print(f"  moved back: {done} images with all their files")
    rm.print_list("notes", notes)
    if not dst.exists():
        print(f"  {dst} is empty and deleted")
    elif left:
        rm.print_list(f"{dst} stays; files left in it", left)
    else:
        print(f"  {dst} is empty, but Windows still holds it; delete it later")
    if errors:
        rm.print_list("could not move back", errors)
        print(f"{len(errors)} images were not moved back; fix the cause and run --undo again")
        return False
    return True


def run_dry(src: Path, dst: Path, mpath: Path, settings: dict) -> bool:
    old = rm.read_manifest(mpath)
    if old is not None and any(it["status"] == "moved" for it in old["items"]):
        raise rm.Refused(f"{mpath} records moved images; run --undo first, or give another --manifest")
    if dst.exists():
        print(f"  note: the destination {dst} exists; --apply will refuse until it is gone")
    m = plan_run(src, settings)
    rm.write_manifest(mpath, m)
    print_summary(m)
    print(f"  plan written to {mpath}")
    print("Dry run: nothing moved. Re-run with --apply.")
    return True


# --- main ---------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Move the images whose captions contain the keywords, with all their files, masks and AI "
                    "Toolkit cache files, out of a dataset into <dataset>_<first keyword>. A dry run without "
                    "--apply; --undo moves them back.",
        epilog='Keywords one after another: any of them. Joined by AND: all of them. '
               'Example: D:\\train\\set "black-and-white" AND 1940s')
    ap.add_argument("folder", help="the dataset folder, with all its subfolders")
    ap.add_argument("keywords", nargs="+", help="keywords; put a phrase in quotes; AND joins keywords that "
                                                "must all be in the caption")
    ap.add_argument("--partial", action="store_true", help="match inside words too (\"car\" finds \"cart\")")
    ap.add_argument("--case-sensitive", action="store_true", help="letter case must match")
    ap.add_argument("--dest", metavar="NAME", help="the new folder, next to the dataset folder "
                                                   "(default <folder>_<first keyword>)")
    ap.add_argument("--manifest", metavar="PATH",
                    help="the manifest (default <folder>_<first keyword>_extract.json next to the dataset)")
    ap.add_argument("--apply", action="store_true", help="move the files; without it nothing changes")
    ap.add_argument("--undo", action="store_true", help="move every file of an applied run back")
    args = ap.parse_args(argv)
    if args.apply and args.undo:
        ap.error("--apply and --undo do not go together")
    try:
        groups = parse_query(args.keywords)
    except ValueError as e:
        ap.error(str(e))

    src = Path(args.folder).resolve()
    if not src.is_dir():
        print(f"not a folder: {args.folder}")
        return 1
    dst = (src.parent / args.dest).resolve() if args.dest else \
        src.with_name(f"{src.name}_{folder_name_part(groups[0][0])}")
    mpath = Path(args.manifest).resolve() if args.manifest else dst.with_name(dst.name + MANIFEST_SUFFIX)
    if dst == src or dst.is_relative_to(src) or src.is_relative_to(dst):
        print(f"the destination {dst} must be outside the dataset folder")
        return 1
    settings = {"input": str(src), "dest": str(dst), "keywords": groups, "partial": args.partial,
                "case_sensitive": args.case_sensitive}
    print(f"input:    {src}")
    try:
        if args.undo:
            ok = run_undo(src, mpath)
        else:
            print(f"dest:     {dst}")
            ok = run_apply(src, dst, mpath, settings) if args.apply else run_dry(src, dst, mpath, settings)
    except rm.Refused as e:
        print(f"refused: {e}")
        return 1
    except KeyboardInterrupt:
        print("interrupted; the manifest has every finished image; run the same command again to go on")
        return 130
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
