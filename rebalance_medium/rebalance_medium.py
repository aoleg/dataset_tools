#!/usr/bin/env python3
"""
Teach a model what a prompt without a medium means: a share of the colour and
black-and-white photographs of a captioned dataset moves into a sibling
dataset folder, with a caption that no longer names the medium.

    python rebalance_medium.py D:\\train\\1_main-dataset_1024
    python rebalance_medium.py D:\\train\\1_main-dataset_1024 --apply
    python rebalance_medium.py D:\\train\\1_main-dataset_1024 --undo

Every caption opens with its medium: "A color photograph from 1975 of ...",
"A black-and-white photograph ...", with up to three words around the medium
word ("A high-angle color photograph", "A color studio photograph").

How many move is set by the "medium not stated" set the model learns from:
--drop-share percent of the colour and black-and-white captions lose the
medium (default 30), and --color-share percent of those are colour (default
60), whatever the sizes of the two classes. --max-class-pct (default 50) keeps
at least half of each class with its medium, so "color photograph" and
"black-and-white photograph" stay strong; when the shares would take more,
the drop share is lowered and the colour share kept. --color and --bw give
the percent of each class instead. The images are drawn at random (--seed)
from the whole dataset. Each moves to the same subfolder of the destination,
and its caption there loses the medium word ("A high-angle photograph"). The
captions of the images that stay are not touched.

The caches of Ostris AI Toolkit go with the images. A latent cache file is
named after the image and its crop, never its folder, so it moves with the
image and stays valid. A text-embedding cache file is named after the caption
text, so it is useless for the new caption. The original caption file and
the text-embedding cache files go to <input>_rebalance_backup/<subfolder>
next to the input folder, out of the dataset; --undo puts them back with the
image, and they also allow a restore by hand.

Without --apply nothing moves: the plan is written to the manifest
(<input>_rebalance.json next to the input folder) and summarised. An
interrupted --apply resumes from the manifest when run again.

Usage:    python rebalance_medium.py FOLDER [--drop-share PCT] [--color-share PCT]
          [--max-class-pct PCT | --color PCT --bw PCT] [--seed N] [--dest NAME]
          [--manifest PATH] [--strip-russian] [--apply]
          python rebalance_medium.py FOLDER --undo [--manifest PATH]
Install:  nothing; Python 3.10 or newer
"""
import argparse
import json
import os
import random
import re
import shutil
import stat
import sys
import time
from pathlib import Path

TOOL = "rebalance_medium"
MANIFEST_VERSION = 2                # 2: "selection" and "plan"; 1 had only color_pct and bw_pct
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
CAPTION_EXT = ".txt"
LATENT_DIRNAME = "_latent_cache"    # AI Toolkit: <stem>_<hash>.safetensors, keyed by file name and crop
TE_DIRNAME = "_t_e_cache"           # AI Toolkit: <stem>_<hash>.safetensors, keyed by caption text
BACKUP_DIRNAME = "_backup"          # the other dataset tools' old backup: never scanned, never moved
BACKUP_MARK = "_dataset_tools.json"  # marks <dataset>_backup, the backup folder of the other tools: never scanned
SIZE_DB_NAME = ".aitk_size.json"    # AI Toolkit's image size cache in a dataset folder
SKIP_DIRS = {d.casefold() for d in (LATENT_DIRNAME, TE_DIRNAME, BACKUP_DIRNAME)}
DEST_SUFFIX = "media-rebalanced"
BACKUP_SUFFIX = "_rebalance_backup"  # <input>_rebalance_backup: original captions and text-embedding caches
MANIFEST_SUFFIX = "_rebalance.json"
SAVE_EVERY = 100                    # images between manifest writes
LIST_LIMIT = 20                     # lines of any list printed
RMDIR_TRIES, RMDIR_WAIT = 20, 0.1   # an emptied folder Windows still holds: tries, seconds between
CLASSES = ("color", "bw", "ambiguous", "no_caption")
SETTINGS = ("input", "dest", "seed", "strip_russian")      # and the selection, compared apart
DROP_SHARE, COLOR_SHARE, MAX_CLASS_PCT = 30, 60, 50        # the default selection: shares
COLOR_PCT, BW_PCT = 40, 20          # per class, when only one of --color and --bw is given


# --- captions -----------------------------------------------------------------

# The medium at the start of the caption: the article, up to three words, the
# medium word, up to three words, then "photograph" (or "photo"). A word holds
# no white space or sentence punctuation, so the phrase never leaves its
# sentence.
_WORD = r"[^\s.,;:!?()\"]+"
_BW = r"black[\s-]+(?:and|&)[\s-]+white"
MEDIUM_RX = re.compile(
    rf"(?P<lead>[\s\ufeff]*)(?P<art>an?)(?P<pre>(?:\s+{_WORD}){{0,3}}?)\s+"
    rf"(?P<medium>colou?r|{_BW})"
    rf"(?P<post>(?:\s+{_WORD}){{0,3}}?)\s+photo(?:graph)?(?![\w-])",
    re.IGNORECASE)
# Words around the medium word that make it unsure: the other medium, or a
# tinted or toned print ("A hand-colored black-and-white photograph").
UNSURE_RX = re.compile(rf"colou?r|tint|sepia|monochrom|cyanotype|duotone|{_BW}", re.IGNORECASE)

RU_SEP = "., "                      # the English description ends here when a Russian tail follows
RU_MEDIUM_RX = re.compile(
    r"(?<!\w)(?:цветн|ч[её]рно-бел)(?:ая|ое|ой|ые|ый|ую|ого|ому|ом|ых|ым|ыми|ою|ей)(?!\w)",
    re.IGNORECASE)


def classify(caption: str):
    """("color" | "bw" | "ambiguous", the medium match or None)."""
    m = MEDIUM_RX.match(caption)
    if not m or UNSURE_RX.search(f"{m['pre']} {m['post']}"):
        return "ambiguous", None
    return ("color" if m["medium"][0] in "cC" else "bw"), m


def drop_medium(caption: str, m: re.Match) -> str:
    """The caption without the medium word and the space before it; "a" becomes
    "an" when the next word starts with a vowel. A leading byte order mark
    goes too."""
    new = caption[:m.end("pre")] + caption[m.end("medium"):]
    if not m["pre"]:                # the article stood right before the medium word
        nxt = new[m.end("art"):].lstrip()
        want = "an" if nxt[:1] and nxt[0].casefold() in "aeiou" else "a"
        art = m["art"]
        if art.casefold() != want:
            fixed = ("A" if art[0].isupper() else "a") + ("n" if want == "an" else "")
            new = new[:m.start("art")] + fixed + new[m.end("art"):]
    return new.lstrip("\ufeff")


def russian_tail(caption: str) -> str:
    i = caption.find(RU_SEP)
    return caption[i + len(RU_SEP):] if i >= 0 else ""


def strip_russian(caption: str) -> str:
    """The caption without the Russian medium words of its tail, each with the
    space after it (or before it, at the end of a phrase). The next word takes
    the capital letter of a removed word that opened a sentence."""
    i = caption.find(RU_SEP)
    if i < 0:
        return caption
    head, tail = caption[:i + len(RU_SEP)], caption[i + len(RU_SEP):]
    for m in reversed(list(RU_MEDIUM_RX.finditer(tail))):
        s, e = m.span()
        after = tail[e:]
        if after[:1] == " ":
            e += len(after) - len(after.lstrip(" "))
        elif s > 0 and tail[s - 1] == " ":
            s -= 1
        rest = tail[e:]
        if m.group()[0].isupper() and rest[:1].islower():
            rest = rest[0].upper() + rest[1:]
        tail = tail[:s] + rest
    return head + tail


def new_caption(caption: str, m: re.Match, russian: bool) -> str:
    new = drop_medium(caption, m)
    return strip_russian(new) if russian else new


# --- scan ---------------------------------------------------------------------

def is_link(entry: os.DirEntry) -> bool:
    """A symbolic link or a junction; never followed."""
    if entry.is_symlink():
        return True
    try:
        st = entry.stat(follow_symlinks=False)
    except OSError:
        return False
    return getattr(st, "st_reparse_tag", 0) == getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", -1)


def cache_rx(stem: str) -> re.Pattern:
    """The cache files of one image. A glob on stem_* would also take the files of
    another image whose name starts with stem_."""
    return re.compile(rf"^{re.escape(stem)}_[A-Za-z0-9_-]{{22}}\.safetensors$")


def list_files(folder: Path) -> list[str]:
    try:
        return sorted(e.name for e in os.scandir(folder) if e.is_file(follow_symlinks=False) and not is_link(e))
    except OSError:
        return []


def read_text(path: Path) -> str:
    """The caption as written: UTF-8, a byte order mark and line ends kept."""
    return path.read_bytes().decode("utf-8")


def write_text(path: Path, text: str):
    """UTF-8 with no byte order mark added, line ends as given; a half-written
    caption never replaces a whole one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_bytes(text.encode("utf-8"))
    os.replace(part, path)


def scan(root: Path):
    """The images of every folder that has at least one image with a caption,
    and the links that were not followed. A folder without such a pair is
    skipped, its subfolders are not."""
    items, links = [], []

    def walk(here: Path, parts: tuple):
        try:
            entries = sorted(os.scandir(here), key=lambda e: e.name)
        except OSError as e:
            print(f"  cannot read {here}: {e}")
            return
        subdirs, files = [], []
        for e in entries:
            if is_link(e):
                links.append("/".join(parts + (e.name,)))
            elif e.is_dir(follow_symlinks=False):
                if (not e.name.startswith(".") and e.name.casefold() not in SKIP_DIRS
                        and not os.path.isfile(os.path.join(e.path, BACKUP_MARK))):
                    subdirs.append(e.name)
            elif e.is_file(follow_symlinks=False):
                files.append(e.name)
        captions = {}
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.casefold() == CAPTION_EXT:
                captions.setdefault(stem.casefold(), f)
        images = [f for f in files if os.path.splitext(f)[1].casefold() in IMAGE_EXTS]
        if any(os.path.splitext(f)[0].casefold() in captions for f in images):
            latents, tes = list_files(here / LATENT_DIRNAME), list_files(here / TE_DIRNAME)
            stems: dict[str, int] = {}
            for f in images:
                k = os.path.splitext(f)[0].casefold()
                stems[k] = stems.get(k, 0) + 1
            for f in images:
                stem = os.path.splitext(f)[0]
                cap = captions.get(stem.casefold())
                rx = cache_rx(stem)
                it = {"rel": "/".join(parts + (f,)), "class": "no_caption", "selected": False,
                      "status": "planned", "caption_file": cap, "caption_old": None, "caption_new": None,
                      "latent_cache": [n for n in latents if rx.match(n)],
                      "te_cache": [n for n in tes if rx.match(n)], "russian_medium": False}
                if stems[stem.casefold()] > 1:
                    it["shared_name"] = True        # two images, one caption and one cache name: never moved
                if cap is not None:
                    try:
                        text = read_text(here / cap)
                    except (OSError, UnicodeDecodeError) as e:
                        it["class"], it["unreadable"] = "ambiguous", str(e)
                    else:
                        it["caption_old"] = text
                        it["class"] = classify(text)[0]
                        it["russian_medium"] = bool(RU_MEDIUM_RX.search(russian_tail(text)))
                items.append(it)
        for d in subdirs:
            walk(here / d, parts + (d,))

    walk(root, ())
    return items, links


# --- plan ---------------------------------------------------------------------

def eligible(it: dict) -> bool:
    return it["class"] in ("color", "bw") and not it.get("shared_name")


def round_half_up(x: float) -> int:
    return int(x + 0.5 + 1e-9)


def class_counts(items: list[dict]) -> dict:
    """The images that can move, per class."""
    return {c: sum(1 for it in items if it["class"] == c and eligible(it)) for c in ("color", "bw")}


def plan_counts(n: dict, sel: dict) -> dict:
    """How many images of each class move. Per class: the rounded percent of
    each. Shares: drop_share percent of all colour and black-and-white images,
    color_share percent of them colour; the drop share is lowered until no class
    loses more than max_class_pct percent of its images, and the colour share
    stays."""
    if sel["mode"] == "per_class":
        return {"color": round_half_up(n["color"] * sel["color_pct"] / 100),
                "bw": round_half_up(n["bw"] * sel["bw_pct"] / 100), "drop_share": None, "lowered": False}
    cs, ds, cap = sel["color_share"] / 100, sel["drop_share"] / 100, sel["max_class_pct"] / 100
    total = n["color"] + n["bw"]
    if cs > 0 and ds > 0 and not n["color"]:
        raise Refused("no colour image can move; give --color-share 0, or --color and --bw")
    if cs < 1 and ds > 0 and not n["bw"]:
        raise Refused("no black-and-white image can move; give --color-share 100, or --color and --bw")
    if not total:
        return {"color": 0, "bw": 0, "drop_share": 0, "lowered": False}
    limits = [ds]
    if cs > 0:
        limits.append(cap * n["color"] / (cs * total))
    if cs < 1:
        limits.append(cap * n["bw"] / ((1 - cs) * total))
    used = min(limits)
    moved = round_half_up(used * total)
    k_color = round_half_up(cs * moved)
    k_bw = moved - k_color
    # rounding can pass the cap by one image; never more than the cap allows
    k_color = min(k_color, int(cap * n["color"] + 1e-9))
    k_bw = min(k_bw, int(cap * n["bw"] + 1e-9))
    return {"color": k_color, "bw": k_bw, "drop_share": round(100 * used, 2), "lowered": used < ds - 1e-12}


def select(items: list[dict], counts: dict, seed: int, russian: bool):
    """Draw the counted images of each class from the whole dataset: from a
    sorted list shuffled with the seed, so a run is the same on every
    platform."""
    rng = random.Random(seed)
    by_rel = {it["rel"]: it for it in items}
    for cls in ("color", "bw"):
        pool = sorted(it["rel"] for it in items if it["class"] == cls and eligible(it))
        rng.shuffle(pool)
        for rel in pool[:counts[cls]]:
            it = by_rel[rel]
            it["selected"] = True
            it["caption_new"] = new_caption(it["caption_old"], classify(it["caption_old"])[1], russian)


def summarize(items: list[dict]) -> dict:
    cls = {c: sum(1 for it in items if it["class"] == c) for c in CLASSES}
    sel = {c: sum(1 for it in items if it["selected"] and it["class"] == c) for c in ("color", "bw")}
    chosen = [it for it in items if it["selected"]]
    return {
        "images": len(items), "classes": cls,
        "russian_medium": sum(1 for it in items if it["russian_medium"]),
        "selected": sel,
        "stated_after": cls["color"] + cls["bw"] - sel["color"] - sel["bw"],
        "dropped": sel["color"] + sel["bw"],
        "latent_cache_files": sum(len(it["latent_cache"]) for it in chosen),
        "te_cache_files": sum(len(it["te_cache"]) for it in chosen),
        "folders_touched": len({str(Path(it["rel"]).parent) for it in chosen}),
        "shared_name": sum(1 for it in items if it.get("shared_name")),
        "unreadable": sum(1 for it in items if it.get("unreadable")),
    }


def pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f} %" if total else "-"


def print_list(title: str, lines: list[str]):
    if not lines:
        return
    print(f"  {title}: {len(lines)}")
    for s in lines[:LIST_LIMIT]:
        print(f"    {s}")
    if len(lines) > LIST_LIMIT:
        print(f"    ... and {len(lines) - LIST_LIMIT} more")


def print_summary(m: dict):
    s, items = m["summary"], m["items"]
    c, sel = s["classes"], s["selected"]
    print(f"  images found: {s['images']}: color {c['color']}, bw {c['bw']}, "
          f"ambiguous {c['ambiguous']}, no_caption {c['no_caption']}")
    print(f"  Russian tail names the medium: {s['russian_medium']}"
          f"{' (stripped from moved captions)' if m['strip_russian'] else ''}")
    sl, plan = manifest_selection(m), m.get("plan", {})
    if sl["mode"] == "shares":
        line = (f"  selection: shares: drop {sl['drop_share']} % of the captions, {sl['color_share']} % of them "
                f"colour, at most {sl['max_class_pct']} % of a class")
        if plan.get("lowered"):
            line += f"; the cap lowered the drop share to {plan['drop_share']} %"
        print(line)
    else:
        print(f"  selection: per class: color {sl['color_pct']} %, bw {sl['bw_pct']} %")
    print(f"  moved: color {sel['color']} of {c['color']} ({pct(sel['color'], c['color'])}), "
          f"bw {sel['bw']} of {c['bw']} ({pct(sel['bw'], c['bw'])})")
    total = s["stated_after"] + s["dropped"]
    stated_color = c["color"] - sel["color"]
    print(f"  medium dropped, in the destination: {s['dropped']} ({pct(s['dropped'], total)} of the captions), "
          f"{pct(sel['color'], s['dropped'])} colour"
          + (f"; also without a known medium: {c['ambiguous']} ambiguous captions" if c["ambiguous"] else ""))
    print(f"  medium stated, in the input folder after the move: {s['stated_after']} "
          f"({pct(s['stated_after'], total)}): color {stated_color}, bw {c['bw'] - sel['bw']}, "
          f"{pct(stated_color, s['stated_after'])} colour")
    print(f"  latent cache files to move: {s['latent_cache_files']}; to back up: {s['dropped']} original captions "
          f"and {s['te_cache_files']} text-embedding cache files")
    print(f"  folders touched: {s['folders_touched']}")
    if s["shared_name"]:
        print(f"  not moved, another image has the same name: {s['shared_name']}")
    print_list("not read as UTF-8, not moved", [it["rel"] for it in items if it.get("unreadable")])
    print_list("ambiguous captions (no medium, both, or tinted), not moved",
               [f"{it['rel']}: {one_line(it['caption_old'])}" for it in items
                if it["class"] == "ambiguous" and not it.get("unreadable")])


def one_line(text: str, width: int = 90) -> str:
    t = " ".join(text.split())
    return t if len(t) <= width else t[:width - 3] + "..."


# --- manifest -----------------------------------------------------------------

def write_manifest(path: Path, m: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with open(part, "w", encoding="utf-8") as fh:
        json.dump(m, fh, ensure_ascii=False, indent=1)
    os.replace(part, path)


def read_manifest(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def manifest_selection(m: dict) -> dict:
    """The selection options of a manifest; a version 1 manifest had per-class
    percents only."""
    return m.get("selection") or {"mode": "per_class", "color_pct": m["color_pct"], "bw_pct": m["bw_pct"]}


def same_settings(m: dict, settings: dict) -> bool:
    return all(m.get(k) == settings[k] for k in SETTINGS) and manifest_selection(m) == settings["selection"]


# --- moves --------------------------------------------------------------------

class Refused(Exception):
    """A reason to stop before anything changes."""


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


def paths(it: dict, src: Path, dst: Path, bk: Path) -> dict:
    rel = Path(it["rel"])
    sdir, ddir = src / rel.parent, dst / rel.parent
    return {"sdir": sdir, "ddir": ddir, "simg": sdir / rel.name, "dimg": ddir / rel.name,
            "scap": sdir / it["caption_file"], "dcap": ddir / (rel.stem + CAPTION_EXT),
            "bcap": bk / rel.parent / it["caption_file"], "bte": bk / rel.parent / TE_DIRNAME}


def text_or_none(path: Path) -> str | None:
    try:
        return read_text(path)
    except FileNotFoundError:
        return None


def apply_item(it: dict, src: Path, dst: Path, bk: Path, counts: dict):
    """Move one image; every step may have been done before an interruption."""
    p = paths(it, src, dst, bk)
    if p["simg"].exists() and p["dimg"].exists():
        raise OSError(f"the image is in both folders: {p['dimg']}")
    if not p["simg"].exists() and not p["dimg"].exists():
        raise OSError("the image is missing")
    scap = text_or_none(p["scap"])
    if scap is not None and scap != it["caption_old"]:
        raise OSError(f"the caption changed after the plan: {p['scap']}")
    dcap = text_or_none(p["dcap"])
    if dcap is not None and dcap != it["caption_new"]:
        raise OSError(f"another caption is in the destination: {p['dcap']}")
    bcap = text_or_none(p["bcap"])
    if bcap is not None and bcap != it["caption_old"]:
        raise OSError(f"another caption is in the backup folder: {p['bcap']}")

    if it["latent_cache"]:
        (p["ddir"] / LATENT_DIRNAME).mkdir(parents=True, exist_ok=True)
    if p["simg"].exists():
        move(p["simg"], p["dimg"])
    write_text(p["dcap"], it["caption_new"])
    if scap is not None:
        if bcap is None:
            move(p["scap"], p["bcap"])      # the original caption file itself, kept for a restore by hand
        else:
            p["scap"].unlink()              # its backup is there already, the same text
    elif bcap is None:
        write_text(p["bcap"], it["caption_old"])
    for name in it["latent_cache"]:
        s, d = p["sdir"] / LATENT_DIRNAME / name, p["ddir"] / LATENT_DIRNAME / name
        if s.exists():
            move(s, d)
            counts["latent"] += 1
    for name in it["te_cache"]:
        s, b = p["sdir"] / TE_DIRNAME / name, p["bte"] / name
        if s.exists():
            if b.exists():
                s.unlink()          # the same name is the same caption: the backup is this file
            else:
                move(s, b)
            counts["te"] += 1
    it["status"] = "moved"


def undo_item(it: dict, src: Path, dst: Path, bk: Path, counts: dict, notes: list[str]):
    """Put one image back with its original caption (the backed-up file, or the
    manifest's text when that file is gone), its latent caches and its
    text-embedding caches; delete the text-embedding caches made for the new
    caption."""
    p = paths(it, src, dst, bk)
    if p["simg"].exists() and p["dimg"].exists():
        raise OSError(f"the image is in both folders: {p['simg']}")
    if not p["simg"].exists() and not p["dimg"].exists():
        raise OSError("the image is missing from both folders")
    scap = text_or_none(p["scap"])
    if scap is not None and scap != it["caption_old"]:
        raise OSError(f"another caption is in the input folder: {p['scap']}")

    if p["dimg"].exists():
        move(p["dimg"], p["simg"])
    bcap = text_or_none(p["bcap"])
    if scap is None and bcap is not None:
        move(p["bcap"], p["scap"])
        if bcap != it["caption_old"]:
            notes.append(f"{it['rel']}: the backed-up caption was edited; the edited one is back in the input folder")
    elif scap is None:
        write_text(p["scap"], it["caption_old"])
    elif bcap is not None:
        if bcap == it["caption_old"]:
            p["bcap"].unlink()              # the same text is in place already
        else:
            notes.append(f"{it['rel']}: the backed-up caption was edited; it stays in the backup folder")
    dcap = text_or_none(p["dcap"])
    if dcap is not None:
        if dcap == it["caption_new"]:
            p["dcap"].unlink()
        else:
            notes.append(f"{it['rel']}: the caption in the destination was edited; it stays there")
    rx = cache_rx(Path(it["rel"]).stem)
    for name in list_files(p["ddir"] / LATENT_DIRNAME):
        if rx.match(name):
            s, d = p["ddir"] / LATENT_DIRNAME / name, p["sdir"] / LATENT_DIRNAME / name
            if d.exists():
                s.unlink()          # the same name is the same image and crop
            else:
                move(s, d)
            counts["latent"] += 1
    for name in list_files(p["bte"]):
        if rx.match(name):
            b, d = p["bte"] / name, p["sdir"] / TE_DIRNAME / name
            if d.exists():
                b.unlink()
            else:
                move(b, d)
            counts["te"] += 1
    for name in list_files(p["ddir"] / TE_DIRNAME):
        if rx.match(name):
            (p["ddir"] / TE_DIRNAME / name).unlink()
            counts["te_deleted"] += 1
    it["status"] = "restored"


def remove_dir(d: Path) -> bool:
    """Remove a folder that holds no files. Windows can hold a folder whose files
    were just deleted for a moment (a virus scanner or the search indexer still
    has a handle), so a folder without files is tried again for up to 2 s."""
    for _ in range(RMDIR_TRIES):
        try:
            os.rmdir(d)
            return True
        except FileNotFoundError:
            return True
        except OSError:
            if any(p.is_file() for p in d.rglob("*")):
                return False
            time.sleep(RMDIR_WAIT)
    return False


def clear_folder(root: Path) -> list[str]:
    """Delete root when no file is left in it but AI Toolkit's size cache, which
    goes too; else return the files left."""
    if not root.is_dir():
        return []
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    if files == [SIZE_DB_NAME]:         # the trainer's size cache of the destination, nothing else
        (root / SIZE_DB_NAME).unlink()
        files = []
    if files:
        return files
    for d in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        remove_dir(d)
    remove_dir(root)
    return []


def remove_empty_dirs(dirs: set[Path], stop: Path):
    """Remove each folder that is empty, deepest first, and then its parents up
    to stop (kept)."""
    chain = set()
    for d in dirs:
        while d != stop and stop in d.parents:
            chain.add(d)
            d = d.parent
    for d in sorted(chain, key=lambda p: len(p.parts), reverse=True):
        remove_dir(d)


# --- runs ---------------------------------------------------------------------

def plan_run(src: Path, settings: dict) -> dict:
    items, links = scan(src)
    n = class_counts(items)
    plan = plan_counts(n, settings["selection"])
    select(items, plan, settings["seed"], settings["strip_russian"])
    summary = summarize(items)
    c = summary["classes"]          # the percent of each whole class, as the summary prints it
    m = {"tool": TOOL, "version": MANIFEST_VERSION, **settings,
         "color_pct": round(100 * plan["color"] / c["color"], 2) if c["color"] else 0,
         "bw_pct": round(100 * plan["bw"] / c["bw"], 2) if c["bw"] else 0, "plan": plan,
         "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "summary": summary, "items": items}
    print_list("links and junctions, not followed", links)
    return m


def backfill_captions(m: dict, src: Path, dst: Path, bk: Path) -> int:
    """Write the original caption of every moved image whose backup file is
    missing, from the manifest: a run made before the captions were backed up
    gets them on its next --apply."""
    n = 0
    for it in m["items"]:
        if it["status"] == "moved":
            p = paths(it, src, dst, bk)
            if not p["bcap"].exists() and not p["scap"].exists():
                write_text(p["bcap"], it["caption_old"])
                n += 1
    return n


def check_stale(m: dict, src: Path):
    stale = [it["rel"] for it in m["items"] if it["status"] == "moved" and (src / it["rel"]).exists()]
    if stale:
        raise Refused(f"the manifest is stale: {len(stale)} images it lists as moved are in the input folder "
                      f"(first: {stale[0]})")


def run_apply(src: Path, dst: Path, mpath: Path, settings: dict, chosen: bool) -> bool:
    """chosen: the selection was given on the command line; when it was not, a
    resumed run takes the manifest's."""
    bk = Path(settings["backup"])
    m = read_manifest(mpath)
    resume = m is not None and any(
        it["status"] == "moved" or (it["selected"] and it["status"] == "planned" and dst.exists())
        for it in m["items"])
    if resume:
        if not chosen:
            settings["selection"] = manifest_selection(m)
        if not same_settings(m, settings):
            raise Refused(f"{mpath} records an applied run with other settings; run --undo first")
        check_stale(m, src)
        print(f"  resuming the run of {m['created']} from {mpath}")
    else:
        if dst.exists():
            raise Refused(f"the destination {dst} exists; remove it or give another --dest")
        if bk.exists():
            raise Refused(f"the backup folder {bk} exists; it may hold the caches of an earlier run")
        m = plan_run(src, settings)
    todo = [it for it in m["items"] if it["selected"] and it["status"] == "planned"]
    write_manifest(mpath, m)
    backfilled = backfill_captions(m, src, dst, bk)
    if backfilled:
        print(f"  original captions written to the backup folder from the manifest: {backfilled}")
    counts = {"latent": 0, "te": 0}
    errors, done = [], 0
    try:
        for it in todo:
            try:
                apply_item(it, src, dst, bk, counts)
                done += 1
            except OSError as e:
                errors.append(f"{it['rel']}: {e}")
            if (done + len(errors)) % SAVE_EVERY == 0:
                write_manifest(mpath, m)
                print(f"  {done + len(errors)}/{len(todo)}")
    finally:
        m["applied"] = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "images": done,
                        "latent_cache_moved": counts["latent"], "te_cache_backed_up": counts["te"]}
        write_manifest(mpath, m)
    print_summary(m)
    print(f"  moved now: {done} images, {counts['latent']} latent cache files; "
          f"original captions and {counts['te']} text-embedding cache files backed up to {bk}")
    moved = sum(1 for it in m["items"] if it["status"] == "moved")
    print(f"  moved in all: {moved} images into {dst}")
    if errors:
        print_list("could not move", errors)
        print(f"{len(errors)} images were not moved; fix the cause and run --apply again")
        return False
    return True


def run_undo(src: Path, mpath: Path) -> bool:
    m = read_manifest(mpath)
    if m is None:
        raise Refused(f"no manifest at {mpath}")
    if Path(m["input"]) != src:
        raise Refused(f"{mpath} is the manifest of {m['input']}, not of {src}")
    dst, bk = Path(m["dest"]), Path(m["backup"])
    todo = [it for it in m["items"] if it["status"] == "moved"
            or (it["selected"] and it["status"] == "planned" and (dst / it["rel"]).exists())]
    print(f"  undoing the run of {m['created']}: {len(todo)} images back from {dst}")
    counts = {"latent": 0, "te": 0, "te_deleted": 0}
    errors, notes, done = [], [], 0
    try:
        for it in todo:
            try:
                undo_item(it, src, dst, bk, counts, notes)
                done += 1
            except OSError as e:
                errors.append(f"{it['rel']}: {e}")
            if (done + len(errors)) % SAVE_EVERY == 0:
                write_manifest(mpath, m)
                print(f"  {done + len(errors)}/{len(todo)}")
    finally:
        m["undone"] = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "images": done,
                       "latent_cache_moved": counts["latent"], "te_cache_restored": counts["te"],
                       "te_cache_deleted": counts["te_deleted"]}
        write_manifest(mpath, m)
    dirs = set()
    for it in todo:
        rel = Path(it["rel"]).parent
        dirs |= {dst / rel / LATENT_DIRNAME, dst / rel / TE_DIRNAME, bk / rel / TE_DIRNAME}
    remove_empty_dirs({d for d in dirs if d.is_relative_to(dst)}, dst)
    remove_empty_dirs({d for d in dirs if d.is_relative_to(bk)}, bk)
    left = clear_folder(dst)
    left_bk = clear_folder(bk)
    print_summary(m)
    print(f"  moved back: {done} images with their original captions, {counts['latent']} latent cache files, "
          f"{counts['te']} text-embedding cache files from {bk.name}; "
          f"{counts['te_deleted']} text-embedding cache files of the new captions deleted")
    print_list("notes", notes)
    for folder, files in ((dst, left), (bk, left_bk)):
        if not folder.exists():
            print(f"  {folder} is empty and deleted")
        elif files:
            print_list(f"{folder} stays; files left in it", files)
        else:
            print(f"  {folder} is empty, but Windows still holds it; delete it later")
    if errors:
        print_list("could not move back", errors)
        print(f"{len(errors)} images were not moved back; fix the cause and run --undo again")
        return False
    return True


def run_dry(src: Path, dst: Path, mpath: Path, settings: dict) -> bool:
    old = read_manifest(mpath)
    if old is not None and any(it["status"] == "moved" for it in old["items"]):
        raise Refused(f"{mpath} records moved images; run --undo first, or give another --manifest")
    for d, what in ((dst, "destination"), (Path(settings["backup"]), "backup folder")):
        if d.exists():
            print(f"  note: the {what} {d} exists; --apply will refuse until it is gone")
    m = plan_run(src, settings)
    write_manifest(mpath, m)
    print_summary(m)
    print(f"  plan written to {mpath}")
    print("Dry run: nothing moved. Re-run with --apply.")
    return True


# --- main ---------------------------------------------------------------------

def default_dest(src: Path) -> Path:
    """<name>_media-rebalanced_<bucket> when the name ends in _<digits>, else
    <name>_media-rebalanced, next to the input folder."""
    m = re.fullmatch(r"(.+)_(\d+)", src.name)
    return src.with_name(f"{m[1]}_{DEST_SUFFIX}_{m[2]}" if m else f"{src.name}_{DEST_SUFFIX}")


def backup_dir(src: Path) -> Path:
    """<input>_rebalance_backup next to the input folder: out of the dataset,
    so the trainer never walks it."""
    return src.with_name(src.name + BACKUP_SUFFIX)


def percent(s: str) -> float:
    v = float(s)
    if not 0 <= v <= 100:
        raise argparse.ArgumentTypeError(f"{s} is not between 0 and 100")
    return int(v) if v.is_integer() else v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Move a share of the colour and black-and-white photographs of a dataset, with captions "
                    "that no longer name the medium, into a sibling dataset folder. A dry run without --apply.")
    ap.add_argument("folder", help="the dataset folder, with all its subfolders")
    ap.add_argument("--drop-share", type=percent, metavar="PCT",
                    help=f"percent of the colour and black-and-white captions that lose the medium "
                         f"(default {DROP_SHARE})")
    ap.add_argument("--color-share", type=percent, metavar="PCT",
                    help=f"percent of colour images among those (default {COLOR_SHARE})")
    ap.add_argument("--max-class-pct", type=percent, metavar="PCT",
                    help=f"at most this percent of either class moves; the drop share is lowered to keep it "
                         f"(default {MAX_CLASS_PCT})")
    ap.add_argument("--color", type=percent, metavar="PCT",
                    help=f"instead of the shares: percent of the colour images to move (default {COLOR_PCT} "
                         f"when only --bw is given)")
    ap.add_argument("--bw", type=percent, metavar="PCT",
                    help=f"instead of the shares: percent of the black-and-white images to move (default "
                         f"{BW_PCT} when only --color is given)")
    ap.add_argument("--dest", metavar="NAME", help="the destination folder, next to the input folder "
                                                   "(default <name>_media-rebalanced_<bucket>)")
    ap.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    ap.add_argument("--manifest", metavar="PATH", help="the manifest (default <folder>_rebalance.json next to it)")
    ap.add_argument("--strip-russian", action="store_true",
                    help="also remove the Russian medium words from the tail of the moved captions")
    ap.add_argument("--apply", action="store_true", help="move the images; without it nothing changes")
    ap.add_argument("--undo", action="store_true", help="move everything of an applied run back")
    args = ap.parse_args(argv)
    if args.apply and args.undo:
        ap.error("--apply and --undo do not go together")
    per_class = args.color is not None or args.bw is not None
    shares = any(v is not None for v in (args.drop_share, args.color_share, args.max_class_pct))
    if per_class and shares:
        ap.error("--color and --bw do not go together with --drop-share, --color-share and --max-class-pct")
    if per_class:
        selection = {"mode": "per_class", "color_pct": COLOR_PCT if args.color is None else args.color,
                     "bw_pct": BW_PCT if args.bw is None else args.bw}
    else:
        selection = {"mode": "shares",
                     "drop_share": DROP_SHARE if args.drop_share is None else args.drop_share,
                     "color_share": COLOR_SHARE if args.color_share is None else args.color_share,
                     "max_class_pct": MAX_CLASS_PCT if args.max_class_pct is None else args.max_class_pct}

    src = Path(args.folder).resolve()
    if not src.is_dir():
        print(f"not a folder: {args.folder}")
        return 1
    mpath = Path(args.manifest).resolve() if args.manifest else src.with_name(src.name + MANIFEST_SUFFIX)
    dst = (src.parent / args.dest).resolve() if args.dest else default_dest(src)
    bk = backup_dir(src)
    if dst == src or dst.is_relative_to(src) or src.is_relative_to(dst):
        print(f"the destination {dst} must be outside the input folder")
        return 1
    if dst == bk or dst.is_relative_to(bk) or bk.is_relative_to(dst):
        print(f"the destination {dst} must be outside the backup folder {bk}")
        return 1
    settings = {"input": str(src), "dest": str(dst), "backup": str(bk), "seed": args.seed,
                "strip_russian": args.strip_russian, "selection": selection}
    print(f"input:    {src}")
    try:
        if args.undo:
            ok = run_undo(src, mpath)
        else:
            print(f"dest:     {dst}")
            print(f"backup:   {bk}")
            print(f"seed:     {args.seed}{', Russian medium words stripped' if args.strip_russian else ''}")
            if args.apply:
                ok = run_apply(src, dst, mpath, settings, chosen=per_class or shares)
            else:
                ok = run_dry(src, dst, mpath, settings)
    except Refused as e:
        print(f"refused: {e}")
        return 1
    except KeyboardInterrupt:
        print("interrupted; the manifest has every finished image; run the same command again to go on")
        return 130
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
