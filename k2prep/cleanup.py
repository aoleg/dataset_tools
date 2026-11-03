#!/usr/bin/env python3
"""cleanup.py - move undersized images out of a training folder.

Walks a folder and all its subfolders, at any depth, and moves every image
smaller than a given size, together with its sidecars (every other file that
goes by its name: captions, masks, caches), into a sidecar folder named after
the source with a leading underscore. The original folder structure is
preserved inside it:

    cleanup.py T:\\somefolder 1024

    T:\\somefolder\\small.jpg     ->  T:\\_somefolder\\small.jpg
    T:\\somefolder\\1\\small.jpg   ->  T:\\_somefolder\\1\\small.jpg
    T:\\somefolder\\1\\a\\small.jpg ->  T:\\_somefolder\\1\\a\\small.jpg

Nothing is deleted and nothing is overwritten. The images stay on disk, one
folder over, so a threshold chosen too aggressively is undone by moving them
back. Run with --dry-run first.

Stand-alone: shares no code with k2prep.py, only Pillow.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from PIL import Image

__version__ = "1.2.0"

# Mirrors k2prep's IMAGE_EXTENSIONS, for the same reason: these are the
# extensions musubi-tuner recognises. Mixed-case variants such as .Jpg are not
# matched; they are counted and listed so you notice them.
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"]
KNOWN_EXTENSIONS = set(IMAGE_EXTENSIONS) | {e.upper() for e in IMAGE_EXTENSIONS}

# What a caption is: files with these extensions are not "unknown" when no
# image goes by their name. What moves with an image is every file of its name.
CAPTION_EXTENSIONS = (".txt", ".short", ".caption", ".json")

# A leading underscore marks a folder as somebody's output rather than source
# material: k2prep's _prep, and this script's own destination. Skipping the
# whole class means cleanup.py is safe to re-run on a folder it has already
# processed.
SIDECAR_PREFIX = "_"


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def _size_arg(text: str) -> tuple[int, int]:
    """N, meaning N x N, or WxH. Returns (width, height)."""
    parts = text.lower().split("x")
    try:
        if len(parts) == 1:
            n = int(parts[0])
            dims = (n, n)
        elif len(parts) == 2:
            dims = (int(parts[0]), int(parts[1]))
        else:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"size must be N or WxH, e.g. 1024 or 1600x900, got {text!r}")
    if dims[0] < 1 or dims[1] < 1:
        raise argparse.ArgumentTypeError(f"size must be positive, got {text!r}")
    return dims


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="cleanup",
        description="Move images below a size threshold, and their sidecars "
                    "(every other file of the image's name), out of a folder "
                    "and all its subfolders into a _foldername sidecar folder "
                    "next to it.",
    )
    p.add_argument("folder", help="Folder to clean. Subfolders at any depth "
                                  "are cleaned too, except those starting with "
                                  "an underscore or a dot, and links.")
    p.add_argument("size", type=_size_arg, metavar="SIZE",
                   help="Threshold, N (meaning NxN) or WxH. By default an "
                        "image is moved when it has fewer pixels than this, "
                        "measured as area: 1024 means 1048576 pixels, so a "
                        "2048x400 panorama goes and a 1200x900 frame stays.")
    p.add_argument("--dim", action="store_true",
                   help="Compare dimensions instead of area: move an image if "
                        "either side is shorter than the threshold's. Under "
                        "--dim 1024 that same 2048x400 panorama is moved for "
                        "its short side and 1200x900 is moved too.")
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="List what would move and touch nothing.")
    p.add_argument("--version", action="version", version=f"cleanup {__version__}")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def _is_link(entry: os.DirEntry) -> bool:
    """A symlink or a Windows junction: anything that points elsewhere.

    is_dir(follow_symlinks=False) is not enough - Windows reports a junction as
    a real directory there - so ask for the reparse tag, which is non-zero for
    every kind of reparse point. Anything that cannot be classified is treated
    as a link, since the safe direction is not following it. Copied from
    k2prep.py.
    """
    try:
        if entry.is_symlink():
            return True
        st = entry.stat(follow_symlinks=False)
    except OSError:
        return True
    return getattr(st, "st_reparse_tag", 0) != 0


def folders_to_clean(root: Path) -> tuple[list[Path], list[str], list[str]]:
    """The root and every folder below it, at any depth, each parent before
    its subfolders and siblings in name order. Returns (folders, links not
    followed, folders that could not be read).

    Skipped at every depth, as k2prep's scan skips them: folders starting with
    an underscore (sidecars, _prep) or a dot, and links and junctions, which
    are not followed. An unreadable folder is listed and the walk goes on.
    """
    folders: list[Path] = []
    links: list[str] = []
    unreadable: list[str] = []
    stack = [root]
    while stack:
        folder = stack.pop()
        folders.append(folder)
        subs = []
        try:
            entries = list(os.scandir(folder))
        except OSError as exc:
            unreadable.append(f"{folder}: {exc.strerror or exc}")
            continue
        for entry in entries:
            try:
                if not entry.is_dir():
                    continue
            except OSError:
                continue
            if entry.name.startswith((SIDECAR_PREFIX, ".")):
                continue
            if _is_link(entry):
                links.append(entry.path)
                continue
            subs.append(Path(entry.path))
        subs.sort(key=lambda p: p.name.lower(), reverse=True)
        stack.extend(subs)
    return folders, links, unreadable


def stem_key(name: str) -> str:
    """The name without its extension, folded the way the file system folds it:
    x.jpg and X.PNG are one stem on Windows, two on Linux."""
    return os.path.normcase(os.path.splitext(name)[0])


def images_in(folder: Path) -> tuple[list[Path], dict[str, list[Path]], list[str]]:
    """Non-recursive. Returns (image paths by filename, sidecars by stem_key,
    unknown-extension filenames).

    The sidecars of an image are the other files of the folder that go by its
    name, whatever their extension, taken from this one listing. A mixed-case
    image extension (.Jpg) is neither an image nor a sidecar: it is unknown.
    A caption with no image is not unknown: it is caption data."""
    images: list[Path] = []
    others: list[os.DirEntry] = []
    for entry in os.scandir(folder):
        if entry.is_dir():
            continue
        if os.path.splitext(entry.name)[1] in KNOWN_EXTENSIONS:
            images.append(Path(entry.path))
        else:
            others.append(entry)
    images.sort(key=lambda p: p.name)
    stems = {stem_key(p.name) for p in images}
    sidecars: dict[str, list[Path]] = {}
    unknown: list[str] = []
    for entry in sorted(others, key=lambda e: e.name):
        ext = os.path.splitext(entry.name)[1].lower()
        key = stem_key(entry.name)
        if ext in IMAGE_EXTENSIONS:
            unknown.append(entry.name)
        elif key in stems:
            sidecars.setdefault(key, []).append(Path(entry.path))
        elif ext not in CAPTION_EXTENSIONS:
            unknown.append(entry.name)
    return images, sidecars, unknown


def plural(n: int, word: str, plural_form: str | None = None) -> str:
    """"1 image", "2 images". Counts land in almost every line this script
    prints, and "1 images" reads like a bug in the counting."""
    if n == 1:
        return f"{n} {word}"
    return f"{n} {plural_form or word + 's'}"


def is_undersized(size: tuple[int, int], threshold: tuple[int, int],
                  by_dim: bool) -> bool:
    if by_dim:
        return size[0] < threshold[0] or size[1] < threshold[1]
    return size[0] * size[1] < threshold[0] * threshold[1]


# ---------------------------------------------------------------------------
# Moving
# ---------------------------------------------------------------------------

class Counts:
    def __init__(self):
        self.scanned = 0
        self.moved = 0
        self.kept = 0
        self.skipped = 0
        self.errors = 0
        self.unknown = 0


def clean_folder(folder: Path, dest: Path, threshold: tuple[int, int],
                 by_dim: bool, dry_run: bool, counts: Counts) -> None:
    images, sidecars, unknown = images_in(folder)
    counts.unknown += len(unknown)
    gone: set[Path] = set()           # sidecars of a shared name already moved

    moved = kept = skipped = errors = 0
    for path in images:
        counts.scanned += 1
        # Image.open reads the header and stops; the pixels are never decoded,
        # which is what makes this usable on a folder of 40 MP originals.
        try:
            with Image.open(path) as im:
                size = im.size
        except Exception as exc:
            print(f"    error  {path.name}: {type(exc).__name__}: {exc}")
            errors += 1
            continue

        if not is_undersized(size, threshold, by_dim):
            kept += 1
            continue

        # x.jpg and x.png share x.txt and every other file of the name x: the
        # first of them to move takes it along.
        caps = [s for s in sidecars.get(stem_key(path.name), []) if s not in gone]
        pairs = [(path, dest / path.name)] + [(cap, dest / cap.name) for cap in caps]

        # Never overwrite. Renaming on collision would be friendlier right up
        # until the image and its caption disagree about their new stem, so a
        # colliding pair is left where it is and reported instead.
        clash = [dst for _, dst in pairs if dst.exists()]
        if clash:
            print(f"    skip   {path.name}: {clash[0].name} already exists in "
                  f"{dest}")
            skipped += 1
            continue

        gone.update(caps)
        if dry_run:
            cap_note = "".join(f" (+ {cap.suffix})" for cap in caps)
            print(f"    move   {path.name}  {size[0]}x{size[1]}{cap_note}")
            moved += 1
            continue

        dest.mkdir(parents=True, exist_ok=True)
        try:
            for src, dst in pairs:
                shutil.move(str(src), str(dst))
        except Exception as exc:
            print(f"    error  {path.name}: {type(exc).__name__}: {exc}")
            errors += 1
            continue
        moved += 1

    counts.moved += moved
    counts.kept += kept
    counts.skipped += skipped
    counts.errors += errors

    parts = [plural(len(images), "image"), f"{moved} moved", f"{kept} kept"]
    if skipped:
        parts.append(f"{skipped} skipped")
    if errors:
        parts.append(plural(errors, "error"))
    if unknown:
        parts.append(f"{len(unknown)} unknown ext")
    print(f"    {', '.join(parts)}")


def main(argv=None) -> int:
    args = parse_args(argv)

    root = Path(args.folder).expanduser()
    if not root.exists():
        print(f"error: folder does not exist: {root}", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2
    root = root.resolve()
    if root.parent == root:
        print(f"error: {root} is a drive root; there is no name to build a "
              f"sidecar folder from", file=sys.stderr)
        return 2

    dest_root = root.parent / f"{SIDECAR_PREFIX}{root.name}"
    w, h = args.size
    rule = (f"either side shorter than {w}x{h}" if args.dim
            else f"fewer than {w * h:,} pixels ({w}x{h})")

    print(f"source      {root}")
    print(f"sidecar     {dest_root}")
    print(f"moving      images with {rule}")
    if args.dry_run:
        print("dry run     nothing will be moved")
    print()

    counts = Counts()
    folders, links, unreadable = folders_to_clean(root)
    for folder in folders:
        dest = dest_root / folder.relative_to(root)
        print(f"{folder}")
        try:
            clean_folder(folder, dest, args.size, args.dim, args.dry_run, counts)
        except OSError as exc:
            unreadable.append(f"{folder}: {exc.strerror or exc}")

    print()
    for link in links:
        print(f"not followed: {link} is a link or junction")
    for line in unreadable:
        print(f"cannot read: {line}")
    verb = "would move" if args.dry_run else "moved"
    print(f"{plural(counts.scanned, 'image')} scanned, {counts.moved} {verb}, "
          f"{counts.kept} kept.")
    if counts.skipped:
        print(f"{plural(counts.skipped, 'image')} left in place: a file of that "
              f"name is already in the sidecar folder.")
    if counts.unknown:
        was = "has" if counts.unknown == 1 else "have"
        print(f"{plural(counts.unknown, 'file')} {was} an extension this script "
              f"does not recognise and {'was' if counts.unknown == 1 else 'were'} "
              f"not looked at.")
    if unreadable:
        print(f"{plural(len(unreadable), 'folder')} could not be read.")
    if counts.errors:
        print(f"{plural(counts.errors, 'image')} could not be read or moved.")
    if counts.errors or unreadable:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
