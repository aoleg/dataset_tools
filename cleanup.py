#!/usr/bin/env python3
"""cleanup.py - move undersized images out of a training folder.

Walks a folder and its first-level subfolders and moves every image smaller
than a given size, together with its .txt caption sidecar, into a sidecar
folder named after the source with a leading underscore. The original folder
structure is preserved inside it:

    cleanup.py T:\\somefolder 1024

    T:\\somefolder\\small.jpg     ->  T:\\_somefolder\\small.jpg
    T:\\somefolder\\1\\small.jpg   ->  T:\\_somefolder\\1\\small.jpg

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

__version__ = "1.0.0"

# Mirrors k2prep's IMAGE_EXTENSIONS, for the same reason: these are the
# extensions musubi-tuner recognises. Mixed-case variants such as .Jpg are not
# matched; they are counted and listed so you notice them.
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"]
KNOWN_EXTENSIONS = set(IMAGE_EXTENSIONS) | {e.upper() for e in IMAGE_EXTENSIONS}

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
        description="Move images below a size threshold, and their .txt "
                    "sidecars, out of a folder and its first-level subfolders "
                    "into a _foldername sidecar folder next to it.",
    )
    p.add_argument("folder", help="Folder to clean. First-level subfolders are "
                                  "cleaned too, except those starting with an "
                                  "underscore.")
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

def folders_to_clean(root: Path) -> list[Path]:
    """The root itself, then its first-level subfolders in name order."""
    subs = []
    for entry in os.scandir(root):
        if entry.is_dir() and not entry.name.startswith(SIDECAR_PREFIX):
            subs.append(Path(entry.path))
    subs.sort(key=lambda p: p.name.lower())
    return [root] + subs


def images_in(folder: Path) -> tuple[list[Path], list[str]]:
    """Non-recursive. Returns (image paths by filename, unknown-extension
    filenames). .txt is not unknown: it is caption data."""
    images: list[Path] = []
    unknown: list[str] = []
    for entry in os.scandir(folder):
        if entry.is_dir():
            continue
        ext = os.path.splitext(entry.name)[1]
        if ext in KNOWN_EXTENSIONS:
            images.append(Path(entry.path))
        elif ext.lower() != ".txt":
            unknown.append(entry.name)
    images.sort(key=lambda p: p.name)
    unknown.sort()
    return images, unknown


def caption_for(path: Path) -> Path | None:
    cap = path.parent / f"{path.stem}.txt"
    return cap if cap.is_file() else None


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
    images, unknown = images_in(folder)
    counts.unknown += len(unknown)

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

        pairs = [(path, dest / path.name)]
        cap = caption_for(path)
        if cap is not None:
            pairs.append((cap, dest / cap.name))

        # Never overwrite. Renaming on collision would be friendlier right up
        # until the image and its caption disagree about their new stem, so a
        # colliding pair is left where it is and reported instead.
        clash = [dst for _, dst in pairs if dst.exists()]
        if clash:
            print(f"    skip   {path.name}: {clash[0].name} already exists in "
                  f"{dest}")
            skipped += 1
            continue

        if dry_run:
            cap_note = " (+ .txt)" if cap is not None else ""
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
    for folder in folders_to_clean(root):
        dest = dest_root if folder == root else dest_root / folder.name
        print(f"{folder}")
        clean_folder(folder, dest, args.size, args.dim, args.dry_run, counts)

    print()
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
    if counts.errors:
        print(f"{plural(counts.errors, 'image')} could not be read or moved.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
