"""Write MOSS-Image JSON caption sidecars from .txt and .short captions,
recursively.

For every image "name.png" (or jpg/jpeg/gif/webp) the same folder may hold
"name.txt" (the long caption) and "name.short" (the one-sentence caption).
The script writes "name.json" next to them:

    {
      "caption": {
        "long": "<content of name.txt>",
        "short": "<content of name.short>"
      }
    }

When only one of the two captions exists (or the other is empty), both
fields get the one that exists. When neither exists, no .json is written.
The .txt and .short files are not changed. An existing "name.json" is
overwritten. Folders whose names start with a dot are not scanned.

Captions are read as UTF-8 (a UTF-8 or UTF-16 byte order mark is honoured);
the .json is written as UTF-8 without escaping, so Cyrillic text stays
readable. A caption that is not valid UTF-8 is reported and its image is
skipped.

Usage:
    python moss-caption.py [folder] [--dry-run] [--quiet]
"""

import argparse
import codecs
import json
import os
import sys
import time

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
LONG_EXT = ".txt"
SHORT_EXT = ".short"
OUT_EXT = ".json"


def read_caption(path):
    """Caption text with surrounding whitespace removed, "" if empty."""
    with open(path, "rb") as f:
        data = f.read()
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        text = data.decode("utf-16")
    else:
        text = data.decode("utf-8-sig")
    return text.strip()


def scan(root, dry_run, quiet):
    written = single = skipped = failed = folders = 0
    stack = [root]
    while stack:
        folder = stack.pop()
        folders += 1
        if not quiet:
            print(f"Scanning: {folder}")

        # One directory listing per folder; names are matched case-insensitively.
        images = []
        captions = {}   # (stem.lower(), ext) -> path
        subdirs = []
        try:
            with os.scandir(folder) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if not entry.name.startswith("."):
                                subdirs.append(entry.path)
                            continue
                    except OSError:
                        continue
                    stem, ext = os.path.splitext(entry.name)
                    ext = ext.lower()
                    if ext in IMAGE_EXTS:
                        images.append(stem)
                    elif ext in (LONG_EXT, SHORT_EXT):
                        captions[(stem.lower(), ext)] = entry.path
        except OSError as e:
            print(f"  Cannot read folder: {e}", file=sys.stderr)
            continue

        for stem in sorted(images, key=str.lower):
            texts = {}
            bad = False
            for ext in (LONG_EXT, SHORT_EXT):
                path = captions.get((stem.lower(), ext))
                if path is None:
                    continue
                try:
                    texts[ext] = read_caption(path)
                except (OSError, UnicodeDecodeError) as e:
                    print(f"  FAILED to read {path}: {e}", file=sys.stderr)
                    bad = True
            if bad:
                failed += 1
                continue

            long_text = texts.get(LONG_EXT, "")
            short_text = texts.get(SHORT_EXT, "")
            if not long_text and not short_text:
                skipped += 1
                continue
            if not long_text or not short_text:
                single += 1
            doc = {"caption": {"long": long_text or short_text,
                               "short": short_text or long_text}}

            out = os.path.join(folder, stem + OUT_EXT)
            if dry_run:
                print(f"  Would write: {stem + OUT_EXT}")
                written += 1
                continue
            try:
                with open(out, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(doc, f, ensure_ascii=False, indent=2)
                    f.write("\n")
            except OSError as e:
                print(f"  FAILED to write {out}: {e}", file=sys.stderr)
                failed += 1
            else:
                written += 1

        # Visit subfolders in name order; the stack is LIFO, so push reversed.
        stack.extend(sorted(subdirs, key=str.lower, reverse=True))
    return folders, written, single, skipped, failed


def main():
    # A Cyrillic path printed to a cp1251 or cp437 console must not crash the run.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")

    ap = argparse.ArgumentParser(
        description="Recursively write MOSS-Image JSON caption sidecars (name.json) "
                    "from name.txt (long) and name.short (short).")
    ap.add_argument("folder", nargs="?", default=".",
                    help="root folder (default: current folder)")
    ap.add_argument("-n", "--dry-run", action="store_true",
                    help="only list what would be written")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="do not print the folder being scanned")
    args = ap.parse_args()

    root = os.path.abspath(args.folder)
    if not os.path.isdir(root):
        sys.exit(f"Not a folder: {root}")

    mode = " (dry run, nothing is written)" if args.dry_run else ""
    print(f"===== Writing MOSS-Image JSON captions in {root} and all subfolders{mode} =====")
    t0 = time.perf_counter()
    folders, written, single, skipped, failed = scan(root, args.dry_run, args.quiet)
    dt = time.perf_counter() - t0

    print()
    verb = "Would write" if args.dry_run else "Written"
    line = (f"Folders: {folders}   {verb}: {written}   (one caption only: {single})"
            f"   No caption: {skipped}")
    if failed:
        line += f"   Failed: {failed}"
    print(f"{line}   ({dt:.2f} s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
