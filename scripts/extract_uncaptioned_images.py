"""Move images that have no .txt caption sidecar into "<root>_uncaptioned", recursively.

An image "name.png" counts as captioned when the same folder holds
"name.txt" (case-insensitive). Every uncaptioned png/jpg/jpeg/gif/webp is
moved to "<root>_uncaptioned/<same relative path>", next to the root, so the
original directory tree is kept. The folder is next to the root, not inside
it, because a trainer that reads all subfolders (Ostris AI Toolkit trains
every subfolder whose name does not start with a dot) would train the
uncaptioned images. Until 2026-10-11 it was "<root>/_uncaptioned"; such a
folder is moved out on the next run, or restored from where it is.

Not scanned: folders whose names start with a dot, the output folders of the
dataset tools (EXCLUDES: face masks, face crops, duplicates and the others,
which have no captions by design), and any folder the dataset tools marked
with MARK.

After captioning, --restore moves every file in "<root>_uncaptioned" (images
and their new .txt files) back to the same place in the tree and removes the
emptied folders.

Usage:
    python extract_uncaptioned_images.py [folder] [--dry-run] [--quiet]
    python extract_uncaptioned_images.py [folder] --restore [--dry-run]
"""

import argparse
import os
import sys
import time

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
OUT_SUFFIX = "_uncaptioned"     # <root>_uncaptioned next to the root
OLD_OUT_NAME = "_uncaptioned"   # the same folder inside the root until 2026-10-11
# The output folders of the dataset tools, at any depth: _controls holds the
# face masks (masks before 2026-10-11), faces the face crops, and the rest
# hold copies or originals that have their captions elsewhere.
EXCLUDES = {OLD_OUT_NAME, "_controls", "masks", "faces", "_backup", "_duplicates", "_prep",
            "_classify", "_embeddings", "_reframed"}
MARK = "_dataset_tools.json"    # marks a folder the dataset tools made next to a dataset


def skipped(entry):
    name = entry.name
    return (name.startswith(".") or name.casefold() in EXCLUDES
            or os.path.isfile(os.path.join(entry.path, MARK)))


def move(src, dst, dry_run, label):
    """Move src to dst without ever overwriting. Returns True on success."""
    if os.path.lexists(dst):
        print(f"  SKIPPED, target exists: {dst}", file=sys.stderr)
        return False
    if dry_run:
        print(f"  Would move {label}: {src}")
        return True
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.rename(src, dst)
    except OSError as e:
        print(f"  FAILED to move {src}: {e}", file=sys.stderr)
        return False
    print(f"  Moving {label}: {src}")
    return True


def extract(root, out, dry_run, quiet):
    captioned = moved = failed = folders = 0
    stack = [root]
    while stack:
        folder = stack.pop()
        folders += 1
        if not quiet:
            print(f"Scanning: {folder}")

        # One directory listing per folder; no per-file existence checks.
        images = []
        texts = set()
        subdirs = []
        try:
            with os.scandir(folder) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if (os.path.normcase(entry.path) != os.path.normcase(out)
                                    and not skipped(entry)):
                                subdirs.append(entry.path)
                            continue
                    except OSError:
                        continue
                    stem, ext = os.path.splitext(entry.name)
                    ext = ext.lower()
                    if ext in IMAGE_EXTS:
                        images.append(entry)
                    elif ext == ".txt":
                        texts.add(stem.lower())
        except OSError as e:
            print(f"  Cannot read folder: {e}", file=sys.stderr)
            continue

        for entry in sorted(images, key=lambda e: e.name.lower()):
            if os.path.splitext(entry.name)[0].lower() in texts:
                captioned += 1
                continue
            dst = os.path.join(out, os.path.relpath(entry.path, root))
            if move(entry.path, dst, dry_run, "uncaptioned image"):
                moved += 1
            else:
                failed += 1

        # Visit subfolders in name order; the stack is LIFO, so push reversed.
        stack.extend(sorted(subdirs, key=str.lower, reverse=True))
    return folders, captioned, moved, failed


def restore(root, out, dry_run):
    moved = failed = 0
    for folder, dirs, files in os.walk(out):
        dirs.sort(key=str.lower)
        for name in sorted(files, key=str.lower):
            src = os.path.join(folder, name)
            dst = os.path.join(root, os.path.relpath(src, out))
            if move(src, dst, dry_run, "back"):
                moved += 1
            else:
                failed += 1

    # Remove folders that are empty now, deepest first; keep anything left over.
    if not dry_run:
        for folder, _, _ in os.walk(out, topdown=False):
            try:
                os.rmdir(folder)
            except OSError:
                pass
    return moved, failed


def move_old_out(root, out, dry_run):
    """An old <root>/_uncaptioned: renamed to out when out is not there yet.
    -> the folders that --restore has to empty: out and, when both exist, old."""
    old = os.path.join(root, OLD_OUT_NAME)
    if not os.path.isdir(old):
        return [out]
    if not os.path.lexists(out):
        if dry_run:
            print(f"Would move {old} out of the folder, to {out}")
            return [old]
        os.rename(old, out)
        print(f"Moved {old} out of the folder, to {out}")
        return [out]
    print(f"Both {old} and {out} exist; --restore empties both", file=sys.stderr)
    return [out, old]


def main():
    ap = argparse.ArgumentParser(
        description=f"Recursively move png/jpg/jpeg/gif/webp images with no "
                    f".txt sidecar into <folder>{OUT_SUFFIX} next to the folder, keeping the folder tree.")
    ap.add_argument("folder", nargs="?", default=".",
                    help="root folder (default: current folder)")
    ap.add_argument("-r", "--restore", action="store_true",
                    help=f"move everything in <folder>{OUT_SUFFIX} back into the tree")
    ap.add_argument("-n", "--dry-run", action="store_true",
                    help="only list what would be moved")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="do not print the folder being scanned")
    args = ap.parse_args()
    # A piped or redirected console may not encode every file name.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")

    root = os.path.abspath(args.folder)
    if not os.path.isdir(root):
        sys.exit(f"Not a folder: {root}")
    if os.path.dirname(root) == root:
        sys.exit(f"{root}: a drive root has no folder next to it for {OUT_SUFFIX}; use a subfolder")
    out = root + OUT_SUFFIX
    outs = move_old_out(root, out, args.dry_run)
    mode = " (dry run, nothing is moved)" if args.dry_run else ""
    verb = "Would move" if args.dry_run else "Moved"
    t0 = time.perf_counter()

    if args.restore:
        outs = [o for o in outs if os.path.isdir(o)]
        if not outs:
            sys.exit(f"Nothing to restore, no folder: {out}")
        moved = failed = 0
        for o in outs:
            print(f"===== Restoring {o} into {root}{mode} =====")
            m, f = restore(root, o, args.dry_run)
            moved += m
            failed += f
        line = f"{verb} back: {moved}"
    else:
        print(f"===== Moving uncaptioned images in {root} and all subfolders "
              f"to {out}{mode} =====")
        folders, captioned, moved, failed = extract(
            root, out, args.dry_run, args.quiet)
        line = f"Folders: {folders}   Captioned: {captioned}   {verb}: {moved}"

    dt = time.perf_counter() - t0
    print()
    if failed:
        line += f"   Failed: {failed}"
    print(f"{line}   ({dt:.2f} s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
