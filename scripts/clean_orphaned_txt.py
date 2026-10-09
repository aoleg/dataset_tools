"""Delete .txt caption sidecars that have no matching image, recursively.

A sidecar "name.txt" is kept when the same folder holds "name.png",
"name.jpg", "name.jpeg" or "name.gif" (case-insensitive). Every other
.txt file is deleted.

Usage:
    python clean_orphaned_txt.py [folder] [--dry-run] [--quiet]
"""

import argparse
import os
import sys
import time

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif"}


def scan(root, dry_run, quiet):
    kept = deleted = failed = folders = 0
    stack = [root]
    while stack:
        folder = stack.pop()
        folders += 1
        if not quiet:
            print(f"Scanning: {folder}")

        # One directory listing per folder; no per-file existence checks.
        images = set()
        texts = []
        subdirs = []
        try:
            with os.scandir(folder) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            subdirs.append(entry.path)
                            continue
                    except OSError:
                        continue
                    stem, ext = os.path.splitext(entry.name)
                    ext = ext.lower()
                    if ext in IMAGE_EXTS:
                        images.add(stem.lower())
                    elif ext == ".txt":
                        texts.append(entry)
        except OSError as e:
            print(f"  Cannot read folder: {e}", file=sys.stderr)
            continue

        for entry in texts:
            if os.path.splitext(entry.name)[0].lower() in images:
                kept += 1
                continue
            if dry_run:
                print(f"  Would delete orphan (no matching image): {entry.name}")
                deleted += 1
                continue
            try:
                os.remove(entry.path)
            except OSError as e:
                print(f"  FAILED to delete {entry.name}: {e}", file=sys.stderr)
                failed += 1
            else:
                print(f"  Deleting orphan (no matching image): {entry.name}")
                deleted += 1

        # Visit subfolders in name order; the stack is LIFO, so push reversed.
        stack.extend(sorted(subdirs, key=str.lower, reverse=True))
    return folders, kept, deleted, failed


def main():
    ap = argparse.ArgumentParser(
        description="Recursively delete .txt sidecars with no matching "
                    "png/jpg/jpeg/gif in the same folder.")
    ap.add_argument("folder", nargs="?", default=".",
                    help="root folder (default: current folder)")
    ap.add_argument("-n", "--dry-run", action="store_true",
                    help="only list what would be deleted")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="do not print the folder being scanned")
    args = ap.parse_args()

    root = os.path.abspath(args.folder)
    if not os.path.isdir(root):
        sys.exit(f"Not a folder: {root}")

    mode = " (dry run, nothing is deleted)" if args.dry_run else ""
    print(f"===== Checking sidecar TXT files in {root} and all subfolders{mode} =====")
    t0 = time.perf_counter()
    folders, kept, deleted, failed = scan(root, args.dry_run, args.quiet)
    dt = time.perf_counter() - t0

    print()
    verb = "Would delete" if args.dry_run else "Deleted"
    line = f"Folders: {folders}   Kept: {kept}   {verb}: {deleted}"
    if failed:
        line += f"   Failed: {failed}"
    print(f"{line}   ({dt:.2f} s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
