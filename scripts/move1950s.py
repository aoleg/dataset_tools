"""Move images captioned with a pre-1950 year out of a dataset, recursively.

For every image under <folder> the caption sidecar is read: "<stem>.json"
when it exists and parses (every string value in it is searched), otherwise
"<stem>.txt". A four-digit year starting with 18 or 19, alone or as a decade
("1930s", "1930's"), that is earlier than the cutoff marks the image as old.
Digit runs that are part of a longer number or token ("1920x1080", "M1911",
"159911334") are not years.

An old image and every sidecar of its name (all non-image files in the same
folder whose name without extension equals the image's) are moved to the
sibling folder "<folder>_pre_1950s", keeping the directory tree. Images with
no year, or only years at or past the cutoff, stay.

Usage:
    python move1950s.py <folder> [--dry-run] [--cutoff 1950] [--quiet]
"""

import argparse
import json
import os
import re
import shutil
import sys

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".avif"}
CAPTION_JSON = ".json"
CAPTION_TXT = ".txt"

# 18xx or 19xx, optionally a decade suffix, not glued to another word character.
YEAR_RE = re.compile(r"(?<!\w)((?:18|19)\d{2})(?:s|'s|’s)?(?!\w)")


def read_text(path):
    with open(path, "rb") as f:
        return f.read().decode("utf-8-sig", errors="replace")


def json_strings(doc):
    """Yield every string value in a parsed JSON document."""
    if isinstance(doc, str):
        yield doc
    elif isinstance(doc, dict):
        for v in doc.values():
            yield from json_strings(v)
    elif isinstance(doc, list):
        for v in doc:
            yield from json_strings(v)


def caption_text(folder, stem):
    """Text to search for years: the JSON caption, else the .txt caption.

    Returns (text, source) where source is the sidecar used, or (None, None)
    when the image has no readable caption.
    """
    json_path = os.path.join(folder, stem + CAPTION_JSON)
    if os.path.isfile(json_path):
        try:
            raw = read_text(json_path)
        except OSError:
            raw = None
        if raw is not None:
            try:
                doc = json.loads(raw)
            except ValueError:
                return raw, json_path
            return "\n".join(json_strings(doc)), json_path
    txt_path = os.path.join(folder, stem + CAPTION_TXT)
    if os.path.isfile(txt_path):
        try:
            return read_text(txt_path), txt_path
        except OSError:
            pass
    return None, None


def years_in(text):
    return sorted({int(m.group(1)) for m in YEAR_RE.finditer(text)})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", help="dataset folder to scan recursively")
    ap.add_argument("--cutoff", type=int, default=1950,
                    help="images with any year before this move (default 1950)")
    ap.add_argument("--dry-run", action="store_true", help="report only, move nothing")
    ap.add_argument("--quiet", action="store_true", help="print the summary only")
    args = ap.parse_args()

    src_root = os.path.abspath(args.folder.rstrip("\\/"))
    if not os.path.isdir(src_root):
        print(f"not a folder: {src_root}", file=sys.stderr)
        return 2
    dst_root = f"{src_root}_pre_{args.cutoff}s"

    images = moved = old = no_caption = no_year = recent = failed = 0
    sidecars_moved = 0

    for folder, dirs, files in os.walk(src_root):
        dirs.sort()
        by_stem = {}
        for name in files:
            by_stem.setdefault(os.path.splitext(name)[0], []).append(name)
        for name in sorted(files):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in IMAGE_EXTS:
                continue
            images += 1
            text, source = caption_text(folder, stem)
            if text is None:
                no_caption += 1
                continue
            years = years_in(text)
            if not years:
                no_year += 1
                continue
            if years[0] >= args.cutoff:
                recent += 1
                continue
            old += 1

            rel = os.path.relpath(folder, src_root)
            dst_folder = dst_root if rel == "." else os.path.join(dst_root, rel)
            group = [name] + [n for n in by_stem.get(stem, [])
                              if n != name and os.path.splitext(n)[1].lower() not in IMAGE_EXTS]
            if not args.quiet:
                tag = "would move" if args.dry_run else "move"
                print(f"{tag}: {os.path.join(rel, name) if rel != '.' else name}"
                      f"  [{', '.join(str(y) for y in years)}]"
                      f"  +{len(group) - 1} sidecar(s)")
            if args.dry_run:
                moved += 1
                sidecars_moved += len(group) - 1
                continue
            try:
                os.makedirs(dst_folder, exist_ok=True)
                for n in group:
                    src = os.path.join(folder, n)
                    dst = os.path.join(dst_folder, n)
                    if not os.path.exists(src):
                        continue  # shared sidecar already moved with another image
                    if os.path.exists(dst):
                        raise FileExistsError(dst)
                    shutil.move(src, dst)
                    if n == name:
                        moved += 1
                    else:
                        sidecars_moved += 1
            except OSError as e:
                failed += 1
                print(f"FAILED: {os.path.join(folder, name)}: {e}", file=sys.stderr)

    verb = "would move" if args.dry_run else "moved"
    print(f"\n{src_root}\n  -> {dst_root}")
    print(f"images: {images}  pre-{args.cutoff}: {old}  {args.cutoff}+ only: {recent}"
          f"  no year: {no_year}  no caption: {no_caption}")
    print(f"{verb}: {moved} image(s), {sidecars_moved} sidecar(s)"
          + (f"  FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
