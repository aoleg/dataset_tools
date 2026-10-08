"""Cut every caption at the first sentence that contains a given word or phrase.

For each .txt file under the folder and all its subfolders, the first sentence
that contains the phrase is removed together with the space before it and all
text after it. The match ignores letter case and also finds the phrase inside
longer words ("watermark" matches "watermarked", "pastvu" matches "pastvu.com").
Any run of spaces in the phrase matches any run of whitespace in the text.

A sentence starts after ". ", "! " or "? " (optionally with a closing quote
before the space) when the next word starts with a capital letter or a quote.
So a period inside a quotation, as in 'reads "Ялта 1957г." and a watermark',
does not count as a sentence break.

Usage:
    python strip_watermark_sentence.py <phrase> [folder] [--dry-run]

Examples:
    python strip_watermark_sentence.py watermark
    python strip_watermark_sentence.py pastvu --dry-run
    python strip_watermark_sentence.py "small print" MIX-Dataset
"""

import argparse
import codecs
import re
import sys
from pathlib import Path

BOM = codecs.BOM_UTF8
BOUNDARY = re.compile(r"""[.!?]["'»”]?(?=\s+["'«“A-ZА-ЯЁ])""")


def phrase_pattern(phrase):
    """Compile a case-insensitive pattern for the phrase, flexible about whitespace."""
    words = phrase.split()
    if not words:
        raise ValueError("the phrase is empty")
    return re.compile(r"\s+".join(map(re.escape, words)), re.IGNORECASE)


def strip(text, pattern):
    """Return the text cut before the sentence that matches, or None if nothing matches."""
    match = pattern.search(text)
    if not match:
        return None
    start = 0
    for boundary in BOUNDARY.finditer(text, 0, match.start()):
        start = boundary.end()
    return text[:start].rstrip()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog='Quote a phrase that has spaces: "small print".')
    parser.add_argument("phrase", help="word or phrase that marks the sentence to cut")
    parser.add_argument("folder", nargs="?", default=Path.cwd(), type=Path,
                        help="folder to scan recursively (default: the current folder)")
    parser.add_argument("--dry-run", action="store_true", help="show the results without writing files")
    args = parser.parse_args()

    try:
        pattern = phrase_pattern(args.phrase)
    except ValueError as error:
        parser.error(str(error))
    if not args.folder.is_dir():
        parser.error(f"folder not found: {args.folder}")

    scanned = changed = skipped = 0
    for path in sorted(args.folder.rglob("*.txt")):
        scanned += 1
        raw = path.read_bytes()
        bom = raw.startswith(BOM)
        text = raw.decode("utf-8-sig")
        result = strip(text, pattern)
        if result is None:
            continue
        if not result:
            print(f"SKIP (the phrase is in the first sentence): {path}")
            skipped += 1
            continue
        changed += 1
        if args.dry_run:
            print(f"{path}\n  -> ...{result[-100:]}\n")
            continue
        path.write_bytes((BOM if bom else b"") + result.encode("utf-8"))

    action = "would change" if args.dry_run else "changed"
    print(f'scanned {scanned} file(s) for "{args.phrase}", {action} {changed}, skipped {skipped}')


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
