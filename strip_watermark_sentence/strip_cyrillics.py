"""Cut every caption at the first sentence written in Cyrillic.

For each .txt file under the folder and all its subfolders, the first Cyrillic
sentence is removed together with the space before it and all text after it.
Typical captions are an English description followed by the original Russian
text, joined by "., ":

    ... Low-angle view, bright daylight., Калязинский радиотелескоп: ...

The cut is made at the first sentence that
- follows the "., " join, when the rest of the caption from there has any
  Cyrillic letter and the join is not inside a quotation, so the join is cut even when the Russian text opens with a
  year or a Latin name ("., 1965-1974. Дин Конгер ...", "., Vogue, СССР ..."),
- follows a closing quote and a comma ('reads "Анапа 1972г.", Те самые ...'),
  or a "., " join that seems to be inside a quotation, when the text from
  there to the next "., " join has more Cyrillic than Latin letters once the
  quotations in it are left out. In English a quotation is also followed by a
  comma and more of the sentence ('reads "Мир меняется...", with a seating
  area', 'signs read "ЧТО ДЕЛАТЬ?", "ГАМЛЕТ", and ...'), and quotes that do
  not pair up ('"„ОКНО В ЕВРОПУ" ЧТОБ ... ЛЕНИНГРАД!"') make the rest of a
  caption look quoted.
- has more Cyrillic than Latin letters and starts with a Cyrillic letter,
  once the quotations in it are left out.
So Latin words in Russian text do not save it, and an English sentence that
quotes Russian text stays ('A sign reads "ГАСТРОНОМ".', '"СЛАВА КПСС" is
painted on the roof.').

A sentence ends at ".", "!", "?" or "…" (optionally with closing quotes and the
comma of the join) followed by whitespace, but not inside "double quotes" or
«guillemets»: in 'text reads "г. Норильск. Драмтеатр."' the periods inside the
quotation are not sentence breaks.

A caption whose first sentence is Cyrillic would become empty, so it is left
as it is and listed.

Usage:
    python strip_cyrillics.py [folder] [--dry-run]
"""

import argparse
import codecs
import re
import sys
from pathlib import Path

BOM = codecs.BOM_UTF8
CYRILLIC = re.compile(r"[\u0400-\u04FF]")
LATIN = re.compile(r"[A-Za-z]")
LETTER = re.compile(r"[^\W\d_]")
# End of a sentence: the punctuation, closing quotes, the comma of a join, whitespace.
BOUNDARY = re.compile(r"""[.!?…](["'»”)]*)(,?)\s+""")
QUOTATION = re.compile(r'"[^"]*"|«[^»]*»')
SURE, WEAK = "sure", "weak"  # a "., " join; a '.", ' join or one that looks quoted


def outside_quotes(text, end):
    """True when text[:end] has no open quotation."""
    head = text[:end]
    return head.count('"') % 2 == 0 and head.count("«") <= head.count("»")


def sentences(text):
    """Yield (start, join) for every sentence; join is SURE, WEAK or None."""
    yield 0, None
    for match in BOUNDARY.finditer(text):
        if match.end() == len(text):
            continue
        outside = outside_quotes(text, match.end())
        if match.group(2):
            yield match.end(), SURE if outside and not match.group(1) else WEAK
        elif outside:
            yield match.end(), None


def mostly_cyrillic(text):
    return len(CYRILLIC.findall(text)) > len(LATIN.findall(text))


def is_cyrillic(sentence):
    sentence = QUOTATION.sub("", sentence)
    first = LETTER.search(sentence)
    return bool(first and CYRILLIC.match(first.group())) and mostly_cyrillic(sentence)


def weak_join_cyrillic(rest):
    """True when rest, up to the next "., " join and without quotations, is mostly Cyrillic."""
    sure = re.search(r"[.!?…],\s", rest)
    return mostly_cyrillic(QUOTATION.sub("", rest[:sure.start() if sure else len(rest)]))


def strip(text):
    """Return the text cut before the first Cyrillic sentence, or None if there is none."""
    starts = list(sentences(text))
    for (start, join), (end, _) in zip(starts, starts[1:] + [(len(text), None)]):
        rest = text[start:]
        if ((join == SURE and CYRILLIC.search(rest))
                or (join == WEAK and weak_join_cyrillic(rest))
                or is_cyrillic(text[start:end])):
            head = text[:start].rstrip()
            return head[:-1] if join else head  # drop the comma of the join
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", nargs="?", default=Path.cwd(), type=Path,
                        help="folder to scan recursively (default: the current folder)")
    parser.add_argument("--dry-run", action="store_true", help="show the results without writing files")
    args = parser.parse_args()
    if not args.folder.is_dir():
        parser.error(f"folder not found: {args.folder}")

    scanned = changed = skipped = 0
    for path in sorted(args.folder.rglob("*.txt")):
        scanned += 1
        raw = path.read_bytes()
        bom = raw.startswith(BOM)
        text = raw.decode("utf-8-sig")
        result = strip(text)
        if result is None:
            continue
        if not result:
            print(f"SKIP (the caption starts in Cyrillic): {path}")
            skipped += 1
            continue
        changed += 1
        if args.dry_run:
            print(f"{path}\n  -> ...{result[-100:]}\n")
            continue
        path.write_bytes((BOM if bom else b"") + result.encode("utf-8"))

    action = "would change" if args.dry_run else "changed"
    print(f"scanned {scanned} file(s), {action} {changed}, skipped {skipped}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
