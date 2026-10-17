"""Shared query engine: one DuckDB scan over local Parquet, one img2dataset-ready CSV.

A dataset is described by a DatasetSpec (see duck_datasets/). The spec says which
columns can be searched, which numeric filters exist, what rides along in the
CSV and how rows are ordered. Everything else, from argument parsing to the CSV
writer, lives here and is the same for every dataset.

Matching rules (identical for every dataset):

* Inclusion terms are case-insensitive substrings, and every term must match,
  so "cowboy" also finds "cowboys". Inside one term, "|" separates alternatives
  ("cowboy|cowgirl") and "+" stands for a space ("cowboy+hat" matches the phrase
  "cowboy hat").
* Exclusion terms are whole words, matched case-insensitively, and any one of
  them drops the row. "|" and "+" work the same way.
* Any option may be given as "*", which means "not set", so a sample .bat can
  list every option and be edited in place.
"""

import argparse
import csv
import datetime as _dt
import glob as _glob
import json
import os
import re
import sys
from dataclasses import dataclass, field

import duckdb


# --------------------------------------------------------------------------- spec


@dataclass
class TextColumn:
    """A text column that can be searched (--<flag>) and excluded (--exclude-<flag>)."""

    column: str
    flag: str
    help: str = ""


@dataclass
class NumericFilter:
    """A numeric threshold, e.g. --min-width 1024 -> width >= 1024."""

    flag: str
    column: str
    op: str
    cast: type
    help: str = ""


@dataclass
class RangeFilter:
    """An integer that is given exactly ("1") or as an inclusive range ("1-2")."""

    flag: str
    column: str
    help: str = ""


@dataclass
class OutputColumn:
    """A column copied into the CSV after url and text."""

    header: str
    expr: str
    is_text: bool = False


@dataclass
class DatasetSpec:
    name: str
    title: str
    hf_id: str
    file_glob: str
    quick_test_glob: str
    text_columns: list
    default_caption: str
    output_columns: list
    order_by: str
    numeric_filters: list = field(default_factory=list)
    range_filters: list = field(default_factory=list)
    splits: dict = None
    add_arguments: callable = None
    caption_expr: callable = None
    notes: str = ""

    def text_column(self, column):
        for tc in self.text_columns:
            if tc.column == column:
                return tc
        raise KeyError(column)


# --------------------------------------------------------------------------- helpers

REGEX_SPECIAL = set("\\.^$|()[]{}*+?")

# Whitespace is never valid in a url, and a stray line break there breaks both
# the CSV parse and the download. Always applied, whatever --newlines says.
URL_EXPR = "trim(regexp_replace(url, '[\\r\\n]+', '', 'g'))"


def fail(message):
    print(f"Error: {message}")
    sys.exit(1)


def unset(value):
    """A "*" placeholder (or nothing at all) means: leave this filter out."""
    return value is None or str(value).strip() in ("", "*")


def regex_literal(word):
    """Escape a word so RE2 treats it literally."""
    return "".join("\\" + c if c in REGEX_SPECIAL else c for c in word)


def parse_number(value, label, cast):
    try:
        return cast(str(value).strip())
    except ValueError:
        fail(f"--{label} expects a number or '*', got {value!r}")


def sanitize_newlines(expr, mode):
    """Fold line breaks out of a text column.

    A newline inside a quoted CSV value is legal, but img2dataset reads the list
    with pyarrow, which splits files into 1 MB blocks for multithreaded parsing
    and then dies with "Expected N columns, got M" when a block boundary lands
    inside one. Small lists slip through and only big pulls break, so the
    default is to fold them away.
    """
    if mode == "keep":
        return expr
    if mode == "escape":
        backslash_n = "chr(92) || 'n'"
        return (
            f"replace(replace(replace({expr}, chr(13) || chr(10), {backslash_n}), "
            f"chr(10), {backslash_n}), chr(13), {backslash_n})"
        )
    return f"trim(regexp_replace({expr}, '\\s*[\\r\\n]+\\s*', ' ', 'g'))"


def split_terms(value):
    """"cowboy|cowgirl hat+band" -> [["cowboy", "cowgirl"], ["hat band"]]"""
    terms = []
    for word in str(value).split():
        alts = [a.replace("+", " ") for a in word.split("|") if a]
        if alts:
            terms.append(alts)
    return terms


def argname(flag):
    return flag.replace("-", "_")


# --------------------------------------------------------------------------- parser


def build_parser(spec, prog):
    parser = argparse.ArgumentParser(
        prog=prog,
        description=f"{spec.title} ({spec.hf_id}). Any option may be '*', meaning 'not set'.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Matching rules", 1)[1].join(["Matching rules", ""]),
    )
    parser.add_argument("--dir", default="*", help="Folder holding the .parquet files (overrides duck.conf).")
    if spec.splits:
        parser.add_argument("--split", default="all", choices=sorted(spec.splits),
                            help="Which source split to search (default: all).")

    search = parser.add_argument_group("search (every term must match; substring)")
    default_tc = spec.text_column(spec.default_caption)
    search.add_argument("--caption", default="*",
                        help=f"Terms that must appear in the caption ({default_tc.column}).")
    for tc in spec.text_columns:
        search.add_argument(f"--{tc.flag}", default="*",
                            help=tc.help or f"Terms that must appear in {tc.column}.")
    for rf in spec.range_filters:
        search.add_argument(f"--{rf.flag}", default="*", help=rf.help)
    for nf in spec.numeric_filters:
        search.add_argument(f"--{nf.flag}", default="*", help=nf.help)

    exclude = parser.add_argument_group("exclusions (whole word; any one drops the row)")
    exclude.add_argument("--exclude", default="*",
                         help=f"Words that must NOT appear in the caption ({default_tc.column}).")
    for tc in spec.text_columns:
        exclude.add_argument(f"--exclude-{tc.flag}", default="*",
                             help=f"Words that must NOT appear in {tc.column}.")

    output = parser.add_argument_group("output")
    output.add_argument("--limit", default="*", help="Max number of rows to write ('*' = no limit).")
    output.add_argument("--start", default="0", help="Number of matching rows to skip (for paging).")
    output.add_argument("--caption-col", default=spec.default_caption,
                        choices=[tc.column for tc in spec.text_columns],
                        help=f"Column written as the CSV 'text' column, i.e. the downloaded .txt caption "
                             f"(default: {spec.default_caption}).")
    output.add_argument("--newlines", default="space", choices=["space", "escape", "keep"],
                        help="Line breaks in text: fold to a space (default), write as a literal "
                             "backslash-n, or keep raw.")
    output.add_argument("--out", default="*", help="Output .csv path (default: results/<dataset>/<terms>-<start>.csv).")
    output.add_argument("--count", action="store_true",
                        help="Only count the matching rows; write nothing.")
    output.add_argument("--quick-test", action="store_true",
                        help=f"Scan only the first shard(s): {spec.quick_test_glob}")
    output.add_argument("--no-progress", action="store_true", help="Hide the progress bar during the scan.")
    output.add_argument("--progress-after", default="2000", help="Show the progress bar after this many ms.")

    if spec.add_arguments:
        spec.add_arguments(parser)
    return parser


# --------------------------------------------------------------------------- query


def add_range(value, column, label, conditions, params):
    text = str(value).strip()
    ranged = re.fullmatch(r"(\d+)\s*-\s*(\d+)", text)
    if ranged:
        low, high = int(ranged.group(1)), int(ranged.group(2))
        if low > high:
            fail(f"--{label} range is backwards: {text!r}")
        conditions.append(f"{column} BETWEEN ? AND ?")
        params.extend([low, high])
        return
    if re.fullmatch(r"\d+", text):
        conditions.append(f"{column} = ?")
        params.append(int(text))
        return
    fail(f"--{label} expects a value like '1', a range like '1-2', or '*', got {text!r}")


def build_where(spec, args):
    """Return (sql_conditions, parameters). Values are bound, never interpolated."""
    conditions = []
    params = []

    inclusions = [(spec.default_caption, args.caption)]
    inclusions += [(tc.column, getattr(args, argname(tc.flag))) for tc in spec.text_columns]
    for column, value in inclusions:
        if unset(value):
            continue
        for alts in split_terms(value):
            parts = []
            for alt in alts:
                parts.append(f"coalesce(contains(lower({column}), ?), false)")
                params.append(alt.lower())
            conditions.append("(" + " OR ".join(parts) + ")")

    exclusions = [(spec.default_caption, args.exclude)]
    exclusions += [(tc.column, getattr(args, "exclude_" + argname(tc.flag))) for tc in spec.text_columns]
    for column, value in exclusions:
        if unset(value):
            continue
        for alts in split_terms(value):
            pattern = "|".join(regex_literal(a) for a in alts)
            conditions.append(f"NOT coalesce(regexp_matches({column}, ?, 'i'), false)")
            params.append(r"\b(" + pattern + r")\b")

    for rf in spec.range_filters:
        value = getattr(args, argname(rf.flag))
        if not unset(value):
            add_range(value, rf.column, rf.flag, conditions, params)

    for nf in spec.numeric_filters:
        value = getattr(args, argname(nf.flag))
        if unset(value):
            continue
        conditions.append(f"{nf.column} {nf.op} ?")
        params.append(parse_number(value, nf.flag, nf.cast))

    return conditions, params


def build_slug(spec, args):
    terms = []
    for value in [args.caption] + [getattr(args, argname(tc.flag)) for tc in spec.text_columns]:
        if not unset(value):
            terms.extend(str(value).split())
    parts = []
    for t in terms:
        t = re.sub(r"[|+]", "-", t.lower())
        t = re.sub(r"[^a-z0-9-]+", "", t).strip("-")
        if t:
            parts.append(t)
    slug = "-".join(parts) or "all"
    return slug[:80]


def unique_path(path):
    """lighthouse-0.csv, lighthouse-0-2.csv, lighthouse-0-3.csv, ..."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 2
    while os.path.exists(f"{stem}-{n}{ext}"):
        n += 1
    return f"{stem}-{n}{ext}"


def resolve_glob(spec, args):
    pattern = spec.quick_test_glob if args.quick_test else spec.file_glob
    if spec.splits:
        pattern = pattern.format(split=spec.splits[args.split])
    return pattern


def active_filters(spec, args, parser):
    """The options that differ from their defaults, for the sidecar .json."""
    out = {}
    for action in parser._actions:
        if not action.option_strings or action.dest in ("help", "dir", "out", "no_progress", "progress_after"):
            continue
        value = getattr(args, action.dest, None)
        if value != action.default and not (isinstance(value, str) and unset(value) and unset(action.default)):
            out[action.option_strings[-1]] = value
    return out


def run(spec, args, parser, data_dir, results_dir):
    limit = None if unset(args.limit) else parse_number(args.limit, "limit", int)
    start = parse_number("0" if unset(args.start) else args.start, "start", int)
    if limit is not None and limit <= 0:
        fail("--limit must be positive, or '*' for no limit")
    if start < 0:
        fail("--start cannot be negative")

    data_dir = os.path.normpath(data_dir)
    if not os.path.isdir(data_dir):
        fail(f"dataset folder does not exist: {data_dir}\n"
             f"Set [paths] {spec.name} in duck.conf, or pass --dir.")
    pattern = resolve_glob(spec, args)
    file_glob = os.path.join(data_dir, pattern).replace("\\", "/")
    n_files = len(_glob.glob(file_glob))
    if n_files == 0:
        fail(f"no files match {pattern} in {data_dir}")

    conditions, params = build_where(spec, args)
    where_clause = " AND ".join(conditions) if conditions else "TRUE"

    con = duckdb.connect()
    if args.no_progress:
        con.execute("SET enable_progress_bar = false")
    else:
        con.execute("SET enable_progress_bar = true")
        con.execute("SET enable_progress_bar_print = true")
        con.execute(f"SET progress_bar_time = {parse_number(args.progress_after, 'progress-after', int)}")

    split_note = f" | split: {args.split}" if spec.splits else ""
    print(f"Dataset: {spec.title}{split_note} | {n_files} file(s)"
          + (" (quick test)" if args.quick_test else ""))
    print(f"Filters: {len(conditions)} active | caption column: {args.caption_col}")

    if args.count:
        print("Counting matches...")
        sys.stdout.flush()
        n = con.execute(f"SELECT count(*) FROM read_parquet(?) WHERE {where_clause}", [file_glob] + params).fetchone()[0]
        print(f"\n{n:,} row(s) match.")
        return

    if unset(args.out):
        out_dir = os.path.join(results_dir, spec.name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = unique_path(os.path.join(out_dir, f"{build_slug(spec, args)}-{start}.csv"))
    else:
        out_path = args.out
        parent = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(parent, exist_ok=True)

    # The chosen caption goes first, right after url, because img2dataset takes
    # its .txt companion from --caption_col. The rest follow for reference.
    if spec.caption_expr:
        caption_expr, caption_params = spec.caption_expr(args.caption_col, args)
    else:
        caption_expr, caption_params = args.caption_col, []
    caption_expr = sanitize_newlines(caption_expr, args.newlines)
    extras = ", ".join(
        f"{sanitize_newlines(oc.expr, args.newlines) if oc.is_text else oc.expr} AS {oc.header}"
        for oc in spec.output_columns
    )
    limit_clause = "" if limit is None else f"LIMIT {limit} "
    query = f"""
        SELECT {URL_EXPR} AS url, {caption_expr} AS text, {extras}
        FROM read_parquet(?)
        WHERE {where_clause}
        ORDER BY {spec.order_by}
        {limit_clause}OFFSET {start}
    """

    print(f"Writing {out_path}")
    # DuckDB writes its bar straight to the C-level stdout, so push our own
    # buffered output out first or these lines come out in the wrong order.
    sys.stdout.flush()

    started = _dt.datetime.now()
    try:
        cursor = con.execute(query, caption_params + [file_glob] + params)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["url", "text"] + [oc.header for oc in spec.output_columns])
            target = "" if limit is None else f" / {limit}"
            found = 0
            while True:
                row = cursor.fetchone()
                if row is None:
                    break
                writer.writerow(row)
                f.flush()
                found += 1
                sys.stdout.write(f"\rFound and saved: {found}{target}")
                sys.stdout.flush()
    except Exception as e:
        print(f"\nSomething went wrong: {e}")
        sys.exit(1)

    sidecar = {
        "dataset": spec.name,
        "hf_id": spec.hf_id,
        "created": started.isoformat(timespec="seconds"),
        "data_dir": data_dir,
        "files": pattern,
        "options": active_filters(spec, args, parser),
        "rows": found,
        "csv": os.path.basename(out_path),
    }
    with open(os.path.splitext(out_path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2, ensure_ascii=False)

    print(f"\n\nDone! Saved {found} row(s) to {out_path}")
    if found == 0:
        print("No rows matched. Loosen a filter, or check that '*' is set where you meant 'any'.")
    else:
        print(f"Next: download.bat \"{out_path}\"")
