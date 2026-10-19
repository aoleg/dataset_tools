"""pastvu: download historic photographs from PastVu by region and years,
cut the PastVu band and scanned borders, paint out watermarks, sort them into
the categories of a classify samples folder, and write each with a one-line
caption, one image at a time with every model resident. The base folder, the
samples folder, the defaults and the region aliases come from pastvu.conf.

    run.bat run --region belarus                      belarus_1917-1991 (years from the conf)
    run.bat run --region ukraine --decade 1970        ukraine_1970s
    run.bat run --region baltics --years 1945 1960    baltics_1945-1960
    run.bat run --scope belarus_1917-1991             resume
    run.bat reclassify belarus_1917-1991 --dry-run    sort again with the current samples
    run.bat captions belarus_1917-1991                rewrite the captions from the records
    run.bat export belarus_1917-1991 --format parquet
    run.bat scopes | aliases | regions --find Minsk | count --region belarus

See README.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import conf as cf  # noqa: E402
import index as ix  # noqa: E402
import layout  # noqa: E402
import sidecars  # noqa: E402
from client import Client, filter_string  # noqa: E402
from store import DB_NAME, Store  # noqa: E402

USSR = [1, 20, 22, 189, 267, 149, 154, 92, 45, 35, 29, 24, 30, 32, 31]   # the order of the countries section
GROUPS = {"ussr": [1, 20, 22, 24, 29, 30, 31, 32, 35, 45, 92, 149, 154, 189, 267], "baltics": [149, 154, 267],
          "caucasus": [35, 45, 92], "central_asia": [24, 29, 30, 31, 32]}
CAPITALS = {"Moscow", "Saint Petersburg", "Sevastopol", "Kiev", "Minsk", "Tashkent", "Astana", "Bishkek", "City of Osh",
            "Ashgabat", "Dushanbe", "Yerevan", "Tbilisi", "Chișinău", "Bălți"}
CITY_MIN_PHOTOS = 1000


# --- setup -------------------------------------------------------------------------------

def open_base(args, conf: cf.Conf) -> tuple[Path, Store]:
    base = Path(args.base).resolve() if getattr(args, "base", None) else conf.need_base()
    base.mkdir(parents=True, exist_ok=True)
    return base, Store(base / DB_NAME)


def memory_store(client: Client) -> Store:
    st = Store(":memory:")
    st.put_regions(client.regions())
    return st


def years_of(args, conf: cf.Conf) -> tuple[tuple[int, int], int | None]:
    if getattr(args, "decade", None) is not None:
        d = args.decade
        if d % 10:
            raise SystemExit(f"--decade {d}: give the first year of a decade, e.g. 1970")
        return (d, d + 9), d
    years = tuple(args.years) if getattr(args, "years", None) else conf.years
    if years[0] > years[1]:
        raise SystemExit("--years: the first year must not be after the second")
    return years, None


def scope_arg(base: Path, store: Store, value: str) -> dict:
    """A scope by name or by its folder."""
    name = Path(value).name if ("/" in value or "\\" in value) else value
    sc = store.scope(name)
    if not sc:
        known = ", ".join(s["name"] for s in store.scopes()) or "none yet"
        raise SystemExit(f"no scope {name!r} in {base}. Scopes: {known}")
    return sc


def scope_from_args(args, conf: cf.Conf, client: Client, store: Store) -> dict:
    ix.ensure_regions(client, store)
    years, decade = years_of(args, conf)
    regions, labels = ix.resolve(store, conf.aliases, args.region)
    if args.name:
        name = layout.check_scope_name(args.name)
    else:
        if len(labels) > 3:
            raise SystemExit("more than three regions: give the scope a --name, or define an alias for them in the conf")
        name = layout.check_scope_name(layout.scope_name(labels, years, decade))
    cids = sorted(r["cid"] for r in regions)
    sc = store.scope(name)
    if sc:
        if sc["regions"] != cids or tuple(sc["years"]) != tuple(years) or sc["decade"] != decade:
            raise SystemExit(f"scope {name} exists with regions {sc['regions']}, years {list(sc['years'])}"
                             f"{', decade ' + str(sc['decade']) if sc['decade'] is not None else ''}; "
                             f"give another --name")
        return sc
    split = args.split if getattr(args, "split", None) is not None else conf.split
    where = ", ".join(ix.region_path(store, r) for r in regions)
    rule = " (by start year)" if decade is not None else ""
    size = f"at most {split:,} images per folder" if split else "no folder split"
    print(f"New scope {name}: {where}, {layout.years_label(years, decade)}{rule}, {size}")
    return store.add_scope(name, cids, years, decade, split)


def do_index(args, client: Client, store: Store, sc: dict) -> dict:
    t0 = time.time()
    last = [0.0]

    def progress(done, total, text):
        if time.time() - last[0] > 3 or done == total:
            last[0] = time.time()
            print(f"  index: page {done:,} of {total:,}  {text}", flush=True)

    print(f"Indexing {sc['name']} ...", flush=True)
    stats = ix.run_index(client, store, sc, getattr(args, "full_parent", False), getattr(args, "reindex", False),
                         getattr(args, "shard_max", None) or ix.SHARD_MAX, progress=progress)
    counts = store.scope_counts(sc["name"])
    if sc["split"] and not sc["width"]:
        store.set_scope_width(sc["name"], layout.chunk_width(counts["total"], sc["split"]))
    print(f"  {stats['pages']:,} pages read, {stats['new']:,} new photos; the scope lists {counts['total']:,} photos "
          f"({time.time() - t0:.0f} s)")
    return store.scope(sc["name"])


# --- subcommands -------------------------------------------------------------------------

def cmd_regions(args, conf, client) -> int:
    st = memory_store(client)
    if args.children is not None:
        parent = st.region(args.children)
        if not parent:
            raise SystemExit(f"no region {args.children}")
        rows = st.children(args.children)
        print(f"{ix.region_path(st, parent)}: {len(rows)} children")
    elif args.find:
        rows = st.find_regions(args.find)
    else:
        rows = sorted([r for r in st.all_regions() if not r["parents"]], key=lambda r: -r["phc"])
    for r in rows:
        print(f"{r['cid']:>6}  {r['phc']:>9,} photos  {ix.region_path(st, r)}  ({r['title_local']})")
    return 0


def cmd_aliases(args, conf, client) -> int:
    if args.generate:
        Path(args.generate).write_text(conf_example(client.regions()), encoding="utf-8")
        print(f"wrote {args.generate}")
        return 0
    if not conf.aliases:
        print(f"no aliases in {conf.path or cf.DEFAULT_PATH}")
        return 0
    st = memory_store(client)
    for group in cf.ALIAS_GROUPS:
        names = [n for n, g in conf.groups.items() if g == group]
        if not names:
            continue
        print(f"[aliases.{group}]  {len(names)}")
        for n in names:
            parts = []
            for c in conf.aliases[n]:
                r = st.region(c)
                parts.append(f"{c} {r['title_en']}" if r else f"{c} (not in PastVu's tree)")
            print(f"  {n:30} {', '.join(parts)}")
    return 0


def cmd_count(args, conf, client) -> int:
    st = memory_store(client)
    regions, _ = ix.resolve(st, conf.aliases, args.region)
    years, decade = years_of(args, conf)
    total = 0
    for r in regions:
        n = client.count(filter_string([r["cid"]], years))
        total += n
        print(f"{r['cid']:>6}  {n:>9,}  {ix.region_path(st, r)}")
    if len(regions) > 1:
        print(f"{'':>6}  {total:>9,}  total (a photo in two of these regions counts twice)")
    if decade is not None:
        print("PastVu counts photos whose dating overlaps the decade; a decade scope keeps those that start in it.")
    return 0


def cmd_scopes(args, conf, client) -> int:
    base, store = open_base(args, conf)
    scopes = store.scopes()
    if not scopes:
        print(f"no scopes in {base}")
    for sc in scopes:
        c = store.scope_counts(sc["name"])
        print(f"{sc['name']:36} {c['total']:>9,} listed  {c['done']:>9,} saved  {c['skipped']:>7,} skipped  "
              f"{c['failed']:>5,} failed  {c['elsewhere']:>7,} in other scopes  {c['pending']:>9,} left")
    return 0


def cmd_index(args, conf, client) -> int:
    base, store = open_base(args, conf)
    sc = scope_from_args(args, conf, client, store)
    do_index(args, client, store, sc)
    return 0


def cmd_run(args, conf, client) -> int:
    import stream as sm
    base, store = open_base(args, conf)
    if args.scope:
        sc = scope_arg(base, store, args.scope)
        if args.reindex:
            ix.ensure_regions(client, store)
            sc = do_index(args, client, store, sc)
        print(f"Resuming {sc['name']}")
    elif args.region:
        sc = scope_from_args(args, conf, client, store)
        sc = do_index(args, client, store, sc)
    else:
        raise SystemExit("give --region (an alias, a region number or a title) or --scope NAME")
    fields = parse_fields(args.txt_fields)
    n = sm.cleanup_interrupted(store, base, sc["name"])
    if n:
        print(f"  {n} photos were being written when the last run stopped; they are done again")
    opts = sm.Options(base=base, scope=sc, min_side=args.min_side, png=args.png, wm_conf=args.wm_conf,
                      wm_paint_conf=args.wm_paint_conf, wm_mode=args.wm_mode, borders=not args.no_borders,
                      dark_borders=not args.no_dark, fetchers=args.fetchers or conf.fetchers, depth=args.depth,
                      writers=args.writers, report_every=args.report_every, limit=args.limit, retries=args.retries,
                      txt_fields=fields, json_files=conf.json_files, json_sidecar=args.json_sidecar,
                      comments=args.comments,
                      display="json" if args.json_progress else "quiet" if args.quiet else "live")

    print("Loading the models ...", flush=True)
    t0 = time.time()
    import numpy as np
    import torch
    detector = lama = None
    if not args.no_watermark:
        import watermark as wm
        detector = wm.detector_lazy()
        lama = wm.lama_lazy()
        print(f"  watermark detector {detector[0]}, LaMa", flush=True)
    classifier = None
    if not args.no_classify:
        import classifier as clf
        samples = Path(args.samples).resolve() if args.samples else conf.samples
        classifier = clf.Classifier(samples, args.patches, args.min_confidence)
        classifier.embed(np.zeros((64, 64, 3), np.uint8))         # load the encoder now
        print(f"  classifier: {len(classifier.class_names)} classes from {samples}", flush=True)
    if torch.cuda.is_available():
        print(f"  loaded in {time.time() - t0:.0f} s; GPU memory in use {torch.cuda.memory_allocated() / 2**30:.1f} GB",
              flush=True)
    import encode
    encode.turbo()

    counts = store.scope_counts(sc["name"])
    cids = store.pending_cids(sc["name"], args.retries)
    completed = counts["total"] - len(cids)
    s = sm.Stream(client, store, opts, classifier, detector, lama)
    res = s.run(cids, counts["total"], completed)
    left = len(store.pending_cids(sc["name"], args.retries))
    print(f"This run: {res['done']:,} saved, {res['skipped']:,} skipped, {res['failed']:,} failed in "
          f"{res['seconds'] / 60:.1f} min. Left to do in {sc['name']}: {left:,}.")
    return 0


def cmd_captions(args, conf, client) -> int:
    base, store = open_base(args, conf)
    sc = scope_arg(base, store, args.scope)
    fields = parse_fields(args.txt_fields)
    n = changed = missing = 0
    for cid, path, category, sha in store.done_rows(sc["name"]):
        rec = store.record(cid)
        img = base / path
        if rec is None or not img.is_file():
            missing += 1
            continue
        text = sidecars.caption(rec.get("record") or {}, fields)
        txt = img.with_suffix(".txt")
        old = txt.read_text(encoding="utf-8") if txt.is_file() else None
        if old != text:
            if not args.dry_run:
                txt.write_text(text, encoding="utf-8")
            changed += 1
            if changed <= args.show:
                print(f"{cid}:\n  - {old}\n  + {text}")
        n += 1
    verb = "would change" if args.dry_run else "changed"
    print(f"{n:,} captions checked, {changed:,} {verb}, {missing:,} photos without a record or image")
    return 0


def cmd_reclassify(args, conf, client) -> int:
    import classifier as clf
    import reclassify as rc
    base, store = open_base(args, conf)
    sc = scope_arg(base, store, args.scope)
    if args.undo:
        res = rc.undo(store, base, sc)
        print(f"Undid reclassify run {res['run']}: {res['back']:,} photos moved back, {res['skipped']:,} left where "
              f"they are, {res['pruned']} empty folders removed")
        return 0
    samples = Path(args.samples).resolve() if args.samples else conf.samples
    classifier = clf.Classifier(samples, args.patches, args.min_confidence)
    print(f"Sorting {sc['name']} again with {len(classifier.class_names)} classes from {samples} ...", flush=True)
    p = rc.plan(store, base, sc, classifier, args.override_manual)
    print(rc.summary(p))
    if args.dry_run:
        report = base / "_reports" / f"{sc['name']}-reclassify-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        report.parent.mkdir(parents=True, exist_ok=True)
        with open(report, "w", encoding="utf-8") as f:
            for m in p["moves"]:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"Dry run: nothing moved. The planned moves are in {report}")
        return 0
    res = rc.apply(store, base, sc, p, str(samples))
    print(f"Reclassify run {res['run']}: {res['moved']:,} photos moved, {res['failed']:,} not moved, "
          f"{res['manual']:,} hand-sorted photos recorded where they are, {res['pruned']} empty folders removed. "
          f"--undo puts the moves back.")
    return 0


def cmd_export(args, conf, client) -> int:
    import export as ex
    base, store = open_base(args, conf)
    sc = scope_arg(base, store, args.scope)
    ext = ".jsonl" if args.format == "jsonl" else ".parquet"
    dst = Path(args.to).resolve() if args.to else base / f"{sc['name']}_records{ext}"
    n = ex.export(store, sc["name"], dst, args.format)
    print(f"{n:,} records written to {dst}")
    return 0


def parse_fields(spec: str | None) -> list[str]:
    fields = [f.strip() for f in spec.split(",") if f.strip()] if spec else list(sidecars.TXT_FIELDS)
    for f in fields:
        if f not in sidecars.ALL_TXT_FIELDS:
            raise SystemExit(f"--txt-fields: unknown field {f!r}; choose from {', '.join(sidecars.ALL_TXT_FIELDS)}")
    return fields


# --- the example conf ---------------------------------------------------------------------

def is_city(r: dict) -> bool:
    title = r["title_en"]
    return title.endswith(" Administration") or layout.slug(title) in {layout.slug(t) for t in CAPITALS}


def conf_example(regions: list[dict]) -> str:
    """pastvu.conf.example with the aliases built from PastVu's region tree:
    the countries of the former Soviet Union and groups of them, their
    first-level regions, and the cities and smaller parts (capitals and
    federal cities, then second-level places with at least CITY_MIN_PHOTOS
    photos)."""
    by = {r["cid"]: r for r in regions}
    taken: dict[str, int] = {g: -1 for g in GROUPS}
    from collections import Counter
    shared = Counter(layout.slug(r["title_en"]) for r in regions
                     if r["parents"] and r["parents"][0] in USSR and len(r["parents"]) <= 2)

    def name_for(r: dict, parent: dict | None = None) -> str:
        n = layout.slug(r["title_en"]) or f"r{r['cid']}"
        if parent is not None and (shared[n] > 1 or (n in taken and taken[n] != r["cid"])):
            n = f"{n}_{layout.slug(parent['title_en'])}"
        while n in taken and taken[n] != r["cid"]:
            n = f"{n}_{r['cid']}"
        taken[n] = r["cid"]
        return n

    def line(n: str, cids: list[int], comment: str) -> str:
        return f"{n:32} = {', '.join(map(str, cids)):20} ; {comment}"

    def desc(r: dict) -> str:
        return f"{r['title_en']} / {r['title_local']}, {r['phc']:,} photos"

    countries = [line(name_for(by[c]), [c], desc(by[c])) for c in USSR]
    countries += [line(g, cids, ", ".join(by[c]["title_en"] for c in cids)) for g, cids in GROUPS.items()]
    region_lines, city_lines = [], []
    for c in USSR:
        first = sorted([r for r in regions if r["parents"] == [c]], key=lambda r: -r["phc"])
        regs = [r for r in first if not is_city(r)]
        cities = [r for r in first if is_city(r)]
        seconds = sorted([r for r in regions if len(r["parents"]) == 2 and r["parents"][0] == c
                          and r["phc"] >= CITY_MIN_PHOTOS], key=lambda r: -r["phc"])
        if regs:
            region_lines.append(f"; {by[c]['title_en']}")
            region_lines += [line(name_for(r), [r["cid"]], desc(r)) for r in regs]
        if cities or seconds:
            city_lines.append(f"; {by[c]['title_en']}")
            city_lines += [line(name_for(r), [r["cid"]], desc(r)) for r in cities]
            city_lines += [line(name_for(r, by.get(r["parents"][-1])), [r["cid"]],
                                f"{desc(r)}, in {by[r['parents'][-1]]['title_en']}") for r in seconds]
    body = ["[aliases.countries]", *countries, "", "[aliases.regions]", *region_lines, "", "[aliases.cities]",
            *city_lines]
    return CONF_HEAD + "\n".join(body) + "\n"


CONF_HEAD = """\
; pastvu configuration. install.bat copies this file to pastvu.conf (next to
; pastvu.py), which is yours to edit. This example is made by
;   run.bat aliases --generate pastvu.conf.example
; Relative paths are resolved against the folder of this file.

[paths]
; The base folder of the downloads. Each scope gets a folder in it
; (belarus_1917-1991, ukraine_1970s, ...), and the store pastvu.db sits in it.
base = D:\\pastvu

; The classify samples folder: one subfolder of example images per category.
samples = D:\\samples

[defaults]
; Years of a run without --years or --decade (PastVu's overlap rule).
years = 1917 1991
; At most this many images per folder (each with its caption, so twice as many
; files); 0 puts every image of a category into one folder.
split = 5000
; Download threads. 8 is faster and doubles the load on PastVu.
fetchers = 4
; Also write each record as a JSON file in <base>\\_records\\<number>\\. The store
; holds every record anyway, and `export` writes them as JSON Lines or Parquet.
json_files = no

; Aliases: a name for one or more PastVu region numbers, used with --region.
; Three groups: countries (and groups of countries), regions (the first level
; under each country), cities and smaller parts (capitals and federal cities,
; then places of the second level with at least 1,000 photos). The names come
; from the English titles; each comment gives the title and the photo count at
; the time this file was made.
;
; The aliases follow PastVu's region tree, which files some territories under
; another country than you may want and marks them as a territorial dispute:
; Crimea (556) and Sevastopol (669) are under Russia, Abkhazia and South Ossetia
; under Georgia, Transnistria under Moldova. "ukraine = 20" does not contain
; Crimea or Sevastopol. Change the aliases to fit your dataset.

"""


# --- arguments ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="pastvu", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", metavar="FILE", help="the configuration file (default pastvu.conf next to pastvu.py)")
    ap.add_argument("--base", metavar="DIR", help="the base folder of the downloads (default: [paths] base of the conf)")
    ap.add_argument("--max-per-second", type=float, default=None, metavar="N",
                    help="cap the requests to PastVu at N per second (default: no cap; back-off on errors is always on)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("regions", help="list PastVu's region tree: top level, --find a name, or --children of a number")
    p.add_argument("--find", metavar="TEXT")
    p.add_argument("--children", type=int, metavar="CID")

    p = sub.add_parser("aliases", help="list the aliases of the conf, or --generate the example conf")
    p.add_argument("--generate", metavar="FILE", help="write an example conf with aliases from PastVu's region tree")

    def when(p):
        g = p.add_mutually_exclusive_group()
        g.add_argument("--years", type=int, nargs=2, metavar=("FROM", "TO"),
                       help="photos whose dating overlaps these years (default: [defaults] years of the conf)")
        g.add_argument("--decade", type=int, metavar="YEAR",
                       help="photos whose dating starts in this decade, e.g. 1970 for 1970-1979")

    def scope_opts(p, required):
        p.add_argument("--region", action="append", required=required, metavar="ALIAS|CID|TITLE",
                       help="an alias of the conf, a PastVu region number or a region title; repeat for several")
        when(p)
        p.add_argument("--name", metavar="NAME", help="the scope's name (default: <aliases>_<years>, e.g. ukraine_1970s)")
        p.add_argument("--split", type=int, metavar="N", help="at most N images per folder, for a new scope "
                                                              "(default: [defaults] split of the conf; 0: no split)")
        p.add_argument("--full-parent", action="store_true",
                       help="page a large region itself instead of through its children (slower for the server)")
        p.add_argument("--reindex", action="store_true", help="page the scope again (finds photos published since)")
        p.add_argument("--shard-max", type=int, help=argparse.SUPPRESS)

    p = sub.add_parser("count", help="the number of photos PastVu lists for regions and years")
    p.add_argument("--region", action="append", required=True, metavar="ALIAS|CID|TITLE")
    when(p)

    sub.add_parser("scopes", help="list the scopes of the base folder with their progress")

    p = sub.add_parser("index", help="list the photos of a scope into the store, no downloads")
    scope_opts(p, True)

    p = sub.add_parser("run", help="index (with --region) and download, process, sort and write a scope")
    scope_opts(p, False)
    p.add_argument("--scope", metavar="NAME", help="resume a scope by its name or folder")
    p.add_argument("--samples", metavar="DIR", help="classify samples folder (default: [paths] samples of the conf)")
    p.add_argument("--no-classify", action="store_true", help="no categories: images go into numbered folders of the scope")
    p.add_argument("--min-confidence", type=float, default=0.7, metavar="X", help="below this an image goes to _unsure")
    p.add_argument("--patches", type=int, default=576, metavar="N", help="encoder patch budget (default 576)")
    p.add_argument("--min-side", type=int, default=512, metavar="PX",
                   help="skip images whose picture is smaller than this on either side (default 512)")
    p.add_argument("--png", action="store_true", help="write images with painted watermarks as PNG instead of JPEG q97")
    p.add_argument("--no-watermark", action="store_true", help="no watermark detection or painting")
    p.add_argument("--wm-conf", type=float, default=0.1, metavar="X",
                   help="watermark detections from this confidence go into the record (default 0.1)")
    p.add_argument("--wm-paint-conf", type=float, default=0.4, metavar="X",
                   help="only detections from this confidence are painted out (default 0.4)")
    p.add_argument("--wm-mode", choices=["inpaint", "trim"], default="inpaint",
                   help="paint watermarks out (default) or cut the largest clear rectangle when it keeps half the image")
    p.add_argument("--no-borders", action="store_true", help="no border analysis (the PastVu band is still cut)")
    p.add_argument("--no-dark", action="store_true", help="leave dark borders alone")
    p.add_argument("--txt-fields", metavar="LIST",
                   help=f"caption parts in order, from {','.join(sidecars.ALL_TXT_FIELDS)} "
                        f"(default {','.join(sidecars.TXT_FIELDS)})")
    p.add_argument("--json-sidecar", action="store_true", help="also write the record next to the image")
    p.add_argument("--comments", action="store_true", help="add the comment tree to the record (one more request)")
    p.add_argument("--fetchers", type=int, metavar="N", help="download threads (default: [defaults] fetchers of the conf)")
    p.add_argument("--depth", type=int, default=16, metavar="N", help="images fetched ahead of the GPU (default 16)")
    p.add_argument("--writers", type=int, default=2, metavar="N", help="encoder and writer threads (default 2)")
    p.add_argument("--limit", type=int, metavar="N", help="do at most N photos in this run")
    p.add_argument("--retries", type=int, default=3, metavar="N", help="attempts for a photo that failed (default 3)")
    p.add_argument("--report-every", type=int, default=500, metavar="N", help="a report block every N photos")
    p.add_argument("--quiet", action="store_true", help="report blocks only, no live display")
    p.add_argument("--json-progress", action="store_true", help="one JSON line per photo on stdout")

    p = sub.add_parser("captions", help="rewrite the captions of a scope from the records, no downloads")
    p.add_argument("scope", metavar="SCOPE", help="the scope's name or folder")
    p.add_argument("--txt-fields", metavar="LIST", help="caption parts in order (see run --help)")
    p.add_argument("--dry-run", action="store_true", help="count and show the changes, write nothing")
    p.add_argument("--show", type=int, default=5, metavar="N", help="print the first N changes (default 5)")

    p = sub.add_parser("reclassify", help="sort the saved photos of a scope again with the current samples folder")
    p.add_argument("scope", metavar="SCOPE", help="the scope's name or folder")
    p.add_argument("--samples", metavar="DIR", help="classify samples folder (default: [paths] samples of the conf)")
    p.add_argument("--min-confidence", type=float, default=0.7, metavar="X", help="below this an image goes to _unsure")
    p.add_argument("--patches", type=int, default=576, metavar="N", help="encoder patch budget (default 576)")
    p.add_argument("--dry-run", action="store_true", help="report the moves and write them to <base>/_reports, move nothing")
    p.add_argument("--override-manual", action="store_true",
                   help="also sort the photos that were moved to another category by hand (default: they stay)")
    p.add_argument("--undo", action="store_true", help="put back the moves of the last reclassify of the scope")

    p = sub.add_parser("export", help="write the records of a scope as JSON Lines or Parquet")
    p.add_argument("scope", metavar="SCOPE", help="the scope's name or folder")
    p.add_argument("--format", choices=["jsonl", "parquet"], default="jsonl")
    p.add_argument("--to", metavar="FILE", help="the output file (default <base>/<scope>_records.<format>)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    os.environ.setdefault("YOLO_OFFLINE", "1")
    conf = cf.load(Path(args.conf) if args.conf else None)
    fetchers = getattr(args, "fetchers", None) or conf.fetchers
    client = Client(max_per_second=args.max_per_second, connections=max(8, fetchers * 2))
    try:
        cmds = {"regions": cmd_regions, "aliases": cmd_aliases, "count": cmd_count, "scopes": cmd_scopes,
                "index": cmd_index, "run": cmd_run, "captions": cmd_captions, "reclassify": cmd_reclassify,
                "export": cmd_export}
        return cmds[args.cmd](args, conf, client)
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
