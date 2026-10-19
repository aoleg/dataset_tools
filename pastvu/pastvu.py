"""pastvu: download historic photographs from PastVu by region and years,
cut the PastVu band and scanned borders, paint out watermarks, sort them into
the categories of a classify samples folder, and write each with a one-line
caption and its full record, one image at a time with every model resident.

    run.bat regions [--find TEXT] [--children CID]
    run.bat count --region Belarus [--years 1917 1991]
    run.bat index --out D:/pastvu/belarus --region Belarus [--years 1917 1991]
    run.bat run --out D:/pastvu/belarus --region Belarus --samples D:/samples
    run.bat run --out D:/pastvu/belarus              (resume: the scope is in the store)
    run.bat captions --out D:/pastvu/belarus         (rewrite the captions from the records)

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

import index as ix  # noqa: E402
import sidecars  # noqa: E402
from client import Client, filter_string  # noqa: E402
from store import DB_NAME, Store  # noqa: E402

MARK = "_dataset_tools.json"


def meta_dir(out: Path) -> Path:
    return out.with_name(out.name + "_meta")


def open_store(out: Path) -> Store:
    meta = meta_dir(out)
    meta.mkdir(parents=True, exist_ok=True)
    mark = meta / MARK
    if not mark.is_file():
        mark.write_text(json.dumps({"made_by": "dataset_tools", "tool": "pastvu", "dataset": out.name}) + "\n",
                        encoding="utf-8")
    return Store(meta / DB_NAME)


def memory_store(client: Client) -> Store:
    st = Store(Path(":memory:"))
    st.put_regions(client.regions())
    return st


def years_of(args) -> tuple[int, int]:
    a, b = args.years
    if a > b:
        raise SystemExit("--years: the first year must not be after the second")
    return (a, b)


# --- subcommands ------------------------------------------------------------------------

def cmd_regions(args, client: Client) -> int:
    st = memory_store(client)
    if args.children is not None:
        rows = st.children(args.children)
        parent = st.region(args.children)
        if not parent:
            raise SystemExit(f"no region {args.children}")
        print(f"{ix.region_path(st, parent)}: {len(rows)} children")
    elif args.find:
        rows = st.find_regions(args.find)
    else:
        with st.lock:
            cids = [r[0] for r in st.db.execute("SELECT cid FROM regions WHERE parent IS NULL ORDER BY phc DESC")]
        rows = [st.region(c) for c in cids]
    for r in rows:
        print(f"{r['cid']:>6}  {r['phc']:>9,} photos  {ix.region_path(st, r)}  ({r['title_local']})")
    return 0


def cmd_count(args, client: Client) -> int:
    st = memory_store(client)
    regions = ix.resolve_regions(st, args.region)
    years = years_of(args)
    total = 0
    for r in regions:
        n = client.count(filter_string([r["cid"]], years))
        total += n
        print(f"{r['cid']:>6}  {n:>9,}  {ix.region_path(st, r)}")
    if len(regions) > 1:
        print(f"{'':>6}  {total:>9,}  total (a photo in two of these regions counts twice)")
    return 0


def index_scope(args, client: Client, store: Store) -> list[dict]:
    ix.ensure_regions(client, store)
    years = years_of(args)
    regions = ix.resolve_regions(store, args.region)
    scope = store.get_meta("scope", {"regions": [], "years": None})
    if scope["years"] and tuple(scope["years"]) != years:
        raise SystemExit(f"this output folder holds the years {scope['years'][0]}-{scope['years'][1]}; "
                         f"use another --out for {years[0]}-{years[1]}")
    t0 = time.time()
    last = [0.0]

    def progress(done, total, text):
        if time.time() - last[0] > 2 or done == total:
            last[0] = time.time()
            print(f"  index: page {done:,} of {total:,}  {text}", flush=True)

    print(f"Indexing {', '.join(ix.region_path(store, r) for r in regions)}, {years[0]}-{years[1]} ...")
    stats = ix.run_index(client, store, regions, years, args.full_parent, args.reindex, progress=progress)
    cids = sorted(set(scope["regions"]) | {r["cid"] for r in regions})
    store.set_meta("scope", {"regions": cids, "years": list(years)})
    c = store.counts()
    print(f"  {stats['pages']:,} pages read, {stats['new']:,} new photos, {c['total']:,} in the index "
          f"({time.time() - t0:.0f} s)")
    return regions


def cmd_index(args, client: Client) -> int:
    store = open_store(Path(args.out).resolve())
    index_scope(args, client, store)
    return 0


def cmd_captions(args, client: Client) -> int:
    """Rewrite the caption of every saved photo from its record, with the
    current caption rules and --txt-fields; no downloads."""
    out = Path(args.out).resolve()
    meta = meta_dir(out)
    store = open_store(out)
    fields = parse_fields(args.txt_fields)
    n = changed = missing = 0
    for cid, path, *_ in store.done_rows():
        rec = meta / f"{cid}.json"
        img = out / path
        if not rec.is_file() or not img.is_file():
            missing += 1
            continue
        d = json.loads(rec.read_text(encoding="utf-8"))
        text = sidecars.caption(d["record"] or {}, fields)
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


def parse_fields(spec: str | None) -> list[str]:
    fields = [f.strip() for f in spec.split(",") if f.strip()] if spec else list(sidecars.TXT_FIELDS)
    for f in fields:
        if f not in sidecars.ALL_TXT_FIELDS:
            raise SystemExit(f"--txt-fields: unknown field {f!r}; choose from {', '.join(sidecars.ALL_TXT_FIELDS)}")
    return fields


def cmd_run(args, client: Client) -> int:
    import stream as sm
    out = Path(args.out).resolve()
    store = open_store(out)
    scope = store.get_meta("scope")
    if args.region:
        index_scope(args, client, store)
    elif not scope:
        raise SystemExit("this output folder has no scope yet: give --region (and --years)")
    else:
        print(f"Resuming the scope in {meta_dir(out)}: regions {scope['regions']}, years "
              f"{scope['years'][0]}-{scope['years'][1]}")
    fields = parse_fields(args.txt_fields)
    opts = sm.Options(out=out, meta=meta_dir(out), min_side=args.min_side, png=args.png, wm_conf=args.wm_conf,
                      wm_paint_conf=args.wm_paint_conf, wm_mode=args.wm_mode, borders=not args.no_borders, dark_borders=not args.no_dark,
                      fetchers=args.fetchers, depth=args.depth, writers=args.writers, report_every=args.report_every,
                      limit=args.limit, retries=args.retries, txt_fields=fields, json_sidecar=args.json_sidecar,
                      comments=args.comments,
                      display="json" if args.json_progress else "quiet" if args.quiet else "live")
    out.mkdir(parents=True, exist_ok=True)

    print("Loading the models ...", flush=True)
    t0 = time.time()
    import torch
    detector = lama = None
    if not args.no_watermark:
        import watermark as wm
        detector = wm.detector_lazy()
        lama = wm.lama_lazy()
        print(f"  watermark detector {detector[0]}, LaMa", flush=True)
    classifier = out_index = None
    if not args.no_classify:
        if not args.samples:
            raise SystemExit("give --samples <folder> (a classify samples folder), or --no-classify")
        import classifier as clf
        classifier = clf.Classifier(Path(args.samples).resolve(), args.patches, args.min_confidence)
        out_index = clf.OutputIndex(out, classifier.encoder.model_id, args.patches)
        print(f"  classifier: {len(classifier.class_names)} classes from {args.samples}", flush=True)
    if torch.cuda.is_available():
        print(f"  loaded in {time.time() - t0:.0f} s; GPU memory in use {torch.cuda.memory_allocated() / 2**30:.1f} GB",
              flush=True)
    import encode
    encode.turbo()

    counts = store.counts()
    cids = store.pending_cids(retry_failed=args.retries)
    completed = counts["total"] - len(cids)
    s = sm.Stream(client, store, opts, classifier, out_index, detector, lama)
    res = s.run(cids, counts["total"], completed)
    left = len(store.pending_cids(retry_failed=args.retries))
    print(f"This run: {res['done']:,} saved, {res['skipped']:,} skipped, {res['failed']:,} failed in "
          f"{res['seconds'] / 60:.1f} min. Left to do: {left:,}.")
    return 0


# --- arguments --------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="pastvu", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-per-second", type=float, default=None, metavar="N",
                    help="cap the requests to PastVu at N per second (default: no cap; back-off on errors is always on)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("regions", help="list the region tree: top level, --find a name, or --children of a cid")
    p.add_argument("--find", metavar="TEXT")
    p.add_argument("--children", type=int, metavar="CID")

    def scope(p, required):
        p.add_argument("--region", action="append", required=required, metavar="CID|NAME",
                       help="a PastVu region by cid or name; repeat for several")
        p.add_argument("--years", type=int, nargs=2, default=[1917, 1991], metavar=("FROM", "TO"),
                       help="photos whose dating overlaps these years (default 1917 1991)")

    p = sub.add_parser("count", help="the number of photos PastVu lists for a scope")
    scope(p, True)

    for name, helptext in (("index", "list the photos of a scope into the store, no downloads"),
                           ("run", "index (when --region is given) and download, process, sort and write")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--out", required=True, metavar="DIR", help="the dataset folder; the store goes to DIR_meta")
        scope(p, False)
        p.add_argument("--full-parent", action="store_true",
                       help="page a large region itself instead of through its children (slower for the server)")
        p.add_argument("--reindex", action="store_true", help="page every shard again (finds photos published since)")

    p = sub.add_parser("captions", help="rewrite the captions of the saved photos from their records, no downloads")
    p.add_argument("--out", required=True, metavar="DIR", help="the dataset folder of an earlier run")
    p.add_argument("--txt-fields", metavar="LIST", help="caption parts in order (see run --help)")
    p.add_argument("--dry-run", action="store_true", help="count and show the changes, write nothing")
    p.add_argument("--show", type=int, default=5, metavar="N", help="print the first N changes (default 5)")

    p = sub.choices["run"]
    p.add_argument("--samples", metavar="DIR", help="classify samples folder: one subfolder of examples per category")
    p.add_argument("--no-classify", action="store_true", help="no categories: every image goes into the dataset folder")
    p.add_argument("--min-confidence", type=float, default=0.7, metavar="X", help="below this an image goes to _unsure")
    p.add_argument("--patches", type=int, default=576, metavar="N", help="encoder patch budget (default 576)")
    p.add_argument("--min-side", type=int, default=512, metavar="PX",
                   help="skip images whose picture is smaller than this on either side (default 512)")
    p.add_argument("--png", action="store_true", help="write images with painted watermarks as PNG instead of JPEG q97")
    p.add_argument("--no-watermark", action="store_true", help="no watermark detection or painting")
    p.add_argument("--wm-conf", type=float, default=0.1, metavar="X",
                   help="watermark detections from this confidence go into the record (default 0.1)")
    p.add_argument("--wm-paint-conf", type=float, default=0.4, metavar="X",
                   help="only detections from this confidence are painted out (default 0.4); lower ones were often "
                        "scene text and objects in the pilot")
    p.add_argument("--wm-mode", choices=["inpaint", "trim"], default="inpaint",
                   help="paint watermarks out (default) or cut the largest clear rectangle when it keeps half the image")
    p.add_argument("--no-borders", action="store_true", help="no border analysis (the PastVu band is still cut)")
    p.add_argument("--no-dark", action="store_true", help="leave dark borders alone")
    p.add_argument("--txt-fields", metavar="LIST",
                   help=f"caption parts in order, from {','.join(sidecars.ALL_TXT_FIELDS)} "
                        f"(default {','.join(sidecars.TXT_FIELDS)})")
    p.add_argument("--json-sidecar", action="store_true", help="also write the record next to the image")
    p.add_argument("--comments", action="store_true", help="add the comment tree to the record (one more request)")
    p.add_argument("--fetchers", type=int, default=4, metavar="N", help="download threads (default 4)")
    p.add_argument("--depth", type=int, default=16, metavar="N", help="images fetched ahead of the GPU (default 16)")
    p.add_argument("--writers", type=int, default=2, metavar="N", help="encoder and writer threads (default 2)")
    p.add_argument("--limit", type=int, metavar="N", help="do at most N photos in this run")
    p.add_argument("--retries", type=int, default=3, metavar="N", help="attempts for a photo that failed (default 3)")
    p.add_argument("--report-every", type=int, default=500, metavar="N", help="a report block every N photos")
    p.add_argument("--quiet", action="store_true", help="report blocks only, no live display")
    p.add_argument("--json-progress", action="store_true", help="one JSON line per photo on stdout")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    os.environ.setdefault("YOLO_OFFLINE", "1")
    client = Client(max_per_second=args.max_per_second, connections=max(8, getattr(args, "fetchers", 4) * 2))
    try:
        return {"regions": cmd_regions, "count": cmd_count, "index": cmd_index, "run": cmd_run,
                "captions": cmd_captions}[args.cmd](args, client)
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
