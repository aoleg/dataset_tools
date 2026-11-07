"""pastvu: download historic photographs from PastVu by region and years,
cut the PastVu band and scanned borders, paint out watermarks, sort them into
the categories of a classify samples folder, and write each with a one-line
caption, one image at a time with every model resident. The base folder, the
samples folder, the defaults and the region aliases come from pastvu.conf.

    run.bat run --region belarus                      belarus_1917-1991 (years from the conf)
    run.bat run --region ukraine --decade 1970        ukraine_1970s
    run.bat run --region baltics --years 1945 1960    baltics_1945-1960
    run.bat run --scope belarus_1917-1991             resume
    run.bat run --region estonia --by decade          estonia_1910s ... estonia_1990s in one download
    run.bat run --region estonia --regroup            pull saved Estonia photos from other scopes, then download
    run.bat run --region belarus --raw                download only, into <base>/_storage, no model and no GPU
    run.bat run --region belarus --pull               download and process; the storage fills the GPU's idle time
    run.bat pull                                      process the storage, no downloads
    run.bat regroup --by decade estonia_1917-1991     split a scope into decades, no downloads (--undo)
    run.bat drop estonia_1950s --yes                  delete a scope's files and forget it
    run.bat reclassify belarus_1917-1991 --dry-run    sort again with the current samples
    run.bat reclassify                                every scope of the base (asks first)
    run.bat captions belarus_1917-1991                rewrite the captions from the records
    run.bat autocaption belarus_1917-1991             caption with the VLM of [caption] in the conf
    run.bat autocaption <base>/belarus_1917-1991/5_transport --limit 100
    run.bat export "estonia_*" --format parquet       scope names take * and ? patterns
    run.bat scopes | aliases | regions --find Minsk | count --region belarus

See README.md.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import conf as cf  # noqa: E402
import index as ix  # noqa: E402
import layout  # noqa: E402
import sidecars  # noqa: E402
from client import IMAGE_DEADLINE, READ_TIMEOUT, Client, filter_string  # noqa: E402
from store import DB_NAME, Store  # noqa: E402

USSR = [1, 20, 22, 189, 267, 149, 154, 92, 45, 35, 29, 24, 30, 32, 31]   # the order of the countries section
GROUPS = {"ussr": [1, 20, 22, 24, 29, 30, 31, 32, 35, 45, 92, 149, 154, 189, 267],
          "ussr_without_russia": [20, 22, 24, 29, 30, 31, 32, 35, 45, 92, 149, 154, 189, 267],
          "baltics": [149, 154, 267],
          "caucasus": [35, 45, 92], "central_asia": [24, 29, 30, 31, 32]}
CAPITALS = {"Moscow", "Saint Petersburg", "Sevastopol", "Kiev", "Minsk", "Tashkent", "Astana", "Bishkek", "City of Osh",
            "Ashgabat", "Dushanbe", "Yerevan", "Tbilisi", "Chișinău", "Bălți"}
CITY_MIN_PHOTOS = 1000
CRIMEA = [556, 669]          # Crimea and Sevastopol: under Russia in PastVu's tree, listed apart in the conf
UKRAINE_PRESETS = [          # name, regions, comment
    ("ukraine", [20, 556, 669], "the modern state of Ukraine in its internationally recognized borders: "
                                "Ukraine (20), Crimea (556), Sevastopol (669)"),
    ("ukrainian_ssr", [20, 556, 669], "the Ukrainian SSR from 1954 to 1991 (see the note above)"),
    ("ukraine_pre_1954", [20], "the Ukrainian SSR before 1954, without Crimea and Sevastopol (see the note above)"),
]


# --- setup -------------------------------------------------------------------------------

IMPORT_COMMANDS = ("reclassify", "cleanup", "cherrypick", "autocaption", "scopes", "reprocess", "niqe", "detect",
                   "deduplicate")
IMPORT_TOPS = ("color", "category", "none")
# The options of the processing: a raw run (run --raw) refuses them, since it processes nothing.
RAW_REFUSES = ("samples", "no_classify", "min_confidence", "priority_margin", "patches", "png", "no_watermark",
               "wm_conf", "wm_paint_conf", "wm_mode", "no_borders", "no_dark", "txt_fields", "json_sidecar", "depth",
               "writers", "keep_rejects", "storage_threads", "max_pixels", "prepare_workers")


def prepare_workers_of(args) -> int:
    """--prepare-workers, or prepare.default_workers() when not given."""
    n = getattr(args, "prepare_workers", None)
    if n is None:
        import prepare as pp
        return pp.default_workers()
    return max(0, n)


def no_open_review(store: Store, base: Path, command: str) -> None:
    n = store.review_count()
    if n:
        import cleanup as cu
        raise SystemExit(f"{n:,} photos of {base} are in the review folder {cu.review_dir(store, base)}; settle the "
                         f"review with cleanup --done or cleanup --undo before {command}")


def open_base(args, conf: cf.Conf) -> tuple[Path, Store]:
    base = Path(args.base).resolve() if getattr(args, "base", None) else conf.need_base()
    base.mkdir(parents=True, exist_ok=True)
    store = Store(base / DB_NAME)
    if store.is_import:
        if args.cmd not in IMPORT_COMMANDS:
            raise SystemExit(f"{base} is the import of {store.get_meta('source')}; {args.cmd} works on PastVu base "
                             f"folders only (an import takes {', '.join(IMPORT_COMMANDS)} and import)")
        return base, store
    base_layout(conf, store, base)
    return base, store


def base_layout(conf: cf.Conf, store: Store, base: Path) -> str:
    """The folder tree of the base. A base with scopes keeps the tree it was
    made with (flat for bases from before the option); an empty one takes the
    conf's. A conf without the setting gets it written in."""
    if store.scopes():
        used = store.layout()
        if store.get_meta("layout") is None:
            store.set_meta("layout", used)
    else:
        used = conf.layout or layout.DEFAULT_LAYOUT
        store.set_meta("layout", used)
    if conf.layout is None:
        if conf.path:
            cf.save_layout(conf.path, used)
            print(f"{conf.path.name}: layout = {used} written into [defaults] (the folder tree of {base})")
        conf.layout = used
    elif conf.layout != used:
        print(f"{base} keeps its folder tree {used}; layout = {conf.layout} in the conf applies to new base folders")
    # black-and-white and colour at the top of the tree: the same rules
    if store.scopes():
        split = bool(store.get_meta("bw_color", False))
        if store.get_meta("bw_color") is None:
            store.set_meta("bw_color", split)
    else:
        split = True if conf.bw_color is None else conf.bw_color
        store.set_meta("bw_color", split)
    if conf.bw_color is None:
        if conf.path:
            cf.save_default(conf.path, "bw_color", "yes" if split else "no", cf.BW_COLOR_COMMENT)
            print(f"{conf.path.name}: bw_color = {'yes' if split else 'no'} written into [defaults]")
        conf.bw_color = split
    elif conf.bw_color != split:
        print(f"{base} keeps bw_color = {'yes' if split else 'no'}; the conf's setting applies to new base folders")
    return used


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


def is_folder_arg(base: Path, store: Store, value: str) -> bool:
    return "/" in value or "\\" in value or (not store.scope(value) and (base / value).is_dir())


def scope_arg(base: Path, store: Store, value: str) -> dict:
    """A scope by name or by its folder."""
    if is_folder_arg(base, store, value):
        hits = target_arg(base, store, value)
        if len(hits) != 1:
            raise SystemExit(f"{value} holds {len(hits)} scopes: {', '.join(sc['name'] for sc, _ in hits)}; give one")
        return hits[0][0]
    sc = store.scope(value)
    if not sc:
        known = ", ".join(s["name"] for s in store.scopes()) or "none yet"
        raise SystemExit(f"no scope {value!r} in {base}. Scopes: {known}")
    return sc


def is_pattern(value: str) -> bool:
    return any(c in value for c in "*?[")


def scopes_arg(base: Path, store: Store, values: list[str]) -> list[dict]:
    """Scopes by name, folder or name pattern (estonia_*), each once, in the order given."""
    out, seen = [], set()
    for v in values:
        if is_pattern(v):
            hits = [sc for sc in store.scopes() if fnmatch.fnmatchcase(sc["name"], v)]
            if not hits:
                raise SystemExit(f"no scope of {base} matches {v!r}")
        elif is_folder_arg(base, store, v):
            hits = [sc for sc, _ in target_arg(base, store, v)]
        else:
            hits = [scope_arg(base, store, v)]
        for sc in hits:
            if sc["name"] not in seen:
                seen.add(sc["name"])
                out.append(sc)
    return out


def confirm(question: str, yes: bool) -> bool:
    if yes:
        return True
    try:
        answer = input(question)
    except EOFError:
        print("\nno answer; add --yes to go ahead without asking")
        return False
    return answer.strip().lower() in ("y", "yes")


@contextmanager
def locked(store: Store, command: str, mode: str = "exclusive"):
    store.acquire_lock(command, mode)
    try:
        yield
    finally:
        store.release_lock()


def target_arg(base: Path, store: Store, value: str) -> list[tuple[dict, str]]:
    """A scope name, or a folder: a scope's folder, one inside it, or one that
    holds scope folders (the 1970s folder of a years/place base). A relative
    folder is taken from the current folder, else from the base.
    -> [(scope, path prefix relative to the base, ending in /, or "" for the whole scope)]."""
    if "/" not in value and "\\" not in value and store.scope(value):
        return [(store.scope(value), "")]
    p = Path(value)
    if not p.is_absolute():
        p = Path.cwd() / p if (Path.cwd() / p).exists() or not (base / p).exists() else base / p
    p = p.resolve()
    try:
        rel = p.relative_to(base.resolve())
    except ValueError:
        raise SystemExit(f"{p} is not inside the base folder {base}")
    if not rel.parts:
        raise SystemExit("give a scope or a folder inside the base folder, not the base folder")
    if not p.is_dir():
        raise SystemExit(f"{p} is not a folder")
    prefix = rel.as_posix().rstrip("/") + "/"
    hits = []
    for sc in store.scopes():
        for _, root in layout.scope_roots(sc):
            f = root + "/"
            if prefix.startswith(f):
                return [(sc, prefix)]
            if f.startswith(prefix):
                hits.append((sc, f))
    if not hits:
        raise SystemExit(f"{p} is not the folder of a scope, inside one, or holding any")
    return hits


def vlm_client(conf: cf.Conf):
    import autocaption as ac
    client = ac.VLMClient(conf.caption)
    try:
        served = client.check()
    except ac.EndpointDown as e:
        raise SystemExit(str(e))
    print(f"Caption endpoint {client.base}: {served}; prompts {conf.caption.prompt.name} and "
          f"{conf.caption.system_prompt.name}", flush=True)
    return client


def scope_from_args(args, conf: cf.Conf, client: Client, store: Store, create: bool = True) -> dict:
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
    if not create:
        raise SystemExit(f"--resume: there is no scope {name} in this base yet; run the command once without --resume")
    split = args.split if getattr(args, "split", None) is not None else conf.split
    where = ", ".join(ix.region_path(store, r) for r in regions)
    rule = " (by start year)" if decade is not None else ""
    size = f"at most {split:,} images per folder" if split else "no folder split"
    sc = store.add_scope(name, cids, years, decade, split, label=args.name or "+".join(labels))
    print(f"New scope {name} in {sc['folder']}: {where}, {layout.years_label(years, decade)}{rule}, {size}")
    return sc


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


def skip_pairs(store: Store, names: list[str], pairs: list[tuple[int, str]], skip: int) -> list[tuple[int, str]]:
    """run --skip N: leave out the first N photos the scopes list, in PastVu
    number order (the order of a run), whether done or not. They are not
    downloaded and stay to do for a run without --skip."""
    if not skip:
        return pairs
    cids = store.member_cids(names)
    if skip >= len(cids):
        return []
    first = cids[skip]
    return [p for p in pairs if p[0] >= first]


def run_rule(args, store: Store):
    """The rule a --region run splits by, or None for one scope. In a
    years/place or place/years base every scope is one decade, so a run over a
    span of years splits by decade, and --by country means country,decade."""
    import partition as pt
    rule = pt.parse_rule(args.by) if getattr(args, "by", None) else None
    if store.layout() != "flat" and getattr(args, "decade", None) is None:
        rule = pt.Rule(True, rule.place if rule else None)
    return rule


def family_from_args(args, conf: cf.Conf, client: Client, store: Store, rule,
                     index: bool = True) -> tuple[dict, dict]:
    """run --by: index the regions and years of the arguments once and put each
    photo into the scope its rule names, made on first sight. index False
    (run --resume): no index, the scopes the family has listed so far.
    -> (the family as a pseudo scope, {name: scope} of every scope it fills)."""
    import partition as pt
    ix.ensure_regions(client, store)
    years, decade = years_of(args, conf)
    regions, labels = ix.resolve(store, conf.aliases, args.region)
    if args.name:
        label = layout.check_scope_name(args.name)
    elif len(labels) > 3:
        raise SystemExit("more than three regions: give the family a --name, or define an alias for them in the conf")
    else:
        label = "+".join(labels)
    cids = sorted(r["cid"] for r in regions)
    split = args.split if getattr(args, "split", None) is not None else conf.split
    part = pt.Partitioner(store, conf.aliases, conf.groups, rule, label, cids)
    inside = ix.member_test(years, decade)
    if rule.decade and decade is None:
        # a decade split keeps the decades the years cover: 1917-1991 gives the 1910s to the 1990s; a photo
        # dated 1890-1920 is listed by PastVu's overlap filter but starts in the 1890s, so it joins no scope
        lo, hi = years[0] // 10 * 10, years[1] // 10 * 10 + 9
        inside = lambda p: p.get("year") is not None and lo <= int(p["year"]) <= hi  # noqa: E731

        def of_this_run(sc):
            return sc["decade"] is not None and lo <= sc["decade"] <= hi
    else:
        def of_this_run(sc):
            target = (decade, decade + 9) if rule.decade else years
            return tuple(sc["years"]) == tuple(target) and sc["decade"] == decade
    family = f"{label}@{rule.text}"
    made: dict[str, dict] = {}
    cache: dict[tuple, str] = {}

    def place(t):
        lab, regs = (f"{label}_unplaced", cids) if t.label == "unplaced" else (t.label, list(t.regions))
        if t.decade is not None:
            y, d = (t.decade, t.decade + 9), t.decade
        else:
            y, d = years, decade
        return layout.check_scope_name(layout.scope_name([lab], y, d)), regs, y, d, lab

    def assign(photo):
        if not inside(photo):
            return None
        t = part.target(photo)
        key = (t.label, t.regions, t.decade)
        name = cache.get(key)
        if name is None:
            name, regs, y, d, lab = place(t)
            if name not in made:
                is_new = store.scope(name) is None
                made[name] = store.ensure_scope(name, regs, y, d, split, lab, 0)
                store.add_family(family, [name])           # at once: an index cut short keeps its scopes
                if is_new:
                    print(f"  new scope {name}", flush=True)
            cache[key] = name
        return name

    fam = {"name": family, "regions": cids, "years": years, "decade": decade, "split": split, "width": 0}
    where = ", ".join(ix.region_path(store, r) for r in regions)
    if not index:
        scopes = {n: store.scope(n) for n in store.family(family)}
        scopes = {n: sc for n, sc in scopes.items() if sc and of_this_run(sc)}
        if not scopes:
            raise SystemExit(f"--resume: nothing of {where}, {layout.years_label(years, decade)} by {rule.text} is "
                             f"indexed in this base yet; run the command once without --resume")
        p = ix.progress_of(store, family)
        print(f"Resuming {where}, {layout.years_label(years, decade)} by {rule.text}: {len(scopes)} scopes, no "
              f"indexing. Index: {ix.describe_progress(p)}.", flush=True)
        if p["partial"] or p.get("not_started"):
            print("  The index is not finished: this run downloads what is listed; run without --resume to list "
                  "the rest.", flush=True)
        elif p["planned"] is None:
            print("  An older version made this index without recording its plan, so it may lack shards that were "
                  "never started; a run without --resume finishes it.", flush=True)
        return fam, scopes
    print(f"Indexing {where}, {layout.years_label(years, decade)}, into scopes by {rule.text} ...", flush=True)
    t0 = time.time()
    last = [0.0]

    def progress(done, total, text):
        if time.time() - last[0] > 3 or done == total:
            last[0] = time.time()
            print(f"  index: page {done:,} of {total:,}  {text}", flush=True)

    stats = ix.run_index(client, store, fam, getattr(args, "full_parent", False), getattr(args, "reindex", False),
                         getattr(args, "shard_max", None) or ix.SHARD_MAX, progress=progress, assign=assign)
    store.add_family(family, made)
    scopes = {}
    for name in store.family(family):
        sc = store.scope(name)
        if not sc or not of_this_run(sc):          # the family also lists scopes of runs with other years
            continue
        if sc["split"] and not sc["width"]:
            store.set_scope_width(name, layout.chunk_width(store.scope_counts(name)["total"], sc["split"]))
            sc = store.scope(name)
        scopes[name] = sc
    total = sum(store.scope_counts(n)["total"] for n in scopes)
    print(f"  {stats['pages']:,} pages read, {stats['new']:,} new photos; {len(scopes)} scopes list {total:,} photos "
          f"({time.time() - t0:.0f} s)")
    return fam, scopes


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
        k = store.counters(sc["name"])
        where = f"  {sc['folder']}" if sc["folder"] != sc["name"] else ""
        modes = f"  bw {int(k.get('c.mode.bw', 0)):,} / colour {int(k.get('c.mode.color', 0)):,}" \
            if k.get("c.mode.bw") or k.get("c.mode.color") else ""
        stored = f"  {c['stored']:>7,} stored" if c["stored"] else ""
        print(f"{sc['name']:36} {c['total']:>9,} listed  {c['done']:>9,} saved{stored}  {c['skipped']:>7,} skipped  "
              f"{c['failed']:>5,} failed  {c['elsewhere']:>7,} in other scopes  {c['pending']:>9,} left{modes}{where}")
    n = store.stored_count()
    if n:
        print(f"storage: {n:,} photos, {store.stored_bytes() / 2**30:.1f} GB in {base / layout.STORAGE_DIR}; "
              f"pull, or run --pull, processes them")
    return 0


def cmd_index(args, conf, client) -> int:
    base, store = open_base(args, conf)
    rule = run_rule(args, store)
    if rule:
        family_from_args(args, conf, client, store, rule)
        return 0
    sc = scope_from_args(args, conf, client, store)
    do_index(args, client, store, sc)
    return 0


def cmd_run(args, conf, client) -> int:
    import stream as sm
    base, store = open_base(args, conf)
    with locked(store, "run", "run"):
        return run_locked(args, conf, client, base, store, sm)


def run_locked(args, conf, client, base: Path, store: Store, sm) -> int:
    if args.skip < 0:
        raise SystemExit("--skip takes a number of photos, 0 or more")
    scopes = None
    if args.by and (args.scope or not args.region):
        raise SystemExit("--by takes --region (the regions and years to split), not --scope")
    if args.resume and args.reindex:
        raise SystemExit("--resume skips the index and --reindex pages it again: give one of them")
    rule = run_rule(args, store) if args.region and not args.scope else None
    if rule:
        sc, scopes = family_from_args(args, conf, client, store, rule, index=not args.resume)
    elif args.scope:
        sc = scope_arg(base, store, args.scope)
        if args.reindex:
            ix.ensure_regions(client, store)
            sc = do_index(args, client, store, sc)
        print(f"Resuming {sc['name']}")
    elif args.region:
        sc = scope_from_args(args, conf, client, store, create=not args.resume)
        if args.resume:
            p = ix.progress_of(store, sc["name"])
            print(f"Resuming {sc['name']}, no indexing. Index: {ix.describe_progress(p)}.", flush=True)
        else:
            sc = do_index(args, client, store, sc)
    else:
        raise SystemExit("give --region (an alias, a region number or a title) or --scope NAME")
    names = sorted(scopes) if scopes else [sc["name"]]
    fields = parse_fields(args.txt_fields)
    n = sum(sm.cleanup_interrupted(store, base, name) for name in names)
    if n:
        print(f"  {n} photos were being written when the last run stopped; they are done again")
    if args.regroup:
        import regroup as rg
        print("Pulling the saved photos these scopes list from other scopes ...", flush=True)
        p = rg.plan(store, base, rg.pull_plan(store, names))
        print("\n".join(rg.summary(p)))
        if p["moves"] or p["remove"]:
            res = rg.apply(store, base, p, "pull", " ".join(names))
            print(f"  regroup run {res['run']}: {res['file']:,} photos moved, {res['missing']:,} files gone (downloaded "
                  f"again in this run), {res['removed']} emptied scopes removed; regroup --undo puts it back")
        if scopes:
            scopes = {name: store.scope(name) for name in names}
        else:
            sc = store.scope(sc["name"])
    opts = sm.Options(base=base, scope=sc, scopes=scopes, min_side=args.min_side, max_aspect=aspect_of(conf),
                      max_pixels=0 if args.raw else pixels_of(conf, args.max_pixels), png=args.png, wm_conf=args.wm_conf,
                      **({} if args.raw else qualities_of(conf)),
                      wm_paint_conf=args.wm_paint_conf, wm_mode=args.wm_mode, borders=not args.no_borders,
                      dark_borders=not args.no_dark, fetchers=args.fetchers or conf.fetchers, depth=args.depth,
                      writers=args.writers, report_every=args.report_every, limit=args.limit, retries=args.retries,
                      txt_fields=fields, json_files=conf.json_files, json_sidecar=args.json_sidecar,
                      comments=args.comments, raw=args.raw, keep_rejects=args.keep_rejects,
                      storage_threads=args.storage_threads, prepare_workers=prepare_workers_of(args),
                      display="json" if args.json_progress else "quiet" if args.quiet else "live")

    total = sum(store.scope_counts(name)["total"] for name in names)
    pairs = store.pending_pairs(names, args.retries)
    completed = total - len(pairs)
    if args.skip:
        n = len(pairs)
        pairs = skip_pairs(store, names, pairs, args.skip)
        print(f"  --skip {args.skip:,}: starting at photo {args.skip + 1:,} of {total:,}; {n - len(pairs):,} photos "
              f"before it that are not done are left for later", flush=True)
    where = sc["name"] if len(names) == 1 else f"its {len(names)} scopes"
    if args.raw:
        s = sm.Stream(client, store, opts)
        res = s.run(pairs, total, completed)
        left = len(store.pending_pairs(names, args.retries))
        print(f"This run: {res['stored']:,} photos stored, {res['skipped']:,} skipped, {res['failed']:,} failed in "
              f"{res['seconds'] / 60:.1f} min. Left to download in {where}: {left:,}. In the storage: "
              f"{store.stored_count():,} photos, {store.stored_bytes() / 2**30:.1f} GB; pull, or run --pull, processes them.")
        return 0
    stored = []
    if args.pull:
        # a stored photo of any scope may have been cut short while written: back to the storage
        n = sum(sm.cleanup_interrupted(store, base, s["name"]) for s in store.scopes() if s["name"] not in names)
        if n:
            print(f"  {n} stored photos were being written when an earlier run stopped; they are in the storage again")
        stored = store.stored_pairs()
    elif store.stored_count():
        print(f"  {store.stored_count():,} photos wait in the storage {base / layout.STORAGE_DIR}; add --pull, or run "
              f"pull, to process them", flush=True)
    in_scope = sum(1 for _, s in stored if s in names)      # listed by the run's scopes: in total already
    return process_stream(args, conf, client, base, store, sm, opts, names, pairs, total + len(stored) - in_scope,
                          completed - in_scope, stored, where)


def process_stream(args, conf, client, base: Path, store: Store, sm, opts, names: list[str], pairs: list,
                   total: int, completed: int, stored: list, where: str) -> int:
    """Load the models, run the stream over the downloads (pairs) and the storage (stored),
    then caption what has no VLM caption yet. names: the scopes to report and caption."""
    caption_client = None
    if args.autocaption:
        caption_client = vlm_client(conf)
    vram, detector, lama, classifier, mode_probe = load_models(args, conf)
    if vram.ok:
        print(f"  {vram.text()}", flush=True)
    exporter = None
    if args.export or args.export_reset:
        import liveexport as lx
        exporter = lx.open_for_run(store, base, conf.split, reset=args.export_reset, order=export_layout_of(conf))
    else:
        import liveexport as lx
        if lx.Exporter.exists_for(store):
            print(f"  This base has an export ({lx.root_of(base)}); add --export to link the new photos into it too")
    s = sm.Stream(client, store, opts, classifier, detector, lama, caption_client, vram, mode_probe, exporter)
    res = s.run(pairs, total, completed, stored=stored)
    left = len(store.pending_pairs(names, args.retries))
    text = f"This run: {res['done']:,} saved"
    if stored:
        text += f" ({res['pulled']:,} of them from the storage)"
    text += (f", {res['skipped']:,} skipped, {res['failed']:,} failed in {res['seconds'] / 60:.1f} min. "
             f"Left to do in {where}: {left:,}")
    if stored:
        text += f"; in the storage: {store.stored_count():,}"
    print(text + ".")
    if caption_client is not None and not s.stop_requested_by_captioner():
        # photos saved earlier or left behind: caption them now
        import autocaption as ac
        from progress import Display
        for name in names:
            prefix = ""
            targets, _ = ac.select(store, base, name, prefix, False, args.retries)
            if not targets:
                continue
            print(f"Captioning {len(targets):,} saved photos of {name} that have no VLM caption yet ...")
            run_id = store.start_caption_run(name, name, caption_client.model, caption_client.prompt_sha)
            mode = "json" if args.json_progress else "quiet" if args.quiet else "live"
            with Display(len(targets), 0, mode, vram=vram) as d:
                cres = ac.run(store, base, name, targets, caption_client, d, run_id, args.report_every, vram)
            if cres.get("stopped"):
                break
    vram.stop()
    return 0


def cmd_reprocess(args, conf, client) -> int:
    """reprocess --downscale [N]: the saved photos over N^2 pixels rewritten at that area."""
    import reprocess as rp
    from progress import Display
    base, store = open_base(args, conf)
    no_open_review(store, base, "reprocess")
    side = args.downscale
    if side is None:
        raise SystemExit("give --downscale [N]: the side of the square whose area is the limit (2048 assumed)")
    if side < 64:
        raise SystemExit(f"--downscale {side}: give at least 64")
    scopes = [sc["name"] for sc in scopes_arg(base, store, args.scope)] if args.scope else None
    with locked(store, "reprocess"):
        items = rp.candidates(store, scopes, side)
        total_mb = sum(it["bytes"] for it in items) / 2**20
        where = f"{len(scopes)} scope{'s' if len(scopes) != 1 else ''}" if scopes else "the whole base"
        quality = qualities_of(conf)["jpeg_quality_downscaled"]
        print(f"{len(items):,} saved photos of {where} have more than {side} x {side} = {side * side:,} pixels "
              f"({total_mb:,.0f} MB now); they are downscaled to that area in place, as JPEG at quality {quality}"
              + (" or PNG" if not args.png else ", every one as PNG") + ".")
        if not items:
            return 0
        if args.dry_run:
            report = base / "_reports" / f"reprocess-downscale-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
            report.parent.mkdir(parents=True, exist_ok=True)
            with open(report, "w", encoding="utf-8") as f:
                for it in items:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
            print(f"Dry run: nothing changed; the list is in {report}")
            return 0
        if not confirm("This cannot be undone (the larger pixels are gone). Go on? [y/N] ", args.yes):
            print("Nothing changed.")
            return 1
        exporter = None
        import liveexport as lx
        if lx.Exporter.exists_for(store):
            exporter = lx.Exporter(store, base, conf.split, order=export_layout_of(conf))
            print(f"  the export {exporter.root} is relinked as the photos are rewritten")
        swept = rp.sweep_temp(base, items)
        if swept:
            print(f"  {swept} temporary files of a stopped run deleted")
        mode = "json" if args.json_progress else "quiet" if args.quiet else "live"
        t0 = time.time()
        with Display(len(items), 0, mode) as d:
            def progress(n, total):
                d.advance(saved=True)
                d.set(f"downscaled {n:,} of {total:,}")
            res = rp.apply(store, base, items, side, png=args.png, workers=args.workers, exporter=exporter, log=d.print,
                           progress=progress, quality=quality)
        if res["stopped"]:
            print(f"Stopped. {len(rp.candidates(store, scopes, side)):,} photos are left; the same command continues.")
        print(f"Downscaled {res['done']:,} photos in {(time.time() - t0) / 60:.1f} min: {res['before'] / 2**20:,.0f} MB "
              f"to {res['after'] / 2**20:,.0f} MB; {res['failed']} failed, {res['repaired']} repaired (rewritten by a "
              f"run that stopped before their rows), "
              f"{res['caches']} caches of the old pixels deleted"
              + (f", {res['relinked']:,} export links made again" if exporter is not None else "") + ".")
    return 0


def cmd_pull(args, conf, client) -> int:
    """The processing of the storage alone: every stored photo (or those of the given scopes)
    through the complete pipeline into its scope, no downloads."""
    import stream as sm
    base, store = open_base(args, conf)
    with locked(store, "pull", "run"):
        names = [sc["name"] for sc in scopes_arg(base, store, args.scope)] if args.scope else None
        n = sum(sm.cleanup_interrupted(store, base, sc["name"]) for sc in store.scopes())
        if n:
            print(f"  {n} photos were being written when an earlier run stopped; the stored ones are in the storage "
                  f"again, the others are downloaded again")
        stored = store.stored_pairs(names)
        if not stored:
            print(f"Nothing to do: the storage {base / layout.STORAGE_DIR} holds no photos"
                  + (" of these scopes." if names else "."))
            return 0
        mine = sorted({s for _, s in stored})
        scopes = {name: store.scope(name) for name in mine}
        if any(sc is None for sc in scopes.values()):
            gone = ", ".join(n for n, sc in scopes.items() if sc is None)
            raise SystemExit(f"stored photos of a scope that no longer exists: {gone}")
        opts = sm.Options(base=base, scope=scopes[mine[0]], scopes=scopes, min_side=args.min_side,
                          max_aspect=aspect_of(conf), max_pixels=pixels_of(conf, args.max_pixels), png=args.png,
                          wm_conf=args.wm_conf, **qualities_of(conf),
                          wm_paint_conf=args.wm_paint_conf, wm_mode=args.wm_mode, borders=not args.no_borders,
                          dark_borders=not args.no_dark, fetchers=conf.fetchers, depth=args.depth,
                          writers=args.writers, report_every=args.report_every, limit=args.limit,
                          retries=args.retries, txt_fields=parse_fields(args.txt_fields), json_files=conf.json_files,
                          json_sidecar=args.json_sidecar, keep_rejects=args.keep_rejects,
                          storage_threads=args.storage_threads, prepare_workers=prepare_workers_of(args),
                          display="json" if args.json_progress else "quiet" if args.quiet else "live")
        print(f"Processing {len(stored):,} photos ({store.stored_bytes(names) / 2**30:.1f} GB) of the storage "
              f"{base / layout.STORAGE_DIR}, {len(mine)} scope{'s' if len(mine) > 1 else ''} ...", flush=True)
        where = mine[0] if len(mine) == 1 else f"its {len(mine)} scopes"
        return process_stream(args, conf, client, base, store, sm, opts, mine, [], len(stored), 0, stored, where)


def load_models(args, conf: cf.Conf, quality: float | None = None, colour_probe: bool = True,
                quality_niqe: float | None = None) -> tuple:
    """The watermark detector and LaMa, the classifier with its quality probe (at quality and
    quality_niqe, default the conf's thresholds), and the colour probe, all resident on the GPU.
    -> (vram monitor, detector, lama, classifier, colour probe)"""
    import vram as vr
    vram = vr.VramMonitor().start()
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
    classifier = mode_probe = None
    if not getattr(args, "no_classify", False):
        import classifier as clf
        samples = Path(args.samples).resolve() if args.samples else conf.samples
        classifier = clf.Classifier(samples, args.patches, args.min_confidence,
                                    priority_margin=margin_of(args, conf),
                                    quality_threshold=quality_of(conf) if quality is None else quality,
                                    quality_threshold_niqe=quality_niqe_of(conf) if quality_niqe is None
                                    else quality_niqe)
        classifier.embed(np.zeros((64, 64, 3), np.uint8))         # load the encoder now
        print(f"  classifier: {len(classifier.class_names)} classes from {samples}", flush=True)
        if colour_probe:
            import colour
            mode_probe = colour.probe_from_samples(samples, classifier.sample_vectors)
    if torch.cuda.is_available():
        print(f"  loaded in {time.time() - t0:.0f} s; GPU memory in use {torch.cuda.memory_allocated() / 2**30:.1f} GB",
              flush=True)
    import encode
    encode.turbo()
    return vram, detector, lama, classifier, mode_probe


def import_out(args, src: Path) -> Path:
    """-o, or <source>_pastvu next to the source; it must be writable and must not nest with the source."""
    out = Path(args.o).resolve() if args.o else src.parent / f"{src.name}_pastvu"
    if out == src or src in out.parents or out in src.parents:
        raise SystemExit(f"the output {out} and the source {src} must not be inside each other; give another -o")
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".pastvu_write_test"
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as e:
        raise SystemExit(f"cannot write into {out} ({e.strerror or e}); give an output folder with -o")
    return out


def cmd_import(args, conf, client) -> int:
    import importer
    import stream as sm
    src = Path(args.dataset).resolve()
    if not src.is_dir():
        raise SystemExit(f"--dataset {src}: no such folder")
    out = import_out(args, src)
    store = Store(out / DB_NAME)
    if store.scopes() and not store.is_import:
        raise SystemExit(f"{out} is a PastVu base folder; an import needs a folder of its own (-o)")
    if store.is_import:
        if Path(store.get_meta("source")) != src:
            raise SystemExit(f"{out} is the import of {store.get_meta('source')}, not of {src}")
        split = bool(store.get_meta("bw_color"))
        if args.top and (args.top != "none") != split:
            raise SystemExit(f"{out} was made {'with' if split else 'without'} black-and-white and colour folders; "
                             f"--top {args.top} cannot change that. Import into a new folder (-o) instead")
    else:
        top = args.top or ("none" if conf.bw_color is False else
                           "color" if export_layout_of(conf) == layout.COLOR_FIRST else "category")
        split = top != "none"
        store.set_meta("kind", importer.KIND)
        store.set_meta("source", str(src))
        store.set_meta("layout", "flat")
        store.set_meta("bw_color", split)
        store.set_meta("tree_order", top_order(top) or export_layout_of(conf))
        layout.use_names(store.names())
    if args.top:
        store.set_meta("top", args.top)
    quality = quality_for(store, conf, args.quality_threshold)
    quality_niqe = quality_niqe_for(store, conf, args.quality_threshold_niqe)
    if args.against:
        store.set_meta("against", str(Path(args.against).resolve()))
    against_base = store.get_meta("against")
    order = import_order(store, conf)
    name = layout.slug(src.name) or "import"
    sc = store.scope(name)
    if sc is None:
        sc = store.add_scope(name, [], (0, 0), None, args.split, label=name, width=3 if args.split > 0 else 0,
                             folder="")
    with locked(store, "import"):
        if split and store.get_meta("tree_order") != order:
            print(f"Moving the saved photos into {order} folders ({import_order_source(store)}) ...", flush=True)
            n = importer.reorder(store, out, sc, order)
            print(f"  {n:,} photos moved")
            sc = store.scope(name)
        res = importer.index_source(store, name, src)
        print(f"Import of {src} into {out}: {res['new']:,} new images listed ({res['known']:,} known before"
              + (f"; {res['renamed']:,} renamed because the name was taken" if res["renamed"] else "") + ")",
              flush=True)
        tree = ("<category>" if not split else
                "<category>/<bw|color>" if order == layout.CATEGORY_FIRST else "<bw|color>/<category>")
        print(f"  tree {tree}, quality threshold {quality:g}"
              + (" (off)" if not quality else ""), flush=True)
        n = sm.cleanup_interrupted(store, out, name)
        if n:
            print(f"  {n} photos were being written when the last import stopped; they are done again")
        added = importer.sync_sidecars(store, out, src, name)
        if added:
            print("  sidecars copied for photos imported before: "
                  + ", ".join(f"{n:,} {ext}" for ext, n in sorted(added.items())), flush=True)
        total = store.scope_counts(name)["total"]
        pairs = store.pending_pairs([name], args.retries)
        if not pairs:
            print(f"Nothing to do: all {total:,} images are processed.")
            return 0
        args.no_classify = False
        vram, detector, lama, classifier, mode_probe = load_models(args, conf, quality, colour_probe=split,
                                                                   quality_niqe=quality_niqe)
        against = None
        if against_base:
            against = importer.Against(Path(against_base), classifier.model_id, classifier.patches)
        opts = sm.Options(base=out, scope=sc, min_side=args.min_side, max_aspect=aspect_of(conf),
                          max_pixels=pixels_of(conf, getattr(args, "max_pixels", None)), png=args.png,
                          **qualities_of(conf),
                          wm_conf=args.wm_conf, wm_paint_conf=args.wm_paint_conf, wm_mode=args.wm_mode,
                          borders=not args.no_borders, dark_borders=not args.no_dark, fetchers=args.threads,
                          depth=args.depth, writers=args.writers, report_every=args.report_every, limit=args.limit,
                          retries=args.retries, json_files=False, colour=split,
                          display="json" if args.json_progress else "quiet" if args.quiet else "live")
        s = sm.Stream(client, store, opts, classifier, detector, lama, None, vram, mode_probe, None,
                      source=src, against=against)
        res = s.run(pairs, total, total - len(pairs))
        left = len(store.pending_pairs([name], args.retries))
        print(f"This import: {res['done']:,} saved, {res['skipped']:,} skipped, {res['failed']:,} failed in "
              f"{res['seconds'] / 60:.1f} min. Left to do: {left:,}.")
        vram.stop()
    return 0


def cmd_captions(args, conf, client) -> int:
    base, store = open_base(args, conf)
    scopes = scopes_arg(base, store, args.scope)
    fields = parse_fields(args.txt_fields)
    with locked(store, "captions", "shared"):
        for sc in scopes:
            n = changed = missing = vlm = 0
            autocaptioned = store.captioned_cids(sc["name"])
            for cid, path, category, sha in store.done_rows(sc["name"]):
                if cid in autocaptioned:
                    vlm += 1
                    continue
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
            print(f"{sc['name']}: {n:,} captions checked, {changed:,} {verb}, {missing:,} photos without a record or "
                  f"image")
            if vlm:
                print(f"  {vlm:,} photos have a VLM caption and were left alone (autocaption --undo brings their facts "
                      f"line back)")
    return 0


def cmd_reclassify(args, conf, client) -> int:
    import reclassify as rc
    base, store = open_base(args, conf)
    if args.scope:
        scopes = scopes_arg(base, store, args.scope)
    else:
        scopes = [sc for sc in store.scopes() if not args.undo or store.last_reclass(sc["name"])]
        if not scopes:
            print(f"no scopes {'with a reclassify to undo ' if args.undo else ''}in {base}")
            return 0
        if not args.dry_run:
            photos = sum(len(store.done_rows(sc["name"])) for sc in scopes)
            what = "undo the last reclassify of" if args.undo else \
                "move the _unsure photos into their best-guess categories in" if args.unsure else "reclassify"
            if not confirm(f"This will {what} {len(scopes)} scope folders of {base} ({photos:,} saved photos): "
                           f"{', '.join(sc['name'] for sc in scopes)}. Continue? [y/N] ", args.yes):
                print("Nothing done.")
                return 1
    with locked(store, "reclassify", "shared" if args.dry_run else "exclusive"):
        import liveexport as lx
        exporter = lx.Exporter(store, base, conf.split, order=export_layout_of(conf)) \
            if lx.Exporter.exists_for(store) and not args.dry_run else None
        if store.is_import and store.get_meta("bw_color") and not args.dry_run and not args.undo:
            import importer
            order = import_order(store, conf)
            if store.get_meta("tree_order") != order:
                print(f"Moving the saved photos into {order} folders ({import_order_source(store)}) ...", flush=True)
                for sc in scopes:
                    print(f"  {importer.reorder(store, base, sc, order):,} photos moved")
                scopes = [store.scope(sc["name"]) for sc in scopes]
        if args.undo:
            for sc in scopes:
                res = rc.undo(store, base, sc, exporter=exporter)
                print(f"{sc['name']}: undid reclassify run {res['run']}: {res['back']:,} photos moved back, "
                      f"{res['skipped']:,} left where they are, {res['pruned']} empty folders removed")
            return 0
        if args.unsure:
            classifier, samples = None, "unsure: best guess"
        else:
            import classifier as clf
            samples = Path(args.samples).resolve() if args.samples else conf.samples
            classifier = clf.Classifier(samples, args.patches, args.min_confidence,
                                        priority_margin=margin_of(args, conf),
                                        quality_threshold=quality_for(store, conf),
                                        quality_threshold_niqe=quality_niqe_for(store, conf))
        against = None
        if classifier is not None and store.is_import and store.get_meta("against"):
            import importer
            against = importer.Against(Path(store.get_meta("against")), classifier.model_id, classifier.patches)
        plans = []
        for sc in scopes:
            if classifier is None:
                print(f"Moving the photos of {sc['name']} out of _unsure into their best-guess categories ...",
                      flush=True)
                p = rc.plan_unsure(store, base, sc)
            else:
                print(f"Sorting {sc['name']} again with {len(classifier.class_names)} classes from {samples} ...",
                      flush=True)
                p = rc.plan(store, base, sc, classifier, args.override_manual, max_aspect=aspect_of(conf),
                            against=against)
            print(rc.summary(p))
            if args.dry_run:
                report = base / "_reports" / f"{sc['name']}-reclassify-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
                report.parent.mkdir(parents=True, exist_ok=True)
                with open(report, "w", encoding="utf-8") as f:
                    for m in p["moves"]:
                        f.write(json.dumps(m, ensure_ascii=False) + "\n")
                    for m in p["low_quality"] + p["aspect"] + p["copies"]:
                        f.write(json.dumps({**m, "delete": m["reason"]}, ensure_ascii=False) + "\n")
                print(f"Dry run: nothing moved or deleted. The planned moves and deletions are in {report}")
                continue
            plans.append((sc, p))
        low = sum(len(p["low_quality"]) for _, p in plans)
        wide = sum(len(p["aspect"]) for _, p in plans)
        copies = sum(len(p["copies"]) for _, p in plans)
        if low + wide + copies:
            what = []
            samples_bad = sum(1 for _, p in plans for m in p["low_quality"] if m.get("sample"))
            by_niqe = sum(1 for _, p in plans for m in p["low_quality"] if m.get("niqe"))
            if low - samples_bad - by_niqe:
                what.append(f"{low - samples_bad - by_niqe:,} low quality by the probe "
                            f"(P(bad) >= {classifier.quality.threshold:g})")
            if by_niqe:
                what.append(f"{by_niqe:,} low quality by the probe + NIQE "
                            f"(P(bad) >= {classifier.quality.threshold_niqe:g})")
            if samples_bad:
                what.append(f"{samples_bad:,} copies of _quality_bad samples")
            if wide:
                what.append(f"{wide:,} with an aspect ratio over {conf.max_aspect:.2f}:1")
            if copies:
                what.append(f"{copies:,} copies of photos of {store.get_meta('against')}")
            again = "imported" if store.is_import else "downloaded"
            if not confirm(f"This deletes {low + wide + copies:,} photos, {', '.join(what)}; they are never {again} "
                           f"again, and --undo cannot bring them back. Continue? [y/N] ", args.yes):
                print("Nothing done. quality_threshold = 0, quality_threshold_niqe = 0 and max_aspect = 0 in "
                      "[defaults] of the conf reclassify "
                      "without deleting" + (" (this import keeps the quality threshold it was given: import "
                                            "--quality-threshold 0 changes it)"
                                            if store.is_import and store.get_meta("quality_threshold") is not None
                                            else "") + ".")
                return 1
        for sc, p in plans:
            res = rc.apply(store, base, sc, p, str(samples), exporter=exporter)
            print(f"Reclassify run {res['run']}: {res['moved']:,} photos moved, {res['failed']:,} not moved, "
                  f"{res['deleted']:,} photos deleted (low quality, aspect ratio or copies), "
                  f"{res['manual']:,} hand-sorted photos recorded where they are, {res['pruned']} empty folders "
                  f"removed. --undo puts the moves back.")
    return 0


def cmd_cleanup(args, conf, client) -> int:
    import cleanup as cu
    base, store = open_base(args, conf)
    if args.done or args.undo:
        if args.done and args.undo:
            raise SystemExit("give --done or --undo, not both")
        if args.threshold is not None or args.scope:
            raise SystemExit("--done and --undo settle the whole open review; they take no threshold or scope")
        with locked(store, "cleanup"):
            import liveexport as lx
            exporter = lx.Exporter(store, base, conf.split, order=export_layout_of(conf)) \
                if lx.Exporter.exists_for(store) else None
            res = cu.settle(store, base, keep_good=args.done, exporter=exporter)
        how = "kept for good (no reclassify or cleanup sends them away again for their quality)" if args.done \
            else "put back as they were"
        print(f"Review settled: {res['back']:,} photos {how}; {res['gone']:,} you deleted are skipped for good"
              + (f"; {res['stuck']:,} could not go back and stay in the review folder" if res["stuck"] else ""))
        if res["left"]:
            print(f"  {res['folder']} still holds {res['left']:,} files that are not photos of the review; it stays")
        return 0
    if args.threshold is None:
        raise SystemExit("give the threshold, a P(bad) such as 0.8 (cleanup 0.8 [scope ...]); or --done, --undo")
    try:
        threshold = float(args.threshold)
    except ValueError:
        raise SystemExit(f"cleanup {args.threshold}: the first value is the threshold, a P(bad) such as 0.8")
    if not 0 < threshold <= 1:
        raise SystemExit(f"threshold {threshold}: give a P(bad) above 0 and at most 1")
    scopes = scopes_arg(base, store, args.scope) if args.scope else store.scopes()
    if not scopes:
        print(f"no scopes in {base}")
        return 0
    with locked(store, "cleanup", "shared" if args.dry_run else "exclusive"):
        import classifier as clf
        samples = Path(args.samples).resolve() if args.samples else conf.samples
        classifier = clf.Classifier(samples, args.patches, 0.7, quality_threshold=threshold, categories=False,
                                    quality_threshold_niqe=threshold if quality_niqe_of(conf) > 0 else 0)
        if classifier.quality is None:
            raise SystemExit(f"no quality probe: {samples} needs at least 20 images in _quality_good and in "
                             f"_quality_bad")
        plans = []
        for sc in scopes:
            p = cu.plan(store, base, sc, classifier, threshold, args.override_manual)
            print(cu.summary(sc["name"], p, threshold), flush=True)
            if p["picks"]:
                plans.append((sc, p))
        n = sum(len(p["picks"]) for _, p in plans)
        folder = cu.review_dir(store, base)
        if args.dry_run:
            report = base / "_reports" / f"cleanup-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
            report.parent.mkdir(parents=True, exist_ok=True)
            with open(report, "w", encoding="utf-8") as f:
                for sc, p in plans:
                    for m in p["picks"]:
                        f.write(json.dumps({**m, "scope": sc["name"]}, ensure_ascii=False) + "\n")
            print(f"Dry run: nothing moved. The {n:,} photos to review are in {report}")
            return 0
        if not n:
            print("Nothing to review.")
            return 0
        if not confirm(f"This moves {n:,} photos (P(bad) >= {threshold}) with their captions into {folder}. "
                       f"Continue? [y/N] ", args.yes):
            print("Nothing done.")
            return 1
        moved = sum(cu.move_to_review(store, base, sc, p["picks"], threshold) for sc, p in plans)
    total = store.review_count()
    print(f"{moved:,} photos moved into {folder}" + (f" ({total:,} in the review now)" if total != moved else "")
          + ". Delete the ones that are bad, then run cleanup --done to put the rest back and keep them for good, "
            "or cleanup --undo to put the rest back as they were.")
    return 0


def caption_targets(base: Path, store: Store, values: list[str]) -> list[tuple[dict, str]]:
    out, seen = [], set()
    for v in values:
        hits = [(sc, "") for sc in scopes_arg(base, store, [v])] if is_pattern(v) \
            else target_arg(base, store, v)
        for sc, prefix in hits:
            if (sc["name"], prefix) not in seen:
                seen.add((sc["name"], prefix))
                out.append((sc, prefix))
    return out


def cmd_autocaption(args, conf, client) -> int:
    import autocaption as ac
    import vram as vr
    from progress import Display
    base, store = open_base(args, conf)
    targets_of = caption_targets(base, store, args.target)
    with locked(store, "autocaption", "shared"):
        if args.undo:
            for name in dict.fromkeys(sc["name"] for sc, _ in targets_of):
                res = ac.undo(store, base, name)
                print(f"Undid autocaption run {res['run']} ({res['target']}): {res['restored']:,} captions put back, "
                      f"{res['kept']:,} edited since and kept, {res['missing']:,} photos not found")
            return 0
        jobs = []
        left = args.limit
        for sc, prefix in targets_of:
            targets, counts = ac.select(store, base, sc["name"], prefix, args.recaption, args.retries)
            if left is not None:
                targets = targets[:max(0, left)]
                left -= len(targets)
            print(f"{prefix.rstrip('/') or sc['name']}: {counts['listed']:,} saved photos, {counts['captioned before']:,} captioned "
                  f"before, {counts['gave up']:,} failed {args.retries} times; {len(targets):,} to caption")
            if targets:
                jobs.append((sc, prefix, targets))
        if not jobs:
            return 0
        capt = vlm_client(conf)
        vram = vr.VramMonitor().start()
        if vram.ok:
            print(f"  {vram.text()}")
        mode = "json" if args.json_progress else "quiet" if args.quiet else "live"
        code = 0
        for sc, prefix, targets in jobs:
            run_id = store.start_caption_run(sc["name"], prefix or sc["name"], capt.model, capt.prompt_sha)
            with Display(len(targets), 0, mode, vram=vram) as d:
                res = ac.run(store, base, sc["name"], targets, capt, d, run_id, args.report_every, vram)
            if res.get("stopped"):
                code = 0 if "Ctrl+C" in res["stopped"] else 1
                break
        vram.stop()
    return code


def cmd_niqe(args, conf, client) -> int:
    """niqe [target ...]: NIQE rows for the saved photos that have none of the current version
    (niqe.py). Shared lock: it runs beside run, pull, import, detect and autocaption."""
    import niqe
    import vram as vr
    from progress import Display
    base, store = open_base(args, conf)
    targets_of = caption_targets(base, store, args.target) if args.target else [(sc, "") for sc in store.scopes()]
    if not targets_of:
        print(f"{base} has no scopes yet")
        return 0

    def pending() -> list[tuple[int, str]]:
        out = []
        for sc, prefix in targets_of:
            out += store.niqe_pending(sc["name"], prefix, niqe.VERSION, redo=False)
        return out

    with locked(store, "niqe", "shared"):
        targets, saved, scored = [], 0, 0
        for sc, prefix in targets_of:
            rows = store.niqe_pending(sc["name"], prefix, niqe.VERSION, redo=args.redo)
            targets += rows
            if prefix:
                print(f"{prefix.rstrip('/')}: {len(rows):,} saved photos to score")
                continue
            c = store.niqe_count(sc["name"], niqe.VERSION)
            saved, scored = saved + c["saved"], scored + c["scored"]
            if args.target:                  # every scope of the base: one line for all, below
                print(f"{sc['name']}: {c['saved']:,} saved photos, {c['scored']:,} with NIQE {niqe.VERSION}; "
                      f"{len(rows):,} to score")
        if not args.target:
            print(f"{base}: {saved:,} saved photos in {len(targets_of)} scopes, {scored:,} with NIQE {niqe.VERSION}; "
                  f"{len(targets):,} to score")
        if args.limit is not None:
            targets = targets[:max(0, args.limit)]
        if not targets and not args.follow:
            return 0
        scorer = niqe.Niqe()
        print(f"  NIQE at {niqe.PIXELS:,} pixels on {scorer.device}, {args.threads} read threads"
              + ("; then waiting for new photos until Ctrl+C" if args.follow else ""), flush=True)
        vram = vr.VramMonitor().start()
        mode = "json" if args.json_progress else "quiet" if args.quiet else "live"
        with Display(len(targets), 0, mode, vram=vram) as d:
            niqe.run(store, base, targets, scorer, d, args.report_every, args.threads,
                     follow=pending if args.follow else None, vram=vram)
        vram.stop()
    return 0


def cmd_deduplicate(args, conf, client) -> int:
    """deduplicate [target ...]: copies of one photo inside the base (copies.py). First the hashes of
    the saved photos that have no row of the current version, then the copies, then the moves.
    Shared lock while it only reads (--hashes-only, --dry-run): it runs beside run, pull, import,
    niqe and detect. Moving and --undo take the lock for themselves."""
    import copies
    from progress import Display
    if args.undo and (args.dry_run or args.report or args.target or args.hashes_only):
        raise SystemExit("--undo puts back the last deduplicate run of the whole base; give it alone")
    if args.report and (args.target or args.hashes_only):
        raise SystemExit("--report moves the groups of that report; give no scopes or --hashes-only with it")
    features = args.features or ("report" if args.dry_run else "off")
    kinds = copies.KINDS if features == "copies" else copies.SURE_KINDS
    base, store = open_base(args, conf)
    reads_only = args.hashes_only or args.dry_run
    log = lambda text: print(text, flush=True)          # noqa: E731

    def exporter_of():
        import liveexport as lx
        if not lx.Exporter.exists_for(store):
            return None
        return lx.Exporter(store, base, conf.split, order=export_layout_of(conf))

    def move(groups, how):
        n = sum(1 for g in groups if g["keeper"] is not None for m in g["members"] if m.get("copy"))
        rules = Counter(g["rule"] for g in groups if g["keeper"] is not None for m in g["members"] if m.get("copy"))
        print(f"{len(groups):,} groups ({how}; pairs: {', '.join(kinds)}): {n:,} copies to move ("
              + ", ".join(f"{v:,} {k}" for k, v in rules.items()) + ")", flush=True)
        if args.dry_run:
            print("Dry run: nothing moved.")
            return 0
        res = copies.apply(store, base, groups, {"how": how, "kinds": list(kinds), "targets": args.target or None},
                           exporter=exporter_of(), log=log)
        print(f"Moved {res['moved']:,} copies to {res['folder']} (run {res['run']}); "
              f"{res['missing']:,} not found where the store says, {res['failed']:,} failed. "
              f"`deduplicate --undo` puts them back.")
        return 0

    if args.undo:
        with locked(store, "deduplicate"):
            res = copies.undo(store, base, exporter=exporter_of(), log=log)
        print(f"Undid deduplicate run {res['run']}: {res['back']:,} copies put back, {res['gone']:,} deleted from "
              f"{res['folder']} by hand stay skipped, {res['stuck']:,} could not be put back")
        return 0
    if args.report:
        with locked(store, "deduplicate", "shared" if args.dry_run else "exclusive"):
            return move(copies.groups_from_report(Path(args.report), kinds), f"from {Path(args.report).name}")

    targets_of = caption_targets(base, store, args.target) if args.target else [(sc, "") for sc in store.scopes()]
    if not targets_of:
        print(f"{base} has no scopes yet")
        return 0
    with locked(store, "deduplicate", "shared" if reads_only else "exclusive"):
        targets, saved, hashed = [], 0, 0
        for sc, prefix in targets_of:
            rows = store.hashes_pending(sc["name"], prefix, copies.HASH_VERSION, redo=args.redo)
            targets += rows
            if prefix:
                print(f"{prefix.rstrip('/')}: {len(rows):,} saved photos to hash")
                continue
            c = store.hashes_count(sc["name"], copies.HASH_VERSION)
            saved, hashed = saved + c["saved"], hashed + c["hashed"]
            if args.target:
                print(f"{sc['name']}: {c['saved']:,} saved photos, {c['hashed']:,} hashed; {len(rows):,} to hash")
        if not args.target:
            print(f"{base}: {saved:,} saved photos in {len(targets_of)} scopes, {hashed:,} hashed "
                  f"({copies.HASH_VERSION}); {len(targets):,} to hash")
        if args.limit is not None:
            targets = targets[:max(0, args.limit)]
        if targets:
            print(f"  hashing with {args.threads} threads", flush=True)
            mode = "json" if args.json_progress else "quiet" if args.quiet else "live"
            with Display(len(targets), 0, mode) as d:
                res = copies.hash_photos(store, base, targets, d, args.report_every, args.threads)
            if res.get("stopped"):
                return 1
        if args.hashes_only:
            return 0
        import process  # noqa: F401 - puts the classify folder on sys.path
        import classify as cl
        in_targets = None
        if args.target:
            in_targets = {cid for sc, prefix in targets_of
                          for cid, _ in store.hashes_pending(sc["name"], prefix, copies.HASH_VERSION, redo=True)}
            print(f"  copies of {len(in_targets):,} saved photos in the targets, among the whole base")
        review = False if args.review is None else None if args.review <= 0 else args.review
        print(f"Finding the copies: the {args.neighbours} nearest photos of each at cosine {args.min_cos:g} or more; "
              f"feature test {features}", flush=True)
        summary = copies.find(store, base, cl.MODEL_REPO, args.patches, targets=in_targets, min_cos=args.min_cos,
                              k=args.neighbours, features=features != "off", threads=args.threads, review=review,
                              copy_kinds=kinds, log=log)
        print(copies.summary_text(summary))
        if args.dry_run:
            print(f"Dry run: nothing moved ({summary['minutes']:.1f} min).")
            return 0
        return move(copies.groups_from_report(Path(summary["reports"]["groups"]), kinds), "found now")


def cmd_detect(args, conf, client) -> int:
    """detect [target ...]: boxes rows for the saved photos that have none of the current vocabulary
    (detect.py). Shared lock: it runs beside run, pull, import, niqe and autocaption."""
    import detect as dt
    import vram as vr
    from progress import Display
    base, store = open_base(args, conf)
    targets_of = caption_targets(base, store, args.target) if args.target else [(sc, "") for sc in store.scopes()]
    if not targets_of:
        print(f"{base} has no scopes yet")
        return 0
    vocab = dt.vocab_hash()

    def pending() -> list[tuple[int, str]]:
        out = []
        for sc, prefix in targets_of:
            out += store.boxes_pending(sc["name"], prefix, vocab, redo=False)
        return out

    with locked(store, "detect", "shared"):
        targets, saved, boxed = [], 0, 0
        for sc, prefix in targets_of:
            rows = store.boxes_pending(sc["name"], prefix, vocab, redo=args.redo)
            targets += rows
            if prefix:
                print(f"{prefix.rstrip('/')}: {len(rows):,} saved photos to box")
                continue
            c = store.boxes_count(sc["name"], vocab)
            saved, boxed = saved + c["saved"], boxed + c["boxed"]
            if args.target:                  # every scope of the base: one line for all, below
                print(f"{sc['name']}: {c['saved']:,} saved photos, {c['boxed']:,} with boxes of vocabulary {vocab}; "
                      f"{len(rows):,} to box")
        if not args.target:
            print(f"{base}: {saved:,} saved photos in {len(targets_of)} scopes, {boxed:,} with boxes of vocabulary "
                  f"{vocab}; {len(targets):,} to box")
        if args.limit is not None:
            targets = targets[:max(0, args.limit)]
        if not targets and not args.follow:
            return 0
        models = dt.load_models(args.imgsz)
        print(f"  YOLOE-26l and the face model at {args.imgsz} px, batches of {args.batch}, vocabulary {vocab}"
              + ("; then waiting for new photos until Ctrl+C" if args.follow else ""), flush=True)
        vram = vr.VramMonitor().start()
        mode = "json" if args.json_progress else "quiet" if args.quiet else "live"
        with Display(len(targets), 0, mode, vram=vram) as d:
            dt.run(store, base, targets, models, d, args.report_every, vram, args.batch,
                   follow=pending if args.follow else None)
        vram.stop()
    return 0


def cmd_export(args, conf, client) -> int:
    import export as ex
    base, store = open_base(args, conf)
    scopes = scopes_arg(base, store, args.scope)
    if args.to and len(scopes) > 1:
        raise SystemExit("--to takes one scope; without it each scope goes to <base>/<scope>_records.<format>")
    ext = ".jsonl" if args.format == "jsonl" else ".parquet"
    for sc in scopes:
        dst = Path(args.to).resolve() if args.to else base / f"{sc['name']}_records{ext}"
        n = ex.export(store, sc["name"], dst, args.format)
        print(f"{n:,} records written to {dst}")
    return 0


def cmd_extract(args, conf, client) -> int:
    import extract as xt
    base, store = open_base(args, conf)
    if args.out and len(args.category) > 1:
        raise SystemExit("-o takes one category; without it each goes to <base parent>/<category and filters>")
    scopes = xt.scope_filter(store.scopes(), args.scope)
    years, decade = None, None
    if args.decade is not None:
        if args.decade % 10:
            raise SystemExit(f"--decade {args.decade}: give the first year of a decade, e.g. 1970")
        decade = args.decade
    elif args.years:
        years = tuple(args.years)
    if not 0.0 <= args.colour_share <= 1.0:
        raise SystemExit("--colour-share is a fraction from 0 to 1, e.g. 0.4 for 40 % colour")
    if args.target is not None and args.target < 1:
        raise SystemExit("--target takes a number of photos, 1 or more")
    filters = {"years": years, "decade": decade, "scopes": args.scope or [],
               "mode": "bw" if args.bw else "color" if args.color else None, "regions": [], "region_labels": [],
               "min_res": args.min_res, "target": args.target, "colour_share": args.colour_share}
    if args.region:
        regions, labels = ix.resolve(store, conf.aliases, args.region)
        filters["regions"] = sorted(r["cid"] for r in regions)
        filters["region_labels"] = labels
    with locked(store, "extract", "shared"):
        picked = xt.select(store, base, scopes, args.category, filters)
        for label, sel in picked.items():
            out = Path(args.out).resolve() if args.out else base.parent / xt.folder_name(label, filters)
            photos = sel["photos"]
            if args.target:
                chosen, table = xt.pick_target(sel["info"], args.target, args.colour_share)
                print(f"{label}: --target {args.target:,} of {len(photos):,} photos that pass the filters; "
                      f"picked {len(chosen):,} (the largest first in each decade):")
                for m in ("bw", "color"):
                    row = sorted((d, v) for (mm, d), v in table.items() if mm == m)
                    if row:
                        cells = ", ".join(f"{d if d is not None else 'undated'}{'s' if d is not None else ''} "
                                          f"{c:,}/{n:,}" for d, (c, n) in row)
                        print(f"  {'bw' if m == 'bw' else 'colour'} {sum(c for _, (c, _) in row):,}: {cells}")
                photos = {c: photos[c] for c in chosen}
            modes = Counter(m or "none" for m, _ in photos.values())
            also = f" (also written {', '.join(h for h in sel['hits'] if h != label)})" if len(sel["hits"]) > 1 else ""
            no_mode = f" / no mode {modes['none']:,}" if modes.get("none") else ""
            print(f"{label}{also}: {len(photos):,} photos, bw {modes.get('bw', 0):,} / colour "
                  f"{modes.get('color', 0):,}{no_mode} -> {out}", flush=True)
            criteria = {"category": label, "argument": sel["arg"], "folders": sel["hits"], "base": str(base),
                        **{k: v for k, v in filters.items() if v}}
            res = xt.extract(photos, out, criteria, args.split, args.link, args.prune, args.dry_run)
            if args.dry_run:
                print(f"  dry run: {res.get('new', 0):,} to copy, {res.get('there', 0):,} there already, "
                      f"{res.get('stale', 0):,} no longer matching")
                continue
            done = res.get("linked", 0) + res.get("copied", 0)
            how = f"{res.get('linked', 0):,} linked, {res.get('copied', 0):,} copied" if args.link \
                else f"{done:,} copied"
            print(f"  {how}, {res.get('there', 0):,} there already, {res.get('failed', 0) + res.get('gone', 0):,} "
                  f"not copied; {res.get('stale', 0):,} files from earlier extracts no longer match"
                  + (" and were deleted" if args.prune else (" (--prune deletes them)" if res.get("stale") else "")))
    if not args.link:
        print(xt.LINK_TIP)
    return 0


def cmd_cherrypick(args, conf, client) -> int:
    import cherrypick as cp
    base, store = open_base(args, conf)
    lines = conf.cherrypick
    c = cp.parse_values(args.value, args.quality, args.top, args.per_category, require=not lines)
    given = any(v is not None for v in c.values())
    if args.balance is not None and not 0 <= args.balance <= 100:
        raise SystemExit(f"--balance {args.balance}: a percentage of colour photos, 0 to 100")
    no_modes = store.is_import and not store.get_meta("bw_color")
    if no_modes and args.balance is not None:
        raise SystemExit("--balance: this import does not tell black-and-white from colour (--top none)")
    default = None                                   # the setting of a category without a line of its own
    if given:
        default = {**c, "color": args.balance, "skip": False}
    elif "default" in lines:
        default = dict(lines["default"])
        if args.balance is not None:
            default["color"] = args.balance
    if default is not None:
        default = cp.effective(default, modes=not no_modes)
    balance = default["color"] if default else None
    decades = cp.parse_decades(args.decades)
    if decades and store.is_import:
        raise SystemExit("--decades: an import has no years")
    out = Path(args.out).resolve()
    if out == base or base in out.parents:
        raise SystemExit(f"-o {out} is inside the base folder; give a folder outside it")
    if default is not None and not default["skip"]:
        print(cp.describe(default, balance, decades) + (" (the default)" if lines else ""), flush=True)
    else:
        print("Decades: " + (", ".join(f"{d}s" for d in decades) if decades else "all") + ".", flush=True)
    settings = None
    with locked(store, "cherrypick", "shared"):
        photos = cp.load_photos(store, decades, args.category, outliers=bool(lines))
        if lines:
            with store.lock:
                names = [r[0] for r in store.db.execute(
                    "SELECT DISTINCT category FROM status WHERE state=? AND category IS NOT NULL", (cp.DONE,))]
            try:
                resolved, warnings = cp.resolve_settings(lines, names, default)
            except ValueError as e:
                raise SystemExit(f"{conf.path}: {e}")
            for w in warnings:
                print(f"  {w}")
            used = {p["cat"] for p in photos.values()}
            missing = sorted(n for n in used if resolved[n] is None)
            if missing:
                raise SystemExit(f"no setting for {', '.join(missing)}: give values on the command line or a "
                                 f"default line in [cherrypick] of {conf.path}")
            settings = {n: cp.effective(resolved[n], modes=not no_modes) for n in used}
            if no_modes and any(lines[k]["color"] is not None for k in lines):
                print("  --color on a [cherrypick] line is ignored: this import does not tell black-and-white "
                      "from colour")
            print(f"Per category: the [cherrypick] lines of {conf.path} ({len(lines)} lines); each row of the "
                  f"table ends with its category's setting", flush=True)
        if not photos:
            print("No saved photos match.")
            return 0
        import classifier as clf
        samples = Path(args.samples).resolve() if args.samples else conf.samples
        classifier = clf.Classifier(samples, args.patches, 0.7, quality_threshold=0.5, categories=False)
        if classifier.quality is None:
            raise SystemExit(f"no quality probe: {samples} needs at least 20 images in _quality_good and in "
                             f"_quality_bad")
        scores = cp.score(store, classifier.quality, classifier.model_id, classifier.patches, set(photos))
        tiers = None
        if args.interest:
            import detect as dt
            import interest as it
            cats = {p["cat"] for p in photos.values() if not (settings and settings[p["cat"]]["skip"])}
            print(it.describe(cats, conf.interest), flush=True)
            store.refill_boxes(dt.stats_from_json, dt.DERIVE)     # once after a change of the derived columns
            tiers = it.tiers(store, photos, conf.interest, dt.vocab_hash())
            known = sum(1 for t in tiers.values() if t != it.UNKNOWN)
            print(f"  {known:,} of {len(tiers):,} photos have a boxes row and a rule"
                  + ("" if known else "; run detect first for the ordering to mean anything"), flush=True)
        picked, table = cp.select(photos, scores, default, balance, tiers, settings)
        print(f"{len(photos):,} saved photos in {len({p['cat'] for p in photos.values()})} categories; "
              f"pool -> picked per category:")
        print(cp.table_text(table, interest=args.interest), flush=True)
        criteria = {"base": str(base), **{k: v for k, v in c.items() if v is not None}, "balance": balance,
                    "decades": decades, "categories": args.category or [], "samples": str(samples)}
        if settings:
            criteria["settings"] = {n: ({"skip": True} if v["skip"] else
                                        {k: x for k, x in v.items() if x is not None and k != "skip"})
                                    for n, v in sorted(settings.items())}
        if args.interest:
            criteria["interest"] = {"vocab": dt.vocab_hash(), "rules": it.rules_of(cats, conf.interest),
                                    **conf.interest.thresholds()}
        res = cp.write(picked, base, out, criteria, None if no_modes else export_layout_of(conf), args.split,
                       args.link, args.prune, args.dry_run, record=not store.is_import)
        if args.dry_run:
            print(f"Dry run: {res['new']:,} to copy into {out}, {res['there']:,} there already, "
                  f"{res['stale']:,} files of this base there that are no longer picked")
            return 0
        done = res["linked"] + res["copied"]
        how = f"{res['linked']:,} linked, {res['copied']:,} copied" if args.link else f"{done:,} copied"
        print(f"{out}: {how}, {res['there']:,} there already, {res['failed'] + res['gone']:,} not copied; "
              f"{res['stale']:,} files of this base from earlier picks are no longer picked"
              + (" and were deleted" if args.prune else (" (--prune deletes them)" if res["stale"] else "")))
    if not args.link:
        print(xt_link_tip())
    return 0


def xt_link_tip() -> str:
    import extract as xt
    return xt.LINK_TIP


def cmd_regroup(args, conf, client) -> int:
    import partition as pt
    import regroup as rg
    base, store = open_base(args, conf)
    if not args.dry_run:
        no_open_review(store, base, "regroup")
    with locked(store, "regroup", "shared" if args.dry_run else "exclusive"):
        if args.undo:
            res = rg.undo(store, base)
            print(f"Undid regroup run {res['run']} ({res['rule']}, {res['target']}): {res['file']:,} photos moved back, "
                  f"{res['owner']:,} skipped or failed photos returned, {res['members']:,} listings put back, "
                  f"{res['skipped']:,} left where they are, {res['removed']} scopes it made removed")
            if res["missing"]:
                print(f"  {res['missing']:,} photos whose files were gone stay marked not downloaded")
            return 0
        if not args.by:
            raise SystemExit("regroup needs --by RULE (decade, country, region, country,decade, region,decade) or --undo")
        rule = pt.parse_rule(args.by)
        if store.layout() != "flat" and not rule.decade:
            rule = pt.Rule(True, rule.place)
            print(f"In a {store.layout()} base every scope is one decade: regrouping by {rule.text}")
        whole = not args.target
        sources = scopes_arg(base, store, args.target) if args.target else store.scopes()
        if not sources:
            raise SystemExit(f"no scopes in {base}")
        if not store.regions_count():
            ix.ensure_regions(client, store)
        what = "every scope of the base" if whole else ", ".join(sc["name"] for sc in sources)
        print(f"Regrouping {what} by {rule.text} ...", flush=True)
        rp = rg.rule_plan(store, conf.aliases, conf.groups, rule, sources, whole)
        p = rg.plan(store, base, rp["wanted"], rp["members"], rp["defs"], pull_from={sc["name"] for sc in sources})
        print("\n".join(rg.summary(p)))
        if args.dry_run:
            print("Dry run: nothing moved.")
            return 0
        if not p["moves"] and not p["members"]:
            return 0
        res = rg.apply(store, base, p, rule.text, "*" if whole else " ".join(args.target))
        print(f"Regroup run {res['run']}: {res['file']:,} photos moved, {res['owner']:,} skipped or failed photos "
              f"reassigned, {res['missing']:,} files gone (marked not downloaded), {res['members']:,} listings moved, "
              f"{res['created']} scopes made, {res['removed']} removed, {res['failed']} not moved. --undo puts it back.")
        into = {}
        for m in p["moves"]:
            if m.kind == "file":
                into.setdefault(m.dst, set()).add(m.src)
        if any(len(v) > 1 for v in into.values()):
            print("  Scopes sorted with different samples may now mix category names; reclassify them if so.")
    return 0


def cmd_drop(args, conf, client) -> int:
    import regroup as rg
    from store import DONE
    base, store = open_base(args, conf)
    no_open_review(store, base, "drop")
    scopes = scopes_arg(base, store, args.scope)
    with locked(store, "drop"):
        for sc in scopes:
            rows = store.owned_rows(sc["name"])
            saved = sum(1 for r in rows if r[1] == DONE)
            print(f"{sc['name']}: {saved:,} saved photos (image and caption files), {len(rows) - saved:,} skipped or "
                  f"failed, {len(store.members(sc['name'])):,} listed; "
                  f"{', '.join(str(base / root) for _, root in layout.scope_roots(sc))}")
        if not args.yes:
            print("Nothing deleted. Add --yes to delete these files and forget the scopes; this cannot be undone.")
            return 1
        import liveexport as lx
        exporter = lx.Exporter(store, base, conf.split, order=export_layout_of(conf)) \
            if lx.Exporter.exists_for(store) else None
        for sc in scopes:
            res = rg.drop_scope(store, base, sc, exporter=exporter)
            print(f"Dropped {sc['name']}: {res['files']:,} images deleted, {res['photos']:,} photos marked not "
                  f"downloaded")
        if (base / layout.STORAGE_DIR).is_dir():            # the scope's raw files went with its paths
            import folders
            folders.prune_empty(base / layout.STORAGE_DIR)
    return 0


def quality_of(conf: cf.Conf) -> float:
    """[defaults] quality_threshold of the conf; written into the file with its default when it is not there."""
    q = conf.quality_threshold
    if not 0 <= q <= 1:
        raise SystemExit(f"{conf.path}: [defaults] quality_threshold {q}: give a P(bad) from 0 (off) to 1")
    if not conf.quality_in_file and conf.path and Path(conf.path).is_file():
        cf.save_default(conf.path, "quality_threshold", f"{q:g}", cf.QUALITY_COMMENT)
        conf.quality_in_file = True
        print(f"  quality_threshold = {q:g} written into [defaults] of {conf.path}")
    return q


def quality_niqe_of(conf: cf.Conf) -> float:
    """[defaults] quality_threshold_niqe of the conf; written into the file with its default when it is not there."""
    q = conf.quality_threshold_niqe
    if not 0 <= q <= 1:
        raise SystemExit(f"{conf.path}: [defaults] quality_threshold_niqe {q}: give a P(bad) from 0 (off) to 1")
    if not conf.quality_niqe_in_file and conf.path and Path(conf.path).is_file():
        cf.save_default(conf.path, "quality_threshold_niqe", f"{q:g}", cf.QUALITY_NIQE_COMMENT)
        conf.quality_niqe_in_file = True
        print(f"  quality_threshold_niqe = {q:g} written into [defaults] of {conf.path}")
    return q


def aspect_of(conf: cf.Conf) -> float:
    """[defaults] max_aspect of the conf; written into the file with its default when it is not there."""
    if not conf.aspect_in_file and conf.path and Path(conf.path).is_file():
        cf.save_default(conf.path, "max_aspect", "21:9", cf.ASPECT_COMMENT)
        conf.aspect_in_file = True
        print(f"  max_aspect = 21:9 written into [defaults] of {conf.path}")
    return conf.max_aspect


def pixels_of(conf: cf.Conf, given: int | None = None) -> int:
    """--max-pixels, else [defaults] max_pixels of the conf, written into the file with its
    default when it is not there. The side of the square whose area is the limit; 0: never."""
    if given is not None:
        if given < 0:
            raise SystemExit("--max-pixels takes the side of the square (2048), or 0 for never")
        return given
    if not conf.pixels_in_file and conf.path and Path(conf.path).is_file():
        cf.save_default(conf.path, "max_pixels", str(cf.Conf.max_pixels), cf.PIXELS_COMMENT)
        conf.pixels_in_file = True
        print(f"  max_pixels = {cf.Conf.max_pixels} written into [defaults] of {conf.path}")
    return conf.max_pixels


def qualities_of(conf: cf.Conf) -> dict:
    """[defaults] jpeg_quality and jpeg_quality_downscaled of the conf, as Options keywords;
    a key the file lacks is written into it with its default."""
    keys = ("jpeg_quality", "jpeg_quality_downscaled")
    missing = [k for k in keys if not getattr(conf, k + "_in_file")]
    if missing and conf.path and Path(conf.path).is_file():
        for i, key in enumerate(missing):
            value = getattr(cf.Conf, key)
            cf.save_default(conf.path, key, str(value), cf.JPEG_COMMENT if i == 0 else [])
            setattr(conf, key + "_in_file", True)
            print(f"  {key} = {value} written into [defaults] of {conf.path}")
    return {k: getattr(conf, k) for k in keys}


def export_layout_of(conf: cf.Conf) -> str:
    """[defaults] export_layout of the conf; written into the file with its default when it is not there."""
    if not conf.export_layout_in_file and conf.path and Path(conf.path).is_file():
        cf.save_default(conf.path, "export_layout", conf.export_layout, cf.EXPORT_LAYOUT_COMMENT)
        conf.export_layout_in_file = True
        print(f"  export_layout = {conf.export_layout} written into [defaults] of {conf.path}")
    return conf.export_layout


def top_order(top: str | None) -> str | None:
    """--top of an import -> the order of its tree (None: none, or not given)."""
    return {"color": layout.COLOR_FIRST, "category": layout.CATEGORY_FIRST}.get(top)


def import_order(store: Store, conf: cf.Conf) -> str:
    """The order of an import's tree: the one its --top named, else export_layout of the conf."""
    return top_order(store.get_meta("top")) or export_layout_of(conf)


def import_order_source(store: Store) -> str:
    top = store.get_meta("top")
    return f"--top {top}" if top_order(top) else "export_layout of the conf"


def quality_for(store: Store, conf: cf.Conf, given: float | None = None) -> float:
    """The quality threshold of a base: the one given (an import remembers it), else the
    one the import was given before, else [defaults] quality_threshold of the conf."""
    if given is not None:
        if not 0 <= given <= 1:
            raise SystemExit(f"--quality-threshold {given}: give a P(bad) from 0 (off) to 1")
        if store.is_import:
            store.set_meta("quality_threshold", given)
        return given
    q = store.get_meta("quality_threshold") if store.is_import else None
    return quality_of(conf) if q is None else float(q)


def quality_niqe_for(store: Store, conf: cf.Conf, given: float | None = None) -> float:
    """The NIQE stack's threshold of a base, remembered by an import like quality_for's."""
    if given is not None:
        if not 0 <= given <= 1:
            raise SystemExit(f"--quality-threshold-niqe {given}: give a P(bad) from 0 (off) to 1")
        if store.is_import:
            store.set_meta("quality_threshold_niqe", given)
        return given
    q = store.get_meta("quality_threshold_niqe") if store.is_import else None
    return quality_niqe_of(conf) if q is None else float(q)


def margin_of(args, conf: cf.Conf) -> float:
    m = args.priority_margin if getattr(args, "priority_margin", None) is not None else conf.priority_margin
    if not 0 <= m < 1:
        raise SystemExit(f"priority margin {m}: give a value from 0 (off) to below 1")
    return m


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

    countries = []
    for c in USSR:
        if c == 20:
            for n, cids, comment in UKRAINE_PRESETS:
                taken[n] = -1
                countries.append(line(n, cids, comment))
            continue
        comment = desc(by[c])
        if c == 1:
            comment += "; in PastVu's tree this contains Crimea and Sevastopol (see the note above)"
        countries.append(line(name_for(by[c]), [c], comment))
    countries += [line(g, cids, ", ".join(by[c]["title_en"] for c in cids)) for g, cids in GROUPS.items()]
    region_lines, city_lines = [], []

    def block(header: str, first: list[dict], seconds: list[dict]) -> None:
        regs = [r for r in first if not is_city(r)]
        cities = [r for r in first if is_city(r)]
        if regs:
            region_lines.append(f"; {header}")
            region_lines.extend(line(name_for(r), [r["cid"]], desc(r)) for r in regs)
        if cities or seconds:
            city_lines.append(f"; {header}")
            city_lines.extend(line(name_for(r), [r["cid"]], desc(r)) for r in cities)
            city_lines.extend(line(name_for(r, by.get(r["parents"][-1])), [r["cid"]],
                                   f"{desc(r)}, in {by[r['parents'][-1]]['title_en']}") for r in seconds)

    def seconds_of(test) -> list[dict]:
        return sorted([r for r in regions if len(r["parents"]) == 2 and test(r) and r["phc"] >= CITY_MIN_PHOTOS],
                      key=lambda r: -r["phc"])

    for c in USSR:
        first = sorted([r for r in regions if r["parents"] == [c] and r["cid"] not in CRIMEA], key=lambda r: -r["phc"])
        block(by[c]["title_en"], first,
              seconds_of(lambda r, c=c: r["parents"][0] == c and r["parents"][1] not in CRIMEA))
        if c == 20:
            block("Crimea and Sevastopol (see the note above)",
                  sorted([by[x] for x in CRIMEA if x in by], key=lambda r: -r["phc"]),
                  seconds_of(lambda r: r["parents"][1] in CRIMEA))
    body = ["[aliases.countries]", *countries, "", "[aliases.regions]", *region_lines, "", "[aliases.cities]",
            *city_lines]
    return CONF_HEAD + "\n".join(body) + "\n"


CONF_HEAD = """\
; pastvu configuration. install.bat copies this file to pastvu.conf (next to
; pastvu.py), which is yours to edit. This example is made by
;   run.bat aliases --generate pastvu.conf.example
; Relative paths are resolved against the folder of this file.

[paths]
; The base folder of the downloads. Each scope gets a folder in it (see layout
; below), and the store pastvu.db sits in it.
base = D:\\pastvu

; The classify samples folder: one subfolder of example images per category.
samples = D:\\samples

[defaults]
; The folder tree of a base folder: years/place puts a scope into
; <base>\\1970s\\russia\\, place/years into <base>\\russia\\1970s\\, flat into
; <base>\\russia_1970s\\. A base keeps the tree it was made with.
layout = years/place
; Years of a run without --years or --decade (PastVu's overlap rule).
years = 1917 1991
; At most this many images per folder (each with its caption, so twice as many
; files): a category holds its images directly until it would pass this, then
; they move into the numbered folder 001 and the next go into 002, 003, ...;
; 0 puts every image of a category into one folder.
split = 5000
; Download threads. 8 is faster and doubles the load on PastVu.
fetchers = 4
; Also write each record as a JSON file in <base>\\_records\\<number>\\. The store
; holds every record anyway, and `export` writes them as JSON Lines or Parquet.
json_files = no
; The numbers of the samples folders are a priority: when categories score
; within this margin of the top one, the lowest-numbered of them wins. 0 turns
; it off (the top score always wins).
priority_margin = 0.15
; Photos the quality probe (trained on _quality_good and _quality_bad in the
; samples folder) scores at or above this P(bad) are skipped and never saved,
; and reclassify deletes the saved ones. 0 turns it off.
quality_threshold = 0.88
; The same for photos with a NIQE score (all new ones; `niqe` scores the older
; ones): the probe stacked with NIQE above 12, where print screens and noise
; survive the downscale to 1024² pixels and film grain does not reach. Photos
; without a NIQE score keep quality_threshold. 0: the probe alone for all.
quality_threshold_niqe = 0.8
; Photos whose long side is more than this times the short side (panoramas,
; strips) are skipped like the ones under 512 pixels, and reclassify deletes
; the saved ones. W:H or a number; 0 turns it off.
max_aspect = 21:9
; A photo whose picture (without the PastVu band) has more pixels than this
; number squared is downscaled to that area before it is processed and is saved
; from the downscaled pixels (JPEG at jpeg_quality_downscaled, or PNG with --png). Training needs no
; more than 2048 x 2048. 0: never downscale. reprocess --downscale does the same
; to photos saved before.
max_pixels = 2048
; The JPEG quality (1 to 100, saved without chroma subsampling) of a photo
; re-encoded at its own size (a painted watermark), and of a photo re-encoded
; while it is downscaled to max_pixels (run, pull, import, reprocess). A
; downscaled photo is still larger than training needs, so it can take a little
; less. Untouched photos and lossless crops keep the downloaded bytes.
jpeg_quality = 97
jpeg_quality_downscaled = 95
; The folder order of the live export (run --export) and of an import:
; categories_first (<category>\\bw, <category>\\color) or color_first
; (bw\\<category>, color\\<category>). A change reorders an existing export
; at its next run --export or reclassify, and an import at its next reclassify.
export_layout = categories_first

[caption]
; autocaption: a vision-language model writes each caption from the image and
; its facts line, through an OpenAI-compatible chat endpoint (KoboldCpp, the
; llama.cpp server, LM Studio). The prompts are files; {tags} in the main prompt
; is the existing caption.
api_url = http://localhost:5001/v1
; Empty: whatever model the endpoint has loaded.
model =
api_key =
system_prompt = tagging/sysprompt.txt
prompt = tagging/main_prompt.txt
max_tokens = 400
temperature = 0.3
top_p = 0.95
; The longest side of the image sent to the model, in pixels.
image_side = 1024
; Seconds to wait for one caption.
timeout = 300
; Let a reasoning model think before it answers (slower).
reasoning = no

[interest]
; cherrypick --interest: the saved photos are ordered by what detect found in
; them before the quality ceilings cut each category. A rule per category reads
; the boxes row (sizes as shares of the frame); a photo whose rule passes comes
; first, then one without a row, then one whose rule fails; nothing is removed.
; The rule of a category comes from its name after the number: transport ->
; vehicle; people_adult, people_children -> people; group_photo_adult,
; group_photo_children -> group; futurism, objects, military -> thing;
; environment_indoors -> life; environment_outdoors -> street; school and every
; other name -> pass (no ordering). rule.<name> = <rule> overrides one category, e.g.
;   rule.monuments = thing
; vehicle: the largest vehicle covers between vehicle_min and vehicle_max of the
; frame and is at least vehicle_width of its width.
vehicle_min = 0.06
vehicle_max = 0.70
vehicle_width = 0.30
; people (bodies: clothes and activity): the tallest person is at least
; person_height of the frame's height, or people_pair persons are each at least
; 0.20 tall (0: not this way); face_height above 0 also asks for a face that tall.
person_height = 0.35
people_pair = 2
face_height = 0
; group: at least group_min persons, the tallest face at least group_face_height.
group_min = 4
group_face_height = 0.03
; thing: the largest vehicle or object covers at least thing_min of the frame.
thing_min = 0.08
; presence: at least presence_persons persons with presence_faces faces inside
; them (a street or an interior with people before one without).
presence_persons = 1
presence_faces = 1
; life: at least life_persons persons 0.08 tall or more, or life_vehicles
; vehicles (0: not this way): an interior with someone in it before an empty one.
life_persons = 1
life_vehicles = 1
; street: at least street_persons persons 0.08 tall or more, or street_crowd
; persons of any size, or a vehicle covering street_vehicle_area of the frame
; (0: not this way): passers-by, a parade or a tram before a facade with one
; pedestrian or a car parked far off.
street_persons = 2
street_crowd = 5
street_vehicle_area = 0.02

[cherrypick]
; cherrypick per category: a line per category, by its folder name with or
; without the number, or a pattern (group_photo_*). A line takes the values of
; the command line, in any order: a threshold with a dot (0.85), a percentile
; from 1 to 100 (30), a number per category over 100 (5000; a smaller number is
; --per-category 50), and --color PERCENT, the share of colour photos (the
; --balance of that category; a number per category alone means 50). 0 leaves
; the category out. A line is the whole setting of its category. default is the
; line of every category without one; values on the command line replace it for
; that run. 200-outliers is left out unless a line names it. With lines here,
; cherrypick runs without values: run.bat cherrypick -o D:\\training
;default = 0.85 30
;churches = 0
;aerial = 0.85 15
;environment_outdoors = 0.85 12000
;school = 0.5
;group_photo_* = 1000 0.5 --color 100

; Aliases: a name for one or more PastVu region numbers, used with --region.
; Three groups: countries (and groups of countries), regions (the first level
; under each country), cities and smaller parts (capitals and federal cities,
; then places of the second level with at least 1,000 photos). The names come
; from the English titles; each comment gives the title and the photo count at
; the time this file was made.
;
; Where the aliases leave PastVu's region tree. PastVu's tree files Crimea (556)
; and Sevastopol (669) under Russia and marks them as a territorial dispute; that
; is PastVu's wording, kept in the titles below. In this file they are listed
; apart, and three presets cover Ukraine:
;   ukraine          = 20, 556, 669   the modern state of Ukraine in its
;                                     internationally recognized borders
;   ukrainian_ssr    = 20, 556, 669   the Ukrainian SSR from 1954 to 1991
;   ukraine_pre_1954 = 20             the Ukrainian SSR before 1954
; The Crimean Oblast passed from the RSFSR to the Ukrainian SSR by a decree of
; the Presidium of the Supreme Soviet of the USSR of 19 February 1954. Before
; that, Crimea was part of the RSFSR: the Crimean ASSR from 1921, the Crimean
; Oblast from 1945. Sevastopol had been made a city of republican subordination
; of the RSFSR in 1948. The 1954 decree does not name it, but the city was
; administered through the Ukrainian SSR from then on, and the 1978 constitution
; of the Ukrainian SSR lists it among the republic's cities of republican
; subordination.
; Neither Soviet preset is exact for every year. ukraine_pre_1954 still contains
; the regions that joined the Ukrainian SSR in 1939, 1940 and 1945 (western
; Ukraine, northern Bukovina and southern Bessarabia, Zakarpattia), which between
; the wars belonged to Poland, Romania and Czechoslovakia. For the Ukrainian SSR
; as it was when each photo was taken, run ukraine_pre_1954 with --years 1917
; 1953 and ukrainian_ssr with --years 1954 1991.
;
; russia = 1 follows PastVu's tree and so also contains Crimea and Sevastopol. A
; photo is saved once per base folder: a Crimean photo is downloaded by whichever
; of the two scopes runs first, and the other skips it. Abkhazia and South
; Ossetia are filed under Georgia and Transnistria under Moldova, with the same
; mark; the aliases of those countries contain them. Change any alias to fit
; your dataset.

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
        p.add_argument("--by", metavar="RULE",
                       help="split into scopes by decade, country or region, or country,decade / region,decade "
                            "(a scope per value, made as photos turn up); a years/place or place/years base always "
                            "splits a span of years by decade")
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

    def processing_opts(p):
        """The processing (the models, the crop, the writer) and what follows it: run, run --pull and pull."""
        p.add_argument("--samples", metavar="DIR", help="classify samples folder (default: [paths] samples of the conf)")
        p.add_argument("--no-classify", action="store_true",
                       help="no categories: images go into numbered folders of the scope")
        p.add_argument("--min-confidence", type=float, default=0.7, metavar="X",
                       help="below this the record marks the photo as doubtful; it still goes to its best-guess category")
        p.add_argument("--priority-margin", type=float, metavar="X",
                       help="a category within X of the top score wins when its number is lower (default: [defaults] "
                            "priority_margin of the conf, 0.15; 0: the top score always wins)")
        p.add_argument("--patches", type=int, default=576, metavar="N", help="encoder patch budget (default 576)")
        p.add_argument("--max-pixels", type=int, metavar="N",
                       help="downscale a picture over N x N pixels to that area before processing it and save it "
                            "from the pixels; 0: never (default: [defaults] max_pixels of the conf, 2048)")
        p.add_argument("--png", action="store_true",
                       help="write painted and downscaled images as PNG instead of JPEG (jpeg_quality, jpeg_quality_downscaled)")
        p.add_argument("--no-watermark", action="store_true", help="no watermark detection or painting")
        p.add_argument("--wm-conf", type=float, default=0.1, metavar="X",
                       help="watermark detections from this confidence go into the record (default 0.1)")
        p.add_argument("--wm-paint-conf", type=float, default=0.4, metavar="X",
                       help="only detections from this confidence are painted out (default 0.4)")
        p.add_argument("--wm-mode", choices=["inpaint", "trim"], default="inpaint",
                       help="paint watermarks out (default) or cut the largest clear rectangle when it keeps half "
                            "the image")
        p.add_argument("--no-borders", action="store_true", help="no border analysis (the PastVu band is still cut)")
        p.add_argument("--no-dark", action="store_true", help="leave dark borders alone")
        p.add_argument("--txt-fields", metavar="LIST",
                       help=f"caption parts in order, from {','.join(sidecars.ALL_TXT_FIELDS)} "
                            f"(default {','.join(sidecars.TXT_FIELDS)})")
        p.add_argument("--json-sidecar", action="store_true", help="also write the record next to the image")
        p.add_argument("--depth", type=int, default=16, metavar="N", help="images fetched ahead of the GPU (default 16)")
        p.add_argument("--writers", type=int, default=4, metavar="N",
                       help="threads that test the colour, encode and write (default 4)")
        p.add_argument("--keep-rejects", action="store_true",
                       help="a stored photo the processing rejects keeps its raw files in the storage (default: "
                            "deleted, since nothing fetches it again)")
        p.add_argument("--storage-threads", type=int, default=6, metavar="N",
                       help="threads that read, decode and prepare stored photos ahead of the GPU (default 6)")
        p.add_argument("--prepare-workers", type=int, metavar="N",
                       help="processes that decode, plan the borders and make the models' inputs, so the GPU thread "
                            "never waits behind them for the GIL (default: half the logical processors, at most 8; "
                            "0: in the download and storage threads)")
        p.add_argument("--export", action="store_true",
                       help="hard-link every saved photo into <base parent>/pastvu_export/<category>/<bw|color>/<NNN> "
                            "as it is written (photos saved before are linked at the start)")
        p.add_argument("--export-reset", action="store_true",
                       help="with --export: forget the images you deleted from the export and link them again")
        p.add_argument("--autocaption", action="store_true",
                       help="caption each saved photo with the VLM of [caption] in the conf (the run slows to its pace)")

    def pace_opts(p):
        p.add_argument("--limit", type=int, metavar="N", help="do at most N photos in this run")
        p.add_argument("--retries", type=int, default=3, metavar="N", help="attempts for a photo that failed (default 3)")
        p.add_argument("--report-every", type=int, default=500, metavar="N", help="a report block every N photos")
        p.add_argument("--quiet", action="store_true", help="report blocks only, no live display")
        p.add_argument("--json-progress", action="store_true", help="one JSON line per photo on stdout")

    p = run_p = sub.add_parser("run", help="index (with --region) and download, process, sort and write a scope; "
                                           "--raw downloads only, --pull also processes the storage")
    scope_opts(p, False)
    p.add_argument("--scope", metavar="NAME", help="resume a scope by its name or folder")
    p.add_argument("--resume", action="store_true",
                   help="skip the index (no requests to plan or page it) and download what is listed already")
    p.add_argument("--raw", action="store_true",
                   help="download the originals into <base>/_storage with their records and nothing else: no model "
                        "is loaded, no GPU is used; pull, or run --pull, processes them later")
    p.add_argument("--pull", action="store_true",
                   help="also process the photos in the storage: one whenever no download is ready for the GPU, "
                        "the rest at full speed when the downloads are finished (the whole storage, every scope)")
    p.add_argument("--regroup", action="store_true",
                   help="first move the saved photos the run's scopes list into them, from whatever scope holds them")
    p.add_argument("--min-side", type=int, default=512, metavar="PX",
                   help="skip images whose picture is smaller than this on either side (default 512)")
    p.add_argument("--comments", action="store_true", help="add the comment tree to the record (one more request)")
    p.add_argument("--fetchers", type=int, metavar="N", help="download threads (default: [defaults] fetchers of the conf)")
    p.add_argument("--read-timeout", type=float, default=READ_TIMEOUT, metavar="S",
                   help=f"give up a request that sends nothing for S seconds; the photo is tried again later "
                        f"(default {READ_TIMEOUT:.0f})")
    p.add_argument("--image-deadline", type=float, default=IMAGE_DEADLINE, metavar="S",
                   help=f"give up an image download that takes more than S seconds in all; 0: no limit "
                        f"(default {IMAGE_DEADLINE:.0f})")
    p.add_argument("--skip", type=int, default=0, metavar="N",
                   help="start at photo N+1 in PastVu number order: the first N are left out, not downloaded, and "
                        "stay to do for a run without --skip")
    processing_opts(p)
    pace_opts(p)

    p = sub.add_parser("pull", help="process the photos in <base>/_storage (run --raw) into their scopes, no downloads")
    p.add_argument("scope", nargs="*", metavar="SCOPE",
                   help="only the stored photos of these scopes (names, folders or patterns; default: the whole storage)")
    p.add_argument("--min-side", type=int, default=512, metavar="PX",
                   help="skip images whose picture is smaller than this on either side after the crop (default 512)")
    processing_opts(p)
    pace_opts(p)

    p = sub.add_parser("import", help="process a local dataset (not from PastVu) like a download, into a folder of "
                                      "its own; the source is only read")
    p.add_argument("--dataset", required=True, metavar="DIR", help="the source folder (read only)")
    p.add_argument("-o", metavar="DIR",
                   help="the output folder: the imported dataset itself (default: <source>_pastvu next to it)")
    p.add_argument("--against", metavar="BASE",
                   help="a PastVu base folder: photos that are copies of its photos are skipped, and reclassify of "
                        "the import deletes the copies found later (remembered; give it once)")
    p.add_argument("--split", type=int, default=0, metavar="N",
                   help="numbered folders of at most N images (default 0: none); set when the import is made")
    p.add_argument("--top", choices=IMPORT_TOPS,
                   help="the top of the tree: color (bw/<category>), category (<category>/bw) or none (<category>, "
                        "no black-and-white or colour detection). Default: bw_color and export_layout of the conf. "
                        "Remembered; color and category can be swapped later, none cannot be changed")
    p.add_argument("--quality-threshold", type=float, metavar="X",
                   help="skip photos the quality probe scores at or above this P(bad); 0: off (default: [defaults] "
                        "quality_threshold of the conf). Remembered, also by reclassify of the import")
    p.add_argument("--quality-threshold-niqe", type=float, metavar="X",
                   help="the same for the probe stacked with NIQE, which decides for every photo with a NIQE score; "
                        "0: the probe alone (default: [defaults] quality_threshold_niqe of the conf). Remembered")
    p.add_argument("--samples", metavar="DIR", help="classify samples folder (default: [paths] samples of the conf)")
    p.add_argument("--min-confidence", type=float, default=0.7, metavar="X",
                   help="below this the record marks the photo as doubtful; it still goes to its best-guess category")
    p.add_argument("--priority-margin", type=float, metavar="X",
                   help="a category within X of the top score wins when its number is lower (default: [defaults] priority_margin of the conf, 0.15; 0: the top score always wins)")
    p.add_argument("--patches", type=int, default=576, metavar="N", help="encoder patch budget (default 576)")
    p.add_argument("--min-side", type=int, default=512, metavar="PX",
                   help="skip images whose picture is smaller than this on either side (default 512)")
    p.add_argument("--png", action="store_true", help="write images with painted watermarks as PNG instead of JPEG at jpeg_quality")
    p.add_argument("--no-watermark", action="store_true", help="no watermark detection or painting")
    p.add_argument("--wm-conf", type=float, default=0.1, metavar="X",
                   help="watermark detections from this confidence go into the record (default 0.1)")
    p.add_argument("--wm-paint-conf", type=float, default=0.4, metavar="X",
                   help="only detections from this confidence are painted out (default 0.4)")
    p.add_argument("--wm-mode", choices=["inpaint", "trim"], default="inpaint",
                   help="paint watermarks out (default) or cut the largest clear rectangle when it keeps half the image")
    p.add_argument("--no-borders", action="store_true", help="no border analysis")
    p.add_argument("--no-dark", action="store_true", help="leave dark borders alone")
    p.add_argument("--threads", type=int, default=4, metavar="N", help="reading and decoding threads (default 4)")
    p.add_argument("--depth", type=int, default=16, metavar="N", help="images read ahead of the GPU (default 16)")
    p.add_argument("--writers", type=int, default=2, metavar="N", help="encoder and writer threads (default 2)")
    p.add_argument("--limit", type=int, metavar="N", help="do at most N images in this run")
    p.add_argument("--retries", type=int, default=3, metavar="N", help="attempts for an image that failed (default 3)")
    p.add_argument("--report-every", type=int, default=500, metavar="N", help="a report block every N images")
    p.add_argument("--quiet", action="store_true", help="report blocks only, no live display")
    p.add_argument("--json-progress", action="store_true", help="one JSON line per image on stdout")

    p = sub.add_parser("reprocess", help="rewrite saved photos: --downscale [N] brings every photo over N x N pixels "
                                         "down to that area (2048 assumed)")
    p.add_argument("scope", nargs="*", metavar="SCOPE",
                   help="only these scopes (names, folders or patterns; default: the whole base)")
    p.add_argument("--downscale", type=int, nargs="?", const=2048, metavar="N",
                   help="downscale the saved photos over N x N pixels to that area, in place (N: the side of the "
                        "square, 2048 when given alone)")
    p.add_argument("--png", action="store_true", help="write the downscaled photos as PNG (default: JPEG at jpeg_quality_downscaled; a PNG stays PNG)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2), metavar="N",
                   help="worker processes that read, resize and encode (default: half the CPUs)")
    p.add_argument("--dry-run", action="store_true", help="count and list the photos in <base>/_reports, change nothing")
    p.add_argument("--yes", action="store_true", help="do not ask before rewriting the photos")
    p.add_argument("--quiet", action="store_true", help="no live display")
    p.add_argument("--json-progress", action="store_true", help="JSON lines on stdout")

    p = sub.add_parser("captions", help="rewrite the captions of a scope from the records, no downloads")
    p.add_argument("scope", nargs="+", metavar="SCOPE", help="a scope's name, folder or name pattern (estonia_*)")
    p.add_argument("--txt-fields", metavar="LIST", help="caption parts in order (see run --help)")
    p.add_argument("--dry-run", action="store_true", help="count and show the changes, write nothing")
    p.add_argument("--show", type=int, default=5, metavar="N", help="print the first N changes (default 5)")

    p = sub.add_parser("reclassify", help="sort the saved photos of a scope again with the current samples folder")
    p.add_argument("scope", nargs="*", metavar="SCOPE",
                   help="a scope's name, folder or name pattern (estonia_*); none: every scope of the base (asks first)")
    p.add_argument("--yes", action="store_true",
                   help="do not ask before working on every scope of the base or deleting low-quality photos")
    p.add_argument("--samples", metavar="DIR", help="classify samples folder (default: [paths] samples of the conf)")
    p.add_argument("--min-confidence", type=float, default=0.7, metavar="X",
                   help="below this the record marks the photo as doubtful; it still goes to its best-guess category")
    p.add_argument("--priority-margin", type=float, metavar="X",
                   help="a category within X of the top score wins when its number is lower (default: [defaults] priority_margin of the conf, 0.15; 0: the top score always wins)")
    p.add_argument("--patches", type=int, default=576, metavar="N", help="encoder patch budget (default 576)")
    p.add_argument("--dry-run", action="store_true", help="report the moves and write them to <base>/_reports, move nothing")
    p.add_argument("--override-manual", action="store_true",
                   help="also sort the photos that were moved to another category by hand (default: they stay)")
    p.add_argument("--undo", action="store_true", help="put back the moves of the last reclassify of the scope")
    p.add_argument("--unsure", action="store_true",
                   help="only move the photos in _unsure folders (made by older versions) into their best-guess "
                        "categories; no classifier")

    p = sub.add_parser("cleanup", help="move the saved photos the quality probe scores at or above a threshold into "
                                       "<base>_review for you to look at; --done or --undo puts back what you kept")
    p.add_argument("threshold", nargs="?", metavar="P", help="P(bad) from which a photo goes to review, e.g. 0.8 "
                                                            "(for this command only; the conf keeps its value)")
    p.add_argument("scope", nargs="*", metavar="SCOPE",
                   help="a scope's name, folder or name pattern (estonia_*); none: every scope of the base")
    p.add_argument("--done", action="store_true",
                   help="settle the review: the photos left in the folder go back and are kept for good")
    p.add_argument("--undo", action="store_true",
                   help="settle the review: the photos left in the folder go back as they were, not marked")
    p.add_argument("--yes", action="store_true", help="do not ask before moving photos into the review folder")
    p.add_argument("--dry-run", action="store_true", help="count and list the photos to review, move nothing")
    p.add_argument("--override-manual", action="store_true",
                   help="also review the photos that were moved to another category by hand (default: they stay)")
    p.add_argument("--samples", metavar="DIR", help="classify samples folder (default: [paths] samples of the conf)")
    p.add_argument("--patches", type=int, default=576, metavar="N", help="encoder patch budget (default 576)")

    p = sub.add_parser("autocaption", help="caption saved photos with a VLM, from each image and its caption")
    p.add_argument("target", nargs="+", metavar="SCOPE|FOLDER",
                   help="a scope's name or name pattern (estonia_*), or its folder or any folder inside it")
    p.add_argument("--recaption", action="store_true",
                   help="caption again the photos that have a VLM caption (from their original caption)")
    p.add_argument("--undo", action="store_true", help="put back the caption files of the last autocaption of the scope")
    p.add_argument("--limit", type=int, metavar="N", help="caption at most N photos in this run")
    p.add_argument("--retries", type=int, default=3, metavar="N", help="attempts for a photo that failed (default 3)")
    p.add_argument("--report-every", type=int, default=100, metavar="N", help="a report block every N photos")
    p.add_argument("--quiet", action="store_true", help="report blocks only, no live display")
    p.add_argument("--json-progress", action="store_true", help="one JSON line per photo on stdout")

    p = sub.add_parser("niqe", help="NIQE of the saved photos at the training resolution, for the quality rule "
                                     "(beside a download)")
    p.add_argument("target", nargs="*", metavar="SCOPE|FOLDER",
                   help="a scope's name or name pattern, or its folder or any folder inside it (default: every scope)")
    p.add_argument("--follow", action="store_true",
                   help="when done, look for newly saved photos without NIQE every minute until Ctrl+C")
    p.add_argument("--redo", action="store_true", help="score every saved photo again, also those with a row")
    p.add_argument("--limit", type=int, metavar="N", help="score at most N photos in this run")
    p.add_argument("--threads", type=int, default=12, metavar="N", help="reading and decoding threads (default 12)")
    p.add_argument("--report-every", type=int, default=5000, metavar="N", help="a report block every N photos")
    p.add_argument("--quiet", action="store_true", help="report blocks only, no live display")
    p.add_argument("--json-progress", action="store_true", help="one JSON line per photo on stdout")

    p = sub.add_parser("deduplicate", help="copies of one photo inside the base, by their hashes and embeddings "
                                            "(the hash pass runs beside a download)")
    p.add_argument("target", nargs="*", metavar="SCOPE|FOLDER",
                   help="a scope's name or name pattern, or its folder or any folder inside it (default: every scope)")
    p.add_argument("--hashes-only", action="store_true", help="hash the saved photos that have no hashes, nothing else")
    p.add_argument("--dry-run", action="store_true",
                   help="find the copies and write the report to <base>/_reports, move nothing")
    p.add_argument("--review", type=int, nargs="?", const=0, metavar="N",
                   help="hard links of the groups next to the base (<base>_duplicates_review): N random groups "
                        "per rule and N pairs of the feature window, or all without N")
    p.add_argument("--min-cos", type=float, default=0.90, metavar="X",
                   help="candidate pairs at this cosine or more (default 0.90)")
    p.add_argument("--neighbours", type=int, default=10, metavar="N", help="nearest photos per photo (default 10)")
    p.add_argument("--features", choices=("off", "report", "copies"),
                   help="the feature test of the pairs in the hash window: off; report (tested and listed, never "
                        "copies; the default of --dry-run); copies (a pass makes a copy). Default off when moving")
    p.add_argument("--report", metavar="FILE",
                   help="move the groups of an earlier dry run (its deduplicate-<time>-groups.jsonl) instead of "
                        "finding them again; the pairs of the feature test count only with --features copies")
    p.add_argument("--undo", action="store_true", help="put back the copies of the last deduplicate run")
    p.add_argument("--patches", type=int, default=576, metavar="N", help="the embeddings of this patch budget (576)")
    p.add_argument("--redo", action="store_true", help="hash every saved photo again, also those with a row")
    p.add_argument("--limit", type=int, metavar="N", help="hash at most N photos in this run")
    p.add_argument("--threads", type=int, default=12, metavar="N",
                   help="reading, hashing, feature-test and scoring threads (default 12)")
    p.add_argument("--report-every", type=int, default=50000, metavar="N", help="a report block every N photos")
    p.add_argument("--quiet", action="store_true", help="report blocks only, no live display")
    p.add_argument("--json-progress", action="store_true", help="one JSON line per photo on stdout")

    p = sub.add_parser("detect", help="box statistics of the saved photos (persons, faces, vehicles, objects) for "
                                       "cherrypick --interest (beside a download)")
    p.add_argument("target", nargs="*", metavar="SCOPE|FOLDER",
                   help="a scope's name or name pattern, or its folder or any folder inside it (default: every scope)")
    p.add_argument("--follow", action="store_true",
                   help="when done, look for newly saved photos without boxes every 30 s until Ctrl+C")
    p.add_argument("--redo", action="store_true", help="box every saved photo again, also those with a row")
    p.add_argument("--limit", type=int, metavar="N", help="box at most N photos in this run")
    p.add_argument("--batch", type=int, default=16, metavar="N", help="photos per detector call (default 16)")
    p.add_argument("--imgsz", type=int, default=640, metavar="PX", help="the long side the detector sees (default 640)")
    p.add_argument("--report-every", type=int, default=500, metavar="N", help="a report block every N photos")
    p.add_argument("--quiet", action="store_true", help="report blocks only, no live display")
    p.add_argument("--json-progress", action="store_true", help="one JSON line per photo on stdout")

    p = sub.add_parser("export", help="write the records of a scope as JSON Lines or Parquet")
    p.add_argument("scope", nargs="+", metavar="SCOPE", help="a scope's name, folder or name pattern (estonia_*)")
    p.add_argument("--format", choices=["jsonl", "parquet"], default="jsonl")
    p.add_argument("--to", metavar="FILE", help="the output file, for one scope (default <base>/<scope>_records.<format>)")

    p = sub.add_parser("extract", help="copy the photos of a category into a flat folder for training (bw/ and "
                                        "color/ kept)")
    p.add_argument("category", nargs="+", metavar="CATEGORY",
                   help="a category by number (050 or 50), name (transport) or full name (050-transport)")
    p.add_argument("-o", "--out", metavar="DIR",
                   help="the output folder, for one category (default: <base parent>/<category>[_<filters>])")
    p.add_argument("--region", action="append", metavar="ALIAS|CID|TITLE",
                   help="only photos placed in this region (the photo's own region chain); repeat for several")
    when(p)
    p.add_argument("--scope", action="append", metavar="NAME|PATTERN", help="only these scopes (estonia_*)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--bw", action="store_true", help="only black-and-white photos")
    g.add_argument("--color", action="store_true", help="only colour photos")
    p.add_argument("--min-res", type=int, default=0, metavar="N",
                   help="only photos of at least N x N pixels, as area (512, 768, 1024, 1280, ...)")
    p.add_argument("--target", type=int, metavar="N",
                   help="pick N photos, spread over the decades in proportion to their photos, the largest first")
    p.add_argument("--colour-share", type=float, default=0.5, metavar="X",
                   help="with --target: the share of colour photos (default 0.5; a short side is filled from the "
                        "other)")
    p.add_argument("--split", type=int, default=0, metavar="N",
                   help="numbered folders of at most N images under bw/ and color/ (default 0: flat)")
    p.add_argument("--link", action="store_true",
                   help="hard-link the images instead of copying them (same drive; captions are still copied)")
    p.add_argument("--prune", action="store_true",
                   help="delete the files of earlier extracts into this folder that no longer match")
    p.add_argument("--dry-run", action="store_true", help="count, copy nothing")

    p = sub.add_parser("cherrypick", help="pick a training set by quality: per category and mode, a threshold "
                                          "(0.85), a top percentile (30) and/or a number per category (5000), "
                                          "balanced between colour and black-and-white")
    p.add_argument("value", nargs="*", metavar="VALUE",
                   help="a P(good) threshold with a dot (0.85), a top percentile from 1 to 100 (30), a number per "
                        "category over 100 (5000); any of them, in any order; none: the [cherrypick] lines of the conf, "
                        "whose default line they replace when given")
    p.add_argument("-o", "--out", required=True, metavar="DIR", help="the output folder (outside the base)")
    p.add_argument("--quality", type=float, metavar="P", help="the threshold, named")
    p.add_argument("--top", type=int, metavar="X", help="the percentile, named")
    p.add_argument("--per-category", type=int, metavar="N", help="the number per category, named (implies --balance)")
    p.add_argument("--balance", nargs="?", type=int, const=50, metavar="PERCENT",
                   help="the share of colour photos to aim at, per category, as far as the pools allow "
                        "(--balance alone: 50; always on, at 50, with a number per category)")
    p.add_argument("--decades", nargs="+", metavar="DECADE", help="only these decades, e.g. 1960s 1970s (default: all)")
    p.add_argument("--category", nargs="+", metavar="CATEGORY",
                   help="only these categories, by number, name or full name (default: all but 200-outliers)")
    p.add_argument("--interest", action="store_true",
                   help="order each category by what detect found (a subject at a readable size, people with "
                        "faces) before the ceilings cut it; the rules and thresholds are [interest] in the conf")
    p.add_argument("--link", action="store_true", help="hard-link the images instead of copying them")
    p.add_argument("--prune", action="store_true", help="delete the files of this base in the folder that are no "
                                                        "longer picked")
    p.add_argument("--split", type=int, default=0, metavar="N",
                   help="numbered folders of at most N images under each category and mode (default 0: none)")
    p.add_argument("--dry-run", action="store_true", help="count and report, copy nothing")
    p.add_argument("--samples", metavar="DIR", help="classify samples folder (default: [paths] samples of the conf)")
    p.add_argument("--patches", type=int, default=576, metavar="N", help="encoder patch budget (default 576)")

    p = sub.add_parser("regroup", help="move saved photos into scopes by decade, country or region, no downloads")
    p.add_argument("target", nargs="*", metavar="SCOPE",
                   help="scopes to split (names, folders or patterns); none: group every scope of the base")
    p.add_argument("--by", metavar="RULE", help="decade, country, region, country,decade or region,decade")
    p.add_argument("--dry-run", action="store_true", help="show what would move, move nothing")
    p.add_argument("--undo", action="store_true", help="put back the last regroup (or run --regroup) of the base")

    p = sub.add_parser("drop", help="delete the files of a scope and forget it; its photos count as not downloaded")
    p.add_argument("scope", nargs="+", metavar="SCOPE", help="a scope's name, folder or name pattern")
    p.add_argument("--yes", action="store_true", help="delete; without it, only show what would be deleted")
    args = ap.parse_args(argv)
    if args.cmd in ("niqe", "detect") and args.follow and (args.limit is not None or args.redo):
        ap.error("--follow waits for new photos; give --limit and --redo without it")
    if args.cmd == "run" and args.raw:
        if args.pull:
            ap.error("--raw downloads without processing and --pull processes the storage: give one of them")
        for dest in ("regroup", "export", "export_reset", "autocaption"):
            if getattr(args, dest):
                ap.error(f"--raw downloads into the storage without processing; --{dest.replace('_', '-')} belongs to "
                         f"run --pull or pull")
        for dest in RAW_REFUSES:
            if getattr(args, dest) != run_p.get_default(dest):
                ap.error(f"--raw downloads into the storage without processing; --{dest.replace('_', '-')} belongs to "
                         f"run --pull or pull")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    os.environ.setdefault("YOLO_OFFLINE", "1")
    conf = cf.load(Path(args.conf) if args.conf else None)
    fetchers = getattr(args, "fetchers", None) or conf.fetchers
    client = Client(max_per_second=args.max_per_second, connections=max(8, fetchers * 2),
                    read_timeout=getattr(args, "read_timeout", READ_TIMEOUT),
                    image_deadline=getattr(args, "image_deadline", IMAGE_DEADLINE))
    try:
        cmds = {"regions": cmd_regions, "aliases": cmd_aliases, "count": cmd_count, "scopes": cmd_scopes,
                "index": cmd_index, "run": cmd_run, "pull": cmd_pull, "reprocess": cmd_reprocess,
                "captions": cmd_captions,
                "reclassify": cmd_reclassify,
                "export": cmd_export, "autocaption": cmd_autocaption, "regroup": cmd_regroup, "drop": cmd_drop,
                "extract": cmd_extract, "import": cmd_import, "cleanup": cmd_cleanup,
                "cherrypick": cmd_cherrypick, "niqe": cmd_niqe, "detect": cmd_detect,
                "deduplicate": cmd_deduplicate}
        return cmds[args.cmd](args, conf, client)
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
