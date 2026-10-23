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
from client import Client, filter_string  # noqa: E402
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

def open_base(args, conf: cf.Conf) -> tuple[Path, Store]:
    base = Path(args.base).resolve() if getattr(args, "base", None) else conf.need_base()
    base.mkdir(parents=True, exist_ok=True)
    store = Store(base / DB_NAME)
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
        print(f"{sc['name']:36} {c['total']:>9,} listed  {c['done']:>9,} saved  {c['skipped']:>7,} skipped  "
              f"{c['failed']:>5,} failed  {c['elsewhere']:>7,} in other scopes  {c['pending']:>9,} left{modes}{where}")
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
    opts = sm.Options(base=base, scope=sc, scopes=scopes, min_side=args.min_side, png=args.png, wm_conf=args.wm_conf,
                      wm_paint_conf=args.wm_paint_conf, wm_mode=args.wm_mode, borders=not args.no_borders,
                      dark_borders=not args.no_dark, fetchers=args.fetchers or conf.fetchers, depth=args.depth,
                      writers=args.writers, report_every=args.report_every, limit=args.limit, retries=args.retries,
                      txt_fields=fields, json_files=conf.json_files, json_sidecar=args.json_sidecar,
                      comments=args.comments,
                      display="json" if args.json_progress else "quiet" if args.quiet else "live")

    caption_client = None
    if args.autocaption:
        caption_client = vlm_client(conf)
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
    if not args.no_classify:
        import classifier as clf
        samples = Path(args.samples).resolve() if args.samples else conf.samples
        classifier = clf.Classifier(samples, args.patches, args.min_confidence,
                                    priority_margin=margin_of(args, conf))
        classifier.embed(np.zeros((64, 64, 3), np.uint8))         # load the encoder now
        print(f"  classifier: {len(classifier.class_names)} classes from {samples}", flush=True)
        import colour
        mode_probe = colour.probe_from_samples(samples, classifier.sample_vectors)
    if torch.cuda.is_available():
        print(f"  loaded in {time.time() - t0:.0f} s; GPU memory in use {torch.cuda.memory_allocated() / 2**30:.1f} GB",
              flush=True)
    import encode
    encode.turbo()

    total = sum(store.scope_counts(name)["total"] for name in names)
    pairs = store.pending_pairs(names, args.retries)
    completed = total - len(pairs)
    if args.skip:
        n = len(pairs)
        pairs = skip_pairs(store, names, pairs, args.skip)
        print(f"  --skip {args.skip:,}: starting at photo {args.skip + 1:,} of {total:,}; {n - len(pairs):,} photos "
              f"before it that are not done are left for later", flush=True)
    if vram.ok:
        print(f"  {vram.text()}", flush=True)
    exporter = None
    if args.export or args.export_reset:
        import liveexport as lx
        exporter = lx.open_for_run(store, base, conf.split, reset=args.export_reset)
    else:
        import liveexport as lx
        if lx.Exporter.exists_for(store):
            print(f"  This base has an export ({lx.root_of(base)}); add --export to link the new photos into it too")
    s = sm.Stream(client, store, opts, classifier, detector, lama, caption_client, vram, mode_probe, exporter)
    res = s.run(pairs, total, completed)
    left = len(store.pending_pairs(names, args.retries))
    where = sc["name"] if len(names) == 1 else f"its {len(names)} scopes"
    print(f"This run: {res['done']:,} saved, {res['skipped']:,} skipped, {res['failed']:,} failed in "
          f"{res['seconds'] / 60:.1f} min. Left to do in {where}: {left:,}.")
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
        exporter = lx.Exporter(store, base, conf.split) if lx.Exporter.exists_for(store) else None
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
                                        priority_margin=margin_of(args, conf))
        for sc in scopes:
            if classifier is None:
                print(f"Moving the photos of {sc['name']} out of _unsure into their best-guess categories ...",
                      flush=True)
                p = rc.plan_unsure(store, base, sc)
            else:
                print(f"Sorting {sc['name']} again with {len(classifier.class_names)} classes from {samples} ...",
                      flush=True)
                p = rc.plan(store, base, sc, classifier, args.override_manual)
            print(rc.summary(p))
            if args.dry_run:
                report = base / "_reports" / f"{sc['name']}-reclassify-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
                report.parent.mkdir(parents=True, exist_ok=True)
                with open(report, "w", encoding="utf-8") as f:
                    for m in p["moves"]:
                        f.write(json.dumps(m, ensure_ascii=False) + "\n")
                print(f"Dry run: nothing moved. The planned moves are in {report}")
                continue
            res = rc.apply(store, base, sc, p, str(samples), exporter=exporter)
            print(f"Reclassify run {res['run']}: {res['moved']:,} photos moved, {res['failed']:,} not moved, "
                  f"{res['manual']:,} hand-sorted photos recorded where they are, {res['pruned']} empty folders "
                  f"removed. --undo puts the moves back.")
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


def cmd_regroup(args, conf, client) -> int:
    import partition as pt
    import regroup as rg
    base, store = open_base(args, conf)
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
        exporter = lx.Exporter(store, base, conf.split) if lx.Exporter.exists_for(store) else None
        for sc in scopes:
            res = rg.drop_scope(store, base, sc, exporter=exporter)
            print(f"Dropped {sc['name']}: {res['files']:,} images deleted, {res['photos']:,} photos marked not "
                  f"downloaded")
    return 0


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

    p = sub.add_parser("run", help="index (with --region) and download, process, sort and write a scope")
    scope_opts(p, False)
    p.add_argument("--scope", metavar="NAME", help="resume a scope by its name or folder")
    p.add_argument("--resume", action="store_true",
                   help="skip the index (no requests to plan or page it) and download what is listed already")
    p.add_argument("--export", action="store_true",
                   help="hard-link every saved photo into <base parent>/pastvu_export/<category>/<bw|color>/<NNN> "
                        "as it is written (photos saved before are linked at the start)")
    p.add_argument("--export-reset", action="store_true",
                   help="with --export: forget the images you deleted from the export and link them again")
    p.add_argument("--regroup", action="store_true",
                   help="first move the saved photos the run's scopes list into them, from whatever scope holds them")
    p.add_argument("--samples", metavar="DIR", help="classify samples folder (default: [paths] samples of the conf)")
    p.add_argument("--no-classify", action="store_true", help="no categories: images go into numbered folders of the scope")
    p.add_argument("--min-confidence", type=float, default=0.7, metavar="X",
                   help="below this the record marks the photo as doubtful; it still goes to its best-guess category")
    p.add_argument("--priority-margin", type=float, metavar="X",
                   help="a category within X of the top score wins when its number is lower (default: [defaults] priority_margin of the conf, 0.15; 0: the top score always wins)")
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
    p.add_argument("--skip", type=int, default=0, metavar="N",
                   help="start at photo N+1 in PastVu number order: the first N are left out, not downloaded, and "
                        "stay to do for a run without --skip")
    p.add_argument("--retries", type=int, default=3, metavar="N", help="attempts for a photo that failed (default 3)")
    p.add_argument("--report-every", type=int, default=500, metavar="N", help="a report block every N photos")
    p.add_argument("--quiet", action="store_true", help="report blocks only, no live display")
    p.add_argument("--json-progress", action="store_true", help="one JSON line per photo on stdout")
    p.add_argument("--autocaption", action="store_true",
                   help="caption each saved photo with the VLM of [caption] in the conf (downloads slow to its pace)")

    p = sub.add_parser("captions", help="rewrite the captions of a scope from the records, no downloads")
    p.add_argument("scope", nargs="+", metavar="SCOPE", help="a scope's name, folder or name pattern (estonia_*)")
    p.add_argument("--txt-fields", metavar="LIST", help="caption parts in order (see run --help)")
    p.add_argument("--dry-run", action="store_true", help="count and show the changes, write nothing")
    p.add_argument("--show", type=int, default=5, metavar="N", help="print the first N changes (default 5)")

    p = sub.add_parser("reclassify", help="sort the saved photos of a scope again with the current samples folder")
    p.add_argument("scope", nargs="*", metavar="SCOPE",
                   help="a scope's name, folder or name pattern (estonia_*); none: every scope of the base (asks first)")
    p.add_argument("--yes", action="store_true", help="do not ask before working on every scope of the base")
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

    p = sub.add_parser("regroup", help="move saved photos into scopes by decade, country or region, no downloads")
    p.add_argument("target", nargs="*", metavar="SCOPE",
                   help="scopes to split (names, folders or patterns); none: group every scope of the base")
    p.add_argument("--by", metavar="RULE", help="decade, country, region, country,decade or region,decade")
    p.add_argument("--dry-run", action="store_true", help="show what would move, move nothing")
    p.add_argument("--undo", action="store_true", help="put back the last regroup (or run --regroup) of the base")

    p = sub.add_parser("drop", help="delete the files of a scope and forget it; its photos count as not downloaded")
    p.add_argument("scope", nargs="+", metavar="SCOPE", help="a scope's name, folder or name pattern")
    p.add_argument("--yes", action="store_true", help="delete; without it, only show what would be deleted")
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
                "export": cmd_export, "autocaption": cmd_autocaption, "regroup": cmd_regroup, "drop": cmd_drop,
                "extract": cmd_extract}
        return cmds[args.cmd](args, conf, client)
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
