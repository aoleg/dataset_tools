"""The index of a scope: every photo PastVu lists for the scope's regions and
years, pulled through the gallery (photo.givePS) in pages of 100 into the
store, so the stream knows what to fetch and a resume knows what is left.

The gallery's year filter is an overlap (a photo dated 1965-1975 is in
1970-1979). A year range keeps that rule; a decade keeps the photos whose
start year lies in it, so every photo belongs to exactly one decade.

A gallery page costs the server a count over the region's photos, and a deep
skip grows with the page number, so a region with many photos is paged
through its children (the region tree) until each shard is under SHARD_MAX,
plus a shard of the rest of the parent: its photos in none of the children,
by the gallery's exclusion filter, which takes at most 10 regions. A parent with
more children gets a geo!0 shard (photos without coordinates) instead, and its
geotagged photos outside every child are reported and taken only with
--full-parent, which pages the parent itself.

The gallery pages by skipping, so page N makes the server read and drop N x
100 photos first: a long shard gets slower with every page. A shard of more
than BAND_MAX photos that is not finished is therefore paged in year bands,
the years halved until each band holds at most BAND_MAX photos (or one year).
The year filter is an overlap, so a photo dated across a band boundary is
listed in both bands, which the store takes once. Finished shards stay as
they are.
"""

from __future__ import annotations

import time

from client import Client, PastvuError, filter_string
from layout import slug
from store import Store

PAGE_SIZE = 100
SHARD_MAX = 50_000
MAX_EXCLUDE = 10           # the gallery filter takes at most 10 regions to exclude
BAND_MAX = 5_000           # a shard larger than this is paged in year bands (deep skips get slow)
REGIONS_MAX_AGE = 30 * 86400


def ensure_regions(client: Client, store: Store, refresh: bool = False) -> int:
    pulled = store.get_meta("regions_pulled", 0)
    if refresh or store.regions_count() == 0 or time.time() - pulled > REGIONS_MAX_AGE:
        store.put_regions(client.regions())
        store.set_meta("regions_pulled", time.time())
    return store.regions_count()


def region_path(store: Store, region: dict) -> str:
    names = []
    for p in region["parents"]:
        r = store.region(p)
        if r:
            names.append(r["title_en"])
    return " > ".join(names + [region["title_en"]])


def resolve(store: Store, aliases: dict[str, list[int]], specs: list[str]) -> tuple[list[dict], list[str]]:
    """Region specs -> (region rows, labels for the scope name). A spec is an
    alias of the conf, else a PastVu region number, else a region title
    (English or local, matched exactly)."""
    regions, labels = [], []
    for spec in specs:
        spec = str(spec).strip()
        key = spec.lower()
        if key in aliases:
            for cid in aliases[key]:
                r = store.region(cid)
                if not r:
                    raise SystemExit(f"alias {key}: region {cid} is not in PastVu's region tree")
                regions.append(r)
            labels.append(key)
            continue
        if spec.isdigit():
            r = store.region(int(spec))
            if not r:
                raise SystemExit(f"region {spec}: no such number in PastVu's region tree")
            regions.append(r)
            labels.append(slug(r["title_en"]) or spec)
            continue
        found = store.find_regions(spec)
        exact = [r for r in found if spec.casefold() in (r["title_en"].casefold(), r["title_local"].casefold())]
        top = [r for r in exact if not r["parents"]]
        if len(exact) == 1 or len(top) == 1:
            r = exact[0] if len(exact) == 1 else top[0]
            regions.append(r)
            labels.append(slug(r["title_en"]))
        elif exact:
            names = "; ".join(f"{r['cid']} {region_path(store, r)} ({r['phc']:,} photos)" for r in exact)
            raise SystemExit(f"region {spec!r} is ambiguous: {names}. Give the number or an alias.")
        else:
            close = ", ".join(f"{r['cid']} {r['title_en']}" for r in found[:10])
            raise SystemExit(f"region {spec!r}: not an alias of the conf and not a region title"
                             + (f"; close: {close}" if close else ""))
    seen, unique = set(), []
    for r in regions:
        if r["cid"] not in seen:
            seen.add(r["cid"])
            unique.append(r)
    return unique, labels


def member_test(years: tuple[int, int], decade: int | None):
    if decade is None:
        return lambda p: True
    lo, hi = decade, decade + 9
    return lambda p: p.get("year") is not None and lo <= int(p["year"]) <= hi


def plan_shards(client: Client, store: Store, scope: str, region: dict, years: tuple[int, int], full_parent: bool,
                shard_max: int, log=print) -> list[dict]:
    f = filter_string([region["cid"]], years)
    count = client.count(f)
    one = [{"key": f"{scope}|{f}", "region": region["cid"], "filter": f, "count": count,
            "parts": {"regions": [region["cid"]]}}]
    if count <= shard_max or full_parent or not region["child_len"]:
        return one
    children = store.children(region["cid"])
    if not children:
        return one
    shards = []
    for child in children:
        shards.extend(plan_shards(client, store, scope, child, years, full_parent, shard_max, log))
    if len(children) <= MAX_EXCLUDE:
        # the rest of the region: its photos in none of the children, by the gallery's exclusion filter
        fr = filter_string([region["cid"]], years, exclude=[c["cid"] for c in children])
        rest = client.count(fr)
        shards.append({"key": f"{scope}|{fr}", "region": region["cid"], "filter": fr, "count": rest,
                       "parts": {"regions": [region["cid"]], "exclude": [c["cid"] for c in children]}})
        log(f"  {region_path(store, region)}: {count:,} photos, paged through {len(children)} children and "
            f"{rest:,} in none of them")
        return shards
    f0 = filter_string([region["cid"]], years, geo="0")
    count0 = client.count(f0)
    shards.append({"key": f"{scope}|{f0}", "region": region["cid"], "filter": f0, "count": count0,
                   "parts": {"regions": [region["cid"]], "geo": "0"}})
    covered = sum(s["count"] for s in shards)
    log(f"  {region_path(store, region)}: {count:,} photos, paged through {len(children)} children and "
        f"{count0:,} without coordinates; up to {max(0, count - covered):,} geotagged photos outside every child "
        f"are left out (--full-parent takes them; the gallery excludes at most {MAX_EXCLUDE} regions)")
    return shards


def progress_of(store: Store, name: str) -> dict:
    """How far the index of a scope or family has got, from the store alone:
    planned shards (None when no plan was recorded), finished, partly paged, rows."""
    plan = store.get_meta(f"plan:{name}")
    prefix = name + "|"
    with store.lock:
        rows = store.db.execute("SELECT key, done, rows FROM shards WHERE substr(key, 1, ?) = ?",
                                (len(prefix), prefix)).fetchall()
    state = {key: (done, n) for key, done, n in rows}
    out = {"planned": None, "done": sum(1 for d, _ in state.values() if d),
           "partial": sum(1 for d, _ in state.values() if not d), "rows": sum(n for _, n in state.values())}
    if plan:
        keys = [k for k, _ in plan["shards"]]
        out.update(planned=len(keys), not_started=sum(1 for k in keys if k not in state),
                   expected=sum(c for _, c in plan["shards"]))
    return out


def describe_progress(p: dict) -> str:
    if p["planned"] is None:
        text = f"{p['done']:,} shards finished, {p['partial']:,} partly paged; the full plan was not recorded"
    else:
        text = (f"{p['done']:,} of {p['planned']:,} shards finished, {p['partial']:,} partly paged, "
                f"{p['not_started']:,} not started")
    text += f" ({p['rows']:,} photos listed"
    if p.get("expected"):
        text += f" of about {p['expected']:,}"
    return text + ")"


def band_shards(client: Client, store: Store, scope: str, shard: dict, years: tuple[int, int],
                band_max: int = BAND_MAX, log=print) -> list[dict]:
    """A shard over band_max photos, not finished, as year bands of at most band_max photos."""
    prev = store.shard(shard["key"])
    if shard["count"] <= band_max or (prev and prev["done"]) or "parts" not in shard:
        return [shard]
    parts = shard["parts"]
    out = []

    def split(y0: int, y1: int) -> None:
        mid = (y0 + y1) // 2
        for a, b in ((y0, mid), (mid + 1, y1)):
            f = filter_string(parts["regions"], (a, b), exclude=parts.get("exclude"), geo=parts.get("geo"))
            c = client.count(f)
            if c > band_max and a < b:
                split(a, b)
            elif c:
                out.append({"key": f"{scope}|{f}", "region": shard["region"], "filter": f, "count": c,
                            "parts": parts, "band": [a, b]})

    if years[0] >= years[1]:
        return [shard]
    split(years[0], years[1])
    log(f"  {shard['filter']}: {shard['count']:,} photos, paged in {len(out)} year bands of at most {band_max:,}")
    return out


def run_index(client: Client, store: Store, scope: dict, full_parent: bool = False, refresh: bool = False,
              shard_max: int = SHARD_MAX, log=print, progress=None, assign=None) -> dict:
    """Page every shard of the scope into the store; a finished shard is not
    paged again unless refresh. progress(done_pages, total_pages, text).

    assign: for `run --by`, scope is a family (its name keys the shards) and
    assign(photo) names the scope each photo joins, or None for no scope."""
    years, decade = tuple(scope["years"]), scope["decade"]
    member = assign or member_test(years, decade)
    shards = []
    for cid in scope["regions"]:
        shards.extend(plan_shards(client, store, scope["name"], store.region(cid), years, full_parent, shard_max, log))
    shards = [b for s in shards for b in band_shards(client, store, scope["name"], s, years, BAND_MAX, log)]
    # unfinished shards the plan no longer has (a long one now paged in bands): their photos stay listed
    store.forget_shards(scope["name"], {s["key"] for s in shards})
    # the plan, so run --resume can tell how much of the index is done without asking PastVu again
    store.set_meta(f"plan:{scope['name']}", {"years": list(years), "time": time.time(),
                                             "shards": [[s["key"], s["count"]] for s in shards]})
    total_pages = sum(-(-s["count"] // PAGE_SIZE) for s in shards)
    stats = {"shards": len(shards), "pages": 0, "rows": 0, "new": 0, "count": sum(s["count"] for s in shards)}
    done_pages = 0
    for s in shards:
        prev = store.shard(s["key"])
        pages = -(-s["count"] // PAGE_SIZE)
        if prev and prev["done"] and not refresh:
            done_pages += pages
            continue
        store.start_shard(s["key"], scope["name"], s["region"], s["filter"], s["count"], restart=refresh)
        page = store.shard(s["key"])["pages_done"]
        done_pages += page
        while True:
            try:
                res = client.give_ps(s["filter"], page * PAGE_SIZE, PAGE_SIZE)
            except PastvuError as e:
                log(f"  shard {s['filter']}: {e}; this shard stops here")
                break
            photos = res.get("photos") or []
            if photos:
                stats["new"] += store.add_page(s["key"], scope["name"], page, photos, member)
                stats["rows"] += len(photos)
                stats["pages"] += 1
                page += 1
                done_pages += 1
                if progress:
                    progress(done_pages, max(total_pages, done_pages), f"{s['filter']} page {page}/{pages}")
            if len(photos) < PAGE_SIZE:
                break
        store.finish_shard(s["key"])
    return stats
