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
"""

from __future__ import annotations

import time

from client import Client, PastvuError, filter_string
from layout import slug
from store import Store

PAGE_SIZE = 100
SHARD_MAX = 50_000
MAX_EXCLUDE = 10           # the gallery filter takes at most 10 regions to exclude
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
    one = [{"key": f"{scope}|{f}", "region": region["cid"], "filter": f, "count": count}]
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
        shards.append({"key": f"{scope}|{fr}", "region": region["cid"], "filter": fr, "count": rest})
        log(f"  {region_path(store, region)}: {count:,} photos, paged through {len(children)} children and "
            f"{rest:,} in none of them")
        return shards
    f0 = filter_string([region["cid"]], years, geo="0")
    count0 = client.count(f0)
    shards.append({"key": f"{scope}|{f0}", "region": region["cid"], "filter": f0, "count": count0})
    covered = sum(s["count"] for s in shards)
    log(f"  {region_path(store, region)}: {count:,} photos, paged through {len(children)} children and "
        f"{count0:,} without coordinates; up to {max(0, count - covered):,} geotagged photos outside every child "
        f"are left out (--full-parent takes them; the gallery excludes at most {MAX_EXCLUDE} regions)")
    return shards


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
