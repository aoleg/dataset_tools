"""The index of a scope: every photo PastVu lists for the regions and years,
pulled through the gallery (photo.givePS) in pages of 100 into the store, so
the stream knows what to fetch and a resume knows what is left.

A gallery page costs the server a count over the region's photos, and a deep
skip grows with the page number, so a region with many photos is paged
through its children (the region tree) until each shard is under SHARD_MAX.
A photo that lies in the parent but in none of its children is caught by a
geo!0 shard of the parent (photos without coordinates); geotagged photos
outside every child are the residual, reported and fetched with
--full-parent, which pages the parent itself.
"""

from __future__ import annotations

import time

from client import Client, PastvuError, filter_string
from store import Store

PAGE_SIZE = 100
SHARD_MAX = 50_000
REGIONS_MAX_AGE = 30 * 86400


def ensure_regions(client: Client, store: Store, refresh: bool = False) -> int:
    pulled = store.get_meta("regions_pulled", 0)
    if refresh or store.regions_count() == 0 or time.time() - pulled > REGIONS_MAX_AGE:
        store.put_regions(client.regions())
        store.set_meta("regions_pulled", time.time())
    return store.regions_count()


def resolve_regions(store: Store, specs: list[str]) -> list[dict]:
    """Region specs (cids, or names matched against the English and local
    titles) -> region rows. An ambiguous name lists its candidates."""
    out = []
    for spec in specs:
        spec = str(spec).strip()
        if spec.isdigit():
            r = store.region(int(spec))
            if not r:
                raise SystemExit(f"region {spec}: no such cid in the region tree")
            out.append(r)
            continue
        found = store.find_regions(spec)
        exact = [r for r in found if spec.casefold() in (r["title_en"].casefold(), r["title_local"].casefold())]
        top = [r for r in exact if not r["parents"]]
        if len(exact) == 1:
            out.append(exact[0])
        elif len(top) == 1:
            out.append(top[0])
        elif exact:
            names = "; ".join(f"{r['cid']} {region_path(store, r)} ({r['phc']:,} photos)" for r in exact)
            raise SystemExit(f"region {spec!r} is ambiguous: {names}. Give the cid.")
        else:
            close = ", ".join(f"{r['cid']} {r['title_en']}" for r in found[:10])
            raise SystemExit(f"region {spec!r}: not found" + (f"; close: {close}" if close else ""))
    return out


def region_path(store: Store, region: dict) -> str:
    names = []
    for p in region["parents"]:
        r = store.region(p)
        if r:
            names.append(r["title_en"])
    return " > ".join(names + [region["title_en"]])


def shard_key(region: int, filter_: str) -> str:
    return f"{region}|{filter_}"


def plan_shards(client: Client, store: Store, region: dict, years: tuple[int, int], full_parent: bool,
                log=print) -> list[dict]:
    f = filter_string([region["cid"]], years)
    count = client.count(f)
    one = [{"key": shard_key(region["cid"], f), "region": region["cid"], "filter": f, "count": count}]
    if count <= SHARD_MAX or full_parent or not region["child_len"]:
        return one
    children = store.children(region["cid"])
    if not children:
        return one
    shards = []
    for child in children:
        shards.extend(plan_shards(client, store, child, years, full_parent, log))
    f0 = filter_string([region["cid"]], years, geo="0")
    count0 = client.count(f0)
    shards.append({"key": shard_key(region["cid"], f0), "region": region["cid"], "filter": f0, "count": count0})
    covered = sum(s["count"] for s in shards)
    log(f"  {region_path(store, region)}: {count:,} photos, paged through {len(children)} children and "
        f"{count0:,} without coordinates; up to {max(0, count - covered):,} geotagged photos outside every child "
        f"are left out (--full-parent takes them)")
    return shards


def run_index(client: Client, store: Store, regions: list[dict], years: tuple[int, int], full_parent: bool = False,
              refresh: bool = False, log=print, progress=None) -> dict:
    """Page every shard of the scope into the store; a finished shard is not
    paged again unless refresh. progress(done_pages, total_pages, text)."""
    shards = []
    for r in regions:
        shards.extend(plan_shards(client, store, r, years, full_parent, log))
    total_pages = sum(-(-s["count"] // PAGE_SIZE) for s in shards)
    stats = {"shards": len(shards), "pages": 0, "rows": 0, "new": 0, "count": sum(s["count"] for s in shards)}
    done_pages = 0
    for s in shards:
        prev = store.shard(s["key"])
        pages = -(-s["count"] // PAGE_SIZE)
        if prev and prev["done"] and not refresh:
            done_pages += pages
            if progress:
                progress(done_pages, total_pages, f"{s['filter']} (indexed before)")
            continue
        page = 0 if refresh or not prev else prev["pages_done"]
        store.start_shard(s["key"], s["region"], s["filter"], s["count"])
        done_pages += page
        while True:
            try:
                res = client.give_ps(s["filter"], page * PAGE_SIZE, PAGE_SIZE)
            except PastvuError as e:
                log(f"  shard {s['filter']}: {e}; this shard stops here")
                break
            photos = res.get("photos") or []
            if photos:
                stats["new"] += store.add_page(s["key"], page, photos)
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
