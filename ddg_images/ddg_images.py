#!/usr/bin/env python3
"""
Collect and download image search results (Bing or DuckDuckGo via ddgs), page by page.

ddgs fetches one result page per call; this script loops over pages and queries,
deduplicates by URL and file content, and logs every image (including failures)
to a CSV.

Usage:
    pip install ddgs
    python ddg_images.py "query one" "query two" -o posters
    python ddg_images.py -f queries.txt -o posters --pages 30
    python ddg_images.py "query" --backend duckduckgo

Bing returns 35 images per page, DuckDuckGo about 100.
"""

import argparse
import csv
import hashlib
import mimetypes
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ddgs import DDGS
from ddgs.exceptions import DDGSException

# Workaround for a ddgs bug: its DuckDuckGo images engine sends "Connection: keep-alive",
# a header HTTP/2 forbids, so every request to duckduckgo.com fails with
# "user error: malformed headers". Drop that header before the engine is created.
try:
    from ddgs.engines.duckduckgo_images import DuckduckgoImages
    DuckduckgoImages.headers_update = {
        k: v for k, v in DuckduckgoImages.headers_update.items() if k.lower() != "connection"
    }
except (ImportError, AttributeError):
    pass

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif",
       "image/webp": ".webp", "image/bmp": ".bmp", "image/tiff": ".tif"}
FIELDS = ["query", "page", "title", "image", "url", "source", "width", "height",
          "file", "bytes", "sha256", "status"]


def search(queries, pages, region, delay, backend):
    """Yield unique result dicts across all queries and pages."""
    seen = set()
    ddgs = DDGS(timeout=20)
    any_results = False
    for q in queries:
        seen_q = set()  # URLs seen for this query; detects the end of its results
        for page in range(1, pages + 1):
            for attempt in range(3):
                try:
                    batch = ddgs.images(q, region=region, safesearch="off",
                                        backend=backend, page=page, max_results=None)
                    break
                except DDGSException as ex:
                    msg = str(ex)
                    if "No results" in msg:
                        batch = []
                        break
                    wait = delay * 5 * (attempt + 1)
                    print(f"  [{q} p{page}] error: {msg[:400]}; retry in {wait:.0f}s")
                    time.sleep(wait)
            else:
                print(f"  [{q} p{page}] giving up on this query")
                break

            urls = {r["image"] for r in batch if r.get("image")}
            any_results = any_results or bool(urls)
            fresh_q = urls - seen_q
            seen_q |= urls
            new = [r for r in batch if r.get("image") and r["image"] not in seen]
            print(f"[{q}] page {page}: {len(batch)} results, {len(new)} new")
            for r in new:
                seen.add(r["image"])
                yield {"query": q, "page": page, **r}
            if not fresh_q:  # the engine keeps returning the same page: end of results
                break
            time.sleep(delay + random.uniform(0, delay))

    if not any_results and backend == "duckduckgo":
        # ddgs reports an HTTP 403 from i.js as "No results found", so a blocked
        # engine looks the same as an empty search.
        print("\nWARNING: DuckDuckGo returned no images for any query. This usually means it\n"
              "rejected the requests: as of October 2026 its image API needs tokens that\n"
              "only its own page script computes, and ddgs does not send them. Use --backend bing.")


def download(row, outdir):
    url = row["image"]
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Referer": row.get("url") or "",
        "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            ctype = resp.headers.get_content_type()
            data = resp.read()
    except Exception as ex:  # noqa: BLE001
        return {**row, "status": f"error: {type(ex).__name__}: {str(ex)[:80]}"}
    if not ctype.startswith("image/"):
        return {**row, "status": f"not an image ({ctype})"}
    if len(data) < 2048:
        return {**row, "status": f"too small ({len(data)} bytes)"}
    sha = hashlib.sha256(data).hexdigest()
    ext = EXT.get(ctype) or mimetypes.guess_extension(ctype) or ".img"
    path = outdir / f"{sha[:16]}{ext}"
    if path.exists():
        return {**row, "file": path.name, "bytes": len(data), "sha256": sha, "status": "duplicate"}
    path.write_bytes(data)
    return {**row, "file": path.name, "bytes": len(data), "sha256": sha, "status": "ok"}


def main():
    for stream in (sys.stdout, sys.stderr):  # avoid crashes on non-UTF-8 consoles or redirected output
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queries", nargs="*", help="search queries")
    ap.add_argument("-f", "--file", help="text file with one query per line")
    ap.add_argument("-o", "--out", default="images", help="output folder (default: images)")
    ap.add_argument("--backend", choices=["bing", "duckduckgo"], default="bing",
                    help="search backend (default: bing, 35 images per page; duckduckgo gives ~100 per page)")
    ap.add_argument("--pages", type=int, default=30, help="max result pages per query (default: 30)")
    ap.add_argument("--region", default="ru-ru", help="DuckDuckGo region (default: ru-ru)")
    ap.add_argument("--delay", type=float, default=3.0, help="seconds between page requests (default: 3)")
    ap.add_argument("--threads", type=int, default=8, help="parallel downloads (default: 8)")
    args = ap.parse_args()

    queries = list(args.queries)
    if args.file:
        raw = Path(args.file).read_bytes()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:  # file saved by Notepad as "ANSI" on a Russian Windows
            text = raw.decode("cp1251")
        queries += [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not queries:
        ap.error("give at least one query or -f FILE")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = list(search(queries, args.pages, args.region, args.delay, args.backend))
    print(f"\n{len(rows)} unique image URLs found; downloading...")

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        results = list(ex.map(lambda r: download(r, outdir), rows))

    csv_path = outdir / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    stats = {}
    for r in results:
        key = r["status"].split(":")[0].split(" (")[0]
        stats[key] = stats.get(key, 0) + 1
    print(f"Done. {stats}")
    print(f"Images and {csv_path.name} are in {outdir.resolve()}")


if __name__ == "__main__":
    main()