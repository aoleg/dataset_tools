#!/usr/bin/env python3
"""
Collect and download image search results, page by page, from Bing or DuckDuckGo
(via the ddgs package) or from a SearXNG instance (via its JSON API).

The script loops over pages and queries, deduplicates by URL and file content,
and logs every image (including failures) to a CSV.

Usage:
    pip install ddgs                      # not needed for --backend searxng
    python ddg_images.py "query one" "query two" -o posters
    python ddg_images.py -f queries.txt -o posters --pages 30
    python ddg_images.py "query" --backend searxng --searxng-url http://localhost:8080

Bing returns 35 images per page, DuckDuckGo about 100, SearXNG depends on its engines.
"""

import argparse
import csv
import hashlib
import json
import mimetypes
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif",
       "image/webp": ".webp", "image/bmp": ".bmp", "image/tiff": ".tif"}
FIELDS = ["query", "page", "title", "image", "url", "source", "width", "height",
          "file", "bytes", "sha256", "status"]


class SearchError(Exception):
    """A failed page request that is worth a retry."""


def ddgs_pager(backend, region):
    """-> function(query, page) returning the result dicts of one ddgs page."""
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

    ddgs = DDGS(timeout=20)

    def page_of(q, page):
        try:
            return ddgs.images(q, region=region, safesearch="off",
                               backend=backend, page=page, max_results=None)
        except DDGSException as ex:
            if "No results" in str(ex):
                return []
            raise SearchError(str(ex)) from ex
    return page_of


def searxng_language(region):
    """ddgs region (ru-ru, wt-wt) -> SearXNG language (ru-RU, all)."""
    if not region or region.lower() in ("wt-wt", "all"):
        return "all"
    lang, _, country = region.partition("-")
    return f"{lang.lower()}-{country.upper()}" if country else lang.lower()


def searxng_pager(base_url, region, engines):
    """-> function(query, page) returning the result dicts of one SearXNG page."""
    base_url = base_url.rstrip("/")
    language = searxng_language(region)
    reported = set()

    def page_of(q, page):
        params = {"q": q, "format": "json", "pageno": page, "safesearch": 0, "language": language}
        # SearXNG adds the engines of "categories" to those of "engines", so an
        # explicit engine list must be sent without the category.
        if engines:
            params["engines"] = engines
        else:
            params["categories"] = "images"
        req = urllib.request.Request(f"{base_url}/search?{urllib.parse.urlencode(params)}",
                                     headers={"User-Agent": UA, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as ex:
            if ex.code == 403:
                sys.exit(f"SearXNG at {base_url} refused format=json (HTTP 403). Add json to "
                         "search: formats: in its settings.yml and restart it. See the README.")
            raise SearchError(f"SearXNG HTTP {ex.code}") from ex
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as ex:
            raise SearchError(f"SearXNG: {type(ex).__name__}: {ex}") from ex

        # Engines that failed for this request (CAPTCHA, rate limit, access denied).
        # Reported once per engine, so a blocked engine does not hide behind good ones.
        for name, reason in data.get("unresponsive_engines") or []:
            if name not in reported:
                reported.add(name)
                print(f"  SearXNG engine '{name}' did not answer: {reason}")

        out = []
        for r in data.get("results", []):
            img = r.get("img_src") or ""
            if img.startswith("//"):
                img = "https:" + img
            if not img.startswith(("http://", "https://")):
                continue
            m = re.match(r"\s*(\d+)\s*[x×]\s*(\d+)", str(r.get("resolution") or ""))
            out.append({
                "title": r.get("title") or "",
                "image": img,
                "url": r.get("url") or "",
                "source": ",".join(r.get("engines") or [r.get("engine") or ""]),
                "width": m.group(1) if m else "",
                "height": m.group(2) if m else "",
            })
        return out
    return page_of


def search(queries, pages, delay, backend, page_of):
    """Yield unique result dicts across all queries and pages."""
    seen = set()
    any_results = False
    for q in queries:
        seen_q = set()  # URLs seen for this query; detects the end of its results
        for page in range(1, pages + 1):
            for attempt in range(3):
                try:
                    batch = page_of(q, page)
                    break
                except SearchError as ex:
                    msg = str(ex)
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


def ascii_url(url):
    """Make a URL sendable by urllib, which accepts only ASCII.

    Search engines return internationalised hosts percent-encoded
    (https://%D0%B2%D0%B4%D0%BF%D0%BE.%D1%80%D1%84/...) or raw, and paths with
    raw Cyrillic. The host goes to IDNA (xn--...), the rest is percent-encoded;
    existing %XX escapes are kept.
    """
    try:
        parts = urllib.parse.urlsplit(url)
        host = urllib.parse.unquote(parts.hostname or "")
        if not host.isascii():
            host = host.encode("idna").decode("ascii")
        netloc = host + (f":{parts.port}" if parts.port else "")
        if parts.username:
            netloc = parts.username + (f":{parts.password}" if parts.password else "") + "@" + netloc
        safe = "/%:@!$&'()*+,;=-._~?"
        return urllib.parse.urlunsplit((parts.scheme, netloc,
                                        urllib.parse.quote(parts.path, safe=safe),
                                        urllib.parse.quote(parts.query, safe=safe), ""))
    except (ValueError, UnicodeError):
        return url


# Wikimedia's User-Agent policy requires automated clients to identify themselves,
# and it throttles generic browser user agents with HTTP 429.
# https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy
TOOL_UA = f"ddg_images/1.0 (https://github.com/aoleg/dataset_tools) python-urllib/{sys.version_info[0]}.{sys.version_info[1]}"
TOOL_UA_HOSTS = ("wikimedia.org", "wikipedia.org")


def download(row, outdir):
    url = ascii_url(row["image"])
    host = urllib.parse.urlsplit(url).hostname or ""
    ua = TOOL_UA if host.endswith(TOOL_UA_HOSTS) else UA
    req = urllib.request.Request(url, headers={
        "User-Agent": ua,
        "Referer": ascii_url(row.get("url") or ""),
        "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
    })
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                ctype = resp.headers.get_content_type()
                data = resp.read()
            break
        except urllib.error.HTTPError as ex:
            if ex.code != 429 or attempt == 2:
                return {**row, "status": f"error: HTTPError: {ex.code} {str(ex.reason)[:60]}"}
            # Rate limited: wait as long as the server asks (capped), then try again.
            try:
                wait = min(float(ex.headers.get("Retry-After", "")), 60.0)
            except ValueError:
                wait = 5.0 * (attempt + 1)
            time.sleep(wait + random.uniform(0, 1))
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
    ap.add_argument("--backend", choices=["bing", "duckduckgo", "searxng"], default="bing",
                    help="search backend (default: bing, 35 images per page; duckduckgo gives ~100 per page; "
                         "searxng uses the instance at --searxng-url)")
    ap.add_argument("--pages", type=int, default=30, help="max result pages per query (default: 30)")
    ap.add_argument("--region", default="ru-ru",
                    help="DuckDuckGo region, also the SearXNG language: ru-ru becomes ru-RU, wt-wt "
                         "becomes all (default: ru-ru); Bing ignores it")
    ap.add_argument("--searxng-url", default="http://localhost:8080",
                    help="SearXNG base URL (default: http://localhost:8080)")
    ap.add_argument("--engines", default="",
                    help="SearXNG only: comma-separated engine names, e.g. \"bing images,flickr\" "
                         "(default: the instance's enabled image engines)")
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

    if args.backend == "searxng":
        page_of = searxng_pager(args.searxng_url, args.region, args.engines)
    else:
        page_of = ddgs_pager(args.backend, args.region)
    rows = list(search(queries, args.pages, args.delay, args.backend, page_of))
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