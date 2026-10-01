# Image search downloader

`ddg_images.py` runs image searches on Bing or DuckDuckGo through the [ddgs](https://pypi.org/project/ddgs/) package, goes through the result pages, and downloads every image it finds into one folder. A CSV file records each image with its source page and the download result. Use it to collect raw material for a dataset, which you then sort and caption.

## Install

It needs Python 3.10 or newer and one package:

```
pip install ddgs
```

## Usage

```
python ddg_images.py "query one" "query two" -o posters
python ddg_images.py -f queries.txt -o posters --pages 30
python ddg_images.py "query" --backend duckduckgo
```

Give the queries on the command line, in a text file with one query per line (`-f`), or both. The file can be UTF-8 or the Windows Cyrillic code page (cp1251) that Notepad writes as "ANSI".

| option | default | meaning |
|---|---|---|
| `-f`, `--file` | none | text file with one query per line; empty lines are ignored |
| `-o`, `--out` | `images` | output folder; it is created if it does not exist |
| `--backend` | `bing` | `bing` (about 35 images per page) or `duckduckgo` (about 100 per page, but see [DuckDuckGo returns nothing](#duckduckgo-returns-nothing)) |
| `--pages` | 30 | maximum number of result pages per query |
| `--region` | `ru-ru` | search region, for example `us-en` or `de-de`; only DuckDuckGo uses it, Bing ignores it |
| `--delay` | 3 | seconds between page requests; the real wait is a random value between this and twice this |
| `--threads` | 8 | number of parallel downloads |

Safe search is always off.

## What it does

1. **Search.** For each query, the script requests page 1, 2, 3 and so on, up to `--pages`. It stops a query early when a page has no URL that the query did not already return, because the engine then repeats its last page. A "No results" answer also ends the query. On other errors it waits and tries again, three times in total, then goes to the next query.
2. **Deduplicate URLs.** An image URL that an earlier page or query already returned is not downloaded again.
3. **Download.** When all searches are done, the images are downloaded in parallel. Each request sends the result page as the referrer, because some sites refuse image requests without one. A response is rejected if it is not an image or is smaller than 2 KB.
4. **Deduplicate content.** Each file is named by the first 16 hex characters of the SHA-256 of its content, for example `29aa5d031150b92e.jpg`. The extension comes from the content type that the server sends. The same image from two different URLs gives the same name, so it is saved once and the second copy is marked `duplicate`.
5. **Log.** `metadata.csv` in the output folder gets one row for each URL, including the failed ones.

## metadata.csv

The file is UTF-8 with a BOM, so Excel opens Cyrillic text correctly.

| column | content |
|---|---|
| `query`, `page` | the query and the result page that found the image |
| `title` | the title of the search result |
| `image` | the image URL |
| `url` | the page that shows the image |
| `source` | the source that the engine reports, if any |
| `width`, `height` | the size that the engine reports; this can differ from the downloaded file |
| `file` | the saved file name, empty if the download failed |
| `bytes`, `sha256` | size and hash of the downloaded data |
| `status` | `ok`, `duplicate`, `too small (N bytes)`, `not an image (type)`, or `error: ...` |

At the end the script prints a count for each status.

## Notes

- A second run into the same folder skips images it already has, because the file names come from the content. But it downloads them again to find that out, and it replaces `metadata.csv` with the rows of the new run only. To keep the old log, use a new output folder or rename the file first.
- All search results are collected before the first download starts. With many queries and pages, the search phase takes a while: at the default delay, each page waits 3 to 6 seconds.
- Search engines limit automated requests. If you get many errors, increase `--delay`.
- The script removes the `Connection: keep-alive` header from the ddgs DuckDuckGo engine. Some ddgs versions send it, HTTP/2 does not allow it, and every DuckDuckGo request then fails with "malformed headers". Tested with ddgs 9.16.0.
- The downloaded images belong to their owners. Check the license before you use them.

## DuckDuckGo returns nothing

As of October 2026, `--backend duckduckgo` finds no images: every query shows `0 results` on page 1. The script then prints a warning. Use the Bing backend.

The cause is on the DuckDuckGo side. Its image API (`duckduckgo.com/i.js`) now expects extra parameters (`jsa`, `jsa_hash`, `dp`, `j_id`) that the script of its own search page computes in the browser. ddgs 9.16.0, the newest version at that time, does not send them. DuckDuckGo then answers 403 and marks the client as a bot, and ddgs reports this as "No results found". A normal browser on the same computer gets results, so the block is against the client, not the IP address. The backend may work again if a later ddgs version supports the new parameters.
