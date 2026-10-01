# Image search downloader

`ddg_images.py` runs image searches on Bing or DuckDuckGo through the [ddgs](https://pypi.org/project/ddgs/) package, or on a [SearXNG](https://docs.searxng.org/) instance through its JSON API. It goes through the result pages and downloads every image it finds into one folder. A CSV file records each image with its source page and the download result. Use it to collect raw material for a dataset, which you then sort and caption.

## Install

It needs Python 3.10 or newer. On Windows, run `install.bat`. It creates a `venv` folder next to the script and installs `requirements.txt` into it. Running it again updates the packages.

Without `install.bat`, install the one package yourself:

```
pip install ddgs
```

Only the Bing and DuckDuckGo backends need `ddgs`. The SearXNG backend needs no package, but the instance must allow JSON output. See [SearXNG](#searxng).

## Usage

On Windows, use `run.bat`. It runs the script with the venv Python and passes all arguments to it unchanged. Relative paths, for `-f` and `-o`, are relative to the current folder.

```
run.bat "query one" "query two" -o posters
run.bat -f queries.txt -o posters --pages 30
run.bat "historic poster" --backend searxng
run.bat "historic poster" --backend searxng --engines "bing images,flickr,wikicommons.images"
```

Elsewhere, or with the package installed yourself, use `python ddg_images.py` with the same arguments.

`run.bat` pauses at the end when it is started by double-click, or when the script fails. PowerShell starts a `.bat` file in the same way as Explorer, so it also pauses there. Set `NOPAUSE=1` to prevent this.

Give the queries on the command line, in a text file with one query per line (`-f`), or both. The file can be UTF-8 or the Windows Cyrillic code page (cp1251) that Notepad writes as "ANSI".

| option | default | meaning |
|---|---|---|
| `-f`, `--file` | none | text file with one query per line; empty lines are ignored |
| `-o`, `--out` | `images` | output folder; it is created if it does not exist |
| `--backend` | `bing` | `bing` (about 35 images per page), `duckduckgo` (about 100 per page, but see [DuckDuckGo returns nothing](#duckduckgo-returns-nothing)) or `searxng` (the instance at `--searxng-url`) |
| `--pages` | 30 | maximum number of result pages per query |
| `--region` | `ru-ru` | search region, for example `us-en` or `de-de`; DuckDuckGo uses it as it is, SearXNG as its language (`ru-ru` becomes `ru-RU`, `wt-wt` becomes `all`), Bing ignores it |
| `--searxng-url` | `http://localhost:8080` | base URL of the SearXNG instance |
| `--engines` | all enabled | SearXNG only: comma-separated engine names, as SearXNG shows them in Preferences, Engines, Images |
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
| `source` | the source that the engine reports, if any; with SearXNG, the engines that found the image |
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
- Image URLs with non-Latin characters, for example on `.рф` domains, are converted before the download: the host to its IDNA form (`xn--...`), the rest to percent-encoding.
- Downloads from Wikimedia (`upload.wikimedia.org`) use a user agent that names the tool and the repository, as the [Wikimedia User-Agent policy](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy) requires. All other downloads use a browser user agent.
- On HTTP 429 (Too Many Requests) the script waits for the time that the server gives in `Retry-After` (at most 60 seconds) and tries again, three times in total.
- Some sites fail with `CERTIFICATE_VERIFY_FAILED`, usually because they send an incomplete certificate chain or use a certificate authority that Python does not trust. The script does not turn off certificate checks; these images are skipped and logged.
- Some sites refuse direct image downloads with HTTP 403. These are logged as errors.
- The downloaded images belong to their owners. Check the license before you use them.

## DuckDuckGo returns nothing

As of October 2026, `--backend duckduckgo` finds no images: every query shows `0 results` on page 1. The script then prints a warning. Use the Bing backend.

The cause is on the DuckDuckGo side. Its image API (`duckduckgo.com/i.js`) now expects extra parameters (`jsa`, `jsa_hash`, `dp`, `j_id`) that the script of its own search page computes in the browser. ddgs 9.16.0, the newest version at that time, does not send them. DuckDuckGo then answers 403 and marks the client as a bot, and ddgs reports this as "No results found". A normal browser on the same computer gets results, so the block is against the client, not the IP address. The backend may work again if a later ddgs version supports the new parameters.

## SearXNG

[SearXNG](https://docs.searxng.org/) is a metasearch engine that you can run on your own computer. It sends each query to many image engines at once, for example Bing, Flickr, Wikimedia Commons, Openverse and DeviantArt, and merges the results. With `--backend searxng`, the script asks your instance through its JSON API and pages through the results like with the other backends.

### Allow JSON output

By default SearXNG allows only HTML output, and the script then stops with "refused format=json (HTTP 403)". To allow JSON:

1. Open the `settings.yml` of your instance. In the official Docker setup it is in the folder that is mounted as `/etc/searxng`.
2. Find the `search:` section and add `json` to `formats`. If the file has no `formats` list, add one:

   ```yaml
   search:
     formats:
       - html
       - json
   ```

   If `search:` already exists, add only the `formats` lines under it. A second `search:` key replaces the first one.
3. Restart SearXNG, for example `docker restart searxng` or `docker compose restart`.
4. Check it: open `http://localhost:8080/search?q=test&categories=images&format=json` in a browser. You must get JSON text, not "Forbidden".

Only do this on an instance that is not reachable from the internet. A public instance with JSON output is easy to use for automated scraping.

If the script gets HTTP 429 (Too Many Requests), the SearXNG limiter is on. Turn it off for a private instance (`server: limiter: false` in `settings.yml`), or add `127.0.0.1` to `pass_ip` in `limiter.toml`.

### Engines

The script uses the image engines that are enabled in your instance, unless you give `--engines`. With `--engines`, only the named engines are asked, and these can include engines that are disabled in the SearXNG preferences. For example, `--engines "yandex images"` works on an instance where Yandex is off. When an engine does not answer, the script prints its name and the reason that SearXNG gives, once per run. Typical reasons are a CAPTCHA, "too many requests" or "access denied". The DuckDuckGo engine of SearXNG is blocked for the same reason as the DuckDuckGo backend of this script. For Russian-language queries, `yandex images` can be worth enabling in the SearXNG preferences.

The number of results per page depends on the engines. SearXNG removes duplicate results across engines, and the script removes duplicate URLs across pages and queries.

In a test on 2026-10-02 with three queries for historic photographs and 3 pages each, SearXNG found 1,339 unique image URLs, against about 260 from the Bing backend for 30 pages. Google Images supplied 866 of them. The stock photo engines (Pexels, Unsplash, Art Institute of Chicago) return mostly unrelated images for such queries; leave them out with `--engines`, for example `--engines "google images,bing images,yandex images,wikicommons.images,flickr"`.

### Pace

Each page request goes to every selected engine, so SearXNG sends many more upstream requests than one backend does. After that test run, Google answered "access denied" and SearXNG suspended its Google engine. SearXNG retries a suspended engine after a few minutes, for an "access denied" after 180 seconds by default, but Google can keep refusing for longer. For long runs, use a larger `--delay`, for example 10, and watch the "did not answer" lines.
