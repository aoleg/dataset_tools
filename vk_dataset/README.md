# VK dataset

Downloads the photographs of public [VK](https://vk.com) (VKontakte) groups, from their albums and their walls, at full size and with their captions, for an image dataset. Every photo gets a `.txt` with its cleaned caption, every group a manifest of all its photos, and every API call is counted against VK's monthly quota. Its output is a raw folder per group that the [pastvu](../pastvu/README.md) import, [deduplicate](../deduplicate/README.md) and `scripts\unrelated.py` take from there ([From download to collection](#from-download-to-collection)).

The tool needs a VK access token of your own application. Read [The photographs are not ours](#the-photographs-are-not-ours) before you publish images you downloaded with it. Everything the tool prints, and its `--help`, is in Russian: it was written for a partner who runs it through a local agent.

## What it is

Groups of old photographs ("old Moscow", Soviet buses, a factory's history) keep thousands of scans in their albums and on their walls. The tool takes a group by its short name, id or link and downloads everything, because the files come from VK's CDN, which costs nothing; only the listing costs API calls, and few of them: one call lists 1,000 photos of an album or 100 posts of a wall. Curation happens afterwards, on the downloaded folder, with the tools of this repository.

- **Finding groups:** `search groups` finds groups by their name and description; `sample` downloads a dozen photos of one group, with a contact sheet on request, for the choice before any download.
- **Albums:** `scan albums` lists the albums of a group with their sizes; `albums` downloads them, every album into a folder named after its title.
- **The wall:** `scan wall` indexes the posts of a wall into `_wall.json` with their full text; `wall` downloads the photos of the posts you select, by date, text or id, and gives every photo the text of its post as its caption. For a repost the text and the owner of the original post are used.
- **Resume and reruns:** a rerun downloads only what is missing. An album or a post whose photos are all on disk costs no call at all.
- **Dead photos:** VK often keeps a photo's record after its file is gone from the CDN. The tool then tries every smaller copy, registers the photo as dead only when all of them are gone, and never asks for it again unless told to.
- **Moved files:** a file that you or another tool moved out of the download (deduplicate, `unrelated.py`, a hand sort) is not downloaded again.
- **The quota:** a ledger of this checkout counts every attempt and every success per day and method, and records the limit errors.

## Install

Run `install.bat` in the repository root, or `install.bat` in this folder after the root one has created the shared venv. It installs `requests` into `..\venv` and makes `vkds.conf` from `vkds.conf.example` when there is none.

Put your token into a file `vk_token.txt` next to `vkds.conf`, or set `VK_TOKEN`, or pass `--token`. Use the token of an application you registered yourself in VK's developer cabinet (in the partner's tests a service key was enough for public groups); the partner who uses the tool observed that VK has restricted accounts seen using tokens of other people's applications. The token never leaves your machine except in the requests to `api.vk.com`, and the tool never prints it.

## The conf file

`vkds.conf`, next to `vkds.py`, holds the paths and limits. Relative paths resolve against the folder of the conf.

```
[paths]
token = vk_token.txt
base =
ledger = _vkds\ledger.json

[limits]
monthly = 10000
delay = 0.35
```

- `token`: the file with the token; `VK_TOKEN` and `--token` take precedence.
- `base`: the datasets root. A run without `--out` writes here, a relative `--out` resolves under it, an absolute `--out` is used as it is. With `base` empty, `--out` is required.
- `ledger`: the ledger of this checkout ([The ledger and the quota](#the-ledger-and-the-quota)).
- `monthly`: the quota of your application, for the report of `budget`.
- `delay`: seconds between two API calls, about 3 per second.

## Usage

Run the tool with `run.bat` in this folder, from anywhere:

```
run.bat search groups --q "<query>" [--q "<query>" ...] [--count N] [--type group|page|all]
run.bat sample <group> [--n 12] [--from wall|albums] [--album-id ID] [--sheet] [--min-side N]
run.bat scan albums <group>
run.bat scan wall <group> [--since YYYY-MM-DD] [--until YYYY-MM-DD] [--limit N] [--all]
run.bat albums <group> [<group> ...] [--album-id ID ...] [--title-regex RX] [--include-wall] [--min-area N] [--min-side N] [--dry-run] [--recheck-dead] [--redownload-missing] [--no-meta] [--workers N]
run.bat wall <group> [--since YYYY-MM-DD] [--until YYYY-MM-DD] [--text-regex RX] [--photos-min N] [--post-id ID ...] [--min-area N] [--min-side N] [--dry-run] [--recheck-dead] [--redownload-missing] [--workers N]
run.bat peek <group> --album-id ID [--limit 6] [--offset N] [--min-side N]
run.bat whoami [--group <group>]
run.bat budget [--seed-ledger]
run.bat recheck-dead <group> [--limit N] [--sizes]
run.bat migrate <dataset> [--dry-run]
```

A `<group>` is a short name (`examplegroup`), an id (`12345`) or a link (`https://vk.com/examplegroup`, `https://vk.com/club12345`). The common options work before and after the verb: `--out <folder>` (the dataset root, see `base`), `--token`, `--delay`, `--budget N` (a hard cap of API attempts for this run), `--no-cache` and `--refresh`. `run.bat --help` and `run.bat <verb> --help` list everything, in Russian. `run.bat` pauses at the end when started by double-click or when the tool fails; set `NOPAUSE=1` to prevent this.

What each verb costs in API calls (the files themselves come from the CDN for free):

| verb | what it does | calls |
|---|---|---|
| `search groups` | searches groups by name and description, merges the results into `_candidates.json` and prints a short table | 1 per query, plus 1 per 500 groups found |
| `sample` | downloads a few photos of a group into `_samples\` for a look, with a contact sheet on request | 1 from the wall, 2 from an album; 1 more for a group not yet in `_candidates.json` or `_groups.json` |
| `scan albums` | lists the albums of a group with sizes and dates into `_albums.json` | 1, plus 1 per 100 albums |
| `scan wall` | indexes the wall into `_wall.json`, prints posts and photos per year and the cost of a download | 1, plus 1 per 100 posts |
| `albums` | downloads albums; an album with nothing missing is checked against the manifest and costs nothing | 1, plus 1 per 100 albums, plus 1 per 1,000 photos of each album it lists |
| `wall` | downloads the photos of the selected posts, listing only the posts that still miss photos | 1 per 100 such posts |
| `peek` | downloads a few sample photos of an album into `_peek\` for a look | 2 |
| `whoami` | checks the token | 1 |
| `budget` | prints the ledger and the dataset's journal; `--seed-ledger` imports the journal's calls into the ledger | 0 |
| `recheck-dead` | tries the dead photos of a group again, from the stored URLs | 0 |
| `migrate` | renames a dataset made before version 0.5 | 0 |

- **`scan albums`** is the cheap first look at a group. It reads the response cache (7 days), so a second scan is nearly free.
- **`scan wall`** pages the wall from the newest post. It stops at the first page that reaches back past `--since`, at the end of the wall, after `--limit` posts, or at a post it already knows, so a rescan costs one call per 100 new posts. With an older `--since` than before it jumps below the posts it knows instead of paging through them again; `--refresh` pages through everything. The pinned post is kept whatever its date. `--all` takes every post of the wall, not only the group's own. `--since` and `--until` are calendar days in UTC, both included.
- **`albums`** downloads every album of the group, or those of `--album-id` (also `wall`, `saved` and `profile`, the system albums) or of `--title-regex`. `--include-wall` adds the system album of wall photos, without their post texts; `wall` is the verb that carries the post text. `--min-side 512` skips photos with a side under 512 pixels, the gate of the pastvu import; `--min-area N` skips photos under N pixels; both may be given. `--dry-run` prints the plan and the estimated calls without listing any photo.
- **`wall`** works from `_wall.json` and needs a `scan wall` first. It selects posts by `--since`, `--until`, `--text-regex` (over the post's own text and, for a repost, the original's), `--photos-min` and `--post-id`, then lists the selected posts live with `wall.getById`, 100 per call, so the download URLs are fresh. `--dry-run` prints the selected posts, their months and the estimated calls, and needs neither a token nor a call.
- **`peek`** shows what an album holds before you decide: `--offset` takes the samples from the middle or the end.
- **`--recheck-dead`** checks the known dead photos again during a download; **`--redownload-missing`** downloads again the photos whose files were moved out of the dataset.

Examples:

```
run.bat scan albums examplegroup
run.bat albums examplegroup --min-side 512 --dry-run
run.bat albums examplegroup --min-side 512
run.bat scan wall examplegroup
run.bat wall examplegroup --since 2015-01-01 --min-side 512 --dry-run
run.bat wall examplegroup --text-regex "трамва|tram" --min-side 512
run.bat budget
```

## Finding groups

Before a download comes the choice of groups. `search groups` asks VK's group search once for every `--q`, so a grid of queries (synonyms, abbreviations, a transliteration) costs one call per query; then it fetches the member count, the description and the counters of the groups found with `groups.getById`, 500 groups per call, because the search itself returns little more than the names. `--count N` takes up to N groups per query (20 by default, at most 1,000), `--type group` or `--type page` limits the search to groups or to public pages. The results of all queries are merged by group id into `<base>\_candidates.json`: earlier candidates stay, fresh data updates them, and each keeps the queries that found it.

The table that `search groups` prints has one line per group found, the largest first: the id, the short name, the title, the members, whether the group is open, how many of the queries found it, the estimated calls of a `scan albums` plus a `scan wall`, and the first 60 characters of the description. The estimate is a lower bound until the number of albums and wall posts is known; a closed group can be neither scanned nor sampled.

Whether VK lets a service key search groups is not known on our side: VK documents `groups.search` for user keys. `tools\capture_fixtures.py` makes one search call to settle it ([Fixtures for the offline tests](#fixtures-for-the-offline-tests)). When the key may not search, `search groups` stops after one call with a message, and groups can still be named by hand and sampled.

`sample <group>` downloads a dozen photos of a group for a look. From the wall, the default, it reads one page of 100 posts and spreads the photos evenly over the posts that have photos, one photo per post; with fewer such posts than photos, each post gives several. With `--from albums` it reads the album list and one page of up to 1,000 photos of the largest album, or of `--album-id`, and spreads the photos over that page. A group found in `_candidates.json` or in the dataset's `_groups.json` costs no lookup. The photos go to `<base>\_samples\<group folder>\`, named as in a download and with their cleaned captions as `.txt` (the post text for a wall photo), next to `_sample.json` with the group, the source and the photos. A photo whose every copy is gone from the CDN is skipped and replaced by a neighbour, and is not registered as dead; `--min-side N` skips small photos the same way. The sample also writes the number of wall posts or albums into the group's candidate entry, which makes its estimate exact.

`--sheet` adds `sheet.jpg`, a contact sheet: the group's title and member count on top, then tiles about 320 pixels wide, four per row, each with its number and the first 60 characters of its caption. One image per group is what a vision model or a person needs to judge a group at a glance. The sheet needs Pillow, which the shared venv has from the other tools, so `requirements.txt` stays `requests`; without Pillow the sample is still written and the sheet is refused with a message. The captions are drawn with Arial or Segoe UI from the Windows fonts folder, for the Cyrillic.

```
run.bat search groups --q "старые автобусы" --q "советский автобус" --q "ЛиАЗ"
run.bat sample examplegroup --sheet
run.bat sample examplegroup --from albums --sheet
```

## Output

```
<base>\
  _groups.json                    the folder of every group in this dataset root
  _vkds\budget.json               the journal of the calls made for this dataset root
  _vkds\cache\                    the response cache of scan and --dry-run
  _peek\<short name>\album_<id>\  samples of peek
  _candidates.json                the groups found by search groups
  _samples\<group folder>\        the photos of sample, _sample.json and sheet.jpg
  <group folder>\                 named after the group's title
    _group.json, _albums.json     the group, and the album index with every album's state
    _wall.json                    the wall index of scan wall
    _manifest.csv                 one row per photo of the group, albums and wall
    _dead_photos.json             the registry of dead photos
    <album folder>\               named after the album's title
      photo<owner>_<id>.jpg
      photo<owner>_<id>.txt       the cleaned caption, when the photo has one
      album.txt, metadata.json    the album, and the data of all its photos
    wall\<YYYY-MM>\               the month of the post (UTC)
      photo<owner>_<id>.jpg
      photo<owner>_<id>.txt       the cleaned text of the post
```

The file name is VK's own id of the photo: `photo-12345_456239017` is photo 456239017 of the group 12345 (`vk.com/photo<owner>_<id>` opens a photo on VK), `photo123_456` a photo of user 123. The names are unique across groups, so folders of several groups can be flattened and deduplicated without collisions. For a repost the owner is the original owner.

**Captions.** The `.txt` holds the caption without links, hashtags and mentions (`[club123|Name]` becomes `Name`), on one line. A wall photo gets the text of its post, or the original's text when the post itself says nothing. A caption with nothing left writes no `.txt`. An existing `.txt` is never rewritten or deleted, so a caption you edited or a model wrote survives every rerun. The raw text stays in the manifest.

**The manifest** `_manifest.csv` (UTF-8 with BOM, opens in Excel) has one row per photo: `photo_id`, `owner_id`, `group_id`, `group_name`, `group_screen`, `album_id`, `album_title`, `album_dir`, `file`, `path` (relative to the dataset root), `txt`, `status`, `size_type`, `width`, `height`, `bytes`, `date`, `likes`, `comments`, `text` (the raw caption), and for wall photos `post_id`, `post_date`, `repost_from` (`<owner>_<post>` of the original) and `reposts`, with `album_id` `post_<post id>` and the post's likes and comments. `moved` holds the date a file was first seen gone from its path. The statuses:

- `ok` downloaded, `exists` already there;
- `dup` the photo already has a file elsewhere in the group (an album, an earlier post); `path` points at that file;
- `small` under `--min-area` or `--min-side`;
- `dead` gone from the CDN, see below;
- `forbidden` (403), `failed` (download errors), `no_url` (no copy in the API response): a rerun tries them again.

## Dead photos

VK keeps the records of many photos whose files are gone from the CDN: the API lists them, the file answers 404. When the best copy answers 404, the tool tries every smaller copy of the photo, which costs no call; copies under `--min-area` or `--min-side` are not taken. A copy that answers is used and recorded in `metadata.json`. Only when every copy answers 404 is the photo dead: its full record (caption, date, likes, the URLs of every size) goes into `_dead_photos.json`, and later runs do not ask the CDN for it for 30 days. A record whose check date cannot be read counts as stale.

`run.bat recheck-dead <group>` tries the dead photos of a group again without any API call, from the URLs in the registry: the best copy, or every stored size with `--sizes`. It prints how many answered 404 on every copy, how many had a copy that answered, and how many failed otherwise. A recovered photo gets its file, its `.txt` and its manifest row, and leaves the registry. The group is found in the dataset root's `_groups.json` by id, short name, folder or title, or give its folder.

## The ledger and the quota

VK allows an application 10,000 API calls per month without a verified business profile. Downloads from the CDN and answers from the cache do not count.

- **The ledger** (`[paths] ledger`, one per checkout of the tool) keeps daily buckets by UTC date: the attempts and the successes of every method. Every HTTP attempt counts, retried and failed ones included, because it is not known which of them VK counts. `--budget N` caps the attempts of one run.
- **The month** that `budget` prints starts on the first of the calendar month. Once you know the day your quota resets, write it into the ledger by hand as `"reset_day": 14` (a day of the month); the period then starts on the last such day.
- **The limit error.** VK error 29 ("Rate limit reached"), and any other non-retried error whose message mentions a limit, stops the run at once, without retries, and is written to the ledger with its code, message, method and time; the first success afterwards is written too. VK documents error 29 as the limit of one method, so a hit on one method is not proof that the month is spent. The two dates tell when the limit was lifted.
- **The dataset journal** `<base>\_vkds\budget.json` keeps the runs of that dataset root with their calls and attempts. Calls made before the ledger existed are only there: `run.bat budget --seed-ledger --out <dataset>` adds the journal's recorded calls to the ledger, each as one attempt and one success on its UTC day, and remembers them, so a second run adds nothing.

## Migrate

Versions before 0.5 named the files `photo_<id>.jpg`. A rerun still recognises them, so an old dataset resumes as it is. `run.bat migrate <dataset> --dry-run` lists the renames to `photo<owner>_<id>`, every sidecar (`.txt`, `.json` and any other file of the same name) with its image; `run.bat migrate <dataset>` renames them and rewrites the manifest and every `metadata.json`. A name that already exists is skipped and reported. Those versions also wrote the raw caption into the `.txt`: a `.txt` whose content is still exactly the raw text of its manifest row is rewritten with the cleaned caption, or removed when nothing is left, and every other `.txt` (edited by hand or by a model) is left alone; `--dry-run` lists these too. `<dataset>` is a group folder or a dataset root.

## Exit codes

- `0` done;
- `1` an error: a VK API error other than the limit, or an unexpected failure;
- `2` a refusal: no token, no `base` and no `--out`, no wall index, a search the key may not make, a closed or empty group for `sample`, a sheet without Pillow, the run cap of `--budget`, or the limit error;
- `130` interrupted with Ctrl+C; what was downloaded stays, and the same command resumes.

## Fixtures for the offline tests

`tools\capture_fixtures.py` records the raw answers of VK for the tool's offline tests, for whoever has a token: `..\venv\Scripts\python.exe tools\capture_fixtures.py <group> [--out-dir <folder>] [--dry-run]`, run from this folder. It makes about seven calls, with a cap of eight: the group, the first page of its albums, 50 photos of the largest album and 50 of the oldest one, two pages of 100 wall posts, and a search for the word `музей` (five results), whose answer, results or an error, tells whether the key may search groups. Choose a public group whose wall has a pinned post and reposts. Each answer is written as a JSON file, with `_capture.json` listing the calls; a file that would contain the token is refused. `--dry-run` lists the calls and writes nothing. Send the folder back by hand.

## From download to collection

The download is raw: copies, rejects and modern photos are still in it. Curation runs on the raw group folder, the one `scan` prints, before any import, because deduplicate and `unrelated.py` move files and an import is a pastvu base with a store of its own. Where the download and the collection live on different machines, the steps split:

On the machine that downloads, from the repository root:

1. `vk_dataset\run.bat scan albums <group>` and `vk_dataset\run.bat scan wall <group>`
2. `vk_dataset\run.bat albums <group> --min-side 512` and `vk_dataset\run.bat wall <group> --min-side 512`
3. `deduplicate\run.bat <group folder> --gpu`: the copies inside the download move to `<group folder>_duplicates`
4. `venv\Scripts\python.exe scripts\unrelated.py <group folder> --endpoint http://127.0.0.1:11434/v1/`: a local LLM (here Ollama's OpenAI-compatible endpoint; the script asks the first model it lists) reads the captions and moves the photos off the topic to `<group folder>_unrelated`; photos without a caption stay

Then the group folder goes to the machine that holds the collection, where in `pastvu\`:

5. `run.bat import --dataset <group folder> -o <import folder> --against <PastVu base>`: the pastvu import cleans, sorts and filters the photos and skips the copies of photos the base already has; without a base, leave out `--against`
6. `run.bat --base <import folder> reclassify`, and `cherrypick`, as on any import
7. `run.bat --base <import folder> autocaption <import folder>` for the captions of a vision model, on either machine's GPU

The import keeps the file names, so `photo<owner>_<id>` carries the provenance into the collection; the VK manifest and `_wall.json` stay the record of where each file came from. The era of a photo is decided by the classifier and the LLM: modern photos go to the outliers of the samples folder, which holds examples of them.

## The photographs are not ours

The photographs in VK groups belong to their photographers, archives, museums and publishers, not to VK, not to the group that posted them and not to whoever downloads them. VK's terms of use and the rules of its API govern the access; read them for your use, and use only a token of your own application. Downloading for a personal or research dataset is one thing, publishing the images again is another: that needs the rights holder of each image.

Most photographs of the Soviet era are still protected somewhere. In Russia a photograph is protected for the author's life plus 70 years, plus 4 years for those who served in the Second World War, and an anonymous work for 70 years from publication; other countries of the former Soviet Union have their own terms. Text and data mining exceptions cover the analysis and a local copy, not the publication of the files. This is not legal advice.

The manifest keeps what attribution and a takedown need: the group, the album or post, the photo's id (which opens it on VK), its date and its caption.

## For the partner

The folder `partner\` holds the handover for the partner who runs the tool through his local agent, in Russian. `partner\README.md` is his manual: the two scenarios of his agent (finding groups with `search groups` and `sample --sheet`, and downloading groups with a selection by a condition through [judge](../judge/README.md)), the wiring of OpenCode for them and two sample tasks. `ПЕРЕХОД.md` is the one-time move from the old package. The four OpenCode skills (`skill_vk-dataset.md`, `skill_dataset-tools-postprocess.md`, `skill_vk-agent-scenarios.md`, `skill_dataset-tools-menu.md`) complete it; the last one gives his agent a menu at the start of a session and the rule by which it writes one skill per tool of this repository from the tool's `--help` and README the first time the tool is chosen. His ready-made agent files (AGENTS.md, opencode.jsonc, the skills tree) and the installation note carry his machine paths, so they live in two git-ignored folders under `partner\` and are passed to him by hand.

## Lint

The code is checked with ruff from the repository root's `ruff.toml` (check only, no formatting). ruff is installed by hand, never by an `install.bat`: `venv\Scripts\python.exe -m pip install ruff==0.17.3`, then `venv\Scripts\python.exe -m ruff check vk_dataset` from the repository root.

## Project files

| file | purpose |
|---|---|
| `vkds.py` | the entry point |
| `vkds\cli.py` | the command line |
| `vkds\conf.py` | reads `vkds.conf`: the token, `base`, the ledger, the limits |
| `vkds\api.py` | the API client: throttle, retries, the limit error, live listing |
| `vkds\budget.py` | the run journal of a dataset and the ledger of the checkout |
| `vkds\cache.py` | the response cache of scan and --dry-run |
| `vkds\albums.py` | albums: listing, download, size fallback, dead registry, manifest, reconcile |
| `vkds\wall.py` | the wall: `scan wall` and `wall` |
| `vkds\discover.py` | `search groups` and `sample` |
| `vkds\sheet.py` | the contact sheet of `sample --sheet` |
| `vkds\captions.py` | caption cleaning for the `.txt` |
| `vkds\recheck.py` | `recheck-dead` |
| `vkds\migrate.py` | `migrate` |
| `vkds\fs.py`, `vkds\models.py` | file helpers and the CDN download, the group and album models |
| `tools\capture_fixtures.py` | the capture of raw VK answers for the offline tests |
| `vkds.conf.example` | the template of `vkds.conf` |
| `partner\` | the Russian handover for the partner |
