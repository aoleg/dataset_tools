# PastVu

Downloads historic photographs from [PastVu](https://pastvu.com) by region and years, and prepares each one for a training dataset while it downloads: it cuts the grey PastVu band and any scanned border, paints out watermarks and printed captions, sorts the image into the categories of a [classify](../classify/README.md) samples folder, and writes it with a one-line caption for the captioner. One image after another, with every model kept on the GPU, and with a store that lets a stopped run continue where it was.

The tool uses PastVu's public API (no account, no key). Read [The photographs are not PastVu's](#the-photographs-are-not-pastvus) before you publish images you downloaded with it.

## Install

Run `install.bat` in the repository root first: it installs remove_borders, watermark and classify, whose code and models this tool uses, and then this tool. Or run `install.bat` in this folder after the root one. It installs `httpx`, `rich`, PyTurboJPEG and `zstandard` into the shared `..\venv`, puts `turbojpeg.dll` into `bin\` (PyTurboJPEG does not include the library, and libjpeg-turbo publishes its Windows builds only as installers, so the installer takes the DLL out of conda-forge's `libjpeg-turbo` package, pinned by version and sha256), and makes `pastvu.conf` from `pastvu.conf.example` when there is none.

## The conf file

`pastvu.conf`, next to `pastvu.py`, holds what stays the same from run to run. Edit `[paths]` before the first run:

```
[paths]
base = D:\pastvu                      the base folder of the downloads
samples = D:\samples                  the classify samples folder

[defaults]
years = 1917 1991                     years of a run without --years or --decade
split = 5000                          at most this many images per folder; 0: no split
fetchers = 4                          download threads
json_files = no                       also write each record as a JSON file
priority_margin = 0.15                within this of the top score, the lower-numbered category wins
```

The rest of the file is the **aliases**: names for PastVu region numbers, used with `--region`, in three groups. `[aliases.countries]` has the 15 countries of the former Soviet Union (Ukraine with three presets, see [Ukraine, Crimea and Sevastopol](#ukraine-crimea-and-sevastopol)) and the groups `ussr`, `baltics`, `caucasus` and `central_asia`. `[aliases.regions]` has the first level under each country (oblasts, krais, provinces, counties, republics within Russia). `[aliases.cities]` has the capitals and federal cities, then the places of the second level with at least 1,000 photos (city districts, towns, urban okrugs), with the parent's name added where two places share a name (`leninsky_district_minsk`). Each line has a comment with the PastVu title and the photo count when the file was made:

```
belarus                          = 22                   ; Belarus / Беларусь, 23,422 photos
baltics                          = 149, 154, 267        ; Latvia, Lithuania, Estonia
moscow_oblast                    = 4                    ; Moscow Oblast / Московская область, 97,530 photos
minsk                            = 636                  ; Minsk / Минск, 6,910 photos
```

An alias may name several regions. `run.bat aliases` lists them with the titles PastVu gives the numbers today, and `run.bat aliases --generate FILE` writes a fresh example from today's region tree. Change any alias to fit your dataset.

### Ukraine, Crimea and Sevastopol

The aliases are built from PastVu's region tree, with one exception. PastVu files Crimea (556) and Sevastopol (669) under Russia and marks both as a territorial dispute; that is PastVu's own wording, and it stays in the region titles that PastVu returns and that `count` and `regions` print. The conf lists Crimea and Sevastopol apart from Russia, and Ukraine has three presets:

| alias | regions | what it means |
|---|---|---|
| `ukraine` | 20, 556, 669 | the modern state of Ukraine in its internationally recognized borders |
| `ukrainian_ssr` | 20, 556, 669 | the Ukrainian SSR from 1954 to 1991 |
| `ukraine_pre_1954` | 20 | the Ukrainian SSR before 1954 |

`ukraine` and `ukrainian_ssr` name the same regions; the two names say which of the two meanings a dataset follows. The year 1954 comes from these administrative decisions:

- The Crimean Oblast passed from the RSFSR to the Ukrainian SSR by a decree of the Presidium of the Supreme Soviet of the USSR of 19 February 1954. Before that, Crimea was part of the RSFSR: the Crimean ASSR from 1921, the Crimean Oblast from 1945.
- Sevastopol had been made a city of republican subordination of the RSFSR in 1948. The 1954 decree does not name it, but the city was administered through the Ukrainian SSR from then on, and the 1978 constitution of the Ukrainian SSR lists it among the republic's cities of republican subordination.

Neither Soviet preset is exact for every year. `ukraine_pre_1954` still contains the regions that joined the Ukrainian SSR in 1939, 1940 and 1945 (western Ukraine, northern Bukovina and southern Bessarabia, Zakarpattia), which between the wars belonged to Poland, Romania and Czechoslovakia. For the Ukrainian SSR as it was when each photo was taken, run the two presets for their years:

```
run.bat run --region ukraine_pre_1954 --years 1917 1953
run.bat run --region ukrainian_ssr --years 1954 1991
```

`russia = 1` follows PastVu's tree and so also contains Crimea and Sevastopol. A photo is saved once per base folder, so a Crimean photo is downloaded by whichever of the two scopes runs first, and the other skips it. Abkhazia and South Ossetia are filed under Georgia and Transnistria under Moldova, with the same mark; the aliases of those countries contain them.

`--region` takes an alias first, then a PastVu region number, then a region title (English or local, matched exactly). `run.bat regions --find TEXT` finds numbers and titles.

## The samples folder

The classifier learns the categories from examples that you sort by hand: one subfolder per category in the samples folder that `[paths] samples` names. It is the samples folder of [classify](../classify/README.md#the-samples-folder), and the same folder serves both tools.

1. **One numbered folder per category; the number is a priority.** The name after the number is only a label: it becomes the name of the category folder in the output, and the classifier looks only at the images inside. When an image fits two categories about equally, the one with the lower number takes it: if `60_people` scores within `priority_margin` (0.15) of `100_monuments`, a photo of people in front of a monument goes to `60_people`. The margin is set in the conf or with `--priority-margin`; 0 turns the priority off, and then the higher score always wins. Images of a kind you want in a particular folder, put into that folder, steer the classifier too.
2. **About 40 images per category.** More helps, within reason: from 20 to about 150 per category has worked best. Below 20 the tool warns, below 5 it stops. Choose typical images, not only the clearest ones, and include the ones on the border with a neighbouring category: the classifier learns each border from what you give it.
3. **The last folder for outliers.** Everything you want out of the main categories goes into a last folder named `outliers` after its number, in subfolders by kind: `postcards`, `blurry`, `screenshots`, or a theme you want to keep apart. Each subfolder is learned as its own kind, so unrelated rejects do not blur each other, and all of them are sorted into the one outliers folder in the output. Any category can have subfolders in the same way. As the last folder, outliers has the lowest priority: an image that is about as much a reject as a real category stays in the real category. Give outliers a low number if you want such images out instead.

A structure to start from:

```
samples\
  10_posters\
  20_postcards_and_art\
  30_futurism\
  40_military\
  50_transport\
  60_people\
  70_objects\
  80_environment_indoors\
  90_environment_outdoors\
  100_monuments\
  110_group_photo\
  200_outliers\
    blurry\
    screenshots\
```

Number in steps of 10, so a category can go in between later without renaming the others. The folders sort by the value of the number, so `100_monuments` comes after `90_environment_outdoors`; a folder without a number comes after all numbered ones. The priority is this tool's: [classify](../classify/README.md) itself sorts by the higher score alone. Folders whose names start with `_` are ignored; the samples' embeddings are cached in `_embeddings` inside the samples folder. After you change the samples, `reclassify` sorts what you already downloaded again, in seconds.

## Usage

```
run.bat run --region <alias> [--region ...] [--years FROM TO | --decade YEAR] [options]
run.bat run --scope <name>
run.bat reclassify <scope> [--dry-run] [--undo]
run.bat captions <scope> [--txt-fields LIST] [--dry-run]
run.bat export <scope> [--format jsonl|parquet]
run.bat scopes | aliases | count --region <alias> | regions [--find TEXT]
```

Examples:

```
run.bat count --region baltics --decade 1960
run.bat run --region belarus
run.bat run --region ukraine --decade 1970
run.bat run --region minsk --region brest_oblast --years 1945 1960
run.bat run --scope belarus_1917-1991
run.bat scopes
```

A run downloads a **scope**: regions and years. Its name is the aliases (or region names) joined by `+`, an underscore, and the years: `belarus_1917-1991`, `ukraine_1970s`, `minsk+brest_oblast_1945-1960`. More than three regions need `--name` or an alias of their own. The scope's folder is `<base>\<name>`, and running the same command again continues it, as does `run --scope <name>`.

`--years` takes the photos whose dating overlaps the years, as PastVu's own filter does: a photo dated 1910-1925 is in 1917-1991. `--decade 1970` takes the photos whose dating starts in 1970-1979, so a photo dated 1965-1975 belongs to the 1960s only, and decade runs never download a photo twice. Only photographs are taken, not paintings.

**A photo is downloaded once per base folder.** The store at `<base>\pastvu.db` is shared by every scope, so a photo already saved under one scope (Minsk) is not downloaded again for another (Belarus); the report counts it as already saved under another scope, and `scopes` shows how many there are.

## What happens to one photo

1. **Fetch** (download threads): the PastVu record; the photo is skipped when it is smaller than `--min-side` (512) on either side, a painting, or not public. Then the original file, which is the picture with PastVu's grey band appended at the bottom (`pastvu.com/<cid> uploaded by ...`). The file must have the size the record gives, picture plus band, or the photo counts as failed.
2. **Band and borders** (download threads): the band rows are known from the record and come off exactly. [remove_borders](../remove_borders/README.md) analyses the picture without the band for scanned frames, mounts, thin lines and text banners.
3. **Watermarks** (GPU): the [watermark](../watermark/README.md) detector runs on the picture without the band, so the band's own text is never taken for a watermark. Detections from `--wm-conf` (0.1) go into the record; only those from `--wm-paint-conf` (0.4) are painted out with LaMa. Below 0.4 the detector often finds scene text and objects (a shop sign, a building facade); from 0.4 up it found printed captions and credits in every case checked.
4. **Crop**: the band, the border cuts and a watermark trim (`--wm-mode trim`) make one crop, put on the JPEG block grid: the top and left edges move inward to the next block where a cut set them, so the cut part goes completely, and the far edges are exact.
5. **Classify** (GPU): the final pixels are embedded with SigLIP 2 and scored by classify's probe, trained once at start on the samples folder. Among the categories within `priority_margin` of the top score, the lowest-numbered takes the image ([The samples folder](#the-samples-folder)). Then classify's gates: the summed score of those near categories against the confidence threshold (`--min-confidence`, 0.7) and their summed share of the 10 nearest examples against 0.3; an image that fails goes to `_unsure` under the chosen category. A byte-identical copy of a sample takes the sample's category. The record keeps the score of the chosen category and, when the priority chose it over a higher score, which category that was.
6. **Write** (writer threads): the image and its caption go into the next folder of the category with room; then the photo's status, its record, its embedding and the scope's counters are committed to the store in one transaction.

Pixels stay in memory from the download to the write; no temporary file is made. How an image is written:

| what was done | written as |
|---|---|
| nothing | the downloaded bytes, unchanged |
| only cuts (band, borders, trim) | JPEG: cropped in the DCT domain with libjpeg-turbo, every coefficient kept, EXIF kept with the new size. PNG: the exact pixels in the same format and mode. |
| a watermark painted | JPEG at quality 97 without chroma subsampling, or PNG with `--png` |

A JPEG crop keeps every coefficient of the kept blocks; the decoded pixels can still differ in the last row or column at a cut, where the decoder's chroma upsampling no longer sees the cut-off neighbours. An image that is smaller than `--min-side` after its crop is skipped.

## Output

```
<base>\
  pastvu.db                                     the store of every scope
  belarus_1917-1991\
    10_environment_outdoors\001\163150.jpg      the image, named by its PastVu number
    10_environment_outdoors\001\163150.txt      its caption
    10_environment_outdoors\002\...             the next 5,000 images of the category
    _unsure\5_transport\001\165001.jpg          below the confidence gate, under its best guess
  ukraine_1970s\
    ...
  _records\016\163150.json                      with json_files = yes only
  _reports\                                     reclassify --dry-run plans
  belarus_1917-1991_records.jsonl               written by export
```

Each category is split into numbered folders `001`, `002`, ... of at most `split` images (5,000 by default, so 10,000 files with the captions; Explorer slows down in folders much larger than that). A category stays one tree: AI Toolkit and the dataset tools read subfolders, so point them at the category folder. `--split` sets another size for a new scope; with `--no-classify` the numbered folders are directly in the scope folder.

**The caption** is one line of facts for the captioner, in the record's language with the English region names added: title, years, regions, address and description, in that order, empty parts left out (`--txt-fields` chooses the parts and their order; `direction` adds the camera direction). Example:

```
Холмские ворота. Брестская крепость. 1900-1930. Беларусь, Брестская область, Брест, Брестская крепость (Belarus, Brest Oblast, Brest, Brest Fortress).
```

PastVu's placeholder titles ("Где это?", alone or in front of a guess) are left out, and so are the parts of a description that held a link, a publication permission, a note on where the map point stands, a pointer to another upload, or a separator line. No URLs and no user names go into the caption. The [taggui_captioning](../taggui_captioning/README.md) prompt reads it as the existing caption; change its sentence that says English words in the caption describe image quality, because here they are region names. `captions <scope>` rewrites every caption of a scope from the records with the current rules and `--txt-fields`, without downloading anything.

**The record** of each photo is in the store: the PastVu record as the API gives it (title, description, address, author, source, years, coordinates, direction, regions, uploader), the download (URL, size, sha256), what was done (band, border cuts, every watermark detection with its confidence, the painted boxes, the crop, how the file was written) and the classification. `export <scope>` writes the records of a scope as JSON Lines (one full record per line) or, with `--format parquet`, as Parquet with the main fields as columns and the full record as a JSON string. `json_files = yes` in the conf also writes each record to `<base>\_records\<number>\<cid>.json`, where the number is the PastVu number divided by 10,000, so no folder holds more than 10,000 files; `--json-sidecar` writes it next to the image instead.

## Reclassify

```
run.bat reclassify belarus_1917-1991 --dry-run
run.bat reclassify belarus_1917-1991
run.bat reclassify belarus_1917-1991 --undo
```

Sorts the saved photos of a scope again with the samples folder as it is now (or `--samples`) and the priority margin of the conf (or `--priority-margin`), and moves each image and its caption into its new category folder, in place. The file keeps its name, so the caption and the record stay joined. The embeddings are in the store, so the GPU is needed only for images without one, and the probe trains in seconds.

A photo you moved to another category by hand keeps its place: reclassify sees that its file is not where the store says, records where it is now, and leaves it there (`--override-manual` sorts it like the rest). A photo whose file is gone is reported and left alone. `--dry-run` prints the moves per category pair and writes them to `<base>\_reports\`. `--undo` puts the moves of the last reclassify back.

## Progress and resume

The live display shows the current activity on a status line ("163150: detecting watermarks", "163150: saving 5_transport/001/163150.jpg", "waiting for downloads (4 in flight)") above a progress bar with the rate and the time left. Every `--report-every` photos (500) a report block is printed above them and stays: photos saved, skipped by reason and already saved under another scope, band and border cuts, watermarks painted and detected below the threshold, how the files were written, the categories, and the time per stage. `--quiet` prints the report blocks only; `--json-progress` prints one JSON line per photo.

Ctrl+C finishes the photos already downloaded and stops; a second Ctrl+C stops at once. Run the same command again and it continues with every photo that is not done. Before a photo is written its path is claimed in the store; a photo found claimed but not done at the start of a run was interrupted while writing, so its files are deleted and it is done again. A photo that failed is tried again on the next run, up to `--retries` (3) times.

## Speed

The downloads set the pace: on an RTX 5090 the GPU work (detection, painting, classification) takes about 0.17 s per photo, and the models together hold 2 GB of GPU memory. Measured on Belarus, 19,523 photos:

| download threads | photos per second | requests to PastVu per second |
|---|---|---|
| 4 (default) | 5.2 | about 7 |
| 8 | 6.6 to 9 | about 13 |

Each saved photo costs one API call and one image download; a photo skipped as too small costs the call only. There is no delay between requests. PastVu is run by volunteers: `--max-per-second N` caps the requests, and the client always backs off when the server answers 429 or 5xx or a Cloudflare page.

## Options of `run`

| option | default | meaning |
|---|---|---|
| `--region ALIAS\|CID\|TITLE` | | a region; repeat for several; needed for a new scope |
| `--years FROM TO` | conf | photos whose dating overlaps these years |
| `--decade YEAR` | | photos whose dating starts in this decade |
| `--name NAME` | `<aliases>_<years>` | the scope's name |
| `--scope NAME` | | resume a scope by its name or folder |
| `--split N` | conf (5000) | at most N images per folder, for a new scope; 0: no split |
| `--samples DIR` | conf | the classify samples folder |
| `--no-classify` | off | no categories |
| `--min-confidence X` | 0.7 | below this an image goes to `_unsure` |
| `--priority-margin X` | conf (0.15) | within this of the top score, the lower-numbered category wins; 0: off |
| `--min-side PX` | 512 | skip pictures smaller than this on either side, before and after the crop |
| `--png` | off | write painted images as PNG |
| `--no-watermark` | off | no watermark detection or painting |
| `--wm-conf X` | 0.1 | detections from this confidence go into the record |
| `--wm-paint-conf X` | 0.4 | detections from this confidence are painted out |
| `--wm-mode inpaint\|trim` | inpaint | paint, or cut the largest clear rectangle when it keeps half the image |
| `--no-borders` | off | no border analysis (the band is still cut) |
| `--no-dark` | off | leave dark borders alone |
| `--txt-fields LIST` | title,years,regions,address,desc | the caption parts in order; `direction` is the extra one |
| `--json-sidecar` | off | also write the record next to the image |
| `--comments` | off | add the comment thread to the record |
| `--fetchers N` | conf (4) | download threads; 8 is about 30 to 70 % faster and doubles the load on PastVu |
| `--depth N` | 16 | photos downloaded ahead of the GPU |
| `--writers N` | 2 | encoder and writer threads |
| `--limit N` | | do at most N photos in this run |
| `--retries N` | 3 | attempts for a photo that failed |
| `--report-every N` | 500 | a report block every N photos |
| `--quiet`, `--json-progress` | | report blocks only, or one JSON line per photo |
| `--full-parent` | off | page a large region itself instead of through its children |
| `--reindex` | off | page the scope again, to find photos published since |

Before the subcommand: `--conf FILE` another conf, `--base DIR` another base folder, `--max-per-second N` a cap on the requests to PastVu.

## How the index works

PastVu's API has no text search and no topics; it filters by region, years and type. The index pages PastVu's gallery for the scope, 100 photos a page, into the store. A region with more than 50,000 photos in scope is paged through its children in the region tree, plus the photos of the region that have no coordinates; geotagged photos that lie in the region but in none of its children are reported and taken only with `--full-parent`, which pages the region itself and costs the server more. A finished index is not paged again unless `--reindex`.

## The photographs are not PastVu's

PastVu's [rules](https://docs.pastvu.com/en/rules) let site users reproduce the published images free of charge for personal, non-commercial purposes (rule 1.15) and let anyone save them (rule 1.20). Commercial use needs the permission of the rights holder of each image (rule 1.16), and PastVu says it is not that rights holder (rule 1.19): the photographs belong to their photographers, archives, museums and publishers, which the record's `author` and `source` fields name. Nothing in the rules forbids bulk downloads, and nothing grants the right to publish the images again.

Most photographs of the Soviet era are still protected somewhere. In Russia a photograph is protected for the author's life plus 70 years, plus 4 years for those who served in the Second World War, and the 2008 Civil Code restored protection to works the short Soviet terms had released; an anonymous work is protected for 70 years from publication. Other countries of the former Soviet Union have their own terms. In the United States, foreign works published from 1931 to 1977 that were still protected at home on 1 January 1996 are protected for 95 years from publication. Text and data mining exceptions cover the analysis and a local copy, not the publication of the files. This is not legal advice.

Each record carries what attribution and a takedown need: the PastVu page, the uploader, the author, the source and the years.

## Project files

| file | purpose |
|---|---|
| `pastvu.py` | the command line, and the example conf generator |
| `conf.py` | reads `pastvu.conf` |
| `client.py` | the API client: retries, back-off, optional cap |
| `store.py` | the SQLite store of a base folder |
| `index.py` | the index of a scope, and how `--region` is resolved |
| `layout.py` | scope names, folders and the numbered-folder allocator |
| `process.py` | header, decode, band, borders, crop composition |
| `encode.py` | the save rules, in memory |
| `classifier.py` | classify's sorter, one image at a time or in batches |
| `reclassify.py` | sorting a scope again, and undo |
| `export.py` | records as JSON Lines or Parquet |
| `sidecars.py` | the caption and the record |
| `progress.py` | the live display and the report blocks |
| `stream.py` | the threads and the run |
| `fetch_turbojpeg.py` | fetches `turbojpeg.dll` into `bin\` |
| `pastvu.conf.example` | the example conf with the aliases |
| `install.bat`, `run.bat`, `requirements.txt` | install into the shared venv, run with it |
