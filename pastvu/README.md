# PastVu

Downloads historic photographs from [PastVu](https://pastvu.com) by region and years, and prepares each one for a training dataset while it downloads: it cuts the grey PastVu band and any scanned border, paints out watermarks and printed captions, sorts the image into the categories of a [classify](../classify/README.md) samples folder, and writes it with a one-line caption of its facts. A local vision-language model can then write the final captions, during the download or afterwards ([Autocaption](#autocaption)). One image after another, with every model kept on the GPU, and with a store that lets a stopped run continue where it was.

The tool uses PastVu's public API (no account, no key). Read [The photographs are not PastVu's](#the-photographs-are-not-pastvus) before you publish images you downloaded with it.

## What it is

The tool turns PastVu, the big archive of historic photos of the former Soviet Union, into a ready training dataset. You name a place and a time span; it downloads the photos, cleans them, sorts them into your categories and gives each one a caption.

### What one run does

You give a place and years, for example “Belarus, 1917 to 1991” or “Ukraine in the 1970s”. Places have short names in a settings file: all 15 former Soviet republics, groups such as “baltics”, every oblast and region, and the main cities. The tool asks PastVu for the list of matching photos, then handles them one at a time:

1. **Download:** it fetches the full-size original. It skips photos smaller than 512 pixels on either side, and paintings.
2. **Clean:**
   - It cuts off PastVu’s grey “pastvu.com, uploaded by…” strip at the bottom.
   - It trims scanned frames and white borders.
   - It paints out printed captions, photographer credits and watermarks. It paints only marks it is confident about, so real shop signs and buildings are left alone.
   - Cropping never lowers JPEG quality; only painting re-saves an image.
3. **Black-and-white or colour:** a fast pixel test tells colour photos from black-and-white ones, and counts sepia, toned prints and yellowed paper as black-and-white.
4. **Sort:** it first leaves out low-quality photos, such as film and TV frame grabs and blurry or washed-out scans, learned from two folders of examples. Then it puts the photo into your categories, such as transport, people or monuments, learned from example images you put in a samples folder. When two categories fit about equally, the one with the lower folder number wins. Every photo goes into a category, a doubtful one into its best guess.
5. **Caption:** it writes a short facts line from PastVu’s record: title, years, place and description.
6. **Describe with a model (optional):** a vision model on your own machine, currently Gemma 4 in KoboldCpp, writes a full English description. It starts from the facts line and adds what is visible in the photo.

Downloads are split into black-and-white and colour first, then filed by decade and place (`bw\1970s\russia\transport`, `color\1970s\russia\transport`), or by place first, or in one folder per download, as the settings file says. A category that grows past 5,000 images is split into numbered folders, so Explorer stays fast.

### Safety and convenience

- **Stop and resume:** stop at any time; the same command continues where it left off. I killed runs on purpose several times, and nothing was lost or half-written.
- **No double downloads:** a photo is downloaded only once, even if two runs cover the same area.
- **Re-sorting:** after you improve your samples, `reclassify` re-sorts everything already downloaded in seconds. It never undoes your manual moves, and it can be undone. Only the deletion of low-quality photos cannot be undone, so it asks first.
- **Splitting and regrouping:** a country can be downloaded straight into one folder per decade, region or country. Photos you already have can be moved between such folders later without downloading them again, and every regroup can be undone ([Splitting and regrouping](#splitting-and-regrouping)).
- **Captioning later:** `autocaption` captions photos you already have, a whole set or one folder, with a progress bar, statistics and resume. Old captions are backed up and can be restored.
- **Live display:** a status line shows what is happening and how much GPU memory is in use, above a progress bar. Regular summaries report counts, skips, paints and speed.
- **Export:** every photo’s full PastVu record is kept, and `export` writes all records as one file for publishing.

### Speed

- **Download and clean:** about 5 photos per second; PastVu’s server is the limit.
- **With model descriptions:** about 1,300 photos per hour, roughly 3 seconds each.

### Already done

- **Belarus:** 18,218 photos, cleaned and sorted.
- **Estonia:** a county from 1917 to 1991, and the whole country for the 1970s.
- **Captions:** one test folder is fully described by the model.

The README covers setup, the samples folder, the history behind the three Ukraine options, and the copyright status of Soviet-era photos.

## Install

Run `install.bat` in the repository root first: it installs remove_borders, watermark and classify, whose code and models this tool uses, and then this tool. Or run `install.bat` in this folder after the root one. It installs `httpx`, `rich`, PyTurboJPEG, `zstandard` and `nvidia-ml-py` into the shared `..\venv`, puts `turbojpeg.dll` into `bin\` (PyTurboJPEG does not include the library, and libjpeg-turbo publishes its Windows builds only as installers, so the installer takes the DLL out of conda-forge's `libjpeg-turbo` package, pinned by version and sha256), and makes `pastvu.conf` from `pastvu.conf.example` when there is none.

## The conf file

`pastvu.conf`, next to `pastvu.py`, holds what stays the same from run to run. Edit `[paths]` before the first run:

```
[paths]
base = D:\pastvu                      the base folder of the downloads
samples = D:\samples                  the classify samples folder

[defaults]
years = 1917 1991                     years of a run without --years or --decade
layout = years/place                  the folder tree: years/place, place/years or flat
bw_color = yes                        black-and-white and colour photos in two trees, bw\ and color\
split = 5000                          at most this many images per folder; 0: no split
fetchers = 4                          download threads
json_files = no                       also write each record as a JSON file
priority_margin = 0.15                within this of the top score, the lower-numbered category wins
quality_threshold = 0.88              skip photos the quality probe scores at or above this; 0: off

[caption]                             the VLM of autocaption; see Autocaption
api_url = http://localhost:5001/v1
system_prompt = tagging/sysprompt.txt
prompt = tagging/main_prompt.txt
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

### Low-quality photos

Two more folders in the samples folder teach a second probe which photos to leave out: `_quality_good` and `_quality_bad`. Put sharp, ordinary photos into the first. Put the kind you do not want into the second: film and TV frame grabs, and blurry, soft, foggy or washed-out scans. Use photos of at least 512 pixels on the short side; the download skips smaller ones anyway, so they teach the probe little. Each folder needs at least 20 images, or the filter is off.

The probe reads the same SigLIP 2 embedding as the category classifier, so it needs no second model on the GPU, and its time per photo is too small to measure. A photo that it scores at or above `quality_threshold` (P(bad) 0.88 by default) is skipped like a photo under 512 pixels. It is not saved, the store marks it skipped with the reason `low quality` and its score, and no later run downloads it again. `reclassify` deletes the saved photos that score that high ([Reclassify](#reclassify)). The threshold is set in the conf only, in `[defaults]`; a conf without it gets `quality_threshold = 0.88` written into it the first time it is used, and 0 turns the filter off.

At 0.88, with 94 good and 131 bad samples, about 3 % of 28,620 downloaded photos scored that high. Nearly all of them were film and TV frame grabs and soft scans; about one in eight was a photo worth keeping. The scores move whenever the two folders change, so check the threshold again after you edit them: `reclassify --dry-run` lists every photo it would delete in `<base>\_reports\`. Label both kinds: when you add only bad examples, all scores go up.

Other folders whose names start with `_`, such as a folder of photos that you did not sort yet, are ignored by both probes.

## Usage

```
run.bat run --region <alias> [--region ...] [--years FROM TO | --decade YEAR] [options]
run.bat run --region <alias> --by decade|country|region|country,decade|region,decade [--regroup]
run.bat run --scope <name>
run.bat regroup --by RULE [<scope> ...] [--dry-run] [--undo]
run.bat drop <scope> [...] [--yes]
run.bat reclassify [<scope> ...] [--dry-run] [--undo] [--yes]
run.bat captions <scope> [...] [--txt-fields LIST] [--dry-run]
run.bat autocaption <scope or folder> [...] [--limit N] [--recaption] [--undo]
run.bat export <scope> [...] [--format jsonl|parquet]
run.bat extract <category> [...] [--region R] [--years A B | --decade D] [--bw | --color] [--min-res N]
                   [--target N [--colour-share X]] [-o DIR]
run.bat scopes | aliases | count --region <alias> | regions [--find TEXT]
```

Where a command takes scopes, a scope can be given by its name, by its folder, or by a name pattern with `*` and `?`: `estonia_*` is every scope whose name starts with `estonia_`. A folder that holds scope folders stands for all of them: `color` (or `<base>\color`) is the colour tree of every scope, and `color\1970s` the colour photos of every scope of the 1970s. In Git Bash or PowerShell, put a pattern in quotes.

Examples:

```
run.bat count --region baltics --decade 1960
run.bat run --region belarus
run.bat run --region ukraine --decade 1970
run.bat run --region minsk --region brest_oblast --years 1945 1960
run.bat run --scope ukraine_1970s
run.bat scopes
```

### Common tasks

**Download 5,000 photos of a region (here Ukraine):**

```
run.bat run --region ukraine --limit 5000
```

The first run indexes all of Ukraine for the years of the conf (one gallery page per 100 photos, so a few minutes for a large country), then processes 5,000 photos and stops. `--limit` counts the photos processed in the run: those saved and those skipped as too small (about 7 % in Belarus), so expect about 4,650 images. Photos are taken in the order of their PastVu number, which is the order of upload, oldest first, from all decades at once: in the default layout they go into `1910s\ukraine\`, `1920s\ukraine\` and so on up to `1990s\ukraine\`.

**Download the next 1,000 (photos 5,001 to 6,000):**

```
run.bat run --region ukraine --limit 1000
```

The same command continues with the photos that are not done yet, in the same order, so it takes the next 1,000; the index is not paged again. A photo that failed in an earlier run is tried again first when its number is lower.

**Download 1,000 photos starting from photo 5,001, without the first 5,000:**

```
run.bat run --region ukraine --skip 5000 --limit 1000
```

`--skip N` starts at photo N+1 of everything the scope (or the scopes of a decade split) lists, in the same PastVu number order. It counts positions in the whole list, done or not: after a first run of 5,000, `--skip 5000 --limit 1000` takes the same 1,000 as the example above. The photos before that position that are not done are not downloaded and stay to do; a later run without `--skip` downloads them. This lets two base folders, or two machines, share out one region: one runs `--limit 5000`, the other `--skip 5000`.

**Resume a download:**

```
run.bat run --region ukraine
run.bat run --region ukraine --resume
run.bat run --scope ukraine_1970s
```

Without `--limit`, the same command runs until every photo is done. Running it again first goes over the index: finished shards are not paged again, but planning the shards asks PastVu for the photo count of every large region and its children, which takes a while for a big country. `--resume` skips the index altogether and downloads what the store lists already; it says how much of the index is done ("189 of 189 shards finished, 0 partly paged, 0 not started"), and when the index is not finished, a run without `--resume` lists the rest. Ctrl+C finishes the photos in progress and stops; a second Ctrl+C stops at once. `--scope` continues one scope only, here the 1970s. A photo that failed is tried again on the next run, up to `--retries` (3) times; a photo skipped as too small is not. `--reindex` pages the index again to find photos published since the first run.

A run downloads a **scope**: regions and years. Its name is the aliases (or region names) joined by `+`, an underscore, and the years: `belarus_1917-1991`, `ukraine_1970s`, `minsk+brest_oblast_1945-1960`. More than three regions need `--name` or an alias of their own. Running the same command again continues it, as does `run --scope <name>`. The scope's folder follows `layout` in the conf (see [Folder layouts](#folder-layouts)): `<base>\1970s\ukraine\` by default.

`--years` takes the photos whose dating overlaps the years, as PastVu's own filter does: a photo dated 1910-1925 is in 1917-1991. `--decade 1970` takes the photos whose dating starts in 1970-1979, so a photo dated 1965-1975 belongs to the 1960s only, and decade runs never download a photo twice. Only photographs are taken, not paintings.

**A photo is downloaded once per base folder.** The store at `<base>\pastvu.db` is shared by every scope, so a photo already saved under one scope (Minsk) is not downloaded again for another (Belarus); the report counts it as already saved under another scope, and `scopes` shows how many there are.

**One command at a time changes a base folder.** The store holds a lock. A second download into the same base folder stops at the start with the name of the command that holds the lock, and so does a `regroup`, `drop` or `reclassify` while anything else runs. `autocaption` and `captions` can run beside a download. A lock left by a process that no longer runs is taken over.

## Splitting and regrouping

```
run.bat run --region estonia --by decade
run.bat run --region baltics --by country,decade
run.bat run --region estonia --regroup
run.bat regroup --by decade estonia_1917-1991 --dry-run
run.bat regroup --by country
run.bat regroup --undo
run.bat drop estonia_1950s --yes
```

**`run --by RULE`** downloads the regions and years of the run into one scope per value of the rule, made as the photos turn up (in a years/place or place/years base the decade is always part of the rule, see [Folder layouts](#folder-layouts)). `run --region estonia --by decade` indexes Estonia once for the years of the conf and fills `estonia_1910s` up to `estonia_1990s`, each with its own folder, categories and numbered folders, in one download with one progress bar; the final report adds a line per scope. The rules:

- **decade:** the decade of the photo's start year, so every photo belongs to exactly one decade. A run keeps the decades its years cover: 1917-1991 gives the 1910s to the 1990s. PastVu's filter also lists photos whose dating only overlaps the years, such as 1890-1920; such a photo starts in another decade, joins no scope and is not downloaded. `--decade` or `--years` reaches those decades when you want them.
- **country:** the country alias of the conf that names the most specific region of the photo; on a tie the alias higher in the conf wins. Aliases that name several countries (`ussr`, `baltics`, ...) are never used. So Crimea and Sevastopol go to `ukraine`, which names them, and not to `russia`.
- **region:** the first level under the country in PastVu's tree (an oblast, a county, a federal city, Crimea), named by its alias in the conf or else by its English title. A photo placed at country level only goes to its country.
- **country,decade** and **region,decade** combine the two: `estonia_1970s`, `harju_county_1980s`.

A scope made by a rule is the same as one made by hand: `estonia_1970s` from `--by decade` is the scope that `--region estonia --decade 1970` makes, so the two meet instead of clashing. A scope of that name with other regions or years stops the run. The same command continues every scope of the split; `run --scope estonia_1970s` continues one of them.

**`run --regroup`** first moves every saved photo that the run's scopes list into them, from whatever scope holds it now, and then downloads the rest. `run --region estonia --decade 1960 --regroup` pulls the 1960s photos out of `estonia_1917-1991`; `run --region estonia --regroup` pulls everything back from the decade scopes into `estonia_1917-1991`; `run --region estonia --by decade --regroup` moves an all-years download into its decades. The last regroup wins.

**`regroup --by RULE [SCOPE ...]`** moves photos by a rule and downloads nothing. With scopes it splits them and keeps their label: `regroup --by decade estonia_1917-1991` makes `estonia_1920s` and so on. Without scopes it groups every scope of the base folder: by country, photos of `tartu_county_1970s` join `estonia_1970s`; by decade alone, the regions of all scopes are combined into `all_1970s` (scopes that all have the same regions keep their label). The years of a scope made by the country or region rule are those of the scopes its photos come from: the decade when they share one, else the span of their years. With two keys, photos of different countries or different decades never meet; photos of the same country and decade from different scopes do. The photos' listings move with them, so a later run of a new scope continues it. `--dry-run` prints the moves per pair of scopes, and the scopes made and removed.

Both kinds of regroup move files the same way:

- The image and its caption are renamed into the new scope (same drive, so at once), into the lowest numbered folder with room, under the category folder the file is in now: photos you sorted by hand keep their category. Records, embeddings and VLM captions are kept by the photo's number and follow.
- A saved photo whose file is gone (deleted by hand) is marked not downloaded, and the next run downloads it again.
- A scope left with no photos, whose listed photos all belong to the new scopes, is removed with its empty folders. Files in it that are not this tool's stay, and so does the folder.
- The counters of every scope involved are counted again from the records.
- The moves are logged in the store: `regroup --undo` puts the last regroup (or `run --regroup`) back and makes again the scopes it removed. A reclassify of a moved photo can no longer be undone. An undo puts each photo back into the category it came from, in a numbered folder with room when that category was chunked in the meantime.
- Scopes sorted with different samples can mix category names when they are combined; `reclassify` sorts them again.

**`drop SCOPE`** deletes the scope's images and captions and forgets the scope. Its photos count as not downloaded, with no record, embedding or VLM caption, so a later run downloads and processes them from the start; photos it lists that are saved under another scope stay there. A drop cannot be undone: without `--yes` it only shows what it would delete.

## What happens to one photo

1. **Fetch** (download threads): the PastVu record; the photo is skipped when it is smaller than `--min-side` (512) on either side, a painting, or not public. Then the original file, which is the picture with PastVu's grey band appended at the bottom (`pastvu.com/<cid> uploaded by ...`). The file must have the size the record gives, picture plus band, or the photo counts as failed.
2. **Band and borders** (download threads): the band rows are known from the record and come off exactly. [remove_borders](../remove_borders/README.md) analyses the picture without the band for scanned frames, mounts, thin lines and text banners.
3. **Watermarks** (GPU): the [watermark](../watermark/README.md) detector runs on the picture without the band, so the band's own text is never taken for a watermark. Detections from `--wm-conf` (0.1) go into the record; only those from `--wm-paint-conf` (0.4) are painted out with LaMa. Below 0.4 the detector often finds scene text and objects (a shop sign, a building facade); from 0.4 up it found printed captions and credits in every case checked.
4. **Crop**: the band, the border cuts and a watermark trim (`--wm-mode trim`) make one crop, put on the JPEG block grid: the top and left edges move inward to the next block where a cut set them, so the cut part goes completely, and the far edges are exact.
5. **Classify** (GPU): the final pixels are embedded with SigLIP 2 and scored by classify's probe, trained once at start on the samples folder. Among the categories within `priority_margin` of the top score, the lowest-numbered takes the image ([The samples folder](#the-samples-folder)). Then classify's gates: the summed score of those near categories against the confidence threshold (`--min-confidence`, 0.7) and their summed share of the 10 nearest examples against 0.3; an image that fails still goes into the chosen category, its best guess, and the record keeps the reason (`low confidence` or `disagree`), so doubtful photos can be found later. A byte-identical copy of a sample takes the sample's category. The record keeps the score of the chosen category and, when the priority chose it over a higher score, which category that was. Before the categories, the quality probe scores the same embedding, and a photo at or above `quality_threshold` is skipped ([Low-quality photos](#low-quality-photos)).
6. **Black-and-white or colour** (CPU, about 0.06 s): the pixel test on the final picture, with the painted watermark boxes left out; a borderline photo is decided by the probe on its SigLIP 2 embedding ([Black-and-white and colour](#black-and-white-and-colour)).
7. **Write** (writer threads): the image and its caption go into the next folder of the category with room, under `bw\` or `color\`; then the photo's status, its record, its embedding and the scope's counters are committed to the store in one transaction.

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
  bw\                                           black-and-white photos
    1970s\
      belarus\                                  the black-and-white photos of the scope belarus_1970s
        50_transport\163150.jpg                 the image, named by its PastVu number
        50_transport\163150.txt                 its caption
        90_environment_outdoors\001\165001.jpg  a category past 5,000 images: numbered folders
        90_environment_outdoors\002\...         the next 5,000 images of the category
      ukraine\
    1980s\
      ...
  color\                                        colour photos, the same tree
    1970s\
      belarus\
  _records\016\163150.json                      with json_files = yes only
  _reports\                                     reclassify --dry-run plans
  belarus_1970s_records.jsonl                   written by export
```

This is the default layout. [Folder layouts](#folder-layouts) shows the other two.

### Folder layouts

`layout` in `[defaults]` of `pastvu.conf` decides how the scope folders of a base folder are arranged. There are three. With `bw_color = yes` (the default) the whole tree is there twice, once under `bw\` and once under `color\` ([Black-and-white and colour](#black-and-white-and-colour)); the examples below leave that level out.

| layout | the tree | `ukraine_1970s` goes into |
|---|---|---|
| `years/place` (default) | decade, then place | `<base>\1970s\ukraine\` |
| `place/years` | place, then decade | `<base>\ukraine\1970s\` |
| `flat` | one folder per scope | `<base>\ukraine_1970s\` |

How they work:

- **A scope has a place and years.** The place is the label of the run: the alias you give (`--region estonia` gives `estonia`, `--region baltics` gives `baltics`), the aliases joined by `+`, `--name`, or the country or region that `--by` finds for each photo. The years are a decade (`1970s`) or a span (`1917-1991`).
- **`years/place` and `place/years`: every scope is one decade.** A run over a span of years puts each photo into the decade it starts in, as if `--by decade` were given. It keeps the decades the span covers: 1917-1991 gives `1910s` to `1990s`, and a photo dated 1890-1920, which PastVu also lists, joins no scope. `--decade 1970` makes the one scope of that decade.
- **`flat`: one folder per scope, named like the scope.** A run makes one scope for all its years (`estonia_1917-1991`) unless `--by` splits it.
- **Countries need `--by country`.** A group alias is one place: `--region baltics` makes `1970s\baltics\`. `--by country` splits it into `1970s\estonia\`, `1970s\latvia\` and `1970s\lithuania\` (in the tree layouts the decade is added to the rule by itself). `--by region` makes one folder per oblast, county or region (`1970s\harju_county\`), without a country level.
- **A base folder keeps its layout.** The store records the layout when the first scope is made, and a base made before this setting existed is `flat`. When the conf has no `layout` line, the tool writes one into `[defaults]`: the layout of the base when it already has scopes, else `years/place`. A `layout` in the conf that differs from the layout of an existing base is reported, and the base keeps its own. To use another layout, set it in the conf and start a new base folder: change `[paths] base`, or give `--base` before the command. An existing base cannot be changed to another layout.

#### Decade, then country (default)

```
[defaults]
layout = years/place
```

```
run.bat run --region estonia                       1910s\estonia\, 1920s\estonia\, ... 1990s\estonia\
run.bat run --region baltics --by country          1970s\estonia\, 1970s\latvia\, 1970s\lithuania\, ... per decade
run.bat run --region ussr --by country             every former Soviet republic, a folder per decade
run.bat run --region ukraine --decade 1970         1970s\ukraine\ only
```

```
<base>\
  1960s\
    estonia\
    latvia\
  1970s\
    estonia\
      50_transport\
      90_environment_outdoors\
    latvia\
```

Each decade is one folder with a folder per country in it, so a trainer can be given one folder per decade with its own number of repeats.

#### Country, then decade

```
[defaults]
layout = place/years
```

The commands are the same as above:

```
run.bat run --region estonia                       estonia\1910s\, estonia\1920s\, ... estonia\1990s\
run.bat run --region baltics --by country          estonia\1970s\, latvia\1970s\, lithuania\1970s\, ...
```

```
<base>\
  estonia\
    1960s\
    1970s\
      50_transport\
      90_environment_outdoors\
  latvia\
    1960s\
    1970s\
```

#### Flat

```
[defaults]
layout = flat
```

```
run.bat run --region estonia                       estonia_1917-1991\ (all the years in one folder)
run.bat run --region estonia --decade 1970         estonia_1970s\
run.bat run --region estonia --by decade           estonia_1910s\, estonia_1920s\, ... estonia_1990s\
run.bat run --region baltics --by country,decade   estonia_1970s\, latvia_1970s\, lithuania_1980s\, ...
```

```
<base>\
  estonia_1917-1991\
    50_transport\
    90_environment_outdoors\
  latvia_1970s\
```

A conf serves every base folder, and each base keeps the layout it was made with. To keep a flat base next to a default one, set `layout = flat` before the first run of the flat base:

```
run.bat --base D:\pastvu_flat run --region estonia
```

### Black-and-white and colour

`bw_color = yes` in `[defaults]` (the default for a new base) puts every photo into one of two trees, `<base>\bw\` and `<base>\color\`, each arranged by `layout`: `bw\1970s\russia\50_transport\` and `color\1970s\russia\50_transport\`. Training needs the two balanced, and the decades differ widely in how many colour photos they hold, so `scopes` and the report of every run give the black-and-white and colour counts of each scope.

**The test.** A toned print (sepia, selenium, cyanotype, yellowed paper) has a colour that depends only on brightness; a colour photo also has colour in other hues. The tool fits the tone curve of the picture (the median colour per brightness band, in Lab), takes the colour that is left, and keeps the part that points away from the tone's own hue. A photo is colour when at least 1 % of the picture has more than 6 units of such off-hue colour, or 3 % has more than 12 units of leftover colour. The test runs on the final picture (after the band, border and watermark cuts) with the painted watermark boxes left out, on its central 90 % reduced to 128 pixels, in about 0.06 s on the CPU. Sepia, toned and split-toned prints, poor scans and yellowed paper count as black-and-white; aged and faded colour prints count as colour.

**Borderline photos.** About 10 % of photos lie close to those lines. A run that sorts into categories (the default) already has each photo's SigLIP 2 embedding, and a small probe decides the borderline ones: a logistic regression trained at start on the samples folder images that the pixel test is sure about, so it needs no labels of its own. The pixel scores of the samples are cached in `<samples>\_embeddings\pastvu_colour.json`, so only the first run pays the few seconds. With `--no-classify` the pixel test decides alone; the encoder is never loaded for this test only.

**How good it is.** Calibrated on 1,280 sample images, 264 of them checked by hand, most of those borderline: the pixel test alone missed 3 colour photos and called 10 black-and-white photos colour; with the probe it missed no colour photo and called 8 black-and-white photos colour, half of them one-ink posters, documents on cream paper or photos with a coloured stamp. That is about 1 % of all photos either way, and the misses are near-neutral colour photos (a grey river scene, a white TV set) and black-and-white prints with uneven toning or stains.

**Correcting by hand.** Move a photo's image (with its caption) from `bw\` to `color\` or back. `reclassify` then treats it as a hand-sorted photo: it records the new mode in the store and the record, and leaves the photo where you put it. The record keeps the test's numbers in `processing.colour` (`mode`; `by`: `rule`, `probe` or `hand`; `off_hue`, `strong`, `tone`, and `probe` for a borderline photo), so another threshold can be applied later without downloading again.

**Existing bases.** Like the layout, the setting is fixed per base when its first scope is made, and a base made before it existed is not split; its photos still get the test, the counts and the record. A conf without the line gets `bw_color` written into `[defaults]`.

### Numbered folders

A category holds its images directly until it would pass `split` images (5,000 by default, so 10,000 files with the captions; Explorer slows down in folders much larger than that). Then its images move into the numbered folder `001`, and the next ones go into `002`, `003`, and so on, each of at most `split` images. The move takes a few seconds and happens during the download, as soon as no image of the category is being written; the store follows each file, and a move cut short by a crash is finished the next time the scope is written to. A category stays chunked when reclassify or regroup later takes images out of it, and the lowest numbered folder with room takes the next image. A category stays one tree: AI Toolkit and the dataset tools read subfolders, so point them at the category folder. `--split` sets another size for a new scope; with `--no-classify` the images and their numbered folders are directly in the scope folder.

**The caption** is one line of facts for the captioner, in the record's language with the English region names added: title, years, regions, address and description, in that order, empty parts left out (`--txt-fields` chooses the parts and their order; `direction` adds the camera direction). Example:

```
Холмские ворота. Брестская крепость. 1900-1930. Беларусь, Брестская область, Брест, Брестская крепость (Belarus, Brest Oblast, Brest, Brest Fortress).
```

PastVu's placeholder titles ("Где это?", alone or in front of a guess) are left out, and so are the parts of a description that held a link, a publication permission, a note on where the map point stands, a pointer to another upload, or a separator line. No URLs and no user names go into the caption. The [taggui_captioning](../taggui_captioning/README.md) prompt reads it as the existing caption; change its sentence that says English words in the caption describe image quality, because here they are region names. `captions <scope>` rewrites every caption of a scope from the records with the current rules and `--txt-fields`, without downloading anything.

**The record** of each photo is in the store: the PastVu record as the API gives it (title, description, address, author, source, years, coordinates, direction, regions, uploader), the download (URL, size, sha256), what was done (band, border cuts, every watermark detection with its confidence, the painted boxes, the crop, how the file was written) and the classification. `export <scope>` writes the records of a scope as JSON Lines (one full record per line) or, with `--format parquet`, as Parquet with the main fields as columns and the full record as a JSON string. `json_files = yes` in the conf also writes each record to `<base>\_records\<number>\<cid>.json`, where the number is the PastVu number divided by 10,000, so no folder holds more than 10,000 files; `--json-sidecar` writes it next to the image instead.

## Autocaption

```
run.bat autocaption belarus_1917-1991
run.bat autocaption D:\pastvu\belarus_1917-1991\5_transport --limit 500
run.bat autocaption belarus_1917-1991 --undo
run.bat run --region belarus --autocaption
```

A vision-language model on your own machine writes each photo's caption from the image and its facts line, and the caption file then holds the model's caption alone: it is built from the facts line, so the date, place and names are in it. The model runs in a separate server with an OpenAI-compatible chat endpoint, such as [KoboldCpp](https://github.com/LostRuins/koboldcpp) with a vision model loaded; `[caption]` in the conf names its address and the two prompt files. The prompts are always files: `tagging\sysprompt.txt`, a short system prompt, and `tagging\main_prompt.txt`, which explains what the facts line is and how a caption is to be written; `{tags}` in it is replaced by the photo's existing caption. The API part is TagGUI's: one request per photo with the image inline (its longest side cut to `image_side`, 1024 px), reasoning switched off, thinking blocks removed from the answer, a retry at half the size when the endpoint rejects a request, and a stop when three photos in a row fail.

`autocaption <target>` captions saved photos, one after another, with the same live display as a download (the photo being captioned, the VRAM in use, a progress bar with the rate and the time left) and report blocks every `--report-every` photos (100): photos captioned, photos per hour, seconds per photo, tokens per second, failures, and the VRAM now and at its peak. The target is a scope's name, its folder, or any folder inside it, such as one category. Several targets and name patterns are captioned one after another (`autocaption "estonia_*"`), and `--limit` counts across them. A photo with a VLM caption is not captioned again, so a stopped run continues where it was when you start it again; `--limit N` takes at most N photos in this run, and `--recaption` captions photos again from their original facts line (with a changed prompt, for example). A photo that fails is tried again on later runs, up to `--retries` (3) times. Ctrl+C finishes the photo being captioned and stops.

**Backups and undo.** Before a caption file is written, its old text goes into the store, and the caption a photo had before its first VLM caption is kept as its original. `autocaption <target> --undo` puts back the caption files of the last autocaption run of the scope, one run at a time; a file you edited since that run keeps your edit. `captions <scope>` leaves photos with a VLM caption alone, and `export` writes the VLM caption next to the facts line.

`run --autocaption` captions each photo as soon as it is saved, from the pixels still in memory, and then the saved photos of the scope that have no VLM caption yet. A run with `--by` keeps one caption run per scope, so `--undo` works per scope. The downloads slow down to the model's pace: a queue of four photos sits between them, so the downloads never run far ahead. If the endpoint stops answering, the run stops; start `autocaption` later for the photos left.

**Speed and memory.** With Gemma 4 26B-A4B (NVFP4) in KoboldCpp 1.122 on an RTX 5090, a caption takes 2.4 to 3.3 s, about 1,300 photos per hour, against 5 to 9 photos per second for a download alone. The model holds about 23 GB of GPU memory; this tool's models add 2.1 GB, and up to 5.9 GB while LaMa paints a large window, so a download with `--autocaption` peaked at 27 GB of the card's 32. The VRAM shown is the whole card's, read through NVML as nvidia-smi reads it: on Windows a program does not see the memory other programs hold, and the driver does not fail when the card is full but moves memory to system RAM, which slows everything down. The status line warns when less than 1 GB is free.

## Reclassify

```
run.bat reclassify belarus_1917-1991 --dry-run
run.bat reclassify belarus_1917-1991
run.bat reclassify belarus_1917-1991 --undo
run.bat reclassify "estonia_*"
run.bat reclassify
```

Sorts the saved photos of a scope again with the samples folder as it is now (or `--samples`) and the priority margin of the conf (or `--priority-margin`), and moves each image and its caption into its new category folder, in place. The file keeps its name, so the caption and the record stay joined. The embeddings are in the store, so the GPU is needed only for images without one, and the probe trains in seconds.

A photo you moved to another category by hand keeps its place: reclassify sees that its file is not where the store says, records where it is now, and leaves it there (`--override-manual` sorts it like the rest). A photo whose file is gone is reported and left alone. `--dry-run` prints the moves per category pair and writes them to `<base>\_reports\`. `--undo` puts the photos of the last reclassify back into the categories they came from.

A saved photo that the quality probe scores at or above `quality_threshold` is deleted: its image, caption, JSON file and export links. The store marks it skipped (`low quality`), so no run downloads it again, and `--undo` cannot bring it back. Before it deletes, reclassify asks `This deletes N low-quality photos ... Continue? [y/N]`, and `--yes` skips the question. `--dry-run` lists these photos in its report. With `quality_threshold = 0` in the conf, reclassify sorts without deleting. A photo that you moved by hand is kept, unless you give `--override-manual`.

Versions before 2026-10-23 put doubtful photos into `_unsure\<category>`. `reclassify --unsure` moves them into `<category>`, their best guess, without the classifier or the GPU: `run.bat reclassify --unsure` for the whole base (it asks first), or name scopes. `--dry-run` and `--undo` work as for a full reclassify.

Several scopes and name patterns are sorted one after another with one load of the classifier. Without a scope, reclassify works on every scope of the base folder; it first asks `This will reclassify N scope folders ... Continue? [y/N]` with the number of saved photos, and `--yes` skips the question. `reclassify --undo` without a scope undoes the last reclassify of every scope that has one, after the same question.

## Extract

```
run.bat extract 050
run.bat extract transport --decade 1970
run.bat extract 100-environment_outdoors --region moscow --color
run.bat extract 050 060 --years 1960 1979 --scope "russia_*"
run.bat extract 050 -o D:\train\transport --link
run.bat extract 140-aerial --target 15000 --min-res 1024
run.bat extract 050 --target 5000 --colour-share 0.4 --decade 1970
```

Copies the photos of a category out of the base folder into a flat folder for training. Only the `bw\` and `color\` level stays; there are no decade, place or numbered folders:

```
<base parent>\050-transport_1970s\
  bw\123456.jpg, 123456.txt, 123456.short
  color\...
  extract.json
```

- **The category** is named by its number (`050` or `50`), its name (`transport`) or its full name (`050-transport`). Folders written differently but meaning the same category (`50_transport` from older samples) are taken together. When a number or name matches two different categories, for example `140-aerial` and `140-monuments`, extract refuses and asks for the full name. Several categories at once each go into their own folder.
- **The photos are found where they are on disk**, in the category folders of every scope, so a photo you moved by hand to another category, or between `bw\` and `color\`, is extracted from where it sits now. Only finished photos are taken, so extract can run beside a download.
- **Filters narrow the selection and never add folders:** `--region` (one or more, matched against the photo's own region chain, so `--region crimea` finds Crimean photos in `russia_1970s`), `--years` (dating overlaps the years), `--decade` (dating starts in the decade), `--scope` (names or patterns), `--bw` or `--color`.
- **`--min-res N`** keeps photos of at least N×N pixels, measured as area as in [k2prep](../k2prep/README.md): `--min-res 1024` is 1,048,576 pixels whatever the shape, so 1536×768 passes and 1200×800 does not.
- **`--target N`** picks N photos for balancing categories against each other. `--colour-share` of them are colour (0.5 by default; 0.4 is 40 % colour), and a side that has too few is filled from the other. Within black-and-white and within colour, each decade gets a share proportional to its photos in the selection, so the decades keep the proportions they have; the scope options (`--decade`, `--years`, `--region`) decide which decades take part. Inside a decade the largest photos come first: the quality filter only keeps or drops a photo and gives no ranking, and size is stable, so a top-up keeps what it picked before. The run prints what it took from each decade, for example `bw 50: 1930s 12/53, 1940s 5/22, ...`.
- **The output folder** is `<base parent>\<category>` with the filters added to the name (`050-transport_1970s`, `050-transport_ukraine_1960-1979`, `140-aerial_min1024_n15000`), or `-o` for one category. `--split N` puts at most N images into each numbered folder under `bw\` and `color\` when a flat folder grows too large.
- **Each image comes with its caption** (`.txt`, the VLM caption where there is one) and its `.short` caption if [recaption_json](../recaption_json/README.md) made one.
- **Running it again tops the folder up:** new photos are copied, files already there are left as they are (captions you edited for training stay), and files from earlier extracts that no longer match (reclassified, moved, dropped) are reported; `--prune` deletes them. Files you put there yourself are never pruned. `extract.json` holds the criteria and the files extract copied.
- **Copy or link.** Images are copied by default. `--link` hard-links them instead, so they take no extra disk space on the same drive; captions are always copied. Deleting a hard-linked file removes that link only: the base keeps its file, and the disk space comes back only when both are gone. Editing is what to watch: the dataset tools of this repository (remove_borders, watermark, jpeg_cleanup) write a new file and rename it over the old one, which breaks the link and leaves the base as it was; an editor that saves into the same file changes both. Every run without `--link` prints a reminder that it exists. `--dry-run` counts and copies nothing.

## Live export: `run --export`

```
run.bat run --region ussr --by country --export
```

Hard-links every saved photo into `pastvu_export` next to the base folder as it is written, one folder per category, then `bw\` and `color\`, then numbered folders of at most `split` images (the conf's split, 5,000 by default):

```
<base parent>\pastvu_export\
  050-transport\bw\001\123456.jpg, 123456.txt
                     002\...
               color\001\...
  100-environment_outdoors\...
```

- **Hard links, image and caption.** The export takes no extra disk space on the drive of the base, and a caption that autocaption writes later lands in both places, as it writes the file in place. Deleting a hard-linked file removes that link only: the base keeps its file, and the disk space comes back only when both are gone. Editing is what to watch: the dataset tools of this repository (remove_borders, watermark, jpeg_cleanup) write a new file and rename it over the old one, which breaks the link and leaves the base as it was; an editor that saves into the same file changes both. On another drive, or a file system without hard links, the files are copied and the run says so.
- **Photos saved before** the export existed are linked when a run with `--export` starts; that pass only creates links, so it is fast.
- **Curating by deleting.** An image you delete from the export is remembered as excluded and never linked again, and its caption link is tidied away. A photo you move to another folder inside the export is recorded where you put it. `--export-reset` forgets the exclusions and links those photos again.
- **The export follows the base.** `reclassify` moves a photo's links to its new category or into the other of `bw\` and `color\` (unless you moved it inside the export yourself), and `drop` deletes the links of the dropped photos. Moves inside the base (regroup, a category passing the split) keep the links valid by themselves, since a link points to the file, not to its folder.
- **Only with the flag.** A run without `--export` on a base that has an export says so and does not link; the next run with `--export` catches up.

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
| `--by RULE` | | split the download into scopes by `decade`, `country`, `region`, `country,decade` or `region,decade` |
| `--regroup` | off | first move the saved photos the run's scopes list into them |
| `--export` | off | hard-link every saved photo into `<base parent>\pastvu_export\<category>\<bw or color>\<NNN>\` as it is written |
| `--export-reset` | off | with `--export`: link again the images you deleted from the export |
| `--split N` | conf (5000) | at most N images per folder, for a new scope; 0: no split |
| `--samples DIR` | conf | the classify samples folder |
| `--no-classify` | off | no categories |
| `--min-confidence X` | 0.7 | below this the record marks the photo as doubtful; it still goes into its best-guess category |
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
| `--skip N` | 0 | start at photo N+1 in PastVu number order; the first N are not downloaded and stay to do |
| `--retries N` | 3 | attempts for a photo that failed |
| `--report-every N` | 500 | a report block every N photos |
| `--quiet`, `--json-progress` | | report blocks only, or one JSON line per photo |
| `--autocaption` | off | caption each saved photo with the VLM of `[caption]`; the downloads slow to its pace |
| `--full-parent` | off | page a large region itself instead of through its children, to take the photos placed at the region itself (slower for the server) |
| `--reindex` | off | page the scope again, to find photos published since |
| `--resume` | off | skip the index (no requests to plan or page it) and download what is listed already |

Before the subcommand: `--conf FILE` another conf, `--base DIR` another base folder, `--max-per-second N` a cap on the requests to PastVu.

## How the index works

PastVu's API has no text search and no topics; it filters by region, years and type. The index pages PastVu's gallery for the scope, 100 photos a page, into the store. A region with more than 50,000 photos in scope is paged through its children in the region tree. With at most 10 children, the photos in none of them are a shard of their own, by the gallery's exclusion filter. With more children (Russia, Moscow, Saint Petersburg) those photos, placed at the region itself and not in any of its parts, are reported and taken only with `--full-parent`, which pages the region itself and costs the server more; for Russia they are about 0.15 % of its photos. A photo without coordinates still names its oblast or city, so its child lists it: a separate pass over the region's photos without coordinates would cost a query over the whole region per page (about 7 s a page for Russia) and find almost nothing new, and the index no longer makes one. The gallery pages by skipping, so every page of a long shard costs the server more than the one before; a shard of more than 5,000 photos that is not finished is therefore paged in year bands of at most 5,000 photos each (a photo dated across a band boundary is listed in both and kept once). A finished index is not paged again unless `--reindex`.

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
| `partition.py` | the rules of `--by`: which scope a photo belongs to |
| `regroup.py` | moving photos between scopes, drop, and undo |
| `colour.py` | the black-and-white or colour test, and the probe for borderline photos |
| `quality.py` | the quality probe: low-quality photos are skipped, or deleted by reclassify |
| `folders.py` | finding and moving files, and the move of a category into numbered folders |
| `runlock.py` | whether the process that holds the lock still runs |
| `layout.py` | scope names, the folder tree, and the numbered-folder allocator |
| `process.py` | header, decode, band, borders, crop composition |
| `encode.py` | the save rules, in memory |
| `classifier.py` | classify's sorter, one image at a time or in batches |
| `reclassify.py` | sorting a scope again, and undo |
| `autocaption.py` | the VLM client, the caption runs, and undo |
| `vram.py` | the GPU memory of the whole card, through NVML |
| `tagging\` | the default prompts of autocaption |
| `export.py` | records as JSON Lines or Parquet |
| `extract.py` | a category copied into a flat folder for training |
| `liveexport.py` | `run --export`: the hard-linked export of every saved photo |
| `sidecars.py` | the caption and the record |
| `progress.py` | the live display and the report blocks |
| `stream.py` | the threads and the run |
| `fetch_turbojpeg.py` | fetches `turbojpeg.dll` into `bin\` |
| `pastvu.conf.example` | the example conf with the aliases |
| `install.bat`, `run.bat`, `requirements.txt` | install into the shared venv, run with it |
