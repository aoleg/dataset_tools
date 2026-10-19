# PastVu

Downloads historic photographs from [PastVu](https://pastvu.com) by region and years, and prepares each one for a training dataset while it downloads: it cuts the grey PastVu band and any scanned border, paints out watermarks and printed captions, sorts the image into the categories of a [classify](../classify/README.md) samples folder, and writes it with a one-line caption for the captioner and its full PastVu record. One image after another, with every model kept on the GPU, and with a store that lets a stopped run continue where it was.

The tool uses PastVu's public API (no account, no key). Read [The photographs are not PastVu's](#the-photographs-are-not-pastvus) before you publish images you downloaded with it.

## Install

Run `install.bat` in the repository root first: it installs remove_borders, watermark and classify, whose code and models this tool uses. Then run `install.bat` in this folder. It installs `httpx`, `rich`, PyTurboJPEG and `zstandard` into the shared `..\venv`, and puts `turbojpeg.dll` into `bin\`: PyTurboJPEG does not include the library, and libjpeg-turbo publishes its Windows builds only as installers, so the installer takes the DLL out of conda-forge's `libjpeg-turbo` package (pinned by version and sha256).

## Usage

```
run.bat regions [--find TEXT] [--children CID]
run.bat count --region <cid or name> [--region ...] [--years FROM TO]
run.bat run --out <folder> --region <cid or name> [--years FROM TO] --samples <folder> [options]
run.bat run --out <folder>
run.bat captions --out <folder> [--txt-fields LIST] [--dry-run]
```

Examples:

```
run.bat regions --find Georgia
run.bat count --region Belarus --region 3
run.bat run --out D:\pastvu\belarus --region Belarus --samples D:\samples
run.bat run --out D:\pastvu\belarus
```

`captions` rewrites the caption of every saved photo from its record, with the current caption rules and `--txt-fields`, and downloads nothing; `--dry-run` shows the first changes and counts them.

The first example lists the regions whose name contains the text, with their cids and photo counts (Georgia the country is 92, the US state is 726). The second prints how many photos PastVu lists for Belarus and for Moscow (cid 3) in the default years, 1917 to 1991. The third indexes Belarus and starts the stream into `D:\pastvu\belarus`. The fourth continues that run: the scope is kept in the store.

`--years` selects the photos whose dating overlaps the years, as PastVu's own filter does: a photo dated 1910-1925 is in 1917-1991. Only photographs are taken, not paintings.

## What happens to one photo

1. **Fetch** (download threads): the PastVu record; the photo is skipped when it is smaller than `--min-side` (512) on either side, a painting, or not public. Then the original file, which is the picture with PastVu's grey band appended at the bottom (`pastvu.com/<cid> uploaded by ...`). The file must have the size the record gives, picture plus band, or the photo counts as failed.
2. **Band and borders** (download threads): the band rows are known from the record and come off exactly. [remove_borders](../remove_borders/README.md) analyses the picture without the band for scanned frames, mounts, thin lines and text banners.
3. **Watermarks** (GPU): the [watermark](../watermark/README.md) detector runs on the picture without the band, so the band's own text is never taken for a watermark. Detections from `--wm-conf` (0.1) go into the record; only those from `--wm-paint-conf` (0.4) are painted out with LaMa.
4. **Crop**: the band, the border cuts and a watermark trim (`--wm-mode trim`) make one crop, put on the JPEG block grid: the top and left edges move inward to the next block where a cut set them, so the cut part goes completely, and the far edges are exact.
5. **Classify** (GPU): the final pixels are embedded with SigLIP 2 and sorted by classify's probe, trained once at start on the samples folder, with classify's rules: the confidence gate (`--min-confidence`, 0.7), the share of the 10 nearest examples, and a byte-identical copy of a sample taking the sample's category.
6. **Write** (writer threads): the image, its caption and its record, then its status in the store.

Pixels stay in memory from the download to the write; no temporary file is made. How an image is written:

| what was done | written as |
|---|---|
| nothing | the downloaded bytes, unchanged |
| only cuts (band, borders, trim) | JPEG: cropped in the DCT domain with libjpeg-turbo, every coefficient kept, EXIF kept with the new size. PNG: the exact pixels in the same format and mode. |
| a watermark painted | JPEG at quality 97 without chroma subsampling, or PNG with `--png` |

A JPEG crop keeps every coefficient of the kept blocks; the decoded pixels can still differ in the last row or column at a cut, where the decoder's chroma upsampling no longer sees the cut-off neighbours.

An image that is smaller than `--min-side` after its crop is skipped.

## Output

```
<out>\
  10_environment_outdoors\163150.jpg      the image, named by its PastVu number
  10_environment_outdoors\163150.txt      its caption
  _unsure\5_transport\165001.jpg          below the confidence gate, under its best guess
  _unsure\5_transport\165001.txt
  _classify\plan.csv                      classify's plan layout, one row per image
  _embeddings\                            classify's embedding cache of the output
<out>_meta\
  163150.json                             the full record and what was done
  pastvu.db                               the index, the status of every photo, the counters
  _dataset_tools.json                     marks the folder, so the other tools skip it
```

The output is a classify output: `classify --extract` works on it, and the embeddings are already cached. With `--no-classify` every image goes into `<out>` itself.

**The caption** is one line of facts for the captioner, in the record's language with the English region names added: title, years, regions, address and description, in that order, empty parts left out (`--txt-fields` chooses the parts and their order; `direction` adds the camera direction). Example:

```
Холмские ворота. Брестская крепость. 1900-1930. Беларусь, Брестская область, Брест, Брестская крепость (Belarus, Brest Oblast, Brest, Brest Fortress).
```

PastVu's placeholder titles ("Где это?") are left out, and so are the parts of a description that held a link, a publication permission, a note on where the map point stands, or a pointer to another upload. No URLs and no user names go into the caption. The [taggui_captioning](../taggui_captioning/README.md) prompt reads it as the existing caption; change its sentence that says English words in the caption describe image quality, because here they are region names.

**The record** `<cid>.json` is in `<out>_meta`, outside the dataset, because deduplicate, k2prep and the pipeline move only `.txt` files with an image; the PastVu number in the file name joins it to the image in any later folder. It holds the PastVu record as the API gives it (title, description, address, author, source, years, coordinates, direction, regions, uploader), the download (URL, size, sha256), what was done (band, border cuts, every watermark detection with its confidence, the painted boxes, the crop, how the file was written) and the classification. `--json-sidecar` also writes it next to the image; `--comments` adds the PastVu comment thread.

## Progress and resume

The live display shows the current activity on a status line ("163150: detecting watermarks", "waiting for downloads (6 in flight)") above a progress bar with the rate and the time left. Every `--report-every` photos (500) a report block is printed above them and stays: photos saved and skipped by reason, band and border cuts, watermarks painted and detected below the threshold, how the files were written, the categories, and the time per stage. `--quiet` prints the report blocks only; `--json-progress` prints one JSON line per photo.

Ctrl+C finishes the photos already downloaded and stops; a second Ctrl+C stops at once. Run the same command again, or `run.bat run --out <folder>`, and it continues with every photo that is not done. A photo's status becomes done only after its image, caption and record are written, so a photo that was being written when the process died is done again. A photo that failed is tried again on the next run, up to `--retries` (3) times.

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
| `--out DIR` | required | the dataset folder; the store goes to `DIR_meta` next to it |
| `--region CID\|NAME` | | a PastVu region; repeat for several; needed on the first run only |
| `--years FROM TO` | 1917 1991 | photos whose dating overlaps these years |
| `--samples DIR` | | classify samples folder, one subfolder of examples per category |
| `--no-classify` | off | no categories |
| `--min-confidence X` | 0.7 | below this an image goes to `_unsure` |
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
| `--fetchers N` | 4 | download threads; 8 is about 30 to 70 % faster and doubles the load on PastVu |
| `--depth N` | 16 | photos downloaded ahead of the GPU |
| `--writers N` | 2 | encoder and writer threads |
| `--limit N` | | do at most N photos in this run |
| `--retries N` | 3 | attempts for a photo that failed |
| `--report-every N` | 500 | a report block every N photos |
| `--quiet`, `--json-progress` | | report blocks only, or one JSON line per photo |
| `--full-parent` | off | page a large region itself instead of through its children |
| `--reindex` | off | page the scope again, to find photos published since |
| `--max-per-second N` | | before the subcommand: cap the requests to PastVu |

## How the index works

PastVu's API has no text search and no topics; it filters by region, years and type. The index pages PastVu's gallery for the scope, 100 photos a page, into the store. A region with more than 50,000 photos in scope is paged through its children in the region tree, plus the photos of the region that have no coordinates; geotagged photos that lie in the region but in none of its children are reported and taken only with `--full-parent`, which pages the region itself and costs the server more. A finished index is not paged again unless `--reindex`.

## The photographs are not PastVu's

PastVu's [rules](https://docs.pastvu.com/en/rules) let site users reproduce the published images free of charge for personal, non-commercial purposes (rule 1.15) and let anyone save them (rule 1.20). Commercial use needs the permission of the rights holder of each image (rule 1.16), and PastVu says it is not that rights holder (rule 1.19): the photographs belong to their photographers, archives, museums and publishers, which the record's `author` and `source` fields name. Nothing in the rules forbids bulk downloads, and nothing grants the right to publish the images again.

Most photographs of the Soviet era are still protected somewhere. In Russia a photograph is protected for the author's life plus 70 years, plus 4 years for those who served in the Second World War, and the 2008 Civil Code restored protection to works the short Soviet terms had released; an anonymous work is protected for 70 years from publication. Other countries of the former Soviet Union have their own terms. In the United States, foreign works published from 1931 to 1977 that were still protected at home on 1 January 1996 are protected for 95 years from publication. Text and data mining exceptions cover the analysis and a local copy, not the publication of the files. This is not legal advice.

Each record carries what attribution and a takedown need: the PastVu page, the uploader, the author, the source and the years.

## Project files

| file | purpose |
|---|---|
| `pastvu.py` | the command line |
| `client.py` | the API client: retries, back-off, optional cap |
| `store.py` | the SQLite store |
| `index.py` | the index of a scope |
| `process.py` | header, decode, band, borders, crop composition |
| `encode.py` | the save rules, in memory |
| `classifier.py` | classify's sorter, one image at a time |
| `sidecars.py` | the caption and the record |
| `progress.py` | the live display and the report blocks |
| `stream.py` | the threads and the run |
| `fetch_turbojpeg.py` | fetches `turbojpeg.dll` into `bin\` |
| `install.bat`, `run.bat`, `requirements.txt` | install into the shared venv, run with it |
