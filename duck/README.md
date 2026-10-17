# duck

Search large image-caption datasets by caption text and metadata, and download the matching images.

duck runs [DuckDB](https://duckdb.org/) over the Parquet files that the dataset authors publish on Hugging Face. One command scans a dataset for captions that contain your terms, applies size, aesthetic, watermark and face filters, and writes a CSV. A second command feeds that CSV to [img2dataset](https://github.com/rom1504/img2dataset), which downloads the images and writes each caption next to its image as a `.txt` file. The result is a folder ready for training a text-to-image model or a LoRA.

No index is built and nothing is uploaded. A query is a full scan of the Parquet files, which takes seconds on the two smaller datasets and a few minutes on COYO-700M.

## Datasets

| name | Hugging Face | rows | on disk | captions |
|---|---|---|---|---|
| `coyo700m` | [kakaobrain/coyo-700m](https://huggingface.co/datasets/kakaobrain/coyo-700m) | 747M | 126 GB | alt text from the web page |
| `coyo-hd-11m` | [CaptionEmporium/coyo-hd-11m-llavanext](https://huggingface.co/datasets/CaptionEmporium/coyo-hd-11m-llavanext) | 11.4M | 15 GB Arrow, 4.5 GB Parquet | LLaVA-NeXT long and short captions, alt text, Open Images and booru tags |
| `pixelprose` | [tomg-group-umd/pixelprose](https://huggingface.co/datasets/tomg-group-umd/pixelprose) | 16.9M | 9.2 GB | dense Gemini captions plus the source caption |

COYO-700M is the raw web crawl: every image size, alt text of every quality, and many thumbnails. The other two are curated subsets with synthetic captions. coyo-hd-11m is COYO filtered to at least 512 px on the short edge and to images with real content, then captioned. PixelProse combines CC12M, CommonPool and RedCaps with long descriptive captions, and its RedCaps split is the densest in high-resolution images.

All three datasets hold URLs, not images. The images live on the original web servers, and a share of them is gone, so every download list loses some rows.

### Getting the datasets

Run duck's `install.bat` first (next section), because the Hugging Face CLI comes with it. Then download each dataset you want into a folder of your choice. The examples use `T:\DATASETS`.

COYO-700M (126 GB):

```bat
..\venv\Scripts\hf download kakaobrain/coyo-700m --repo-type dataset --include "data/*" --local-dir T:\DATASETS\coyo-700m
```

PixelProse (9 GB):

```bat
..\venv\Scripts\hf download tomg-group-umd/pixelprose --repo-type dataset --include "data/*" --local-dir T:\DATASETS\pixelprose
```

coyo-hd-11m-llavanext (15 GB) ships as Arrow shards instead of Parquet, so convert it once after downloading. The converter writes a `parquet` folder next to the `data` folder, 4.5 GB, and verifies the row count against the dataset's own metadata:

```bat
..\venv\Scripts\hf download CaptionEmporium/coyo-hd-11m-llavanext --repo-type dataset --local-dir T:\DATASETS\coyo-hd-11m-llavanext
```

```bat
tools\arrow_to_parquet.bat --src T:\DATASETS\coyo-hd-11m-llavanext\data
```

The Arrow shards can be deleted after the conversion.

## Installing

Requirements: Windows, Python 3.10 or newer on PATH, and git on PATH. git is needed because img2dataset must be installed from GitHub (see [Downloading images](#downloading-images) for why).

```bat
install.bat
```

This installs duck's dependencies into the shared `venv` of the repository, one level up, creating it when it is missing, and copies `duck.conf.example` to `duck.conf`. See the repository README for how the shared environment works.

## Configuring paths

Edit `duck.conf` in the duck folder. It has one line per dataset under `[paths]`, pointing at the folder that holds the Parquet files, and two output folders:

```ini
[paths]
coyo700m = T:\DATASETS\coyo-700m\data
coyo-hd-11m = T:\DATASETS\coyo-hd-11m-llavanext\parquet
pixelprose = T:\DATASETS\pixelprose\data

[output]
results = results
downloads = downloads
```

Relative paths are resolved against the duck folder. A dataset you do not have can stay in the file. `--dir` on the command line overrides the configured folder for one run.

## Searching

```bat
run.bat --list-datasets
```

```bat
run.bat --dataset pixelprose --caption "lighthouse" --watermark-class 1 --min-width 1024 --limit 200
```

```bat
run.bat --dataset pixelprose --help
```

Every dataset takes `--caption` for the search terms and `--exclude` for words that must not appear. The matching rules are the same everywhere:

- Terms are case-insensitive substrings, and every term must match. `cowboy` also finds `cowboys`.
- Inside one term, `|` separates alternatives and `+` stands for a space. `lighthouse|beacon` matches either word, and `stone+tower` matches the phrase `stone tower`.
- Exclusions are whole words, and any one of them drops the row. `--exclude night` removes `a lighthouse at night` but keeps `nightingale`.
- Any option can be given as `*`, which means "not set". This lets a batch file list every option and be edited in place. The `samples` folder has one such file per dataset.

Common options on every dataset:

| option | meaning |
|---|---|
| `--caption TERMS` | terms that must appear in the dataset's main caption column |
| `--exclude WORDS` | words that must not appear in it |
| `--min-width N`, `--min-height N` | minimum image size in pixels (the real image, not a resized copy) |
| `--min-aesthetic X` | minimum aesthetic score, 0 to 10 |
| `--caption-col COL` | which column becomes the `.txt` caption on download |
| `--limit N` | stop after N rows, `*` for all |
| `--start N` | skip the first N matches, for paging through a large result |
| `--count` | only count the matches, write nothing |
| `--quick-test` | scan the first shard only, to try a query cheaply |
| `--out FILE` | write to this file instead of the default name |

Dataset-specific options, shown in full by `--help`:

- **coyo700m**: `--faces` (exact count or range such as `1-2`), `--max-watermark`, `--max-nsfw`, `--min-clip` (how well the alt text fits the image, useful because alt text is noisy).
- **coyo-hd-11m**: search any of the five text columns directly with `--caption_llava`, `--caption_llava_short`, `--caption_original`, `--tags_open_images` and `--tags_booru`, each with a matching `--exclude-` form; `--faces`, `--max-watermark`, `--max-nsfw`. Tags are searched as substrings too, so use full tag names such as `baseball_cap`.
- **pixelprose**: `--split` (`all`, `cc12m`, `commonpool`, `redcaps`), `--watermark-class` (0 has a watermark, 1 is clean, 2 has text but no watermark, ranges such as `1-2` allowed), `--max-watermark-score`, `--original_caption` and `--exclude-original_caption`, and `--strip-preamble` (on by default, removes the "This image displays:" boilerplate that starts most PixelProse captions).

Sizing a query before running it:

```bat
run.bat --dataset coyo700m --caption "lighthouse" --min-width 1024 --min-height 1024 --count
```

Paging: the rows are always returned in the same order, so `--start 1000 --limit 1000` continues exactly where `--limit 1000` stopped.

Quoting from PowerShell: PowerShell removes the quotes around an argument that has no spaces before cmd sees it, and cmd then treats `|` as a pipe. Write such a term as `'"lighthouse|beacon"'` in PowerShell, or run the command from cmd or a batch file.

## What comes out

Each search writes two files under the results folder, in a sub-folder per dataset:

```
results\pixelprose\lighthouse-0.csv
results\pixelprose\lighthouse-0.json
```

The name is built from the search terms and the `--start` value. An existing file is never overwritten; a second run gets `-2`, `-3` and so on. The `.json` records the dataset, the folder scanned, every option that was set, and the number of rows, so a CSV can always be traced back to the query that produced it.

The CSV starts with `url` and `text`, the two columns img2dataset reads. `text` is the chosen caption column with line breaks folded to spaces. The remaining columns ride along for inspection and filtering in a spreadsheet:

| dataset | columns after `url,text` |
|---|---|
| `coyo700m` | width, height, num_faces, clip_similarity_vitb32, nsfw_score_opennsfw2, watermark_score, aesthetic_score_laion_v2, id |
| `coyo-hd-11m` | caption_llava, caption_llava_short, caption_original, tags_open_images, tags_booru, num_faces, width, height, aesthetic_score_laion_v2, watermark_score, nsfw_score_opennsfw2, key |
| `pixelprose` | vlm_caption, original_caption, watermark_class_id, watermark_class_score, aesthetic_score, width, height, uid |

## Downloading images

```bat
download.bat results\pixelprose\lighthouse-0.csv
```

Images go to the downloads folder from `duck.conf`, in a sub-folder named after the CSV, in img2dataset's `files` layout: shards of up to 10,000 images, each image with a `.txt` holding its caption and a `.json` holding the metadata and any download error.

```
downloads\lighthouse-0\00000\000000000.jpg
downloads\lighthouse-0\00000\000000000.txt
downloads\lighthouse-0\00000\000000000.json
downloads\lighthouse-0\00000.parquet
downloads\lighthouse-0\00000_stats.json
```

Running the same command again resumes: finished shards are skipped. Options, all shown by `download.bat --help`:

- `--encode original` (default, also `0`, `off`, `no`, `none` or `skip`) stores the downloaded bytes without decoding them and names each file by its real content: `.jpg`, `.png`, `.gif`, `.webp` or `.bmp`. `--encode jpg` keeps JPEG sources exactly as downloaded and converts anything else to JPEG at quality 95. `--encode png` decodes every image and writes PNG, which is lossless from the decoded pixels but makes a JPEG about ten times larger.
- `--resize N` shrinks images whose long edge exceeds N pixels. `--min-size N` drops images smaller than N pixels on either edge. Both need the image decoded, so with the default encoding they switch to `jpg`.
- `--threads` (64), `--processes` (1), `--timeout` (10 s) and `--retries` (0) control the download itself.
- `--format webdataset` or `--format parquet` writes tar shards or Parquet instead of loose files.

Why img2dataset comes from GitHub: the PyPI release and the GitHub main branch both call themselves 1.47.0, but only the GitHub build passes an explicit TLS context to urllib. Without it, Python's default TLS handshake is rejected by some CDNs, most visibly WordPress Photon (`i0.wp.com`, very common in COYO-700M), which answers with a bogus `404 Not Found`. Reinstalling img2dataset from PyPI brings those failures back. `requirements.txt` has the details.

## Adding a dataset

A dataset is one module in `duck_datasets` that defines a `SPEC`, plus one line in `duck_datasets/__init__.py` to register it. The spec lists the searchable text columns, the numeric and range filters with the column each maps to, the columns to copy into the CSV, the order-by key for stable paging, optional splits, and optional dataset-only arguments and caption transforms. `pixelprose.py` shows all of these. The engine in `duck_engine.py` builds the command line and the SQL from the spec, so a new dataset needs no code beyond its description.

## License

MIT, with the rest of the repository. The datasets have their own licenses: COYO-700M and PixelProse are CC BY 4.0, coyo-hd-11m-llavanext is CC BY-SA 4.0. The images belong to their respective owners.
