# JPEG cleanup

Finds the images of a dataset that were saved with heavy JPEG compression, restores them with [FBCNN](https://github.com/jiaxi-jiang/FBCNN), and replaces only those where the restoration removes enough of the damage. Originals and captions go to `_backup`; `--review` draws them next to the restored files, and `--undo` puts them back.

## Why

Images collected from the web have often been saved as JPEG several times: by the camera, an editor, a website, a messenger, every repost. Each save at a lower quality adds 8 x 8 blocks, ringing around edges and colour blotches, and a model trained on such images learns to draw them.

The JPEG header records only the last save. A video frame compressed hard and then saved again at quality 85 says 85. FBCNN predicts the quality factor (QF) from the pixels, so it reads that frame as 56 to 66, and a PNG made from a JPEG as about the quality of that JPEG.

Size counts as much as quality. Training scales images to about 1024 x 1024: a 3 MP image at QF 75 loses most of its artifacts in that downscale, a 0.3 MP image at QF 75 keeps all of them. Images above 2048 x 2048 are skipped for that reason.

A restoration can also do harm. FBCNN smooths film grain, print dots and fine texture together with the artifacts, and on a scan of an old print the grain is part of the picture. Every re-save costs disk space too: a restored JPEG at quality 97 takes about 2.6 times the space of its QF 60 to 70 original. So I wanted a tool that changes as few files as possible: only images with real damage, and only when the restoration removes enough of it.

## What we found

The tool was developed on a dataset of 10,000 photos, mostly scans of old prints re-saved several times, and on sample folders sorted by eye.

- On a JPEG saved once, FBCNN's QF matches the header quality within 1 to 3 points. It earns its cost on re-saved images.
- QF alone did not match the eye. Two images of the same source at QF 72 and about the same size were judged one good, one poor; large images at QF 75 to 80 looked fine.
- Measuring QF after a downscale fails: any resampling breaks the 8 x 8 grid, and FBCNN reads the image as clean (a 6% downscale moved one image from QF 71 to 83). The tool measures the stored pixels at full size.
- By eye, QF 85 and above looked fine, 79 and below needed fixing, and 80 to 84 (55% of the dataset) was mixed.
- The mean pixel change of a restoration measures grain. The images it changed most were the grainiest, and its correlation with the drop in blockiness was -0.48. It cannot tell a useful restoration from a harmful one.
- The gray FBCNN model smoothed the grain of black-and-white photos away and hardly reacted to the strength setting. The colour model, given the gray image as three equal channels, kept the grain and removed the blocks as well.
- Dithered and screened prints read QF 40 with no JPEG damage at all. FBCNN cannot help them.
- Of the 3,258 images under QF 80, 1,774 restorations removed enough to be worth saving (section [Enough benefit](#enough-benefit)).

## What it does differently

A typical artifact-removal script runs the model over a folder and overwrites every file. This tool:

- **Measures first.** `extract.bat` copies the poor images into one folder per quality band, outside the dataset, so the threshold is set by eye on the dataset itself. The dataset stays untouched.
- **Leaves good images alone.** An image at or above the threshold (QF 80 by default) is never restored, and one above 2048 x 2048 is never touched.
- **Saves a restoration only for a real benefit.** Every restoration is judged in memory with two measures that film grain cannot fake. Without enough benefit the image stays as it is, and nothing goes to `_backup`.
- **Keeps the grain.** FBCNN is told a quality 10 points above the image's own, so it removes a little less, and black-and-white images go through the colour model.
- **Skips what it cannot fix.** Formats with their own kind of compression (lossy WebP, AVIF, HEIC, GIF), transparent, animated, CMYK and 16-bit images are listed and left alone.
- **Shows the result first.** `--dry-run` writes a report and contact sheets with every image before and after, at 100% and magnified, split into the restorations it would save and the ones it would leave. A real run saves exactly what its dry run listed.
- **Backs up before it writes.** The original and its captions go to `_backup` under the same relative path. A backup is never overwritten, and an earlier original from another tool keeps its place.
- **Writes safely.** Each image is logged before it is touched. The restoration goes to a `.part` file and replaces the image in one step, with the original's name, times, read-only flag, EXIF block and ICC profile.
- **Shows what it wrote.** `--review` draws the contact sheets of the last run from the originals in `_backup` and the files now in place.
- **Undoes a run.** `--undo` puts back the last run, also one that was interrupted.
- **Runs offline.** The models are downloaded once and checked against their SHA-256 sums.

## How to use it

1. Run `install.bat` once.
2. Run `extract.bat <dataset>` and look through the band folders in `<dataset>_jpeg_extract`, from the lowest up, to find the QF where the images stop looking compressed. New `--bands` re-sort the copies in seconds.
3. Run `run.bat <dataset> --dry-run --threshold <QF>` and look through the contact sheets in `<dataset>\_backup\_jpeg_cleanup\sheets`. The `fix_*` sheets show what would be saved, the `nofix_*` sheets what would be left.
4. Run `run.bat <dataset> --threshold <QF>`, then `run.bat <dataset> --review` to look at what was written.
5. If you do not like the result, run `run.bat <dataset> --undo`.

## Install

On Windows, run `install.bat`. It needs Python 3.10 or newer and a CUDA GPU; without a GPU the tool runs on the CPU, much more slowly.

It creates the shared `..\venv` folder when it is missing, installs torch from the PyTorch CUDA index and then `requirements.txt` (Pillow, numpy), and downloads the two FBCNN models (288 MB each) into the `models` folder next to the script.

| file | used for |
|---|---|
| `fbcnn_color.pth` | the QF of colour images, and the restoration of every image |
| `fbcnn_gray_double.pth` | the QF of black-and-white images |

`network_fbcnn.py` is the FBCNN network from the FBCNN repository, under the Apache 2.0 licence in `LICENSE-FBCNN`.

## Extract

```
extract.bat <folder> [<folder> ...] [options]
```

Examples:

```
extract.bat D:\photos
extract.bat D:\photos --bands 50,60,70,75,80,85,90
extract.bat D:\photos --out E:\check\photos
extract.bat D:\photos_jpeg_extract --bands 60,70,75,80,85
```

The first command measures every image of `D:\photos` and its subfolders and copies the images with a QF under 85 into `D:\photos_jpeg_extract\60`, `70`, `80` and `85`. The second splits the same images into other bands, with the measurements from the cache. The third writes the copies to another folder. The fourth re-sorts an existing extract folder in place, without the dataset: band `80` splits into `75` and `80`.

| option | what it does |
|---|---|
| `--bands LIST` | the band limits, comma-separated (default `60,70,80,85`; for an extract folder, the limits of its last run) |
| `--out DIR` | the output folder (default `<folder>_jpeg_extract` next to the dataset folder); with several folders, one subfolder per folder name |
| `--max-pixels N` | skip images larger than N pixels (default 4194304, which is 2048 x 2048) |
| `--exclude NAME` | another folder name to skip, at any depth; may repeat |
| `--sidecars LIST` | the extensions of the files that travel with an image, comma-separated (default `.txt`) |
| `--reanalyse` | ignore the cache and measure every image again |
| `--threads N` | threads that decode the images (default 8) |

What a run does:

1. Scans the folder at any depth. Images are `.jpg`, `.jpeg`, `.jpe`, `.jfif`, `.png`, `.webp`, `.bmp`, `.tif`, `.tiff`, `.gif`, `.avif`, `.heic` and `.heif`. A `.txt` file with the same name as an image is its caption. Folders named `_backup`, `_duplicates`, `_prep`, `_classify`, `_embeddings`, `masks` and `faces` are skipped, at any depth.
2. Measures the QF of every image on its stored pixels, before EXIF rotation, so the 8 x 8 grid stays where the model expects it. A black-and-white image (one channel, or three channels within 2 levels of each other) is measured with both models, and the gray model decides. An unchanged image is taken from the cache.
3. Puts each image in the band of the first limit above its QF. With the default limits, band `60` holds QF under 60, `70` holds 60 to 69.9, `80` holds 70 to 79.9 and `85` holds 80 to 84.9. Each image is in one band, so neighbouring bands can be compared side by side; a threshold of 80 takes the bands `60`, `70` and `80`.
4. Copies each image of a band, with its captions, as `q<QF>__<name>` (for example `q057__photo.jpg` and `q057__photo.txt`), so a folder sorted by name is sorted by QF. When two images of different folders have the same name and QF, the second gets `~2`.
5. Writes `extract.csv`, `summary.txt` and `manifest.json` and prints the summary.

A new run into the same output folder first deletes the copies of the last run, which `manifest.json` lists; other files in the band folders stay.

Skipped, and counted in the summary:

| skip | why |
|---|---|
| `large` | more than `--max-pixels`: the downscale to the training resolution hides the artifacts |
| `format GIF`, `format AVIF`, `format lossy WEBP` and other formats | their own compression artifacts, which FBCNN does not know |
| `mode CMYK`, `mode I;16` and other modes | CMYK, 16-bit and 1-bit images |
| `transparent` | an alpha channel that is not fully opaque |
| `animated` | more than one frame |
| `unreadable` | the file cannot be decoded |

The run prints a progress line every 500 images with the time left. Ctrl+C stops it; the measurements made so far stay in the cache, and the next run continues from there.

### Re-sorting an extract folder

When the folder given is an extract folder, it is re-sorted in place from its `extract.csv` and `manifest.json`, without the dataset or the model. New `--bands` move copies with their captions between the band folders, and copies above a lower top limit are deleted with their captions. Images above the old top limit were never copied; the run says how many there are, and a run on the dataset folder adds them.

`extract.csv` and `manifest.json` are written first, then the moves are made. A stopped re-sort is finished by the next run, from `moves.json`.

## Fix

```
run.bat <folder> [<folder> ...] [--dry-run] [options]
run.bat <folder> --review
run.bat <folder> --undo
```

Examples:

```
run.bat D:\photos --dry-run
run.bat D:\photos --dry-run --sheet-offsets 0,20
run.bat D:\photos
run.bat D:\photos --review
run.bat D:\photos --undo
```

The first command restores every image of `D:\photos` with a QF under 80 in memory, judges each restoration, and writes the report and the contact sheets into `D:\photos\_backup\_jpeg_cleanup\`. The second also shows the restorations at offsets 0 and 20 on the sheets, to choose `--qf-offset` by eye. The third makes the fix. The fourth draws the contact sheets of that run from the backups. The fifth puts back what the last run changed.

| option | what it does |
|---|---|
| `--dry-run` | restore and judge in memory, write the report and the sheets; change no image |
| `--threshold QF` | restore the images with a QF under this (default 80) |
| `--qf-offset N` | added to the QF that FBCNN is told (default 10) |
| `--min-block-drop X` | save a restoration whose blockiness drops by at least X (default 0.10) |
| `--min-qf-gain N` | or whose QF rises by at least N on an image with a visible JPEG grid (default 25) |
| `--sheet-offsets LIST` | more offsets shown side by side on the sheets, comma-separated |
| `--no-sheets` | with `--dry-run`: the report only |
| `--sheets` | with a real run: the sheets too |
| `--quality Q` | the JPEG quality of a restored JPEG (default 97) |
| `--review` | draw the contact sheets of the last run: each original in `_backup` next to the file now in place |
| `--undo` | put back what the last run of each folder changed |

`--max-pixels`, `--exclude`, `--sidecars`, `--reanalyse` and `--threads` work as for `extract.bat`, and the QF measurements come from the same cache.

What the fix does with each image under the threshold:

1. Restores the stored pixels with the colour model of FBCNN. A black-and-white image goes in as three equal channels, and only the luma of the result is kept, so no colour can appear. FBCNN is told the QF the colour model predicts, plus `--qf-offset`; told a higher quality, it removes less and keeps more grain. The effect is mild: from +0 to +20 the fine detail kept rises from about 78% to 87% of the original's.
2. Encodes the result as it would be written: a JPEG source as JPEG at `--quality` with full-resolution colour (4:4:4, so the re-save adds no colour bleeding), a PNG, BMP, TIFF or lossless WebP source in its own format. The re-save at 97 differs from the restoration by 46 to 51 dB PSNR, far below the artifacts removed.
3. Measures the QF of the result, the blockiness before and after, and the mean change from the original.
4. Judges whether the restoration is worth saving. If it is not, the image and its captions stay untouched.
5. In a real run, if it is: logs the image, copies the original and its captions to `_backup`, writes the restoration to a `.part` file and puts it in place of the image in one step. The image keeps its name, modification time, read-only flag, EXIF block (with the orientation) and ICC profile; a gray result leaves out an RGB profile, which does not fit it.

The restorations are computed deterministically, so a real run saves exactly what its dry run listed as `fix`. A second run reads the restored images at QF 90 or so and leaves them alone.

A dry run also takes an extract folder: its copies hold the bytes of their sources, so the report and the sheets show what a run on the dataset would do. A real run refuses an extract folder.

Restored JPEGs are larger than their originals: at quality 97 about 2.6 times the size of a QF 60 to 70 original, at 95 about 2 times, at 92 about 1.6 times.

### Enough benefit

Two measures decide whether a restoration is saved, and film grain fakes neither:

- **Blockiness**: the mean luma step across the edges of the 8 x 8 JPEG blocks, divided by the mean step between other neighbouring pixels. About 1.0 when no grid shows, 1.1 to 1.5 on blocky images. Grain and texture raise both steps alike and leave it near 1.0. A drop of at least `--min-block-drop` (0.10) means the grid visibly fades.
- **QF gain**: how much FBCNN's own QF rises from the original to the result. It also sees ringing and mosquito noise, which blockiness misses: an image at QF 45 whose ringing goes reads 90 afterwards. A dithered or screened print fools it, so a gain of at least `--min-qf-gain` (25) counts only on an image whose blockiness is at least 1.05.

On the 3,258 images under QF 80 of the dataset above, the rule saved 68 of 75 under QF 60, 349 of 424 at 60 to 69, 975 of 1,750 at 70 to 74 and 382 of 1,009 at 75 to 79. By eye, restorations with a blockiness drop under 0.03 were indistinguishable from their originals, those at 0.06 to 0.10 showed small gains at 3x, and those from 0.15 up removed visible blocks.

### Backup and undo

The original of a restored image goes to `_backup\<relative path>`, with copies of its captions. When `_backup` already holds an original of that path (from `remove_borders`, the pipeline, or an earlier run), that one stays: a backup is the one untouched original of an image, shared by every tool, and it is never overwritten.

`--undo` takes the last run that is not undone yet and returns every image it restored to its original in `_backup`, newest first; an earlier run's change to the same image goes with it, since the only backup is the original. A backup the run made is removed; one that was there before stays. Caption copies the run made are removed while the caption in place is unchanged, and a missing caption comes back. Each `--undo` goes one run further back. A run that wrote nothing logs no run.

### Contact sheets

`fix_NNNN_qLO-HI.jpg` for the restorations worth saving and `nofix_NNNN_qLO-HI.jpg` for the ones left alone, 8 images per sheet in QF order, in `_backup\_jpeg_cleanup\sheets`. The label of each row says SAVE with the reason, or LEAVE, with the QF, blockiness and change. Each row has a thumbnail with the crop marked in red, a 256 x 256 crop at 100% (the original, then each restoration), and the most changed 96 x 96 part of that crop, magnified 3 times. The crop sits where the restoration changed the image most: on a blocky image where the blocks were, on a grainy one where the most grain went. Judge both: the artifacts should be gone, and the grain of a film photo or the dots of a print should stay. The sheets are JPEG at quality 95 with 4:4:4 colour. A run deletes the sheets of the last one. A real run draws sheets only with `--sheets`; `--review` draws them afterwards, from the log of the last run that is not undone: every image it wrote, the original from where the run kept it next to the file in place, with columns `original` and `written`. It reads files only, so it needs no GPU (417 images take about 10 seconds).

## Output

```
<folder>_jpeg_extract\
  60\q050__photo.jpg         the images with QF under 60, and their captions
  70\ 80\ 85\                the other bands
  extract.csv                one row per image of the dataset
  summary.txt                counts per band, a QF histogram, the skipped images
  manifest.json              the files the last extract copied
<folder>\_backup\
  <relative path>\<image>    the original of a restored image
  <relative path>\<name>.txt its captions
  _jpeg_cleanup\
    cache.json               the QF measurements, reused while a file is unchanged
    report.csv               the last fix, dry or real, one row per image
    sheets\                  its contact sheets
    log.jsonl                every real run and undo, step by step
```

`extract.csv` columns: `path` (relative to the dataset folder), `format`, `mode`, `width`, `height`, `megapixels`, `gray` (1 for a black-and-white image), `header_q` (the quality of the last save, estimated from the JPEG luminance table), `qf_color` and `qf_gray` (the QF of each model), `qf` (the one that decides), `band`, `copy` (relative to the output folder), `status` (copied, above, or the skip reason). A `qf` well under `header_q` means the image was compressed harder before its last save.

`report.csv` columns: `path`, `format`, `mode`, `width`, `height`, `gray`, `header_q`, `qf`, `action` (fix, little benefit, keep, or the skip reason), `benefit` (why a fix is worth saving), `qf_used` (the QF FBCNN was told), `write` (jpeg, or the lossless format), `qf_after`, `change` (the mean difference from the original, in 8-bit levels), `block_before`, `block_after`, `result` (in a real run: written, or the error).

Both files are UTF-8 with a byte order mark, so Excel shows Cyrillic names correctly.

`extract.bat` and `--dry-run` write only into `_backup\_jpeg_cleanup`. The other dataset tools skip `_backup`. When you are sure of the fix, delete the backup files to free the space, and keep `_jpeg_cleanup` while you may want `--undo`.

## Speed

On an RTX 5090: about 35 images per second to measure a mix of 0.2 to 4 MP images, seconds to re-band from the cache, and about 8 images per second to restore with sheets, 10 without. Each extra sheet offset costs one more restoration per image.
