# JPEG cleanup

`jpeg_cleanup.py` finds heavily compressed images in a large dataset with [FBCNN](https://github.com/jiaxi-jiang/FBCNN), a network that predicts the JPEG quality factor (QF) of an image from its pixels. It sees through re-saves: video frames whose JPEG header says quality 85 read 56 to 66 because they were compressed harder before, and a PNG made from a JPEG reads about the quality of that JPEG.

`extract.bat` measures every image and copies the poor ones into one folder per quality band, outside the dataset, so the threshold for the cleanup can be chosen by eye. `run.bat` restores the images under the threshold with FBCNN, decides for each whether the restoration removes enough to be worth saving, and only then puts the original and its captions into a `_backup` folder and the restoration in its place, under the same name. `run.bat --dry-run` does all of that but the writing and draws contact sheets with each image before and after; `run.bat --undo` puts back what the last run changed.

## Install

On Windows, run `install.bat`. It needs Python 3.10 or newer and a CUDA GPU; without one the tool runs on the CPU, much more slowly.

It creates the shared `..\venv` folder when it is missing, installs torch from the PyTorch CUDA index and then `requirements.txt` (Pillow, numpy), and downloads the two FBCNN models (288 MB each) into the `models` folder next to the script. Both files are checked against their SHA-256 checksums. After that the tool runs offline: it loads the models from `models` only.

| file | used for |
|---|---|
| `fbcnn_color.pth` | the QF of colour images, and the fix of every image |
| `fbcnn_gray_double.pth` | the QF of black-and-white images, also those saved as RGB; the fix uses the colour model for them too |

`network_fbcnn.py` is the FBCNN network from the FBCNN repository, under the Apache 2.0 licence in `LICENSE-FBCNN`.

## Usage

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

The first command measures every image of `D:\photos` and its subfolders and copies the images with a QF under 85 into `D:\photos_jpeg_extract\60`, `70`, `80` and `85`. The second splits the same images into other bands; the measurements are taken from the cache, so it takes seconds. The third writes the copies to another folder. The fourth sorts an existing extract folder again, in place and without the dataset: band `80` is split into `75` and `80` by moving its copies (section [Sorting an extract folder again](#sorting-an-extract-folder-again)).

| option | what it does |
|---|---|
| `--bands LIST` | the band limits, comma-separated (default `60,70,80,85`; for an extract folder, the limits of its last run) |
| `--out DIR` | the output folder (default `<folder>_jpeg_extract` next to the dataset folder); with several folders, one subfolder per folder name |
| `--max-pixels N` | skip images larger than N pixels (default 4194304, which is 2048 x 2048) |
| `--exclude NAME` | another folder name to skip, at any depth; may repeat |
| `--sidecars LIST` | the extensions of the files that travel with an image, comma-separated (default `.txt`) |
| `--reanalyse` | ignore the cache and measure every image again |
| `--threads N` | threads that decode the images (default 8) |

## What a run does

1. Scans the folder with all its subfolders. Images are `.jpg`, `.jpeg`, `.jpe`, `.jfif`, `.png`, `.webp`, `.bmp`, `.tif`, `.tiff`, `.gif`, `.avif`, `.heic` and `.heif`. A `.txt` file with the same name as an image is its caption. Folders named `_backup`, `_duplicates`, `_prep`, `_classify`, `_embeddings`, `masks` and `faces` are skipped, at any depth.
2. Measures the QF of every image on its stored pixels, before EXIF rotation, so the 8 x 8 block grid of the JPEG stays where the model expects it. A black-and-white image (one channel, or three channels that differ by at most 2 levels) is measured with both models, and the gray model decides. An image that has not changed since the last run is taken from the cache.
3. Puts each image in the band of the first limit above its QF. With the default limits, band `60` holds QF under 60, band `70` holds 60 to 69.9, band `80` holds 70 to 79.9 and band `85` holds 80 to 84.9. An image is in one band only, so two neighbouring bands can be compared side by side; a threshold of 80 later takes the bands `60`, `70` and `80`. Images at 85 or above are not copied.
4. Copies each image of a band, with its captions, into the band folder as `q<QF>__<name>`, for example `q057__photo.jpg` and `q057__photo.txt`, so a folder sorted by name is sorted by QF. When two images of different folders have the same name and QF, the second gets `~2`. The copy keeps the modification time.
5. Writes `extract.csv`, `summary.txt` and `manifest.json` into the output folder and prints the summary.

A new run into the same output folder first deletes the copies of the last run (they are listed in `manifest.json`); other files in the band folders stay.

Not measured, and counted as skipped in the summary:

| skip | why |
|---|---|
| `large` | more than `--max-pixels`: the downscale to the training resolution hides the artifacts |
| `format GIF`, `format AVIF`, `format lossy WEBP` and other formats | their own compression artifacts, which FBCNN does not know |
| `mode CMYK`, `mode I;16` and other modes | CMYK, 16-bit and 1-bit images |
| `transparent` | an alpha channel that is not fully opaque |
| `animated` | more than one frame |
| `unreadable` | the file cannot be decoded |

JPEG, MPO, PNG, BMP, TIFF and lossless WebP are measured.

The run prints a progress line every 500 images with the time left. Ctrl+C stops it; the measurements made so far stay in the cache, and the next run continues from there.

## Sorting an extract folder again

An extract folder given instead of a dataset folder is sorted in place from its `extract.csv` and its `manifest.json`; the dataset is not read and the model is not loaded. New `--bands` move copies with their captions between the band folders. Copies above a lower new top limit are deleted with their captions. Images above the old top limit were never copied, so a higher new top limit cannot add them; the run says how many there are, and a run on the dataset folder adds them.

`extract.csv` and `manifest.json` are written first, then the moves are made. When a run is stopped, the next run finishes its moves from `moves.json` before anything else.

## The fix

```
run.bat <folder> [<folder> ...] [--dry-run] [options]
run.bat <folder> --undo
```

Examples:

```
run.bat D:\photos --dry-run
run.bat D:\photos --dry-run --sheet-offsets 0,20
run.bat D:\photos
run.bat D:\photos --undo
```

The first command restores every image of `D:\photos` with a QF under 80 in memory, decides for each whether the restoration is worth saving, and writes the report and the contact sheets into `D:\photos\_backup\_jpeg_cleanup\`; no image changes. The second shows two more restorations on the sheets, at offsets 0 and 20, next to the default one, to choose `--qf-offset` by eye. The third makes the fix: the restorations worth saving replace their images. The fourth puts back everything the last run changed.

| option | what it does |
|---|---|
| `--threshold QF` | fix the images with a QF under this (default 80) |
| `--qf-offset N` | added to the predicted QF that FBCNN is told (default 10); see below |
| `--min-block-drop X` | a restoration is worth saving when the blockiness drops by at least X (default 0.10) |
| `--min-qf-gain N` | ... or when the QF rises by at least N on an image with a visible JPEG grid (default 25) |
| `--sheet-offsets LIST` | more offsets shown side by side on the sheets, comma-separated |
| `--no-sheets` | with `--dry-run`: the report only |
| `--sheets` | with a real run: the contact sheets too |
| `--undo` | put back everything the last run of each folder changed |
| `--quality Q` | the JPEG quality a restored JPEG is saved with (default 97) |

`--max-pixels`, `--exclude`, `--sidecars`, `--reanalyse` and `--threads` work as for `extract.bat`. The QF measurements come from the same cache, so after an extract the fix starts restoring at once.

What the fix does with each image under the threshold:

1. Restores the stored pixels (before EXIF rotation) with the colour model of FBCNN. A black-and-white image goes in as three equal channels, and only the luma of the result is kept, so no colour can appear. The gray model is not used for the fix: on this kind of dataset it smoothed the film grain of black-and-white photos away and hardly reacted to `--qf-offset`, where the colour model kept the grain and removed the blocks as well. FBCNN is told the quality of the image: the QF the colour model predicts, plus `--qf-offset`. Told a higher quality, it removes less, so a positive offset keeps more film grain and fine texture, at the price of a little more of the artifacts. The effect is mild: from +0 to +20 the fine detail kept rises from about 78% to 87% of the original's.
2. Encodes the result as it would be written: a JPEG source as JPEG at `--quality` with full-resolution colour (4:4:4, so the re-save adds no colour bleeding), a PNG, BMP, TIFF or lossless WebP source in its own format. The JPEG re-save at 97 differs from the restoration by 46 to 51 dB PSNR, far below the artifacts removed.
3. Measures the QF of the result again (`qf_after`), the blockiness before and after, and the mean change from the original.
4. Decides whether the restoration is worth saving (section [Enough benefit](#enough-benefit)). If it is not, the image stays as it is and its original never goes to `_backup`.
5. If it is, and this is not a dry run: logs the image, copies the original and its captions to `_backup` under the same relative path, writes the restoration to a `.part` file and puts it in place of the image in one step. The image keeps its name, its modification time and its read-only flag, its EXIF block (with the orientation) and its ICC profile; a gray result leaves out an RGB profile, which does not fit it.

Images at or above the threshold are kept as they are. The restorations are computed deterministically, so a real run saves exactly the images its dry run listed as `fix`. A second run finds the restored images clean (their QF is now around 90) and leaves them alone.

An extract folder can be given to a dry run instead of the dataset: its copies hold the bytes of their sources, so the report and the sheets show what a run on the dataset would do. The dry run then writes into the `_backup` folder of the extract, which a re-sort of the extract leaves alone. A real run refuses an extract folder.

The restored JPEGs are larger than their originals: quality 97 with full-resolution colour takes about 2.6 times the space of the quality 70 to 80 files it replaces (417 images went from 45 to 119 MB). At `--quality 95` they are about 2 times the size of the originals, at 92 about 1.6 times.

### Backup and undo

The original of a restored image goes to `_backup\<relative path>`, with copies of its captions. When `_backup` holds an earlier, different original of the same path already (for example from `remove_borders`), that one stays, and this run's original goes to `_backup\_jpeg_cleanup\originals\<run>\<relative path>` instead. A backup that exists is never overwritten.

`--undo` takes the last run that is not undone yet and puts every image it restored back from where the run kept its original, newest first; a backup the run made is removed, one that was there before stays. Caption copies the run made are removed when the caption in place is unchanged; a caption that went missing comes back. An interrupted run can be undone too, since every image is logged before it is touched. Each `--undo` goes one run further back. A run that wrote nothing logs no run.

### Enough benefit

A restoration always changes the image a little; it is saved only when it removes enough of the JPEG artifacts to be worth a re-saved file. Two measures decide, and film grain fakes neither:

- **Blockiness**: the mean luma step across the edges of the 8 x 8 JPEG blocks, divided by the mean step between other neighbouring pixels. About 1.0 when no block grid shows; 1.1 to 1.5 on the blocky images of a typical dataset. Grain and fine texture raise both steps alike and leave it near 1.0. A drop of at least `--min-block-drop` (0.10) means the grid visibly fades.
- **QF gain**: how much FBCNN's own QF rises from the original to the result. It also sees ringing and mosquito noise around edges, which blockiness misses; an image at QF 45 whose ringing goes reads 90 afterwards. A dithered or screened print fools it (such a print reads QF 40 with no JPEG damage at all), so a gain of at least `--min-qf-gain` (25) counts only on an image whose blockiness is at least 1.05.

The mean change of the pixels is no measure of benefit: on scanned prints it mostly counts the grain a restoration takes away.

On a dataset of 10,000 photos, mostly scans of old prints that had been re-saved several times (3,258 under QF 80), the rule saved 1,774 restorations: 68 of 75 under QF 60, 349 of 424 at 60 to 69, 975 of 1,750 at 70 to 74 and 382 of 1,009 at 75 to 79. By eye, the restorations with a blockiness drop under 0.03 were indistinguishable from their originals, those at 0.06 to 0.10 showed small gains at 3x, and those from 0.15 up removed visible blocks.

### The contact sheets

`_backup\_jpeg_cleanup\sheets\fix_NNNN_qLO-HI.jpg` for the restorations worth saving and `nofix_NNNN_qLO-HI.jpg` for the ones left alone, 8 images per sheet, in QF order, the QF range in the name. The label of each row says SAVE with the reason, or LEAVE, and gives the QF, the blockiness and the change before and after. Each row has a thumbnail with the crop marked in red; a 256 x 256 crop at 100%, first the original, then each restoration; and the most changed 96 x 96 part of that crop, magnified 3 times, in the same order. The crop sits where the restoration changed the image most: on a blocky image that is where the blocks were, on a grainy one where the most grain went. Judge both: the artifacts should be gone, and the grain of a film photo or the dots of a printed one should not be smoothed into plastic. The sheets are JPEG at quality 95 with 4:4:4 colour, so their own compression stays far below what they show. A run deletes the sheets of the last one.

### The report

`_backup\_jpeg_cleanup\report.csv`, one row per image: `path`, `format`, `mode`, `width`, `height`, `gray`, `header_q`, `qf`, `action` (fix: worth saving; little benefit: restored but left alone; keep: at or above the threshold; or the skip reason), `benefit` (why a fix is worth saving), `qf_used` (the QF FBCNN was told), `write` (jpeg, or the lossless format), `qf_after` (the QF of the result), `change` (the mean difference between the original and the result, in 8-bit levels), `block_before` and `block_after` (the blockiness), `result` (in a real run: written, or the error).

## Output

```
<folder>_jpeg_extract\
  60\q050__photo.jpg       the images with QF under 60, and their captions
  70\ 80\ 85\              the other bands
  extract.csv              one row per image of the dataset
  summary.txt              counts per band, a QF histogram, the skipped images
  manifest.json            the files this run copied
<folder>\_backup\
  <relative path>\<image>  the original of a restored image
  <relative path>\<name>.txt  its captions
  _jpeg_cleanup\
    cache.json             the measurements, reused while a file is unchanged
    report.csv             the fix of the last run (dry or real), one row per image
    sheets\                its contact sheets
    log.jsonl              every real run and undo, step by step
    originals\<run>\       originals whose place in _backup was taken
```

`extract.csv` columns: `path` (relative to the dataset folder), `format`, `mode`, `width`, `height`, `megapixels`, `gray` (1 for a black-and-white image), `header_q` (the quality the JPEG header was saved with, estimated from its luminance table), `qf_color` and `qf_gray` (the QF of each model), `qf` (the one that decides), `band`, `copy` (the copy, relative to the output folder), `status` (copied, above, or the skip reason). The file is UTF-8 with a byte order mark, so Excel shows Cyrillic names correctly.

`header_q` and `qf` agree within a few points on a JPEG saved once. A `qf` well under `header_q` means the image was compressed harder before its last save.

`extract.bat` and a dry run write only into `_backup\_jpeg_cleanup`; a real run also writes the restored images and their originals in `_backup`. The other dataset tools skip `_backup`. When you are sure of the fix, delete the backup files to free the space; keep `_jpeg_cleanup` if you may want `--undo`.

## Choosing the threshold

Look through the band folders from the lowest up and find the band where the images stop looking compressed at the size they will be trained at. Things to watch:

- Size matters as much as QF. A 3 MP image at QF 75 is downscaled to 1 MP for training, which hides most of its artifacts; a 0.3 MP image at QF 75 is not downscaled. `megapixels` in `extract.csv` shows which is which.
- FBCNN sees JPEG artifacts only. A blurry frame from a video, or an image upscaled from a small original, can read QF 85 or more and still look poor.

## Speed

On an RTX 5090 a mix of 0.2 to 4 MP images is measured at about 35 images per second. A second run with other bands takes seconds. The fix restores about 8 images per second with sheets, 10 without; each extra sheet offset costs one more restoration per image.
