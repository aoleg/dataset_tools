# JPEG cleanup

`jpeg_cleanup.py` finds heavily compressed images in a large dataset with [FBCNN](https://github.com/jiaxi-jiang/FBCNN), a network that predicts the JPEG quality factor (QF) of an image from its pixels. It sees through re-saves: video frames whose JPEG header says quality 85 read 56 to 66 because they were compressed harder before, and a PNG made from a JPEG reads about the quality of that JPEG.

The tool is built in phases. This version has the first one, `extract.bat`: it measures every image and copies the poor ones into one folder per quality band, outside the dataset, so the threshold for the cleanup can be chosen by eye. The dataset is not changed. The cleanup itself (FBCNN restores the images in place, the originals go to `_backup`) comes in a later version.

## Install

On Windows, run `install.bat`. It needs Python 3.10 or newer and a CUDA GPU; without one the tool runs on the CPU, much more slowly.

It creates the shared `..\venv` folder when it is missing, installs torch from the PyTorch CUDA index and then `requirements.txt` (Pillow, numpy), and downloads the two FBCNN models (288 MB each) into the `models` folder next to the script. Both files are checked against their SHA-256 checksums. After that the tool runs offline: it loads the models from `models` only.

| file | used for |
|---|---|
| `fbcnn_color.pth` | colour images |
| `fbcnn_gray_double.pth` | black-and-white images, also those saved as RGB |

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

## Output

```
<folder>_jpeg_extract\
  60\q050__photo.jpg       the images with QF under 60, and their captions
  70\ 80\ 85\              the other bands
  extract.csv              one row per image of the dataset
  summary.txt              counts per band, a QF histogram, the skipped images
  manifest.json            the files this run copied
<folder>\_backup\_jpeg_cleanup\cache.json
                           the measurements, reused while a file is unchanged
```

`extract.csv` columns: `path` (relative to the dataset folder), `format`, `mode`, `width`, `height`, `megapixels`, `gray` (1 for a black-and-white image), `header_q` (the quality the JPEG header was saved with, estimated from its luminance table), `qf_color` and `qf_gray` (the QF of each model), `qf` (the one that decides), `band`, `copy` (the copy, relative to the output folder), `status` (copied, above, or the skip reason). The file is UTF-8 with a byte order mark, so Excel shows Cyrillic names correctly.

`header_q` and `qf` agree within a few points on a JPEG saved once. A `qf` well under `header_q` means the image was compressed harder before its last save.

The cache is the only file the tool writes into the dataset folder. The other dataset tools skip `_backup`.

## Choosing the threshold

Look through the band folders from the lowest up and find the band where the images stop looking compressed at the size they will be trained at. Things to watch:

- Size matters as much as QF. A 3 MP image at QF 75 is downscaled to 1 MP for training, which hides most of its artifacts; a 0.3 MP image at QF 75 is not downscaled. `megapixels` in `extract.csv` shows which is which.
- FBCNN sees JPEG artifacts only. A blurry frame from a video, or an image upscaled from a small original, can read QF 85 or more and still look poor.

## Speed

On an RTX 5090 a mix of 0.2 to 4 MP images is measured at about 35 images per second. A second run with other bands takes seconds.
