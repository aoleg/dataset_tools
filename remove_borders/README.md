# Remove borders

`remove_borders.py` finds images with borders in a large dataset and cuts the borders off, in place. It finds light and dark frames (also thin ones of 1 to 2 px, slanted ones from rotated scans, and a print on a card scanned on a white bed), borders on one or two sides, and banners with text at the bottom or top, such as the grey strip with a site address and an uploader name that photo-sharing sites add. JPEG files are cropped losslessly: the compressed data is cut, never decoded and saved again. Before an image changes, the original and its caption go to a `_backup` folder, and `--undo` puts everything back.

The tool is for training datasets where a border would teach the model to draw one. It leaves an image alone when it is not sure: a product shot on a white background, a sky across the top of a photo or a coloured band that belongs to a poster's design is not a border.

## Install

On Windows, run `install.bat`. It needs Python 3.10 or newer; no GPU.

It creates the shared `..\venv` folder when it is missing and installs `requirements.txt`: Pillow, numpy, jpeglib (the lossless JPEG crop) and OpenCV (only for 16-bit colour PNG, which Pillow reads as 8 bit).

## Usage

```
run.bat <folder> [<folder> ...] [options]
run.bat <folder> --undo
```

Examples:

```
run.bat D:\photos --dry-run
run.bat D:\photos
run.bat D:\photos --undo
```

The first command analyses every image and writes a report and contact sheets into `D:\photos\_backup\_remove_borders\`, without changing an image. The second makes the crops. It reuses the analysis of the dry run for every file that has not changed since, so it starts cropping at once. The third puts back everything the last run changed.

Each folder on the command line is scanned with all its subfolders and has its own `_backup` folder. Do not give a folder together with one of its subfolders.

| option | what it does |
|---|---|
| `--dry-run` | analyse and write the plan, the report and the contact sheets; change nothing |
| `--no-sheets` | with `--dry-run`: no contact sheets |
| `--sheets` | with a real run: write the contact sheets too |
| `--undo` | put back everything the last run of each folder changed |
| `--exclude NAME` | another folder name to skip, at any depth; may repeat |
| `--sidecars LIST` | the extensions of the files that travel with an image, comma-separated (default `.txt`) |
| `--min-area N` | an image smaller than N pixels after its crop is not cropped but moved to `_backup` (default 65536, which is 256 x 256) |
| `--no-dark` | leave dark frames and borders alone |
| `--reanalyse` | ignore the analysis of the last run and analyse every image again |
| `--threads N` | worker processes (default 8) |

## What a run does

1. Scans the folder with all its subfolders. Images are `.jpg`, `.jpeg`, `.jpe`, `.jfif`, `.png`, `.webp`, `.bmp`, `.tif`, `.tiff`, `.gif`, `.avif`, `.heic` and `.heif`. A `.txt` file with the same name as an image is its caption. Folders named `_backup`, `_duplicates`, `_prep`, `_classify`, `_embeddings`, `masks` and `faces` are skipped, at any depth; every other folder is scanned, also one whose name starts with `_`. Animated images and files that cannot be read are skipped and listed.
2. Analyses every image (section [What counts as a border](#what-counts-as-a-border)) and plans the crop (section [How an image is cropped](#how-an-image-is-cropped)).
3. Writes `plan.json`, `report.csv` and `cache.json` into `<folder>\_backup\_remove_borders\`, and the contact sheets into its `sheets\` folder. With `--dry-run` the run stops here.
4. For every image to crop: copies the image, its captions and its face mask to `_backup` under the same relative path, then writes the crop in place. An image too small after its crop moves to `_backup` with its captions instead.
5. Logs every step in `log.jsonl` and adds the result of every image to `report.csv`.

The run prints a progress line every 500 images with the time left. Ctrl+C stops it: the images being cropped at that moment are finished, the rest are not started, and `--undo` puts back what was done.

## What counts as a border

The tool looks at the stored pixels (before EXIF rotation) and repeats the search on what is left, up to four times, so a frame inside a frame (a cream mount with a dark line) comes off in two or three rounds.

**Banners** at the bottom or top: a band of one neutral colour (grey, white or black) with text in it and a sharp straight edge to the picture, up to 12% of the height. Without text, a band is a frame side or nothing. A coloured band with text is part of a poster's design and stays.

**Frames** on three or four sides: light (mean at least 160) or dark (mean at most 70) bands whose inner edge is a straight line along each side. The edge may slant by up to 2 degrees (a print scanned slightly rotated); the cut is then at its deepest point, so no wedge of the border stays. The picture may touch the frame in light places (a pale sky) without breaking it. The inner corners of the frame must be picture: a rounded object on a white background has corners in the background colour and is no frame. The open fourth side of a three-sided frame must not be the frame colour too, or the "frame" is a studio background.

**Borders on one or two sides**: the same, but stricter: one straight depth without slant, almost no gaps, at most 5% of the side, and no other edge of the image in the same neutral colour. A border of up to 3 px is a **line**.

Mid tones are never borders. The colour of a border is matched within 24 levels per channel, which covers JPEG noise and scanned paper.

## How an image is cropped

| format | crop |
|---|---|
| JPEG, MPO | lossless: the DCT blocks outside the box are dropped, the rest is copied unchanged. The left and top edges move inward to the next block boundary (8 or 16 px), so the border goes completely; this can cost up to 15 px of picture there. The right and bottom edges are exact. EXIF stays, without its thumbnail and with the new size; the ICC profile stays. |
| PNG, lossless WebP, TIFF, GIF, BMP | exact to the pixel, saved in the same format and mode, with palette, transparency, ICC profile, EXIF and PNG text. 16-bit PNG stays 16-bit. |
| lossy WebP, AVIF and other lossy formats | exact to the pixel of the decoded image, saved as `<name>.png` next to it; the original moves to `_backup`, the caption stays for the PNG. When `<name>.png` exists, the image is skipped. |

A thin line on a JPEG is cut only when the crop keeps the image in the same k2prep bucket (aspect-ratio family and resolution tier of 1024, 768 or 512). The block alignment at the left or top can take more than the line, and that must not cost an image its tier. When the bucket would change, the line stays and the report says so. An image too small for the 512 tier anyway is cut.

A face mask made by `face_masks` (`<folder>\masks\<name>.png`) is cropped with the same box, turned with the EXIF orientation of its image.

Every write goes to a `.part` file first and replaces the image in one step. The image keeps its modification time and its read-only flag.

## The `_backup` folder

```
<folder>\_backup\
  <relative path>\<image>             the original of a cropped image, or a too-small image
  <relative path>\<name>.txt          its captions
  <relative path>\masks\<name>.png    its face mask
  _remove_borders\
    report.csv                        one row per image: what was found, what was done
    plan.json                         the analysis and the crop plan of every image
    cache.json                        the analysis, reused while a file is unchanged
    log.jsonl                         every run and undo, step by step
    sheets\                           contact sheets of the last dry run (or real run with --sheets)
```

A backup that exists is never overwritten, so `_backup` keeps the first original of an image even after several runs. The other dataset tools skip `_backup`. When you are sure of the crops, delete the backup files to free the space; keep `_remove_borders` if you may want `--undo` for the newest run.

## Checking the crops

The contact sheets show every image to crop, grouped as `frame_NNN.jpg`, `border_NNN.jpg`, `banner_NNN.jpg` and `review_NNN.jpg`, 12 per sheet. Each tile has a thumbnail with the box that stays in red, and the four corners of that box magnified four times, so a cut of 1 or 2 px can be judged.

`report.csv` columns: `path`; the file facts `format`, `kind`, `mode`, `bits`, `width`, `height`, `mcu`, `orientation`, `frames`, `sidecars`, `mask`; `action` (crop, too small, review, skip or empty); `write` (jpeg, exact or png); `new_width`, `new_height`; `cut_top`, `cut_bottom`, `cut_left`, `cut_right` in pixels; `mcu_lost` (pixels of picture lost at the left and top to the JPEG block grid); `cuts` (each cut with its side, kind, tone, depth and slope); `reason`; `result` of a real run.

`review` means the crop would keep less than half of the image. Such an image is not changed.

## Undo

`--undo` takes the last run that is not undone yet. A cropped image gets its original back from `_backup`; a lossy image gets its original back and the PNG is deleted; a too-small image moves back with its captions and mask. Captions are never overwritten: a caption edited after the run keeps the edit. Each `--undo` takes one run further back. An interrupted run can be undone too, since every image is logged before its crop starts.

A second run can find a little more on a few images: when a frame fades into the picture over many rows, the first cut ends inside the fade, and its pale rest is a thin line on the next run. A third run finds nothing.

## Limits

- A border that is neither light nor dark (a mid-grey mat) is not found.
- A frame whose edge is torn, uneven or slanted by more than 2 degrees may be missed on that side or as a whole.
- A frame with corners rounded by more than about 6 px (1% of the short side on large images) fails the corner test, the price of leaving product shots alone.
- A frame around a product shot on a white background is not cut when the floor shows on the open side.
- Text printed over the picture, such as a watermark, is not a border; no crop can remove it.
- A rectangular object with sharp corners on a white background looks like a framed picture and is cut to the object.
