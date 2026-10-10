# Pipeline

Runs the repair tools of this repository over a dataset from a job file: [remove_borders](../remove_borders/README.md), [watermark](../watermark/README.md), [reframe](../reframe/README.md), [jpeg_cleanup](../jpeg_cleanup/README.md) and [face_masks](../face_masks/README.md). It runs one model at a time: detection passes over all images first, each loading one model and caching what it finds, then the plan, then one apply pass that decodes each image once, runs the in-memory chain, and writes it once. There are no temporary files, and never more than one detector in memory. It is made to be driven by another program (a TagGUI dock) as a subprocess, so progress goes to stdout as JSON lines, but it runs from the command line just the same.

## The passes

Every pass goes over all the images of the job (or its file list) and prints one JSON line per image and a summary line. The detection passes only read the images and can overlap with other read-only work; the apply pass changes files and must run alone.

1. **borders** (CPU): remove_borders analyses the stored pixels, as it does on its own, and keeps the cuts it finds.
2. **watermark** (one YOLO model): the watermark boxes of the full original.
3. **reframe** (its three YOLO models, one pass): the people of the full original, with their measurements.
4. **faces** (the face YOLO model): the face boxes of the full original.
5. **quality** (the FBCNN quality predictor): the JPEG quality factor of the original stored pixels.
6. **compose** (no model): one crop box from the border cuts, the watermark trim and the subject crop, on the JPEG block grid; the watermark boxes still inside it; the face boxes moved into it (a crop is a pure translation, so detection on the original is the same as detection on the crop); and whether the image is eligible for the cleanup (quality factor under the threshold, and the composed crop within the size limit).
7. **benefit** (FBCNN again, on the eligible images only): jpeg_cleanup decides to save a restoration only after restoring in memory and measuring the drop in blockiness, so this pass restores the composed crop of each eligible image and keeps the verdict. The apply pass restores again: FBCNN runs twice on each saved image, the price of an accurate dry run.
8. **apply** (LaMa and FBCNN, each loaded on first use; no YOLO model): decode once upright, crop from the composed box, paint the watermark boxes left inside the crop, restore when the verdict says so, write once, then the face mask of the final image from the moved boxes.

A dry run is passes 1 to 7: it writes `plan.json` and the previews and changes nothing. Every detection is cached in `<folder>/_backup/_pipeline/cache.json` by the file's size and modification time and the model's signature, so a pass whose results are all cached decodes nothing and loads no model, and a real run after a dry run does no new detection work: only the painting and the restoration, which produce the pixels, run in the apply pass.

## What happens to one image

**Compose.** The crop is the border crop first, then the subject crop inside it, then the watermark trim (with `"mode": "trim"`, the largest watermark-free rectangle, when it keeps `trim_min_keep` of the picture; a trim that would cut more than half of the subject crop is dropped and the watermark is painted instead). The box is put on the JPEG block grid in stored pixels: the top and left edges move inward where a border or a trim set them, so the border goes completely, and outward elsewhere, so the subject keeps its margins; the far edges stay. This holds also when the pixels are re-encoded, because jpeg_cleanup measures the quality on the stored 8x8 grid. An image that remove_borders would mark for review or as too small is left as it is and the reason is logged.

**Apply.** The file is read once into an upright RGB array (EXIF orientation applied, transparency composited over white) and cropped. The watermark boxes inside the crop are painted out with LaMa, each in a window of context around it. When the benefit verdict says so, FBCNN restores the pixels, told the image's own quality factor plus the offset. Then the image is written once. If no pixel stage touched it (no painting, no restoration), the file itself is cropped with the composed box: whole DCT blocks for JPEG (no re-encoding, EXIF kept with the orientation, the thumbnail dropped), the exact pixels for PNG and the other lossless formats, and a lossy format (WebP, AVIF, HEIF) into a PNG. Otherwise the array is written: a JPEG source as JPEG at quality 97 without chroma subsampling, with its EXIF (orientation reset, since the pixels are upright now) and ICC profile; any other source as PNG; with `"save_png": true` every touched image as PNG. An image that no stage changed is not written at all in place, and copied in parallel mode. Last, the face mask of the final pixels goes to `masks/<stem>.png` next to the image (white where the training loss counts, black over the faces; `invert` the other way round), as face_masks writes it.

Captions (`.txt` next to the image) are never changed or deleted. In place, the caption of a changed image is copied to the backup with the image; in parallel mode every caption is copied to the output. The image keeps its stem, also when a JPEG becomes a PNG, so the caption stays its caption.

## The job file

```
run.bat --job job.json [--dry-run] [--undo] [--threads N]
```

`run.bat --example` prints a job with every option at its default. Stages that are not in `stages` do not run; a stage with `{}` runs with its defaults.

```json
{
 "folder": "D:/datasets/set",
 "files": ["optional/relative/path.jpg"],
 "stages": {
  "borders": {"dark": true, "min_area": 65536},
  "watermark": {"conf": 0.1, "dilate": 15, "mode": "inpaint", "trim_min_keep": 0.5, "max_size": 2048},
  "reframe": {"ratios": ["9:16", "2:3", "4:5", "1:1", "5:4", "3:2", "16:9"], "overrides": true},
  "jpeg_cleanup": {"threshold": 80, "qf_offset": 10, "min_block_drop": 0.1, "min_qf_gain": 25, "max_pixels": 4194304},
  "face_masks": {"conf": 0.3, "grow": 1.35, "feather": 12, "include_hair": false, "invert": false}
 },
 "output": {"mode": "in_place"},
 "save_png": false,
 "dry_run": false,
 "undo": false,
 "threads": 4
}
```

| key | meaning |
|---|---|
| `folder` | the dataset folder, scanned at any depth; the output folders of the tools (`_backup`, `_duplicates`, `_prep`, `_classify`, `_embeddings`, `masks`, `faces`) are skipped, `exclude` adds names |
| `files` | optional: only these images, as paths relative to the folder |
| `stages` | the stages to run with their options; every option has the default of its tool; `reframe.overrides` reads `overrides.txt` in the folder as reframe does |
| `output` | `{"mode": "in_place"}` changes the dataset folder; `{"mode": "parallel", "folder": "..."}` writes the whole tree into that folder, which must be outside the dataset, and copies the unchanged images and all captions along |
| `save_png` | write every touched image as PNG |
| `dry_run` | the detection passes, the plan and the previews; change nothing (also `--dry-run`) |
| `undo` | return the images of the last run of the folder to their originals (also `--undo`) |
| `threads` | decoder threads and writer processes (also `--threads`); the models run one image at a time |

## Output

Each pass prints one JSON line per image with `pass`, `index`, `total` and `path` (relative to the folder): a detection pass adds `cached` when the result was in the cache, or `skipped` or `error`; the compose pass adds `changed` (the stages that change the image, empty when none does), `notes` (what each stage found) and `box` (the crop in upright pixels, when there is one); the apply pass adds `write` (`lossless`, `array` or `none`), `written` (the path of the output), `mask`, and `removed` when a JPEG became a PNG. Each pass ends with a summary line (`"summary": true` with the pass name and its counts), and the run with a final summary: the counts, the seconds, and the peak VRAM. The plan goes to `_backup/_pipeline/plan.json`, the apply lines also to `_backup/_pipeline/report.jsonl`.

## Backup and undo

A backup is the one untouched original of an image, shared with the other tools: `<folder>/_backup/<same relative path>`, with its caption next to it. In place, the apply pass copies the original there right before its single write, once per image however many stages change it, and never over an original that is there already (from remove_borders, jpeg_cleanup or an earlier pipeline run). A mask that exists already is backed up the same way. The run is logged in `_backup/_pipeline/log.jsonl`, one line per image before and after its write, so `--undo` works after an interruption too.

`--undo` returns the images of the last run to their originals: every image the run wrote comes back from `_backup`, a renamed output (a JPEG written as PNG) is removed, the masks are restored or removed, and in parallel mode the files written into the output folder are removed. Since the only backup is the original, undoing a later run also reverts earlier runs' changes to the same image. Backups that the run copied are removed by the undo; earlier ones stay. Each `--undo` goes one run further back.

## Previews

A dry run writes into `_backup/_pipeline/previews`: `reframe/<path>.jpg` with the numbered people and the planned crop (the numbers go into `overrides.txt`), `watermark/<path>.jpg` with the painted area in red and a trim box in yellow, `borders/` with the contact sheets of remove_borders (the cut lines magnified at the four corners), `cleanup/` with the before-and-after sheets of jpeg_cleanup for the eligible images. The previews pass reads the images once more.

## How it runs

Within a detection pass the images are decoded in threads, a few ahead of the model, which runs one image at a time in the main thread; the border analysis, which needs no GPU, runs in the decoder threads. The model of a pass is loaded on its first image and dropped at the end of the pass, so the peak memory is that of the largest single model plus, in the apply pass, LaMa and FBCNN together. In the apply pass the writes run in worker processes (jpeglib for the lossless JPEG crop must not share a process with other threads that use Pillow), a few at a time, so the GPU does not wait for the disk. Each image is decoded once per pass that needs its pixels; the passes whose results are cached decode nothing.

## Install

Run `install.bat` in the repository root. It runs the `install.bat` of remove_borders, reframe, jpeg_cleanup, face_masks and watermark one after the other; each creates the shared `..\venv` when it is missing and installs only its own dependencies, so running it again installs what is missing. The models are downloaded by those installers and found locally from then on.

## Project files

| file | purpose |
|---|---|
| `pipeline.py` | the tool; imports the five tools from their folders |
| `run.bat` | runs `pipeline.py` with the shared venv and passes all arguments through |
| `../install.bat` | installs the five tools in sequence |
