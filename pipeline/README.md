# Pipeline

Runs the repair tools of this repository over a dataset image by image, from a job file: [remove_borders](../remove_borders/README.md), [watermark](../watermark/README.md), [reframe](../reframe/README.md), [jpeg_cleanup](../jpeg_cleanup/README.md) and [face_masks](../face_masks/README.md), in that order. Each image is decoded once, passes through the chosen stages in memory, and is written once; there are no temporary files. It is made to be driven by another program (a TagGUI dock) as a subprocess, so progress goes to stdout as one JSON line per image, but it runs from the command line just the same.

## What happens to one image

1. **Decode.** The file is read once into an upright RGB array (EXIF orientation applied, transparency composited over white) together with its header: format, stored size, EXIF orientation, the JPEG block size, the EXIF and ICC blocks.
2. **Borders.** remove_borders finds frames, thin lines and text banners on the stored pixels, as it does on its own, and gives a crop box. An image the tool would mark for review or as too small is left as it is and the reason is logged.
3. **Watermark detection.** The watermark detector runs on the picture inside the borders. With `"mode": "trim"` the largest watermark-free rectangle becomes a crop box when it keeps enough of the picture; otherwise the boxes are painted later.
4. **Reframe.** On the same picture, the three reframe detectors find the people, the subject is chosen (with `overrides.txt` in the dataset folder, as reframe reads it), and the crop is planned in one of the k2prep aspect ratios.
5. **Compose.** One crop box comes out of the three: inside the borders, then the reframe crop, then the trim (a trim that would cut more than half of the reframe crop is dropped and the watermark is painted instead). The box is put on the JPEG block grid in stored pixels: the top and left edges move inward where a border or a trim set them, so the border goes completely, and outward elsewhere, so the subject keeps its margins; the far edges stay. This holds also when the pixels are re-encoded, because jpeg_cleanup measures the quality on the stored 8x8 grid.
6. **Inpaint.** The watermark boxes still inside the crop are painted out with LaMa, in a window of context around each.
7. **JPEG cleanup.** FBCNN measures the quality of the pixels as they are now and restores them when the quality is under the threshold and the restoration removes enough of the artifacts, by the rules of jpeg_cleanup.
8. **Write.** If no pixel stage touched the image (no painting, no restoration), the file itself is cropped once with the composed box: whole DCT blocks for JPEG (no re-encoding, EXIF kept with the orientation, the thumbnail dropped), the exact pixels for PNG and the other lossless formats, and a lossy format (WebP, AVIF, HEIF) into a PNG. Otherwise the array is written once: a JPEG source as JPEG at quality 97 without chroma subsampling, with its EXIF (orientation reset, since the pixels are upright now) and ICC profile; any other source as PNG; with `"save_png": true` every touched image as PNG. An image that no stage changed is not written at all in place, and copied in parallel mode.
9. **Face mask.** On the final pixels, the face detector finds the faces and the mask is written to `masks/<stem>.png` next to the image (white where the training loss counts, black over the faces; `invert` the other way round), as face_masks writes it.

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
| `stages` | the stages to run with their options; every option has the default of its tool |
| `output` | `{"mode": "in_place"}` changes the dataset folder; `{"mode": "parallel", "folder": "..."}` writes the whole tree into that folder, which must be outside the dataset, and copies the unchanged images and all captions along |
| `save_png` | write every touched image as PNG |
| `dry_run` | plan, report and preview; change nothing (also `--dry-run`) |
| `undo` | put the last run of the folder back (also `--undo`) |
| `threads` | decoder threads and writer processes (also `--threads`); the GPU stages run one image at a time |

## Output

Each image gives one JSON line on stdout: `index` and `total`, `path` (relative to the folder), `changed` (the stages that changed it, empty when nothing did), `notes` (what each stage found), `box` (the crop in upright pixels, when there is one), `write` (`lossless`, `array` or `none`), and after the write `written` (the path of the output) and `mask`; a problem gives `skipped` or `error` instead. The last line is a summary with the counts, the seconds, and the peak VRAM of the run. The same lines go to `_backup/_pipeline/report.jsonl`.

In place, the original of every changed image and its caption are copied to `<folder>/_backup/<same relative path>` before the write, never over an earlier backup (one of remove_borders, say): when that place holds another original, this run's goes to `_backup/_pipeline/originals/<run>/<same relative path>`. A mask that exists already is backed up the same way. The run is logged in `_backup/_pipeline/log.jsonl`, one line per image before and after its write, so `--undo` puts the last run back after an interruption too: the images from their backups, a renamed output (a JPEG written as PNG) removed, the masks restored or removed, and in parallel mode the files written into the output folder removed. Backups that this run copied are removed by the undo; earlier ones stay.

A dry run writes the report and the previews into `_backup/_pipeline/previews`: `reframe/<path>.jpg` with the numbered people and the planned crop (the numbers go into `overrides.txt`), `watermark/<path>.jpg` with the painted area in red and a trim box in yellow, `borders/` with the contact sheets of remove_borders (the cut lines magnified at the four corners), `cleanup/` with the before-and-after sheets of jpeg_cleanup. The detections of every stage go to `_backup/_pipeline/cache.json`, keyed by the file and the stage's inputs, so the real run that follows a dry run detects nothing again: borders, watermark boxes, people, quality measurements and face boxes come from the cache, and only the painting and the restoration, which produce the pixels, run on the GPU once more.

## How it runs

The images are decoded in threads, a few ahead of the GPU; the border analysis, which needs no GPU, runs in those threads too. The GPU stages run one image at a time in the main thread, each model loaded on its first use and kept: a job without a stage never loads that stage's models. The writes run in worker processes (jpeglib for the lossless JPEG crop must not share a process with other threads that use Pillow), a few at a time, so the GPU does not wait for the disk.

## Install

Run `install.bat` in the repository root. It runs the `install.bat` of remove_borders, reframe, jpeg_cleanup, face_masks and watermark one after the other; each creates the shared `..\venv` when it is missing and installs only its own dependencies, so running it again installs what is missing. The models are downloaded by those installers (reframe, jpeg_cleanup, face_masks and watermark) and found locally from then on.

## Project files

| file | purpose |
|---|---|
| `pipeline.py` | the tool; imports the five tools from their folders |
| `run.bat` | runs `pipeline.py` with the shared venv and passes all arguments through |
| `../install.bat` | installs the five tools in sequence |
