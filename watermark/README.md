# Watermark

Finds watermarks in the images of a dataset with a YOLO detector and paints them out with LaMa, or cuts them off. The output goes to a new folder next to the dataset; the dataset itself is not changed. The [pipeline](../pipeline/README.md) tool runs the same detection and painting in place, together with the other repair tools.

This is the per-image core of [watermark_remover](https://github.com/aoleg/watermark_remover), brought into this repository without its multi-GPU workers and live console: one process, the images one after the other through the detector, the painting and writing around it in threads and worker processes.

## What it does

1. Scans every folder given at any depth. The output folders of the dataset tools (`_backup`, `_duplicates`, `_prep`, `_classify`, `_embeddings`, `masks`, `faces`) are never scanned; `--exclude` adds names.
2. Runs the detector on each image (upright, as EXIF shows it) and keeps the boxes above `--conf` (default 0.1, low on purpose: the detector is confident about real watermarks and a missed one costs more than a painted patch of background).
3. Paints the boxes out: each box is grown by `--dilate` pixels (default 15), and LaMa fills the area from a window of context around it (at least 128 px, or half the watermark's size), so the rest of the image is not touched at all. A window with a side over `--max-size` (default 2048) is scaled down for LaMa and its fill scaled back; the image itself keeps its size.
4. With `--trim`, the largest rectangle that holds no watermark is cut out instead, when it keeps at least `--trim-min-keep` of the image (default 0.5); otherwise the image is painted. A JPEG trim is lossless: whole DCT blocks, no re-encoding, the top and left edges moved inward to the block grid so that the watermark goes completely. A tiled watermark (more than 20 boxes) is never trimmed.
5. Writes the output tree with the same relative paths. Images without a watermark are copied unchanged, with their `.txt` captions (`--skip-clean` leaves them out). A painted JPEG is written as JPEG at quality 97 without chroma subsampling, with its EXIF (orientation reset, since the pixels written are upright) and ICC profile; a painted image of any other format becomes a PNG; `--png` writes every painted or trimmed image as PNG. An output that exists already is skipped, so a stopped run can be started again; `--overwrite` writes everything anew.

The detections are cached in `<folder>/_backup/_watermark/cache.json` by file size, time and ID, and a run reuses them while the file and the detector settings are unchanged. `--dry-run` only detects: it writes `report.csv` next to the cache, one row per image with its boxes and what a real run would do, and with `--previews` draws the painted area in red and the trim box in yellow over each image with a watermark, into `previews/`. A real run after a dry run detects nothing again.

## Models

| model | source | used for |
|---|---|---|
| `yolov12x-dino3-watermark-detection.pt` | [corzent/yolov12x-dino3-watermark-detection](https://huggingface.co/corzent/yolov12x-dino3-watermark-detection) | detection (default) |
| `yolo11x-train28-best.pt` | [fancyfeast/joycaption-watermark-detection](https://huggingface.co/spaces/fancyfeast/joycaption-watermark-detection) | detection, when the first cannot be loaded |
| `big-lama.pt` | [simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting) release v0.1.0 (the TorchScript export of [LaMa](https://github.com/advimman/lama)) | painting |

The DINOv3 checkpoint was trained with the [DINOV3-YOLOV12](https://github.com/Sompote/DINOV3-YOLOV12) fork of ultralytics. `dino3_compat.py` lets stock ultralytics load and run it: it provides the fork's `DINO3Backbone` layer for inference and teaches the area-attention layer the fork's weight layout. It needs `transformers` and takes about twice the time of the YOLOv11 model per image. The shim is tried on a small image when the model is loaded; if a change of ultralytics or transformers breaks it, the run says so and uses the YOLOv11 detector instead. `--detector yolo11` chooses it outright.

The detectors are downloaded into the Hugging Face cache on first use and looked up there first, with the network off, on every later run; a checkpoint copied into `models/` next to `watermark.py` is used before the cache. LaMa goes into the torch hub checkpoints folder (`%USERPROFILE%\.cache\torch\hub\checkpoints`), where simple-lama-inpainting keeps it, so a copy it fetched is reused. The LaMa model is loaded with the 30 lines of simple-lama-inpainting's loader built in here, because that package pins Pillow below 10 and would downgrade the shared `venv`.

## Install

Run `install.bat`. It creates the shared `..\venv` when it is missing, installs torch from the PyTorch CUDA index, then `requirements.txt` (ultralytics, transformers, huggingface_hub, Pillow, numpy, OpenCV, jpeglib), checks that torch is a CUDA build, and downloads and loads the models once (`watermark.py --fetch-models`). Running it again installs what is missing.

## Usage

```
run.bat <folder> [<folder> ...] [options]
```

| option | meaning |
|---|---|
| `--out DIR` | output folder (default `<folder>_watermark_removed` next to the folder; one folder only) |
| `--conf X` | detector confidence threshold (default 0.1) |
| `--dilate N` | grow the detected boxes by N pixels before painting (default 15) |
| `--trim` | cut the largest watermark-free rectangle out instead of painting, when it keeps enough |
| `--trim-min-keep X` | with `--trim`: the share of the image a trim must keep (default 0.5) |
| `--max-size N` | longest side of a LaMa window; larger windows are scaled down for LaMa (default 2048) |
| `--png` | write every painted or trimmed image as PNG |
| `--skip-clean` | do not copy the images without a watermark |
| `--dry-run` | detect only: write the report, and with `--previews` the masks drawn over the images |
| `--overwrite` | write outputs that exist already |
| `--detector dino3\|yolo11` | use this detector only |
| `--reanalyse` | ignore the detection cache |
| `--threads N` | decoder threads and writer processes (default 4); detection runs one image at a time |
| `--fetch-models` | download the detector and LaMa and load them once |

## As a library

`plan(array, head, options)` finds the watermarks of one upright RGB array and decides between painting and a trim; `apply(array, plan)` paints or cuts. `inpaint(array, boxes)` paints given boxes, `largest_clear_rect(boxes, w, h)` finds the trim rectangle, `detector_lazy()` and `lama_lazy()` load the models on first use. The pipeline tool uses these on the image it holds in memory, with the boxes shifted into the crop it has composed.

## Project files

| file | purpose |
|---|---|
| `watermark.py` | the tool |
| `dino3_compat.py` | lets stock ultralytics load the DINOv3-YOLOv12 checkpoint |
| `install.bat`, `run.bat` | install into the shared `venv`, run with it |
| `requirements.txt` | the PyPI dependencies; torch is not in it |
| `models/` | optional local copies of the checkpoints, used before the caches (git-ignored) |
