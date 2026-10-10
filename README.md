# Dataset tools

Tools that prepare image datasets for LoRA and fine-tune training, mainly with [Ostris AI Toolkit](https://github.com/ostris/ai-toolkit) and [musubi-tuner](https://github.com/kohya-ss/musubi-tuner). Each tool is in its own folder with its own README.

| folder | what it does |
|---|---|
| [face_masks](face_masks/README.md) | Makes per-image loss masks that hide faces from training, and cuts close-up face crops sorted into 512, 768 and 1024 buckets. |
| [telegram_dataset](telegram_dataset/README.md) | Turns the photos of a Telegram channel export into images with clean one-line captions. |
| [taggui_captioning](taggui_captioning/README.md) | A TagGUI prompt that turns a photo and its editor's caption into an English text-to-image prompt, with the TagGUI settings it needs. |
| [ddg_images](ddg_images/README.md) | Downloads the image results of Bing, DuckDuckGo or SearXNG searches, page by page, removes duplicates and logs every image to a CSV file. |
| [deduplicate](deduplicate/README.md) | Finds copies of the same picture across folders (hashes plus a feature check for crops, borders and mock-ups) and moves the worse copies, with their captions, to a `_duplicates` folder; a kept copy without a caption gets a copy of one. A curated collection given with `--sorted` receives the better copies from raw folders under its own names and keeps its captions. `compare.bat` shows which files were judged copies of which, side by side, for a check by eye; `review.bat` (or `--review` after a run) opens the groups full screen and lets you change the kept copy with a click. `--undo` puts everything back, including review changes. `--gpu` matches the image features on the GPU when torch is in the shared venv, which cuts the slow stage from minutes to seconds on large sets. |
| [reframe](reframe/README.md) | Crops photos of people to the subject (one person or a group, without passers-by) in a [k2prep](k2prep/README.md) aspect ratio; lossless for JPEG, and with `--resize` straight into k2prep buckets, for Ostris AI Toolkit (`--ostris`) or musubi-tuner (`--musubi`). |
| [classify](classify/README.md) | Sorts a dataset into category folders from a folder of hand-sorted examples per category: every image is embedded once with SigLIP 2, a classifier is trained on the examples, and each image is copied with its caption into its category folder, or into `_unsure` when the classifier is not confident. A dry run writes a report with a confusion matrix, a confidence histogram and contact sheets; `--undo` puts everything back. |
| [remove_borders](remove_borders/README.md) | Finds images with borders (light and dark frames, thin lines, slanted frames of rotated scans, banners with text at the bottom or top) and cuts the borders off in place; lossless for JPEG, exact for PNG and other lossless formats, lossy formats become PNG. Originals and captions go to a `_backup` folder first, an image too small after its crop moves there instead, and `--undo` puts everything back. A dry run writes a report and contact sheets with the cut lines magnified. |
| [watermark](watermark/README.md) | Finds watermarks with a YOLO detector (a YOLOv12 with a DINOv3 backbone, or the YOLOv11 fallback) and paints them out with LaMa in a window of context around each, or with `--trim` cuts the largest watermark-free rectangle out instead, losslessly for JPEG. The output goes to `<folder>_watermark_removed` next to the dataset with the captions; `--dry-run` writes a report and previews of the masks. |
| [pipeline](pipeline/README.md) | Runs remove_borders, watermark, reframe, jpeg_cleanup and face_masks over a dataset image by image from a job file: each image is decoded once, the crop boxes are composed into one on the JPEG block grid, the watermarks inside it are painted, the quality is restored, and the image is written once (a lossless crop of the file when no stage touched the pixels), then its face mask. In place with backups, a log and `--undo`, or into a new folder. One JSON line per image on stdout, for a GUI to drive it. |
| [flatten_dataset](flatten_dataset/README.md) | Moves the images and captions of all subfolders into the dataset folder under normalised names (`<folder>__<subfolder>-datasetNNNNNN.ext`: transliterated, no spaces, at most 80 characters), always an image and its caption together. `--undo` puts every file back from the manifest in `_flatten_dataset`. |
| [jpeg_cleanup](jpeg_cleanup/README.md) | Finds heavily compressed images with the FBCNN quality predictor, which also sees through re-saves. `extract.bat` copies the poor images with their captions into one folder per quality band (`60`, `70`, `80`, `85`), outside the dataset, so the threshold for the cleanup can be chosen by eye. `run.bat` restores the images under the threshold with FBCNN and replaces only those whose restoration removes enough of the artifacts, with the originals and captions in a `_backup` folder; `--dry-run` writes a report and before/after contact sheets instead, and `--undo` puts everything back. |
| [extract_keywords](extract_keywords/README.md) | Moves the images whose captions contain any of the keywords (or all of them, joined by `AND`) with their captions out of a dataset into `<dataset>_<first keyword>` next to it, in the same subfolders: one concept for its own training, or every image whose caption mentions a watermark. `--undo` moves them back. |
| [move_alone](move_alone/move_alone.bat) | Moves the images that have no caption (no `.txt` file with the same name) into a `single_files` subfolder, to caption them or leave them out. Copy `move_alone.bat` into the dataset folder and run it there; it handles that folder only, not its subfolders. |
| [scripts](scripts) | Small single-file scripts, run from the dataset folder or given it as an argument. `extract_uncaptioned_images.py` moves the images without a caption from all subfolders into `_uncaptioned`, in the same subfolders, so only those need captioning; `--restore` moves them back with their new captions. `clean_orphaned_txt.py` deletes the `.txt` captions that have no image. Both take `--dry-run`. |
| [k2prep](k2prep/README.md) | Builds a training set for Ostris AI Toolkit (`--ostris`) or the musubi-tuner Krea 2 trainer (`--musubi`) from a folder of mixed photos: each image that passes a quality score is cropped and resized onto that trainer's exact bucket sizes, on one of 7 aspect ratios in 3 resolution tiers (1024, 768, 512), so the trainer uses it with no second resize or crop. Buckets too small for a batch are merged into their nearest neighbour, and the result goes to `_prep` with its captions, plus a `dataset.toml` for musubi. `--by-tier` gives one folder per tier (`_1024` to `_256`), each holding the whole source tree, as AI Toolkit wants a tree of datasets. The source folder is not changed. `--report` is a dry run; `--sort` files the originals into quality folders instead, and `--copy-to` copies the best originals of a tree into another folder, tier first with `--by-tier`. `cleanup.bat` moves undersized images out of a folder. |

## Workflow

The stages of a dataset and the tools that come in at each of them. Tools marked (*) are in separate repositories: [taggui](https://github.com/aoleg/taggui) and [krea-2-merge-tool](https://github.com/aoleg/krea-2-merge-tool).

```
 1. COLLECT  (side by side)
  ________________________________    ________________________________
 | telegram_dataset               |  | ddg_images                     |
 |   Telegram channel export ->   |  |   Bing / DDG / SearXNG image   |
 |   photos, one-line captions    |  |   search results, CSV log      |
 |________________________________|  |________________________________|
                  |                                   |
                  +-----------------+-----------------+
                                    v
 2. GATHER  (in this order)
  __________________________________________________________________
 | deduplicate               moves worse copies of a picture aside, |
 |                           across folders, into a curated set     |
 | flatten_dataset           all subfolders -> one folder           |
 |__________________________________________________________________|
                                    |
                                    v
 3. SORT AND SELECT  (cleanup.bat first; the rest only copy, so
                      they can run side by side)
  __________________________________________________________________
 | k2prep cleanup.bat        moves images too small to train on out |
 | classify                  sorts into category folders, learned   |
 |                           from a few hand-sorted examples        |
 | k2prep --sort             files the originals by quality tier    |
 | jpeg_cleanup extract.bat  copies badly compressed JPEGs by       |
 |                           quality band, to pick a threshold      |
 |__________________________________________________________________|
                                    |
                                    v
 4. REPAIR  (pipeline runs them all in one pass, image by image,       <-+
             or one at a time in this order)                           |
  __________________________________________________________________   |
 | pipeline                  remove_borders, watermark, reframe,    |  |
 |                           jpeg_cleanup and face_masks in one     |  |
 |                           pass from a job file: decoded once,    |  |
 |                           one composed crop, written once        |  |
 | remove_borders            cuts off frames, lines, text banners   |  |
 | watermark                 finds watermarks, paints them out or   |  |
 |                           trims them off                         |  |
 | reframe                   crops photos of people to the subject; |  |
 |                           with --resize it also does the k2prep  |  |
 |                           step of stage 6, so skip k2prep then   |  |
 | jpeg_cleanup run.bat      fixes compression artefacts in         |  |
 |                           severely damaged JPEGs                 |  |
 |__________________________________________________________________|  |
                                    |                                  |
                                    v                                  |
 5. CAPTION  (in this order)                                           |
  __________________________________________________________________   |
 | taggui (*)                writes the captions with a vision      |  |
 |   + taggui_captioning     model; the prompt turns a photo and    |  |
 |                           its editor's caption into a prompt     |  |
 | move_alone                parks images still without a caption   |  |
 | extract_uncaptioned_images (scripts)                             |  |
 |                           the same for a whole tree, into        |  |
 |                           _uncaptioned; --restore puts back      |  |
 | extract_keywords          moves images out by caption keyword:   |  |
 |                           a concept to train apart, or           |  |
 |                           "watermark" back to watermark         |--+
 | strip_watermark_sentence  cuts the watermark sentence out of     |
 |                           the captions                           |
 |__________________________________________________________________|
                                    |
                                    v
 6. BUILD  (in this order)
  __________________________________________________________________
 | k2prep                    crops and resizes onto 7 ratios in 3   |
 |                           resolution tiers, drops images under   |
 |                           a quality threshold, merges small      |
 |                           buckets; for AI Toolkit and musubi,    |
 |                           with a dataset.toml for musubi         |
 | face_masks                loss masks that hide faces, made on    |
 |                           the final images (AI Toolkit)          |
 |   + extract.bat           face crops in 512/768/1024 buckets,    |
 |                           for a separate face dataset            |
 |__________________________________________________________________|
                                    |
                                    v
 7. TRAIN  (AI Toolkit, musubi-tuner)
                                    |
                                    v
 8. MERGE
  __________________________________________________________________
 | krea-2-merge-tool (*)     merges the trained Krea 2 LoRAs and    |
 |                           checkpoints into one model             |
 |__________________________________________________________________|

 (*) separate repository: aoleg/taggui,
     aoleg/krea-2-merge-tool
```

## Install

All tools share one virtual environment: the `venv` folder in the repository root, next to the tool folders. It needs Python 3.10 or newer on Windows. To install a tool, run the `install.bat` in its folder. It creates the shared `venv` when it is missing and installs only that tool's dependencies into it, so a tool that needs no GPU never pulls torch in. Install the tools you need, in any order; running an `install.bat` again installs what is missing. Every `run.bat` uses the shared `venv`.

| tool | needs |
|---|---|
| `telegram_dataset`, `taggui_captioning`, `flatten_dataset`, `extract_keywords` | Python only, nothing to install |
| `move_alone` | nothing, it is a batch file |
| `scripts` | Python only, nothing to install |
| `ddg_images` | the `ddgs` package for the Bing and DuckDuckGo backends; the SearXNG backend needs nothing |
| `deduplicate` | Pillow, numpy, imagehash, OpenCV; runs on the CPU. With `--gpu` it uses the torch of the GPU tools when one of them is installed, and falls back to the CPU otherwise. |
| `remove_borders` | Pillow, numpy, jpeglib, OpenCV; runs on the CPU |
| `k2prep` | Pillow, numpy, tqdm; runs on the CPU. `--sort --vl` also asks a local vision model server (llama.cpp, koboldcpp or LM Studio), set in `.env` |
| `face_masks`, `reframe`, `classify`, `jpeg_cleanup`, `watermark` | a CUDA GPU; their `install.bat` installs torch from the PyTorch CUDA index and downloads their models, after which they run offline |
| `pipeline` | the five tools it runs; the `install.bat` in the repository root installs remove_borders, reframe, jpeg_cleanup, face_masks and watermark one after the other |

The GPU tools install the same torch build, so they share it in the `venv`.

MIT License.
