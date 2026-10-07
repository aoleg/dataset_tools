# Dataset tools

Tools that prepare image datasets for LoRA and fine-tune training, mainly with [Ostris AI Toolkit](https://github.com/ostris/ai-toolkit). Each tool is in its own folder with its own README.

| folder | what it does |
|---|---|
| [face_masks](face_masks/README.md) | Makes per-image loss masks that hide faces from training, and cuts close-up face crops sorted into 512, 768 and 1024 buckets. |
| [telegram_dataset](telegram_dataset/README.md) | Turns the photos of a Telegram channel export into images with clean one-line captions. |
| [taggui_captioning](taggui_captioning/README.md) | A TagGUI prompt that turns a photo and its editor's caption into an English text-to-image prompt, with the TagGUI settings it needs. |
| [ddg_images](ddg_images/README.md) | Downloads the image results of Bing, DuckDuckGo or SearXNG searches, page by page, removes duplicates and logs every image to a CSV file. |
| [deduplicate](deduplicate/README.md) | Finds copies of the same picture across folders (hashes plus a feature check for crops, borders and mock-ups) and moves the worse copies, with their captions, to a `_duplicates` folder; a kept copy without a caption gets a copy of one. A curated collection given with `--sorted` receives the better copies from raw folders under its own names and keeps its captions. `compare.bat` shows which files were judged copies of which, side by side, for a check by eye; `review.bat` (or `--review` after a run) opens the groups full screen and lets you change the kept copy with a click. `--undo` puts everything back, including review changes. `--gpu` matches the image features on the GPU when torch is in the shared venv, which cuts the slow stage from minutes to seconds on large sets. |
| [reframe](reframe/README.md) | Crops photos of people to the subject (one person or a group, without passers-by) in a k2prep aspect ratio; lossless for JPEG, and with `--resize` straight into k2prep buckets. |
| [classify](classify/README.md) | Sorts a dataset into category folders from a folder of hand-sorted examples per category: every image is embedded once with SigLIP 2, a classifier is trained on the examples, and each image is copied with its caption into its category folder, or into `_unsure` when the classifier is not confident. A dry run writes a report with a confusion matrix, a confidence histogram and contact sheets; `--undo` puts everything back. |
| [remove_borders](remove_borders/README.md) | Finds images with borders (light and dark frames, thin lines, slanted frames of rotated scans, banners with text at the bottom or top) and cuts the borders off in place; lossless for JPEG, exact for PNG and other lossless formats, lossy formats become PNG. Originals and captions go to a `_backup` folder first, an image too small after its crop moves there instead, and `--undo` puts everything back. A dry run writes a report and contact sheets with the cut lines magnified. |
| [flatten_dataset](flatten_dataset/README.md) | Moves the images and captions of all subfolders into the dataset folder under normalised names (`<folder>__<subfolder>-datasetNNNNNN.ext`: transliterated, no spaces, at most 80 characters), always an image and its caption together. `--undo` puts every file back from the manifest in `_flatten_dataset`. |
| [jpeg_cleanup](jpeg_cleanup/README.md) | Finds heavily compressed images with the FBCNN quality predictor, which also sees through re-saves. `extract.bat` copies the poor images with their captions into one folder per quality band (`60`, `70`, `80`, `85`), outside the dataset, so the threshold for the cleanup can be chosen by eye. `run.bat` restores the images under the threshold with FBCNN and replaces only those whose restoration removes enough of the artifacts, with the originals and captions in a `_backup` folder; `--dry-run` writes a report and before/after contact sheets instead, and `--undo` puts everything back. |
| [extract_keywords](extract_keywords/README.md) | Moves the images whose captions contain any of the keywords (or all of them, joined by `AND`) with their captions out of a dataset into `<dataset>_<first keyword>` next to it, in the same subfolders: one concept for its own training, or every image whose caption mentions a watermark. `--undo` moves them back. |

## Install

All tools share one virtual environment: the `venv` folder in the repository root, next to the tool folders. It needs Python 3.10 or newer on Windows. To install a tool, run the `install.bat` in its folder. It creates the shared `venv` when it is missing and installs only that tool's dependencies into it, so a tool that needs no GPU never pulls torch in. Install the tools you need, in any order; running an `install.bat` again installs what is missing. Every `run.bat` uses the shared `venv`.

| tool | needs |
|---|---|
| `telegram_dataset`, `taggui_captioning`, `flatten_dataset`, `extract_keywords` | Python only, nothing to install |
| `ddg_images` | the `ddgs` package for the Bing and DuckDuckGo backends; the SearXNG backend needs nothing |
| `deduplicate` | Pillow, numpy, imagehash, OpenCV; runs on the CPU. With `--gpu` it uses the torch of the GPU tools when one of them is installed, and falls back to the CPU otherwise. |
| `remove_borders` | Pillow, numpy, jpeglib, OpenCV; runs on the CPU |
| `face_masks`, `reframe`, `classify`, `jpeg_cleanup` | a CUDA GPU; their `install.bat` installs torch from the PyTorch CUDA index and downloads their models, after which they run offline |

The GPU tools install the same torch build, so they share it in the `venv`.

MIT License.
