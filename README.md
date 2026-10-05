# Dataset tools

Tools that prepare image datasets for LoRA and fine-tune training, mainly with [Ostris AI Toolkit](https://github.com/ostris/ai-toolkit). Each tool is in its own folder with its own README.

| folder | what it does |
|---|---|
| [face_masks](face_masks/README.md) | Makes per-image loss masks that hide faces from training, and cuts close-up face crops sorted into 512, 768 and 1024 buckets. |
| [telegram_dataset](telegram_dataset/README.md) | Turns the photos of a Telegram channel export into images with clean one-line captions. |
| [taggui_captioning](taggui_captioning/README.md) | A TagGUI prompt that turns a photo and its editor's caption into an English text-to-image prompt, with the TagGUI settings it needs. |
| [ddg_images](ddg_images/README.md) | Downloads the image results of Bing, DuckDuckGo or SearXNG searches, page by page, removes duplicates and logs every image to a CSV file. |
| [deduplicate](deduplicate/README.md) | Finds copies of the same picture across folders (hashes plus a feature check for crops, borders and mock-ups) and moves the worse copies, with their captions, to a `_duplicates` folder; a kept copy without a caption gets a copy of one. A curated collection given with `--sorted` receives the better copies from raw folders under its own names and keeps its captions. `compare.bat` shows which files were judged copies of which, side by side, for a check by eye; `review.bat` (or `--review` after a run) opens the groups full screen and lets you change the kept copy with a click. `--undo` puts everything back, including review changes. |
| [reframe](reframe/README.md) | Crops photos of people to the subject (one person or a group, without passers-by) in a k2prep aspect ratio; lossless for JPEG, and with `--resize` straight into k2prep buckets. |

`face_masks` is written for Windows and needs a CUDA GPU. `telegram_dataset` needs only Python. `ddg_images` needs Python, and the `ddgs` package for the Bing and DuckDuckGo backends; on Windows, its `install.bat` sets this up. `deduplicate` needs Python and runs on the CPU; its `install.bat` sets it up. `reframe` is written for Windows and needs a CUDA GPU; its `install.bat` sets it up.

MIT License.
