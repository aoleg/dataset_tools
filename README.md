# Dataset tools

Tools that prepare image datasets for LoRA and fine-tune training, mainly with [Ostris AI Toolkit](https://github.com/ostris/ai-toolkit). Each tool is in its own folder with its own README.

| folder | what it does |
|---|---|
| [face_masks](face_masks/README.md) | Makes per-image loss masks that hide faces from training, and cuts close-up face crops sorted into 512, 768 and 1024 buckets. |
| [telegram_dataset](telegram_dataset/README.md) | Turns the photos of a Telegram channel export into images with clean one-line captions. |
| [taggui_captioning](taggui_captioning/README.md) | A TagGUI prompt that turns a photo and its editor's caption into an English text-to-image prompt, with the TagGUI settings it needs. |
| [ddg_images](ddg_images/README.md) | Downloads the image results of Bing, DuckDuckGo or SearXNG searches, page by page, removes duplicates and logs every image to a CSV file. |
| [deduplicate](deduplicate/README.md) | Finds copies of the same picture across folders (hashes plus a feature check for crops, borders and mock-ups) and moves the worse copies, with their captions, to a `_duplicates` folder; a kept copy without a caption gets a copy of one. |

`face_masks` is written for Windows and needs a CUDA GPU. `telegram_dataset` needs only Python. `ddg_images` needs Python, and the `ddgs` package for the Bing and DuckDuckGo backends; on Windows, its `install.bat` sets this up. `deduplicate` needs Python and runs on the CPU; its `install.bat` sets it up.

MIT License.
