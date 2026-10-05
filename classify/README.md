# Classify

`classify.py` sorts a large image dataset into category folders. You define the categories by example: one folder per category, with a few dozen images you sorted by hand. The tool embeds every image once with a vision-language encoder, trains a small classifier on your examples, and copies each dataset image with its caption into the matching category folder. Images it cannot place with confidence go to an `_unsure` folder, so the category folders hold only images you can trust.

The tool is for datasets of tens of thousands of images where a category label per image is needed before training, and where losing the doubtful images is better than sorting them by hand.

## Install

On Windows, run `install.bat`. It needs Python 3.10 or newer and a CUDA GPU.

It creates a `venv` folder and installs `torch` and `torchvision` from the PyTorch CUDA 13.2 index, never from PyPI, because the PyPI torch for Windows has no CUDA. Then it installs `requirements.txt` from PyPI, with the installed torch builds pinned so that nothing can replace them. Last, it downloads the encoder into `models\`:

| model | source | size |
|---|---|---|
| SigLIP 2 so400m, NaFlex | `google/siglip2-so400m-patch16-naflex` on Hugging Face, Apache 2.0 | 4.5 GB |

After the install, the tool does not use the network. If the download fails, the installer prints the repository link, the names of the files and the folder to put them in. Download them by hand and run `install.bat` again.

## Usage

```
run.bat --dataset <folder> [--dataset <folder> ...] --samples <folder> [-o <folder>] [options]
run.bat --dataset <folder> --undo
run.bat -o <folder> --undo
```

Examples:

```
run.bat --dataset D:\photos\raw --samples D:\photos\samples --dry-run --sheets
run.bat --dataset D:\photos\raw --samples D:\photos\samples
run.bat --dataset D:\photos\raw --samples D:\photos\samples --retrain
run.bat --dataset D:\photos\raw --undo
```

The first command embeds, trains, classifies and writes the plan, the report and the contact sheets into `D:\photos\raw_classified`, without copying anything. The second copies the images into the category folders there. The third does the same in two passes, see below. The fourth deletes the copies again.

Without `-o`, the output folder is `<dataset>_classified` next to the dataset folder. With several `--dataset` folders, `-o` is required.

## The samples folder

The samples folder holds one subfolder per category. Its name is the category name and the name of the output folder. A numeric prefix keeps the folders in the order you want:

```
samples\
  1_posters\
  2_postcards_art\
  3_people\
  4_transport\
  5_outliers\
    blurry\
    text_pages\
```

Fill each folder with copies of dataset images that belong to the category. Aim for 40 per category; the tool warns below 20 and stops below 5. Choose typical images, not only the most obvious ones: the classifier learns the border of each category from what you give it, and a folder of only perfect examples makes a narrow category.

A subfolder inside a category folder is a sub-category. The tool trains on it separately and reports it as its parent. Use this for a category that is a mix of unrelated things, such as outliers: blurry scans, text pages and screenshots have nothing in common, and separate folders teach them better than one. Images directly in a category folder that also has sub-category folders form one more sub-category.

A category named `outliers` (after the prefix) is counted as rejects in the report and is otherwise an ordinary category. Folders whose names start with `_` are ignored.

The samples folder is training material only. Nothing in it is written to the output. Copy images into it, do not move them: the dataset copy is what reaches the output, with its caption.

## What a run does

1. Scans the dataset folders with all their subfolders. Images are `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tif`, `.tiff`, `.gif` and `.avif`. A `.txt` file with the same name as an image is its caption and travels with it. Folders named `_embeddings`, `_classify`, `_duplicates` and `_unsure` are skipped, and so is the output folder when it lies inside a dataset folder. Any other folder name is scanned, including names that start with `_`.
2. Embeds every image that is not in the cache. The cache is `_embeddings\` inside each dataset folder and inside the samples folder. A second run embeds only new or changed files, so retraining after a change to the samples takes seconds.
3. Cross-validates the examples and prints a table with precision and recall per category and a confusion matrix. This is the first check of your samples: a category with low recall is not consistent, or it overlaps another.
4. Trains the classifier on all examples and predicts a category and a confidence for every dataset image.
5. Applies the gates. An image goes to `_unsure` when fewer than 3 of its 10 nearest examples belong to the predicted category (reason `disagree`), when its confidence is below `--min-confidence` (reason `confidence`), or when it could not be decoded (reason `undecodable`). A dataset image that is a byte-identical copy of an example takes the example's category with no gates (reason `example`).
6. Writes `plan.csv` and `report.txt` into `<output>\_classify\`, and with `--sheets` one contact sheet per output folder into `<output>\_classify\sheets\`. With `--dry-run` the run stops here.
7. Creates the category folders and `_unsure` in the output folder and copies the files, or moves them with `--move`.

The output folder must be empty before a real run, apart from its own `_classify` folder. A dry run has no such rule.

## Reading the report

The report starts with the cross-validation table. Then comes a histogram of the confidence of every dataset image, the counts per output folder and per reason, and the number of examples found in the dataset.

```
Confidence histogram of the dataset (probe probability of the chosen class):
  0.5-0.6    1059  #######
  0.6-0.7    1090  ########
  0.7-0.8    1200  ######## <- --min-confidence
  0.8-0.9    1516  ##########
  0.9-1.0    5782  ########################################
```

The confidence threshold is yours to choose from the histogram and the sheets. The default of 0.7 is a precision-first choice: on a dataset of historical photographs with a draft set of samples, about four fifths of the images between 0.7 and 0.85 were in the right folder, and about two thirds of those between 0.5 and 0.7. The misses sit on the borders between categories that are a matter of judgement, such as a person in a scene against the scene itself. More typical examples in the samples folder move images up the histogram; a lower threshold moves the border down.

The sheets show 48 random images and the 48 lowest-confidence images of each category folder, and for `_unsure` 48 random images per reason, with the predicted category on each tile. A category whose sheet looks wrong needs more or better examples. An image on an `_unsure` sheet that belongs to a category can be copied into the samples folder, and the next run places it and its kind.

## Two passes with `--retrain`

Hand-picked examples tend to be the clearest images of each category, so the first classifier draws narrow categories, and the typical images of the dataset fall between them. `--retrain` widens the categories with the dataset's own images:

1. The first pass runs at a strict confidence, 0.9 by default (`--retrain 0.95` sets another value). Only clear cases reach a category folder.
2. The first-pass placements that have at least half their nearest examples in their category join the examples as training images, at most 5 per hand example of the category, best confidence first. The cap keeps a large category from drowning your 40 examples in its own first-pass opinion.
3. The classifier is trained again on the enlarged set and scores the unsure images once more, now at `--min-confidence`. The ones that pass go to their category folder.

Every image is still placed once, at its final destination; the second pass happens in memory before anything is copied. The plan has a `pass` column, the report lists what was added per category and how many unsure images the second pass placed, and the sheets add one per category for the second-pass placements. Look at those sheets: the second pass is the classifier agreeing with itself, and its errors sit on the same borders as before. The cross-validation table stays a check of your hand examples only.

On the dataset of historical photographs with the draft samples, the second pass placed about half of the first-pass unsure images, and the unsure share went from 36 to 28 percent.

## Output

```
sorted\
  1_posters\
  2_postcards_art\
  ...
  _unsure\
  _classify\
    plan.csv
    report.txt
    moves.jsonl
    sheets\
```

The category folders are flat. An image keeps its name with its path inside the dataset folder in front, separators replaced by two underscores: `album17\000304.jpg` becomes `album17__000304.jpg`. Datasets with many subfolders usually repeat the same names in each, and this keeps every image traceable to its source. A caption is renamed together with its image. A name that still collides, which happens only with two dataset folders, gets `-2`, `-3` and so on before the extension.

Point a trainer at the category folders, not at the output folder: `_classify\sheets\` holds JPEG files that are not training images.

## Running again

The output is a function of the dataset, the samples and the threshold. To run again after a change to the samples, delete the category folders in the output (or the whole output folder) and run. The embeddings are cached, so the second run costs the time of copying the files. After a run with `--move`, run `--undo` first; it moves every file back to where it was.

`--undo` reads `_classify\moves.jsonl`, deletes the copies or moves the moved files back, in reverse order, and leaves a file alone when its size changed since the run. It then renames the log to `moves-undone-<date>.jsonl` and removes the folders it emptied.

## Options

| option | default | meaning |
|---|---|---|
| `--dataset PATH` | required | a dataset folder; repeat the option for several |
| `--samples PATH` | required | the samples folder |
| `-o PATH`, `--output PATH` | `<dataset>_classified` | the output folder; must be empty apart from `_classify` |
| `--dry-run` | off | write the plan, the report and the sheets, copy nothing |
| `--move` | off | move the dataset images instead of copying them |
| `--undo` | off | undo the last run from `<output>\_classify\moves.jsonl`; needs only `-o` or `--dataset` |
| `--min-confidence X` | 0.7 | images below this confidence go to `_unsure` |
| `--retrain [X]` | off | two passes: the first at confidence X (0.9 without a value), then retrain on the confident placements and score the unsure images again at `--min-confidence` |
| `--isolation-pct X` | 0 | also send the X percent of images that are farthest from all others to `_unsure` (reason `isolated`); off at 0 |
| `--sidecars EXT,EXT` | `.txt` | the extensions that travel with an image |
| `--sheets` | off | write the contact sheets |
| `--patches N` | 576 | the number of 16-pixel patches the encoder sees per image; more is slower and sharper |
| `--threads N` | 8 | image decoding workers; 1 runs without worker processes |
| `--batch N` | 64 | encoder batch size |
| `--reembed` | off | ignore the embedding caches |
| `--fetch-models` | off | download the encoder and exit; `install.bat` runs this |

The isolation gate marks images that have no look-alikes in the dataset. On a dataset of historical photographs it marked clean product-style photos of objects, not junk, which is why it is off by default. Deduplicate the dataset before using it: copies of one image make each other look well connected.

**Speed.** The encoder is the limit. On an RTX 5090 it embeds about 180 images per second at the default patch budget, so 40,000 images take under four minutes when the files are in the disk cache and about seven when they are not. Training, prediction and the plan take seconds; copying takes the time of copying. `--patches 256` is about twice as fast at the cost of resolution.

`run.bat` pauses at the end when it is started by double-click, or when the script fails. Set `NOPAUSE=1` to prevent this.

## Tests

```
venv\Scripts\python -m pip install pytest
venv\Scripts\python -m pytest tests
```

The tests replace the encoder with a fake that embeds the mean colour of an image, so they need no model and no GPU and run in a few seconds.
