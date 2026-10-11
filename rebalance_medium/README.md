# Rebalance medium

`rebalance_medium.py` teaches a model what a prompt without a medium means. When every caption of a dataset opens with its medium (`A color photograph from 1975 of ...`, `A black-and-white photograph from 1962 of ...`), a prompt that names no medium is out of distribution, and the model falls back to the medium that is cheaper to predict, usually black-and-white. The tool moves a share of the colour and black-and-white images into a sibling dataset folder and removes the medium word from their captions there. By default 30 % of the captions lose the medium, and 60 % of those are colour, whatever the sizes of the two classes; the colour majority counters the bias towards black-and-white.

It is made for a training run in [Ostris AI Toolkit](https://github.com/ostris/ai-toolkit) that is already in progress. The latent cache of each moved image moves with it and stays valid. The original caption files and the text-embedding caches, which are keyed by the caption text, are kept in a backup folder outside the dataset and come back with `--undo`.

## Install

Nothing to install. The tool needs Python 3.10 or newer. `run.bat` uses the shared `..\venv` when it exists, and the Python on PATH when it does not.

## Usage

```
run.bat <dataset folder> [options]
run.bat <dataset folder> --apply [options]
run.bat <dataset folder> --undo
```

Examples:

```
run.bat D:\train\1_main-dataset_1024
run.bat D:\train\1_main-dataset_1024 --apply
run.bat D:\train\1_main-dataset_1024 --undo
```

The first command is a dry run: it shows how many images would move, writes the plan to the manifest and moves nothing. The second moves the images into `D:\train\1_main-dataset_media-rebalanced_1024`. The third moves everything back.

| option | what it does |
|---|---|
| `--drop-share PCT` | percent of the colour and black-and-white captions that lose the medium, default 30 |
| `--color-share PCT` | percent of colour images among them, default 60 |
| `--max-class-pct PCT` | at most this percent of either class moves, default 50; the drop share is lowered to keep it |
| `--color PCT` | instead of the shares: percent of the colour images to move (40 when only `--bw` is given) |
| `--bw PCT` | instead of the shares: percent of the black-and-white images to move (20 when only `--color` is given) |
| `--seed N` | the random seed of the draw, default 0 |
| `--dest NAME` | the destination folder, next to the dataset folder (section [Folders](#folders)) |
| `--manifest PATH` | the manifest, default `<dataset>_rebalance.json` next to the dataset folder |
| `--strip-russian` | also remove the Russian medium words from the moved captions (section [Captions](#captions)) |
| `--apply` | move the images; without it the run is a dry run |
| `--undo` | move everything of an applied run back (section [Undo](#undo)) |

## Captions

The medium must be at the start of the caption: the article `A` or `An`, up to three words, the medium word, up to three more words, then `photograph` or `photo`. Letter case does not matter.

| caption starts with | class | moved caption starts with |
|---|---|---|
| `A color photograph from 1975` | color | `A photograph from 1975` |
| `A colour photograph` | color | `A photograph` |
| `A high-angle color photograph` | color | `A high-angle photograph` |
| `A color studio photograph` | color | `A studio photograph` |
| `A color aerial photograph` | color | `An aerial photograph` |
| `A black-and-white photograph` | bw | `A photograph` |
| `A black and white press photograph` | bw | `A press photograph` |

Only the medium word and the space before it are removed. `A` becomes `An` when the next word now starts with a vowel. The rest of the caption, its line breaks included, does not change.

A caption is ambiguous, and its image never moves, when:

- it does not start with the medium (`A photograph of ...`, `Color photograph of ...`, `A street. A color photograph ...`);
- the words around the medium name the other medium or a tinted or toned print (`A hand-colored black-and-white photograph`, `A sepia photograph`, `A monochrome photograph`);
- it is not UTF-8.

The run lists the ambiguous captions, so you can fix them or leave them.

A caption can end with a Russian translation after `., `, for example `... Low-angle shot., Цветная фотография улицы.` The run counts the captions whose Russian part names the medium (forms of `цветной`, `чёрно-белый` and `черно-белый`). With `--strip-russian` these words are also removed from the moved captions: `Цветная фотография улицы.` becomes `Фотография улицы.` Without it the Russian part is not changed.

## Which images move

What the model learns for a prompt without a medium comes from the captions that lose it: the "medium not stated" set. Its colour share is what counts, so the tool sets that set directly:

- `--drop-share`: the percent of all colour and black-and-white captions that lose the medium, default 30.
- `--color-share`: the percent of colour images in that set, default 60. Above 50 counters the bias towards black-and-white.

From these, with C colour and B black-and-white images: moved = drop share × (C + B), colour moved = colour share × moved, black-and-white moved = the rest.

| colour | black-and-white | moved | colour moved | black-and-white moved |
|---|---|---|---|---|
| 5,000 | 5,000 | 3,000 | 1,800 (36 %) | 1,200 (24 %) |
| 3,000 | 7,000 | 2,500 (cap) | 1,500 (50 %) | 1,000 (14 %) |
| 3,000 | 7,000, `--max-class-pct 60` | 3,000 | 1,800 (60 %) | 1,200 (17 %) |

The images that keep their medium are what make "color photograph" and "black-and-white photograph" strong prompt words. So `--max-class-pct` (default 50) limits how much of either class moves. When the shares would take more, the drop share is lowered until both classes are within the limit, and the colour share stays: in the second row 30 % would take 60 % of the colour images, so the drop share becomes 25 %.

`--color` and `--bw` give the percent of each class instead, as the first version of the tool did. They cannot be given with the share options.

The summary of every run shows the selection both ways: the percent of each class that moves, the colour share of the "medium not stated" set, and the colour share of the captions that keep their medium. It also says when the limit lowered the drop share. Ambiguous captions without a known medium are already "not stated"; their number is shown next to the set.

The images of each class are drawn from the whole dataset at random. The draw is the same on every computer for the same dataset and `--seed`. A subfolder can lose more or fewer of its images than the share. With repeats: the shares hold in training when the destination gets the same repeat setting as the dataset.

The dataset is searched with all its subfolders. A subfolder is used when it holds at least one image with its caption: `<name>.jpg`, `.jpeg`, `.png` or `.webp` with `<name>.txt` next to it. Images without a caption in such a subfolder are counted and never moved. The folders `_latent_cache`, `_t_e_cache` and `_backup`, and every folder whose name starts with `.`, are not searched. Links and junctions are not followed.

Two images with the same name and different extensions (`a.jpg` and `a.png`) share one caption and one cache name, so neither of them moves.

## Folders

All folders are next to the dataset folder:

| folder or file | what it holds |
|---|---|
| `1_main-dataset_1024` | the dataset; it keeps the images that do not move, with their captions unchanged |
| `1_main-dataset_media-rebalanced_1024` | the moved images with their new captions and their latent caches, in the same subfolders |
| `1_main-dataset_1024_rebalance_backup` | the original caption files and the text-embedding caches of the moved images, in the same subfolders |
| `1_main-dataset_1024_rebalance.json` | the manifest: every image, its class, its old and new caption, its caches and its state |

The destination name is the dataset name with `_media-rebalanced` before its last `_<number>`, or after the name when it does not end in a number: `photos` becomes `photos_media-rebalanced`. `--dest NAME` gives another name.

## What --apply does

1. Searches the dataset and draws the images, as a dry run does, and writes the plan to the manifest before the first move.
2. For each drawn image:
   - moves the image to the same subfolder of the destination;
   - writes the new caption next to it, then moves the original caption file from the dataset to the same subfolder of the backup folder;
   - moves its files in `_latent_cache` to `_latent_cache` in the destination. AI Toolkit names a latent cache file after the image file name and its crop, never its folder, so the file stays valid;
   - moves its files in `_t_e_cache` to the backup folder. AI Toolkit names a text-embedding cache file after the caption text, so these files do not match the new caption. The trainer makes new ones for the destination.
3. Writes the manifest after every 100 images and at the end.

Every original file of a moved image is kept in one of the folders: the image and its latent caches in the destination, its original caption and its text-embedding caches in the backup folder. No file is deleted.

A run made before the original captions were backed up as files keeps them in the manifest only. Give the same `--apply` command again: it writes the missing caption files to the backup folder from the manifest and moves nothing else.

A run stops before it moves anything when the destination or the backup folder exists. An interrupted run (Ctrl+C, a crash, a full disk) goes on from where it stopped when you give the same command again; images that moved are not moved twice. Without selection options, a resumed run takes the selection of the manifest, also of a run made by the first version (`--color 40 --bw 20`). When the manifest records moved images, a run with other settings stops, and so does a dry run, which would write over the manifest; run `--undo` first, or give another `--manifest`. A run also stops when the manifest lists moved images that are in the dataset again.

The tool moves files and never copies them, so the destination and the backup folder should be on the same drive as the dataset. Across drives every file is copied and then deleted, which is slower.

## Undo

`--undo` reads the manifest and, for each moved image:

- moves the image back, and its original caption file from the backup folder. When that file is missing, the caption is written from the manifest, byte for byte;
- moves its latent caches back from the destination, also the ones that the trainer made there;
- moves its text-embedding caches back from the backup folder, so the trainer finds them again and does not encode the original captions;
- deletes the text-embedding caches that the trainer made in the destination for the new caption.

Then it deletes the empty subfolders of the destination and the backup folder, and the folders themselves when they are empty. AI Toolkit's size cache `.aitk_size.json` in the destination is deleted when it is the last file there. Files that you added are listed and stay.

When a caption in the destination was edited after the move, the original caption still goes back to the dataset, and the edited one stays in the destination and is listed. When a backed-up caption was edited, the edited one goes back and is listed. An interrupted undo can be started again.

## Restore by hand

The folders hold every file, so the dataset can be restored without the tool and without the manifest:

1. Move every file of the destination except the `.txt` captions (the images and the `_latent_cache` folders) back into the dataset folder, keeping the subfolders. Skip `.aitk_size.json`.
2. Copy or move the contents of the backup folder into the dataset folder, keeping the subfolders. This puts back the original captions and the `_t_e_cache` files.
3. Delete the destination folder, the backup folder and the manifest.

## Training with the result

1. Stop the training job before `--apply`. The trainer keeps the file list in memory.
2. Run `--apply`.
3. Add the destination folder to the job as a new dataset, with the same resolution and repeat settings as the dataset folder. With other settings the latent caches do not match and are made again.
4. Resume the job. The trainer finds the latent caches of the moved images in place and encodes the new captions once.

## Limits

- Captions are read as UTF-8. A caption in another encoding is ambiguous and does not move.
- Keep the manifest and the backup folder for as long as you may want `--undo`. Without the manifest, restore by hand.
- A run of another tool that renames or moves files in the dataset or the destination between `--apply` and `--undo` changes the paths that the manifest knows. Undo that run first.
