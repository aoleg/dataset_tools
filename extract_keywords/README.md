# Extract keywords

`extract_keywords.py` moves the images whose captions contain one or more keywords, with their captions, out of a dataset into a new folder next to it. The subfolders of the dataset are kept in the new folder. Use it to take the images of one concept out of a large dataset for its own training, or to pull out every image whose caption mentions a text such as a watermark. `--undo` moves everything back.

## Install

Nothing to install. The tool needs Python 3.10 or newer. `run.bat` uses the shared `..\venv` when it exists, and the Python on PATH when it does not.

## Usage

```
run.bat <keyword> [<keyword> ...] --dataset <folder> [options]
run.bat <keyword> AND <keyword> [AND <keyword> ...] --dataset <folder>
run.bat <keyword> ... --dataset <folder> --undo
run.bat --undo <extracted folder>
```

Examples:

```
run.bat 1940s --dataset D:\photos --dry-run
run.bat 1940s --dataset D:\photos
run.bat 1940s --dataset D:\photos --undo
```

The first command shows how many images would move, without moving a file. The second moves every image whose caption has the word `1940s` into `D:\photos_1940s`. The third moves them back.

| option | what it does |
|---|---|
| `--dataset FOLDER` | the dataset, searched with all its subfolders |
| `--out FOLDER` | the new folder; by default `<dataset>_<first keyword>` next to the dataset |
| `--partial` | match inside words too: `car` also finds `cart` and `scarf` |
| `--case-sensitive` | the letter case must match |
| `--dry-run` | show what would move; move nothing |
| `--undo` | move the files of an extraction back (section [Undo](#undo)) |

## Keywords

- Keywords one after another are alternatives: a caption with any of them matches. `run.bat 1940s 1950s --dataset D:\photos` takes both decades.
- `AND` between keywords means that all of them must be in the caption. `run.bat soldier AND 1940s --dataset D:\photos` takes only captions with both words.
- `AND` binds closer than a space: `run.bat soldier AND 1940s tank --dataset D:\photos` takes captions with both `soldier` and `1940s`, and captions with `tank`. `OR` may be written between keywords and means the same as a space.
- Only `AND` and `OR` in capital letters are operators. To search for the word "and", write it in small letters.
- Put a phrase in quotes: `"street scene"`. The words of a phrase may be separated by any spaces or line breaks in the caption.
- A keyword matches whole words, in any letter case: `1940s` finds `1940s`, `1940S` and `mid-1940s`, but `1940` does not find `1940s`. Use `--partial` to match inside words, for example for a watermark that is part of a longer name.

The new folder is named after the first keyword: `run.bat "street scene" AND night --dataset D:\photos` makes `D:\photos_street_scene`. Spaces and characters that Windows does not allow in a folder name become `_`. When the folder exists already, the run stops and moves nothing: run `--undo` first, or give another folder with `--out`.

The run prints how many captions have each keyword, so you can see what each one adds before you change the query.

## What a run does

1. Searches the `.txt` caption of every image in the dataset and its subfolders. Images are `.jpg`, `.jpeg`, `.jpe`, `.jfif`, `.png`, `.webp`, `.bmp`, `.tif`, `.tiff`, `.gif`, `.avif`, `.heic` and `.heif`.
2. Skips the folders `_backup`, `_duplicates`, `_prep`, `_classify`, `_embeddings`, `_flatten_dataset`, `masks` and `faces`, at any depth. These are the output folders of the dataset tools. Links and junctions are not followed.
3. Writes the list of files to move to `<new folder>\_extract_keywords\manifest.json`, then moves them. Each file keeps its subfolder path: `D:\photos\2024\beach\a.jpg` goes to `D:\photos_1940s\2024\beach\a.jpg`.
4. Removes the subfolders of the dataset that are empty after the run. A subfolder that was empty before the run stays.

With an image go all the files of the same name in its folder: its caption, other images of the same name (`a.jpg` and `a.png`) and other sidecars (`a.npz`). A face mask made by `face_masks` (`<subfolder>\masks\<name>.png`) moves to the same place in the new folder.

A caption without an image is not searched and stays where it is; the run lists these captions.

When the new folder is on another drive, every file is copied and then deleted, which is slower than a move on the same drive.

## Undo

`--undo` reads the manifest and moves every file back to its old place in the dataset. Give it the same keywords and `--dataset` (or `--out`) as the run, or the extracted folder itself: `run.bat --undo D:\photos_1940s`. Changes made to the extracted files go back with them, for example edited captions.

- A file that you deleted from the extracted folder is listed and left out.
- A file that you added to the extracted folder stays there, and so does the folder.
- When the old place of a file is taken by a new file, that file stays in the extracted folder and the manifest is kept. Move the new file away and run `--undo` again.
- An interrupted run (Ctrl+C, a crash, a full disk) can be undone too: the manifest is written before the first move. An interrupted undo can be started again.

When every file is back and nothing else is in the extracted folder, the folder is deleted.

## Limits

- Captions are read as UTF-8. Text in another encoding may not match.
- Keep `_extract_keywords` in the extracted folder for as long as you may want `--undo`. Training tools ignore it, because it holds no images.
- A run of `deduplicate`, `remove_borders` or `flatten_dataset` on the dataset or the extracted folder between the extraction and its undo changes the paths that the manifest knows. Undo those runs first.
