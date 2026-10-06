# Flatten dataset

`flatten_dataset.py` moves the images and captions of all subfolders of a dataset into the dataset folder itself and gives every file a new, normalised name. The new name keeps the folder path in it, so you can still see where a file came from. An image and its caption always get the same new name, so they stay a pair. `--undo` puts every file back under its old name in its old subfolder.

## Install

Nothing to install. The tool needs Python 3.10 or newer. `run.bat` uses the shared `..\venv` when it exists, and the Python on PATH when it does not.

## Usage

```
run.bat <folder> [<folder> ...] [options]
run.bat <folder> --undo
```

Examples:

```
run.bat D:\photos --dry-run
run.bat D:\photos
run.bat D:\photos --undo
```

The first command shows the plan and writes the full list of renames to `D:\photos\_flatten_dataset\dry_run.csv`, without renaming a file. The second renames the files. The third puts everything back.

Each folder on the command line is flattened on its own. Do not give a folder together with one of its subfolders.

| option | what it does |
|---|---|
| `--dry-run` | show the plan and write `dry_run.csv`; rename nothing |
| `--undo` | put every file of the last run back |
| `--max-length N` | the longest new file name, extension included (default 80) |
| `--sidecars LIST` | the extensions of the files that go with an image, comma-separated (default `.txt`) |
| `--exclude NAME` | another folder name to leave alone, at any depth; may repeat |

## The new names

```
<folder>__<subfolder1>__<subfolder2>-datasetNNNNNN.<ext>
```

For example, `D:\photos\Лето 2024\пляж\IMG 1.JPG` and its caption `IMG 1.txt` become:

```
D:\photos\photos__Leto_2024__plyazh-dataset000042.jpg
D:\photos\photos__Leto_2024__plyazh-dataset000042.txt
```

A file directly in the folder gets `<folder>-datasetNNNNNN.<ext>`.

- The folder names are separated by `__`. In a folder name, Cyrillic is transliterated (`Жук` becomes `Zhuk`), Latin letters lose their accents (`Café` becomes `Cafe`), and a space or any other character that is not a Latin letter, a digit or `-` becomes `_`. A folder name never has `__` in it, so the separator stays clear. A folder name with nothing left (only Chinese characters, for example) becomes `folder`.
- `NNNNNN` is a running number with 6 digits (more when the dataset has a million images or more). The numbers follow the folder order: the files of the folder itself first, then each subfolder before its own subfolders. Folders and files are in natural order, so `IMG 2` comes before `IMG 10`. The number makes every name unique, also when two folders have the same cleaned name.
- The extension is in lower case (`.JPG` becomes `.jpg`).
- No new name is longer than `--max-length` characters. When a name is too long, the deepest subfolder name is cut first, down to 8 characters, then the one above it. When that is not enough, the deepest subfolders are left out of the name, and the folder name itself is cut last. The cut affects the name only: the manifest keeps the full old path.

## What a run does

1. Scans the folder with all its subfolders. Images are `.jpg`, `.jpeg`, `.jpe`, `.jfif`, `.png`, `.webp`, `.bmp`, `.tif`, `.tiff`, `.gif`, `.avif`, `.heic` and `.heif`. A `.txt` file with the same name as an image is its caption. Two images with the same name and different extensions (`a.jpg` and `a.png`) share their caption, so they get the same new name with their own extensions.
2. Skips the folders `_flatten_dataset`, `_backup`, `_duplicates`, `_prep`, `_classify`, `_embeddings`, `masks` and `faces`, at any depth. These are the output folders of the dataset tools. Every other folder is flattened, also one whose name starts with `_`. Links and junctions are not followed.
3. Plans every rename and checks the plan. When a file that stays in the folder already has a new name (for example a caption without an image), nothing is renamed.
4. Writes the plan to `<folder>\_flatten_dataset\manifest.json`, then renames the files.
5. Removes the subfolders that are empty after the run. A subfolder that was empty before the run stays.

These files stay where they are: a caption without an image, and every file that is not an image or a caption. The run lists them. A subfolder that holds such a file is not removed.

A face mask made by `face_masks` (`<subfolder>\masks\<name>.png`) moves with its image to `<folder>\masks\<new name>.png`.

A file that already has a new name of another file (a folder flattened before, for example) is first renamed to a temporary name, so no file is ever overwritten.

## Undo

`--undo` reads `manifest.json` and puts every file back under its old name in its old subfolder. It creates the subfolders that the run removed. When every file is back, it deletes `_flatten_dataset`, so the folder is as it was before the run.

An interrupted run (Ctrl+C, a crash, a full disk) can be undone too: the manifest is written before the first rename, and `--undo` finds out which renames were done. An interrupted undo can be started again.

When a file cannot go back (you deleted a renamed file, or a new file has its old name), the undo lists it and keeps the manifest. Delete `_flatten_dataset` to give up on these files.

A second run on a folder with a manifest is refused. Run `--undo` first.

## Limits

- The original file names are lost from the names themselves. Only the manifest keeps them, so keep `_flatten_dataset` for as long as you may want `--undo`.
- Run the tool before `remove_borders` or `deduplicate`, or when you no longer need their `--undo`. Their logs keep the old paths, and their undo cannot find the renamed files.
- Transliteration covers Russian, Ukrainian and Belarusian letters. Other scripts (Greek, Chinese, Arabic) become `_` and give the fallback name `folder`.
