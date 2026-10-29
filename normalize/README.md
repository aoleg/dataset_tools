# Normalize

`normalize.py` makes the images of a dataset small enough to train and brings them to one format, in place. Every image under the folder, in subfolders of any depth, whose area is larger than `<pixels>` x `<pixels>` is downscaled to that area. By default every image it writes is a `.jpg` of quality 97, and an image in another format (PNG, WebP, GIF, BMP, TIFF, AVIF, HEIC) is re-encoded as `.jpg` and its original deleted, even when it is not downscaled. A `.jpg` that needs no downscale is not touched.

In place there is no undo: the originals are overwritten or deleted. With `-o <folder>` the source folder is not changed: the normalized dataset is written to the output folder, with the captions of every image.

## Install

Run `install.bat`. It installs Pillow and pillow-heif (for HEIC photos) into the shared `..\venv`. No GPU is needed.

## Usage

```
run.bat <folder> <pixels> [options]
```

Examples:

```
run.bat D:\dataset 2048 --dry-run
run.bat D:\dataset 2048
run.bat D:\dataset 1536 --format png
run.bat D:\dataset 2048 --format original
run.bat D:\dataset 2048 -o D:\dataset_2048
```

The first command shows what would change without writing anything. The second asks `This will overwrite existing images. Continue? y/N`; only `y` and Enter continues, any other answer (Enter alone too) stops without changing a file. The last one writes a normalized copy to `D:\dataset_2048` and asks nothing. Folder and file names in Cyrillic or any other script are fine.

| option | what it does |
|---|---|
| `-o <folder>` | write into this folder instead of in place; see [Output folder](#output-folder) |
| `--format jpg [Q]` | the default, `jpg 97`: downscaled images and images in another format are saved as `.jpg` of quality Q (1 to 100) |
| `--format png` | the same with `.png`: downscaled images and images in another format are saved as `.png`, also a `.jpg` |
| `--format original [Q]` | only downscaled images are written, each in its own format under its own name (quality Q for JPEG, WebP, AVIF and HEIC, default 97); no image changes its format |
| `--dry-run` | show what would change; write nothing |
| `--yes`, `-y` | do not ask before overwriting |
| `--workers N` | images processed at once (default 8, or the CPU count when it is lower) |

## The size

`<pixels>` sets an area, not a side: 2048 means at most 2048 x 2048 = 4,194,304 pixels. An image keeps its aspect ratio and gets the largest size within that area, so at 2048 a 3:2 photo becomes 2508 x 1672 and a 4:3 photo 2364 x 1773. Images that are already smaller are never upscaled. This is how AI Toolkit and musubi-tuner count a resolution.

## Output folder

With `-o <folder>` every image goes to the output folder under its path relative to the source folder, in the same subfolders, by the same rules: a downscaled image or an image in another format is written there in the format that `--format` gives, and every other image (with the default, a JPEG within the size) is copied byte for byte. The tool does not ask, because nothing in the source folder changes.

The captions of an image are copied with it: every file in its folder with the same stem and an extension that is not an image extension (`.txt`, `.short`, `.json`, ...). Files that belong to no image are not copied, nor are the captions of an image that was skipped or could not be read.

The output folder may exist. Files in it with the same name are replaced, so a second run over the same folders brings the output up to date; other files in it stay. The output folder and the source folder must not be inside each other.

## What is written

- JPEG without chroma subsampling (4:4:4); Lanczos resampling.
- The EXIF orientation is applied to the pixels of a written image, so it looks the same as before in every viewer and trainer. The colour profile is kept, other metadata is dropped.
- Transparency becomes white in a JPEG. 16-bit grey becomes 8-bit in a JPEG and stays 16-bit in a PNG. CMYK becomes RGB.
- An animated GIF or WebP keeps its first frame; with `--format original` it is left as it is (copied as it is with `-o`).
- A file is written under a temporary name next to the image first and then moved over it, so an interrupted run never leaves a half-written image. Read-only files are replaced too.

The captions are not touched. The name of an image keeps its stem (`photo.png` becomes `photo.jpg`), so its `.txt` caption still belongs to it. AI Toolkit's latent caches of the changed images are out of date after a run; delete them before the next training.

## What is left alone

- Folders whose name starts with `.`, and `_controls` folders, which hold the face masks of `face_masks`; AI Toolkit fits a mask to the size of its image.
- The backup and output folders of the other dataset tools, which hold the `_dataset_tools.json` mark. The tool stops when the folder you give is one of them.
- An image that would be re-encoded to a name another image in its folder already has or would get, for example `a.png` next to `a.jpg`. It is listed as skipped (and not copied with `-o`); rename it and run again.
- Files that are not images, or that Pillow cannot read. They are listed as failed, and the tool exits with an error code.
