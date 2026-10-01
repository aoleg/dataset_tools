# Deduplicate

`dedup.py` finds copies of the same picture in one or more image folders and moves the worse copies out. It is for raw downloads, where the same picture often arrives several times: as other scans, at other sizes, with borders or watermarks, cropped, or as a shop mock-up on a wall or in a frame.

All folders that you give, with all their subfolders, form one pool. In each group of copies the best copy stays. Every other copy, with its `.txt` caption, moves to a `_duplicates` folder. Nothing is deleted, and `--undo` puts the files back.

## Install

On Windows, run `install.bat`. It needs Python 3.10 or newer. It creates a `venv` folder next to the script and installs Pillow, numpy, imagehash and OpenCV (`opencv-python-headless`). It needs no GPU.

Without `install.bat`:

```
pip install -r requirements.txt
```

## Usage

```
run.bat <folder> [<folder> ...] [options]
```

Examples:

```
run.bat D:\data\downloads
run.bat D:\data\posters D:\data\photos --dry-run
run.bat D:\data\posters D:\data\photos --undo
```

You can also drop a folder onto `run.bat` in Explorer. There is no prompt: the tool scans, decides and moves. Use `--dry-run` first if you want to read the plan before anything moves.

| option | default | meaning |
|---|---|---|
| `--match` | `loose` | `loose`: hashes plus a feature check; finds other scans, crops, borders, frames, mock-ups and watermarks. `strict`: resized and recompressed copies, hashes only. `exact`: identical files only. |
| `--dry-run` | off | write `hashes.json` and `plan.json`, move nothing |
| `--undo` | off | move the files of the last run back to where they were |
| `--exclude NAME` | none | also skip folders with this name; can be given more than once |
| `--workers N` | CPU count - 1 | number of worker processes |

Folders whose names start with `_` (for example `_duplicates`, or k2prep's `_prep`) are always skipped. So are folders named `masks` and `faces`, because the face mask tool writes near-identical images there.

`run.bat` pauses at the end when it is started by double-click or drag-and-drop, or when the script fails. Set `NOPAUSE=1` to prevent this.

## Where the files go

Each copy that moves keeps its path relative to the folder that you gave. For example, with `run.bat D:\data`, the file `D:\data\set2\a.jpg` moves to `D:\data\_duplicates\set2\a.jpg`, and `D:\data\set2\a.txt` moves with it. If you give several folders, each one gets its own `_duplicates` folder for its own files, but copies are found across all of them.

A name that is already taken in `_duplicates` gets ` (1)`, ` (2)` and so on. A caption stays in place if another image with the same name (for example `a.png` next to `a.jpg`) stays in the folder.

Each `_duplicates` folder also holds:

| file | content |
|---|---|
| `hashes.json` | hashes and scores of every image of that folder; a second run reads them and processes only new or changed files |
| `verified.json` | feature-check results, keyed by the content of both images |
| `plan.json` | every group: the kept copy, the moved copies, the reason, the tier and score of each, and the match details |
| `moves.jsonl` | one line per moved file; `--undo` reads it |

## How it decides

### Which images are copies

1. **Identical files** (same SHA-256) are always copies.
2. **Hashes.** Each image gets a pHash and a dHash (64 bits each), computed on the image after EXIF rotation. Two images are copies when the pHash distance is 14 or less and the dHash distance 16 or less, or when the pHash distance is 16 or less and the dHash distance 8 or less.
3. **Feature check** (`loose` only). Hashes see the whole image, so a border, a caption strip or a wall around the picture moves them far apart. The tool therefore also takes, for each image, its 30 nearest images within a pHash and dHash distance of 22, and compares them with ORB keypoints. A pair is a copy when at least 40 keypoint matches agree on one geometric transform (RANSAC homography) and cover at least 36% of both images.
4. **Centre check.** The transform must also make the centres agree: one image is warped onto the other, and the centres must correlate at 0.5 or more. This rejects shop mock-ups that put different pictures into the same frame or onto the same wall: there, the matches lie on the template, and the centres show different pictures. The check also applies to hash matches.
5. **Never matched:** a colour image and a black-and-white, sepia or toned copy of it count as different images. Near-uniform images (blank pages, plain backgrounds) and animated files match only when the files are identical.
6. **Groups.** All images connected through matches form one group, also when two of them match only through a third. One copy stays, all others move.

The thresholds were calibrated on 690 downloaded posters and photos, with every borderline pair checked by eye. On that set, true copies had a centre correlation of 0.74 or more, and template false matches 0.27 or less. Synthetic watermarks (corner labels, centre text, diagonal stock-site text) all still matched; a mirrored copy never matched.

### Which copy stays

Quality is measured the k2prep way: each copy is cropped and resized to the bucket it would get in training (512, 768 or 1024, musubi-tuner buckets, up to 1.15x upscale allowed), and the result is scored for detail and for surviving JPEG block artifacts. A copy too small for 512 is scored at its own size. The code is a copy of k2prep's two-pass scoring.

1. The copy that reaches the largest bucket stays.
2. In the same bucket, the higher score stays.
3. Exception: a copy from a smaller bucket stays if it scores 3 or more points higher than the larger copy. A badly compressed large copy then loses to a clean smaller one.
4. Scores less than 0.1 apart count as equal. Then the copy with a `.txt` caption stays; if both or neither have one, the oldest file (modification time) stays; then the first folder on the command line, then the first path in alphabetical order.

At the end, the tool lists every moved copy that takes a caption with it while the kept copy has none, so that you can copy the caption over if you want it.

## Speed

On 690 images: hashing 3 seconds, feature check of 1,500 candidate pairs 15 seconds, scoring 2 seconds, with 7 worker processes. The feature check is the slow part, and it grows with the number of images, not with the number of pairs, because each image has at most 30 candidates. A second run reads the hashes, the feature checks and the scores from the JSON files and only processes new or changed images.

## Limits

- Two different works that share the same central photo, for example two photomontage posters built on one photograph, can match.
- A mirrored copy does not match. A copy cropped by 10% or more on each side is not caught by the hashes. It matches only through the feature check, and only if it is still among the 30 nearest hash candidates of the other copy.
- A watermarked copy matches the clean copy, but the score does not see the watermark. If the watermarked copy is larger, it stays.
- The tool reads image content only. It does not compare captions.
- Moving files breaks references to them, for example in the `metadata.csv` of the image search downloader.
