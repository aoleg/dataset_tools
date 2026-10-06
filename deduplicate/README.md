# Deduplicate

`dedup.py` finds copies of the same picture in one or more image folders and moves the worse copies out. It is for raw downloads, where the same picture often arrives several times: as other scans, at other sizes, with borders or watermarks, cropped, or as a shop mock-up on a wall or in a frame.

All folders that you give, with all their subfolders, form one pool. In each group of copies the best copy stays. Every other copy, with its `.txt` caption, moves to a `_duplicates` folder. If the copy that stays has no caption and a moved copy has one, that caption is also copied next to the copy that stays. Nothing is deleted, and `--undo` puts the files back.

A folder can also be a curated collection, given with `--sorted`. Then a better copy from a raw folder moves into the collection, under the name of the copy it replaces, and the collection's captions stay as they are. See [Sorted folders](#sorted-folders).

## Install

On Windows, run `install.bat`. It needs Python 3.10 or newer. It creates the shared `..\venv` folder when it is missing and installs Pillow, numpy, imagehash and OpenCV into it. It needs no GPU. The virtual environment is shared by all tools of this repository: it is the `venv` folder in the repository root, next to the tool folders, and each tool's `install.bat` installs only its own dependencies into it.

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
run.bat D:\data\downloads --sorted D:\data\collection --dry-run
```

You can also drop a folder onto `run.bat` in Explorer. There is no prompt: the tool scans, decides and moves. Use `--dry-run` first if you want to read the plan before anything moves.

| option | default | meaning |
|---|---|---|
| `--match` | `loose` | `loose`: hashes plus a feature check; finds other scans, crops, borders, frames, mock-ups and watermarks. `strict`: resized and recompressed copies, hashes only. `exact`: identical files only. |
| `--dry-run` | off | write `hashes.json` and `plan.json`, move nothing |
| `--undo` | off | move the files of the last run back to where they were, and remove the captions it copied (a copied caption that was edited since stays). Give the same folders and `--sorted` folders as the run. |
| `--sorted FOLDER` | none | a curated folder whose subfolders are categories; can be given more than once. See [Sorted folders](#sorted-folders). |
| `--sorted-copies` | `keep` | the same picture in several sorted folders: `keep` keeps every folder's copy and gives each the best copy; `one` keeps a single copy and moves the others out. See [Sorted folders](#sorted-folders). |
| `--promote-margin X` | 1.0 | the score margin a copy from an unsorted folder needs, in the same bucket, to replace a copy in a sorted folder; a larger bucket always qualifies |
| `--exclude NAME` | none | also skip folders with this name; can be given more than once |
| `--workers N` | CPU count - 1 | number of worker processes |
| `--gpu` | off | match the feature descriptors on the GPU with torch. torch is not part of this tool's requirements; it is in the shared `venv` once one of the GPU tools has been installed. Without torch or without a CUDA device the CPU is used, with a note. The results are the same either way. |
| `--review` | off | after the run, open the review tool full screen to check the groups by eye and change the kept copy. See [Review full screen](#review-full-screen-and-change-the-kept-copy). Not after `--dry-run`, because nothing has moved. |

Example of a run followed by the review:

```
run.bat D:\data\downloads --sorted D:\data\collection --review
```

The output folders of the dataset tools are never scanned, at any depth: `_duplicates` (this tool), `_prep` (k2prep), `_classify` and `_embeddings` (classify), `masks` and `faces` (face_masks, which writes near-identical images there), and `_backup` (remove_borders, which keeps the originals of cropped images there). Every other folder is scanned whatever its name, so a collection folder called `_1_posters` is fine as a root or below one. `--exclude` adds names to the list.

`run.bat` pauses at the end when it is started by double-click or drag-and-drop, or when the script fails. Set `NOPAUSE=1` to prevent this.

## Deduplicate curated collections

This is the workflow for a curated collection that receives new downloads. The rules are in [Sorted folders](#sorted-folders).

1. **Install once.** Run `install.bat` in this folder. It needs Python 3.10 or newer and creates the shared `..\venv` folder when it is missing. Nothing else to configure.

2. **Dry run first.** Give the raw download folders as plain arguments and each curated collection with `--sorted`:

   ```
   run.bat D:\data\downloads1 D:\data\downloads2 --sorted D:\data\collection --dry-run
   ```

   Nothing moves. The summary at the end tells you how many copies would move, how many raw copies would move into the collection, how many slots would be synced, and how many groups are left for review.

3. **Read the plan.** `plan.json` in the first folder's `_duplicates` opens with a `summary` block. Below it, each group shows the kept copy, every destination, and for collections the `slots`, `promote` and `sync` entries. Groups with a `review` field are the ones the tool refuses to decide, usually because a border, a frame or a footer differs. Those files stay where they are for you to look at. Two things are worth a glance before a real run: the `promote` entries, because they replace files in your collection, and raw sources that carry printed catalogue footers, because a footer passes on a hash match.

4. **Run.** The same command without `--dry-run`. There is no prompt. Nothing is deleted. Every moved file goes to the `_duplicates` folder of its own root, under its original relative path, and every action is logged in `moves.jsonl`. Add `--review` to open the groups full screen right after the run and correct the tool's choices by eye, or run `compare.bat` later for sheets and a text list; see [Check the result by eye](#check-the-result-by-eye).

5. **Undo if needed.** The same folders, the same `--sorted` folders, plus `--undo`:

   ```
   run.bat D:\data\downloads1 D:\data\downloads2 --sorted D:\data\collection --undo
   ```

   It puts everything back, including the copies it placed into the collection, and skips any copied caption you have edited since.

For larger datasets:

- The feature check is the slow step and grows with the number of images. About 2,000 images take around a minute cold. A second run reuses the three cache files in `_duplicates` and finishes in seconds, so run the dry run and the real run back to back.
- The same picture in several collection folders keeps every folder's copy and gives each the best one. If you want a single copy instead, add `--sorted-copies one`.
- The promotion margin defaults to 1.0 score points in the same bucket. Raise it with `--promote-margin` if too many near-equal copies move in, or lower it if you want every improvement.
- `_duplicates` folders are never scanned, so they are safe to leave in place between runs.
- Delete the `_duplicates` folders only when you no longer want to undo.

## Check the result by eye

`compare.bat` shows which files were judged copies of which, from the plan of the last run:

```
compare.bat D:\data\downloads --skip-exact
```

You can also drop a folder onto `compare.bat` in Explorer.

It reads `D:\data\downloads\_duplicates\plan.json` (any folder of the run will do, the plan is the same in each) and writes `_duplicates\compare\pairs.txt`, one block per group with the kept file, every moved copy, where that copy is now, the match kind and the sizes, and one JPG sheet per group with the kept copy and the moved copies side by side, labelled. Groups that matched by features come first, because a border, a crop, a frame or a footer differed there and they are the ones to look at. Nothing is moved or changed.

| option | default | meaning |
|---|---|---|
| `--skip-exact` | off | leave out groups whose moved copies are byte-identical to the kept one; there is nothing to see there |
| `--only hash` or `--only features` | all | only groups with at least one moved copy of this match kind; `features` is the kind to check first |
| `--height N` | 360 | tile height of the sheets in pixels |
| `--out DIR` | `_duplicates\compare` | write the list and the sheets somewhere else |
| `--review` | off | then open the review tool on the same groups |

The sheets are for a quick look in Explorer. To judge quality, and to correct a choice, use the review tool below.

### Review full screen and change the kept copy

`review.bat` opens the groups of the last run full screen, one group per screen, drawn from the files themselves. It does not need the sheets. There are three ways to open it:

```
review.bat D:\data\downloads
run.bat D:\data\downloads --sorted D:\data\collection --review
compare.bat D:\data\downloads --skip-exact --review
```

The first opens the review of the last run in that folder at any time. The second runs the tool and opens the review right after it. The third builds the sheets and the list first, then opens the review.

Each screen shows the kept copy with a green frame and every moved copy, labelled with name, size, bucket, score, match kind and folder. Click a copy to choose it as the kept one; the frame turns yellow. When you move on to another group, the chosen copy returns to its place and the old kept copy moves to `_duplicates`, with its caption, like any other copy. Press `B` after the click to keep both instead: the copy returns and the kept copy stays. Every move is appended to `moves.jsonl`, so `--undo` of the run still puts everything back, including your changes.

| key | action |
|---|---|
| Right, Space | next group, applies your choice |
| Left | previous group, applies your choice |
| Home, End | first and last group |
| click | choose a copy as the kept one; click the kept copy to clear the choice |
| B | keep both: the chosen copy returns, the kept copy stays |
| U | clear the choice on this group |
| 1 to 9 | show one copy alone full screen; the same key or Esc returns |
| H | hide and show the help at the top |
| Esc | exit; the choice on the current group is discarded |

The help is shown at the top until you hide it. Your position and your decisions are saved in `_duplicates\compare\review.json` of the first folder of the run, and the next launch resumes at the group you left. A group with a promotion or a slot sync (sorted folders) is shown but locked: undo the run to change it. A file that is no longer where the plan says is shown as a grey tile and cannot be chosen.

## Where the files go

Each copy that moves keeps its path relative to the folder that you gave. For example, with `run.bat D:\data`, the file `D:\data\set2\a.jpg` moves to `D:\data\_duplicates\set2\a.jpg`, and `D:\data\set2\a.txt` moves with it. If you give several folders, each one gets its own `_duplicates` folder for its own files, but copies are found across all of them. The one exception is a copy that moves into a sorted folder, described in [Sorted folders](#sorted-folders).

A name that is already taken in `_duplicates` gets ` (1)`, ` (2)` and so on. A caption stays in place if another image with the same name (for example `a.png` next to `a.jpg`) stays in the folder.

If the copy that stays has no caption, the tool copies the caption of a moved copy next to it and renames it to match: for `b.jpg` that stays and `a.jpg` that moves, `a.txt` is copied to `b.txt`. If several moved copies have captions, the caption of the best of them is used (same ranking as below, then the oldest file). An existing caption is never overwritten. The moved copy keeps its own caption in `_duplicates`.

Each `_duplicates` folder also holds:

| file | content |
|---|---|
| `hashes.json` | hashes and scores of every image of that folder; a second run reads them and processes only new or changed files. A file that a run moved or copied into a sorted folder is entered there too, so it is not hashed again. |
| `verified.json` | feature-check results, keyed by the content of both images |
| `plan.json` | a `summary` with the counts of the run, then every group: the kept copy, the moved copies with their destinations, the reason, the tier and score of each, the match details, and with sorted folders the role report, the `slots`, the `promote` and `sync` placements, or `review` and `members` for a group nothing moves in |
| `moves.jsonl` | one line per moved file, per copied caption (`action: copy`, with a hash), per promoted file (`action: promote`) and per synced slot (`action: copy`); `--undo` reads it |

## Sorted folders

A sorted folder is a curated tree: its subfolders are the categories, and its files and captions are the ones you want to keep. Give it with `--sorted`. The folders without the option are unsorted downloads. All folders still form one pool.

When a copy in an unsorted folder is better than the copy in a sorted folder, the unsorted copy moves into the sorted copy's place: the same folder, the same name, its own extension. The worse sorted copy goes to the sorted folder's `_duplicates`, like any other copy. Better means a larger bucket, or a score at least `--promote-margin` higher in the same bucket, and the two copies must match by hash, not only by features. Equal copies keep the sorted one. The rest of the group, the other unsorted copies, moves to `_duplicates` as usual.

A caption in a sorted folder is never moved, overwritten or edited:

- The caption of the slot stays, and the new file takes it over by name. The moved-out copy gets a copy of it in `_duplicates`.
- If the new file needs another name, because an image of the same stem stays (for example `1.png` next to the moved-out `1.jpg`), it becomes `1 (1).png`, and `1 (1).txt` is a copy of `1.txt`.
- If the slot has no caption and the new file has one, that caption moves with it and takes the slot's name.
- If both have one, the sorted caption wins. The unsorted caption moves to the unsorted folder's `_duplicates`.

Nothing moves in a group when the unsorted copy and the sorted copy match by features only, so that a border, a crop, a frame or a footer differs. A higher score does not mean a better picture then: on real folders, every such case was a photo of a print or a catalogue page replacing a clean scan. The plan lists such groups with a `review` field, and the summary counts them.

### The same picture in several sorted folders

A sorted tree can hold one picture under several categories, and the copies can differ in quality. The default, `--sorted-copies keep`, keeps every folder's copy and gives each one the best copy of the group. In each folder one slot survives, the best copy in that folder, and a second copy in the same folder moves out as usual. Every surviving slot whose image matches the best copy by hash gets the best copy: the slot's own image moves to `_duplicates`, the best copy is placed under the slot's name, and the slot's caption stays, exactly as for a promotion. A slot without a caption gets a copy of the best copy's caption. A slot whose image is byte-identical to the best copy is left as it is, and so is a slot whose image matches by features only; the plan says so under `slots`. With `--sorted-copies one` a single slot survives across all sorted folders, the one of the best copy, and the other sorted copies move to `_duplicates` together with their captions.

The dry run shows the final name of every file, also the ` (N)` names. `--undo` needs the same `--sorted` arguments as the run.

## How it decides

### Which images are copies

1. **Identical files** (same SHA-256) are always copies.
2. **Hashes.** Each image gets a pHash and a dHash (64 bits each), computed on the image after EXIF rotation. Two images are copies when the pHash distance is 14 or less and the dHash distance 16 or less, or when the pHash distance is 16 or less and the dHash distance 8 or less.
3. **Feature check** (`loose` only). Hashes see the whole image, so a border, a caption strip or a wall around the picture moves them far apart. The tool therefore also takes, for each image, its 30 nearest images within a pHash and dHash distance of 22, and compares them with ORB keypoints. A pair is a copy when at least 40 keypoint matches agree on one geometric transform (RANSAC homography) and cover at least 36% of both images.
4. **Centre check.** The transform must also make the centres agree: one image is warped onto the other, and the centres must correlate at 0.5 or more. This rejects shop mock-ups that put different pictures into the same frame or onto the same wall: there, the matches lie on the template, and the centres show different pictures. The check also applies to hash matches. A hash match for which the feature check finds no shared geometry at all is rejected as well, because two pictures that hash alike but share no keypoints are two different pictures. On a set of 2,000 downloaded posters all nine such pairs were different images, for example different state emblems printed on one card template. The price is that in `loose` mode a copy in which fewer than 10 keypoints can be found matches only when the files are identical. The run reports how many hash matches the feature check vetoed, and why.
5. **Never matched:** a colour image and a black-and-white, sepia or toned copy of it count as different images. Near-uniform images (blank pages, plain backgrounds) and animated files match only when the files are identical.
6. **Groups.** All images connected through matches form one group, also when two of them match only through a third. One copy stays, all others move.

The thresholds were calibrated on 690 downloaded posters and photos, with every borderline pair checked by eye. On that set, true copies had a centre correlation of 0.74 or more, and template false matches 0.27 or less. Synthetic watermarks (corner labels, centre text, diagonal stock-site text) all still matched; a mirrored copy never matched.

### Which copy stays

Quality is measured the k2prep way: each copy is cropped and resized to the bucket it would get in training (512, 768 or 1024, musubi-tuner buckets, up to 1.15x upscale allowed), and the result is scored for detail and for surviving JPEG block artifacts. A copy too small for 512 is scored at its own size. The code is a copy of k2prep's two-pass scoring.

1. The copy that reaches the largest bucket stays.
2. In the same bucket, the higher score stays.
3. Exception: a copy from a smaller bucket stays if it scores 3 or more points higher than the larger copy. A badly compressed large copy then loses to a clean smaller one.
4. Scores less than 0.1 apart count as equal. Then a copy in a sorted folder stays; then the copy with a `.txt` caption; if both or neither have one, the oldest file (modification time) stays; then the first folder on the command line, then the first path in alphabetical order.
5. With sorted folders, a copy from an unsorted folder replaces a copy in a sorted folder only when it reaches a larger bucket or scores `--promote-margin` higher in the same bucket, and only on a hash match. Otherwise the sorted copy stays although the unsorted copy scored higher. The margin was set on a real collection: below 1.0, the same-bucket differences were resolution steps inside the 1024 bucket, which the bucketed score cannot see.

A caption does not make a copy win: quality decides. The caption is copied instead, as described in [Where the files go](#where-the-files-go).

## Speed

On 11,561 photos with 67,108 candidate pairs, 23 worker processes: hashing 15 seconds, feature extraction 23 seconds, pair check 161 seconds on the CPU, scoring 1 second. With `--gpu` the pair check takes 12 seconds: 5 seconds of descriptor matching on the GPU and 7 seconds of RANSAC and centre checks in the workers, with results identical to the CPU path on every pair. Without a GPU the pair check is the slow part: descriptor matching costs about 16 ms per pair on one core, so the time grows with the number of candidate pairs, at most 30 per image. Features are extracted once per image and shared with the workers through memory-mapped files, and the centre check runs only for pairs that can still become a match. A second run reads the hashes, the feature checks and the scores from the JSON files and only processes new or changed images.

## Limits

- Two different works that share the same central photo, for example two photomontage posters built on one photograph, can match.
- A mirrored copy does not match. A copy cropped by 10% or more on each side is not caught by the hashes. It matches only through the feature check, and only if it is still among the 30 nearest hash candidates of the other copy.
- A watermarked copy matches the clean copy, but the score does not see the watermark. If the watermarked copy is larger, it stays.
- The tool reads image content only. It does not compare captions.
- A photo of a print, or a catalogue page with a printed footer, is told apart from a clean scan only when the hashes disagree and the feature check has to decide. When the hashes agree, the score decides, and the footer comes along into a sorted folder.
- Moving files breaks references to them, for example in the `metadata.csv` of the image search downloader. A promotion into a sorted folder keeps the sorted name, so references into the sorted folder keep working.

## Tests

`tests/fixture_test.py` builds a small sorted and unsorted tree from synthetic pictures, runs the tool with `--sorted`, checks every case of the sorted-folder rules (rename on collision, each caption case, the margin, a feature match left for review, slot sync under both policies), runs again to confirm that nothing moves, and undoes, comparing both trees byte for byte with the originals.

```
..\venv\Scripts\python tests\fixture_test.py
```

It writes to `tests\_fixture` unless another folder is given, and prints `ALL PASS` or the failed checks.
