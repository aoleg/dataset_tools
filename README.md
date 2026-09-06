# k2prep

k2prep takes a messy folder of mixed photographs and produces a small, clean,
bucket-tight training set for musubi-tuner's Krea 2 trainer: every accepted image
is cropped and resized onto one of 7 aspect ratios across 3 resolution tiers, and
buckets too small to form a real batch are then consolidated into their nearest
healthy neighbour — so instead of the 15+ buckets an ordinary photo folder
scatters across, the trainer typically sees two or three per tier.

The output is a **filtered subset**, not a transformation of the whole folder.
Images that fail the quality threshold or are too small are skipped entirely, and
the source folder is never written to, moved within, or deleted from — the report
is the only record of a rejection. (The single exception is `--sort --move`,
which exists to move originals and says so.)

It can also just [sort a folder by quality](#sorting-a-folder-by-quality) and
build nothing.

[`cleanup.bat`](#cleanupbat-moving-undersized-images-out) ships alongside it and does
the one thing k2prep will not: move undersized images out of the source folder.

## Install and run

```bash
install.bat
```

Then look before you leap:

```bash
run.bat "L:\train\photos" --report
```

and when the report looks right:

```bash
run.bat "L:\train\photos" --threshold 6
```

That is the whole workflow. One run scans, assigns buckets, consolidates the
undersized ones, renders and scores every image, writes the ones that pass along
with their captions, and emits `_prep/dataset.toml` ready to hand to
musubi-tuner.

**Run `--report` first.** It is a dry run: it analyses everything and writes a
report but no images. Because rejections leave no artifact on disk — nothing is
copied, moved or marked — the report is the only place a rejected file is ever
named. Recovering from too high a threshold means lowering it and re-running,
which is cheap (already-written images are skipped), but you cannot recover the
list of what was dropped after the fact if you never generated it.

Python 3.10+. Dependencies are Pillow, numpy and tqdm; nothing else.

## Options

```
k2prep.py <folder> [options]
```

| Option | Default | Meaning |
|---|---|---|
| `<folder>` | required | Input folder, positional. Non-recursive; see [`-R`](#-r-a-folder-and-its-subfolders) to sweep its subfolders too. |
| `--report` | off | Dry run. Analyse and write a report; write no images. |
| `--threshold N` | 0 | Process only images whose composite score ≥ N. 0 processes everything that fits a tier. Range 0–10. |
| `--png` | off | Write PNG instead of JPEG q97 4:4:4. |
| `--filter NAME` | `lanczos` | `lanczos`, `box`, `bicubic`, `bilinear`. |
| `--threads N` | 4 | Worker threads. Range 1–32. |
| `--force` | off | Overwrite existing outputs instead of skipping, and ignore the score cache. |
| `--no-merge` | off | Skip bucket consolidation; leave every image in the bucket its own aspect ratio picks. |
| `--single-pass` | off | Score the source and reject before rendering, instead of scoring the rendered result. |
| `--sort [N]` | off | Triage mode: score every image and file the **original** by quality. Builds no dataset. Bare `--sort` uses absolute bands (`score1`…`score10`, 10 best); `--sort N` (2–10) cuts N populated tiers from this folder (`quality1` best). |
| `--move` | off | With `--sort` only: move the originals instead of copying them. |
| `--vl CRITERIA` | off | With `--sort` only: also have a local vision model rate each image on your own comma-separated criteria, blended with the measured score. |

`--report` and `--threshold` compose: `--report --threshold 7` shows what a
threshold-7 run *would* do without writing anything.

Use `--filter box` for heavily compressed sources. Box averaging suppresses 8×8
block artifacts more cleanly than Lanczos, which can ring on them.

Output goes to `<folder>/_prep/{1024,768,512}/`, flat, with `.txt` caption
sidecars copied alongside, plus `_prep/dataset.toml`. Every run writes two
timestamped reports to `<folder>/_prep/reports/`:

- `*-preliminary.txt` — the natural, per-family bucket assignment, before any
  consolidation and before anything is scored. This is the bucket explosion in
  its raw form.
- `*-final.txt` — what was actually written, with the rendered scores, what the
  merge pass moved, and what it could not.

Reports are never overwritten, so two threshold settings are diffable — and so
are the two stages of a single run.

`_prep/` belongs to k2prep. Each run makes the tier folders match its own plan
exactly: outputs left by an earlier run that the current one does not place
(because the threshold changed, or merging moved an image elsewhere) are removed
and listed under `SUPERSEDED OUTPUTS` in the final report. The source folder is
never touched, so anything removed is one re-run away from coming back.

## `-R`: a folder and its subfolders

k2prep itself is non-recursive on purpose: one folder is one dataset, with one
`_prep` and one `dataset.toml` written next to it. If your photos are already
split into subfolders — one per subject, per shoot, per concept — pass `-R` (or
`-r`) to `run.bat` and it runs k2prep once for the folder, then once for each
first-level subfolder:

```bash
run.bat -R "L:\train" --report --threshold 6
```

```
L:\train\          ->  L:\train\_prep\
L:\train\alice\    ->  L:\train\alice\_prep\
L:\train\bob\      ->  L:\train\bob\_prep\
```

Every folder keeps its own `_prep`, its own reports and its own `dataset.toml`,
because that is what the trainer wants: separate datasets, not one pile. Nothing
below the first level is visited.

Subfolders whose name starts with an underscore are skipped, which covers
`_prep` and the `_foldername` sidecars that `cleanup.bat` makes. The `score*`
and `quality*` folders that `--sort` produces are **not** skipped — building a
dataset out of one triaged tier is a real thing to want.

The remaining options are passed to k2prep unchanged, once per folder. The
folder is whichever argument names a directory that exists, so it can go
anywhere on the line and is never mistaken for an option's value. A folder that
fails does not stop the sweep: the failures are listed at the end and the exit
code is non-zero.

`-R` is a `run.bat` feature. `python k2prep.py` on its own still takes exactly
one folder.

## `cleanup.bat`: moving undersized images out

k2prep skips images too small for the 512 tier and names them in the report, but
it never removes anything, so a folder full of thumbnails stays a folder full of
thumbnails. `cleanup.bat` is the separate, deliberate step that takes them out
of the way:

```bash
cleanup.bat "L:\train\alice" 1024
```

That moves every image with fewer than 1024×1024 pixels, along with its `.txt`
caption sidecar, out of the folder and its first-level subfolders into a sidecar
folder named after the source with a leading underscore, keeping the structure:

```
L:\train\alice\small.jpg      ->  L:\train\_alice\small.jpg
L:\train\alice\1\small.jpg    ->  L:\train\_alice\1\small.jpg
```

| Option | Default | Meaning |
|---|---|---|
| `<folder>` | required | Folder to clean, positional. Its first-level subfolders are cleaned too, except those starting with an underscore. |
| `<size>` | required | Threshold, positional. `N` means `N`×`N`; `WxH` is also accepted, e.g. `1600x900`. |
| `--dim` | off | Compare dimensions instead of area: move an image if either side is shorter than the threshold's. |
| `--dry-run` | off | List what would move and touch nothing. |

The default test is **area**, not dimensions: `1024` means fewer than 1,048,576
pixels. A 2048×400 panorama has 819,200 and is moved; a 1200×900 frame has
1,080,000 and stays. If you would rather reject anything with a short side under
the threshold — which is closer to what decides whether an image can fill a
bucket — use `--dim`, and that same panorama goes for its 400px side while
1200×900 goes with it.

Nothing is deleted and nothing is overwritten. The images are still on disk, one
folder over, so a threshold set too aggressively is undone by moving them back.
If a file of that name is already in the sidecar folder the pair is left where
it is and reported, rather than renamed — a renamed image and its caption would
stop agreeing about their own name. Unreadable files stay put and are listed.
Run `--dry-run` first.

`cleanup.bat` and `-R` skip the same folders, so the usual order needs no
special care:

```bash
cleanup.bat "L:\train" 1024 --dry-run
cleanup.bat "L:\train" 1024
run.bat -R "L:\train" --report
```

## Sorting a folder by quality

`--sort` is a different job from building a dataset. It scores every image and
files the **original** — untouched, full size, with its `.txt` sidecar — under
the folder for its score:

```bash
run.bat "L:\train\photos" --sort
```

```
_prep/score10/   IMG_0431.jpg  IMG_0431.txt  ...
_prep/score9/    ...
_prep/score2/    ...
```

It writes no tier folders, no resized images and no TOML. The point is triage:
see what you have, keep the good folders, then run k2prep normally on whichever
of them you decide to train from.

### `--sort N`: N tiers, cut from your folder

Bare `--sort` uses absolute bands, so a folder of uniformly good photographs
piles into two or three of the ten folders and the rest stay empty. Give it a
number instead and it divides *that folder* into exactly N populated tiers,
named `quality1` (best) through `qualityN`:

```bash
run.bat "L:\train\photos" --sort 3
```

```
QUALITY TIERS  (3 tiers from 29 images)
tier          images   share  score   quality         break
quality1          13   44.8%  10-9    10.67-9.28      gap 0.89
quality2           7   24.1%  8-7     8.39-7.42       gap 0.99
quality3           9   31.0%  6-2     6.43-2.68       -
```

Three is the useful default — keep / maybe / discard.

**Cuts are made by rank, not by absolute score.** That is what guarantees every
tier is populated: a folder of uniformly excellent images still has a best third
and a worst third, and being told all 400 are `score10` helps nobody. On a set
where every image scores 9 or 10 absolutely, `--sort 3` still returns 5 / 3 / 5.

Tiers are **not forced to equal size**. Each cut may slide up to 35% of a tier's
width from its ideal position to settle on the largest natural break it can find
there, so the split follows real structure in the data where there is any. The
`break` column reports the size of the gap each cut landed on.

That sliding window is also what stops **one or two outliers from defining a
grade**. Given a single superb image among nine ordinary ones, `--sort 3` puts
three images in `quality1`, not one — a cut cannot run away to a distant gap just
because the gap is large.

Ranking uses the composite score *plus its position inside that band* (the
`quality` column). A 1–10 integer leaves at most ten distinct values, which is
far too coarse to order a folder by; the fraction comes from where the
measurement actually sits within its band, and the integer part is unchanged.

If there are fewer images than tiers, the spare tiers are left empty and the
report says so. If a cut has to fall between images of identical quality, it is
flagged as arbitrary rather than passed off as a judgement.

Each `--sort` run writes two reports: `sort-*-preliminary.txt` with the scores
and the absolute 1–10 histogram, before any tier decision, and
`sort-*-final.txt` with the tiers that were chosen and where everything went.

**Every image is scored at the 1024 tier**, whatever tier the dataset pipeline
would have put it in. "How good would this be as a training image" has one
answer, and asking it at three different resolutions would make the scores
incomparable between folders — which is the opposite of what sorting is for.
Tier demotion is a packing decision, not a quality one. An image already at or
below the 1024 tier is scored as it is, never upscaled first; the report marks
those with `=`.

`--report` works here too and places nothing.

### `--vl`: your own criteria, judged by a local model

B and D measure whether an image is technically sound, and they are the only
thing here that can. What they cannot do is look at a photograph and say whether
it is well composed, whether the subject is the thing you wanted, or whether it
is a screenshot of a menu. A vision-language model can — and is hopeless at the
reverse, because a vision projector runs at a few hundred pixels, which is
exactly where compression artifacts and fine detail have already been thrown
away.

So `--vl` adds the model's opinion alongside the measurements rather than in
place of them:

```bash
copy sample.env .env          # then point it at your server
run.bat "L:\train\photos" --sort 3 --vl "sharpness, composition, lighting"
```

```
filename            ...  score    tech    vl   rank  destination
IMG_20181004.jpg    ...     10   10.63   8.0   8.84  quality1/
```

`tech` is the measured score, `vl` the model's mean over your criteria, `rank`
the blend the tiers are cut on. The report also lists every per-criterion score,
so you can see *why* something ranked where it did.

**How the two combine.** `rank` is a weighted geometric mean — the model at 0.65,
the measurement at 0.35. Weighted toward the model because it answers the
question you actually asked; geometric rather than an average because an average
lets a perfect critique carry a technically broken image, which is the thing this
blend exists to prevent:

| model | technical | rank |
|---|---|---|
| 10 | 10 | 10.0 |
| 7 | 7 | 7.0 |
| 10 | 2 | 5.7 |
| 10 | 1 | 4.5 |
| 2 | 10 | 3.5 |

A perfect match on your criteria with a broken image lands at 5.7 — clearly
above a technically perfect image that ignores your criteria (3.5), and clearly
below a merely decent one (7.0). Tune `VL_WEIGHT` at the top of `k2prep.py` if
you disagree.

Per-criterion scores are **averaged**, not minimised. The technical composite
uses `min()` because a compression fault is disqualifying; your criteria describe
what you are looking for, and a partial match is a real answer rather than a
failure.

**Local servers only, and no API key is ever sent.** There is no option to add
one. That is deliberate: no key to leak, no bill to run up, and no folder of
photographs leaving the machine because a URL was wrong. A hosted endpoint will
reject the unauthenticated request, which is the intended outcome. Configure it
in `.env` (copy `sample.env`) — llama.cpp, koboldcpp and LM Studio are what this
was built against, and the server needs a **vision** model loaded (`--mmproj` for
llama.cpp).

**A `--vl` run is not reproducible.** A model can answer differently between runs,
and a different model will disagree outright. The report says so in its header.
The technical columns beside it are still exact, and the opinions are cached in
`_prep/vl-cache.json` so re-running the same folder is stable — the cache is
keyed to the model, the criteria and the prompt, so changing any of them
correctly throws it away.

`--threads` controls how many requests are in flight, defaulting to **1** for the
model because llama.cpp, koboldcpp and LM Studio all serve one request at a time
unless started with parallel slots — and a queue of four in front of a slow model
is how you turn a working setup into timeouts. Raising it is worth a little
anyway: against koboldcpp with a 31B vision model, 29 images took 39s at
`--threads 1` and 32s at `--threads 4`, so expect roughly 20%, not 4×. The same
flag raises the local worker count to match.

Everything degrades rather than failing: a missing or malformed `.env`, an
endpoint that is not listening, or one that demands authentication all stop the
run with one sentence before a single image is opened. A reply the model mangles
is reported per-image and that image falls back to its technical score, which is
neither a reward nor a penalty. Five failures in a row abandon the model
entirely and rank the rest on measurement alone.

### `--move`

```bash
run.bat "L:\train\photos" --sort --move
```

**This is the only thing in k2prep that removes anything from your source
folder.** Everywhere else the input folder is strictly read-only, and that has
not changed — but `--move` is asked for by name and does exactly what it says:
each image and its caption leave the source folder for their score folder.

What it will not do:

- Move a file it could not read. Anything that fails to decode stays exactly
  where it is and is named under `ERRORS`.
- Leave a hole. Each destination is written before its source is unlinked, so an
  interruption leaves a duplicate to clean up, never a missing file.
- Delete a stale copy. If an earlier run filed the same image under a different
  score — including a run in the other naming scheme — that copy is listed under
  `DUPLICATES IN OTHER SCORE FOLDERS` and left alone; under `--move` it may be
  the only copy in existence. Clear those by hand.

`--move` without `--sort` is rejected outright.

## Why the output dimensions look arbitrary

The 1024-tier 4:3 bucket is **1184×880**, not 1168×880. That is not a rounding
error.

musubi-tuner generates a bucket list per resolution and assigns each image to the
nearest entry **by aspect ratio alone**. The list is not "any multiple of 16" —
it is produced by walking widths on a 16px grid and flooring `area // w` to 16,
which yields a specific set of 65 pairs at 1024 (49 at 768, 33 at 512). Rounding
`sqrt(area × AR)` to a multiple of 16 by hand gives values that look plausible and
are not in that list.

k2prep therefore generates its targets with musubi's own algorithm, ported
verbatim. If output dimensions do not match a real bucket exactly, the trainer
snaps by aspect ratio to a nearby bucket and applies a **second blind
center-crop** on top of ours — undoing the careful crop and silently discarding
edge content. When the dimensions do match, musubi skips its resize entirely
(`if bucket_reso == (image_width, image_height): return`) and our resampler is
the final word on what the VAE encoder sees.

The per-tier tables differ, because the 16px grid is coarser relative to a
smaller image — 9:16 is 768×1360 at the 1024 tier but 576×1024 at 768. `--report`
prints the full table it used.

## Bucket merging

Assigning every image to its nearest aspect-ratio family is the right first
answer, but on a real folder it leaves a long tail. A tier that looks like this
trains in batches of three no matter what `batch_size` says:

```
  tier 512
        512x512 1:1           5   *** WARNING: fewer than 8 images
        560x464 5:4           3   *** WARNING: fewer than 8 images
        384x672 9:16          1   *** WARNING: fewer than 8 images
        416x624 2:3           1   *** WARNING: fewer than 8 images
        448x576 4:5           1   *** WARNING: fewer than 8 images
```

So after the preliminary report and before anything is written, k2prep
consolidates buckets holding fewer than 8 images. Moved images are re-rendered
**from the original source**, never rescaled from an already-downsized output,
and go through the same crop-box code as everything else. The rules, in order:

1. **Same tier.** Move to the nearest healthy bucket (8+ images) by aspect ratio.
2. **One tier below.** Same test. Demotion costs 44% of the pixels, so it is only
   reached when no same-tier bucket will take the image.
3. **Rescue.** Images that no healthy bucket will take are pooled per tier, and
   if they total 8 they merge into whichever of their own buckets can absorb the
   most of them. This is what fixes the tier above: it becomes 10 in `512x512`
   plus the one 9:16 that would have needed a 43% crop.

A move is refused, and the image stays where it is, if it would:

- crop away more than **40%** of the source,
- leave the image too small to fill the destination within the upscale tolerance,
- or **flip the image between portrait and landscape**. That last one matters
  more than it looks: 4:5 and 5:4 are only 0.44 apart in linear aspect ratio, so
  without the rule a portrait photograph gets cropped into a landscape frame —
  which throws away the subject, not just the margins.

Everything the pass did is in the final report: a `MERGED` table with each move
and its before/after crop, a `STILL UNDERSIZED` table naming every image it left
behind and why, and a bucket distribution showing `was -> now`. Pass
`--no-merge` to turn it all off and get the raw per-family placement.

## Do not set `bucket_no_upscale = true`

The emitted TOML sets it to `false` and says so in a comment. Setting it `true`
bypasses the bucket list and gives each image its own dimensions floored to 16 —
which is exactly the bucket explosion this tool exists to prevent.

## Reading the bucket distribution

This is the section that tells you whether the exercise worked:

```
BUCKET DISTRIBUTION   (was -> now, across the merge)
  tier 1024
       1136x912 5:4         9 ->    16
       832x1248 2:3         2 ->    13   *** WARNING: odd count, trailing batch of 1
       768x1360 9:16        6 ->     0   (dissolved)
      1024x1024 1:1         3 ->     0   (dissolved)
```

Batches are formed per bucket, and `num_batches = ceil(len(bucket) / batch_size)`:

- **fewer than 8 images** — the bucket produces one or two tiny batches whose
  gradients are noisy relative to the rest of the run. Merging exists to remove
  these; any that survive it are listed under `STILL UNDERSIZED` with a reason.
- **odd count** — always leaves a trailing batch of 1. Merging does not chase
  this one: making a bucket even just makes another bucket odd. Add or drop a
  single image of that shape if it bothers you.

If the final report still shows a long list of populated buckets in one tier,
check the preliminary report first — if the two are identical, merging found
nothing it was allowed to move, and the reasons are in `STILL UNDERSIZED`.

## Quality is scored on the rendered image, not the source

Quality is resolution-dependent, so measuring the source tells you about pixels
the trainer never sees. Take one photograph, save it twice at JPEG q30 — once at
5056×3792, once at 1300×975 — and both land in the same 1136×912 bucket. They are
not the same training image:

| | source-scored (`--single-pass`) | rendered-scored (default) |
|---|---|---|
| 5056×3792 @ q30 | 1 | **10** |
| 1300×975 @ q30 | 1 | **2** |

The large one is downscaled 4×, which averages the 8×8 block edges away
completely; what reaches the VAE is clean. The small one arrives at roughly 1:1
and keeps every artifact it ever had. Scoring the source cannot tell them apart,
and rejects the good one.

So by default k2prep renders every image that fits a tier, scores that, and only
then writes the ones that pass. Rejected images are never written — the rendered
copy exists in memory only long enough to be measured.

- **D** — detail, as high-frequency energy normalised by contrast, on the
  rendered image. This carries most of the scoring.
- **B** — strength of the source's 8×8 block grid *where it survived the resize*.
  An 8px source block lands every `8 × scale` output pixels, so the metric looks
  for periodicity at that (usually fractional) period. Below 3 output pixels per
  block it reports `n/a` — not a gap in the measurement, the artifacts are
  genuinely gone. Only JPEG sources have a grid to look for.
- **Q~** — estimated JPEG quality from the source's quantization tables.
  **Reported only, deliberately excluded from the score.** It describes the
  source encode, which a downscale has already discarded. Letting it into a
  `min()` composite is exactly what made good high-resolution material score 1.
- **R** — resolution headroom against the assigned tier's bucket, a raw factor
  rather than a score.

The composite is the **minimum** of the metrics that apply, so `--threshold 6`
means "every available metric is at least 6".

`--single-pass` restores the old behaviour: Q, B and D measured on the source,
and the threshold applied before anything is rendered. It is faster and it is
what you want if you already know your sources are uniform in resolution.

### Cost, and the score cache

The default mode renders each image twice on a first run — once to score, once
to write. Scores do not depend on `--threshold`, so they are cached in
`_prep/metrics-cache.json`, keyed on file size and mtime. Re-running at a different
threshold reuses every score and rewrites only what changed, which makes the
tune-and-re-run loop essentially free. Delete the file to force a rescore;
`--force` ignores it.

### The bands are uncalibrated

**Both band tables are a starting point and are explicitly uncalibrated.** They
were fitted against a 29-image reference folder and a controlled quality sweep,
which is better than nothing and a long way from calibrated. D in particular
produces genuine false positives on bokeh, fog, snow, and deliberately minimal
compositions.

Run `--report` on your actual folder and read the QUALITY DISTRIBUTION histogram
in the final report — it covers every image that reached a tier, including those
below the threshold, so you can see what a different threshold would recover.
Then adjust `D_RENDERED_BANDS` and `B_RENDERED_BANDS` at the top of `k2prep.py`
before trusting `--threshold` to act on them.

## Tests

```bash
python test_k2prep.py
```

Covers the bucket generation port, the per-tier family table, the geometry half
of the acceptance criteria, the merge rules, and the rendered-image metrics.

## License

MIT.
