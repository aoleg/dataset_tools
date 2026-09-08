# k2prep — dataset preprocessor for musubi-tuner / Krea 2 LoRA training

Build specification, Phase 1.

---

## 1. Purpose

Take a messy folder of mixed photographs and produce a small, clean, bucket-tight
training set for musubi-tuner's Krea 2 trainer.

The problem being solved: musubi-tuner assigns each image to the nearest entry in a
generated bucket list by aspect ratio alone. At `resolution = [1024, 1024]` that list
has 65 entries spaced roughly 1.5% apart near square, so a folder of ordinary
photographs scatters across 15+ buckets. Batches are formed per bucket
(`BucketBatchManager` in `dataset/bucket.py`), so a scattered dataset produces many
undersized batches and wastes the batch-size setting entirely.

This tool crops and resizes every qualifying image to land on exactly one of 7 chosen
aspect ratios across 3 resolution tiers, so the trainer sees at most 21 buckets and
usually far fewer.

Secondary purpose: score input quality and reject material that would teach the LoRA
compression artifacts.

### Selection philosophy

The output is a **subset**, not a transformation of the whole folder. Images that
fail `--threshold` or that are too small are **skipped entirely**: not copied, not
moved, not modified. The source folder is never written to and never has anything
removed from it. The only record of a rejection is the report.

This is deliberate. The input is assumed to be low-to-medium quality material of
mixed provenance, and the goal is to extract the usable part of it.

### Critical correctness requirement

Output dimensions **must exactly match an entry in musubi-tuner's generated bucket
list** for the tier they are placed in. If they do not, the trainer snaps by aspect
ratio to a nearby bucket and applies a second blind center-crop on top of ours.

The target dimensions are therefore **generated with musubi's own algorithm**
(section 4.2), not by rounding `sqrt(area × AR)` to a multiple of 16. Those two
methods disagree: for 4:3 at the 1024 tier, naive rounding gives 1168×880, which is
not in musubi's list; the real bucket is 1184×880.

---

## 2. Repository layout

```
k2prep/
├── k2prep.py            single-file script, all logic
├── requirements.txt
├── install.bat          Windows: create venv, install deps
├── run.bat              Windows: activate venv, run k2prep.py with passthrough args
├── README.md            usage, examples, the "why exact bucket dims" explanation
├── .gitignore
└── LICENSE              MIT
```

Single file is deliberate. The script is roughly 900 lines and splitting it adds
import ceremony for no benefit. If it grows past ~1500 lines, split out `metrics.py`
and `buckets.py` only.

`.gitignore` must include `venv/`, `__pycache__/`, `*.pyc`, `_prep/`.

---

## 3. Dependencies

`requirements.txt`:

```
Pillow>=10.0.0
numpy>=1.24.0
tqdm>=4.66.0
```

Nothing else. No OpenCV: Pillow plus numpy covers every operation here, and avoiding
the OpenCV wheel keeps `install.bat` fast and sidesteps the DLL problems it causes on
Windows.

Python 3.10 or newer.

### install.bat

```
@echo off
setlocal
cd /d "%~dp0"
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Install complete. Use run.bat to run k2prep.
pause
```

### run.bat

```
@echo off
setlocal
cd /d "%~dp0"
if not exist venv\Scripts\activate.bat (
    echo Virtual environment not found. Run install.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
python k2prep.py %*
```

`run.bat` must pass `%*` through unmodified so `run.bat "L:\train\photos" --report`
works. Do not `pause` at the end of `run.bat`; it breaks scripted use.

---

## 4. Core constants and bucket generation

### 4.1 Tiers

```python
TIERS = [1024, 768, 512]          # nominal resolution, descending
UPSCALE_TOLERANCE = 1.15          # max permitted linear upscale into a tier
```

`TIERS` must be a single module-level constant. Krea 2's technical report states that
pretraining progressively scaled through 256px, 512px and 1024px stages, and there is
secondhand advice circulating that 768 should therefore be avoided for training. That
advice is anecdotal rather than measured, and 768 is included here for demotion
granularity: without it the gap between tiers is 4× in area and near-boundary images
lose most of their pixels. Editing one line reverts that decision.

256 is deliberately **not** a tier. Anything that does not reach the 512 tier is
skipped.

`UPSCALE_TOLERANCE` prevents cliff-edge demotion. Lanczos upscaling softens slightly
but fabricates nothing, so a 15% upscale is a far better trade than dropping a tier
and discarding 44% of the pixels.

Resulting minimum **post-crop** source areas:

| Tier | Nominal area | Minimum post-crop area |
|---|---|---|
| 1024 | 1,048,576 | 792,874 |
| 768 | 589,824 | 445,991 |
| 512 | 262,144 | 198,218 |

Below 198,218 px post-crop the image is skipped as too small.

### 4.2 Bucket generation (must match musubi-tuner exactly)

Port of `BucketSelector.__init__` from `src/musubi_tuner/dataset/bucket.py`.
`RESOLUTION_STEPS_KREA2 = 16` (VAE f8 compression × patch size 2).

```python
RESO_STEPS = 16

def divisible_by(n: int, d: int) -> int:
    return n - n % d

def generate_buckets(resolution: int, steps: int = RESO_STEPS) -> list[tuple[int, int]]:
    """Byte-for-byte equivalent to musubi-tuner's bucket list for a square target."""
    area = resolution * resolution
    sqrt_size = int(math.sqrt(area))
    min_size = divisible_by(sqrt_size // 2, steps)
    out = []
    for w in range(min_size, sqrt_size + steps, steps):
        h = divisible_by(area // w, steps)
        out.append((w, h))
        out.append((h, w))
    return sorted(set(out))
```

Expected list sizes: 65 at 1024, 49 at 768, 33 at 512.

There must be a unit test asserting `len(generate_buckets(1024)) == 65` and that
`(1184, 880) in generate_buckets(1024)` while `(1168, 880) not in generate_buckets(1024)`.

### 4.3 Aspect-ratio families

Seven families, selected by nearest ratio at the **1024 tier**:

```python
AR_FAMILIES = ["9:16", "2:3", "4:5", "1:1", "5:4", "3:2", "16:9"]
AR_NOMINAL  = [0.5647, 0.6667, 0.8028, 1.0000, 1.2456, 1.5000, 1.7708]
```

The nominal values are the *actual 1024-tier bucket ratios*, not the idealised ones.
9:16 is 0.5647 not 0.5625; 4:5 is 0.8028 not 0.8000.

**The per-tier bucket dimensions differ**, because the 16px grid is coarser relative
to a smaller image. Resolve each family to its tier's bucket at runtime with
`min(buckets, key=lambda b: abs(b[0]/b[1] - nominal))`. The resulting table, which
`--report` must print for verification:

| Family | 1024 tier | 768 tier | 512 tier |
|---|---|---|---|
| 9:16 | 768×1360 (0.5647) | 576×1024 (0.5625) | 384×672 (0.5714) |
| 2:3 | 832×1248 (0.6667) | 624×944 (0.6610) | 416×624 (0.6667) |
| 4:5 | 912×1136 (0.8028) | 688×848 (0.8113) | 448×576 (0.7778) |
| 1:1 | 1024×1024 (1.0000) | 768×768 (1.0000) | 512×512 (1.0000) |
| 5:4 | 1136×912 (1.2456) | 848×688 (1.2326) | 560×464 (1.2069) |
| 3:2 | 1248×832 (1.5000) | 944×624 (1.5128) | 624×416 (1.5000) |
| 16:9 | 1360×768 (1.7708) | 1024×576 (1.7778) | 672×384 (1.7500) |

Latent token counts are near-constant within a tier: ~4,050–4,096 at 1024,
~2,280–2,304 at 768, ~1,008–1,024 at 512. This matters for the emitted TOML
(section 10).

Because the ratios differ between tiers, **the crop box cannot be computed until the
tier is known**. See section 5.

---

## 5. Processing pipeline, per image

Order matters. Deviating from it produces wrong crops or wrong tiers.

### 5.1 Load and normalise

1. Open with Pillow. Catch and log truncated/corrupt files; do not crash the batch.
2. Apply `ImageOps.exif_transpose(img)` **before reading dimensions**. Phone and DSLR
   portraits are stored landscape with an orientation tag; skipping this
   bucket-assigns them along the wrong axis.
3. Convert to `RGB` if not already (handles `P`, `LA`, `RGBA`, `CMYK`, `I;16`). For
   `RGBA`, composite over white; do not just drop the alpha channel.
4. Record `src_w`, `src_h`, `src_ar = src_w / src_h`.

EXIF is **not** carried to the output. Do not pass `exif=` to `save()`. This is what
prevents orientation being applied twice by downstream viewers. ICC profiles are also
dropped; all inputs are assumed sRGB.

### 5.2 Assign aspect-ratio family

`family = argmin |AR_NOMINAL[i] - src_ar|`.

No guard, no exclusion. A 21:9 panorama is assigned to 16:9 and cropped hard. This is
intentional: the whole point is to stop weird dimensions multiplying buckets. The crop
fraction is recorded and reported so heavy crops remain visible.

### 5.3 Assign tier

For each tier in `TIERS` (descending), resolve the family's bucket at that tier,
compute the exact post-crop source area, and take the first tier that fits:

```python
for tier in TIERS:
    bw, bh = bucket_for(tier, family)
    target_ar = bw / bh
    cw, ch = crop_dims(src_w, src_h, target_ar)   # section 5.4
    if cw * ch >= (bw * bh) / (UPSCALE_TOLERANCE ** 2):
        return tier, (bw, bh), (cw, ch)
return None   # -> skipped, too small
```

The area test uses **post-crop** dimensions. A 2400×600 panorama has 1.44 MP but only
1067×600 = 0.64 MP after cropping to the 768-tier 16:9 ratio, so it belongs in the 768
tier. Testing raw source area would place it in 1024 and force an upscale.

### 5.4 Compute crop box

Minimal crop to the target ratio, in source pixels:

```python
def crop_dims(w, h, target_ar):
    if w / h > target_ar:
        return int(round(h * target_ar)), h     # too wide: trim width
    else:
        return w, int(round(w / target_ar))     # too tall: trim height
```

Crop origin: horizontal always centred. Vertical centred **except** when the target is
portrait (`target_ar < 1.0`), in which case bias the origin to 1/3 of the excess
rather than 1/2. Heads sit above centre in almost all photography and this is right
more often than centre at zero cost.

```python
excess_y = src_h - ch
top  = int(excess_y * (1/3 if target_ar < 1.0 else 1/2))
left = (src_w - cw) // 2
box  = (left, top, left + cw, top + ch)
```

This function is the `--anchor` seam. See section 13.

### 5.5 Crop and resize in one pass

Use Pillow's `box` argument so the resampling filter operates directly on the crop
region. Do not crop to an intermediate image and then resize; that is an extra
allocation and, with some filters, an extra rounding.

```python
out = img.resize((bw, bh), resample=FILTER, box=box)
```

**Filter default is `Image.LANCZOS`.** Not bilinear. Pillow scales the filter support
by the reduction factor so `BILINEAR` is antialiased, but it is soft, and because our
output dimensions match a bucket exactly, musubi will skip its own resize entirely
(`if bucket_reso == (image_width, image_height): return`). Our resampler is the final
word on what the VAE encoder sees.

`--filter` accepts `lanczos` (default), `box`, `bicubic`, `bilinear`. `box` maps to
`Image.BOX` and reproduces what musubi's `cv2.INTER_AREA` would have done; it is the
better choice for heavily compressed sources, since box averaging suppresses 8×8 block
artifacts more cleanly than Lanczos, which can ring on them.

### 5.6 Save

Default: JPEG, `quality=97`, `subsampling=0` (4:4:4), `optimize=True`. Roughly
250–400 KB per 1 MP image and visually indistinguishable from PNG after a VAE encode.

`--png` switches to PNG, `compress_level=6`. Expect 1.5–2.5 MB per image.

Output extension always matches the format actually written, regardless of input
extension.

### 5.7 Caption sidecar

If `<stem>.txt` exists next to the source, copy it into the tier folder beside the
produced image, renamed to match the output stem. Copy verbatim; do not re-encode or
strip whitespace.

Missing captions are counted and listed in the report. They are not an error and do
not prevent processing.

A skipped image's caption is never copied anywhere.

---

## 6. Quality scoring

Four sub-metrics, each scored 1–10 on a **fixed absolute scale**. The bins are not
percentiles. `--threshold 6` must mean the same thing on every folder, on every run,
forever; percentile bins would make it dataset-relative and unreproducible.

All metrics are computed on the **source** image after EXIF transpose and RGB
conversion, before any crop or resize, because `--threshold` gates whether processing
happens at all.

Work on the luma plane: `L = img.convert("L")` as a numpy `float32` array. For speed,
metrics B and D may be computed on a centre crop of at most 1024×1024 pixels when the
source is larger; state this in the report header.

### 6.1 Q — compression quality

From the JPEG quantization tables, available as `img.quantization` in Pillow (dict of
table index to 64-element list). PNG and other lossless inputs score 10.

Method: reconstruct the IJG standard luma table, generate the scaled table for each
quality 1–100 using the standard IJG scaling formula, and pick the quality that
minimises sum of squared differences against the image's table 0.

```
scale = 5000/q if q < 50 else 200 - 2q
value = clamp((std[i] * scale + 50) / 100, 1, 255)
```

If the best-fit residual exceeds a threshold, the encoder used non-standard tables
(common with Adobe and several phone encoders). In that case set `Q = None`, mark it
`n/a` in the report, and exclude it from the composite.

| Estimated JPEG q | Score |
|---|---|
| lossless / ≥98 | 10 |
| 95–97 | 9 |
| 92–94 | 8 |
| 88–91 | 7 |
| 84–87 | 6 |
| 78–83 | 5 |
| 72–77 | 4 |
| 65–71 | 3 |
| 55–64 | 2 |
| <55 | 1 |

### 6.2 B — blockiness

Detects 8×8 DCT block edges.

```
col_diff[x]  = mean(|L[:, x] - L[:, x-1]|)
aligned      = mean over x where x % 8 == 0
non_aligned  = mean over x where x % 8 != 0
ratio_x      = aligned / max(non_aligned, epsilon)
```

Same for rows. `B_ratio = max(ratio_x, ratio_y)`.

| B_ratio | Score |
|---|---|
| ≤1.05 | 10 |
| ≤1.10 | 9 |
| ≤1.15 | 8 |
| ≤1.20 | 7 |
| ≤1.30 | 6 |
| ≤1.40 | 5 |
| ≤1.55 | 4 |
| ≤1.75 | 3 |
| ≤2.00 | 2 |
| >2.00 | 1 |

**Known limitation, must be stated in the report header:** if an image was rescaled
after being JPEG-encoded, the 8px grid no longer aligns and this metric reads clean on
genuinely damaged material. When `Q` is `None` *and* the source dimensions are not both
multiples of 8, append a `?` to the B score in the report to flag it as unreliable.

### 6.3 D — detail

Detects upscaled, out-of-focus, and over-denoised material.

```
small = L.resize((w//2, h//2), Image.BOX)
back  = small.resize((w, h), Image.BILINEAR)
hf    = mean(|L - back|)
D_ratio = hf / max(std(L), epsilon)
```

Normalising by contrast keeps low-contrast but sharp images from being penalised.

| D_ratio | Score |
|---|---|
| ≥0.070 | 10 |
| ≥0.055 | 9 |
| ≥0.045 | 8 |
| ≥0.035 | 7 |
| ≥0.026 | 6 |
| ≥0.019 | 5 |
| ≥0.013 | 4 |
| ≥0.008 | 3 |
| ≥0.004 | 2 |
| <0.004 | 1 |

**These bands are a starting point and are explicitly uncalibrated.** They produce
genuine false positives on bokeh, fog, snow, and deliberately minimal compositions.
Define them as a module-level constant table with a comment saying so, run `--report`
on a real folder, look at the distribution, and adjust before trusting `--threshold`
to act on D.

### 6.4 R — resolution headroom (reported, not scored into the composite)

`R_factor = sqrt(post_crop_area / assigned_bucket_area)`

**Tier-relative.** A clean 640×480 photo assigned to the 512 tier has R_factor ≈ 1.03
and is fine. Measuring against 1024 would score every small image as garbage and
defeat the whole tier system.

Reported as a raw factor to two decimals, not a 1–10 score. It is informative rather
than disqualifying: a factor near 1.0 means the source was barely larger than the
target, so whatever artifacts it has survive at full strength into training. Images
below `1 / UPSCALE_TOLERANCE` are already skipped as too small by section 5.3, so R
needs no separate gate.

### 6.5 Composite

```python
composite = min(s for s in (Q, B, D) if s is not None)
```

**Minimum, not weighted mean.** Quality faults are disqualifying rather than additive:
a 6000px razor-sharp photo saved at JPEG q55 is a bad training image, and an average
would let the good dimensions hide that. `min` makes `--threshold 6` mean "every
available metric is at least 6," which is a claim a human can verify by looking at one
image.

If all three are `None` (should be impossible), composite is 10 and the image is
flagged in the report.

---

## 7. Output layout and file handling

```
<input_folder>/
├── (source images — never modified, never moved, never deleted)
└── _prep/
    ├── 1024/
    ├── 768/
    ├── 512/
    └── reports/
```

A single `_prep` parent means the input scanner skips exactly one directory name.

Tier folders hold accepted images plus their `.txt` sidecars, flat. Each populated
tier folder is one `[[datasets]]` block in the training TOML.

**There is no exclusion folder.** Rejected images are skipped: nothing is written for
them, and the source file stays exactly where it is. The report is the only record.
This makes the full filename lists in the report's rejection sections load-bearing
rather than decorative, so they must not be truncated.

The intended recovery workflow: lower `--threshold` and re-run. Because existing
outputs are skipped (section 7.3), the second run only adds the newly-qualifying
images and costs nothing for the ones already processed.

### 7.1 Scanning

- Non-recursive. Only files directly in the input folder.
- Skip `_prep/` unconditionally.
- Accept extensions: `.jpg .jpeg .png .webp .bmp .avif` and their uppercase forms.
  Deliberately mirrors musubi's `IMAGE_EXTENSIONS`. Mixed-case variants like `.Jpg`
  will not be found; report a count and list of files skipped for unknown extension so
  the user notices.
- Sort by filename before processing, so reports from two runs are diffable.

### 7.2 Filename collisions

Two sources can produce the same output name (`photo.jpg` and `photo.png` both become
`photo.jpg`). Resolve by suffixing `_2`, `_3`, … in scan order.

Caption handling for collisions: if only one of the colliding sources has a `.txt`,
copy that same caption to **both** output stems. Colliding stems are almost always the
same photograph in two formats, so the caption applies to both. Log every collision in
the report.

Collisions are resolved per tier folder, not globally. Two images that collide by name
but land in different tiers do not need suffixing.

### 7.3 Idempotency

Skip an image if its output file already exists in its assigned tier, unless
`--force`. Report the skip count. This script will be run repeatedly while tuning
`--threshold`, and re-encoding 8,000 images to discover nothing changed is a waste.

### 7.4 `--recursive`: several datasets, one `_prep`

Added in 1.1.0; this is the Phase 2 design section 13 deferred.

A **dataset is any directory that directly contains at least one image**, and its
images are only the files directly in it. The output mirrors the source tree
inside a single `_prep` at the scan root, tier folders at the leaves:

```
parent/
├── summer/                     -> _prep/summer/{1024,768,512}/
├── winter/indoor/              -> _prep/winter/indoor/{1024,768,512}/
├── loose.jpg                   -> _prep/{1024,768,512}/   (root = dataset "")
└── _prep/
    ├── reports/                one, shared
    └── dataset.toml            one, shared
```

Two properties fall out of that rule and are the reason for it: non-recursive
mode is the degenerate case (a tree with no subfolders produces byte-identical
output with and without the flag), and each source directory maps 1:1 to one
`[[datasets]]` block. Pooling a subtree is the user's call, made by flattening
the source; k2prep does not guess.

Scan rules, at every depth:

- Skip directories starting with `_` (covers `_prep` — including one left in a
  subfolder by an earlier `run.bat -R` run, which holds already-cropped renders
  of the same photos — and `cleanup.bat` sidecars) and starting with `.`.
- Never follow directory symlinks or junctions (cycles, escapes); a visited set
  of resolved paths is the backstop. Skips are noted in the report.
- An unreadable directory is reported and the scan continues.
- Refuse with a clear error, before any work: a source directory named like a
  tier (`1024`/`768`/`512`) anywhere — it would make `_prep/<x>/1024` ambiguous
  between "dataset" and "tier folder" — and a first-level directory named
  `reports`, which would collide with `_prep/reports`.

Everything per-dataset stays per-dataset, because each populated
`<dataset>/<tier>` leaf is its own `[[datasets]]` block and musubi forms batches
within a dataset only:

- the merge pass runs once per dataset and never moves an image across datasets;
- the report's bucket distribution and its undersized/odd warnings are printed
  per dataset;
- filename collisions are resolved per (dataset, tier);
- idempotency checks and superseded-output sweeping run per dataset, over every
  directory the scan visited — so a dataset whose images all vanished still gets
  its stale outputs cleared, exactly as an emptied folder does in the flat
  layout;
- the score cache is keyed by relpath-qualified name (`summer/img.jpg`), so root
  entries stay compatible with non-recursive runs.

Outputs orphaned by a renamed or deleted source subfolder are reported as stale
and left alone; the regenerated TOML does not reference them, so they cannot
leak into training. `--sort` is refused together with `--recursive` — what
triage means across a tree of datasets is deliberately undecided.

`run.bat -R` remains the other thing: one independent run, `_prep` and TOML per
first-level subfolder (train separately), versus `--recursive`'s one shared
`_prep` and TOML (train together).

---

## 8. Command-line interface

```
k2prep.py <folder> [options]
```

| Option | Default | Meaning |
|---|---|---|
| `<folder>` | required | Input folder, positional. One dataset, unless `--recursive`. |
| `--report` | off | Dry run. Analyse and write a report; write no images. |
| `--recursive` | off | Every subfolder that directly contains images is its own dataset, sharing one `_prep` and one TOML. Section 7.4. |
| `--threshold N` | 0 | Process only images whose composite score ≥ N. 0 processes everything that fits a tier. Range 0–10. |
| `--png` | off | Write PNG instead of JPEG q97 4:4:4. |
| `--filter NAME` | `lanczos` | `lanczos`, `box`, `bicubic`, `bilinear`. |
| `--threads N` | 4 | Worker threads. Range 1–32. |
| `--force` | off | Overwrite existing outputs instead of skipping. |
| `--emit-toml` | off | Also write a ready-to-use musubi dataset TOML (section 10). |

`--report` and `--threshold` compose: `--report --threshold 7` shows what a
threshold-7 run *would* do without writing anything. Given that rejections are not
recoverable from disk, running `--report` first is the recommended workflow and the
README should say so.

Argument validation must be strict and fail fast with a clear message. An out-of-range
`--threshold` or `--threads` exits non-zero; do not silently clamp.

---

## 9. Reports

Written to `_prep/reports/`, plain UTF-8 text, never overwritten:

- `scan-YYYYMMDD-HHMMSS.txt` for `--report` runs
- `process-YYYYMMDD-HHMMSS.txt` for processing runs

Timestamped filenames make two threshold settings diffable.

### 9.1 Layout

```
k2prep report
==================================================================
folder      : L:\train\photos
run         : process           (or: scan / dry run, no files written)
started     : 2026-08-10 14:22:31
finished    : 2026-08-10 14:31:04
options     : threshold=6 filter=lanczos format=jpeg-q97-444 threads=4
tiers       : 1024, 768, 512   (upscale tolerance 1.15)
policy      : rejected images are skipped, not copied. Source folder unmodified.
metric note : B is computed on the 8px grid and is unreliable for images
              rescaled after JPEG encoding; such scores are marked with ?
              B and D are computed on a centre crop of at most 1024x1024.

TARGET BUCKETS
------------------------------------------------------------------
family    1024 tier        768 tier         512 tier
9:16      768x1360         576x1024         384x672
2:3       832x1248         624x944          416x624
4:5       912x1136         688x848          448x576
1:1       1024x1024        768x768          512x512
5:4       1136x912         848x688          560x464
3:2       1248x832         944x624          624x416
16:9      1360x768         1024x576         672x384

PROCESSED  (4,812 images)
------------------------------------------------------------------
filename                  source      AR     family  tier  bucket      crop%   R     Q   B   D  score
IMG_0431.jpg              6000x4000   1.500  3:2     1024  1248x832     0.0%  4.81   9  10   8      8
DSC_9920.jpg              3008x2000   1.504  3:2     1024  1248x832     0.3%  2.41   8   9   7      7
pano_x.jpg                2400x600    4.000  16:9     768  1024x576    55.5%  1.04  10  10   9      9
old_cam.jpg               640x480     1.333  5:4      512  560x464      9.5%  1.03   7   8   6      6

SKIPPED — below threshold 6  (338 images)
------------------------------------------------------------------
filename                  source      score  failed
web_grab_02.jpg           1200x800        2  Q=2 (est. q58)
upscaled_ref.png          2048x1536       3  D=3 (ratio 0.009)

SKIPPED — too small  (57 images)
------------------------------------------------------------------
filename                  source      family  post-crop    needed
thumb_a.jpg               400x300     5:4     362x300      198,218

SKIPPED — already processed  (0 images)
------------------------------------------------------------------

FILENAME COLLISIONS  (3)
------------------------------------------------------------------
photo.png -> photo_2.jpg   (caption copied from photo.txt)

MISSING CAPTIONS  (14)
------------------------------------------------------------------
IMG_0102.jpg

SKIPPED — unknown extension  (2)
------------------------------------------------------------------
notes.Jpg
scan.TIFF

ERRORS  (1)
------------------------------------------------------------------
broken.jpg    OSError: image file is truncated

==================================================================
SUMMARY
==================================================================

TIER DISTRIBUTION
  1024      3,902   74.9%
   768        718   13.8%
   512        192    3.7%
  skipped     395    7.6%   (338 below threshold, 57 too small)
  total     5,207

BUCKET DISTRIBUTION
  tier 1024
    1248x832    3:2      1,890
    1024x1024   1:1        902
     832x1248   2:3        744
    1360x768   16:9        401
    1136x912    5:4        165
     912x1136   4:5        197  *** WARNING: odd count, trailing batch of 1
     768x1360   9:16         3  *** WARNING: fewer than 8 images
  tier 768
    ...
  tier 512
    ...

QUALITY DISTRIBUTION (composite, all images that fit a tier)
  10  ############################              1,204   23.1%
   9  ####################                        861   16.5%
   8  ###############                             702   13.5%
   7  ############                                588   11.3%
   6  ##########                                  485    9.3%
   5  ########                                    391    7.5%
   4  ######                                      288    5.5%
   3  ####                                        180    3.5%
   2  ##                                           98    1.9%
   1  #                                            42    0.8%

CROP COST
  median crop        1.4%
  mean crop          6.8%
  images >25% crop     91
  heaviest 20:
    pano_003.jpg     58.2%   (4000x1000 -> 16:9)

TIMING
  scanned            5,207 images in 8m 33s (10.1 img/s, 4 threads)
```

### 9.2 Notes on the summary

The bucket distribution is the section that tells you whether the exercise worked. If
it still shows 15 populated buckets, something upstream is wrong.

Emit a `*** WARNING` line on any bucket holding fewer than 8 images, because
`num_batches = ceil(len(bucket) / batch_size)` means small buckets produce undersized
batches. Also warn on any bucket with an odd count, which always leaves a trailing
batch of 1.

The quality histogram uses fixed 1–10 bins with counts and a bar. It is a display of
the distribution, not an adaptive binning. It covers every image that reached a tier,
including those below the threshold, so the user can see what a different threshold
would recover.

Report file size: for 8,000 images the per-image table is roughly 1 MB of text. That is
acceptable. Do not truncate any section, especially the skip lists, since they are the
only record of what was rejected.

---

## 10. `--emit-toml`

Writes `_prep/dataset.toml`, one `[[datasets]]` block per populated tier:

```toml
# generated by k2prep on 2026-08-10 14:31:04
#
# Do not set bucket_no_upscale = true. It bypasses the bucket list entirely and
# assigns each image its own dimensions floored to 16, which is exactly the bucket
# explosion this tool exists to prevent.
#
# batch_size values below keep the per-step latent token count roughly constant
# across tiers (~4,080 tokens at 1024, ~2,300 at 768, ~1,010 at 512). Lower them
# if VRAM says otherwise.
#
# num_repeats is 1 everywhere. Per-tier and per-concept balancing is your job;
# an unbalanced dataset combined with a high learning rate is the worst case for
# multi-concept training.

[general]
caption_extension = ".txt"

[[datasets]]
image_directory = "L:/train/photos/_prep/1024"
resolution = [1024, 1024]
enable_bucket = true
bucket_no_upscale = false
batch_size = 2
num_repeats = 1

[[datasets]]
image_directory = "L:/train/photos/_prep/768"
resolution = [768, 768]
enable_bucket = true
bucket_no_upscale = false
batch_size = 3
num_repeats = 1

[[datasets]]
image_directory = "L:/train/photos/_prep/512"
resolution = [512, 512]
enable_bucket = true
bucket_no_upscale = false
batch_size = 8
num_repeats = 1
```

Rules:

- Forward slashes in paths. TOML and Windows backslashes interact badly.
- Skip tiers with zero images.
- `batch_size`: 2 at 1024, 3 at 768, 8 at 512.
- Never overwrite an existing `dataset.toml`; write `dataset-2.toml` and say so in the
  console output and the report.

---

## 11. Concurrency

`concurrent.futures.ThreadPoolExecutor`, default 4 workers, `--threads 1..32`.

Threads rather than processes is correct here: Pillow releases the GIL for JPEG
decode, resize and encode, which is where essentially all the time goes. Processes
would add pickling overhead and complicate the progress bar for no gain.

Requirements:

- Per-image work fully independent. No shared mutable state; collect futures and
  assemble results after.
- Filename-collision resolution (section 7.2) is inherently sequential. Do it in the
  main thread after the parallel analysis phase and before the write phase, or reserve
  output names under a lock.
- Progress via `tqdm` over completed futures.
- Report ordering by filename, not completion order. Sort results before writing.
- One image raising must not kill the run. Catch broadly per image, record the
  exception text in the report's `ERRORS` section, continue.
- `--threads 1` must produce byte-identical output and an identical report (apart from
  the timing line) to `--threads 16`. If it does not, there is shared state.

---

## 12. Acceptance criteria

The implementation is correct when all of these hold.

1. `generate_buckets(1024)` returns 65 entries, `generate_buckets(768)` returns 49,
   `generate_buckets(512)` returns 33.
2. `(1184, 880) in generate_buckets(1024)` and `(1168, 880) not in generate_buckets(1024)`.
3. Every image written to `_prep/<tier>/` has dimensions present in
   `generate_buckets(tier)`. Verify by walking the output tree.
4. A JPEG with EXIF orientation 6 is bucketed on its *displayed* dimensions, and the
   output file has no EXIF block.
5. A 2400×600 image is assigned to the **768** tier, bucket 1024×576.
6. A 640×480 image is assigned to the **512** tier, bucket 560×464.
7. A 400×300 image is skipped as too small, and no file is written for it anywhere.
8. After any run, the input folder contains exactly the files it contained before,
   byte-identical, plus the `_prep/` directory.
9. Running twice without `--force` writes no images on the second run and reports the
   correct already-processed count.
10. `--report` writes exactly one file and creates no tier directories.
11. `--threshold 11` and `--threads 0` exit non-zero with a readable message.
12. A folder containing `photo.jpg`, `photo.png` and `photo.txt` produces two output
    images and two caption files with matching stems.
13. A corrupt/truncated JPEG appears in the report's `ERRORS` section and does not
    abort the run.
14. `--threads 1` and `--threads 8` produce identical reports apart from the timing
    line.
15. `--emit-toml` output parses with `tomllib.load` and every `image_directory` it
    names exists and is non-empty.

---

## 13. Phase 2 hooks and explicit non-goals

### Hooks to leave in place

**`--anchor`.** All crop-box computation must go through a single function with the
signature `(src_w, src_h, target_ar) -> (left, top, right, bottom)`. Centre-with-
portrait-bias is one implementation; a face-aware one is another with the same
signature. Keep every caller going through it and Phase 2 is a one-file change.

**`--recursive`.** Implemented in 1.1.0; the design decision (mirror the source
tree, per-dataset scoping everywhere) is recorded in section 7.4.

**`--keep-rejected`.** If the skip-only policy ever proves too aggressive, the hook is
a single branch at the point of rejection. Do not build the folder structure for it now.

**Deblocking.** If it is ever added it belongs in the DCT domain (ffmpeg's `deblock` /
`spp` / `uspp`), gated on a measured B ratio, and applied only when R_factor is below
about 1.5. A generic spatial denoiser (NLM, bilateral) is not acceptable at any
strength: it cannot distinguish JPEG ringing from skin pores, fabric weave and hair,
and removing those is precisely how you train a LoRA that produces plastic skin.

### Non-goals for Phase 1

- No 256 tier.
- No lossless JPEG cropping via jpegtran. Its only advantage is avoiding a re-encode,
  which evaporates the moment you re-encode at q97 anyway, and the EXIF-rotation
  interaction (`-rotate` then `-crop` in separate passes, then clearing the orientation
  tag) is the most bug-prone part of the whole design for the least benefit.
- No upscaling beyond `UPSCALE_TOLERANCE`.
- No writing to, moving within, or deleting from the source folder under any flag.
- No ICC handling. All inputs assumed sRGB.
- No duplicate or near-duplicate detection.
- No caption generation, editing or validation beyond copying and counting.
- No video, no animated formats. First-frame extraction is not acceptable behaviour;
  reject with a clear message.

---

## 14. README.md contents

Short, in this order:

1. What the tool does, in two sentences, including that it produces a filtered subset
   and never modifies the source folder.
2. `install.bat`, then `run.bat <folder> --report`, then
   `run.bat <folder> --threshold 6 --emit-toml`. State plainly that `--report` first is
   the intended workflow, because rejections leave no artifact on disk.
3. The full option table from section 8.
4. **Why the output dimensions look arbitrary.** This is the question every user will
   ask. Explain that 1184×880 rather than 1168×880 is not a rounding error but an exact
   match to musubi-tuner's generated bucket list, and that any other value causes the
   trainer to re-crop.
5. A warning not to set `bucket_no_upscale = true` in the training TOML, with the
   one-line reason: it bypasses the bucket list and gives each image its own dimensions
   floored to 16.
6. How to read the bucket distribution section of the report and what the warnings mean.
7. A note that the D metric bands are uncalibrated and should be checked against a
   `--report` histogram before `--threshold` is trusted to act on them.
