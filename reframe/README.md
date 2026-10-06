# Reframe

`reframe.py` trims photos of people. It finds the subject of each photo, one person or a group, leaves passers-by out, and crops the photo to one of the k2prep aspect ratios with margins around the subject. A standing person in a landscape photo becomes a portrait crop, and the other way round. Photos without a clear subject, such as crowds and street scenes, stay unchanged.

The tool is for old personal photos where the people fill only a small part of a large frame. The crops make better training images for LoRA and fine-tune training, and they work directly with k2prep.

## Install

On Windows, run `install.bat`. It needs Python 3.10 or newer and a CUDA GPU.

It creates the shared `..\venv` folder when it is missing and installs `torch` and `torchvision` from the PyTorch CUDA 13.2 index, never from PyPI, because the PyPI torch for Windows has no CUDA. Then it installs `requirements.txt` from PyPI, with the installed torch builds pinned so that `ultralytics` cannot replace them. Last, it downloads the three detection models:

| model | source | use |
|---|---|---|
| `yolo26x-seg.pt` | ultralytics | person outlines |
| `face_yolov8m.pt` | `Bingsu/adetailer` on Hugging Face | faces |
| `yolo26x-pose.pt` | ultralytics | body keypoints (head, shoulders, hips, knees, ankles) |

The ultralytics models go into `models` next to the script. The face model goes into the Hugging Face cache, where the face mask tool and ADetailer also find it. After the install, the tool does not use the network.

## Usage

```
run.bat <folder> [<folder> ...] [options]
```

Examples:

```
run.bat D:\photos\2016 --dry-run --verdicts
run.bat D:\photos\2016
run.bat D:\photos\2016 --resize
```

The first command plans the crops and draws a preview of each photo, without writing any photos. The second writes the crops. The third crops and resizes every photo into its k2prep bucket.

Each folder is processed with all its subfolders. Folders whose names start with `_` are skipped. The output goes to `<folder>\_reframed\` with the same relative paths. The photos in the folder itself are never changed.

| option | default | meaning |
|---|---|---|
| `--dry-run` | off | write only `plan.json` and the previews |
| `--verdicts` | off | draw the decision for each photo into `_reframed\_verdicts\` |
| `--resize` | off | crop and resize every photo into its k2prep bucket, sorted into tier folders |
| `--png` | off | with `--resize`: write PNG instead of JPEG |
| `--reencode` | off | crop JPEG photos by decoding and saving them again, not losslessly |
| `--ratios` | all seven | the aspect ratios to choose from, for example `--ratios 2:3,4:5,1:1,5:4,3:2` |
| `--skip-unchanged` | off | do not copy the photos that are not cropped |
| `--overwrite` | off | write all outputs again, also the ones that are up to date |
| `--out DIR` | `<folder>\_reframed` | another output folder; with one input folder only |
| `--previews` | off | draw the raw detections (outlines, faces, skeletons) into `_reframed\_preview\` |
| `--people` | off | draw the people with their measurements into `_reframed\_people\` |
| `--redetect` | off | run the detection again, without the cache |
| `--threads N` | 4 | number of parallel workers, 1 to 32 |

**Speed.** Detection runs one photo at a time on the GPU, because the models cannot run from two threads at once. Decoding, measuring and previews for other photos run in parallel threads meanwhile. The output files are written by parallel worker processes instead of threads, because the library that does the lossless JPEG crop can crash the program when it runs next to other threads. On 29 phone photos of 12 megapixels, a first run takes about 13 seconds with 4 workers and 24 seconds with 1, and a second run reuses the detections.

The aspect ratios are k2prep's: 9:16, 2:3, 4:5, 1:1, 5:4, 3:2 and 16:9.

`run.bat` pauses at the end when it is started by double-click or drag-and-drop, or when the script fails. Set `NOPAUSE=1` to prevent this.

## What is written

| output | when |
|---|---|
| the cropped photo, same name and format | the photo has a subject and a crop |
| a copy of the photo | the photo stays unchanged (not with `--skip-unchanged`) |
| the `.txt` caption, next to the output | the photo has a caption with the same name |

**JPEG photos are cropped losslessly.** The tool cuts whole 8×8 pixel blocks out of the compressed image. Nothing is decoded and saved again, so there is no quality loss. Away from the outer 16 pixels of the crop, every pixel is identical to the original. At the new edges, colours can differ by a few levels, because the colour smoothing no longer sees the pixels that were cut off. For a lossless crop, the top-left corner must be on the block grid of the photo, so the tool moves it up and left by up to 15 pixels and corrects the aspect ratio at the opposite edge. The aspect ratio is then correct to within about 0.1%.

The lossless crop keeps the EXIF data, including the orientation. The EXIF thumbnail is removed, because it shows the whole photo, and the image size in EXIF is set to the crop size. iPhone portrait photos (MPO files) are ordinary JPEG files with extra images appended, such as a depth map. The crop keeps the main image only.

**Other formats**, and JPEG photos with `--reencode`, are cropped from the upright image and saved in their own format. A JPEG keeps its quantization tables and its chroma subsampling, so its quality stays as it was. EXIF and the ICC profile stay, and the orientation is set to normal, because the pixels are already upright. PNG keeps its transparency.

**With `--resize`**, each crop, or each unchanged photo as a whole, is processed as k2prep processes a photo:

1. The nearest k2prep aspect ratio.
2. The largest of the tiers 1024, 768 and 512 whose bucket the crop fills, with an upscale of at most 1.15.
3. k2prep's crop to the exact bucket size: centred, with a third of the spare height above for portrait buckets.
4. One Lanczos resize from the original photo.
5. JPEG at quality 97 without chroma subsampling, or PNG with `--png`, without EXIF and ICC.

The output goes to `<tier>\<name>`, or `<subfolder>\<tier>\<name>` for photos in subfolders, as k2prep does. A name that is taken gets `-2`, `-3` and so on. A photo that is too small for the 512 tier is skipped. On photos that are not cropped, the output is pixel-identical to the output of k2prep itself.

**Next runs.** `_reframed\written.json` records what was written for each photo. A second run skips the photos that are done. It writes a photo again when the photo, its caption, the plan or the options changed. It removes outputs that are no longer wanted, for example the old copy of a photo that is now cropped. Each file is written under a temporary name and renamed when it is complete.

Other files in `_reframed`:

| file | content |
|---|---|
| `detections.json` | the detections and measurements of each photo; a second run reuses them |
| `plan.json` | the decision for each photo: the subject people, the reason, the crop box and the flags |
| `written.json` | the outputs of the last run |

## How the subject is chosen

**People.** Three models look at each photo: one finds person outlines, one finds faces, and one finds body keypoints. No model finds every person, so the tool merges the three into one list. A skeleton starts a person; an outline joins it when they overlap, and a face joins it when it sits on the skeleton's head. An outline or a face that joins nobody is a person of its own.

**Measurements.** For each person the tool measures the head size, the height of the feet in the photo, which body parts are visible, the image edges the person touches, how much the person faces the camera and in which direction, and how sharp the head is.

**Candidates.** A person is a candidate if the head is at least 40% the size of the largest head in the photo. A person who is turned away and touches an image edge is not a candidate: this is typically a passer-by close to the camera, or the photographer's shadow. A person who touches the left or right edge is a candidate only if the head is large and faces the camera.

**Groups.** Candidates form a group when they are at the same distance from the camera and close together, or when they overlap. Same distance means heads within a factor of 1.8 in size and feet at a similar height. Head size is a better measure than body height, because a child's head is almost as large as an adult's.

**The subject.** Each group gets the score of its best person: the head size relative to the largest head, more for facing the camera, less for touching a side edge or for being far from the centre. The group with the best score is the subject. A tall stranger close to the camera at the edge loses to a family in the centre, because he faces away and is cut by the frame. If a second group scores almost the same, both are kept and the photo is flagged.

**People in the group who are hidden.** An outline that lies mostly inside a member of the group, such as a person partly hidden behind a member, joins the group, so that the crop does not cut through it.

**The photo stays unchanged when:**

- the best group scores too low (no clear subject);
- the largest head in the group is smaller than 3.4% of the short side of the photo;
- it is a crowd: 10 or more people of similar size with nobody clearly larger, or a group of 8 or more people among 30 or more;
- the only subject is one person far from the centre, which is typical of a stranger who walks through the photo.

The rules were set on 29 photos where every person was labelled by eye, and they agree with all 27 labels that could be decided. A test on 12 further photos found one crowd seen from above that was cropped, and the rule for large groups among many people was added for it.

## How the crop is made

1. **Margins.** Around the subject group, in head sizes of the largest head: 0.7 above, 0.5 left and right, 0.3 below visible feet. A person who looks to one side gets more room on that side. If a person's legs are not visible and the person does not reach the bottom edge, the crop goes down to the bottom of the photo, because the body continues below the detection.
2. **Aspect ratio.** Of the allowed ratios, the one that holds the group with its margins in the smallest crop. If no ratio can hold the margins, the largest crop of each ratio is tried without margins, and the tool uses it if it keeps at least 98% of the group.
3. **Placement.** The highest head goes to the upper third of the crop where there is room, and the group is centred from left to right. Then the crop moves left, right, up or down so that its edges cut through as few passers-by as possible.
4. **Size.** A crop smaller than k2prep's 512 bucket is made larger. If the photo is too small for that, it stays unchanged.
5. **Savings.** If the crop would keep more than 90% of the photo, the photo stays unchanged.

## Corrections

Some decisions will be wrong. Correct them in a file `overrides.txt` in the photo folder, one photo per line:

```
# comments start with #
IMG_1234.jpg keep
IMG_1240.jpg crop
holiday\IMG_1302.jpg people=0,3
"a name with spaces.jpg" keep
```

| action | effect |
|---|---|
| `keep` | leave the photo unchanged |
| `crop` | crop the photo even though the rules leave it unchanged |
| `people=0,3` | the subject is exactly these people |

Paths are relative to the folder, and capital letters do not matter. The person numbers are the ones in the `--verdicts` previews. Run with `--verdicts` before you write `people=` lines, because the numbers can change when a new version of the tool merges the detections differently. The tool reports lines it cannot read and photos it cannot find.

## Previews

`--verdicts` draws each photo with its decision in the title. The subject group is green, other people are red, and every person has a number. If the photo is cropped, everything outside the crop is darker, and the crop shows its aspect ratio and size. Use these previews to check a folder before you write the crops, and to find the numbers for `overrides.txt`.

## Limits

- **Statues and pictures of people** are found as people. They can join a group or become the subject.
- **Unusual clothing.** People in large costumes or unusual clothing may get no outline. Their size then comes from the body keypoints, which do not include hats, headdresses or wide clothing, so the crop can cut these off.
- **Passers-by who overlap the subject** cannot be cropped out. The photo is flagged "cuts a passer-by".
- **The rules are tuned on family and travel photos** taken with phones. Other kinds of photos, such as stage events, sports or very wide group photos, may need corrections.
- **Lossless crops** keep the EXIF orientation tag. Viewers and tools that ignore this tag show the crop rotated, as they show the original. k2prep and current viewers apply the tag.
