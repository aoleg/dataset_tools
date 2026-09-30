# Face masks and face crops

Two tools that use the same YOLOv8 face detector:

- `make_face_masks.py` (`run.bat`) makes per-image loss masks that hide faces from LoRA/LoKr training in [Ostris AI Toolkit](https://github.com/ostris/ai-toolkit). Point it at a folder of images; it writes a matching folder of mask PNGs.
- `face_extract.py` (`extract.bat`) cuts close-up face crops out of a folder of images and sorts them into 512, 768 and 1024 bucket folders, to make a face dataset for character training. See [face_extract.py: close-up face crops](#face_extractpy-close-up-face-crops).

## What the masks are for

When a dataset shows the same person over and over, the trained adapter learns that person's face along with whatever you actually wanted it to learn (an outfit, a style, a product). AI Toolkit can weight the training loss per pixel with a mask image, so regions painted dark contribute little or nothing to the gradient. This script produces those masks automatically: white everywhere, black over each detected face.

The images themselves are untouched. The model still sees the face as input context; it just isn't rewarded for reproducing it.

## How it works

1. Loads a YOLOv8 face detector (`face_yolov8m.pt` from the `Bingsu/adetailer` repo on Hugging Face; downloaded once and cached).
2. For each image in the input folder, runs the detector and gets one bounding box per face.
3. Creates a white mask the size of the image and draws a black ellipse over each face box, enlarged by `--grow` (default 1.35, so a 35 % margin). With `--include-hair` the ellipse is also extended upward to cover hair.
4. Blurs the mask edge by `--feather` pixels (default 12) so the transition from "ignored" to "trained" is gradual rather than a hard line across the collar.
5. Saves `<mask_dir>/<image basename>.png` (`<mask_dir>` defaults to `<img_dir>\masks`). Images with no detected face get an all-white mask and are listed at the end so you can check them.

With `--preview <dir>` it also writes a copy of each image with the masked area tinted red, which is the quickest way to verify the detector did what you expect.

## Install

Run `install.bat`. It needs Python 3.10 or newer on PATH (`python`, or `py -3.12` from the launcher).

It creates a `venv` folder next to the script and installs `torch==2.13.0` and `torchvision` from the PyTorch CUDA 13.2 index (`https://download.pytorch.org/whl/cu132`). It does not install torch from PyPI, because the PyPI torch for Windows is CPU only. Next it installs `requirements.txt` from PyPI. While it does this, it pins the installed torch and torchvision builds as a pip constraint, so the torch dependency of `ultralytics` cannot replace them. Then it prints the torch and CUDA versions and fails if torch is not a CUDA build.

Last, it downloads the default face detector `face_yolov8m.pt` from `Bingsu/adetailer` and loads it once as a check. The model goes into the Hugging Face cache (`%USERPROFILE%\.cache\huggingface\hub`, or `HF_HOME` if set). ADetailer uses the same cache, so a copy it already downloaded is reused. When `--model` names another file, the script downloads that file on its first run.

After that, the scripts do not use the network. They look for the model in the cache first and contact Hugging Face only when the file is not there. They also set `YOLO_OFFLINE=1` before they load ultralytics, which stops its online check and the usage analytics it sends when it is online.

Running `install.bat` again updates the packages in the existing venv.

To use the AI Toolkit virtual environment instead and avoid a second copy of PyTorch, activate that environment and run `pip install -r requirements.txt`. The script only needs `ultralytics` and `huggingface_hub` on top of what AI Toolkit already has.

## Project files

| file | purpose |
|---|---|
| `install.bat` | creates the venv and installs the dependencies |
| `activate.bat` | activates the venv in the current cmd window; started by double-click, it opens a new cmd window with the venv active |
| `run.bat` | runs `make_face_masks.py` with the venv Python and passes all arguments through |
| `extract.bat` | runs `face_extract.py` in the same way |
| `make_face_masks.py` | the mask tool; also holds the detector code that `face_extract.py` uses |
| `face_extract.py` | the face crop tool |
| `requirements.txt` | the PyPI dependencies; torch and torchvision are not in it |

`install.bat` always pauses at the end. `run.bat` and `extract.bat` pause at the end when they are started by double-click or drag-and-drop, or when the script fails. PowerShell starts a `.bat` file in the same way as Explorer, so they also pause there. Set `NOPAUSE=1` to prevent this.

## Mask usage

```
run.bat <img_dir> [<mask_dir>] [options]
```

This is the same as `python make_face_masks.py ...` with the venv active.

If you do not give `<mask_dir>`, the masks go into a `masks` subfolder of `<img_dir>`, which the script creates if it does not exist. You can also drop an image folder onto `run.bat` in Explorer to get this result.

Examples:

```
run.bat D:\data\studio
run.bat D:\data\studio D:\data\studio_masks --preview D:\data\studio_preview --include-hair
```

The first command writes the masks to `D:\data\studio\masks`.

Options:

| option | default | meaning |
|---|---|---|
| `--grow` | 1.35 | scale factor applied to each detected face box |
| `--feather` | 12 | Gaussian blur radius (px) on the mask edge |
| `--conf` | 0.3 | detector confidence threshold; lower finds more faces, with more false positives |
| `--model` | `face_yolov8m.pt` | which adetailer face model to use (`n`, `s`, `m` variants exist) |
| `--include-hair` | off | extend the ellipse upward to cover hair |
| `--invert` | off | swap the colours: white ellipses over the faces on a black background, so only the faces are trained |
| `--preview DIR` | none | write red-tinted overlay images for checking; the red area is the masked area, so with `--invert` it is everything except the faces |
| `--overwrite` | off | regenerate masks that already exist |

## Using the masks in AI Toolkit

In the dataset entry of your job config:

```yaml
datasets:
  - folder_path: D:\data\studio
    mask_path: D:\data\studio_masks
    mask_min_value: 0.2
```

With the default mask location, set `mask_path: D:\data\studio\masks`.

AI Toolkit finds each mask by the image's basename inside `mask_path` (any image extension), converts it to grayscale, resizes it to the training bucket, and remaps it so that white = 1.0 and black = `mask_min_value`. A `mask_min_value` of 0 removes the face from the loss entirely; 0.1–0.2 leaves a little signal so the model keeps learning that a face belongs there without learning whose it is.

With `--invert` the same setting applies to everything except the faces. This is for training a character's face while the model learns little of the clothes, background and body in the dataset. An image with no detected face gets an all-black mask, so with a `mask_min_value` of 0 nothing in it is trained. The tool lists these images at the end.

## Notes

- Masks are aligned to the original image. If you crop or resize the images later, regenerate the masks.
- Face detection is not identity-aware: it masks every face, including background people. For most subject datasets that is what you want.
- Hair is part of identity too. If the two people in your set have distinctive hairstyles and the hairstyle is not part of what you are training, use `--include-hair`.
- Check the previews once. Typical misses are faces in profile at small size and faces partly covered by hats or hands; raise `--grow` or lower `--conf` if those matter.

## face_extract.py: close-up face crops

```
extract.bat <img_dir> [<out_dir>] [options]
```

For every face found in `<img_dir>`, it writes one crop to `<out_dir>\<tier>\<image basename>_face<N>.png`. `<out_dir>` defaults to `<img_dir>\faces`, and `<tier>` is `512`, `768` or `1024`. `N` counts the faces of one image from the largest down. You can also drop an image folder onto `extract.bat` in Explorer.

Example:

```
extract.bat D:\data\alice --largest-only
```

This writes to `D:\data\alice\faces\512`, `D:\data\alice\faces\768` and `D:\data\alice\faces\1024`.

### How a crop is made

1. The detector finds each face. The face box is grown by `--grow` and extended upward and sideways for hair, as `make_face_masks.py --include-hair` does. This grown box is the "face area".
2. The crop gets a bucket aspect ratio. With `--ratio auto`, the tool tries 2:3, 4:5 and 1:1 first, closest to the shape of the detected face first. It uses 9:16, 5:4, 3:2 and 16:9 only when a face cannot fit the first three, for example near the edge of a wide, low image.
3. The crop size makes the face area cover `--fill` of the crop (default 0.5, a head and a little of the shoulders). The crop is always at least as large as the face area. The face centre sits 42% down from the top of the crop, and the crop is shifted to stay inside the image.
4. The tier is the largest of 1024, 768 and 512 whose bucket the crop can fill. As in k2prep, a crop may be up to 1.15 times smaller than the bucket on each side. So `1024` holds every face whose crop reaches the 1024 bucket, however much larger it is.
5. If the crop is too small for the 512 bucket, the cut is made looser until it is large enough for 512. The face then covers less of the crop.
6. The face is skipped if that is impossible: the image is too small to hold a 512 crop, or the face would cover less than `--min-fill` of the crop (default 0.25). Other faces in the same image are still processed.
7. The crop is resized to the exact bucket size with Lanczos and saved.

The buckets are the musubi-tuner buckets that k2prep uses, for example 912×1136 (4:5), 832×1248 (2:3) and 1024×1024 (1:1) at 1024. `face_extract.py` holds a copy of the k2prep bucket code, which gives the same tables.

At the end the tool prints the number of crops per tier, each skipped face with the reason, and the images with no face.

### Options

| option | default | meaning |
|---|---|---|
| `--fill` | 0.5 | preferred share of the crop covered by the face area; larger is a tighter close-up |
| `--min-fill` | 0.25 | skip a face when the looser cut needed for 512 would drop its share below this |
| `--ratio` | auto | bucket aspect ratio: `auto`, or one of `9:16`, `2:3`, `4:5`, `1:1`, `5:4`, `3:2`, `16:9` |
| `--grow` | 1.35 | scale factor applied to each detected face box |
| `--conf` | 0.5 | detector confidence threshold; higher than the mask tool, because a false detection here becomes a bad training image |
| `--model` | `face_yolov8m.pt` | which adetailer face model to use |
| `--largest-only` | off | keep only the largest face of each image |
| `--allow-others` | off | keep crops that also contain half or more of another detected face; by default such crops are skipped, so that another person's face does not enter the dataset |
| `--keep-size` | off | save the crop at its source pixel size, not resized to the bucket |
| `--format` | png | `png` or `jpg` (quality 95) |
| `--overwrite` | off | process images that already have crops again; their old crops are deleted first, in all tier folders |

Without `--overwrite`, an image that already has a crop in any tier folder is not processed again. An image whose faces were all skipped has no crop, so it is processed again on the next run.

### Notes

- The detector does not know identity. In a group photo it finds every face, and each gets a crop if it is large enough. For a character dataset, use `--largest-only` or delete the other people's crops by hand.
- Captions are not copied. A caption of the full image does not describe the close-up.
- Faces in profile have narrow boxes and usually get a 2:3 crop.
