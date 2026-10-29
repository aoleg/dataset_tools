#!/usr/bin/env python3
"""
Normalize the images of a dataset, in place or into an output folder:
downscale every image larger than <pixels>^2 to that area, and bring every
image to one format.

An image is downscaled when its width times its height is more than
<pixels> * <pixels>; it keeps its aspect ratio and gets the largest size whose
area is not more than that (2048 gives at most 4,194,304 pixels: a 3:2 photo
becomes 2508x1672). Smaller images are never upscaled.

--format decides the format of the images it writes:

    jpg [Q]   (the default, jpg 97) a downscaled image is saved as JPEG of
              quality Q, whatever its format was; an image in another format
              (PNG, WebP, ...) is re-encoded as JPEG too, even when it is not
              downscaled. Its name becomes <stem>.jpg and the original file is
              deleted. A JPEG that is not downscaled is left as it is.
    png       the same with PNG: a downscaled image is saved as PNG, an image
              in another format is re-encoded as PNG.
    original [Q]
              only downscaled images are written, each in its own format and
              under its own name (quality Q for JPEG, WebP and AVIF, default
              97). No image changes its format.

JPEG is written without chroma subsampling (4:4:4). Transparency is flattened
onto white for JPEG. The EXIF orientation of a written image is applied to its
pixels, so it looks as before in every viewer and trainer; its colour profile
is kept. Caption files are not touched: the stem of an image never changes.

Not scanned: folders whose name starts with "." or is "_controls" (the face
masks, which AI Toolkit fits to the image itself), and every folder that holds
the "_dataset_tools.json" mark of the backup and output folders of the other
dataset tools. When an image would be re-encoded to a name that another image
in its folder already has or would get (a.png next to a.jpg), it is skipped
with a warning.

In place there is no undo: the originals are overwritten or deleted. The tool
asks before it starts; --dry-run shows what would change and writes nothing.

With -o <folder> the source folder is not changed and nothing is asked: every
image goes to the output folder under its relative path, written by the same
rules, or copied as it is when it needs no change (a JPEG within the size with
--format jpg). The files next to an image with its stem and another extension
(its captions: .txt, .short, ...) are copied with it. Files of the same name
in the output folder are replaced.

Usage:    python normalize.py <folder> <pixels> [-o <folder>]
          [--format original [Q]|jpg [Q]|png] [--dry-run] [--yes] [--workers N]
Install:  Pillow, pillow-heif (install.bat)
"""
import argparse
import math
import os
import secrets
import shutil
import stat
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:              # HEIC/HEIF files then fail to open and are reported
    pass

Image.MAX_IMAGE_PIXELS = None    # large scans are normal input, not an attack

IMAGE_EXTS = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff",
              ".gif", ".avif", ".heic", ".heif"}
JPEG_EXTS = {".jpg", ".jpeg", ".jpe", ".jfif"}
JPEG_FORMATS = {"JPEG", "MPO"}   # MPO: the multi-picture JPEG of some cameras
LOSSY_FORMATS = {"JPEG", "WEBP", "AVIF", "HEIF"}
DEFAULT_QUALITY = 97
MIN_PIXELS = 64
SKIP_DIRS = {"_controls"}        # face masks; AI Toolkit resizes a mask to its image
BACKUP_MARK = "_dataset_tools.json"  # marks a folder the dataset tools made next to a dataset: never scanned
TMP_PREFIX = ".normalize-"       # a file is written under this name, then moved over the target
ROTATED = {5, 6, 7, 8}           # EXIF orientations that swap width and height
EXIF_ORIENTATION = 0x0112
DEFAULT_WORKERS = 8              # one large image in memory per worker
LIST_LIMIT = 20                  # lines of any list printed
PROMPT = "This will overwrite existing images. Continue? y/N "


# --- options ------------------------------------------------------------------

def parse_format(tokens):
    """--format tokens -> (kind, quality); kind is "original", "jpg" or "png"."""
    kind = tokens[0].lower()
    if kind == "jpeg":
        kind = "jpg"
    if kind not in ("original", "jpg", "png"):
        raise ValueError(f"--format must be original, jpg or png, not {tokens[0]!r}")
    if len(tokens) > 2 or (kind == "png" and len(tokens) > 1):
        extra = " ".join(tokens[1:])
        raise ValueError(f"--format {kind} takes {'no quality' if kind == 'png' else 'one quality'}, "
                         f"not {extra!r} (put the folder and the pixels first)")
    quality = DEFAULT_QUALITY
    if len(tokens) == 2:
        try:
            quality = int(tokens[1])
        except ValueError:
            raise ValueError(f"the quality must be a number from 1 to 100, not {tokens[1]!r}") from None
        if not 1 <= quality <= 100:
            raise ValueError(f"the quality must be from 1 to 100, not {quality}")
    return kind, quality


def describe(kind, quality, pixels, out_root):
    area = pixels * pixels
    lines = [f"Images over {pixels}x{pixels} = {area:,} pixels are downscaled to that area."]
    if kind in ("jpg", "png"):
        ext = "." + kind
        q = f", quality {quality}" if kind == "jpg" else ""
        lines.append(f"Downscaled images and images in another format are saved as {ext}{q};")
        if out_root:
            lines.append("the other images are copied as they are.")
        else:
            lines.append(f"an image that was not a {ext} is deleted after its {ext} is written.")
    else:
        lines.append(f"Downscaled images keep their format and name (quality {quality} for lossy formats);")
        lines.append("the other images are copied as they are." if out_root
                     else "images that need no downscale are not touched.")
    if out_root:
        lines.append(f"Output: {out_root}, with the captions of each image; the source folder is not changed.")
    return "\n".join(lines)


# --- scan ---------------------------------------------------------------------

def scan(root):
    """The image files under root, the set of (folder, stem) pairs that more
    than one image shares, the other files of an image's (folder, stem) pair
    (its captions), and the folders not scanned."""
    files, others, skipped_dirs = [], [], []
    for dirpath, dirnames, filenames in os.walk(root):
        keep = []
        for d in sorted(dirnames):
            full = os.path.join(dirpath, d)
            if d.startswith(".") or d in SKIP_DIRS or os.path.exists(os.path.join(full, BACKUP_MARK)):
                skipped_dirs.append(full)
            else:
                keep.append(d)
        dirnames[:] = keep
        for name in sorted(filenames):
            if name.startswith(TMP_PREFIX):
                continue
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                files.append(os.path.join(dirpath, name))
            elif name != BACKUP_MARK:
                others.append(os.path.join(dirpath, name))
    seen, shared = set(), set()
    for path in files:
        key = stem_key(path)
        (shared if key in seen else seen).add(key)
    sidecars = {}
    for path in others:
        if stem_key(path) in seen:
            sidecars.setdefault(stem_key(path), []).append(path)
    return files, shared, sidecars, skipped_dirs


def stem_key(path):
    return os.path.normcase(os.path.splitext(path)[0])


# --- one image ----------------------------------------------------------------

def target_size(w, h, area):
    """The largest size with the aspect ratio of w x h and at most area pixels."""
    scale = math.sqrt(area / (w * h))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    while nw * nh > area:        # float rounding at the edge
        if nw >= nh:
            nw -= 1
        else:
            nh -= 1
    return nw, nh


def scale_16bit(img):
    """16-bit or 32-bit integer grey to 8-bit L (convert("L") clips instead)."""
    return img.convert("I").point(lambda v: v / 256).convert("L")


def scale_float(img):
    lo, hi = img.getextrema()
    span = max(hi - lo, 1e-6)
    return img.point(lambda v: (v - lo) * 255.0 / span).convert("L")


def over_white(img):
    rgba = img.convert("RGBA")
    bg = Image.new("RGB", rgba.size, (255, 255, 255))
    bg.paste(rgba, mask=rgba.getchannel("A"))
    return bg


def for_resize(img):
    """A mode that Lanczos resizes correctly."""
    if img.mode == "P":
        return img.convert("RGBA" if "transparency" in img.info else "RGB")
    if img.mode == "1":
        return img.convert("L")
    if img.mode.startswith("I;16"):
        return img.convert("I")
    if img.mode == "PA":
        return img.convert("RGBA")
    return img


def for_format(img, fmt):
    """A mode that fmt can store."""
    mode = img.mode
    if mode.startswith("I;16") or mode == "I":
        if fmt == "PNG":         # 16-bit grey stays 16-bit; Pillow writes I;16 only, not I
            return img if mode.startswith("I;16") else img.convert("I;16")
        img, mode = scale_16bit(img), "L"
    elif mode == "F":
        img, mode = scale_float(img), "L"
    if fmt == "JPEG":
        if mode in ("RGB", "L"):
            return img
        if mode in ("RGBA", "LA", "PA") or (mode == "P" and "transparency" in img.info):
            return over_white(img)
        return img.convert("RGB")
    if fmt == "PNG":
        if mode in ("1", "L", "LA", "P", "RGB", "RGBA"):
            return img
        return img.convert("RGBA" if "A" in mode else "RGB")
    if fmt in ("WEBP", "AVIF", "HEIF"):
        if mode in ("RGB", "RGBA"):
            return img
        return img.convert("RGBA" if "A" in mode or "transparency" in img.info else "RGB")
    return img


def save_options(fmt, quality, icc):
    kw = {"icc_profile": icc} if icc else {}
    if fmt == "JPEG":
        kw.update(quality=quality, subsampling=0, optimize=True)
    elif fmt in ("WEBP", "AVIF", "HEIF"):
        kw.update(quality=quality)
    elif fmt == "TIFF":
        kw.update(compression="tiff_lzw")
    return kw


def make_writable(path):
    """A read-only file cannot be replaced or deleted on Windows."""
    try:
        if not os.access(path, os.W_OK):
            os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
    except FileNotFoundError:
        pass


def copy_file(src, dst):
    """src to dst through a temporary file, with its modification time."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = os.path.join(os.path.dirname(dst), f"{TMP_PREFIX}{secrets.token_hex(4)}.tmp")
    try:
        shutil.copyfile(src, tmp)
        st = os.stat(src)
        os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))
        make_writable(dst)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def process(path, shared, area, kind, quality, dry_run, root, out_root):
    """Downscale and/or re-encode one image, in place (out_root None) or into
    out_root. Returns a dict: status is "unchanged" (copied as it is into
    out_root), "written", "skipped" or "failed"."""
    def dest(p):
        return p if out_root is None else os.path.join(out_root, os.path.relpath(p, root))

    res = {"path": path, "status": "unchanged", "downscaled": False, "converted": False,
           "target": dest(path), "message": "", "before": 0, "after": 0}
    tmp = None
    try:
        res["before"] = res["after"] = os.path.getsize(path)
        with Image.open(path) as im:
            fmt = im.format or ""
            w, h = im.size
            if im.getexif().get(EXIF_ORIENTATION) in ROTATED:
                w, h = h, w
            downscale = w * h > area
            if kind == "original":
                out_fmt = "JPEG" if fmt in JPEG_FORMATS else fmt
                same = True
            elif kind == "jpg":
                out_fmt, same = "JPEG", fmt in JPEG_FORMATS
            else:
                out_fmt, same = "PNG", fmt == "PNG"
            animated = kind == "original" and getattr(im, "n_frames", 1) > 1
            if animated and out_root is None:
                res.update(status="skipped", message="animated, left as it is with --format original")
                return res
            if (not downscale and same) or animated:
                if out_root is not None and not dry_run:   # into out_root a plain copy
                    im.close()
                    copy_file(path, res["target"])
                return res

            ext = os.path.splitext(path)[1]
            if same or (out_fmt == "JPEG" and ext.lower() in JPEG_EXTS) or (out_fmt == "PNG" and ext.lower() == ".png"):
                target = path
            else:
                target = os.path.splitext(path)[0] + (".jpg" if out_fmt == "JPEG" else ".png")
            renamed = os.path.normcase(target) != os.path.normcase(path)
            # in out_root a file of the target name is an earlier output and is replaced
            if renamed and (stem_key(path) in shared or (out_root is None and os.path.exists(target))):
                res.update(status="skipped", target=dest(target), after=0,
                           message=f"another image has the name {os.path.basename(target)} or its stem")
                return res
            target = dest(target)
            res.update(status="written", downscaled=downscale, converted=not same, target=target)
            if downscale:
                res["size"] = target_size(w, h, area)
            if dry_run:
                return res

            im.seek(0)
            im.load()
            icc = im.info.get("icc_profile")
            img = ImageOps.exif_transpose(im)
        if downscale:
            img = for_resize(img).resize(res["size"], Image.Resampling.LANCZOS, reducing_gap=3.0)
        img = for_format(img, out_fmt)

        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp = os.path.join(os.path.dirname(target), f"{TMP_PREFIX}{secrets.token_hex(4)}.tmp")
        img.save(tmp, out_fmt, **save_options(out_fmt, quality, icc))
        res["after"] = os.path.getsize(tmp)
        make_writable(target)
        os.replace(tmp, target)
        tmp = None
        if renamed and out_root is None:
            make_writable(path)
            os.remove(path)
        return res
    except UnidentifiedImageError:
        res.update(status="failed", after=0, message="not an image, or a format Pillow cannot read")
        return res
    except Exception as e:  # one bad file must not stop a dataset
        res.update(status="failed", after=0, message=f"{type(e).__name__}: {e}")
        return res
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# --- main ---------------------------------------------------------------------

def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"


def print_list(title, items):
    if not items:
        return
    print(f"\n{title} ({len(items)}):")
    for line in items[:LIST_LIMIT]:
        print(f"  {line}")
    if len(items) > LIST_LIMIT:
        print(f"  ... and {len(items) - LIST_LIMIT} more")


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="the dataset folder; all its subfolders are scanned")
    ap.add_argument("pixels", type=int, help="images over pixels x pixels in area are downscaled to that area")
    ap.add_argument("-o", "--output", metavar="FOLDER",
                    help="write into this folder, with the captions, and leave the source folder as it is")
    ap.add_argument("--format", nargs="+", default=["jpg", str(DEFAULT_QUALITY)], metavar="FORMAT",
                    help="original [Q] | jpg [Q] | png (default: jpg 97)")
    ap.add_argument("--dry-run", action="store_true", help="show what would change; write nothing")
    ap.add_argument("-y", "--yes", action="store_true", help="do not ask before overwriting")
    ap.add_argument("--workers", type=int, default=min(DEFAULT_WORKERS, os.cpu_count() or 1),
                    help=f"images processed at once (default {DEFAULT_WORKERS} or the CPU count)")
    args = ap.parse_args(argv)

    try:
        kind, quality = parse_format(args.format)
    except ValueError as e:
        ap.error(str(e))
    if args.pixels < MIN_PIXELS:
        ap.error(f"pixels must be at least {MIN_PIXELS}")
    root = os.path.abspath(args.folder)
    if not os.path.isdir(root):
        print(f"Not a folder: {root}", file=sys.stderr)
        return 1
    if os.path.exists(os.path.join(root, BACKUP_MARK)):
        print(f"{root} holds {BACKUP_MARK}: it is a backup or output folder of the dataset tools. Stopped.",
              file=sys.stderr)
        return 1
    out_root = None
    if args.output:
        out_root = os.path.abspath(args.output)
        a, b = os.path.normcase(root), os.path.normcase(out_root)
        if a == b:
            print("The output folder is the source folder; leave -o out to normalize in place.", file=sys.stderr)
            return 1
        try:
            common = os.path.commonpath([a, b])
        except ValueError:                         # on different drives
            common = None
        if common in (a, b):                       # one inside the other: a run would read its own output
            print(f"The output folder and the source folder must not be inside each other:\n"
                  f"  {root}\n  {out_root}", file=sys.stderr)
            return 1
        if os.path.exists(out_root) and not os.path.isdir(out_root):
            print(f"The output is a file, not a folder: {out_root}", file=sys.stderr)
            return 1
        if os.path.exists(os.path.join(out_root, BACKUP_MARK)):
            print(f"{out_root} holds {BACKUP_MARK}: it is a backup or output folder of the dataset tools. Stopped.",
                  file=sys.stderr)
            return 1

    print(f"Scanning {root} ...")
    files, shared, sidecars, skipped_dirs = scan(root)
    print(f"{len(files):,} images found.")
    for d in skipped_dirs[:LIST_LIMIT]:
        print(f"  not scanned: {d}")
    if not files:
        return 0
    print(describe(kind, quality, args.pixels, out_root))

    if args.dry_run:
        print("Dry run: nothing is written.")
    elif not args.yes and out_root is None:
        try:
            answer = input(PROMPT)
        except EOFError:
            answer = ""
        if answer.strip().lower() != "y":
            print("Cancelled. Nothing was changed.")
            return 0

    area = args.pixels * args.pixels
    results = []
    tty = sys.stdout.isatty()
    lead, end = ("\r", "") if tty else ("", "\n")  # one progress line on a console, a line each in a log
    t0 = last = time.monotonic()
    ex = ThreadPoolExecutor(max_workers=max(1, args.workers))
    try:
        futures = [ex.submit(process, p, shared, area, kind, quality, args.dry_run, root, out_root)
                   for p in files]
        for fut in as_completed(futures):
            results.append(fut.result())
            now = time.monotonic()
            if now - last >= (0.5 if tty else 30) or len(results) == len(files):
                last = now
                print(f"{lead}{len(results):,}/{len(files):,} images", end=end, flush=True)
        if tty:
            print()
    except KeyboardInterrupt:
        print("\nStopping: the images in progress are finished, the rest are not touched.")
        ex.shutdown(wait=True, cancel_futures=True)
        print(f"Stopped after {len(results):,} of {len(files):,} images.")
        return 1
    ex.shutdown()

    # The captions of every image that reached out_root, once per stem, here
    # and not in the workers: two images of one stem share them.
    caption_count, caption_failed = 0, []
    if out_root is not None:
        done = {stem_key(r["path"]) for r in results if r["status"] in ("written", "unchanged")}
        for key in sorted(done):
            for src in sidecars.get(key, []):
                caption_count += 1
                if args.dry_run:
                    continue
                try:
                    copy_file(src, os.path.join(out_root, os.path.relpath(src, root)))
                except OSError as e:
                    caption_failed.append({"path": src, "message": f"{type(e).__name__}: {e}"})

    rel = lambda p: os.path.relpath(p, root)
    written = [r for r in results if r["status"] == "written"]
    down = sum(r["downscaled"] for r in written)
    conv = sum(r["converted"] for r in written)
    unchanged = sum(r["status"] == "unchanged" for r in results)
    skipped = sorted((r for r in results if r["status"] == "skipped"), key=lambda r: r["path"])
    failed = sorted([r for r in results if r["status"] == "failed"] + caption_failed, key=lambda r: r["path"])

    verb = "would be written" if args.dry_run else "written"
    print(f"\n{len(written):,} images {verb}: {down:,} downscaled, {conv:,} re-encoded to another format.")
    if out_root is None:
        print(f"{unchanged:,} unchanged, {len(skipped):,} skipped, {len(failed):,} failed"
              f" ({time.monotonic() - t0:.1f} s).")
    else:
        copied = "would be copied" if args.dry_run else "copied"
        print(f"{unchanged:,} images {copied} as they are, {caption_count:,} captions {copied}.")
        print(f"{len(skipped):,} skipped, {len(failed):,} failed ({time.monotonic() - t0:.1f} s).")
    if not args.dry_run and written:
        before = sum(r["before"] for r in written)
        after = sum(r["after"] for r in written)
        print(f"The written images went from {human(before)} to {human(after)}.")
    if args.dry_run:
        examples = []
        for r in sorted(written, key=lambda r: r["path"]):
            what = []
            if r["downscaled"]:
                what.append("downscale to {}x{}".format(*r["size"]))
            if r["converted"]:
                what.append(f"re-encode as {os.path.basename(r['target'])}")
            examples.append(f"{rel(r['path'])}: {', '.join(what)}")
        print_list("Planned", examples)
    print_list("Skipped", [f"{rel(r['path'])}: {r['message']}" for r in skipped])
    print_list("Failed", [f"{rel(r['path'])}: {r['message']}" for r in failed])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
