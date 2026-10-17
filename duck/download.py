r"""Download the images listed in a results CSV with img2dataset.

    download.bat results\pixelprose\lighthouse-0.csv [options]

Images land in <downloads>/<csv name>/ (downloads folder from duck.conf), one
file per image plus a .txt holding the caption from the CSV's text column.
Re-running on the same CSV resumes: shards that already finished are skipped.

--encode decides what is written:
  original  the downloaded bytes are stored untouched and never decoded. Default.
            Files are named by their real content (.jpg, .png, .gif, .webp, .bmp).
            Aliases: 0, off, no, none, skip.
  jpg       JPEG sources are stored byte for byte; anything else (PNG, WebP, GIF)
            is re-encoded to JPEG at quality 95.
  png       every image is decoded and written as PNG. Lossless from the decoded
            pixels, but a JPEG grows about ten times.
--resize and --min-size need the image decoded, so they switch original to jpg.
"""

import argparse
import configparser
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Albumentations phones home on import to check for updates; keep it quiet.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

ENCODE_ALIASES = {"0": "original", "off": "original", "no": "original", "none": "original", "skip": "original"}

# img2dataset names every raw file after encode_format, so with --encode original
# a PNG would be saved as .jpg. Sniff the first bytes and rename.
MAGIC = [
    (b"\xff\xd8\xff", "jpg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
]


def sniff(path):
    with open(path, "rb") as f:
        head = f.read(16)
    for magic, ext in MAGIC:
        if head.startswith(magic):
            return ext
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    return None


def rename_by_content(folder):
    """Give every raw .jpg under folder the extension its bytes call for."""
    renamed = 0
    unknown = 0
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if not name.lower().endswith(".jpg"):
                continue
            path = os.path.join(root, name)
            ext = sniff(path)
            if ext is None:
                unknown += 1
                continue
            if ext != "jpg":
                os.replace(path, os.path.splitext(path)[0] + "." + ext)
                renamed += 1
    print(f"Renamed {renamed} file(s) to match their content"
          + (f"; {unknown} left as .jpg (format not recognised)" if unknown else "") + ".")


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def downloads_root():
    cp = configparser.ConfigParser(interpolation=None)
    cp.read(os.path.join(HERE, "duck.conf"), encoding="utf-8")
    value = cp.get("output", "downloads", fallback="downloads").strip().strip('"') or "downloads"
    return value if os.path.isabs(value) else os.path.join(HERE, value)


def main():
    parser = argparse.ArgumentParser(
        prog="download.bat",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv", help="A results .csv written by run.bat (columns url, text, ...).")
    parser.add_argument("--out", default=None, help="Output folder (default: <downloads>/<csv name>).")
    parser.add_argument("--encode", default="original",
                        choices=["original", "jpg", "png"] + sorted(ENCODE_ALIASES),
                        metavar="{original,jpg,png}",
                        help="How images are stored (default: original, no re-encoding). See above.")
    parser.add_argument("--resize", type=int, default=0,
                        help="Shrink images whose long edge exceeds N px (default: 0 = never). Implies re-encoding.")
    parser.add_argument("--min-size", type=int, default=0,
                        help="Drop images smaller than N px on either edge (not with --encode original).")
    parser.add_argument("--threads", type=int, default=64, help="Parallel downloads (default: 64).")
    parser.add_argument("--processes", type=int, default=1, help="Worker processes (default: 1).")
    parser.add_argument("--timeout", type=int, default=10, help="Seconds per request (default: 10).")
    parser.add_argument("--retries", type=int, default=0, help="Retries per image (default: 0).")
    parser.add_argument("--format", default="files", choices=["files", "webdataset", "parquet"],
                        help="img2dataset output format (default: files, one image + one .txt each).")
    parser.add_argument("--shard-size", type=int, default=10000, help="Images per shard folder (default: 10000).")
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    if not os.path.isfile(csv_path):
        print(f"Error: not a file: {csv_path}")
        sys.exit(1)
    args.encode = ENCODE_ALIASES.get(args.encode, args.encode)
    if args.encode == "original" and (args.resize or args.min_size):
        print("Note: --resize/--min-size need the image decoded, so --encode original becomes jpg.")
        args.encode = "jpg"

    out = args.out or os.path.join(downloads_root(), os.path.splitext(os.path.basename(csv_path))[0])
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    from img2dataset import download  # slow import, so only after argument checks

    print(f"CSV    : {csv_path}")
    print(f"Images : {out}")
    print(f"Encode : {args.encode}" + (f", long edge <= {args.resize} px" if args.resize else ""))
    print(f"Threads: {args.threads} x {args.processes} process(es)")
    sys.stdout.flush()

    if args.encode == "original":
        image = dict(disable_all_reencoding=True, resize_mode="no", encode_format="jpg")
    else:
        quality = 95 if args.encode == "jpg" else 9
        if args.resize > 0:
            image = dict(image_size=args.resize, resize_mode="keep_ratio_largest", resize_only_if_bigger=True,
                         encode_format=args.encode, encode_quality=quality, skip_reencode=False)
        else:
            # skip_reencode keeps the bytes when the source already is the target
            # format, so with jpg a JPEG is stored as downloaded.
            image = dict(image_size=0, resize_mode="no",
                         encode_format=args.encode, encode_quality=quality, skip_reencode=True)

    download(
        url_list=csv_path,
        input_format="csv",
        url_col="url",
        caption_col="text",
        output_format=args.format,
        output_folder=out,
        processes_count=args.processes,
        thread_count=args.threads,
        timeout=args.timeout,
        retries=args.retries,
        number_sample_per_shard=args.shard_size,
        min_image_size=args.min_size,
        # An empty list disables the X-Robots-Tag check, which otherwise drops
        # images whose CDN sends "noai"/"noindex" headers.
        disallowed_header_directives=[],
        user_agent_token=USER_AGENT,
        incremental_mode="incremental",
        **image,
    )

    if args.encode == "original" and args.format == "files":
        rename_by_content(out)


if __name__ == "__main__":
    main()
