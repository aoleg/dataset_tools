#!/usr/bin/env python3
"""classify.py - sort an image dataset into category folders.

The categories are given as folders of hand-sorted example images (the samples
root). Every image is embedded once with SigLIP 2, a classifier is trained on
the example embeddings, and every dataset image is copied (or moved) with its
caption into the matching class folder under the output root. Images the
classifier cannot place with confidence go to _unsure.

Usage:
    python classify.py --dataset <folder> [--dataset <folder> ...] --samples <folder> -o <folder> [options]
    python classify.py -o <folder> --undo
    python classify.py --fetch-models

Embeddings are cached in <root>\\_embeddings inside every dataset root and the
samples root. The run's own files (plan, report, undo log, sheets) go to
<output>\\_classify.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"

MODEL_REPO = "google/siglip2-so400m-patch16-naflex"
MODEL_FOLDER = MODEL_REPO.split("/")[-1]
MODEL_FILES = [
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
]
PATCH_SIZE = 16
EMBED_DIM = 1152

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".avif"}
DEFAULT_SIDECARS = ".txt"

EMBED_DIRNAME = "_embeddings"
RUN_DIRNAME = "_classify"
UNSURE_DIRNAME = "_unsure"
DEDUP_DIRNAME = "_duplicates"
# Skipped by exact name only. Dataset subfolders may start with "_" too.
SKIP_DIRNAMES = {EMBED_DIRNAME, RUN_DIRNAME, UNSURE_DIRNAME, DEDUP_DIRNAME}

CACHE_FLUSH_EVERY = 2000
WINDOW = 1024  # images handed to the decode pool at a time

MIN_EXAMPLES_WARN = 20
MIN_EXAMPLES_REFUSE = 5


# ----------------------------------------------------------------------------
# Scanning
# ----------------------------------------------------------------------------

@dataclass
class ImageItem:
    root: Path
    rel: str                 # path relative to root, forward slashes
    size: int
    mtime_ns: int
    sidecars: list[str] = field(default_factory=list)  # relative paths

    @property
    def path(self) -> Path:
        return self.root / self.rel


def scan_root(root: Path, sidecar_exts: set[str], skip_abs: set[Path] = frozenset()) -> list[ImageItem]:
    """Walk root recursively and return its images with their sidecars.

    Folders are skipped by exact name (SKIP_DIRNAMES) or by absolute path
    (skip_abs, used for the output root). Any other folder name is scanned,
    including names that start with an underscore.
    """
    root = root.resolve()
    skip_abs = {p.resolve() for p in skip_abs}
    items: list[ImageItem] = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in SKIP_DIRNAMES and (here / d).resolve() not in skip_abs
        )
        by_stem: dict[str, list[str]] = {}
        for fn in filenames:
            by_stem.setdefault(os.path.splitext(fn)[0], []).append(fn)
        for fn in sorted(filenames):
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in IMAGE_EXTS:
                continue
            p = here / fn
            try:
                st = p.stat()
            except OSError:
                continue
            rel_dir = here.relative_to(root).as_posix()
            rel = fn if rel_dir == "." else f"{rel_dir}/{fn}"
            sidecars = [
                (fn2 if rel_dir == "." else f"{rel_dir}/{fn2}")
                for fn2 in by_stem.get(stem, [])
                if os.path.splitext(fn2)[1].lower() in sidecar_exts
            ]
            items.append(ImageItem(root, rel, st.st_size, st.st_mtime_ns, sorted(sidecars)))
    return items


# ----------------------------------------------------------------------------
# Samples root
# ----------------------------------------------------------------------------

@dataclass
class SampleClass:
    name: str                              # folder name, e.g. "11_outliers"
    subclasses: dict[str, list[ImageItem]]  # subclass name -> examples

    @property
    def count(self) -> int:
        return sum(len(v) for v in self.subclasses.values())

    @property
    def is_outliers(self) -> bool:
        return strip_prefix(self.name) == "outliers"


def strip_prefix(name: str) -> str:
    """'11_outliers' -> 'outliers'; '_unsure' stays '_unsure'."""
    head, sep, tail = name.partition("_")
    return tail.lower() if sep and head.isdigit() and tail else name.lower()


def discover_samples(samples_root: Path, sidecar_exts: set[str]) -> list[SampleClass]:
    """Find the class folders and their sub-classes in the samples root."""
    samples_root = samples_root.resolve()
    items = scan_root(samples_root, sidecar_exts)
    classes: dict[str, SampleClass] = {}
    for it in items:
        parts = it.rel.split("/")
        if len(parts) < 2:
            continue  # an image directly in the samples root belongs to no class
        cls = parts[0]
        sub = parts[1] if len(parts) >= 3 else cls
        if len(parts) > 3:
            sub = parts[1]  # deeper levels fold into the first sub-class level
        sc = classes.setdefault(cls, SampleClass(cls, {}))
        sc.subclasses.setdefault(sub, []).append(it)
    ordered = sorted(classes.values(), key=lambda c: natural_key(c.name))
    return ordered


def natural_key(name: str):
    head, sep, tail = name.partition("_")
    if sep and head.isdigit():
        return (0, int(head), tail.lower())
    return (1, 0, name.lower())


def check_samples(classes: list[SampleClass]) -> None:
    if not classes:
        sys.exit("The samples root has no class folders with images.")
    print("Samples:")
    refuse = []
    for c in classes:
        subs = ", ".join(f"{s}: {len(v)}" for s, v in c.subclasses.items()) if len(c.subclasses) > 1 or next(iter(c.subclasses)) != c.name else ""
        flag = ""
        if c.count < MIN_EXAMPLES_REFUSE:
            flag = "  [too few, at least 5 are required]"
            refuse.append(c.name)
        elif c.count < MIN_EXAMPLES_WARN:
            flag = "  [warning: fewer than 20]"
        print(f"  {c.name}: {c.count}{flag}" + (f"  ({subs})" if subs else ""))
    if refuse:
        sys.exit("Fill these class folders with at least 5 images: " + ", ".join(refuse))


# ----------------------------------------------------------------------------
# Embedding cache
# ----------------------------------------------------------------------------

class EmbeddingCache:
    """<root>/_embeddings/index.json + embeddings.npy.

    An entry is reused when path, size, mtime, model id and patch budget match.
    """

    def __init__(self, root: Path, model_id: str, patches: int):
        self.root = root.resolve()
        self.dir = self.root / EMBED_DIRNAME
        self.model_id = model_id
        self.patches = patches
        self.entries: dict[str, tuple[int, int]] = {}   # rel -> (size, mtime_ns)
        self.rows: dict[str, int] = {}                  # rel -> row in self.emb
        self.emb = np.zeros((0, EMBED_DIM), dtype=np.float16)
        self._load()

    def _load(self) -> None:
        idx = self.dir / "index.json"
        npy = self.dir / "embeddings.npy"
        if not (idx.is_file() and npy.is_file()):
            return
        try:
            meta = json.loads(idx.read_text(encoding="utf-8"))
            emb = np.load(npy)
        except Exception as e:  # noqa: BLE001
            print(f"  cache in {self.dir} unreadable ({e}), starting empty")
            return
        if meta.get("model") != self.model_id or meta.get("patches") != self.patches:
            print(f"  cache in {self.dir} was made with another model or patch budget, starting empty")
            return
        files = meta.get("files", [])
        if emb.shape[0] != len(files) or emb.shape[1] != EMBED_DIM:
            print(f"  cache in {self.dir} is inconsistent, starting empty")
            return
        for row, (rel, size, mtime) in enumerate(files):
            self.entries[rel] = (size, mtime)
            self.rows[rel] = row
        self.emb = emb.astype(np.float16, copy=False)

    def has(self, it: ImageItem) -> bool:
        return self.entries.get(it.rel) == (it.size, it.mtime_ns)

    def get(self, it: ImageItem) -> np.ndarray:
        return self.emb[self.rows[it.rel]]

    def put(self, items: list[ImageItem], vecs: np.ndarray) -> None:
        new_rows = []
        for it, v in zip(items, vecs):
            if it.rel in self.rows:
                self.emb[self.rows[it.rel]] = v
            else:
                self.rows[it.rel] = self.emb.shape[0] + len(new_rows)
                new_rows.append(v)
            self.entries[it.rel] = (it.size, it.mtime_ns)
        if new_rows:
            self.emb = np.concatenate([self.emb, np.asarray(new_rows, dtype=np.float16)], axis=0)

    def save(self, keep: set[str] | None = None) -> None:
        """Write the cache. With keep, entries for files not in keep are dropped."""
        rels = [r for r in self.rows if keep is None or r in keep]
        order = sorted(rels, key=lambda r: self.rows[r])
        emb = self.emb[[self.rows[r] for r in order]] if order else np.zeros((0, EMBED_DIM), np.float16)
        meta = {
            "model": self.model_id,
            "patches": self.patches,
            "files": [[r, self.entries[r][0], self.entries[r][1]] for r in order],
        }
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp_npy = self.dir / "embeddings.tmp.npy"
        tmp_idx = self.dir / "index.tmp.json"
        np.save(tmp_npy, emb)
        tmp_idx.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_npy, self.dir / "embeddings.npy")
        os.replace(tmp_idx, self.dir / "index.json")
        # Re-index in the compacted order so later puts append correctly.
        self.rows = {r: i for i, r in enumerate(order)}
        self.entries = {r: self.entries[r] for r in order}
        self.emb = emb


# ----------------------------------------------------------------------------
# Encoder
# ----------------------------------------------------------------------------

def model_dir() -> Path:
    return MODELS_DIR / MODEL_FOLDER


def models_present() -> bool:
    d = model_dir()
    return all((d / f).is_file() for f in MODEL_FILES)


def manual_download_message() -> str:
    d = model_dir()
    lines = [
        "The encoder could not be downloaded.",
        f"Download these files by hand from https://huggingface.co/{MODEL_REPO}/tree/main",
        f"and put them into {d}:",
    ]
    lines += [f"  {f}" for f in MODEL_FILES]
    lines.append("then run install.bat again.")
    return "\n".join(lines)


def fetch_models() -> int:
    if models_present():
        print(f"Encoder already present in {model_dir()}")
    else:
        print(f"Downloading {MODEL_REPO} into {model_dir()} ...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(MODEL_REPO, local_dir=str(model_dir()), allow_patterns=MODEL_FILES + [".gitattributes", "README.md"])
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {e}")
            print(manual_download_message())
            return 1
        if not models_present():
            print(manual_download_message())
            return 1
    print("Loading the encoder once as a check...")
    enc = Encoder(patches=256)
    v = enc.embed_arrays([np.zeros((64, 64, 3), np.uint8)])
    print(f"ok, embedding dim {v.shape[1]}, device {enc.device}")
    return 0


class Encoder:
    """SigLIP 2 NaFlex image tower, loaded from models/ offline.

    Images arrive as uint8 HxWx3 arrays whose sides are multiples of the patch
    size (the decode workers resize them). Patchification, normalisation and
    padding are done on the GPU, which replaces the transformers image
    processor one-to-one.
    """

    def __init__(self, patches: int):
        import torch
        if not models_present():
            sys.exit(manual_download_message())
        self.torch = torch
        self.patches = patches
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        import transformers
        from transformers import Siglip2VisionModel
        transformers.logging.set_verbosity_error()  # the vision tower alone leaves the text weights "unexpected"
        transformers.logging.disable_progress_bar()
        self.model = Siglip2VisionModel.from_pretrained(
            str(model_dir()), dtype=self.dtype, local_files_only=True, attn_implementation="sdpa"
        ).to(self.device).eval()

    @property
    def model_id(self) -> str:
        return MODEL_REPO

    def embed_arrays(self, arrays: list[np.ndarray]) -> np.ndarray:
        """Embed a batch of resized uint8 images. Returns L2-normalised float32 (N, D)."""
        torch = self.torch
        p = PATCH_SIZE
        n = len(arrays)
        pixel = torch.zeros((n, self.patches, p * p * 3), dtype=self.dtype, device=self.device)
        mask = torch.zeros((n, self.patches), dtype=torch.int32, device=self.device)
        shapes = torch.zeros((n, 2), dtype=torch.long, device=self.device)
        for i, a in enumerate(arrays):
            h, w = a.shape[0], a.shape[1]
            nh, nw = h // p, w // p
            t = torch.from_numpy(a).to(self.device, non_blocking=True)
            t = t.permute(2, 0, 1).to(self.dtype)             # C, H, W
            t = (t / 255.0 - 0.5) / 0.5
            t = t.reshape(3, nh, p, nw, p).permute(1, 3, 2, 4, 0).reshape(nh * nw, -1)
            k = min(nh * nw, self.patches)
            pixel[i, :k] = t[:k]
            mask[i, :k] = 1
            shapes[i, 0], shapes[i, 1] = nh, nw
        with torch.inference_mode():
            out = self.model(pixel_values=pixel, pixel_attention_mask=mask, spatial_shapes=shapes)
            feat = out.pooler_output.float()
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return feat.cpu().numpy()


# ----------------------------------------------------------------------------
# Decoding workers
# ----------------------------------------------------------------------------

def target_size(height: int, width: int, patches: int, p: int = PATCH_SIZE) -> tuple[int, int]:
    """The NaFlex resize rule of transformers: largest scale whose patch grid fits the budget."""
    def scaled(scale: float, size: int) -> int:
        s = math.ceil(size * scale / p) * p
        return max(p, int(s))

    lo, hi = 0.0, 1e3  # the processor searches down from a large scale, allowing upscaling
    # Binary search on the scale, as transformers does, then take the fitting size.
    eps = 1e-5
    while (hi - lo) >= eps:
        mid = (lo + hi) / 2
        th, tw = scaled(mid, height), scaled(mid, width)
        if (th // p) * (tw // p) <= patches:
            lo = mid
        else:
            hi = mid
    return scaled(lo, height), scaled(lo, width)


def decode_one(args: tuple[str, int]) -> tuple[np.ndarray | None, str]:
    """Open, convert to RGB and resize one image to its NaFlex size. Runs in a worker."""
    path, patches = args
    try:
        from PIL import Image, ImageOps
        try:
            import pillow_avif  # noqa: F401
        except Exception:  # noqa: BLE001
            pass
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            w, h = im.size
            th, tw = target_size(h, w, patches)
            if (th, tw) != (h, w):
                im = im.resize((tw, th), Image.BILINEAR)
            return np.array(im, dtype=np.uint8), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


# ----------------------------------------------------------------------------
# Embedding a root
# ----------------------------------------------------------------------------

@dataclass
class EmbedResult:
    vectors: dict[str, np.ndarray]     # rel -> embedding
    failed: dict[str, str]             # rel -> error
    seconds: float
    embedded: int                      # newly embedded this run


def embed_root(root: Path, items: list[ImageItem], encoder: Encoder, cache: EmbeddingCache,
               threads: int, batch: int, reembed: bool, label: str) -> EmbedResult:
    t0 = time.time()
    todo = [it for it in items if reembed or not cache.has(it)]
    vectors: dict[str, np.ndarray] = {}
    failed: dict[str, str] = {}
    for it in items:
        if it not in todo:
            vectors[it.rel] = cache.get(it).astype(np.float32)
    print(f"{label}: {len(items)} images, {len(items) - len(todo)} cached, {len(todo)} to embed")
    if todo:
        done = 0
        since_flush = 0
        last_print = time.time()
        with ProcessPoolExecutor(max_workers=threads) as pool:
            for w0 in range(0, len(todo), WINDOW):
                window = todo[w0:w0 + WINDOW]
                args = [(str(it.path), cache.patches) for it in window]
                pending_items: list[ImageItem] = []
                pending_arrays: list[np.ndarray] = []
                for it, (arr, err) in zip(window, pool.map(decode_one, args, chunksize=8)):
                    if arr is None:
                        failed[it.rel] = err
                        continue
                    pending_items.append(it)
                    pending_arrays.append(arr)
                    if len(pending_arrays) >= batch:
                        vecs = encoder.embed_arrays(pending_arrays)
                        cache.put(pending_items, vecs.astype(np.float16))
                        for pit, v in zip(pending_items, vecs):
                            vectors[pit.rel] = v
                        done += len(pending_items)
                        since_flush += len(pending_items)
                        pending_items, pending_arrays = [], []
                if pending_arrays:
                    vecs = encoder.embed_arrays(pending_arrays)
                    cache.put(pending_items, vecs.astype(np.float16))
                    for pit, v in zip(pending_items, vecs):
                        vectors[pit.rel] = v
                    done += len(pending_items)
                    since_flush += len(pending_items)
                if since_flush >= CACHE_FLUSH_EVERY:
                    cache.save()
                    since_flush = 0
                if time.time() - last_print > 5 or done == len(todo):
                    el = time.time() - t0
                    print(f"  {done}/{len(todo)} embedded, {done / el:.0f} img/s", flush=True)
                    last_print = time.time()
        cache.save(keep={it.rel for it in items})
    else:
        # Drop stale entries for files that are gone.
        if len(cache.entries) != len(items):
            cache.save(keep={it.rel for it in items})
    if failed:
        print(f"  {len(failed)} images could not be decoded")
    return EmbedResult(vectors, failed, time.time() - t0, len(todo) - len(failed))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", default=[], metavar="PATH", help="dataset root; may repeat")
    ap.add_argument("--samples", metavar="PATH", help="samples root with one folder per class")
    ap.add_argument("-o", "--output", metavar="PATH", help="output root; must be empty apart from _classify")
    ap.add_argument("--dry-run", action="store_true", help="write the plan, report and sheets; copy nothing")
    ap.add_argument("--move", action="store_true", help="move dataset images instead of copying them")
    ap.add_argument("--undo", action="store_true", help="undo the last run from <output>/_classify/moves.jsonl")
    ap.add_argument("--min-confidence", type=float, default=0.7, metavar="X", help="below this an image goes to _unsure (default 0.7)")
    ap.add_argument("--isolation-pct", type=float, default=3.0, metavar="X", help="the X%% most isolated images go to _unsure (default 3)")
    ap.add_argument("--no-isolation", action="store_true", help="disable the isolation gate")
    ap.add_argument("--sidecars", default=DEFAULT_SIDECARS, metavar="EXT,EXT", help="sidecar extensions (default .txt)")
    ap.add_argument("--sheets", action="store_true", help="write contact sheets per output folder")
    ap.add_argument("--patches", type=int, default=576, metavar="N", help="NaFlex patch budget per image (default 576)")
    ap.add_argument("--threads", type=int, default=8, metavar="N", help="image decoding workers (default 8)")
    ap.add_argument("--batch", type=int, default=64, metavar="N", help="encoder batch size (default 64)")
    ap.add_argument("--reembed", action="store_true", help="ignore the embedding caches")
    ap.add_argument("--fetch-models", action="store_true", help="download the encoder into models/ and exit")
    ap.add_argument("--embed-only", action="store_true", help=argparse.SUPPRESS)  # phase 1: stop after embedding
    return ap.parse_args(argv)


def sidecar_set(spec: str) -> set[str]:
    out = set()
    for s in spec.split(","):
        s = s.strip().lower()
        if not s:
            continue
        out.add(s if s.startswith(".") else "." + s)
    return out


def check_folders(args) -> tuple[list[Path], Path, Path]:
    if not args.dataset or not args.samples or not args.output:
        sys.exit("--dataset, --samples and -o are required (or -o with --undo)")
    datasets = [Path(d).resolve() for d in args.dataset]
    samples = Path(args.samples).resolve()
    output = Path(args.output).resolve()
    for d in datasets:
        if not d.is_dir():
            sys.exit(f"dataset root not found: {d}")
    if not samples.is_dir():
        sys.exit(f"samples root not found: {samples}")
    for d in datasets + [samples]:
        if output == d or output in d.parents or d in output.parents:
            sys.exit(f"the output root must not be inside, equal to or contain {d}")
    if samples in datasets or any(samples in d.parents or d in samples.parents for d in datasets):
        sys.exit("the samples root must not be inside a dataset root or contain one")
    return datasets, samples, output


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.fetch_models:
        return fetch_models()
    if args.undo:
        sys.exit("--undo is not implemented yet")
    datasets, samples, output = check_folders(args)
    sidecars = sidecar_set(args.sidecars)

    t_all = time.time()
    classes = discover_samples(samples, sidecars)
    check_samples(classes)
    dataset_items: list[tuple[Path, list[ImageItem]]] = []
    for d in datasets:
        items = scan_root(d, sidecars, skip_abs={output})
        n_cap = sum(1 for it in items if it.sidecars)
        print(f"Dataset {d}: {len(items)} images, {n_cap} with sidecars")
        dataset_items.append((d, items))

    print("Loading the encoder...")
    t0 = time.time()
    encoder = Encoder(patches=args.patches)
    print(f"  loaded in {time.time() - t0:.1f}s on {encoder.device}, attention {encoder.model.config._attn_implementation}")

    sample_items = [it for c in classes for v in c.subclasses.values() for it in v]
    sample_cache = EmbeddingCache(samples, encoder.model_id, args.patches)
    sample_res = embed_root(samples, sample_items, encoder, sample_cache, args.threads, args.batch, args.reembed, "Samples")
    results = []
    for d, items in dataset_items:
        cache = EmbeddingCache(d, encoder.model_id, args.patches)
        results.append(embed_root(d, items, encoder, cache, args.threads, args.batch, args.reembed, f"Dataset {d.name}"))
    for res, (d, items) in zip(results, dataset_items):
        if res.embedded:
            print(f"  {d.name}: {res.embedded} embedded in {res.seconds:.0f}s, {res.embedded / res.seconds:.0f} img/s")
    print(f"Embedding done in {time.time() - t_all:.0f}s total")
    if args.embed_only:
        return 0
    sys.exit("classification is not implemented yet (phase 2)")


if __name__ == "__main__":
    sys.exit(main())
