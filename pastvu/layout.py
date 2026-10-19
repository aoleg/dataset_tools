"""Names and folders: the scope name (<regions>_<years>), the folder of an
image (<scope>/<category>/<chunk>/<cid>.<ext>), and the chunk allocator that
keeps every folder under --split images.

A scope folder holds a classify-style tree: one folder per category, and
_unsure/<category>, each split into numbered chunks 001, 002, ... of at most
`split` images (each with its .txt caption), so no folder grows past what
Explorer handles well. A category stays one tree: trainers and the dataset
tools read subfolders.
"""

from __future__ import annotations

import math
import re
import threading
import unicodedata
from collections import defaultdict

SCOPE_NAME = re.compile(r"^[a-z0-9][a-z0-9_+\-]*$")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp"}


def slug(text: str) -> str:
    """'Järva County' -> 'jarva_county'; a trailing disambiguation in brackets goes."""
    t = re.sub(r"\s*\([^()]*\)\s*$", "", str(text or ""))
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")


def years_label(years: tuple[int, int], decade: int | None) -> str:
    return f"{decade}s" if decade is not None else f"{years[0]}-{years[1]}"


def scope_name(labels: list[str], years: tuple[int, int], decade: int | None) -> str:
    return "+".join(labels) + "_" + years_label(years, decade)


def check_scope_name(name: str) -> str:
    if not SCOPE_NAME.match(name) or len(name) > 80:
        raise SystemExit(f"scope name {name!r}: use at most 80 lower-case letters, digits, _, + and -")
    return name


def chunk_width(total: int, split: int) -> int:
    if split <= 0:
        return 0
    return max(3, len(str(max(1, math.ceil(total / split)))))


def split_path(rel: str, width: int) -> tuple[str, int | None, str]:
    """'1_transport/002/123.jpg' (relative to the scope) -> ('1_transport', 2, '123.jpg')."""
    parts = rel.replace("\\", "/").split("/")
    name = parts[-1]
    if width and len(parts) >= 2 and parts[-2].isdigit() and len(parts[-2]) == width:
        return "/".join(parts[:-2]), int(parts[-2]), name
    return "/".join(parts[:-1]), None, name


def join_path(folder: str, chunk: int | None, width: int, name: str) -> str:
    parts = [p for p in folder.split("/") if p]
    if chunk is not None:
        parts.append(f"{chunk:0{width}d}")
    return "/".join(parts + [name])


class Chunker:
    """Which chunk of a folder takes the next image: the lowest chunk with
    room, so chunks left with holes by reclassify fill up again. Thread-safe."""

    def __init__(self, split: int, width: int, rels=()):
        self.split, self.width = split, width
        self.counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.lock = threading.Lock()
        for rel in rels:
            folder, chunk, _ = split_path(rel, width)
            if chunk is not None:
                self.counts[folder][chunk] += 1

    def place(self, folder: str, name: str) -> str:
        if self.split <= 0:
            return join_path(folder, None, 0, name)
        with self.lock:
            counts = self.counts[folder]
            chunk = 1
            while counts.get(chunk, 0) >= self.split:
                chunk += 1
            counts[chunk] += 1
        return join_path(folder, chunk, self.width, name)

    def release(self, rel: str) -> None:
        folder, chunk, _ = split_path(rel, self.width)
        if chunk is None:
            return
        with self.lock:
            if self.counts[folder].get(chunk, 0) > 0:
                self.counts[folder][chunk] -= 1


def category_of(rel: str, width: int) -> str:
    """The category folder of a path relative to the scope ('_unsure/5_transport' or '5_transport' or '')."""
    return split_path(rel, width)[0]


def record_file(base, cid: int):
    """Where the optional JSON file of a record goes: <base>/_records/<cid // 10000>/<cid>.json."""
    return base / "_records" / f"{cid // 10000:03d}" / f"{cid}.json"
