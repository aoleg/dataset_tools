"""Names and folders: the scope name (<regions>_<years>), the scope's folder
under the base (by the base's layout), the folder of an image
(<scope folder>/<category>[/<chunk>]/<cid>.<ext>), and the chunk allocator
that keeps every folder under --split images.

A scope folder holds a classify-style tree: one folder per category, and
_unsure/<category>. A category holds its images directly until it would pass
`split` images; then they move into the numbered folder 001 and the next go
into 002, 003, ... of at most `split` images each (each with its .txt caption),
so no folder grows past what Explorer handles well. A category stays one tree:
trainers and the dataset tools read subfolders.
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


LAYOUTS = ("years/place", "place/years", "flat")
DEFAULT_LAYOUT = "years/place"


def scope_folder(layout: str, name: str, label: str, years: tuple[int, int], decade: int | None) -> str:
    """The scope's folder relative to the base: years/place 1970s/russia,
    place/years russia/1970s, flat russia_1970s (the scope's name)."""
    if layout == "flat":
        return name
    y = years_label(years, decade)
    return f"{y}/{label}" if layout == "years/place" else f"{label}/{y}"


def check_scope_name(name: str) -> str:
    if not SCOPE_NAME.match(name) or len(name) > 80:
        raise SystemExit(f"scope name {name!r}: use at most 80 lower-case letters, digits, _, + and -")
    return name


MODES = ("bw", "color")


def scope_roots(sc: dict) -> list[tuple[str, str]]:
    """Where a scope's photos live: (the start of a scope-relative path, the
    folder relative to the base) per root. A base split by black-and-white
    and colour gives every scope two roots, bw/<folder> and color/<folder>,
    and a scope-relative path starts with its mode: bw/50_transport/123.jpg."""
    if sc.get("split_mode"):
        return [(f"{m}/", f"{m}/{sc['folder']}") for m in MODES]
    return [("", sc["folder"])]


def to_base(sc: dict, rel: str) -> str:
    """A scope-relative path (or category folder) -> relative to the base."""
    if sc.get("split_mode"):
        mode, _, rest = rel.partition("/")
        return f"{mode}/{sc['folder']}" + (f"/{rest}" if rest else "")
    return sc["folder"] + (f"/{rel}" if rel else "")


def to_scope(sc: dict, base_rel: str) -> str | None:
    """Relative to the base -> relative to the scope (None when outside it)."""
    for prefix, root in scope_roots(sc):
        if base_rel == root:
            return prefix.rstrip("/")
        if base_rel.startswith(root + "/"):
            return prefix + base_rel[len(root) + 1:]
    return None


def mode_category(sc: dict, folder: str) -> tuple[str | None, str]:
    """A category folder relative to the scope -> (its mode, the category)."""
    if sc.get("split_mode"):
        mode, _, rest = folder.partition("/")
        return mode, rest
    return None, folder


def with_mode(sc: dict, mode: str | None, category: str) -> str:
    """The category folder of a photo relative to the scope."""
    if sc.get("split_mode") and mode:
        return f"{mode}/{category}" if category else mode
    return category


def scope_width(sc: dict) -> int:
    """The digits of the scope's numbered folders (0: it has none)."""
    return max(sc["width"], 3) if sc["split"] > 0 else 0


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
    """Which folder of a category takes the next image. A category holds its
    images directly until it would pass `split`; then it is chunked: the
    images it holds are to move into chunk 1 (ready_moves, done by the
    caller), and new images go into the lowest chunk with room, so chunks
    left with holes by reclassify fill up again. A chunked category stays
    chunked. Thread-safe.

    place() hands out a path; done() says the image is committed there,
    abandon() that it was not written, remove() that an image left a path."""

    def __init__(self, split: int, width: int, rels=()):
        self.split, self.width = split, max(width, 3) if split > 0 else 0
        self.counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.flat: dict[str, int] = defaultdict(int)          # images directly in the category
        self.inflight: dict[str, int] = defaultdict(int)      # direct places not yet done or abandoned
        self.chunked: set[str] = set()
        self.reserved: set[str] = set()                       # chunk 1 held for the images that move
        self.moves: set[str] = set()                          # chunked categories with images still to move
        self.lock = threading.Lock()
        for rel in rels:
            folder, chunk, _ = split_path(rel, self.width)
            if chunk is not None:
                self.counts[folder][chunk] += 1
                self.chunked.add(folder)
            else:
                self.flat[folder] += 1
        for folder, n in self.flat.items():
            if n and (folder in self.chunked or (split > 0 and n > split)):
                self.chunked.add(folder)
                self.moves.add(folder)

    def place(self, folder: str, name: str) -> str:
        if self.split <= 0:
            return join_path(folder, None, 0, name)
        with self.lock:
            if folder not in self.chunked:
                if self.flat[folder] < self.split:
                    self.flat[folder] += 1
                    self.inflight[folder] += 1
                    return join_path(folder, None, 0, name)
                self.chunked.add(folder)
                self.reserved.add(folder)
                self.moves.add(folder)
                self.counts[folder][1] = self.split
            counts = self.counts[folder]
            chunk = 1
            while counts.get(chunk, 0) >= self.split:
                chunk += 1
            counts[chunk] += 1
        return join_path(folder, chunk, self.width, name)

    def done(self, rel: str) -> None:
        folder, chunk, _ = split_path(rel, self.width)
        if chunk is None:
            with self.lock:
                self.inflight[folder] = max(0, self.inflight[folder] - 1)

    def abandon(self, rel: str) -> None:
        folder, chunk, _ = split_path(rel, self.width)
        with self.lock:
            if chunk is None:
                self.inflight[folder] = max(0, self.inflight[folder] - 1)
                self.flat[folder] = max(0, self.flat[folder] - 1)
            elif self.counts[folder].get(chunk, 0) > 0:
                self.counts[folder][chunk] -= 1

    def remove(self, rel: str) -> None:
        folder, chunk, _ = split_path(rel, self.width)
        with self.lock:
            if chunk is None:
                self.flat[folder] = max(0, self.flat[folder] - 1)
            elif self.counts[folder].get(chunk, 0) > 0:
                self.counts[folder][chunk] -= 1

    def ready_moves(self) -> list[str]:
        """Chunked categories whose direct images can move now (none is being written)."""
        with self.lock:
            ready = [f for f in self.moves if self.inflight[f] == 0]
            self.moves.difference_update(ready)
        return ready

    def moved(self, folder: str, placed: dict[int, int]) -> None:
        """The direct images of a category went into chunks: placed[chunk] of them."""
        with self.lock:
            if folder in self.reserved:
                self.counts[folder][1] -= self.split
                self.reserved.discard(folder)
            for chunk, n in placed.items():
                self.counts[folder][chunk] += n
            self.flat[folder] = 0


def category_of(rel: str, width: int) -> str:
    """The category folder of a path relative to the scope ('_unsure/5_transport' or '5_transport' or '')."""
    return split_path(rel, width)[0]


def record_file(base, cid: int):
    """Where the optional JSON file of a record goes: <base>/_records/<cid // 10000>/<cid>.json."""
    return base / "_records" / f"{cid // 10000:03d}" / f"{cid}.json"
