"""Files and folders of a scope: finding images on disk, moving an image with
its caption, removing emptied folders, and chunking a category when it passes
`split` images (its images move into the numbered folder 001).

ScopeChunks is what every writer uses (the download, reclassify, regroup): the
Chunker of a scope, plus the move of a category's images into 001 as soon as
no image of it is being written. The move updates the store in batches and
is journalled in the store, so a move cut short by a crash is finished the
next time the scope is opened for writing.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import layout

JOURNAL = "chunking"          # meta key: [[scope, category], ...] whose move has not finished


def scan_scope(scope_dir: Path) -> dict[int, str]:
    """Photo number -> path of its image relative to the folder (names by layout.cid_of)."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(scope_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in layout.IMAGE_EXTS:
                continue
            cid = layout.cid_of(stem)
            if cid is not None:
                out[cid] = os.path.relpath(os.path.join(dirpath, fn), scope_dir).replace("\\", "/")
    return out


def move_pair(src: Path, dst: Path) -> None:
    """Move an image and its caption (and a JSON sidecar, if any); refuse to overwrite."""
    if dst.exists():
        raise FileExistsError(f"{dst} exists")
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)
    for ext in (".txt", ".json"):
        s2, d2 = src.with_suffix(ext), dst.with_suffix(ext)
        if s2.exists():
            os.replace(s2, d2)


def prune_empty(scope_dir: Path) -> int:
    """Remove the empty folders inside a folder (not the folder itself)."""
    n = 0
    if not scope_dir.is_dir():
        return 0
    for dirpath, dirnames, filenames in os.walk(scope_dir, topdown=False):
        if dirpath != str(scope_dir) and not os.listdir(dirpath):
            os.rmdir(dirpath)
            n += 1
    return n


def remove_folder(base: Path, folder: str) -> bool:
    """Delete a scope's folder when nothing but empty folders is left in it,
    and its parents up to the base when they are left empty (the 1970s folder
    of a years/place base)."""
    d = base / folder
    if d.is_dir():
        prune_empty(d)
        try:
            d.rmdir()
        except OSError:
            return False
    parent = d.parent
    while parent != base and base in parent.parents:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent
    return True


def record_with_path(store, base: Path, cid: int, rel: str, note: tuple[str, dict] | None = None) -> str | None:
    """The photo's record with its output path changed (and a note appended
    to a list in it); the JSON file of the record follows when there is one."""
    rec = store.record(cid)
    if rec is None:
        return None
    rec.setdefault("processing", {}).setdefault("output", {})["path"] = rel
    if note is not None:
        rec.setdefault(note[0], []).append(note[1])
    text = json.dumps(rec, ensure_ascii=False, indent=1, default=str)
    rf = layout.record_file(base, cid)
    if rf.is_file():
        rf.write_text(text, encoding="utf-8")
    return text


def _journal(store, scope: str, category: str, add: bool) -> None:
    items = [tuple(x) for x in store.get_meta(JOURNAL, [])]
    key = (scope, category)
    items = [x for x in items if x != key] + ([key] if add else [])
    store.set_meta(JOURNAL, [list(x) for x in items])


def chunk_category(store, base: Path, sc: dict, category: str, width: int, log=print) -> dict[int, int]:
    """Move the images that lie directly in a category folder into its lowest
    numbered folders with room (001 first), and point the store at every
    image of the category's numbered folders. Safe to run again. -> placed[chunk]."""
    name, split = sc["name"], sc["split"]
    cat_dir = base / layout.to_base(sc, category)
    _journal(store, name, category, True)
    counts: Counter = Counter()
    loose = []
    if cat_dir.is_dir():
        for e in os.scandir(cat_dir):
            stem, ext = os.path.splitext(e.name)
            if e.is_file() and layout.cid_of(stem) is not None and ext.lower() in layout.IMAGE_EXTS:
                loose.append(Path(e.path))
            elif e.is_dir() and e.name.isdigit() and len(e.name) == width:
                counts[int(e.name)] = sum(1 for f in os.scandir(e.path) if f.is_file()
                                          and os.path.splitext(f.name)[1].lower() in layout.IMAGE_EXTS)
    loose.sort(key=lambda f: layout.cid_of(f.stem))
    placed: Counter = Counter()
    owned = {cid: path for cid, path, _, _ in store.done_rows(name)}
    batch = []

    def flush():
        if batch:
            store.move_paths(batch)
            batch.clear()

    for f in loose:
        chunk = 1
        while counts[chunk] >= split:
            chunk += 1
        dst = f.parent / f"{chunk:0{width}d}" / f.name
        try:
            move_pair(f, dst)
        except OSError as e:
            log(f"  {f}: not moved into {dst.parent.name} ({e})")
            continue
        counts[chunk] += 1
        placed[chunk] += 1
        cid = layout.cid_of(f.stem)
        rel = dst.relative_to(base).as_posix()
        if cid in owned and owned[cid] != rel:
            batch.append((cid, rel, record_with_path(store, base, cid, rel)))
            owned[cid] = rel
        if len(batch) >= 100:
            flush()
    flush()
    # anything already in the numbered folders that the store still places elsewhere (a move cut short)
    for chunk in counts:
        cdir = cat_dir / f"{chunk:0{width}d}"
        if not cdir.is_dir():
            continue
        for e in os.scandir(cdir):
            stem, ext = os.path.splitext(e.name)
            if layout.cid_of(stem) is not None and ext.lower() in layout.IMAGE_EXTS:
                cid, rel = layout.cid_of(stem), Path(e.path).relative_to(base).as_posix()
                if cid in owned and owned[cid] != rel and not (base / owned[cid]).is_file():
                    batch.append((cid, rel, record_with_path(store, base, cid, rel)))
                    owned[cid] = rel
                txt = cat_dir / f"{stem}.txt"          # a caption written to the old place during the move
                if txt.is_file():
                    os.replace(txt, Path(e.path).with_suffix(".txt"))
    flush()
    _journal(store, name, category, False)
    if loose:
        log(f"  {layout.to_base(sc, category)}: passed {split:,} images; {len(loose):,} moved into numbered folders")
    return dict(placed)


class ScopeChunks:
    """The Chunker of one scope, with the moves it asks for done at once."""

    def __init__(self, store, base: Path, sc: dict, log=print, defer: bool = False):
        """defer: the moves into 001 wait for settle(), for a caller that moves
        many files of the scope and must find them where it planned."""
        self.store, self.base, self.sc, self.log, self.defer = store, base, sc, log, defer
        self.folder = sc["folder"]
        self.width = max(sc["width"], 3) if sc["split"] > 0 else 0
        for scope, category in [tuple(x) for x in store.get_meta(JOURNAL, [])]:
            if scope == sc["name"] and self.width:          # a move cut short: finish it, the store follows
                chunk_category(store, base, sc, category, self.width, log)
        rels = [r for r in (layout.to_scope(sc, p) for p in store.scope_paths(sc["name"])) if r is not None]
        self.chunker = layout.Chunker(sc["split"], sc["width"], rels)
        if not defer:
            self.settle()

    def rel(self, scope_rel: str) -> str:
        """Relative to the scope -> relative to the base."""
        return layout.to_base(self.sc, scope_rel)

    def place(self, category: str, name: str) -> str:
        return self.chunker.place(category, name)

    def done(self, scope_rel: str) -> None:
        self.chunker.done(scope_rel)
        if not self.defer:
            self.settle()

    def abandon(self, scope_rel: str) -> None:
        self.chunker.abandon(scope_rel)
        if not self.defer:
            self.settle()

    def remove(self, scope_rel: str) -> None:
        self.chunker.remove(scope_rel)

    def settle(self) -> None:
        for category in self.chunker.ready_moves():
            self.chunker.moved(category, chunk_category(self.store, self.base, self.sc, category, self.width,
                                                        self.log))


def scan_rels(base: Path, sc: dict) -> dict[int, str]:
    """Photo number -> scope-relative path of its image, over every root of the scope."""
    out = {}
    for _, root in layout.scope_roots(sc):
        if (base / root).is_dir():
            for cid, rel in scan_scope(base / root).items():
                scope_rel = layout.to_scope(sc, layout._join(root, rel))
                if scope_rel is not None:
                    out[cid] = scope_rel
    return out


def locate(base: Path, sc: dict, cid: int, path: str | None, scans: dict) -> str | None:
    """Where the photo's image is, relative to the base: where the store says,
    else wherever it was moved to inside its scope's roots (scans caches the
    scan of each scope)."""
    if path and (base / path).is_file():
        return path
    if sc["name"] not in scans:
        scans[sc["name"]] = scan_rels(base, sc)
    rel = scans[sc["name"]].get(cid)
    return layout.to_base(sc, rel) if rel else None


def prune_scope(base: Path, sc: dict) -> int:
    return sum(prune_empty(base / root) for _, root in layout.scope_roots(sc))


def remove_scope_folders(base: Path, sc: dict) -> bool:
    """remove_folder for every root of the scope; False when one keeps foreign files."""
    return all([remove_folder(base, root) for _, root in layout.scope_roots(sc)])
