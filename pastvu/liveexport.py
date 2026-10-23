"""run --export: every saved photo hard-linked into <base parent>/pastvu_export
as it is written, in one folder per category:

    pastvu_export/050-transport/bw/001/123456.jpg (+ 123456.txt)
                               color/001/...

- Image and .txt caption are hard links to the base's files: no extra disk
  space on the same drive. A caption the VLM writes later lands in both, as
  autocaption writes the file in place. Deleting a file in the export removes
  that link only; the base keeps its file.
- Numbered folders of at most `split` images (the conf's split) from the
  first image on; the lowest one with room fills first.
- The store records where each photo is in the export. A photo whose image
  you deleted from the export is remembered as excluded and never linked
  again (--export-reset forgets that and links everything anew).
- reclassify moves a photo's links to its new category (or bw/color), and
  drop deletes them, so the export follows the base. Moves inside the base
  (regroup, a category passing the split) keep the links valid by themselves.
- Photos saved before the export existed are linked when a run with --export
  starts.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import layout

ROOT_NAME = "pastvu_export"
META = "export"
EXCLUDED = "-"
SIDECARS = (".txt",)
NO_CATEGORY = "uncategorised"


def root_of(base: Path) -> Path:
    return base.parent / ROOT_NAME


def _width(store, split: int) -> int:
    with store.lock:
        n = store.db.execute("SELECT COUNT(*) FROM status").fetchone()[0]
    return layout.chunk_width(max(n, 1), split) if split > 0 else 0


class Exporter:
    def __init__(self, store, base: Path, split: int, log=print):
        self.store, self.base, self.log = store, base, log
        meta = store.get_meta(META) or {}
        self.root = Path(meta.get("root") or root_of(base))
        self.split = int(meta.get("split", split))
        self.width = int(meta.get("width") or _width(store, self.split))
        if not meta:
            store.set_meta(META, {"root": str(self.root), "split": self.split, "width": self.width})
        self.lock = threading.Lock()
        self.counts: dict[str, Counter] = defaultdict(Counter)
        self.copied_warned = False
        self.stats = Counter()
        for cid, rel in store.export_rows():
            if rel and rel != EXCLUDED:
                self._count(rel, +1)

    @staticmethod
    def exists_for(store) -> bool:
        return bool(store.get_meta(META))

    # --- numbered folders --------------------------------------------------------------------

    def _count(self, rel: str, delta: int) -> None:
        parts = rel.split("/")
        if len(parts) >= 2 and parts[-2].isdigit():
            self.counts["/".join(parts[:-2])][int(parts[-2])] += delta

    def _place(self, folder: str, name: str) -> str:
        if self.split <= 0:
            return f"{folder}/{name}"
        with self.lock:
            c = self.counts[folder]
            chunk = 1
            while c[chunk] >= self.split:
                chunk += 1
            c[chunk] += 1
        return f"{folder}/{chunk:0{self.width}d}/{name}"

    @staticmethod
    def folder_of(mode: str | None, category: str) -> str:
        cat = category.replace("\\", "/").strip("/") or NO_CATEGORY
        return f"{cat}/{mode}" if mode else cat

    # --- files -------------------------------------------------------------------------------

    def _link(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            raise FileExistsError(f"{dst} exists")
        try:
            os.link(src, dst)
        except FileExistsError:
            raise
        except OSError:
            shutil.copy2(src, dst)                 # another drive, or a file system without hard links
            self.stats["copied"] += 1
            if not self.copied_warned:
                self.copied_warned = True
                self.log(f"  export: hard links are not possible from {src.parent} to {dst.parent}; copying "
                         f"instead (twice the disk space)")

    def add(self, cid: int, base_rel: str, mode: str | None, category: str) -> bool:
        """Link a saved photo into the export, unless it is there already or was excluded."""
        if self.store.export_of(cid) is not None:
            return False
        src = self.base / base_rel
        if not src.is_file():
            return False
        rel = self._place(self.folder_of(mode, category), src.name)
        dst = self.root / rel
        try:
            self._link(src, dst)
            for ext in SIDECARS:
                s2 = src.with_suffix(ext)
                if s2.is_file():
                    self._link(s2, dst.with_suffix(ext))
        except OSError as e:
            self._count(rel, -1)
            self.log(f"  export: {cid} not linked ({e})")
            return False
        self.store.set_export(cid, rel)
        self.stats["linked"] += 1
        return True

    def move(self, cid: int, mode: str | None, category: str) -> None:
        """The photo changed category or mode in the base: its links follow, when they are
        where the export put them (a photo you moved inside the export stays where you put it)."""
        rel = self.store.export_of(cid)
        if not rel or rel == EXCLUDED:
            return
        old = self.root / rel
        folder = self.folder_of(mode, category)
        if "/".join(rel.split("/")[:-2 if self.split > 0 else -1]) == folder or not old.is_file():
            return
        new_rel = self._place(folder, old.name)
        new = self.root / new_rel
        new.parent.mkdir(parents=True, exist_ok=True)
        os.replace(old, new)
        for ext in SIDECARS:
            if old.with_suffix(ext).is_file():
                os.replace(old.with_suffix(ext), new.with_suffix(ext))
        with self.lock:
            self._count(rel, -1)
        self.store.set_export(cid, new_rel)
        self.stats["moved"] += 1

    def remove(self, cid: int) -> None:
        """drop: delete the photo's links and forget them (an exclusion stays: it was your choice)."""
        rel = self.store.export_of(cid)
        if rel and rel != EXCLUDED:
            img = self.root / rel
            for f in (img, *(img.with_suffix(e) for e in SIDECARS)):
                if f.is_file():
                    f.unlink()
            with self.lock:
                self._count(rel, -1)
            self.store.set_export(cid, None)

    # --- the start of a run with --export ----------------------------------------------------

    def scan(self) -> dict[int, str]:
        on_disk = {}
        if self.root.is_dir():
            for dirpath, dirnames, filenames in os.walk(self.root):
                for fn in filenames:
                    stem, ext = os.path.splitext(fn)
                    if stem.isdigit() and ext.lower() in layout.IMAGE_EXTS:
                        on_disk[int(stem)] = os.path.relpath(os.path.join(dirpath, fn), self.root).replace("\\", "/")
        return on_disk

    def reconcile(self) -> dict:
        """Find what you did in the export: a recorded image that is gone is excluded for
        good; one you moved inside the export is recorded where it is now."""
        on_disk = self.scan()
        excluded = moved = 0
        updates = []
        for cid, rel in self.store.export_rows():
            if rel == EXCLUDED:
                continue
            now = on_disk.get(cid)
            if now is None:
                updates.append((cid, EXCLUDED))
                excluded += 1
                for ext in SIDECARS:              # the caption left behind is a link too: tidy it away
                    orphan = (self.root / rel).with_suffix(ext)
                    if orphan.is_file():
                        orphan.unlink()
            elif now != rel:
                updates.append((cid, now))
                moved += 1
        if updates:
            self.store.set_exports(updates)
            self.counts.clear()
            for cid, rel in self.store.export_rows():
                if rel and rel != EXCLUDED:
                    self._count(rel, +1)
        return {"excluded": excluded, "moved": moved, "present": len(on_disk)}

    def backfill(self, log_every: int = 5000) -> int:
        """Link every saved photo of the base that the export has not seen yet."""
        rows = self.store.unexported_rows()
        t0, n = time.time(), 0
        for i, (cid, path, mode, category) in enumerate(rows, 1):
            n += self.add(cid, path, mode, category or "")
            if i % log_every == 0:
                self.log(f"  export: {i:,} of {len(rows):,} saved photos looked at ({time.time() - t0:.0f} s)")
        return n

    def reset(self) -> int:
        """--export-reset: forget every exclusion and record the export as it is on disk, so
        the backfill links whatever is missing (what you deleted comes back). -> files found."""
        on_disk = self.scan()
        self.store.clear_exports()
        self.store.set_exports(list(on_disk.items()))
        self.counts.clear()
        for rel in on_disk.values():
            self._count(rel, +1)
        return len(on_disk)


def open_for_run(store, base: Path, split: int, reset: bool = False, log=print) -> Exporter:
    ex = Exporter(store, base, split, log)
    if reset:
        n = ex.reset()
        log(f"Export {ex.root}: exclusions forgotten; {n:,} files found there are kept, the rest is linked again")
    rec = ex.reconcile()
    if rec["excluded"] or rec["moved"]:
        log(f"Export {ex.root}: {rec['excluded']:,} images you deleted stay out, {rec['moved']:,} you moved are "
            f"recorded where they are")
    n = ex.backfill()
    log(f"Export {ex.root}: {n:,} saved photos linked now; new photos are linked as they are saved"
        + (f" ({ex.stats['copied']:,} copied: no hard links possible)" if ex.stats["copied"] else ""))
    return ex
