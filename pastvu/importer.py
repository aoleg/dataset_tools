"""import: a local dataset that this tool did not download, processed like a
download into an output folder of its own, which is a base of its own.

- The source is only read. The output folder holds the images under their
  source names, in <category>/<bw|color>/ or <bw|color>/<category>/ (the
  conf's export_layout), its store pastvu.db and _reports.
- index_source lists the source's images (folders starting with _ or . are
  left out); a new name gets the next number, so running the import again
  picks up files added since.
- Each photo goes through the stages of a download: size and aspect ratio,
  borders, watermarks, quality, black-and-white or colour, category. There is
  no PastVu band to cut (no PastVu record says it is there).
- The caption is the source's .txt without hashtags, links and @mentions; a
  photo whose caption is empty then gets no .txt.
- With a base to check against (--against), a photo that is a copy of one of
  the base's photos is skipped: PastVu holds the original, which Telegram
  channels repost recompressed. reclassify of the import checks the saved
  photos again against the base as it is then, and deletes the copies.

A copy: the cosine of the SigLIP 2 embeddings is COPY_SURE or more, or it is
COPY_CANDIDATE or more and the deduplicate tool's feature test agrees (ORB
keypoints, a RANSAC homography with MIN_INLIERS inliers over MIN_COVER of
both pictures, and the warped centre correlating at MIN_NCC). Calibrated on
34,300 Telegram reposts against 210,801 PastVu photos (72 pairs checked by
eye, all agreeing): copies had 262 to 1,436 inliers, other photos of the same
place fewer than 40; by cosine, 5 % of the pairs at 0.86-0.90 were copies,
30 % at 0.92-0.94, 68 % at 0.94-0.96 and 98 % and more from 0.96.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from pathlib import Path

import numpy as np

import layout

KIND = "import"
COPY_SURE = 0.96
COPY_CANDIDATE = 0.85
CANDIDATES = 3                  # nearest base photos tested per image
FEATURE_SIDE, ORB_FEATURES = 640, 1500
MIN_INLIERS, MIN_COVER, MIN_NCC = 40, 0.36, 0.5
COPY_REASON = "copy in base"
CAPTION_EXTS = (".txt",)

# --- the source ----------------------------------------------------------------------------


def source_images(src: Path) -> list[tuple[str, int, int]]:
    """(path relative to the source, bytes, mtime) of every image, folders starting with _ or . left out."""
    out = []
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(("_", ".")))
        for fn in sorted(filenames):
            if os.path.splitext(fn)[1].lower() in layout.IMAGE_EXTS:
                p = os.path.join(dirpath, fn)
                st = os.stat(p)
                out.append((os.path.relpath(p, src).replace("\\", "/"), st.st_size, st.st_mtime_ns))
    return out


def index_source(store, scope: str, src: Path) -> dict:
    """Give every source image that the store does not know yet a number and a name. -> counts."""
    known = store.name_rows()
    taken = set(store.names())
    new = []
    for rel, size, mtime in source_images(src):
        if rel in known:
            continue
        stem = os.path.splitext(rel.rsplit("/", 1)[-1])[0]
        name, k = stem, 1
        while name in taken:
            k += 1
            name = f"{stem}_{k}"
        taken.add(name)
        new.append((name, rel, size, mtime))
    if new:
        store.add_sources(scope, new)
    return {"known": len(known), "new": len(new), "renamed": sum(1 for n, rel, *_ in new
                                                                 if n != os.path.splitext(rel.rsplit("/", 1)[-1])[0])}


# --- captions ------------------------------------------------------------------------------

_LINK = re.compile(r"(?:https?://|www\.)\S+|\b(?:t|telegram)\.me/\S+", re.IGNORECASE)
_TAG = re.compile(r"(?<![\w&])#[\w\-]+")
_MENTION = re.compile(r"(?<![\w@.])@\w{2,}")


def clean_caption(text: str) -> str:
    """The source's caption without links, hashtags and @mentions, with its whitespace tidied."""
    t = _LINK.sub(" ", text or "")
    t = _TAG.sub(" ", t)
    t = _MENTION.sub(" ", t)
    lines = []
    for line in t.replace("\r\n", "\n").split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        line = re.sub(r"\s+([,.;:!?])", r"\1", line)
        if re.search(r"\w", line):
            lines.append(line)
    return "\n".join(lines).strip()


def read_caption(path: Path) -> str | None:
    for ext in CAPTION_EXTS:
        p = path.with_suffix(ext)
        if p.is_file():
            data = p.read_bytes()
            for enc in ("utf-8-sig", "cp1251"):
                try:
                    return data.decode(enc)
                except UnicodeDecodeError:
                    continue
            return data.decode("utf-8", errors="replace")
    return None


# --- the record ----------------------------------------------------------------------------


def record_json(cid: int, name: str, src_rel: str, data: bytes, raw_caption: str | None, processing: dict,
                classification: dict | None, tool: dict) -> str:
    doc = {"cid": cid, "name": name,
           "source": {"path": src_rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                      "caption": raw_caption},
           "processing": processing, "classification": classification, "tool": tool}
    return json.dumps(doc, ensure_ascii=False, indent=1, default=str)


# --- copies of base photos -----------------------------------------------------------------


def _gray(img) -> np.ndarray:
    from PIL import Image, ImageOps
    if isinstance(img, np.ndarray):
        im = Image.fromarray(np.ascontiguousarray(img))
    else:
        with Image.open(img) as f:
            im = ImageOps.exif_transpose(f)
            im.load()
    im = im.convert("L")
    im.thumbnail((FEATURE_SIDE, FEATURE_SIDE), Image.BILINEAR)
    return np.asarray(im)


def feature_test(a, b) -> tuple[int, float, float | None]:
    """The deduplicate tool's copy test of two pictures (arrays or files). -> (inliers, cover, centre ncc)."""
    import cv2
    ga, gb = _gray(a), _gray(b)
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 10 or len(kb) < 10:
        return 0, 0.0, None
    good = [m[0] for m in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
            if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < 8:
        return len(good), 0.0, None
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
    if H is None:
        return 0, 0.0, None
    inl = mask.ravel().astype(bool)

    def cover(p, g):
        if len(p) < 2:
            return 0.0
        (x0, y0), (x1, y1) = p.min(0), p.max(0)
        return float((x1 - x0) * (y1 - y0) / g.size)
    cov = min(cover(src[inl], ga), cover(dst[inl], gb))
    h, w = gb.shape
    warped = cv2.warpPerspective(ga, H, (w, h))
    valid = cv2.warpPerspective(np.full_like(ga, 255), H, (w, h)) > 0
    win = np.zeros_like(valid)
    win[h // 4:3 * h // 4, w // 4:3 * w // 4] = True
    m = valid & win
    if m.sum() < 500:
        return int(inl.sum()), cov, 0.0
    x, y = warped[m].astype(np.float32), gb[m].astype(np.float32)
    x -= x.mean()
    y -= y.mean()
    ncc = float((x * y).sum() / max(math.sqrt(float((x * x).sum()) * float((y * y).sum())), 1e-6))
    return int(inl.sum()), cov, ncc


def passes(test: tuple[int, float, float | None]) -> bool:
    n, cov, ncc = test
    return n >= MIN_INLIERS and cov >= MIN_COVER and ncc is not None and ncc >= MIN_NCC


class Against:
    """The saved photos of a PastVu base, to find copies of them: their embeddings (by
    the same model and patch budget) and their files. Read only; the base may be
    downloading at the same time."""

    def __init__(self, base: Path, model: str, patches: int, log=print):
        self.base = Path(base)
        db_path = self.base / "pastvu.db"
        if not db_path.is_file():
            raise SystemExit(f"--against {base}: no pastvu.db there")
        db = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
        if (db.execute("SELECT value FROM meta WHERE key='kind'").fetchone() or [None])[0] == json.dumps(KIND):
            raise SystemExit(f"--against {base}: that is an import, not a PastVu base")
        rows = db.execute("SELECT e.cid, e.vec, s.path FROM embeddings e JOIN status s ON s.cid=e.cid "
                          "WHERE s.state='done' AND e.model=? AND e.patches=?", (model, patches)).fetchall()
        db.close()
        self.cids = np.array([r[0] for r in rows], np.int64)
        self.paths = [r[2] for r in rows]
        V = np.stack([np.frombuffer(r[1], np.float16) for r in rows]) if rows else np.zeros((0, 1152), np.float16)
        V = V.astype(np.float32)
        V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
        self.on_gpu = False
        try:
            import torch
            if torch.cuda.is_available() and len(V):
                self.V = torch.from_numpy(V).half().cuda()
                self.on_gpu = True
        except Exception:  # noqa: BLE001
            pass
        if not self.on_gpu:
            self.V = V
        log(f"  looking for copies of {len(rows):,} photos of {self.base}"
            + (" (their embeddings on the GPU)" if self.on_gpu else ""))

    def nearest(self, X) -> tuple[np.ndarray, np.ndarray]:
        """The CANDIDATES nearest base photos of each embedding. -> (cosines, indices), each [n, k]."""
        X = np.asarray(X, np.float32).reshape(-1, self.V.shape[1])
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        k = min(CANDIDATES, len(self.cids))
        if self.on_gpu:
            import torch
            s = torch.from_numpy(X).half().cuda() @ self.V.T
            v, i = s.topk(k, dim=1)
            return v.float().cpu().numpy(), i.cpu().numpy()
        s = X @ self.V.T
        i = np.argsort(-s, axis=1)[:, :k]
        return np.take_along_axis(s, i, 1), i

    def match(self, vec, picture) -> tuple[int, float] | None:
        """The base photo that this picture (array or file) is a copy of: (PastVu number, cosine), or None."""
        if not len(self.cids):
            return None
        cos, idx = self.nearest(vec)
        for c, i in zip(cos[0], idx[0]):
            c = float(c)
            if c < COPY_CANDIDATE:
                break
            if c >= COPY_SURE:
                return int(self.cids[i]), c
            f = self.base / self.paths[i]
            try:
                if f.is_file() and passes(feature_test(picture, f)):
                    return int(self.cids[i]), c
            except Exception:  # noqa: BLE001 - an unreadable file is no copy
                continue
        return None

    def matches(self, vecs, pictures, log=print) -> dict[int, tuple[int, float]]:
        """match() for many photos: vecs [n, d], pictures their files. -> {position: (PastVu number, cosine)}"""
        out = {}
        if not len(self.cids) or not len(vecs):
            return out
        tested = 0
        for s in range(0, len(vecs), 1024):
            cos, idx = self.nearest(np.asarray(vecs[s:s + 1024], np.float32))
            for j in range(cos.shape[0]):
                for c, i in zip(cos[j], idx[j]):
                    c = float(c)
                    if c < COPY_CANDIDATE:
                        break
                    if c >= COPY_SURE:
                        out[s + j] = (int(self.cids[i]), c)
                        break
                    f = self.base / self.paths[i]
                    tested += 1
                    try:
                        if f.is_file() and passes(feature_test(pictures[s + j], f)):
                            out[s + j] = (int(self.cids[i]), c)
                            break
                    except Exception:  # noqa: BLE001
                        continue
        log(f"  copies of base photos: {len(out):,} ({tested:,} pairs checked by their features)")
        return out

    @staticmethod
    def reason(hit: tuple[int, float]) -> str:
        return f"{COPY_REASON} (pastvu {hit[0]}, cos {hit[1]:.3f})"


# --- the tree order ------------------------------------------------------------------------


def reorder(store, base: Path, sc: dict, order: str, log=print) -> int:
    """Move every saved photo of an import into the folders of the other order
    (categories first or colour first). Scope-relative paths do not change. -> moved."""
    import folders
    old = dict(sc)
    new = dict(sc, order=order)
    moved = 0
    batch = []
    for cid, path, _, _ in store.done_rows(sc["name"]):
        rel = layout.to_scope(old, path)
        if rel is None:
            continue
        dst = layout.to_base(new, rel)
        if dst == path or not (base / path).is_file():
            continue
        try:
            folders.move_pair(base / path, base / dst)
        except OSError as e:
            log(f"  {cid}: not moved ({e})")
            continue
        batch.append((cid, dst, folders.record_with_path(store, base, cid, dst)))
        moved += 1
        if len(batch) >= 200:
            store.move_paths(batch)
            batch.clear()
    if batch:
        store.move_paths(batch)
    store.set_meta("tree_order", order)
    folders.prune_empty(base)
    return moved
