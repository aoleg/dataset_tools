"""Black-and-white or colour: a fast pixel test, and a probe on the category
embedding for the borderline photos.

A toned print (sepia, selenium, cyanotype, yellowed paper) has a colour that
depends only on brightness. The test fits that tone curve (the median a*, b*
per brightness band, in Lab), takes what is left, and keeps the part that
points away from the tone's own hue: mixing dark and light pixels of a curved
sepia tone lands off the curve, but along its hue. A colour photo keeps
colour in other hues. Two shares of the picture decide (rule B, calibrated on
1,280 samples and 264 hand labels):

- off_hue: pixels with more than 6 units of off-hue colour after a 3x3
  median filter (thin edge fringes go, colour areas stay);
- strong: pixels with more than 12 units of leftover colour.

colour if off_hue >= 0.01 or strong >= 0.03. Photos near those lines
(about 10 %) are borderline; when the photo has its SigLIP 2 category
embedding, a logistic probe trained on the samples folder decides them. The
probe learns from the samples the pixel test is sure about (off_hue 0 for
bw, >= 0.1 for colour), so it needs no labels of its own.

Measured on the picture after the band, border and watermark cuts, with the
painted watermark boxes left out (LaMa can tint its fill), on the central 90 %
area-averaged to 128 px.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

BW, COLOR = "bw", "color"
MODES = (BW, COLOR)
SIDE = 128
OFF_HUE, STRONG = 6.0, 12.0                    # Lab units
COLOUR_OFF, COLOUR_STRONG = 0.01, 0.03         # rule B
BORDER_OFF = (0.002, 0.03)                     # the probe decides inside these bands
BORDER_STRONG = (0.015, 0.06)
PROBE_COLOUR = 0.4
SURE_BW, SURE_COLOUR = 0.0, 0.1                # what the probe learns from
VERSION = 1


def to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB floats in [0, 1] -> CIE Lab (D65)."""
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], np.float32)
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _shrink(array: np.ndarray, exclude: list | None, dilate: int) -> tuple[np.ndarray, np.ndarray]:
    """Central 90 %, area-averaged to SIDE px on the long side. -> (RGB floats, mask of pixels to leave out)."""
    from PIL import Image
    h, w = array.shape[:2]
    dx, dy = int(w * 0.05), int(h * 0.05)
    a = array[dy:h - dy, dx:w - dx]
    hh, ww = a.shape[:2]
    im = Image.fromarray(np.ascontiguousarray(a))
    im.thumbnail((SIDE, SIDE), Image.BOX)              # as calibrated
    size = im.size
    small = np.asarray(im, dtype=np.float32) / 255.0
    out = np.zeros(small.shape[:2], bool)
    if exclude:
        m = np.zeros((hh, ww), np.uint8)
        for b in exclude:
            x0, y0, x1, y1 = (int(v) for v in b[:4])
            m[max(0, y0 - dy - dilate):max(0, y1 - dy + dilate), max(0, x0 - dx - dilate):max(0, x1 - dx + dilate)] = 255
        out = np.asarray(Image.fromarray(m).resize(size, Image.BOX)) > 0
    return small, out


def _tone_curve(L: np.ndarray, a: np.ndarray, b: np.ndarray, ok: np.ndarray):
    Lv, av, bv = L[ok], a[ok], b[ok]
    edges = np.linspace(Lv.min(), Lv.max() + 1e-3, 17)
    idx = np.digitize(Lv, edges) - 1
    c, ma, mb = [], [], []
    for i in range(16):
        s = idx == i
        if s.sum() >= 8:
            c.append(Lv[s].mean())
            ma.append(np.median(av[s]))
            mb.append(np.median(bv[s]))
    if len(c) < 2:
        return np.full_like(a, np.median(av)), np.full_like(b, np.median(bv))
    return np.interp(L, c, ma), np.interp(L, c, mb)


def scores(array: np.ndarray, exclude: list | None = None, dilate: int = 0) -> dict:
    """The pixel test on an upright RGB uint8 array (or a greyscale one).
    exclude: boxes [x0, y0, x1, y1, ...] in the array's pixels, grown by dilate."""
    from scipy.ndimage import median_filter
    if array.ndim == 2 or array.shape[2] == 1:
        return {"off_hue": 0.0, "strong": 0.0, "tone": 0.0, "grey": True}
    rgb, left_out = _shrink(array[..., :3], exclude, dilate)
    lab = to_lab(rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    ok = (L > 8) & (L < 95) & ~left_out
    if ok.sum() < 200:
        ok = ~left_out if (~left_out).sum() >= 200 else np.ones_like(L, bool)
    ta, tb = _tone_curve(L, a, b, ok)
    ra, rb = a - ta, b - tb
    tn = np.hypot(ta, tb)
    ua = np.where(tn > 3, ta / np.maximum(tn, 1e-6), 0)
    ub = np.where(tn > 3, tb / np.maximum(tn, 1e-6), 0)
    ma, mb = median_filter(ra, 3), median_filter(rb, 3)
    along = ma * ua + mb * ub
    perp = np.hypot(ma - along * ua, mb - along * ub)
    return {"off_hue": round(float((perp[ok] > OFF_HUE).mean()), 4),
            "strong": round(float((np.hypot(ra, rb)[ok] > STRONG).mean()), 4),
            "tone": round(float(np.median(tn[ok])), 2)}


def is_border(sc: dict) -> bool:
    return (BORDER_OFF[0] <= sc["off_hue"] < BORDER_OFF[1]) or (BORDER_STRONG[0] <= sc["strong"] < BORDER_STRONG[1])


def by_rule(sc: dict) -> str:
    return COLOR if sc["off_hue"] >= COLOUR_OFF or sc["strong"] >= COLOUR_STRONG else BW


def decide(sc: dict, probe=None, vec=None) -> dict:
    """-> the record's colour facts: mode, how it was decided, and the scores."""
    out = dict(sc)
    if probe is not None and vec is not None and is_border(sc):
        p = probe.prob(vec)
        out.update(mode=COLOR if p >= PROBE_COLOUR else BW, by="probe", probe=round(p, 3), rule=by_rule(sc))
    else:
        out.update(mode=by_rule(sc), by="rule")
    return out


class Probe:
    """Logistic regression on L2-normalised embeddings: P(colour)."""

    def __init__(self, model, n_bw: int, n_colour: int):
        self.model, self.n_bw, self.n_colour = model, n_bw, n_colour

    def prob(self, vec) -> float:
        v = np.asarray(vec, np.float32).reshape(1, -1)
        v = v / max(float(np.linalg.norm(v)), 1e-9)
        return float(self.model.predict_proba(v)[0, 1])


def _score_file(path: Path) -> dict:
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        if im.format == "JPEG":
            im.draft("RGB", (SIDE * 4, SIDE * 4))
        im = ImageOps.exif_transpose(im)
        if im.mode in ("L", "1", "I", "I;16"):
            return scores(np.asarray(im.convert("L")))
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            rgba = im.convert("RGBA")
            bg.paste(rgba, mask=rgba.getchannel("A"))
            im = bg
        return scores(np.asarray(im.convert("RGB")))


def sample_scores(samples: Path, rels: list[str], log=print) -> dict[str, dict]:
    """The pixel test of each sample image, cached in <samples>/_embeddings/pastvu_colour.json."""
    cache_file = samples / "_embeddings" / "pastvu_colour.json"
    cache = {}
    try:
        meta = json.loads(cache_file.read_text(encoding="utf-8"))
        if meta.get("version") == VERSION:
            cache = meta.get("files", {})
    except (OSError, ValueError):
        pass
    out, todo = {}, []
    for rel in rels:
        p = samples / rel
        try:
            st = p.stat()
        except OSError:
            continue
        hit = cache.get(rel)
        if hit and hit[0] == st.st_size and hit[1] == st.st_mtime_ns:
            out[rel] = {"off_hue": hit[2], "strong": hit[3]}
        else:
            todo.append((rel, st.st_size, st.st_mtime_ns))
    if todo:
        log(f"  colour test of {len(todo):,} sample images (once; cached) ...")

        def one(t):
            try:
                return t, _score_file(samples / t[0])
            except Exception:  # noqa: BLE001
                return t, None

        with ThreadPoolExecutor(max(1, min(8, os.cpu_count() or 1))) as ex:
            for (rel, size, mtime), sc in ex.map(one, todo):
                if sc is not None:
                    out[rel] = sc
                    cache[rel] = [size, mtime, sc["off_hue"], sc["strong"]]
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({"version": VERSION, "files": cache}, ensure_ascii=False),
                                  encoding="utf-8")
        except OSError:
            pass
    return out


def probe_from_samples(samples: Path, vectors: dict, log=print, min_each: int = 20) -> Probe | None:
    """Train the probe on the samples the pixel test is sure about. vectors: rel -> embedding."""
    from sklearn.linear_model import LogisticRegression
    sc = sample_scores(Path(samples), sorted(vectors), log)
    X, y = [], []
    for rel, s in sc.items():
        if s["off_hue"] <= SURE_BW:
            label = 0
        elif s["off_hue"] >= SURE_COLOUR:
            label = 1
        else:
            continue
        v = np.asarray(vectors[rel], np.float32)
        X.append(v / max(float(np.linalg.norm(v)), 1e-9))
        y.append(label)
    n_colour = int(sum(y))
    n_bw = len(y) - n_colour
    if n_bw < min_each or n_colour < min_each:
        log(f"  colour probe off: the samples hold {n_bw} sure bw and {n_colour} sure colour images "
            f"(at least {min_each} of each needed); borderline photos are decided by the pixel test alone")
        return None
    model = LogisticRegression(C=8, max_iter=2000, class_weight="balanced").fit(np.stack(X), np.array(y))
    log(f"  colour probe: trained on {n_bw:,} bw and {n_colour:,} colour sample images")
    return Probe(model, n_bw, n_colour)
