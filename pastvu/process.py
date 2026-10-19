"""What happens to one image between the download and the encoder, on bytes,
arrays and dicts in memory: the header, the decode, the PastVu band, the
border analysis (remove_borders), the composed crop on the JPEG block grid,
the watermark boxes left inside it, and the resize for the classifier.

Coordinates: "display" is the upright image as EXIF shows it, "stored" is the
file's pixel grid. The models see display pixels; the lossless crop is made in
stored pixels. The band and the border analysis are applied first, and the
watermark detector and the classifier see the picture without the band, so the
band's own text ("pastvu.com/<cid> uploaded by ...") is never taken for a
watermark.
"""

from __future__ import annotations

import io
import warnings
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _tool in ("remove_borders", "watermark", "classify"):
    if str(REPO / _tool) not in sys.path:
        sys.path.insert(0, str(REPO / _tool))
import remove_borders as rb  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
# Some PastVu files carry a broken EXIF block; Pillow warns on every read and the
# warnings flood the live display. The block is dropped where it cannot be read.
warnings.filterwarnings("ignore", message="Corrupt EXIF data", category=UserWarning)
warnings.filterwarnings("ignore", message="Possibly corrupt EXIF", category=UserWarning)

BAND_GREY_SPREAD = 14          # the band is a neutral grey: channels within this of each other
BAND_MIN_SHARE = 0.50          # share of band pixels on the band colour (the rest is the white text)
BAND_TOL = 24                  # per-channel distance from the band's median colour
BAND_MIN_LUMA, BAND_MAX_LUMA = 10, 200    # a custom uploader signature comes on a near-black band


class Rejected(Exception):
    """An image the stream does not take, with the reason (status skipped or failed)."""

    def __init__(self, reason: str, failed: bool = False):
        super().__init__(reason)
        self.reason = reason
        self.failed = failed


# --- header and decode --------------------------------------------------------------

def head_of(data: bytes) -> dict:
    """Format, kind, mode, size, JPEG MCU, EXIF orientation, EXIF and ICC bytes
    of image bytes, from the header only."""
    h = {"format": "", "kind": "", "mode": "", "bits": 8, "stored": [0, 0], "mcu": None, "orientation": 1,
         "frames": 1, "exif": None, "icc": None}
    with Image.open(io.BytesIO(data)) as im:
        fmt = im.format or ""
        h.update(format=fmt, mode=im.mode, stored=list(im.size))
        try:
            o = im.getexif().get(274, 1) or 1
            h["orientation"] = o if o in range(1, 9) else 1
        except Exception:  # noqa: BLE001 - a damaged EXIF block is not fatal
            pass
        h["frames"] = 1 if fmt == "MPO" else (getattr(im, "n_frames", 1) or 1)
        if fmt in ("JPEG", "MPO"):
            h["kind"] = rb.KIND_JPEG
            layer = getattr(im, "layer", None)
            if layer:
                h["mcu"] = [8 * max(c[1] for c in layer), 8 * max(c[2] for c in layer)]
        elif fmt in rb.LOSSLESS_FORMATS:
            h["kind"] = rb.KIND_LOSSLESS
        else:
            h["kind"] = rb.KIND_LOSSY
        if im.mode.startswith("I;16") or im.mode in ("I", "F"):
            h["bits"] = 16
        h["exif"] = im.info.get("exif")
        h["icc"] = im.info.get("icc_profile")
    return h


def decode(data: bytes, head: dict) -> np.ndarray:
    """The upright RGB uint8 pixels, transparency composited over white."""
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        if im.mode.startswith("I;16") or im.mode in ("I", "F"):
            a = np.asarray(im.convert("I"), dtype=np.int64)
            a = np.clip(a >> 8 if a.max() > 255 else a, 0, 255).astype(np.uint8)
            stored = np.stack([a, a, a], axis=-1)
        elif "A" in im.mode or "transparency" in im.info:
            rgba = im.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            stored = np.asarray(Image.alpha_composite(bg, rgba).convert("RGB"))
        else:
            stored = np.asarray(im.convert("RGB"))
    return np.ascontiguousarray(rb.to_upright(stored, head["orientation"]))


# --- boxes ----------------------------------------------------------------------------

def display_to_stored(box, orientation: int, sw: int, sh: int) -> list[int]:
    """A box in upright pixels -> the same box in stored pixels (the inverse of
    remove_borders.stored_to_display)."""
    X0, Y0, X1, Y1 = box
    if orientation == 2:
        return [sw - X1, Y0, sw - X0, Y1]
    if orientation == 3:
        return [sw - X1, sh - Y1, sw - X0, sh - Y0]
    if orientation == 4:
        return [X0, sh - Y1, X1, sh - Y0]
    if orientation == 5:
        return [Y0, X0, Y1, X1]
    if orientation == 6:
        return [Y0, sh - X1, Y1, sh - X0]
    if orientation == 7:
        return [sw - Y1, sh - X1, sw - Y0, sh - X0]
    if orientation == 8:
        return [sw - Y1, X0, sw - Y0, X1]
    return [X0, Y0, X1, Y1]


def upright_size(head: dict) -> tuple[int, int]:
    w, h = head["stored"]
    return (h, w) if head["orientation"] in (5, 6, 7, 8) else (w, h)


def intersect(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return [x0, y0, x1, y1] if x1 > x0 and y1 > y0 else None


def shift(box, dx, dy):
    return [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]


def area(box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def cut_sides(inner, outer) -> set:
    """The display sides on which inner lies inside outer."""
    return {s for s, cut in zip("LTRB", (inner[0] > outer[0], inner[1] > outer[1],
                                         inner[2] < outer[2], inner[3] < outer[3])) if cut}


def align_box(display_box, head: dict, inward_display_sides: set) -> tuple[list, list]:
    """The crop on the JPEG block grid in stored pixels: the stored origin moves
    inward on a side that a cut set (so the cut thing goes completely), outward
    elsewhere; the far edges stay exact. -> (display box, stored box). A
    non-JPEG keeps its exact box. The rule of the pipeline tool."""
    o, (sw, sh) = head["orientation"], head["stored"]
    stored = display_to_stored([int(v) for v in display_box], o, sw, sh)
    if not head.get("mcu"):
        return [int(v) for v in display_box], [int(v) for v in stored]
    mw, mh = head["mcu"]
    left_src = {1: "L", 2: "R", 3: "R", 4: "L", 5: "T", 6: "T", 7: "B", 8: "B"}[o]
    top_src = {1: "T", 2: "T", 3: "B", 4: "B", 5: "L", 6: "R", 7: "R", 8: "L"}[o]
    x0, y0, x1, y1 = stored
    nx0 = -(-x0 // mw) * mw if left_src in inward_display_sides else (x0 // mw) * mw
    ny0 = -(-y0 // mh) * mh if top_src in inward_display_sides else (y0 // mh) * mh
    if nx0 >= x1:
        nx0 = (x0 // mw) * mw
    if ny0 >= y1:
        ny0 = (y0 // mh) * mh
    stored = [int(nx0), int(ny0), int(x1), int(y1)]
    return [int(v) for v in rb.stored_to_display(stored, o, sw, sh)], stored


def crop_array(array: np.ndarray, box) -> np.ndarray:
    if not box:
        return array
    x0, y0, x1, y1 = box
    return np.ascontiguousarray(array[y0:y1, x0:x1])


# --- the PastVu band ------------------------------------------------------------------

def band_rows(head: dict, record: dict) -> int:
    """How many stored rows at the bottom are the PastVu band, from the record:
    the original variant is the picture (w x h) with waterh rows appended.
    Raises Rejected when the file does not have the size the record gives."""
    sw, sh = head["stored"]
    w, h, wh = int(record.get("w") or 0), int(record.get("h") or 0), int(record.get("waterh") or 0)
    if (sw, sh) == (w, h + wh):
        return wh
    if (sw, sh) == (w, h):
        return 0
    raise Rejected(f"size {sw}x{sh} is not the record's {w}x{h} plus a band of {wh}", failed=True)


def verify_band(array: np.ndarray, head: dict, rows: int) -> dict:
    """Are the last `rows` stored rows the band: a neutral grey with light text
    on it? -> {"ok", "rows", "note"}."""
    out = {"ok": False, "rows": rows, "note": ""}
    if rows <= 0:
        out["note"] = "no band"
        return out
    stored = rb.to_stored(array, head["orientation"])
    sh = stored.shape[0]
    if rows >= sh - 8:
        out["note"] = "the band would take the whole image"
        return out
    band = stored[sh - rows:sh].reshape(-1, 3).astype(np.int16)
    med = np.median(band, axis=0)
    if med.max() - med.min() > BAND_GREY_SPREAD or not (BAND_MIN_LUMA <= med.mean() <= BAND_MAX_LUMA):
        out["note"] = f"band colour {med.astype(int).tolist()} is not a neutral grey"
        return out
    share = float((np.abs(band - med).max(axis=1) <= BAND_TOL).mean())
    if share < BAND_MIN_SHARE:
        out["note"] = f"only {share:.0%} of the band is on its colour"
        return out
    out.update(ok=True, note=f"grey {med.astype(int).tolist()}, {share:.0%} on colour")
    return out


def picture_box(head: dict, rows: int) -> list[int]:
    """The display box of the picture without the band (the whole image when rows is 0)."""
    sw, sh = head["stored"]
    return [int(v) for v in rb.stored_to_display([0, 0, sw, sh - rows], head["orientation"], sw, sh)]


# --- borders ----------------------------------------------------------------------------

def borders_plan(picture: np.ndarray, options: dict) -> dict:
    """remove_borders' plan of the picture (the upright array without the band).
    The header is that of an upright lossless image: the block alignment is
    done once, on the composed box, in stored pixels of the real file."""
    h, w = picture.shape[:2]
    head = {"format": "PNG", "kind": rb.KIND_LOSSLESS, "mode": "RGB", "bits": 8, "width": w, "height": h,
            "mcu": None, "orientation": 1, "frames": 1, "error": ""}
    return rb.plan(picture, head, options)


def describe_borders(plan: dict) -> dict:
    result = plan.get("result") or {}
    cuts = result.get("cuts") or []
    return {"action": plan.get("action", ""), "cuts": [{"side": c["side"], "kind": c["kind"], "tone": c.get("tone", ""),
                                                        "depth": c["depth"]} for c in cuts],
            "text": rb.describe_cuts(cuts), "note": result.get("review") or (result.get("plan") or {}).get("note") or ""}


# --- compose ----------------------------------------------------------------------------

def compose(head: dict, pic_box: list, band_ok: bool, borders: dict | None, wm_boxes: list, wm_opts: dict) -> dict:
    """One crop from the band, the border cuts and a watermark trim, put on the
    block grid. pic_box: the picture without the band, in display pixels of the
    full image; borders: remove_borders' plan in picture coordinates;
    wm_boxes: the detector's boxes in picture coordinates.
    -> {"box" (display, None for the whole image), "stored_box", "wm_inside"
        (boxes to paint, in crop coordinates), "trim", "changed", "size"}"""
    import watermark as wm
    W, H = upright_size(head)
    full = [0, 0, W, H]
    ox, oy = pic_box[0], pic_box[1]
    box = list(pic_box) if band_ok else list(full)
    inward = cut_sides(box, full)
    changed = ["band"] if band_ok and box != full else []
    if borders and borders.get("action") == "crop" and borders.get("display_box"):
        bx = shift(borders["display_box"], ox, oy)
        nb = intersect(box, bx)
        if nb and nb != box:
            inward |= cut_sides(nb, box)
            box = nb
            changed.append("borders")
    paint_conf = wm_opts.get("paint_conf", 0.0)
    boxes = [shift(b[:4], ox, oy) + [b[4]] for b in wm_boxes if b[4] >= paint_conf]
    trim = False
    if boxes and wm_opts.get("mode") == "trim":
        mask = wm.boxes_mask((H, W), boxes, wm_opts.get("dilate", wm.DEFAULT_DILATE))
        rect = wm.largest_clear_rect(wm.mask_boxes(mask), W, H)
        rect = intersect(rect, box) if rect else None
        if rect and len(boxes) <= 20 and area(rect) >= wm_opts.get("trim_min_keep", wm.DEFAULT_TRIM_MIN_KEEP) * area(box):
            inward |= cut_sides(rect, box)
            box = rect
            trim = True
            changed.append("trim")
    final, stored_box = align_box(box, head, inward)
    final = intersect(final, full) or full
    out = {"box": None, "stored_box": None, "wm_inside": [], "trim": trim, "changed": changed,
           "size": [final[2] - final[0], final[3] - final[1]], "inward": sorted(inward)}
    if final != full:
        out["box"], out["stored_box"] = final, stored_box
    elif "band" in changed or "borders" in changed:
        changed[:] = [c for c in changed if c not in ("band", "borders")]
    if boxes and not trim:
        inside = []
        for b in boxes:
            cut = intersect(b[:4], final)
            if cut:
                inside.append(shift(cut, -final[0], -final[1]) + [b[4]])
        if inside:
            out["wm_inside"] = inside
            changed.append("paint")
    return out


# --- the classifier's view ------------------------------------------------------------

def naflex_resize(array: np.ndarray, patches: int) -> np.ndarray:
    """The array at the NaFlex size classify's encoder takes (classify.target_size)."""
    import classify as cl
    h, w = array.shape[:2]
    th, tw = cl.target_size(h, w, patches)
    if (th, tw) == (h, w):
        return np.array(array, dtype=np.uint8)          # writable: torch.from_numpy warns on a read-only array
    return np.array(Image.fromarray(array).resize((tw, th), Image.BILINEAR), dtype=np.uint8)
