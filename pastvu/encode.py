"""The save rules, all in memory: bytes in, bytes out.

- Nothing changed: the downloaded bytes verbatim.
- Only lossless changes (band, borders, trim): JPEG cropped in the DCT domain
  with libjpeg-turbo (PyTurboJPEG; jpeglib reads and writes files only), EXIF
  kept with the new size and without its thumbnail; PNG and other lossless
  formats cropped to the exact pixels in the same format and mode; a lossy
  non-JPEG format becomes PNG.
- A destructive change (painted watermark): JPEG at quality 97 without chroma
  subsampling, or PNG with --png, from the upright array, EXIF orientation
  reset.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

import process as pr
from process import rb

HERE = Path(__file__).resolve().parent
TURBOJPEG_DLL = HERE / "bin" / "turbojpeg.dll"
JPEG_QUALITY = 97
PNG_LEVEL = 6
EXTENSIONS = {"JPEG": ".jpg", "MPO": ".jpg", "PNG": ".png", "GIF": ".gif", "BMP": ".bmp", "TIFF": ".tif",
              "WEBP": ".webp"}

_tj = None


def turbo():
    global _tj
    if _tj is None:
        from turbojpeg import TurboJPEG
        if not TURBOJPEG_DLL.is_file():
            raise SystemExit(f"{TURBOJPEG_DLL} is missing. Run install.bat in the pastvu folder.")
        _tj = TurboJPEG(str(TURBOJPEG_DLL))
    return _tj


def exif_for(exif: bytes | None, width: int, height: int, reset_orientation: bool) -> bytes | None:
    """The EXIF with the new pixel size, without its thumbnail, and with the
    orientation reset when the pixels are written upright."""
    if not exif:
        return None
    try:
        ex = Image.Exif()
        ex.load(exif)
        sub = ex.get_ifd(0x8769)
        if sub:
            sub[0xA002], sub[0xA003] = width, height
        if reset_orientation and 274 in ex:
            ex[274] = 1
        return ex.tobytes()
    except Exception:  # noqa: BLE001 - a damaged EXIF block is dropped
        return None


def jpeg_crop(data: bytes, head: dict, stored_box) -> bytes:
    """A lossless crop of JPEG bytes. The origin must be on the MCU grid
    (process.align_box puts it there); the far edges are exact."""
    x0, y0, x1, y1 = stored_box
    mw, mh = head["mcu"] or (8, 8)
    if x0 % mw or y0 % mh:
        raise ValueError(f"crop origin {x0},{y0} is not on the {mw}x{mh} block grid")
    out = turbo().crop(data, x0, y0, x1 - x0, y1 - y0)
    exif = exif_for(head.get("exif"), x1 - x0, y1 - y0, reset_orientation=False)
    return rb.rewrite_jpeg_header(out, exif) if head.get("exif") else out


def exact_crop(data: bytes, head: dict, stored_box) -> tuple[bytes, str]:
    """An exact pixel crop of a lossless image in its own format and mode, or
    of a lossy non-JPEG image as PNG. -> (bytes, format)."""
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        fmt = im.format if head["kind"] == rb.KIND_LOSSLESS else "PNG"
        info = dict(im.info)
        cropped = im.crop(tuple(stored_box))
        kw = {}
        if info.get("icc_profile"):
            kw["icc_profile"] = info["icc_profile"]
        if info.get("transparency") is not None:
            kw["transparency"] = info["transparency"]
        exif = exif_for(info.get("exif"), cropped.width, cropped.height, reset_orientation=False)
        if exif:
            kw["exif"] = exif
        if fmt == "PNG":
            text = getattr(im, "text", None) or {}
            if text:
                pnginfo = PngImagePlugin.PngInfo()
                for k, v in text.items():
                    pnginfo.add_text(k, v)
                kw["pnginfo"] = pnginfo
            kw["compress_level"] = PNG_LEVEL
        buf = io.BytesIO()
        cropped.save(buf, fmt, **kw)
    return buf.getvalue(), fmt


def array_bytes(array: np.ndarray, head: dict, png: bool) -> tuple[bytes, str]:
    """A destructively changed upright array as JPEG q97 (or PNG). -> (bytes, format)."""
    im = Image.fromarray(np.ascontiguousarray(array))
    kw = {}
    exif = exif_for(head.get("exif"), im.width, im.height, reset_orientation=True)
    if exif:
        kw["exif"] = exif
    if head.get("icc"):
        kw["icc_profile"] = head["icc"]
    buf = io.BytesIO()
    if png:
        im.save(buf, "PNG", compress_level=PNG_LEVEL, **kw)
        return buf.getvalue(), "PNG"
    im.save(buf, "JPEG", quality=JPEG_QUALITY, subsampling=0, optimize=True, **kw)
    return buf.getvalue(), "JPEG"


def encode(data: bytes, head: dict, plan: dict, painted: np.ndarray | None, png: bool) -> tuple[bytes, str, str]:
    """The output bytes of one image. -> (bytes, extension, write mode:
    verbatim, lossless, exact, png or array)."""
    if painted is not None:
        out, fmt = array_bytes(painted, head, png)
        return out, EXTENSIONS[fmt], "array"
    if not plan.get("stored_box"):
        return data, EXTENSIONS.get(head["format"], ".bin"), "verbatim"
    if head["kind"] == rb.KIND_JPEG:
        return jpeg_crop(data, head, plan["stored_box"]), ".jpg", "lossless"
    out, fmt = exact_crop(data, head, plan["stored_box"])
    return out, EXTENSIONS.get(fmt, ".png"), "exact" if fmt == head["format"] else "png"


def size_of(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as im:
        return im.size


__all__ = ["encode", "jpeg_crop", "exact_crop", "array_bytes", "turbo", "size_of", "pr"]
