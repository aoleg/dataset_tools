"""Contact sheets: one JPEG grid of sample photos for a look by eye or by a VLM.

Pillow is imported lazily inside render_sheet: `requirements.txt` stays
`requests`, the shared venv of dataset_tools has Pillow from the other tools,
and a missing Pillow costs only the sheet, with a Russian message.

Layout: a header with the group's title and its member count, then tiles of
TILE_W pixels, COLS per row, each photo fitted into a TILE_W x TILE_H box,
with its number and the first 60 characters of its caption under it (two
lines at most). The font must have Cyrillic: the Windows Arial or Segoe UI by
path, DejaVu on Linux, else Pillow's default font with a notice printed once.
"""
from __future__ import annotations

from pathlib import Path

TILE_W = 320
TILE_H = 240
COLS = 4
PAD = 8
CAPTION_CHARS = 60
CAPTION_LINES = 2
SHEET_NAME = "sheet.jpg"
FONT_SIZE = 14
TITLE_SIZE = 20
BG = (246, 246, 246)
TILE_BG = (225, 225, 225)
INK = (20, 20, 20)
MUTED = (90, 90, 90)

FONT_CANDIDATES = (
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
)

_warned_font = False


class SheetError(Exception):
    """Pillow is missing or the sheet cannot be written: a readable refusal."""


def _font(ImageFont, size: int):
    """A TrueType font with Cyrillic, or Pillow's default (said once)."""
    global _warned_font
    for path in FONT_CANDIDATES:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    if not _warned_font:
        _warned_font = True
        print("[i] Шрифт с кириллицей (Arial, Segoe UI, DejaVu) не найден: подписи на "
              "листе сделаны шрифтом Pillow по умолчанию, кириллица может не отобразиться.")
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                  # Pillow before 10.1 has no size
        return ImageFont.load_default()


def _text_w(draw, text: str, font) -> float:
    try:
        return draw.textlength(text, font=font)
    except AttributeError:
        return draw.textsize(text, font=font)[0]


def wrap_caption(draw, text: str, font, width: int, lines: int = CAPTION_LINES) -> list[str]:
    """Split a caption into at most `lines` lines that fit `width` pixels; the
    last line ends with an ellipsis when something is cut."""
    words = str(text or "").split()
    out, i = [], 0
    while i < len(words) and len(out) < lines:
        cur = words[i]
        i += 1
        while i < len(words) and _text_w(draw, cur + " " + words[i], font) <= width:
            cur += " " + words[i]
            i += 1
        out.append(cur)
    for k, line in enumerate(out):
        cut = k == len(out) - 1 and i < len(words)
        while _text_w(draw, line + ("…" if cut else ""), font) > width and len(line) > 1:
            line, cut = line[:-1], True     # a word longer than the tile
        out[k] = line + ("…" if cut else "")
    return out


def render_sheet(tiles: list[dict], out_path: Path, title: str, subtitle: str = "") -> Path:
    """Render a contact sheet.

    tiles: dicts with `path` (an image file) and `caption` (the raw or cleaned
    text, cut to its first 60 characters here). Returns the path written.
    Raises SheetError when Pillow is missing.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
    except ImportError as e:
        raise SheetError("Для --sheet нужен Pillow, его нет в этом Python. Он ставится "
                         "в общий venv dataset_tools другими инструментами (например, "
                         "install.bat в корне); фото выборки уже скачаны, лист не сделан.") from e
    out_path = Path(out_path)
    font = _font(ImageFont, FONT_SIZE)
    tfont = _font(ImageFont, TITLE_SIZE)
    n = max(1, len(tiles))
    cols = min(COLS, n)
    rows = (n + cols - 1) // cols
    line_h = FONT_SIZE + 4
    cell_h = TILE_H + PAD + CAPTION_LINES * line_h
    head_h = PAD + TITLE_SIZE + 6 + (FONT_SIZE + 6 if subtitle else 0) + PAD
    width = PAD + cols * (TILE_W + PAD)
    height = head_h + rows * (cell_h + PAD) + PAD
    sheet = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(sheet)

    head = wrap_caption(draw, title, tfont, width - 2 * PAD, lines=1)
    draw.text((PAD, PAD), head[0] if head else "", fill=INK, font=tfont)
    if subtitle:
        sub = wrap_caption(draw, subtitle, font, width - 2 * PAD, lines=1)
        draw.text((PAD, PAD + TITLE_SIZE + 6), sub[0] if sub else "", fill=MUTED, font=font)

    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        x = PAD + c * (TILE_W + PAD)
        y = head_h + r * (cell_h + PAD)
        draw.rectangle([x, y, x + TILE_W - 1, y + TILE_H - 1], fill=TILE_BG)
        try:
            with Image.open(tile["path"]) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((TILE_W, TILE_H))
                sheet.paste(im, (x + (TILE_W - im.width) // 2, y + (TILE_H - im.height) // 2))
        except Exception:
            draw.text((x + PAD, y + TILE_H // 2 - FONT_SIZE), "не открывается",
                      fill=MUTED, font=font)
        cap = " ".join(str(tile.get("caption") or "").split())[:CAPTION_CHARS]
        text = "{}. {}".format(i + 1, cap) if cap else "{}. (без подписи)".format(i + 1)
        for j, line in enumerate(wrap_caption(draw, text, font, TILE_W)):
            draw.text((x, y + TILE_H + PAD // 2 + j * line_h), line, fill=INK, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    try:
        sheet.save(tmp, "JPEG", quality=88)
        tmp.replace(out_path)
    except OSError as e:
        raise SheetError("Лист не записан в {}: {}".format(out_path, e)) from e
    return out_path
