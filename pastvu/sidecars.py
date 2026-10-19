"""The two files that go with each image: the one-line caption <cid>.txt for
the captioner (facts only: no URLs, no logins) and the record <cid>.json (the
full PastVu record and what the stream did to the image)."""

from __future__ import annotations

import html
import json
import re

TXT_FIELDS = ["title", "years", "regions", "address", "desc"]
ALL_TXT_FIELDS = TXT_FIELDS + ["direction"]
DESC_CAP = 400
DIRECTIONS = {"n": "north", "ne": "north-east", "e": "east", "se": "south-east", "s": "south", "sw": "south-west",
              "w": "west", "nw": "north-west", "u": "up", "d": "down", "aero": "an aerial view"}

_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_RULE = re.compile(r"([-_=*~.\u2014\u2013])\1{3,}")     # separator lines of repeated punctuation
_LINK = re.compile(r"<a\b|https?://|www\.|#\d{4,}", re.I)
_SEGMENT = re.compile(r"<br\s*/?>|(?<=[.!?])\s+(?=[A-ZА-ЯЁ0-9«\"(])")
_DISAMBIGUATION = re.compile(r"\s*\([^()]*\)\s*$")
# PastVu's own placeholder titles for unidentified photos
PLACEHOLDER_TITLES = re.compile(r"^\s*(где|что|кто|когда)\s*(это|этот|эта|снято)?\s*\?*\s*$|^\s*без названия\s*$",
                                re.I)
PLACEHOLDER_PREFIX = re.compile(r"^\s*(где|что|кто|когда)\s*(это|этот|эта|снято)?\s*\?+\s*", re.I)
# description segments that say nothing about the picture: publication permissions,
# copyright lines, notes on the map point, pointers to other uploads
DESC_DROP = re.compile(r"разрешен|с согласия|©|copyright|все права"
                       r"|точка (стоит |поставлена )?(условн|примерн|приблизит)|ракурс условн"
                       r"|^\s*вариант\b|более (ч[её]ткий|крупный|качественный) вариант|^\s*(см\.|смотри)\s", re.I)


def clean(text) -> str:
    """Plain text from a record field: tags removed, entities decoded, URLs
    removed, whitespace collapsed."""
    if not text:
        return ""
    t = str(text).replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
    t = html.unescape(_TAG.sub(" ", t))
    t = _URL.sub("", t)
    t = _RULE.sub(" ", t)
    return _SPACE.sub(" ", t).strip(" ,;")


def cap(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:")
    return cut + "..."


def description(raw) -> str:
    """The description without the segments that are not about the picture:
    a segment that held a link, a publication permission, a note on the map
    point or a pointer to another upload is dropped whole."""
    if not raw:
        return ""
    keep = []
    for seg in _SEGMENT.split(str(raw)):
        if not seg or not seg.strip() or _LINK.search(seg):
            continue
        text = clean(seg)
        if text and not DESC_DROP.search(text):
            keep.append(text)
    return " ".join(keep)


def region_name(name) -> str:
    """A region title without its disambiguation in brackets: 'Московский район (Минск)' -> 'Московский район'."""
    return _DISAMBIGUATION.sub("", clean(name))


def years_text(photo: dict) -> str:
    y, y2 = photo.get("year"), photo.get("year2")
    if not y:
        return ""
    return str(y) if not y2 or y2 == y else f"{y}-{y2}"


def regions_text(photo: dict) -> str:
    regions = photo.get("regions") or []
    local = [region_name(r.get("title_local")) for r in regions if r.get("title_local")]
    english = [region_name(r.get("title_en")) for r in regions if r.get("title_en")]
    if not local and not english:
        return ""
    if local == english or not local:
        return ", ".join(english)
    return ", ".join(local) + (f" ({', '.join(english)})" if english else "")


def caption(photo: dict, fields: list[str] | None = None) -> str:
    """The one-line caption of a record, parts in the order of fields, empty
    parts left out, each part ending with a full stop."""
    fields = fields or TXT_FIELDS
    title = clean(photo.get("title"))
    if PLACEHOLDER_TITLES.match(title):
        title = ""
    else:
        title = PLACEHOLDER_PREFIX.sub("", title).strip()
    said = " ".join([title, regions_text(photo)]).casefold()
    parts = []
    for f in fields:
        if f == "title":
            v = title
        elif f == "years":
            v = years_text(photo)
        elif f == "regions":
            v = regions_text(photo)
        elif f == "address":
            v = clean(photo.get("address"))
            if v and v.casefold().strip(" .") in said:
                v = ""
        elif f == "desc":
            v = cap(description(photo.get("desc")), DESC_CAP)
            if v and v.casefold().strip(" .") in said:
                v = ""
        elif f == "direction":
            d = DIRECTIONS.get(str(photo.get("dir") or "").lower())
            v = (f"Camera facing {d}" if d and d not in ("up", "down", "an aerial view") else
                 f"Camera pointing {d}" if d in ("up", "down") else "An aerial view" if d else "")
        else:
            raise ValueError(f"unknown caption field {f!r}; choose from {', '.join(ALL_TXT_FIELDS)}")
        if v:
            parts.append(v if v[-1] in ".!?…" else v + ".")
    return " ".join(parts)


def record_json(cid: int, page: str, result: dict, download: dict, processing: dict, classification: dict | None,
                tool: dict) -> str:
    doc = {"cid": cid, "page": page, "record": result.get("photo"), "can": result.get("can"), "download": download,
           "processing": processing, "classification": classification, "tool": tool}
    return json.dumps(doc, ensure_ascii=False, indent=1, default=str)
