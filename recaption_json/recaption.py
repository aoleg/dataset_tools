#!/usr/bin/env python3
"""
recaption: a short user-style prompt for every long caption of a dataset, by a
local LLM, written as a second sidecar (<stem>.short) next to the image.

    python recaption.py D:\\photos
    python recaption.py D:\\photos --batch 1 --limit 50 --log D:\\photos_recaption_log.csv

An item is every <stem>.txt (or --ext) that has an image next to it. Each item
gets one style letter (A trailing tags, B tag list, C plain phrase), chosen
from a hash of the stem, so a rerun gives the same letter. There is no D (a
request, "Generate a ...": a prompt only describes the picture) and no E
(Russian: every short is in English). The system prompt is
prompt/recaption.txt; the user message holds --batch numbered items, and the
model answers a JSON array [{"id": n, "short": "..."}]. Every short is checked
(length, English, era, color mode, color words, concept names, requests,
banned phrases); an item that fails, or that the
reply leaves out, is asked for again on its own with the reason, at most twice.
A short that never passes is not written, and the log says why.

The .txt is never changed. An item with a non-empty .short is skipped (unless
--overwrite), so a stopped run continues where it stopped.

Usage:    python recaption.py DATASET [--ext .txt] [--endpoint URL] [--model auto]
          [--batch 10] [--styles A=40,B=35,C=25] [--limit N]
          [--overwrite] [--dry-run] [--log recaption_log.csv] [--temperature 0.3]
Install:  httpx (install.bat)
"""
from __future__ import annotations

import argparse
import codecs
import csv
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
PROMPT_FILE = TOOL_DIR / "prompt" / "recaption.txt"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
SHORT_EXT = ".short"
EXCLUDES = {"_controls"}                # face_masks folders: masks, never items
DEFAULT_ENDPOINT = "http://localhost:5001/v1/"
DEFAULT_STYLES = "A=40,B=35,C=25"
DEFAULT_BATCH = 10
STYLE_LETTERS = "ABC"
DROPPED_STYLES = {"D": "D, the request style, was dropped", "E": "E, the Russian style, was dropped: shorts are English"}
MAX_RETRIES = 2                         # single-mode retries after the first try
TOKENS_PER_ITEM = 160                   # max_tokens = this per item + TOKENS_BASE
TOKENS_BASE = 96
TIMEOUT = 600.0                         # seconds for one reply
LOG_FIELDS = ["path", "style", "status", "attempts", "reason", "words", "concepts", "mode", "seconds"]
FIRST_TRY = "first try: "               # a retried item's log reason: why its first answer was rejected

# End-of-turn tokens to stop on (pastvu/autocaption.py); the API allows at most four.
STOP_SEQUENCES = ["<|return|>", "<|im_end|>", "<end_of_turn>", "<|eot_id|>"]
THINKING_BLOCK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.S | re.I)

# --- validation tables ----------------------------------------------------------------------

WORD_LIMITS = {"A": (8, 20), "C": (8, 20)}
TAG_LIMITS = (6, 12)
TAG_MAX_WORDS = 5
# Shorts are English: Cyrillic only inside quotes, where it is a sign's text as the long caption gives it.
CYRILLIC = re.compile(r"[а-яё]+", re.I)
QUOTED = re.compile(r"\"[^\"]*\"|«[^»]*»|“[^”]*”|„[^“”]*[“”]")

# "mid-20th century" too: some long captions date the photo that way and nothing finer; and 20xx for the
# later photos of Soviet things a dataset may hold ("A color photograph from 2009 of a Soviet-era trolleybus").
ERA_EN = re.compile(r"(?<![\w-])(?:mid-)?(?:(?:19|20)\d0s|'?\d0s|(?:19|20)\d\d)(?![\w])|\bsoviet era\b"
                    r"|\b20th[- ]century\b", re.I)

# Color-mode tokens per mode: "bw" black-and-white, "sepia" (its own words or black-and-white), "color",
# "film" film still.
MODE_TOKENS = {
    "bw": ("black-and-white photo", "black and white photo", "b&w photo", "monochrome photo"),
    "color": ("color photo", "colour photo", "vintage color photo", "old color photo"),
    "film": ("film still", "movie still"),
}
MODE_TOKENS["sepia"] = ("sepia photo", "sepia-tone photo", "sepia-toned photo")
# A painting, an illustration or a drawing names its medium instead of a color mode, and is never a photo.
MEDIUM_TOKENS = {"painting": ("painting",), "illustration": ("illustration",), "drawing": ("drawing",)}
PHOTO_WORD = re.compile(r"\bphoto|\bsnapshot|\b(?:film|movie) still", re.I)

# No color words in a short whose long caption begins "A black-and-white"; light and dark are allowed.
COLOR_WORDS = ("red", "orange", "yellow", "green", "blue", "navy", "purple", "violet", "pink", "magenta",
               "brown", "beige", "khaki", "white", "black", "grey", "gray", "silver", "silvery", "gold",
               "golden", "blond", "blonde", "crimson", "scarlet", "maroon", "burgundy", "teal", "turquoise",
               "cyan", "azure", "ivory")
COLOR_RE = re.compile(r"\b(?:%s)(?:ish)?\b" % "|".join(COLOR_WORDS), re.I)
# Proper names with a color word in them, which are places, not colors.
COLOR_NAMES = re.compile(r"\b(?:red square|red army|black sea|white sea)\b", re.I)

TEMPLATE_OPENINGS = ("a black-and-white photograph from", "a color photograph from", "a color film still")
BANNED = re.compile(r"\b(?:this image|this photograph|appears to|seems|likely|possibly|probably)\b", re.I)
# A short prompt only describes the picture: no request or instruction anywhere, no imperative verb first.
REQUEST = re.compile(
    r"\b(?:generate|create|render|gimme|give me|show me|make me|draw me|do an?|i need|i want|i'd like|i would like"
    r"|please|can you|could you|would you|let's|let me)\b", re.I)
IMPERATIVE_FIRST = re.compile(r"^\W*(?:make|draw|show|paint|produce|design|depict|imagine|visualize|give|get|take)\b",
                              re.I)

# Concept names (mirroring prompt/recaption.txt): name, how it shows in a short (the Russian forms catch
# a quoted sign, the only Cyrillic a short may hold), and the condition the long caption must meet. "pioneer tie" and "Soviet school uniform" are not
# concept names but are bound to the long caption's words by the prompt's school rules.
TIE = r"pioneer\s+(?:tie|neckerchief|scarf)"
OCTOBRIST = r"octobrist\w*['’]?s?\s+(?:star|badge|pin)"
UNIFORM = r"school\s+uniform"
AIRCRAFT = r"aircraft|airplane|aeroplane|airliner|\bplane\b|\bjet\b"


def _has(pattern):
    rx = re.compile(pattern, re.I)
    return lambda long: bool(rx.search(long))


def _octobrists(long):
    return bool(re.search(OCTOBRIST, long, re.I)) and not re.search(TIE, long, re.I)


def _uniform_only(long):
    return bool(re.search(UNIFORM, long, re.I)) and not _octobrists(long) and not re.search(TIE, long, re.I)


def _aeroflot(long):
    return bool(re.search(r"aeroflot|аэрофлот", long, re.I)) and bool(re.search(AIRCRAFT, long, re.I))


CONCEPTS = [
    ("pioneers", r"\bpioneers\b|(?<![а-яё])пионер(?:ы|ов|ами|ам|ах)(?![а-яё])", _has(TIE)),
    ("pioneer tie", r"\bpioneer\s+ties?\b|пионерск\w*\s+галстук", _has(TIE)),
    ("Octobrists", r"\boctobrists?\b|\bfirst[- ]graders?\b|октябрят\w*|первоклассни\w*", _octobrists),
    ("schoolchildren in Soviet school uniform",
     r"schoolchildren\s+in\s+(?:a\s+)?soviet\s+school\s+uniforms?|школьники\s+в\s+советской\s+школьной\s+форме",
     _uniform_only),
    ("Soviet school uniform", r"school\s+uniforms?|школьн\w*\s+форм\w*", _has(UNIFORM)),
    ("pedal cars", r"\bpedal[- ]cars?\b|педальн\w*\s+(?:машин|автомобил)\w*",
     _has(r"pedal[- ](?:powered\s+|driven\s+)?cars?\b|toy\s+car\b[^.]{0,80}steering\s+wheel")),
    ("GAZ-21 Volga", r"\bgaz-?\s?21\b|(?<![а-яё])газ-?\s?21\b", _has(r"\bgaz-?\s?21\b")),
    ("GAZ-24 Volga", r"\bgaz-?\s?24\b|(?<![а-яё])газ-?\s?24\b", _has(r"\bgaz-?\s?24\b")),
    ("GAZ-M20 Pobeda", r"\bgaz-?\s?m-?20\b|\bpobeda\b|газ-?\s?м-?20\b|(?<![а-яё])победа(?![а-яё])",
     _has(r"\bgaz-?\s?m-?20\b|\bpobeda\b")),
    ("Moskvich-408", r"\bmoskvich-?\s?408\b|москвич-?\s?408\b", _has(r"\bmoskvich-?\s?408\b")),
    ("VAZ-2101 Zhiguli", r"\bvaz-?\s?2101\b|\b2101\b|ваз-?\s?2101\b",
     _has(r"\bvaz-?\s?2101\b|\bzhiguli\s+2101\b|\blada\s+1200\b")),
    ("LiAZ-677", r"\bliaz-?\s?677\b|лиаз-?\s?677\b", _has(r"\bliaz-?\s?677\b")),
    ("ZiU-9 trolleybus", r"\bziu-?\s?9\b|(?<![а-яё])зиу-?\s?9\b", _has(r"\bziu-?\s?(?:9|682)\b")),
    ("Tu-154", r"\btu-?\s?154\b|(?<![а-яё])ту-?\s?154\b", _has(r"\btu-?\s?154\b")),
    ("Il-62", r"\bil-?\s?62\b|(?<![а-яё])ил-?\s?62\b", _has(r"\bil-?\s?62\b")),
    ("Aeroflot airliner", r"aeroflot|аэрофлот", _aeroflot),
    ("Soviet communal kitchen", r"communal\s+kitchen|коммунальн\w*\s+кухн\w*", _has(r"communal\s+kitchen")),
    ("department store interior", r"department\s+store|универмаг\w*", _has(r"department\s+store")),
    ("Soviet brutalist building", r"brutalis[mt]|брутали\w*", _has(r"brutalis[mt]")),
]
CONCEPTS = [(name, re.compile(rx, re.I), cond) for name, rx, cond in CONCEPTS]


# --- items and styles -----------------------------------------------------------------------

@dataclass
class Item:
    txt: Path
    style: str
    long: str = ""
    short: str = ""
    status: str = ""
    attempts: int = 0
    reason: str = ""
    mode: str = ""
    seconds: float = 0.0
    concepts: list = field(default_factory=list)

    @property
    def short_path(self) -> Path:
        return self.txt.with_suffix(SHORT_EXT)


def parse_styles(spec: str) -> list[tuple[str, float]]:
    """'A=30,B=25,...' -> [(letter, share)], shares summing to 1; a zero share drops the letter."""
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"([A-Za-z])\s*=\s*(\d+(?:\.\d+)?)", part)
        if not m:
            raise ValueError(f"bad style share {part!r}; expected e.g. A=30")
        if m.group(1).upper() not in STYLE_LETTERS:
            dropped = f" ({DROPPED_STYLES[m.group(1).upper()]})" if m.group(1).upper() in DROPPED_STYLES else ""
            raise ValueError(f"no style {m.group(1).upper()}{dropped}; the styles are {', '.join(STYLE_LETTERS)}")
        out[m.group(1).upper()] = float(m.group(2))
    total = sum(out.values())
    if total <= 0:
        raise ValueError("the style shares add up to zero")
    return [(k, out[k] / total) for k in STYLE_LETTERS if out.get(k, 0) > 0]


def style_for(stem: str, styles: list[tuple[str, float]]) -> str:
    """The style letter of a stem: a point in [0, 1) from its SHA-256, placed on the cumulative shares."""
    h = int.from_bytes(hashlib.sha256(stem.encode("utf-8")).digest()[:8], "big") / 2 ** 64
    cum = 0.0
    for letter, share in styles:
        cum += share
        if h < cum:
            return letter
    return styles[-1][0]


def read_text(path: Path) -> str:
    """UTF-16 with a BOM (what Notepad calls "Unicode"), UTF-8 (a BOM is dropped), else cp1251, which some
    hand-written files are in."""
    data = path.read_bytes()
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        text = data.decode("utf-16")
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("cp1251")
    return text.replace("\r\n", "\n")


def has_image(txt: Path) -> bool:
    return any(txt.with_suffix(ext).is_file() for ext in IMAGE_EXTS)


def find_items(root: Path, ext: str, styles) -> list[Item]:
    """Every <stem><ext> under root with an image of the same stem next to it, in path order."""
    items = []
    for txt in sorted(root.rglob(f"*{ext}"), key=lambda p: str(p).lower()):
        if not txt.is_file() or EXCLUDES.intersection(txt.relative_to(root).parts[:-1]):
            continue
        if has_image(txt):
            items.append(Item(txt=txt, style=style_for(txt.stem, styles)))
    return items


def has_short(item: Item) -> bool:
    p = item.short_path
    return p.is_file() and p.stat().st_size > 0


# --- the request ----------------------------------------------------------------------------

def user_message(items: list[Item], reasons: dict[int, str] | None = None) -> str:
    """Numbered items, 1..n; reasons (by number) are appended to a retried item."""
    blocks = []
    for n, it in enumerate(items, 1):
        block = f"Item {n}\nStyle: {it.style}\nLong caption: {' '.join(it.long.split())}"
        if reasons and n in reasons:
            block += (f"\nYour previous short prompt for this item was rejected: {reasons[n]}. "
                      f"Write a new one that follows every rule.")
        blocks.append(block)
    return "\n\n".join(blocks)


def extract_json_array(text: str) -> list | None:
    """The first JSON array of objects in a reply (code fences and text around it are skipped);
    a lone object with "short" counts as an array of one."""
    dec = json.JSONDecoder()
    found_object = None
    for m in re.finditer(r"[\[{]", text):
        try:
            val, _ = dec.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(val, list) and val and all(isinstance(v, dict) for v in val):
            return val
        if found_object is None and isinstance(val, dict) and "short" in val:
            found_object = val
    return [found_object] if found_object is not None else None


def map_reply(reply: list | None, n: int) -> dict[int, str]:
    """{number: short} for the ids 1..n the reply answers with a non-empty string."""
    out = {}
    for obj in reply or []:
        try:
            i = int(obj.get("id"))
        except (TypeError, ValueError):
            continue
        short = obj.get("short")
        if 1 <= i <= n and isinstance(short, str) and short.strip() and i not in out:
            out[i] = " ".join(short.split())
    if n == 1 and not out and reply and len(reply) == 1 and isinstance(reply[0].get("short"), str):
        out[1] = " ".join(reply[0]["short"].split())     # a single item answered without an id
    return out


class EndpointDown(Exception):
    """The endpoint does not answer: stop the run (a rerun continues)."""


class Rejected(Exception):
    """The endpoint answered this request with an error."""


class ChatClient:
    """A thin client for an OpenAI-compatible chat completions endpoint (KoboldCpp, llama.cpp, LM Studio)."""

    def __init__(self, endpoint: str, model: str = "auto", temperature: float = 0.3, http=None):
        import httpx
        self._httpx = httpx
        self.base = endpoint.rstrip("/")
        self.temperature = temperature
        self.http = http or httpx.Client(timeout=httpx.Timeout(connect=15.0, read=TIMEOUT, write=60.0, pool=30.0))
        self.model = None if model in ("", "auto") else model
        self.send_reasoning_effort = True

    def resolve_model(self) -> str:
        if self.model:
            return self.model
        try:
            r = self.http.get(f"{self.base}/models", timeout=15)
            r.raise_for_status()
            data = r.json().get("data") or []
        except (self._httpx.HTTPError, ValueError) as e:
            raise EndpointDown(f"{self.base} does not answer ({e}); start the LLM server or pass --endpoint") from e
        if not data:
            raise EndpointDown(f"{self.base}/models lists no model")
        self.model = data[0]["id"]
        return self.model

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        payload = {"model": self.model or "local-model",
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   "max_tokens": max_tokens, "temperature": self.temperature, "stop": STOP_SEQUENCES,
                   "chat_template_kwargs": {"enable_thinking": False}}
        if self.send_reasoning_effort:
            payload["reasoning_effort"] = "none"
        try:
            r = self.http.post(f"{self.base}/chat/completions", json=payload)
        except self._httpx.TimeoutException as e:
            raise EndpointDown(f"no answer from {self.base} in {TIMEOUT:.0f} s") from e
        except self._httpx.TransportError as e:
            raise EndpointDown(f"{self.base} is unreachable ({e})") from e
        if r.status_code == 400 and "reasoning_effort" in r.text and self.send_reasoning_effort:
            self.send_reasoning_effort = False          # some endpoints reject it: once, then never again
            return self.complete(system, user, max_tokens)
        if r.status_code >= 400:
            raise Rejected(f"HTTP {r.status_code}: {r.text[:200]}")
        try:
            msg = (r.json().get("choices") or [{}])[0].get("message") or {}
        except ValueError as e:
            raise Rejected(f"the reply is not JSON: {r.text[:200]}") from e
        return THINKING_BLOCK.sub("", msg.get("content") or "").strip()


def ask(client, system: str, items: list[Item], reasons: dict[int, str] | None = None) -> tuple[dict[int, str], str]:
    """One request for items -> ({number: short}, why nothing came back)."""
    try:
        text = client.complete(system, user_message(items, reasons), TOKENS_PER_ITEM * len(items) + TOKENS_BASE)
    except Rejected as e:
        return {}, f"the endpoint rejected the request ({e})"
    reply = extract_json_array(text)
    line = " ".join(text.split()).strip("\"'` ")
    if reply is None and len(items) == 1 and line and "\n" not in text.strip() and not re.search(r"[\[\]{}]", line):
        return {1: line}, ""                              # one item answered with the bare short: take it
    if reply is None:
        return {}, f"the reply has no JSON array: {' '.join(text.split())[:100]!r}"
    return map_reply(reply, len(items)), "the reply has no short for this id"


# --- validation -----------------------------------------------------------------------------

def color_mode(long: str) -> str | None:
    """The color mode the long caption's opening states: bw, color, film; or the medium of a picture
    that is not a photo: painting, illustration, drawing; None when the opening says neither."""
    head = " ".join(long.split()).lower()
    if head.startswith("a black-and-white"):
        return "bw"
    opening = re.split(r"\s(?:from|of|in|showing|depicting)\s", head, maxsplit=1)[0]
    if "film still" in opening or "movie still" in opening:
        return "film"
    if re.search(r"photo|snapshot", opening):
        if "sepia" in opening:
            return "sepia"
        if re.search(r"black-and-white|black and white|monochrome", opening):
            return "bw"
        if re.search(r"colou?r", opening):
            return "color"
        return None
    for medium in MEDIUM_TOKENS:
        if medium in opening:
            return medium
    return None


def concepts_in(short: str) -> list[str]:
    return [name for name, rx, _ in CONCEPTS if rx.search(short)]


def verified_concepts(short: str, long: str) -> list[str]:
    return [name for name, rx, cond in CONCEPTS if rx.search(short) and cond(long)]


def word_count(short: str) -> int:
    return len(short.split())


def unquoted_cyrillic(text: str) -> str:
    """The first Cyrillic word outside quotes, or ''."""
    m = CYRILLIC.search(QUOTED.sub(" ", text))
    return m.group(0) if m else ""


def validate(short: str, style: str, long: str) -> list[str]:
    """The reasons a short fails its rules; empty when it passes."""
    why = []
    s = " ".join(short.split())
    low = s.lower()
    n = word_count(s)
    if style == "B":
        tags = [t.strip() for t in s.split(",") if t.strip()]
        lo, hi = TAG_LIMITS
        if not lo <= len(tags) <= hi:
            why.append(f"style B needs {lo} to {hi} comma-separated tags, not {len(tags)}")
        long_tags = [t for t in tags if word_count(t) > TAG_MAX_WORDS]
        if long_tags:
            why.append(f"tags have at most {TAG_MAX_WORDS} words: {long_tags[0]!r}")
    else:
        lo, hi = WORD_LIMITS[style]
        if not lo <= n <= hi:
            why.append(f"style {style} needs {lo} to {hi} words, not {n}")
    russian = unquoted_cyrillic(s)
    if russian:
        why.append(f"write in English only; Cyrillic only in a quoted sign (found {russian!r})")

    if not ERA_EN.search(s):
        why.append('no era (a decade such as "1960s", a year, or "Soviet era" when the long caption has no date)')

    mode = color_mode(long)
    tokens = MODE_TOKENS
    found = {m for m, toks in tokens.items() if any(t in low for t in toks)}
    if mode in MEDIUM_TOKENS:
        if not any(t in low for t in MEDIUM_TOKENS[mode]):
            why.append(f"no medium (the long caption is a {mode}, not a photo)")
        m = PHOTO_WORD.search(s)
        if m:
            why.append(f'a {mode} is not a photo (found {m.group(0)!r})')
    elif mode is not None:
        allowed = {"bw", "sepia"} if mode == "sepia" else {mode}       # a sepia photo may be called b&w
        words = ", ".join(t for m in allowed for t in tokens[m])
        if not found & allowed:
            why.append(f"no color mode, or not the long caption's (use one of: {words})")
        elif found - allowed:
            why.append(f"two color modes (keep only {words})")

    if " ".join(long.split()).lower().startswith("a black-and-white"):
        bare = low
        for toks in MODE_TOKENS.values():
            for t in toks:
                bare = bare.replace(t, " ")
        bare = COLOR_NAMES.sub(" ", bare)
        colors = COLOR_RE.findall(bare)
        if colors:
            why.append(f"a black-and-white photo takes no color words (found {colors[0]!r})")

    for name, rx, cond in CONCEPTS:
        if rx.search(s) and not cond(long):
            why.append(f'"{name}" only when the long caption meets its condition (use a plain noun)')

    if any(t in low for t in TEMPLATE_OPENINGS):
        why.append("the long caption's template opening (write the user form)")
    m = BANNED.search(s)
    if m:
        why.append(f'"{m.group(0)}" is not allowed')
    m = REQUEST.search(s) or IMPERATIVE_FIRST.search(s)
    if m:
        why.append(f'a request or instruction ("{m.group(0).strip()}"); only describe what the picture shows')
    return why


# --- the run --------------------------------------------------------------------------------

def write_short(item: Item) -> None:
    item.short_path.write_bytes(item.short.encode("utf-8"))


def process_chunk(client, system: str, chunk: list[Item], retries: int = MAX_RETRIES, write: bool = True) -> None:
    """First try for the chunk (batch or single), then single-mode retries for every item that failed."""
    mode = "batch" if len(chunk) > 1 else "single"
    t0 = time.time()
    shorts, missing = ask(client, system, chunk)
    share = (time.time() - t0) / len(chunk)
    for n, it in enumerate(chunk, 1):
        it.mode, it.attempts, it.seconds = mode, 1, share
        it.short = shorts.get(n, "")
        reasons = validate(it.short, it.style, it.long) if it.short else [missing]
        first = "; ".join(reasons)
        while reasons and it.attempts <= retries:
            reason = "; ".join(reasons)
            t1 = time.time()
            again, missing1 = ask(client, system, [it], {1: reason} if it.short else None)
            it.seconds += time.time() - t1
            it.attempts += 1
            it.short = again.get(1, "")
            reasons = validate(it.short, it.style, it.long) if it.short else [missing1]
        if reasons:
            it.status, it.reason = "failed", "; ".join(reasons)
            it.concepts = []
            continue
        it.status = "ok" if it.attempts == 1 else "retried"
        it.reason = "" if it.attempts == 1 else FIRST_TRY + first     # why the first answer was rejected
        it.concepts = verified_concepts(it.short, it.long)
        if write:
            write_short(it)


def log_rows(path: Path, items: list[Item], root: Path) -> None:
    new = not path.is_file() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8-sig" if new else "utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(LOG_FIELDS)
        for it in items:
            w.writerow([str(it.txt.relative_to(root)), it.style, it.status, it.attempts, it.reason,
                        word_count(it.short) if it.short else "", "|".join(it.concepts), it.mode,
                        f"{it.seconds:.2f}"])


def load_system_prompt() -> str:
    if not PROMPT_FILE.is_file():
        raise SystemExit(f"the system prompt {PROMPT_FILE} does not exist")
    text = read_text(PROMPT_FILE).strip()
    if "\x00" in text:                     # UTF-16 without a BOM, or another encoding read as cp1251
        raise SystemExit(f"cannot read the encoding of {PROMPT_FILE}; save it as UTF-8")
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write a short user-style prompt (<stem>.short) for every long "
                                             "caption of a dataset with a local LLM.")
    ap.add_argument("dataset", type=Path, help="the dataset folder (searched recursively)")
    ap.add_argument("--ext", default=".txt", help="the long caption extension (default .txt)")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help=f"OpenAI-compatible API (default {DEFAULT_ENDPOINT})")
    ap.add_argument("--model", default="auto", help="model name; auto takes the first one of /models")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH, help=f"items per request, 1 = single mode "
                                                                      f"(default {DEFAULT_BATCH})")
    ap.add_argument("--styles", default=DEFAULT_STYLES, help=f"style shares (default {DEFAULT_STYLES}); "
                                                              "E is Russian, E=0 disables it")
    ap.add_argument("--limit", type=int, default=0, help="process at most N items that have no .short")
    ap.add_argument("--overwrite", action="store_true", help="write a new .short where one exists")
    ap.add_argument("--dry-run", action="store_true", help="list the work and the first request; no calls, no files")
    ap.add_argument("--log", type=Path, default=Path("recaption_log.csv"), help="CSV log, appended (default "
                                                                                "recaption_log.csv)")
    ap.add_argument("--temperature", type=float, default=0.3)
    a = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    root = a.dataset.resolve()
    if not root.is_dir():
        raise SystemExit(f"{root} is not a folder")
    ext = a.ext if a.ext.startswith(".") else "." + a.ext
    if ext.lower() == SHORT_EXT:
        raise SystemExit(f"--ext cannot be {SHORT_EXT}, the output extension")
    if a.batch < 1:
        raise SystemExit("--batch must be 1 or more")
    try:
        styles = parse_styles(a.styles)
    except ValueError as e:
        raise SystemExit(str(e))
    system = load_system_prompt()

    items = find_items(root, ext, styles)
    skipped = [] if a.overwrite else [it for it in items if has_short(it)]
    pending = [it for it in items if a.overwrite or not has_short(it)]
    if a.limit > 0:
        pending = pending[:a.limit]
    counts = {k: sum(1 for it in pending if it.style == k) for k, _ in styles}
    print(f"{len(items)} items in {root}, {len(skipped)} with a .short already, {len(pending)} to do "
          f"({', '.join(f'{k} {v}' for k, v in counts.items())}), batch {a.batch}")
    for it in pending:
        it.long = read_text(it.txt).strip()

    if a.dry_run:
        if pending:
            print("\nThe first request's user message:\n")
            print(user_message(pending[:a.batch]))
        return 0

    for it in skipped:
        it.status, it.reason, it.attempts = "skipped", f"{SHORT_EXT} exists", 0
    log_rows(a.log, skipped, root)
    if not pending:
        return 0

    client = ChatClient(a.endpoint, a.model, a.temperature)
    try:
        model = client.resolve_model()
    except EndpointDown as e:
        raise SystemExit(str(e))
    print(f"model {model}, log {a.log.resolve()}")

    t_start = time.time()
    finished: list[Item] = []
    try:
        for start in range(0, len(pending), a.batch):
            chunk = pending[start:start + a.batch]
            t0 = time.time()
            process_chunk(client, system, chunk)
            log_rows(a.log, chunk, root)
            finished += chunk
            st = {s: sum(1 for it in chunk if it.status == s) for s in ("ok", "retried", "failed")}
            print(f"[{len(finished):>{len(str(len(pending)))}}/{len(pending)}] {st['ok']} ok, {st['retried']} retried, "
                  f"{st['failed']} failed, {(time.time() - t0) / len(chunk):.1f} s/item", flush=True)
    except EndpointDown as e:
        print(f"\nstopped: {e}\nrun again to continue", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nstopped; run again to continue", file=sys.stderr)

    elapsed = time.time() - t_start
    st = {s: sum(1 for it in finished if it.status == s) for s in ("ok", "retried", "failed")}
    print(f"\n{len(finished)} items in {elapsed:.0f} s ({elapsed / max(1, len(finished)):.1f} s/item): "
          f"{st['ok']} ok on the first try, {st['retried']} after a retry, {st['failed']} failed, "
          f"{len(skipped)} skipped")
    for k, _ in styles:
        group = [it for it in finished if it.style == k]
        if group:
            good = sum(1 for it in group if it.status != "failed")
            print(f"  style {k}: {good}/{len(group)} written")
    for status, title in (("failed", "failed"), ("retried", "rejected on the first try, then passed")):
        counts = reason_counts(it.reason for it in finished if it.status == status)
        if counts:
            print(f"  {title}:")
            for r, c in counts[:10]:
                print(f"  {c:6} x {r}")
    return 0 if len(finished) == len(pending) else 1


def reason_counts(texts) -> list[tuple[str, int]]:
    """[(reason, count)], most frequent first, for log reasons: the details after "(" or ":" and the
    actual count (", not 21") dropped."""
    counts: dict[str, int] = {}
    for text in texts:
        for r in text.removeprefix(FIRST_TRY).split("; "):
            key = re.sub(r", not \d+$", "", r.split(" (")[0].split(":")[0])
            counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])


if __name__ == "__main__":
    sys.exit(main())
