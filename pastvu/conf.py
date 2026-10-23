"""pastvu.conf: the base folder of the downloads, the classify samples folder,
the defaults of a run, and the region aliases, in three groups: countries
(with groups of countries), regions, cities and smaller parts.

    [paths]
    base = T:\\pastvu
    samples = T:\\claude\\temp\\learning_dataset

    [defaults]
    years = 1917 1991
    split = 5000
    fetchers = 4
    json_files = no
    priority_margin = 0.15
    quality_threshold = 0.88
    max_aspect = 21:9

    [aliases.countries]
    belarus = 22
    baltics = 149, 154, 267

The file sits next to pastvu.py (install.bat makes it from
pastvu.conf.example); --conf names another. Relative paths are resolved
against the folder of the file.
"""

from __future__ import annotations

import configparser
import re
from dataclasses import dataclass, field
from pathlib import Path

from layout import LAYOUTS

HERE = Path(__file__).resolve().parent
DEFAULT_PATH = HERE / "pastvu.conf"
ALIAS_GROUPS = ("countries", "regions", "cities")
ALIAS_NAME = re.compile(r"^[a-z0-9][a-z0-9_]*$")


@dataclass
class CaptionConf:
    """[caption]: the VLM endpoint and the prompts of autocaption. The prompts
    are always files (paths relative to the conf's folder), never text in the
    conf."""
    api_url: str = "http://localhost:5001/v1"
    api_key: str = ""
    model: str = ""                   # empty: whatever the endpoint has loaded
    system_prompt: Path = HERE / "tagging" / "sysprompt.txt"
    prompt: Path = HERE / "tagging" / "main_prompt.txt"
    max_tokens: int = 400
    temperature: float = 0.3
    top_p: float = 0.95
    image_side: int = 1024            # the longest side of the image sent to the VLM
    timeout: int = 300                # seconds per request
    reasoning: bool = False           # let a reasoning model think before it answers


@dataclass
class Conf:
    path: Path | None = None
    base: Path | None = None
    samples: Path | None = None
    years: tuple[int, int] = (1917, 1991)
    split: int = 5000
    fetchers: int = 4
    json_files: bool = False
    priority_margin: float = 0.15
    quality_threshold: float = 0.88
    quality_in_file: bool = False      # the key is in the file (else it is written there on first use)
    max_aspect: float = 21 / 9         # long side over short side; 0: no limit
    aspect_in_file: bool = False
    layout: str | None = None          # the folder tree of a new base; None: not in the file (written on first use)
    bw_color: bool | None = None       # bw/ and color/ at the top of a new base; None: not in the file
    caption: CaptionConf = field(default_factory=CaptionConf)
    aliases: dict[str, list[int]] = field(default_factory=dict)
    groups: dict[str, str] = field(default_factory=dict)      # alias -> its group

    def need_base(self) -> Path:
        if not self.base:
            where = self.path or DEFAULT_PATH
            raise SystemExit(f"No base folder: set [paths] base in {where}, or give --base.")
        return self.base


def _path(value: str, root: Path) -> Path | None:
    value = (value or "").strip().strip('"')
    if not value:
        return None
    p = Path(value)
    return (p if p.is_absolute() else root / p).resolve()


def parse_cids(value: str, where: str) -> list[int]:
    out = []
    for tok in re.split(r"[,\s]+", value.strip()):
        if not tok:
            continue
        if not tok.isdigit():
            raise SystemExit(f"{where}: {tok!r} is not a PastVu region number")
        out.append(int(tok))
    if not out:
        raise SystemExit(f"{where}: no region numbers")
    return out


LAYOUT_COMMENT = [
    "; The folder tree of a base folder: years/place puts a scope into",
    r"; <base>\1970s\russia\, place/years into <base>\russia\1970s\, flat into",
    r"; <base>\russia_1970s\. A base keeps the tree it was made with.",
]


BW_COLOR_COMMENT = [
    "; Black-and-white and colour photos in two trees: <base>\\bw\\... and",
    "; <base>\\color\\..., each with the layout above. A base keeps the choice it was made with.",
]


QUALITY_COMMENT = ["; Photos the quality probe (trained on _quality_good and _quality_bad in the",
                   "; samples folder) scores at or above this P(bad) are skipped and never saved,",
                   "; and reclassify deletes the saved ones. 0 turns it off."]


ASPECT_COMMENT = ["; Photos whose long side is more than this times the short side (panoramas,",
                  "; strips) are skipped like the ones under 512 pixels, and reclassify deletes",
                  "; the saved ones. W:H or a number; 0 turns it off."]


def parse_aspect(text: str) -> float:
    """21:9 or 2.333 -> the long side over the short side; 0: no limit."""
    t = text.strip()
    try:
        if ":" in t:
            a, b = (float(x) for x in t.split(":", 1))
            v = max(a, b) / min(a, b) if a > 0 and b > 0 else -1.0
        else:
            v = float(t)
    except ValueError:
        v = -1.0
    if v != 0 and v < 1:
        raise ValueError(f"max_aspect {t!r}: give W:H such as 21:9, a number of at least 1, or 0 (no limit)")
    return v


def save_layout(path: Path, value: str) -> None:
    save_default(path, "layout", value, LAYOUT_COMMENT)


def save_default(path: Path, key: str, value: str, comment: list[str]) -> None:
    """Write key = value into [defaults] of the conf file, keeping the rest of
    the file as it is."""
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read()
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    entry = comment + [f"{key} = {value}"]
    head = next((i for i, l in enumerate(lines) if l.strip().lower() == "[defaults]"), None)
    if head is None:
        lines += ["", "[defaults]"] + entry
    else:
        end = next((i for i in range(head + 1, len(lines)) if lines[i].strip().startswith("[")), len(lines))
        old = next((i for i in range(head + 1, end) if lines[i].split("=")[0].strip().lower() == key), None)
        if old is not None:
            lines[old] = f"{key} = {value}"
        else:
            while end > head + 1 and not lines[end - 1].strip():
                end -= 1
            lines[end:end] = entry
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(nl.join(lines) + nl)


def load(path: Path | None = None) -> Conf:
    path = Path(path) if path else DEFAULT_PATH
    conf = Conf()
    if not path.is_file():
        return conf
    cp = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=(";", "#"))
    cp.optionxform = str.lower
    try:
        cp.read(path, encoding="utf-8")
    except configparser.Error as e:
        raise SystemExit(f"{path}: {e}")
    conf.path = path
    root = path.parent
    if cp.has_section("paths"):
        conf.base = _path(cp.get("paths", "base", fallback=""), root)
        conf.samples = _path(cp.get("paths", "samples", fallback=""), root)
    if cp.has_section("defaults"):
        d = cp["defaults"]
        if d.get("years", "").strip():
            ys = d["years"].replace("-", " ").split()
            if len(ys) != 2 or not all(y.isdigit() for y in ys):
                raise SystemExit(f"{path}: [defaults] years must be two years, e.g. 1917 1991")
            conf.years = (int(ys[0]), int(ys[1]))
        lay = d.get("layout", "").strip().lower()
        if lay:
            if lay not in LAYOUTS:
                raise SystemExit(f"{path}: [defaults] layout must be one of {', '.join(LAYOUTS)}")
            conf.layout = lay
        try:
            if d.get("bw_color", "").strip():
                conf.bw_color = d.getboolean("bw_color")
            conf.split = d.getint("split", fallback=conf.split)
            conf.fetchers = d.getint("fetchers", fallback=conf.fetchers)
            conf.json_files = d.getboolean("json_files", fallback=conf.json_files)
            conf.priority_margin = d.getfloat("priority_margin", fallback=conf.priority_margin)
            if d.get("quality_threshold", "").strip():
                conf.quality_threshold = d.getfloat("quality_threshold")
                conf.quality_in_file = True
            if d.get("max_aspect", "").strip():
                conf.max_aspect = parse_aspect(d.get("max_aspect"))
                conf.aspect_in_file = True
        except ValueError as e:
            raise SystemExit(f"{path}: [defaults] {e}")
    if cp.has_section("caption"):
        c, cc = cp["caption"], conf.caption
        try:
            cc.api_url = c.get("api_url", cc.api_url).strip() or cc.api_url
            cc.api_key = c.get("api_key", cc.api_key).strip()
            cc.model = c.get("model", cc.model).strip()
            cc.system_prompt = _path(c.get("system_prompt", ""), root) or cc.system_prompt
            cc.prompt = _path(c.get("prompt", ""), root) or cc.prompt
            cc.max_tokens = c.getint("max_tokens", fallback=cc.max_tokens)
            cc.temperature = c.getfloat("temperature", fallback=cc.temperature)
            cc.top_p = c.getfloat("top_p", fallback=cc.top_p)
            cc.image_side = c.getint("image_side", fallback=cc.image_side)
            cc.timeout = c.getint("timeout", fallback=cc.timeout)
            cc.reasoning = c.getboolean("reasoning", fallback=cc.reasoning)
        except ValueError as e:
            raise SystemExit(f"{path}: [caption] {e}")
    for group in ALIAS_GROUPS:
        sec = f"aliases.{group}"
        if not cp.has_section(sec):
            continue
        for name, value in cp[sec].items():
            if not ALIAS_NAME.match(name):
                raise SystemExit(f"{path}: alias {name!r} in [{sec}]: use lower-case letters, digits and _")
            if name in conf.aliases:
                raise SystemExit(f"{path}: alias {name!r} is defined twice ([{conf.groups[name]}] and [{group}])")
            conf.aliases[name] = parse_cids(value, f"{path}: alias {name}")
            conf.groups[name] = group
    return conf
