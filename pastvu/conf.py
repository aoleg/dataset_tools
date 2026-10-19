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

HERE = Path(__file__).resolve().parent
DEFAULT_PATH = HERE / "pastvu.conf"
ALIAS_GROUPS = ("countries", "regions", "cities")
ALIAS_NAME = re.compile(r"^[a-z0-9][a-z0-9_]*$")


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
        try:
            conf.split = d.getint("split", fallback=conf.split)
            conf.fetchers = d.getint("fetchers", fallback=conf.fetchers)
            conf.json_files = d.getboolean("json_files", fallback=conf.json_files)
            conf.priority_margin = d.getfloat("priority_margin", fallback=conf.priority_margin)
        except ValueError as e:
            raise SystemExit(f"{path}: [defaults] {e}")
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
