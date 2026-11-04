"""vkds.conf: the paths and limits of the tool.

The conf sits next to vkds.py (the tool folder), never in the current
directory, so run.bat finds it from anywhere. install.bat copies
vkds.conf.example to vkds.conf when it is missing. A missing conf means the
defaults, so the tests and a bare checkout still work. Relative paths in the
conf resolve against the folder of the conf.

What the user sees (errors, the token notice) stays Russian, like every
printed line of the tool.
"""
from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PATH = TOOL_DIR / "vkds.conf"


class ConfError(Exception):
    """A conf or path problem the CLI reports as a readable refusal."""


@dataclass
class Conf:
    path: Path = DEFAULT_PATH          # where the conf is or would be
    loaded: bool = False               # True when the file existed
    token_file: Path = TOOL_DIR / "vk_token.txt"
    base: Path | None = None           # the datasets root; None: not set
    ledger: Path = TOOL_DIR / "_vkds" / "ledger.json"   # used from phase 2
    monthly: int = 10000
    delay: float = 0.35


def _path(value: str, root: Path) -> Path | None:
    value = (value or "").strip().strip('"')
    if not value:
        return None
    p = Path(value)
    return p if p.is_absolute() else (root / p).resolve()


def load(path: Path | str | None = None) -> Conf:
    """Read the conf at `path` (default: vkds.conf in the tool folder)."""
    path = Path(path) if path else DEFAULT_PATH
    root = path.resolve().parent
    conf = Conf(path=path, token_file=root / "vk_token.txt",
                ledger=root / "_vkds" / "ledger.json")
    if not path.is_file():
        return conf
    cp = configparser.ConfigParser(interpolation=None,
                                   inline_comment_prefixes=(";", "#"))
    cp.optionxform = str.lower
    try:
        cp.read(path, encoding="utf-8-sig")
    except configparser.Error as e:
        raise ConfError("{}: ошибка в файле настроек: {}".format(path, e)) from e
    conf.loaded = True
    if cp.has_section("paths"):
        s = cp["paths"]
        conf.token_file = _path(s.get("token", ""), root) or conf.token_file
        conf.base = _path(s.get("base", ""), root)
        conf.ledger = _path(s.get("ledger", ""), root) or conf.ledger
    if cp.has_section("limits"):
        s = cp["limits"]
        try:
            if s.get("monthly", "").strip():
                conf.monthly = s.getint("monthly")
            if s.get("delay", "").strip():
                conf.delay = s.getfloat("delay")
        except ValueError as e:
            raise ConfError("{}: [limits] monthly — целое число, delay — число "
                            "секунд ({})".format(path, e)) from e
    return conf


def read_token(explicit: str | None, conf: Conf) -> tuple[str, str]:
    """The token and where it came from: --token > VK_TOKEN > [paths] token.

    Returns ("", "") when there is none.
    """
    if explicit and explicit.strip():
        return explicit.strip(), "--token"
    env = os.environ.get("VK_TOKEN", "").strip()
    if env:
        return env, "VK_TOKEN"
    p = conf.token_file
    if p.is_file():
        tok = p.read_text(encoding="utf-8-sig").strip()
        if tok:
            return tok, str(p)
    return "", ""


def resolve_out(out: str | None, conf: Conf) -> Path:
    """The dataset root of a run.

    An absolute --out is used as it is, a relative one resolves under
    [paths] base (or against the current directory when base is empty), no
    --out means base itself; no --out and no base is an error.
    """
    if out:
        p = Path(out)
        if p.is_absolute():
            return p
        return conf.base / p if conf.base else p.resolve()
    if conf.base:
        return conf.base
    raise ConfError(
        "Не задана папка датасетов: укажите base в разделе [paths] файла {} "
        "или передайте --out.".format(conf.path))
