"""Caption cleaning for the .txt sidecars.

The .txt next to a photo gets the caption without URLs, hashtags and
mentions, with whitespace collapsed to single spaces; the raw text stays in
the manifest `text` column and in metadata.json. A caption with nothing left
(no letter or digit) writes no .txt.
"""
from __future__ import annotations

import re

# [id123|Name], [club123|Name], [examplegroup|Name], [https://...|Name] -> Name
_RE_LINK_MENTION = re.compile(r"\[[^\[\]|\n]+\|([^\[\]\n]*)\]")
# @id123 (Name), @examplegroup (Name) -> Name
_RE_AT_NAMED = re.compile(r"(?<![\w@])@[\w.]+\s*\(([^()\n]*)\)")
# a bare @mention (not the middle of an e-mail address)
_RE_AT = re.compile(r"(?<![\w@])@[\w.]*\w")
_RE_URL = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|(?<![\w.])(?:t\.me|(?:m\.)?vk\.(?:com|ru|cc|me))/\S*",
    re.IGNORECASE)
_RE_HASHTAG = re.compile(r"(?<!\w)#[\w@]+")
_RE_SPACE = re.compile(r"\s+")


def clean_caption(text: str) -> str:
    """The caption for the .txt sidecar; "" when nothing is left."""
    s = str(text or "")
    if not s.strip():
        return ""
    s = _RE_LINK_MENTION.sub(r"\1", s)
    s = _RE_AT_NAMED.sub(r"\1", s)
    s = _RE_URL.sub(" ", s)
    s = _RE_HASHTAG.sub(" ", s)
    s = _RE_AT.sub(" ", s)
    s = _RE_SPACE.sub(" ", s).strip()
    # punctuation left behind by a removed link: "Фото: ." -> "Фото:"
    s = re.sub(r"\s+([,.;:!?)])", r"\1", s)
    s = re.sub(r"\(\s*\)", "", s).strip()
    return s if re.search(r"\w", s) else ""
