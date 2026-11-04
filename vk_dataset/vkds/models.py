"""Data models: a group and an album."""
from __future__ import annotations

from dataclasses import dataclass

from .fs import sanitize_dirname


@dataclass
class GroupInfo:
    """A VK group (id positive; the API owner_id is negative)."""

    id: int
    name: str
    screen_name: str = ""
    description: str = ""
    members_count: int | None = None

    @property
    def owner_id(self) -> int:
        return -int(self.id)

    @property
    def dir_key(self) -> str:
        """The group folder by short name (examplegroup) or club<id>.

        The folder name of vkds 0.2; used to migrate old datasets and for
        service folders (_peek).
        """
        sn = (self.screen_name or "").strip()
        if sn and sn not in ("club{}".format(self.id), "public{}".format(self.id)):
            return sanitize_dirname(sn) or "club{}".format(self.id)
        return "club{}".format(self.id)

    @property
    def name_key(self) -> str:
        """The group folder by the group's title, the default since 0.3 (a
        readable dataset). An empty or useless title falls back to the short
        name (dir_key)."""
        return sanitize_dirname(self.name, max_len=80) or self.dir_key

    def as_dict(self) -> dict:
        return {"id": self.id, "owner_id": self.owner_id, "name": self.name,
                "screen_name": self.screen_name, "description": self.description,
                "members_count": self.members_count}


@dataclass
class AlbumInfo:
    """An album of a group. id is a string: a number, or wall, saved or profile."""

    id: str
    title: str
    description: str = ""
    size: int | None = None   # photos according to the API (None for wall)
    created: int | None = None
    updated: int | None = None
    special: bool = False     # a system album (wall, saved, profile)

    @property
    def is_special(self) -> bool:
        return not str(self.id).isdigit()

    def as_dict(self) -> dict:
        return {"id": self.id, "title": self.title,
                "description": self.description, "size": self.size,
                "created": self.created, "updated": self.updated,
                "special": self.special}
