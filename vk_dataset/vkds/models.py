"""Модели данных: сообщество и альбом."""
from __future__ import annotations

from dataclasses import dataclass

from .fs import sanitize_dirname


@dataclass
class GroupInfo:
    """Сообщество ВК (id положительный; owner_id для API — отрицательный)."""

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
        """Папка группы по короткому имени (examplegroup) или club<id>.

        Историческое имя (vkds 0.2); используется для миграции старых
        датасетов и служебных папок (_peek).
        """
        sn = (self.screen_name or "").strip()
        if sn and sn not in ("club{}".format(self.id), "public{}".format(self.id)):
            return sanitize_dirname(sn) or "club{}".format(self.id)
        return "club{}".format(self.id)

    @property
    def name_key(self) -> str:
        """Папка группы по её НАЗВАНИЮ («Автомобили и Техника СССР…»).

        Основной способ папок с версии 0.3: читаемый датасет. При пустом
        или бесполезном названии — откат на короткое имя (dir_key).
        """
        return sanitize_dirname(self.name, max_len=80) or self.dir_key

    def as_dict(self) -> dict:
        return {"id": self.id, "owner_id": self.owner_id, "name": self.name,
                "screen_name": self.screen_name, "description": self.description,
                "members_count": self.members_count}


@dataclass
class AlbumInfo:
    """Альбом сообщества. id — строка: числовой id либо wall/saved/profile."""

    id: str
    title: str
    description: str = ""
    size: int | None = None   # число фото по данным API (у wall — None)
    created: int | None = None
    updated: int | None = None
    special: bool = False     # системный альбом (wall, saved, profile)

    @property
    def is_special(self) -> bool:
        return not str(self.id).isdigit()

    def as_dict(self) -> dict:
        return {"id": self.id, "title": self.title,
                "description": self.description, "size": self.size,
                "created": self.created, "updated": self.updated,
                "special": self.special}
