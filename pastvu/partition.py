"""The partition rule behind `run --by` and `regroup --by`: which scope a
photo belongs to, by its decade, its country, its region, or a pair of them.

- decade: from the photo's start year (the gallery's `year`), so every photo
  belongs to exactly one decade. A run keeps the decades its years cover
  (1917-1991: the 1910s to the 1990s); a photo PastVu lists for the years
  that starts in another decade (dated 1890-1920) joins no scope.
- country: the alias of the conf's countries section that matches the
  photo's region chain at the deepest level, the first in the file on a tie.
  Aliases that name two or more countries (ussr, baltics, ...) are never
  used. So Crimea (556, filed under Russia in PastVu's tree) goes to the
  alias that names 556 itself: ukraine.
- region: the first level under the country in PastVu's tree (an oblast, a
  county, a federal city, Crimea), named by the alias that names exactly that
  region (regions section first, then cities), else by its English title. A
  photo placed only at country level goes to its country.

A photo's region chain comes from its record (the full list) or, before the
record exists, from the index row: the most specific known region and its
parents in the stored tree.
"""

from __future__ import annotations

from dataclasses import dataclass

import layout

KEYS = ("decade", "country", "region")


@dataclass(frozen=True)
class Rule:
    decade: bool = False
    place: str | None = None          # "country", "region" or None

    @property
    def text(self) -> str:
        return ",".join(([self.place] if self.place else []) + (["decade"] if self.decade else []))


def parse_rule(text: str) -> Rule:
    parts = [p.strip().lower().replace("by_", "") for p in text.replace("+", ",").split(",") if p.strip()]
    bad = [p for p in parts if p not in KEYS]
    if bad or not parts:
        raise SystemExit(f"--by {text}: use decade, country, region, or country,decade / region,decade")
    places = [p for p in parts if p in ("country", "region")]
    if len(places) > 1:
        raise SystemExit("--by: country and region exclude each other")
    return Rule(decade="decade" in parts, place=places[0] if places else None)


@dataclass(frozen=True)
class Target:
    """Where a photo goes. years/decade None: the caller decides (a run's years,
    or the span of the regrouped scopes)."""
    label: str
    regions: tuple
    decade: int | None


class Partitioner:
    def __init__(self, store, aliases: dict[str, list[int]], groups: dict[str, str], rule: Rule, label: str = "all",
                 regions: list[int] | None = None):
        self.store, self.rule = store, rule
        self.label, self.regions = label, tuple(sorted(regions or []))
        self._tree: dict[int, list[int]] = {}
        self.countries = [(n, c) for n, c in aliases.items() if groups.get(n) == "countries" and not self._group(c)]
        self.single = {}
        for section in ("regions", "cities", "countries"):
            for n, c in aliases.items():
                if groups.get(n) == section and len(c) == 1:
                    self.single.setdefault(c[0], n)

    def _parents(self, cid: int) -> list[int] | None:
        if cid not in self._tree:
            r = self.store.region(cid)
            self._tree[cid] = r["parents"] if r else None
        return self._tree[cid]

    def _group(self, cids: list[int]) -> bool:
        return sum(1 for c in cids if self._parents(c) == []) >= 2

    def chain(self, photo: dict) -> list[int]:
        """The region chain, top first. photo: an index row (rs) or a record (regions)."""
        if photo.get("regions"):
            return [int(r["cid"]) for r in photo["regions"] if r.get("cid")]
        for cid in photo.get("rs") or []:
            parents = self._parents(int(cid)) if cid else None
            if parents is not None:
                return list(parents) + [int(cid)]
        return []

    def country(self, chain: list[int]) -> tuple[str, tuple] | None:
        best, best_depth = None, -1
        pos = {c: i for i, c in enumerate(chain)}
        for name, cids in self.countries:
            depth = max((pos[c] for c in cids if c in pos), default=-1)
            if depth > best_depth:
                best, best_depth = (name, tuple(sorted(cids))), depth
        return best

    def region(self, chain: list[int]) -> tuple[str, tuple] | None:
        if len(chain) < 2:
            return self.country(chain)
        first = chain[1]
        name = self.single.get(first)
        if not name:
            r = self.store.region(first)
            name = layout.slug(r["title_en"]) if r else f"r{first}"
        return name, (first,)

    def target(self, photo: dict) -> Target:
        """photo: needs `year` and either `regions` (a record) or `rs` (an index row)."""
        decade = None
        if self.rule.decade:
            y = photo.get("year")
            decade = (int(y) // 10) * 10 if y is not None else None
        if self.rule.place:
            chain = self.chain(photo)
            place = self.country(chain) if self.rule.place == "country" else self.region(chain)
            if place is None:
                place = ("unplaced", ())
            return Target(place[0], place[1], decade)
        return Target(self.label, self.regions, decade)

