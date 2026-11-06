"""Interest: which photos of a category show their subject, read from the boxes
rows that detect wrote (cherrypick --interest).

A rule reads the derived columns of one photo and says pass or fail. The rule
of a category comes from its name after the number (transport -> vehicle,
people_adult -> people, ...; DEFAULT_RULES), and `rule.<name> = <rule>` in the
conf's [interest] overrides one category. A photo gets a tier: 0 when its rule
passes, 1 when it is unknown (no boxes row yet, a row of another vocabulary, or
a category whose rule is `pass`), 2 when its rule fails. cherrypick ranks each
bucket by (tier, quality) and cuts it as it does without interest, so a fail
orders a photo behind the passes and never removes it: a bucket short of
interesting photos fills up with its best others, and a category of rule
`pass` picks exactly as without --interest.

The thresholds live in conf.InterestConf; they come from 110 probed photos and
are meant to be tuned on the full base with cherrypick --interest --dry-run.
"""

from __future__ import annotations

import extract as xt
from conf import InterestConf

RULES = ("vehicle", "people", "group", "thing", "presence", "pass")
DEFAULT_RULES = {
    "transport": "vehicle",
    "people_adult": "people", "people_children": "people", "school": "people",
    "group_photo_children": "group", "group_photo_adult": "group",
    "futurism": "thing", "objects": "thing", "military": "thing",
    "environment_indoors": "presence", "environment_outdoors": "presence",
}
PASS, UNKNOWN, FAIL = 0, 1, 2
TIER_NAMES = ("interesting", "unknown", "not")


def rule_of(category: str, ic: InterestConf) -> str:
    """The rule of a category folder name (050-transport, transport): the conf's override, else
    the default of the name after the number, else pass."""
    name = xt.split_name(category)[1]                   # normalised: underscores and hyphens become spaces
    for rules in (ic.rules, DEFAULT_RULES):
        for key, rule in rules.items():
            if xt.norm(key) == name:
                return rule
    return "pass"


def passes(rule: str, st: dict, ic: InterestConf) -> bool | None:
    """The rule on one photo's derived columns; None for the rule pass (no gate)."""
    if rule == "vehicle":
        return ic.vehicle_min <= st["max_vehicle_area"] <= ic.vehicle_max and st["max_vehicle_w"] >= ic.vehicle_width
    if rule == "people":
        return st["max_person_h"] >= ic.person_height and st["max_face_h"] >= ic.face_height
    if rule == "group":
        return st["n_person"] >= ic.group_min and st["max_face_h"] >= ic.group_face_height
    if rule == "thing":
        return st["max_thing_area"] >= ic.thing_min
    if rule == "presence":
        return st["n_person"] >= ic.presence_persons and st["n_face"] >= ic.presence_faces
    if rule == "pass":
        return None
    raise ValueError(f"unknown interest rule {rule!r}")


def tier(rule: str, st: dict | None, ic: InterestConf, vocab: str | None = None) -> int:
    """0 pass, 1 unknown (no row, a row of another vocabulary, or no gate), 2 fail."""
    if rule == "pass" or st is None or (vocab is not None and st.get("vocab") != vocab):
        return UNKNOWN
    return PASS if passes(rule, st, ic) else FAIL


def tiers(store, photos: dict, ic: InterestConf, vocab: str | None = None) -> dict[int, int]:
    """cid -> tier for the photos of cherrypick (cid -> {"cat", ...}), from one pass over the
    boxes table; vocab: the current vocabulary hash (default detect's), rows of another are unknown."""
    if vocab is None:
        import detect
        vocab = detect.vocab_hash()
    stats = store.boxes_stats(photos)
    rules = {}
    out = {}
    for cid, ph in photos.items():
        cat = ph["cat"]
        if cat not in rules:
            rules[cat] = rule_of(cat, ic)
        out[cid] = tier(rules[cat], stats.get(cid), ic, vocab)
    return out


def rules_of(categories, ic: InterestConf) -> dict[str, str]:
    """category -> rule, for the report and the manifest."""
    return {cat: rule_of(cat, ic) for cat in sorted(categories)}


def describe(categories, ic: InterestConf) -> str:
    """The rules in use, in words, for the command's report."""
    by_rule: dict[str, list[str]] = {}
    for cat, rule in rules_of(categories, ic).items():
        by_rule.setdefault(rule, []).append(cat)
    parts = [f"{rule}: {', '.join(cats)}" for rule, cats in by_rule.items()]
    return ("Interest: photos whose rule passes are picked first, then the unknown, then the rest; "
            "the ceilings decide how many. Rules: " + "; ".join(parts) + ".")
