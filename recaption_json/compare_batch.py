#!/usr/bin/env python3
"""
compare_batch: is a batch of N captions per LLM call as good as one per call?

    python compare_batch.py D:\\photos
    python compare_batch.py D:\\photos --batch 5
    python compare_batch.py --report-only

Takes the first --items items of the dataset (the order recaption.py walks
them, with the same style letters), asks for every short once in single mode
and once in batch mode (first try only, no retries, no .short written), and
writes compare_batch.md next to this script: seconds per item, first-try pass
rate overall and per style, mean word count per style, the share of items
with a verified concept name, cross-item leakage in batch mode, and 20 items
side by side. The raw answers go to compare_batch.json; --report-only rebuilds
the report from it.

Leakage: a short that names a concept, vehicle model, place or year that its
own long caption does not support, but another long caption of the same batch
does.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import recaption as rc

TOOL_DIR = Path(__file__).resolve().parent
OUT_MD = TOOL_DIR / "compare_batch.md"
OUT_JSON = TOOL_DIR / "compare_batch.json"
DEFAULT_DATASET = Path(".")
SIDE_BY_SIDE = 20
PASS_MARGIN = 3.0                       # points: batch within this of single, and no leakage -> batch

VEHICLE_EN = re.compile(r"\b(gaz|zil|zis|ziu|liaz|laz|paz|maz|kraz|kamaz|uaz|vaz|zaz|raf|izh|moskvich|tu|il|an|yak"
                        r"|mi|ikarus)[- ]?(m-?)?(\d{1,4})\b", re.I)
VEHICLE_RU = re.compile(r"(?<![а-яё])(газ|зил|зис|зиу|лиаз|лаз|паз|маз|краз|камаз|уаз|ваз|заз|раф|иж|москвич|ту|ил|ан"
                        r"|як|ми|икарус)[- ]?(м-?)?(\d{1,4})(?!\d)", re.I)
RU_BRANDS = {"газ": "gaz", "зил": "zil", "зис": "zis", "зиу": "ziu", "лиаз": "liaz", "лаз": "laz", "паз": "paz",
             "маз": "maz", "краз": "kraz", "камаз": "kamaz", "уаз": "uaz", "ваз": "vaz", "заз": "zaz", "раф": "raf",
             "иж": "izh", "москвич": "moskvich", "ту": "tu", "ил": "il", "ан": "an", "як": "yak", "ми": "mi",
             "икарус": "ikarus"}
MODEL_NAMES = {"volga": r"\bvolga\b|(?<![а-яё])волг[аиу]", "pobeda": r"\bpobeda\b", "zhiguli": r"\bzhiguli\b|жигул",
               "lada": r"\blada\b|(?<![а-яё])лада", "chaika": r"\bchaika\b|(?<![а-яё])чайк[аи]",
               "zaporozhets": r"\bzaporozhets\b|запорож[её]ц", "niva": r"\bniva\b"}
YEAR = re.compile(r"(?<!\d)(19\d)(\d)(s|-[ех])?(?!\d)")
SHORT_DECADE = re.compile(r"(?<![\w\d])'?([1-9])0s\b")
# Places: these, plus every capitalised name after "in" in the compared long captions.
CITIES = {"Moscow": "москв", "Leningrad": "ленинград", "Kiev": "киев", "Kyiv": "киев", "Minsk": "минск",
          "Riga": "рига", "Tallinn": "таллин", "Vilnius": "вильнюс", "Tbilisi": "тбилиси", "Yerevan": "ереван",
          "Baku": "баку", "Tashkent": "ташкент", "Odessa": "одесс", "Kharkov": "харьков", "Kherson": "херсон",
          "Novocherkassk": "новочеркасск", "Volgograd": "волгоград", "Stalingrad": "сталинград",
          "Sevastopol": "севастопол", "Yalta": "ялт", "Sochi": "сочи", "Lvov": "львов", "Murmansk": "мурманск",
          "Vladivostok": "владивосток", "Novosibirsk": "новосибирск", "Sverdlovsk": "свердловск",
          "Gorky": "горьк", "Kazan": "казан", "Kuibyshev": "куйбышев", "Rostov": "ростов", "Kaliningrad": "калининград",
          "Tokyo": "токио"}
NOT_PLACES = {"Soviet", "Union", "USSR", "Russia", "Eastern", "Europe", "The", "Cyrillic", "Lenin", "Culture", "St"}


def first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s+", " ".join(text.split()), maxsplit=1)[0]


def md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


# --- facts for leakage ----------------------------------------------------------------------

def places_of(longs: list[str]) -> dict[str, str]:
    """{English name: Russian stem or ''} for the cities and the names after "in" in the long captions."""
    out = dict(CITIES)
    for long in longs:
        for m in re.finditer(r"\bin (?:the )?([A-Z][a-z]{2,}(?: [A-Z][a-z]{2,})?)", long):
            name = m.group(1)
            if not set(name.split()) & NOT_PLACES:
                out.setdefault(name, "")
    return out


def facts(short: str, places: dict[str, str]) -> list[tuple[str, str]]:
    """[(kind, key)] the short asserts: concept names, vehicle models, places, years and decades."""
    out = [("concept", name) for name in rc.concepts_in(short)]
    for m in VEHICLE_EN.finditer(short):
        out.append(("vehicle", m.group(1).lower() + (m.group(2) or "").replace("-", "").lower() + m.group(3)))
    for m in VEHICLE_RU.finditer(short):
        brand = RU_BRANDS[m.group(1).lower()]
        out.append(("vehicle", brand + ("m" if m.group(2) else "") + m.group(3)))
    for key, rx in MODEL_NAMES.items():
        if re.search(rx, short, re.I):
            out.append(("vehicle", key))
    for name, ru in places.items():
        if re.search(rf"\b{re.escape(name)}\b", short, re.I) or (ru and re.search(ru, short, re.I)):
            out.append(("place", name))
    for m in YEAR.finditer(short):
        if m.group(3):
            out.append(("decade", m.group(1)))
        else:
            out.append(("year", m.group(1) + m.group(2)))
    for m in SHORT_DECADE.finditer(short):
        out.append(("decade", "19" + m.group(1)))
    return sorted(set(out))


# The vehicle keys a concept name stands for: "VAZ-2101 Zhiguli" for a "Lada 1200" is the prompt's rule.
CONCEPT_VEHICLES = {"VAZ-2101 Zhiguli": {"vaz2101", "zhiguli"}, "GAZ-21 Volga": {"gaz21", "volga"},
                    "GAZ-24 Volga": {"gaz24", "volga"}, "GAZ-M20 Pobeda": {"gazm20", "pobeda"},
                    "Moskvich-408": {"moskvich408"}, "LiAZ-677": {"liaz677"}, "ZiU-9 trolleybus": {"ziu9"},
                    "Tu-154": {"tu154"}, "Il-62": {"il62"}}


def supported(fact: tuple[str, str], long: str, short: str = "") -> bool:
    kind, key = fact
    low = long.lower()
    if kind == "concept":
        return next(cond for name, _, cond in rc.CONCEPTS if name == key)(long)
    if kind == "vehicle":
        if any(key in CONCEPT_VEHICLES.get(name, ()) for name in rc.verified_concepts(short, long)):
            return True
        if key in MODEL_NAMES:
            return bool(re.search(rf"\b{key}\b", low))
        return key in re.sub(r"[\s-]", "", low)
    if kind == "place":
        return bool(re.search(rf"\b{re.escape(key.lower())}\b", low))
    if kind == "decade":
        return bool(re.search(rf"{key}\d|(?<!\d)'?{key[-1]}0s", low))
    return key in low                                    # a year


def unsupported(short: str, long: str, places) -> list[tuple[str, str]]:
    return [f for f in facts(short, places) if not supported(f, long, short)]


# --- the runs -------------------------------------------------------------------------------

def run_mode(client, system, items: list[rc.Item], batch: int) -> tuple[list[dict], float]:
    out = []
    t0 = time.time()
    for start in range(0, len(items), batch):
        chunk = [rc.Item(txt=it.txt, style=it.style, long=it.long) for it in items[start:start + batch]]
        rc.process_chunk(client, system, chunk, retries=0, write=False)
        for k, it in enumerate(chunk):
            out.append({"index": start + k, "batch": start // batch, "path": str(it.txt), "style": it.style,
                        "short": it.short, "ok": it.status == "ok", "reason": it.reason, "seconds": it.seconds})
        done = start + len(chunk)
        print(f"  batch {batch}: {done}/{len(items)}, {sum(r['ok'] for r in out)} pass, "
              f"{(time.time() - t0) / done:.2f} s/item", flush=True)
    return out, time.time() - t0


# --- the report -----------------------------------------------------------------------------

def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.0f} % ({a}/{b})" if b else "-"


def leakage(rows: list[dict], longs: list[str], places) -> list[tuple]:
    """[(row, fact, the other item's index)]: unsupported facts of a short that another long caption
    of its batch supports."""
    leaks = []
    for r in rows:
        mates = [x for x in rows if x["batch"] == r["batch"] and x["index"] != r["index"]]
        for f in unsupported(r["short"], longs[r["index"]], places):
            donor = next((m["index"] for m in mates if supported(f, longs[m["index"]], r["short"])), None)
            if donor is not None:
                leaks.append((r, f, donor))
    return leaks


def report(data: dict) -> str:
    single, batch = data["single"], data["batch"]
    n_batch = data["batch_size"]
    longs = data["longs"]
    styles = [k for k in rc.STYLE_LETTERS if any(r["style"] == k for r in single)]
    places = places_of(longs)
    modes = [("single", single, data["seconds_single"]), (f"batch-{n_batch}", batch, data["seconds_batch"])]

    L = [f"# Batch versus single: recaption.py", "",
         f"{len(longs)} items of `{data['dataset']}` (the first in walk order), model `{data['model']}`, "
         f"temperature {data['temperature']}, {data['date']}. First try only: no retries, nothing written. "
         f"Styles: " + ", ".join(f"{k} {sum(r['style'] == k for r in single)}" for k in styles) + ".", ""]

    L += ["## Speed and first-try pass rate", "", "| mode | wall clock | s/item | pass rate |", "|---|---|---|---|"]
    rates = {}
    for name, rows, secs in modes:
        good = sum(r["ok"] for r in rows)
        rates[name] = 100 * good / len(rows)
        L.append(f"| {name} | {secs:.0f} s | {secs / len(rows):.2f} | {pct(good, len(rows))} |")
    L += ["", "## Pass rate per style", "", "| style | " + " | ".join(m[0] for m in modes) + " |",
          "|---|" + "---|" * len(modes)]
    for k in styles:
        cells = [pct(sum(r["ok"] for r in rows if r["style"] == k), sum(r["style"] == k for r in rows))
                 for _, rows, _ in modes]
        L.append(f"| {k} | " + " | ".join(cells) + " |")

    L += ["", "## Mean word count per style", "", "Over every short returned, pass or fail; B also as tags.", "",
          "| style | " + " | ".join(m[0] for m in modes) + " |", "|---|" + "---|" * len(modes)]
    for k in styles:
        cells = []
        for _, rows, _ in modes:
            shorts = [r["short"] for r in rows if r["style"] == k and r["short"]]
            if not shorts:
                cells.append("-")
                continue
            cell = f"{statistics.mean(rc.word_count(s) for s in shorts):.1f}"
            if k == "B":
                cell += f" ({statistics.mean(len([t for t in s.split(',') if t.strip()]) for s in shorts):.1f} tags)"
            cells.append(cell)
        L.append(f"| {k} | " + " | ".join(cells) + " |")

    eligible = [i for i, long in enumerate(longs) if any(cond(long) for _, _, cond in rc.CONCEPTS)]
    L += ["", "## Verified concept names", "",
          f"A short has a verified concept name when it names a concept whose condition its long caption meets. "
          f"{len(eligible)} of the {len(longs)} long captions meet at least one condition.", "",
          "| mode | all items | eligible items |", "|---|---|---|"]
    for name, rows, _ in modes:
        has = [bool(rc.verified_concepts(r["short"], longs[r["index"]])) for r in rows]
        L.append(f"| {name} | {pct(sum(has), len(rows))} | {pct(sum(has[i] for i in eligible), len(eligible))} |")

    unsupported_rows = []
    for name, rows, _ in modes:
        for r in rows:
            bad = unsupported(r["short"], longs[r["index"]], places)
            if bad:
                unsupported_rows.append((name, r, bad))
    n_unsupported = {name: sum(len(b) for m, _, b in unsupported_rows if m == name) for name, _, _ in modes}
    candidates = leakage(batch, longs, places)
    # A fact the single-mode short of the same item asserts too came from the item's own caption, not the batch.
    leaks = [c for c in candidates if c[1] not in unsupported(single[c[0]["index"]]["short"], longs[c[0]["index"]],
                                                              places)]
    L += ["", "## Cross-item leakage in batch mode", "",
          f"Facts (concept names, vehicle models, places, years and decades) in a short that its own long caption "
          f"does not support: {n_unsupported['single']} in single mode, {n_unsupported[f'batch-{n_batch}']} in "
          f"batch mode. A leak candidate is a batch-mode one that another long caption of the same batch supports: "
          f"{len(candidates)}. A candidate the single-mode short of the same item also asserts is the model's "
          f"reading of its own caption, not leakage; the rest are leaks: {len(leaks)}.", ""]
    if candidates:
        L += ["| item | style | fact | from item | also in single | short | own long caption, first sentence |",
              "|---|---|---|---|---|---|---|"]
        for c in candidates:
            r, (kind, key), donor = c
            L.append(f"| {r['index'] + 1} | {r['style']} | {kind} {md_cell(key)} | {donor + 1} | "
                     f"{'no' if c in leaks else 'yes'} | {md_cell(r['short'])} "
                     f"| {md_cell(first_sentence(longs[r['index']]))} |")
    if unsupported_rows:
        L += ["", "Every unsupported fact:", "", "| mode | item | fact | short |", "|---|---|---|---|"]
        for name, r, bad in unsupported_rows:
            L.append(f"| {name} | {r['index'] + 1} | {', '.join(f'{k} {v}' for k, v in bad)} | {md_cell(r['short'])} |")

    L += ["", "## First-try failures", ""]
    for name, rows, _ in modes:
        counts: dict[str, int] = {}
        for r in rows:
            if not r["ok"]:
                for reason in r["reason"].split("; "):
                    key = reason.split(" (")[0].split(":")[0]
                    counts[key] = counts.get(key, 0) + 1
        L.append(f"{name}: " + ("; ".join(f"{c} x {k}" for k, c in sorted(counts.items(), key=lambda kv: -kv[1]))
                                or "none") + ".")
        L.append("")

    step = max(1, len(longs) // SIDE_BY_SIDE)
    picks = list(range(0, len(longs), step))[:SIDE_BY_SIDE]
    L += ["## Side by side", "", f"Every {step}th item. A failed first try is marked ✗.", "",
          f"| # | style | long caption, first sentence | single | batch-{n_batch} |", "|---|---|---|---|---|"]
    for i in picks:
        s, b = single[i], batch[i]
        cell = lambda r: ("✗ " if not r["ok"] else "") + md_cell(r["short"] or "(none)")
        L.append(f"| {i + 1} | {s['style']} | {md_cell(first_sentence(longs[i]))} | {cell(s)} | {cell(b)} |")

    diff = rates["single"] - rates[f"batch-{n_batch}"]
    verdict = diff <= PASS_MARGIN and not leaks
    L += ["", "## Verdict", "",
          f"Batch-{n_batch} passes {rates[f'batch-{n_batch}']:.0f} % on the first try against {rates['single']:.0f} % "
          f"for single ({diff:+.0f} points for single), with {len(leaks)} leak(s). The rule (within "
          f"{PASS_MARGIN:.0f} points and no leakage) " + ("holds: the default --batch can be "
                                                          f"{n_batch}." if verdict else
                                                          "does not hold: try a smaller batch."), ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare batch and single mode of recaption.py on the first items "
                                             "of a dataset.")
    ap.add_argument("dataset", type=Path, nargs="?", default=DEFAULT_DATASET)
    ap.add_argument("--items", type=int, default=100)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--ext", default=".txt")
    ap.add_argument("--endpoint", default=rc.DEFAULT_ENDPOINT)
    ap.add_argument("--model", default="auto")
    ap.add_argument("--styles", default=rc.DEFAULT_STYLES)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--report-only", action="store_true", help=f"rebuild {OUT_MD.name} from {OUT_JSON.name}")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

    if a.report_only:
        data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    else:
        root = a.dataset.resolve()
        items = rc.find_items(root, a.ext, rc.parse_styles(a.styles))[:a.items]
        for it in items:
            it.long = rc.read_text(it.txt).strip()
        system = rc.load_system_prompt()
        client = rc.ChatClient(a.endpoint, a.model, a.temperature)
        model = client.resolve_model()
        print(f"{len(items)} items, model {model}")
        print("single mode")
        single, t_single = run_mode(client, system, items, 1)
        print(f"batch-{a.batch} mode")
        batch, t_batch = run_mode(client, system, items, a.batch)
        data = {"dataset": str(root), "model": model, "temperature": a.temperature, "batch_size": a.batch,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "longs": [it.long for it in items],
                "single": single, "batch": batch, "seconds_single": t_single, "seconds_batch": t_batch}
        OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT_MD.write_text(report(data), encoding="utf-8")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
