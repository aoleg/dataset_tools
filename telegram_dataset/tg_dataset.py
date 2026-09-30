"""Build an image + sidecar caption dataset from a Telegram channel JSON export.

Usage: python tg_dataset.py EXPORT_DIR [--photos DIR] [--dry-run]
EXPORT_DIR holds result.json and photos/; the dataset goes to EXPORT_DIR/dataset.
If photos/ was renamed (for example photos_watermark/), pass --photos, or leave
it out when EXPORT_DIR has exactly one photos* folder.
--dry-run only prints statistics. In an existing dataset folder, every .txt
sidecar is rewritten and images that are already there are not copied again.
"""
import argparse, json, re, sys
from collections import Counter
from pathlib import Path
from shutil import copy2

sys.stdout.reconfigure(encoding="utf-8")

parser = argparse.ArgumentParser(description="Build an image + caption dataset from a Telegram JSON export.")
parser.add_argument("export", type=Path, help="export folder with result.json and photos/")
parser.add_argument("--photos", help="image folder inside EXPORT_DIR (default: photos, or the only photos* folder)")
parser.add_argument("--dry-run", action="store_true", help="print statistics only, write nothing")
args = parser.parse_args()

EXPORT = args.export
OUT = EXPORT / "dataset"
MAX_WORDS = 80   # a caption keeps whole paragraphs up to this many words
MIN_WORDS = 20   # below this, a paragraph that does not fit is cut at a sentence end
DRY_RUN = args.dry_run

if not (EXPORT / "result.json").is_file():
    sys.exit(f"{EXPORT / 'result.json'} not found; EXPORT_DIR must be the folder that holds result.json")
if args.photos:
    PHOTOS = EXPORT / args.photos
elif (EXPORT / "photos").is_dir():
    PHOTOS = EXPORT / "photos"
else:
    found = [d for d in EXPORT.iterdir() if d.is_dir() and d.name.startswith("photos")]
    if len(found) != 1:
        sys.exit(f"no photos folder in {EXPORT}; found {[d.name for d in found]}; pass --photos")
    PHOTOS = found[0]
if not PHOTOS.is_dir():
    sys.exit(f"{PHOTOS} is not a folder")
print("images from:", PHOTOS)

export = json.loads((EXPORT / "result.json").read_text(encoding="utf-8"))
CHANNEL_NAME = export.get("name", "")

# Found by review, per channel id: announcements, cross-promos, giveaways and disguised ads.
PROMO_IDS = {
    1741939547: {6619, 8182, 8260, 9555, 9622, 9777, 9835, 9890, 9964, 9986, 9999, 10120, 10168, 10221, 10377},  # Прекрасное далёко
    2061579869: {278, 1027, 1734, 2323},  # Советский Фотоальбом
}
# Rules that should find new promos in a later export; anything they flag outside PROMO_IDS is printed.
PROMO_HINT = re.compile(r"\bМ[АA][ХX]\b|\bMAX\b|max\.ru|Вот ссылка|Присоединяйтесь|Подписывайся|"
                        r"Добро пожаловать в «Архив|промокод|скидк|erid|Дарим", re.I)

# Short subscribe calls and "we are now in MAX" notices that end a post
FOOTER_CALL = re.compile(r"подписывайтесь|подписывайся|подпишитесь|переходите|присоединяйтесь|"
                         r"\bв\s+(МАХ|MAX)\b", re.I)
# Sentences inside a caption that address readers about the channel, not the photo.
# Subscribe calls and MAX notices always go; a mention of "the channel" goes only in a short
# sentence, because long ones carry facts, and "канал им./имени" is a real canal.
SUBSCRIBE_TALK = re.compile(r"подписывайтесь|подписывайся|подпишитесь|\bв\s+(МАХ|MAX)\b", re.I)
CHANNEL_MENTION = re.compile(r"\b(на|в)\s+(наш(ем)?\s+)?канал(е)?\b(?!\s+им)|"
                             r"\b(предыдущ|следующ|прошл)\w*\s+пост", re.I)


def channel_talk(s):
    return SUBSCRIBE_TALK.search(s) or (CHANNEL_MENTION.search(s) and len(s.split()) <= 25)

# Latin, Greek and small-capital lookalikes the channel mixes into Cyrillic words
HOMOGLYPHS = str.maketrans(
    "aceopxyACEHKMOPTXBĸᴏοᴍëΓᴙᴦʜΒΜκΚɜΗΠ",
    "асеорхуАСЕНКМОРТХВкоомёГягнВМкКзНП",
)
EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D\u20E3]")

# A period after these does not end a sentence
ABBR = {"г", "гг", "ул", "пл", "им", "др", "см", "км", "св", "ст", "руб", "коп", "тыс", "млн", "млрд",
        "худ", "обл", "пр", "просп", "пер", "наб", "ок", "вв", "стр", "рис", "фот", "акад", "проф", "ген",
        "тов", "т", "е", "д", "н", "э", "р", "с", "ж", "ср", "см", "мм", "кв", "ред", "изд", "арх"}
QUESTION = re.compile(r"\?[?!»\"”)\s.]*$")
QUESTION_TAG = re.compile(r",\s*(а\s+)?(вы\s+)?((помните|узнали|узна[её]те)(\s+[\w-]+){0,2}|правда|не так ли|согласны|верно|да)\s*\?[?!.]*$", re.I)


def fix_word(w):
    return w.translate(HOMOGLYPHS) if re.search(r"[а-яёА-ЯЁ]", w) else w


def normalize(s):
    s = re.sub(r"\w+", lambda w: fix_word(w.group()), s.replace("\xa0", " "))
    return EMOJI.sub("", s)


def is_footer(line, link_texts):
    """The channel's own name, a bare URL, a whole line of channel-link text, or a short subscribe call."""
    l = line.strip().rstrip(":!.").strip()
    name = CHANNEL_NAME.lower()
    return (l.lower() == name or l.lower().startswith(name + " |")
            or re.fullmatch(r"https?://\S+", l) is not None
            or l in link_texts
            or (len(l.split()) <= 12 and FOOTER_CALL.search(l) is not None))


def clean(m):
    link_texts = {normalize(e["text"]).strip() for e in m["text_entities"]
                  if re.search(r"t\.me/|max\.ru", e.get("href") or e["text"])}
    s = normalize("".join(e["text"] for e in m["text_entities"]))
    lines = [l.strip() for l in s.strip().splitlines()]
    while lines and (not lines[-1] or is_footer(lines[-1], link_texts)):
        lines.pop()
    lines = [re.sub(r"\s+([,.;:!?»)])", r"\1", re.sub(r"\s{2,}", " ", l)) for l in lines]
    return [l for l in lines if l]


def sentences(p):
    out, start = [], 0
    for b in re.finditer(r"[.!?…]+[»\"”)]*\s+(?=\S)", p):
        prev = re.search(r"(\w+)$", p[:b.start()])
        nxt = p[b.end()]
        if b.group()[0] == "." and prev and (len(prev.group(1)) == 1 or prev.group(1).lower() in ABBR):
            continue
        if not (nxt.isupper() or nxt.isdigit() or nxt in "«\"“„(—–-"):
            continue
        out.append(p[start:b.end()].strip())
        start = b.end()
    if p[start:].strip():
        out.append(p[start:].strip())
    return out


def end_with_period(s):
    s = s.rstrip(" ,;:—–-!…")
    return s if s.endswith(".") else s + "."


def trim_questions(paras):
    """Drop question sentences; if nothing else is left, keep the caption as it was."""
    out, trimmed = [], False
    for sents in paras:
        kept = []
        for s in sents:
            if not QUESTION.search(s):
                kept.append(s)
                continue
            trimmed = True
            t = QUESTION_TAG.sub("", s).strip()
            if t != s and t:
                kept.append(end_with_period(t))
        if kept:
            out.append(kept)
    if not out or (trimmed and sum(len(s.split()) for p in out for s in p) < 3):
        return paras, "question_only"
    if trimmed:
        out[-1][-1] = end_with_period(out[-1][-1])
        return out, "trimmed"
    return out, None


def truncate(paras, limit=MAX_WORDS, enough=MIN_WORDS):
    """Keep whole paragraphs while they fit in `limit` words. A paragraph that does not fit
    contributes sentences only until the caption reaches `enough` words: less is more, so an
    81-word paragraph can end up as its 21-word first sentence. A first sentence longer than
    `limit` is kept whole rather than cut mid-phrase."""
    out, n = [], 0
    for sents in paras:
        words = sum(len(s.split()) for s in sents)
        if n + words <= limit:
            out.extend(sents)
            n += words
            continue
        for s in sents:
            k = len(s.split())
            if n >= enough or n + k > limit:
                break
            out.append(s)
            n += k
        break
    if not out:
        out = [paras[0][0]]
    return out, len(out) < sum(len(p) for p in paras)


def caption_of(m):
    paras = [[s for s in sentences(l) if not channel_talk(s)] for l in clean(m)]
    paras = [p for p in paras if p]
    if not paras:
        return "", None, False
    paras, qflag = trim_questions(paras)
    for sents in paras[:-1]:  # a line break ends a sentence even without punctuation
        if not re.search(r"[.!?…:;]$", sents[-1]):
            sents[-1] += "."
    sents, cut = truncate(paras)
    return re.sub(r"\s{2,}", " ", " ".join(sents)).strip(), qflag, cut


msgs = export["messages"]
promo_ids = PROMO_IDS.get(export.get("id"), set())
print("channel:", export.get("name"), export.get("id"), "| reviewed promo ids:", len(promo_ids))

rows, prev = [], None
for m in msgs:
    if m.get("type") != "message" or "photo" not in m:
        continue
    same_post = prev and prev["unix"] == m["date_unixtime"] and prev["id"] == m["id"] - 1
    raw = "".join(e["text"] for e in m["text_entities"])
    cap, qflag, cut = caption_of(m)
    rows.append({
        "id": m["id"], "unix": m["date_unixtime"], "date": m["date"],
        "group": prev["group"] if same_post else m["id"],
        "src": (PHOTOS / Path(m["photo"]).name).relative_to(EXPORT).as_posix(),
        "exists": (PHOTOS / Path(m["photo"]).name).is_file(),
        "w": m.get("width"), "h": m.get("height"),
        "raw": raw, "clean": " ".join(clean(m)), "own_caption": cap, "question": qflag, "truncated": cut,
        "reactions": sum(r["count"] for r in m.get("reactions", [])),
    })
    prev = rows[-1]

promo_groups = {r["group"] for r in rows if r["id"] in promo_ids}
by_group = {}
for r in rows:
    by_group.setdefault(r["group"], []).append(r)
for g, members in by_group.items():
    source = next((r for r in members if r["own_caption"]), None)
    for r in members:
        r["drop"] = "promo" if g in promo_groups else "missing" if not r["exists"] else None
        r["caption"] = source["own_caption"] if source else ""
        r["caption_from"] = source["id"] if source else None

kept = [r for r in rows if not r["drop"]]
if rows and not kept:
    sys.exit(f"none of the {len(rows)} photos in result.json were found in {PHOTOS}")
print("photo messages:", len(rows))
print("dropped:", dict(Counter(r["drop"] for r in rows if r["drop"])))
print("kept:", len(kept))
print("  with caption:", sum(bool(r["caption"]) for r in kept),
      "(own:", sum(bool(r["own_caption"]) for r in kept),
      "| from album:", sum(bool(r["caption"]) and not r["own_caption"] for r in kept), ")")
print("  without caption:", sum(not r["caption"] for r in kept))
own = [r for r in kept if r["own_caption"]]
print("  questions trimmed:", sum(r["question"] == "trimmed" for r in own),
      "| question-only kept:", sum(r["question"] == "question_only" for r in own))
print("  shortened:", sum(r["truncated"] for r in own))
lengths = sorted(len(r["own_caption"].split()) for r in own)
if lengths:
    print(f"  words per caption: median {lengths[len(lengths) // 2]}, longest {lengths[-1]}, "
          f"under {MIN_WORDS}: {sum(n < MIN_WORDS for n in lengths)}, "
          f"over {MAX_WORDS} (one long first sentence): {sum(n > MAX_WORDS for n in lengths)}")
print("  multi-line captions:", sum("\n" in r["caption"] for r in kept))

flagged = [r for r in rows if r["id"] not in promo_ids and PROMO_HINT.search(r["clean"])]
print("\npromo rules outside PROMO_IDS:", [(r["id"], r["clean"][:60]) for r in flagged])

print("\ntrimmed questions:")
for r in [r for r in own if r["question"] == "trimmed"][:25]:
    print(f"  {r['id']:6} {' '.join(r['raw'].split())[-90:]!r}\n         -> {r['own_caption'][-90:]!r}")
print("\nquestion-only:")
for r in [r for r in own if r["question"] == "question_only"][:10]:
    print(f"  {r['id']:6} {r['own_caption'][:100]!r}")
print("\ntruncated (tail):")
for r in [r for r in own if r["truncated"]][:12]:
    print(f"  {r['id']:6} {len(r['own_caption'].split()):3}w  ...{r['own_caption'][-110:]!r}")

if not DRY_RUN:
    OUT.mkdir(parents=True, exist_ok=True)
    copied = present = written = 0
    stale_txt, stems = [], set()
    with open(OUT / "manifest.jsonl", "w", encoding="utf-8") as mf:
        for r in rows:
            mf.write(json.dumps(r, ensure_ascii=False) + "\n")
            if r["drop"]:
                continue
            src = EXPORT / r["src"]
            stem = f"{r['id']:06d}"
            stems.add(stem)
            image = OUT / f"{stem}{src.suffix.lower()}"
            if image.exists():
                present += 1
            else:
                copy2(src, image)
                copied += 1
            txt = OUT / f"{stem}.txt"
            if r["caption"]:
                txt.write_text(r["caption"], encoding="utf-8")
                written += 1
            elif txt.exists():
                stale_txt.append(txt.name)
    stale_images = sorted(p.name for p in OUT.iterdir()
                          if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp") and p.stem not in stems)
    print(f"\nwritten to {OUT}: images copied {copied}, already present {present}, captions written {written}")
    if stale_txt:
        print(f"left in place, image now has no caption ({len(stale_txt)}):", stale_txt[:20])
    if stale_images:
        print(f"left in place, image no longer in the dataset ({len(stale_images)}):", stale_images[:20])
