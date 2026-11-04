#!/usr/bin/env python3
"""Move images that a local LLM finds unrelated to the dataset's topic (the
Soviet Union) out of a dataset, into a sibling folder "<dataset>_unrelated".

An item is every "<stem>.txt" caption that has an image of the same stem next
to it, anywhere under <dataset> (any depth, Cyrillic paths included). Folders
whose names start with a dot, the output folders of the dataset tools
(EXCLUDES) and folders marked with MARK are not scanned.

The captions go to the LLM in batches (--batch, default 10) as a JSON array
[{"id": 1, "caption": "..."}, ...] built in memory. The system prompt (PROMPT
below) asks for one verdict per caption, YES, NO or INDETERMINATE, answered as
a JSON array of single letters in input order: ["Y", "N", "I"]. The reply is
as short as it can be, because generation, not prompt reading, is what costs
time on a large model. The reply is parsed in memory; nothing of the exchange
is written to disk unless --log names a CSV. For every NO the image and every
other file of its name in the same folder (.txt, .json, .short, caches, any
extension) are moved to "<dataset>_unrelated/<same relative path>", so the
original tree is kept. YES and INDETERMINATE items stay where they are.

A batch whose reply is unusable (no JSON array, not one letter per caption, a
letter outside Y/N/I) is asked once more; an item still unanswered counts as
INDETERMINATE and is reported, never moved. A rerun is resumable: moved items
are gone, the rest are asked again.

The LLM is any OpenAI-compatible chat endpoint; the default is KoboldCpp at
http://localhost:5001/v1/. Requests ask for no reasoning and temperature 0.

Usage:
    python unrelated.py <dataset> [--dry-run] [--batch 10] [--limit N]
                        [--endpoint URL] [--log verdicts.csv] [--quiet]
"""

import argparse
import codecs
import csv
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".avif"}
CAPTION_EXT = ".txt"
OUT_SUFFIX = "_unrelated"
# Output folders the dataset tools make next to or inside a dataset; they hold no items.
EXCLUDES = {"_controls", "masks", "faces", "_backup", "_duplicates", "_prep", "_classify",
            "_embeddings", "_reframed", "_uncaptioned", "_unrelated"}
MARK = "_dataset_tools.json"    # marks a folder the dataset tools made next to a dataset

DEFAULT_ENDPOINT = "http://localhost:5001/v1/"
DEFAULT_BATCH = 10
VERDICTS = ("YES", "NO", "INDETERMINATE")
LETTERS = {"Y": "YES", "N": "NO", "I": "INDETERMINATE"}
TOKENS_PER_ITEM = 4             # max_tokens = this per item + TOKENS_BASE ("Y", is 3 tokens at most)
TOKENS_BASE = 16
TIMEOUT = 600                   # seconds for one reply
RETRIES = 1                     # extra tries for a batch with an unusable reply
STOP_SEQUENCES = ["<|return|>", "<|im_end|>", "<end_of_turn>", "<|eot_id|>"]
THINKING_BLOCK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.S | re.I)

PROMPT = """You sort captions of historical photographs for a dataset whose topic is the Soviet Union (USSR, 1922-1991). For each caption you decide whether the photograph belongs to that topic.

The user message is a JSON array of objects {"id": n, "caption": "..."}. Answer with exactly one JSON array of single-letter strings, one per input object, in the order of the ids, and nothing else: "Y" for YES, "N" for NO, "I" for INDETERMINATE. For three captions the answer looks like:
["Y", "N", "I"]
Never write the words, the ids, the captions or any explanation.

Judge every caption on its own text only. Never let one caption influence another.

YES: the caption places the photograph in the Soviet Union or ties its subject to it. Signs of this:
- a place in the USSR: "Soviet Union", "USSR", "Soviet", or a republic or city of it (Russia, Ukraine, Belarus, the Baltic republics, Georgia, Armenia, Azerbaijan, Kazakhstan, Uzbekistan and the other Central Asian republics, Moldova; Moscow, Leningrad, Kiev/Kyiv, Minsk, Riga, Tallinn, Vilnius, Tbilisi, Yerevan, Baku, Tashkent, Alma-Ata, Odessa, Kharkov, Yakutsk, Novosibirsk, Sverdlovsk, Gorky, Vladivostok, Murmansk, Sochi, Yalta, Sevastopol and any other Soviet city, town, village or region);
- Soviet institutions, people or things: Soviet army or navy, Red Army, Komsomol, pioneers with pioneer ties, Octobrists, Soviet school uniform, kolkhoz, a Soviet factory, a Soviet leader or celebrity (Lenin, Stalin, Khrushchev, Brezhnev, Gagarin, Vysotsky, Soviet actors and athletes), Soviet cars and vehicles (GAZ, ZIL, Volga, Moskvich, Zhiguli/Lada, Zaporozhets, LiAZ, Ikarus, ZiU, Tu-, Il-, Aeroflot), signs in Cyrillic, Soviet symbols (hammer and sickle, red star, Lenin portraits, Soviet posters);
- a Russian-language caption that names a Soviet place, institution, factory, event or person.
A Soviet subject photographed abroad (a Soviet delegation, a Soviet car shown in New York, Soviet athletes at a foreign competition) is YES.

NO: the caption places the photograph outside the USSR or ties it to a foreign subject with no Soviet connection. Signs of this:
- a foreign place: Paris, London, Berlin, Rome, New York, Chicago, Los Angeles, Hollywood, "the United States", "USA", France, Italy, Germany, Great Britain, Japan, China, Afghanistan or any other place outside the USSR;
- a Western celebrity named without any Soviet place or Soviet connection: Marilyn Monroe, Audrey Hepburn, Elvis Presley, Sharon Stone, Cindy Crawford and the like;
- clearly Western details with no Soviet hint: a US street sign, an American diner, a storefront in English, a Hollywood film set.
A Western celebrity photographed in Moscow or elsewhere in the USSR is YES, not NO.

INDETERMINATE: the caption has no definite hint either way: no place, no Soviet or foreign name, no Soviet or Western detail. A plain scene (a woman on a bench, a family at a table, a man repairing a radio) with no place is INDETERMINATE even if the style of the photograph suggests an era. A Russian-language caption with no place and no Soviet name is INDETERMINATE.

When YES and NO signs both appear, decide by the stronger tie: a Soviet subject abroad is YES; a foreign subject in the USSR is YES; a foreign subject abroad is NO.

Answer with only the JSON array: no code fences, no comments, no text before or after it."""


class EndpointDown(Exception):
    """The endpoint does not answer: nothing more can be done this run."""


# --- files ----------------------------------------------------------------------------------

def read_text(path):
    """UTF-16 with a BOM, UTF-8 (a BOM is dropped), else cp1251."""
    with open(path, "rb") as f:
        data = f.read()
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1251", errors="replace")


def skipped(entry):
    name = entry.name
    return (name.startswith(".") or name.casefold() in EXCLUDES
            or os.path.isfile(os.path.join(entry.path, MARK)))


def walk(root):
    """os.walk that prunes the skipped folders, folders sorted for a stable order."""
    for folder, dirs, files in os.walk(root):
        keep = []
        for d in sorted(dirs):
            with os.scandir(folder) as it:
                entry = next((e for e in it if e.name == d), None)
            if entry is not None and not skipped(entry):
                keep.append(d)
        dirs[:] = keep
        yield folder, files


def find_items(root):
    """[(folder, stem, image name, caption text)] for every <stem>.txt with an image of the same
    stem in the same folder, in path order; with the counts of .txt without an image and of
    empty captions, both skipped."""
    items, no_image, empty = [], 0, 0
    for folder, files in walk(root):
        by_stem = {}
        for name in files:
            by_stem.setdefault(os.path.splitext(name)[0], []).append(name)
        for name in sorted(files, key=str.casefold):
            stem, ext = os.path.splitext(name)
            if ext.lower() != CAPTION_EXT:
                continue
            image = next((n for n in sorted(by_stem[stem]) if os.path.splitext(n)[1].lower() in IMAGE_EXTS), None)
            if image is None:
                no_image += 1
                continue
            text = " ".join(read_text(os.path.join(folder, name)).split())
            if not text:
                empty += 1
                continue
            items.append((folder, stem, image, text))
    return items, no_image, empty


def move_item(root, dst_root, folder, stem, dry_run):
    """Move every file of the stem in folder to the same place under dst_root. Returns the number
    of files moved (or that would be), or -1 when a target exists (then nothing of the item moves)."""
    rel = os.path.relpath(folder, root)
    dst_folder = dst_root if rel == "." else os.path.join(dst_root, rel)
    names = sorted(n for n in os.listdir(folder) if os.path.splitext(n)[0] == stem
                   and os.path.isfile(os.path.join(folder, n)))
    for n in names:
        if os.path.lexists(os.path.join(dst_folder, n)):
            print(f"  SKIPPED, target exists: {os.path.join(dst_folder, n)}", file=sys.stderr)
            return -1
    if dry_run:
        return len(names)
    os.makedirs(dst_folder, exist_ok=True)
    for n in names:
        shutil.move(os.path.join(folder, n), os.path.join(dst_folder, n))
    return len(names)


# --- the LLM --------------------------------------------------------------------------------

class Client:
    def __init__(self, endpoint):
        self.base = endpoint.rstrip("/")
        self.model = "local-model"
        self.send_reasoning_effort = True

    def resolve_model(self):
        try:
            with urllib.request.urlopen(f"{self.base}/models", timeout=15) as r:
                data = json.loads(r.read().decode("utf-8")).get("data") or []
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise EndpointDown(f"{self.base} does not answer ({e}); start the LLM server or pass --endpoint")
        if not data:
            raise EndpointDown(f"{self.base}/models lists no model")
        self.model = data[0]["id"]
        return self.model

    def complete(self, user, max_tokens):
        payload = {"model": self.model,
                   "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": user}],
                   "max_tokens": max_tokens, "temperature": 0.0, "stop": STOP_SEQUENCES,
                   "chat_template_kwargs": {"enable_thinking": False}}
        if self.send_reasoning_effort:
            payload["reasoning_effort"] = "none"
        req = urllib.request.Request(f"{self.base}/chat/completions", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                body = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", errors="replace")
            if e.code == 400 and "reasoning_effort" in text and self.send_reasoning_effort:
                self.send_reasoning_effort = False      # some endpoints reject it: once, then never again
                return self.complete(user, max_tokens)
            return None, f"HTTP {e.code}: {text[:200]}"
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise EndpointDown(f"no answer from {self.base} ({e})")
        except ValueError as e:
            return None, f"the reply is not JSON: {e}"
        msg = (body.get("choices") or [{}])[0].get("message") or {}
        return THINKING_BLOCK.sub("", msg.get("content") or "").strip(), ""


def extract_json_array(text):
    """The first JSON array of strings in the reply (code fences and text around it are skipped)."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\[", text):
        try:
            val, _ = dec.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(val, list) and val and all(isinstance(v, str) for v in val):
            return val
    return None


def parse_verdicts(reply, n):
    """[verdict] for the n captions in order, or (None, reason) when the reply is not one letter per
    caption. A whole word ("YES") is taken by its first letter."""
    if len(reply) != n:
        return None, f"{len(reply)} answer(s) for {n} caption(s)"
    out = []
    for k, s in enumerate(reply, 1):
        letter = s.strip().upper()[:1]
        if letter not in LETTERS or s.strip().upper() not in (letter, LETTERS[letter]):
            return None, f"answer {k} is {s!r}, not Y, N or I"
        out.append(LETTERS[letter])
    return out, ""


def ask(client, captions):
    """Verdicts for a batch of caption texts, in order; None for an item left unanswered."""
    user = json.dumps([{"id": i + 1, "caption": c} for i, c in enumerate(captions)], ensure_ascii=False)
    reason = ""
    for attempt in range(1 + RETRIES):
        text, reason = client.complete(user, TOKENS_PER_ITEM * len(captions) + TOKENS_BASE)
        if text is None:
            continue
        reply = extract_json_array(text)
        if reply is None:
            reason = f"the reply has no JSON array: {' '.join(text.split())[:100]!r}"
            continue
        verdicts, reason = parse_verdicts(reply, len(captions))
        if verdicts is not None:
            return verdicts, ""
    return [None] * len(captions), reason


# --- main -----------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Move images whose captions a local LLM finds unrelated to the "
                                             "Soviet Union to <dataset>_unrelated, keeping the folder tree.")
    ap.add_argument("dataset", help="the dataset folder")
    ap.add_argument("--dry-run", action="store_true", help="ask the LLM and print the verdicts; move nothing")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH, help=f"captions per request (default {DEFAULT_BATCH})")
    ap.add_argument("--limit", type=int, default=0, help="stop after N items (0 = all)")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help=f"OpenAI-compatible base URL (default {DEFAULT_ENDPOINT})")
    ap.add_argument("--log", help="write a CSV with one row per item: path, verdict, moved")
    ap.add_argument("--quiet", action="store_true", help="print only NO items and the totals")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    if args.batch < 1:
        ap.error("--batch must be at least 1")

    root = os.path.abspath(args.dataset)
    if not os.path.isdir(root):
        print(f"not a folder: {root}", file=sys.stderr)
        return 2
    dst_root = root.rstrip("\\/") + OUT_SUFFIX

    items, no_image, empty = find_items(root)
    if args.limit > 0:
        items = items[:args.limit]
    print(f"{root}\n  -> {dst_root}\n{len(items)} item(s); skipped: {no_image} .txt without an image, "
          f"{empty} empty caption(s)")
    if not items:
        return 0

    client = Client(args.endpoint)
    try:
        print(f"model {client.resolve_model()} at {client.base}, batch {args.batch}"
              + (", dry run" if args.dry_run else ""))
    except EndpointDown as e:
        print(e, file=sys.stderr)
        return 1

    log = None
    if args.log:
        log_file = open(args.log, "w", newline="", encoding="utf-8")
        log = csv.writer(log_file)
        log.writerow(["path", "verdict", "moved", "caption"])

    counts = {v: 0 for v in VERDICTS}
    unanswered = moved = files_moved = failed = 0
    verb = "would move" if args.dry_run else "move"
    t0 = time.time()
    rc = 0
    try:
        for start in range(0, len(items), args.batch):
            chunk = items[start:start + args.batch]
            verdicts, reason = ask(client, [c for _, _, _, c in chunk])
            if reason:
                print(f"  batch at item {start + 1} left unanswered: {reason}", file=sys.stderr)
            for (folder, stem, image, text), v in zip(chunk, verdicts):
                rel = os.path.relpath(os.path.join(folder, image), root)
                did_move = 0
                if v is None:
                    unanswered += 1
                    v = "INDETERMINATE"
                    tag = "UNANSWERED"
                else:
                    counts[v] += 1
                    tag = v
                if v == "NO":
                    n = move_item(root, dst_root, folder, stem, args.dry_run)
                    if n < 0:
                        failed += 1
                    else:
                        moved += 1
                        files_moved += n
                        did_move = n
                    print(f"{verb}: {rel}  +{max(n, 1) - 1} sidecar(s)")
                elif not args.quiet:
                    print(f"{tag:>13}: {rel}")
                if log is not None:
                    log.writerow([rel, tag, did_move, text])
            done = start + len(chunk)
            print(f"  {done}/{len(items)}, {(time.time() - t0) / done:.2f} s/item, "
                  f"NO {counts['NO']}  YES {counts['YES']}  INDETERMINATE {counts['INDETERMINATE']}"
                  + (f"  unanswered {unanswered}" if unanswered else ""), flush=True)
    except EndpointDown as e:
        print(f"\nstopped: {e}", file=sys.stderr)
        rc = 1
    except KeyboardInterrupt:
        print("\nstopped by the user", file=sys.stderr)
        rc = 130
    finally:
        if log is not None:
            log_file.close()

    verb = "would move" if args.dry_run else "moved"
    print(f"\n{root}\n  -> {dst_root}")
    print(f"items: {len(items)}  YES: {counts['YES']}  NO: {counts['NO']}  INDETERMINATE: {counts['INDETERMINATE']}"
          + (f"  unanswered (left in place): {unanswered}" if unanswered else ""))
    print(f"{verb}: {moved} image(s), {files_moved - moved} sidecar(s)"
          + (f"  FAILED (target exists): {failed}" if failed else "")
          + f"  in {time.time() - t0:.0f} s")
    if args.log:
        print(f"log: {os.path.abspath(args.log)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
