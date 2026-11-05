#!/usr/bin/env python3
"""judge.py - keep only the images of a dataset that meet a condition in words.

A generalisation of scripts/unrelated.py: the condition is given in natural
language ("the photograph shows a vehicle made in the USSR"), and three
layers decide, cheapest first, each run in turn over the whole folder:

1. captions: every image that has a <stem>.txt is judged on its caption by a
   local LLM on an OpenAI-compatible endpoint, in batches, YES, NO or
   INDETERMINATE (the batching, the reply format and the parsing of
   unrelated.py);
2. images (--images): SigLIP 2 zero-shot. The score of an image is the
   softmax probability of the positive prompts against the negative ones;
   the prompts are the condition itself plus three positive and three
   negative English prompts the LLM writes once from it (--positives and
   --negatives override). The encoder is classify's SigLIP 2 snapshot,
   loaded whole (vision and text towers), and the image embeddings go
   through classify's decode pool and its cache in <folder>/_embeddings;
3. the VLM: the images whose score falls in the band LO..HI (--vlm-band,
   0.35 0.65) go one by one to a vision model on the same endpoint, nearest
   to 0.5 first, at most --vlm-max of them (300); the model answers Y, N or I.

How they combine, per image: a caption YES accepts; a caption NO rejects,
unless --images is on and the image score is at or above HI, which sends
the image to the VLM instead; no caption or an INDETERMINATE one leaves the
decision to the image score when --images is on (at or above HI accepts, at
or below LO rejects, between goes to the VLM), else the image stays. The
VLM's Y accepts, N rejects, anything else stays. Only a final NO moves
anything: the image and every other file of its name go to
<folder>_rejected/<same relative path>. INDETERMINATE never moves a file.

The run's files go to <folder>_rejected/_judge/: verdicts.csv (every layer's
result per image), prompts.json (the condition, the prompts, the band),
moves.jsonl (the undo log) and, with --sheets, accepted.jpg, rejected.jpg
and band.jpg (vk_dataset's contact sheet). A rerun skips the images already
accepted or rejected in verdicts.csv (a rejected one that a --dry-run left in
place is moved now) and reuses the stored layer results of the others, so an
interrupted run or the rest of a capped VLM band continues where it stopped.
--rejudge starts afresh; --undo puts every moved file back.

The GPU is never shared: the captions run first, then the SigLIP pass, after
which the encoder is deleted and the CUDA cache emptied, then the VLM.

What the tool prints, and its --help, is Russian: the partner's agent reads
it (vk_dataset's rule D2).

Usage:
    python judge.py <folder> --condition "<text>" [--images] [--no-captions]
                    [--vlm-band LO HI] [--vlm-max N] [--positives P ...] [--negatives N ...]
                    [--endpoint URL] [--model NAME] [--batch 10] [--timeout S]
                    [--sheets] [--limit N] [--dry-run] [--rejudge]
    python judge.py <folder> --undo
    python judge.py --check-models
"""

from __future__ import annotations

import argparse
import base64
import codecs
import contextlib
import csv
import io
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".avif"}
CAPTION_EXT = ".txt"
OUT_SUFFIX = "_rejected"
RUN_DIRNAME = "_judge"
# Output folders the dataset tools make next to or inside a dataset; they hold no items.
EXCLUDES = {"_controls", "masks", "faces", "_backup", "_duplicates", "_prep", "_classify",
            "_embeddings", "_reframed", "_uncaptioned", "_unrelated", "_rejected", RUN_DIRNAME}
MARK = "_dataset_tools.json"    # marks a folder the dataset tools made next to a dataset

VERDICTS_NAME = "verdicts.csv"
PROMPTS_NAME = "prompts.json"
MOVES_NAME = "moves.jsonl"
SHEET_NAMES = {"YES": "accepted.jpg", "NO": "rejected.jpg", "BAND": "band.jpg"}
SHEET_MAX = 24

DEFAULT_ENDPOINT = "http://localhost:5001/v1/"
DEFAULT_BATCH = 10
DEFAULT_TIMEOUT = 600
DEFAULT_BAND = (0.35, 0.65)
DEFAULT_VLM_MAX = 300
VLM_SIDE = 1024                 # the long side of the image sent to the VLM
PATCHES = 576                   # classify's default patch budget, so its embedding cache is shared
DECODE_THREADS = 8
ENCODE_BATCH = 64

LETTERS = {"Y": "YES", "N": "NO", "I": "INDETERMINATE"}
TOKENS_PER_ITEM = 4
TOKENS_BASE = 16
RETRIES = 1
STOP_SEQUENCES = ["<|return|>", "<|im_end|>", "<end_of_turn>", "<|eot_id|>"]
THINKING_BLOCK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.S | re.I)

CSV_COLS = ["path", "caption_verdict", "image_score", "vlm_verdict", "final", "reason", "moved",
            "caption"]

# reasons in verdicts.csv
R_CAPTION_YES = "caption_yes"
R_CAPTION_NO = "caption_no"
R_IMAGE_HIGH = "image_high"
R_IMAGE_LOW = "image_low"
R_VLM_YES = "vlm_yes"
R_VLM_NO = "vlm_no"
R_VLM_OTHER = "vlm_other"
R_VLM_CAP = "vlm_cap"           # in the band, left out by --vlm-max: a rerun continues
R_UNDECIDED = "undecided"       # no caption verdict and no image layer
R_NO_SCORE = "no_score"         # the image could not be decoded

CAPTION_PROMPT = """You sort captions of photographs by a condition. For each caption you decide whether the photograph it describes meets this condition:

CONDITION: {condition}

The user message is a JSON array of objects {{"id": n, "caption": "..."}}. Answer with exactly one JSON array of single-letter strings, one per input object, in the order of the ids, and nothing else: "Y" for YES, "N" for NO, "I" for INDETERMINATE. For three captions the answer looks like:
["Y", "N", "I"]
Never write the words, the ids, the captions or any explanation.

Judge every caption on its own text only. Never let one caption influence another. The captions may be in any language.

YES: the caption says or clearly implies that the photograph meets the condition.
NO: the caption says or clearly implies that the photograph does not meet it.
INDETERMINATE: the caption gives no definite hint either way. A caption about something else that neither confirms nor rules out the condition is INDETERMINATE, not NO.

Answer with only the JSON array: no code fences, no comments, no text before or after it."""  # noqa: E501

PROMPTS_PROMPT = """You write text prompts for zero-shot image classification with SigLIP. The user message is a condition that a photograph should meet, in any language.

Write three short English captions (5 to 12 words each) of photographs that clearly meet the condition, and three of photographs of the same general kind that clearly do not meet it. Describe what is visible, concretely; never use negations such as "not" or "no".

Answer with exactly one JSON object and nothing else:
{"positive": ["...", "...", "..."], "negative": ["...", "...", "..."]}"""  # noqa: E501

VLM_PROMPT = """You check photographs against a condition. Look at the photograph (and its caption, when one is given) and answer with one letter only: Y when the photograph meets the condition, N when it does not, I when you cannot tell. No other text."""  # noqa: E501


class JudgeError(Exception):
    """A refusal with a readable Russian message (exit 2)."""


class EndpointDown(Exception):
    """The endpoint does not answer: nothing more can be done this run (exit 1)."""


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


def skipped(folder, name):
    return (name.startswith(".") or name.casefold() in EXCLUDES
            or os.path.isfile(os.path.join(folder, name, MARK)))


def walk(root):
    """os.walk that prunes the skipped folders, folders sorted for a stable order."""
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in sorted(dirs) if not skipped(folder, d)]
        yield folder, files


@dataclass
class Item:
    rel: str                    # path of the image relative to the folder, forward slashes
    folder: str                 # absolute folder of the image
    stem: str
    image: str                  # file name of the image
    caption: str = ""           # the caption text, one line; "" when none
    row: dict = field(default_factory=dict)   # its verdicts.csv row

    @property
    def path(self) -> str:
        return os.path.join(self.folder, self.image)


def find_items(root) -> list[Item]:
    """Every image under root (one per stem and folder), with its caption when a <stem>.txt exists."""
    items = []
    for folder, files in walk(root):
        by_stem: dict[str, list[str]] = {}
        for name in files:
            by_stem.setdefault(os.path.splitext(name)[0], []).append(name)
        for stem in sorted(by_stem, key=str.casefold):
            names = sorted(by_stem[stem])
            image = next((n for n in names if os.path.splitext(n)[1].lower() in IMAGE_EXTS), None)
            if image is None:
                continue
            txt = next((n for n in names if os.path.splitext(n)[1].lower() == CAPTION_EXT), None)
            caption = " ".join(read_text(os.path.join(folder, txt)).split()) if txt else ""
            rel_dir = os.path.relpath(folder, root)
            rel = image if rel_dir == "." else Path(rel_dir, image).as_posix()
            items.append(Item(rel, folder, stem, image, caption))
    return items


def move_item(root, dst_root, folder, stem):
    """Move every file of the stem in folder to the same place under dst_root. Returns the names
    moved, or None when a target exists (then nothing of the item moves)."""
    rel = os.path.relpath(folder, root)
    dst_folder = dst_root if rel == "." else os.path.join(dst_root, rel)
    names = sorted(n for n in os.listdir(folder) if os.path.splitext(n)[0] == stem
                   and os.path.isfile(os.path.join(folder, n)))
    for n in names:
        if os.path.lexists(os.path.join(dst_folder, n)):
            print("  ПРОПУЩЕНО, в папке отбраковки уже есть: {}".format(os.path.join(dst_folder, n)),
                  file=sys.stderr)
            return None
    os.makedirs(dst_folder, exist_ok=True)
    for n in names:
        shutil.move(os.path.join(folder, n), os.path.join(dst_folder, n))
    return names


def read_verdicts(path) -> dict[str, dict]:
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["path"]: {k: (r.get(k) or "") for k in CSV_COLS} for r in csv.DictReader(f)}


def write_verdicts(path, rows: dict[str, dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for rel in sorted(rows):
            w.writerow({k: rows[rel].get(k, "") for k in CSV_COLS})
    os.replace(tmp, path)


def write_json(path, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def mark_folder(dst_root) -> None:
    """The mark that keeps the dataset tools from scanning <folder>_rejected."""
    os.makedirs(dst_root, exist_ok=True)
    mark = os.path.join(dst_root, MARK)
    if not os.path.isfile(mark):
        write_json(mark, {"tool": "judge", "made": time.strftime("%Y-%m-%d %H:%M:%S")})


# --- the LLM --------------------------------------------------------------------------------

class Client:
    """An OpenAI-compatible chat endpoint (KoboldCpp, Ollama, LM Studio, llama.cpp)."""

    def __init__(self, endpoint, model=None, timeout=DEFAULT_TIMEOUT):
        self.base = endpoint.rstrip("/")
        self.model = model or ""
        self.timeout = timeout
        self.send_reasoning_effort = True

    def resolve_model(self):
        """Check that the endpoint answers; without --model, take the first model it lists."""
        try:
            with urllib.request.urlopen(f"{self.base}/models", timeout=15) as r:
                data = json.loads(r.read().decode("utf-8")).get("data") or []
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise EndpointDown("{} не отвечает ({}); запустите сервер модели или укажите "
                               "--endpoint".format(self.base, e)) from e
        if not self.model:
            if not data:
                raise EndpointDown("{}/models не называет ни одной модели".format(self.base))
            self.model = data[0]["id"]
        return self.model

    def release(self):
        """Ask an Ollama server to unload the model now (an empty prompt with keep_alive 0), so the
        SigLIP pass has the card; Ollama otherwise keeps it in memory for minutes. KoboldCpp
        emulates Ollama's /api/generate but cannot unload, so a server that answers KoboldCpp's
        /api/extra/version is left alone; a server without /api/generate refuses, which is
        harmless. Returns True only when Ollama confirms the unload."""
        if not self.model:
            return False
        root = re.sub(r"/v1$", "", self.base)
        try:
            with urllib.request.urlopen(f"{root}/api/extra/version", timeout=5):
                return False                        # KoboldCpp
        except (urllib.error.URLError, OSError, ValueError):
            pass
        body = {"model": self.model, "prompt": "", "keep_alive": 0, "stream": False,
                "options": {"num_predict": 1}}
        req = urllib.request.Request(f"{root}/api/generate", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8")).get("done_reason") == "unload"
        except (urllib.error.URLError, OSError, ValueError, AttributeError):
            return False

    def chat(self, system, user, max_tokens):
        """(text, "") or (None, reason); user is a string or a list of content parts."""
        payload = {"model": self.model,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   "max_tokens": max_tokens, "temperature": 0.0, "stop": STOP_SEQUENCES,
                   "chat_template_kwargs": {"enable_thinking": False}}
        if self.send_reasoning_effort:
            payload["reasoning_effort"] = "none"
        req = urllib.request.Request(f"{self.base}/chat/completions",
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", errors="replace")
            if e.code == 400 and "reasoning_effort" in text and self.send_reasoning_effort:
                self.send_reasoning_effort = False      # some endpoints reject it: once, then never again
                return self.chat(system, user, max_tokens)
            return None, f"HTTP {e.code}: {text[:200]}"
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise EndpointDown("нет ответа от {} ({})".format(self.base, e)) from e
        except ValueError as e:
            return None, f"ответ не JSON: {e}"
        msg = (body.get("choices") or [{}])[0].get("message") or {}
        return THINKING_BLOCK.sub("", msg.get("content") or "").strip(), ""


def extract_json(text, kind):
    """The first JSON value of the given type (list or dict) in the reply."""
    dec = json.JSONDecoder()
    opener = "[" if kind is list else "{"
    for m in re.finditer(re.escape(opener), text or ""):
        try:
            val, _ = dec.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(val, kind):
            return val
    return None


def parse_verdicts(reply, n):
    """[verdict] for the n captions in order, or (None, reason). A whole word is taken by its letter."""
    if not isinstance(reply, list) or not all(isinstance(v, str) for v in reply):
        return None, "в ответе нет массива строк"
    if len(reply) != n:
        return None, f"{len(reply)} ответ(ов) на {n} подпис(ей)"
    out = []
    for k, s in enumerate(reply, 1):
        word = s.strip().upper()
        letter = word[:1]
        if letter not in LETTERS or word not in (letter, LETTERS[letter]):
            return None, f"ответ {k}: {s!r}, а не Y, N или I"
        out.append(LETTERS[letter])
    return out, ""


def ask_captions(client, condition, captions):
    """Verdicts for a batch of captions, in order; None for an item left unanswered."""
    system = CAPTION_PROMPT.format(condition=condition)
    user = json.dumps([{"id": i + 1, "caption": c} for i, c in enumerate(captions)], ensure_ascii=False)
    reason = ""
    for _attempt in range(1 + RETRIES):
        text, reason = client.chat(system, user, TOKENS_PER_ITEM * len(captions) + TOKENS_BASE)
        if text is None:
            continue
        verdicts, reason = parse_verdicts(extract_json(text, list), len(captions))
        if verdicts is not None:
            return verdicts, ""
        reason = reason + ": " + " ".join(text.split())[:100]
    return [None] * len(captions), reason


def parse_prompts(text):
    """{"positive": [...], "negative": [...]} from the LLM's reply, up to three of each, or None."""
    obj = extract_json(text, dict)
    if not obj:
        return None
    out = {}
    for key in ("positive", "negative"):
        vals = obj.get(key)
        if not isinstance(vals, list):
            return None
        vals = [" ".join(str(v).split()) for v in vals if isinstance(v, str) and v.strip()][:3]
        if not vals:
            return None
        out[key] = vals
    return out


def make_prompts(client, condition):
    for _attempt in range(1 + RETRIES):
        text, _reason = client.chat(PROMPTS_PROMPT, condition, 400)
        got = parse_prompts(text) if text else None
        if got:
            return got
    raise JudgeError("Модель не написала подсказки для SigLIP (ответ не разобран). Задайте их "
                     "сами: --positives \"...\" и --negatives \"...\" (по три коротких английских "
                     "описания).")


def parse_letter(text):
    """Y, N or I from the VLM's reply; anything else counts as I."""
    word = re.sub(r"[^A-Za-z]", "", (text or "").strip().split()[0] if (text or "").strip() else "").upper()
    if word in LETTERS:
        return word
    for letter, name in LETTERS.items():
        if word == name:
            return letter
    return "I"


def image_data_url(path, side=VLM_SIDE):
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((side, side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def ask_vlm(client, condition, path, caption):
    text = "CONDITION: {}".format(condition)
    if caption:
        text += "\nCAPTION: {}".format(caption[:500])
    text += "\nAnswer Y, N or I."
    try:
        url = image_data_url(path)
    except Exception as e:  # noqa: BLE001
        return "I", "изображение не открывается: {}".format(e)
    reply, reason = client.chat(VLM_PROMPT, [{"type": "text", "text": text},
                                             {"type": "image_url", "image_url": {"url": url}}], 8)
    if reply is None:
        return "I", reason
    return parse_letter(reply), ""


# --- SigLIP 2 -------------------------------------------------------------------------------

def softmax_scores(img, pos, neg, scale):
    """Per image: the softmax probability mass of the positive prompts against all prompts.
    img (N, D), pos (P, D), neg (Q, D), all L2-normalised."""
    import numpy as np
    t = np.concatenate([pos, neg], axis=0)
    logits = scale * (img @ t.T)
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e[:, :len(pos)].sum(axis=1) / e.sum(axis=1)


class _Russian(io.TextIOBase):
    """Passes classify's progress lines through in Russian."""

    SUBS = [(re.compile(r"^(\s*)(\d+)/(\d+) embedded, (\d+) img/s"), r"\1\2/\3 изображений, \4 в секунду"),
            (re.compile(r"^SigLIP: (\d+) images, (\d+) cached, (\d+) to embed"),
             r"SigLIP 2: изображений \1, из кэша \2, считать \3"),
            (re.compile(r"^\s*(\d+) images could not be decoded"), r"  не открылось изображений: \1"),
            (re.compile(r"^\s*cache in (.*?) (.*)$"), r"  кэш \1: начат заново")]

    def __init__(self, out):
        self.out = out
        self.buf = ""

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            for rx, rep in self.SUBS:
                line = rx.sub(rep, line)
            self.out.write(line + "\n")
            self.out.flush()
        return len(s)


class SiglipScorer:
    """The full SigLIP 2 model (vision and text towers) from classify's snapshot, offline."""

    def __init__(self):
        if str(REPO / "classify") not in sys.path:
            sys.path.insert(0, str(REPO / "classify"))
        try:
            import classify as cl
        except ImportError as e:
            raise JudgeError("Не найден classify (папка classify рядом с judge): {}".format(e)) from e
        if not cl.models_present():
            raise JudgeError("Нет модели SigLIP 2 в {}. Запустите judge\\install.bat (или "
                             "classify\\install.bat).".format(cl.model_dir()))
        import torch
        import transformers
        from transformers import AutoTokenizer, Siglip2Model
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
        self.cl = cl
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        d = str(cl.model_dir())
        self.model = Siglip2Model.from_pretrained(d, dtype=self.dtype, local_files_only=True,
                                                  attn_implementation="sdpa").to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(d, local_files_only=True)
        self.scale = float(self.model.logit_scale.detach().float().exp())
        # classify's encoder around the vision tower already loaded: no second copy of the weights
        enc = object.__new__(cl.Encoder)
        enc.torch, enc.patches, enc.device, enc.dtype = torch, PATCHES, self.device, self.dtype
        enc.model = self.model.vision_model
        self.encoder = enc

    def text_features(self, prompts):
        torch = self.torch
        enc = self.tokenizer([p.lower() for p in prompts], padding="max_length", max_length=64,
                             truncation=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.get_text_features(**enc)
        feat = getattr(out, "pooler_output", out).float()
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return feat.cpu().numpy()

    def scores(self, root, items, positives, negatives):
        """(rel -> score, rel -> error, images embedded this run, seconds)."""
        import numpy as np
        cl = self.cl
        its = []
        for it in items:
            try:
                st = os.stat(it.path)
            except OSError:
                continue
            its.append(cl.ImageItem(Path(root).resolve(), it.rel, st.st_size, st.st_mtime_ns))
        cache = cl.EmbeddingCache(Path(root), cl.MODEL_REPO, PATCHES)
        with contextlib.redirect_stdout(_Russian(sys.stdout)):
            res = cl.embed_root(Path(root), its, self.encoder, cache, DECODE_THREADS, ENCODE_BATCH,
                                False, "SigLIP")
        pos = self.text_features(positives)
        neg = self.text_features(negatives)
        rels = [r for r in (it.rel for it in its) if r in res.vectors]
        out = {}
        if rels:
            img = np.stack([res.vectors[r] for r in rels]).astype(np.float32)
            img /= np.linalg.norm(img, axis=1, keepdims=True).clip(1e-8)
            for r, s in zip(rels, softmax_scores(img, pos, neg, self.scale), strict=True):
                out[r] = float(s)
        return out, res.failed, res.embedded, res.seconds

    def close(self):
        import gc
        self.encoder = None
        self.model = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def check_models() -> int:
    """For install.bat: classify's snapshot must hold both towers and the tokenizer."""
    if str(REPO / "classify") not in sys.path:
        sys.path.insert(0, str(REPO / "classify"))
    import classify as cl
    d = cl.model_dir()
    if not cl.models_present():
        print("Модели SigLIP 2 нет в {}; загружаю через classify (те же файлы, одна копия).".format(d))
        rc = cl.fetch_models()
        if rc:
            return rc
    from safetensors import safe_open
    with safe_open(str(d / "model.safetensors"), "pt") as f:
        heads = {k.split(".")[0] for k in f.keys()}
    missing = {"text_model", "vision_model", "logit_scale"} - heads
    if missing:
        print("[!] В {} нет частей модели: {}. Удалите model.safetensors и запустите "
              "classify\\install.bat заново.".format(d, ", ".join(sorted(missing))))
        return 1
    print("SigLIP 2 на месте: {} (визуальная и текстовая части, токенизатор).".format(d))
    return 0


# --- the decision ---------------------------------------------------------------------------

def decide(caption_verdict, score, images, lo, hi):
    """(final, reason) from the caption verdict and the image score; final is YES, NO,
    INDETERMINATE or BAND (to the VLM). caption_verdict: YES, NO, INDETERMINATE or ""."""
    if caption_verdict == "YES":
        return "YES", R_CAPTION_YES
    if caption_verdict == "NO":
        if images and score is not None and score >= hi:
            return "BAND", R_CAPTION_NO
        return "NO", R_CAPTION_NO
    if not images:
        return "INDETERMINATE", R_UNDECIDED
    if score is None:
        return "INDETERMINATE", R_NO_SCORE
    if score >= hi:
        return "YES", R_IMAGE_HIGH
    if score <= lo:
        return "NO", R_IMAGE_LOW
    return "BAND", ""


def band_order(band, vlm_max):
    """The band by rank, nearest to 0.5 first: (visited, left out by the cap)."""
    ranked = sorted(band, key=lambda it: (abs(_score(it.row) - 0.5), it.rel))
    n = max(0, int(vlm_max))
    return ranked[:n], ranked[n:]


def _score(row):
    try:
        return float(row.get("image_score"))
    except (TypeError, ValueError):
        return None


# --- the run --------------------------------------------------------------------------------

def run_dirs(folder):
    root = os.path.abspath(folder)
    dst_root = root.rstrip("\\/") + OUT_SUFFIX
    return root, dst_root, os.path.join(dst_root, RUN_DIRNAME)


def load_prompts(ns, client, run_dir, rejudge):
    """The prompts of this condition: --positives/--negatives, else the stored ones of the same
    condition, else written by the LLM once. Returns (prompts dict, source)."""
    stored = read_json_file(os.path.join(run_dir, PROMPTS_NAME))
    pos = list(ns.positives or [])
    neg = list(ns.negatives or [])
    source = "задано"
    if not (pos and neg):
        same = (not rejudge and stored and stored.get("condition") == ns.condition
                and stored.get("generated"))
        if same:
            gen = stored["generated"]
            source = "из прошлого запуска"
        else:
            client.resolve_model()
            gen = make_prompts(client, ns.condition)
            source = "написаны моделью {}".format(client.model)
        pos = pos or gen["positive"]
        neg = neg or gen["negative"]
    else:
        gen = {"positive": pos, "negative": neg}
    return {"positive": [ns.condition] + pos, "negative": neg, "generated": gen}, source


def read_json_file(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def render_sheets(root, dst_root, run_dir, rows, groups):
    """accepted.jpg, rejected.jpg and band.jpg through vk_dataset's sheet module."""
    if str(REPO / "vk_dataset") not in sys.path:
        sys.path.insert(0, str(REPO / "vk_dataset"))
    try:
        from vkds.sheet import SheetError, render_sheet
    except ImportError:
        print("[!] Листы не сделаны: нет vk_dataset\\vkds\\sheet.py рядом с judge.")
        return []
    written = []
    titles = {"YES": "Принято", "NO": "Отбраковано", "BAND": "Полоса VLM"}
    for key, items in groups.items():
        if not items:
            continue
        step = max(1, len(items) / SHEET_MAX)
        picks = [items[int(i * step)] for i in range(min(SHEET_MAX, len(items)))]
        tiles = []
        for it in picks:
            row = rows[it.rel]
            moved = row.get("moved") == "1"
            path = os.path.join(dst_root if moved else root, *it.rel.split("/"))
            sc = _score(row)
            label = "{}{} {}".format("" if sc is None else "{:.2f} ".format(sc),
                                     row.get("vlm_verdict") or (row.get("caption_verdict") or "")[:1] or "-",
                                     it.caption)
            tiles.append({"path": path, "caption": label})
        out = os.path.join(run_dir, SHEET_NAMES[key])
        try:
            render_sheet(tiles, Path(out), "{}: {} из {}".format(titles[key], len(picks), len(items)),
                         os.path.basename(root))
            written.append(out)
        except SheetError as e:
            print("[!] {}".format(e))
            return written
    return written


def run(ns, client=None, scorer_factory=SiglipScorer) -> int:
    root, dst_root, run_dir = run_dirs(ns.folder)
    if not os.path.isdir(root):
        raise JudgeError("Нет такой папки: {}".format(root))
    lo, hi = ns.vlm_band
    if not (0.0 <= lo <= hi <= 1.0):
        raise JudgeError("--vlm-band: нужно 0 <= LO <= HI <= 1.")
    if not ns.captions and not ns.images:
        raise JudgeError("Нечем судить: --no-captions без --images.")
    client = client or Client(ns.endpoint, ns.model, ns.timeout)
    verdicts_path = os.path.join(run_dir, VERDICTS_NAME)
    prev_prompts = read_json_file(os.path.join(run_dir, PROMPTS_NAME))
    if (prev_prompts and prev_prompts.get("condition") != ns.condition and not ns.rejudge
            and os.path.isfile(verdicts_path)):
        raise JudgeError("В {} уже есть вердикты для другого условия («{}»). Добавьте --rejudge, "
                         "чтобы судить заново, или --undo, чтобы вернуть файлы.".format(
                             run_dir, prev_prompts.get("condition")))
    rows = {} if ns.rejudge else read_verdicts(verdicts_path)
    # prompts.json records the condition of every run (and the prompts of an --images run)
    meta = dict(prev_prompts) if prev_prompts and prev_prompts.get("condition") == ns.condition else {}
    meta.update({"condition": ns.condition, "vlm_band": [lo, hi], "vlm_max": ns.vlm_max,
                 "endpoint": client.base})
    write_json(os.path.join(run_dir, PROMPTS_NAME), meta)
    items = find_items(root)
    if ns.limit:
        items = items[:ns.limit]
    t_start = time.time()
    print("Папка: {}\n  отбраковка -> {}\nУсловие: {}".format(root, dst_root, ns.condition))
    print("Изображений: {}, с подписью: {}; слои: {}{}{}".format(
        len(items), sum(1 for it in items if it.caption),
        " + ".join((["подписи"] if ns.captions else []) + (["SigLIP 2", "VLM"] if ns.images else [])), "",
        " (пробный прогон, ничего не перемещается)" if ns.dry_run else ""))

    todo, settled = [], []
    for it in items:
        prev = rows.get(it.rel)
        if prev and prev.get("final") in ("YES", "NO"):
            it.row = prev
            settled.append(it)
            continue
        it.row = dict(prev or {}, path=it.rel)
        it.row["caption"] = it.caption[:200]
        if not ns.captions:
            it.row["caption_verdict"] = ""
        todo.append(it)
    if settled:
        print("Уже решено в прошлых запусках: {} (пропущены; --rejudge — судить заново)".format(
            len(settled)))
    stats = {"caption_items": 0, "caption_s": 0.0, "siglip_n": 0, "siglip_s": 0.0,
             "vlm_n": 0, "vlm_s": 0.0, "band": 0, "cap_left": 0}
    rc = 0
    prompts = None
    try:
        # 1. captions
        if ns.captions:
            need = [it for it in todo if it.caption and it.row.get("caption_verdict") in ("", None, "UNANSWERED")]
            if need:
                client.resolve_model()
                print("Слой 1, подписи: {} подписей, модель {} на {}, по {} за запрос".format(
                    len(need), client.model, client.base, ns.batch))
                t0 = time.time()
                for start in range(0, len(need), ns.batch):
                    chunk = need[start:start + ns.batch]
                    verdicts, reason = ask_captions(client, ns.condition, [it.caption for it in chunk])
                    if reason:
                        print("  пачка с {}-й подписи без ответа: {}".format(start + 1, reason),
                              file=sys.stderr)
                    for it, v in zip(chunk, verdicts, strict=True):
                        it.row["caption_verdict"] = v or "UNANSWERED"
                    done = start + len(chunk)
                    if done == len(need) or (done // ns.batch) % 10 == 0:
                        print("  {}/{}, {:.2f} с на подпись".format(done, len(need),
                                                                    (time.time() - t0) / done), flush=True)
                stats["caption_items"] = len(need)
                stats["caption_s"] = time.time() - t0
                write_verdicts(verdicts_path, {**rows, **{it.rel: it.row for it in todo}})

        # 2. SigLIP 2 over every image still without a score
        if ns.images:
            prompts, source = load_prompts(ns, client, run_dir, ns.rejudge)
            meta.update({"positive": prompts["positive"], "negative": prompts["negative"],
                         "generated": prompts["generated"], "source": source,
                         "model": client.model or None})
            write_json(os.path.join(run_dir, PROMPTS_NAME), meta)
            print("Подсказки SigLIP ({}):\n  + {}\n  - {}".format(
                source, " | ".join(prompts["positive"]), " | ".join(prompts["negative"])))
            need = [it for it in todo if _score(it.row) is None]
            if need:
                if client.model and client.release():
                    print("Ollama выгрузил модель {} из видеопамяти на время SigLIP".format(client.model))
                scorer = scorer_factory()
                try:
                    scores, failed, embedded, secs = scorer.scores(
                        root, need, prompts["positive"], prompts["negative"])
                finally:
                    scorer.close()             # the encoder leaves the GPU before the VLM
                for it in need:
                    if it.rel in scores:
                        it.row["image_score"] = "{:.4f}".format(scores[it.rel])
                stats["siglip_n"], stats["siglip_s"] = embedded, secs
                if failed:
                    print("  не открылось: {}".format(len(failed)))
                write_verdicts(verdicts_path, {**rows, **{it.rel: it.row for it in todo}})

        # the decision, then 3. the VLM band
        band = []
        for it in todo:
            cv = it.row.get("caption_verdict") or ""
            cv = "INDETERMINATE" if cv == "UNANSWERED" else cv
            final, reason = decide(cv, _score(it.row), ns.images, lo, hi)
            if final == "BAND":
                vv = it.row.get("vlm_verdict") or ""
                if vv in LETTERS:                      # asked in an earlier run
                    final, reason = _vlm_final(vv)
                else:
                    band.append(it)
                    it.row["reason"] = reason
                    continue
            it.row["final"], it.row["reason"] = final, reason
        stats["band"] = len(band)
        if band:
            visit, left = band_order(band, ns.vlm_max)
            for it in left:
                it.row["final"], it.row["reason"] = "INDETERMINATE", R_VLM_CAP
            stats["cap_left"] = len(left)
            if visit:
                client.resolve_model()
                print("Слой 3, VLM: полоса {:.2f}..{:.2f} — {} изображений, смотрю {} (--vlm-max {}), "
                      "модель {}".format(lo, hi, len(band), len(visit), ns.vlm_max, client.model))
                t0 = time.time()
                for k, it in enumerate(visit, 1):
                    letter, why = ask_vlm(client, ns.condition, it.path, it.caption)
                    if why:
                        print("  {}: {}".format(it.rel, why), file=sys.stderr)
                    it.row["vlm_verdict"] = letter
                    it.row["final"], it.row["reason"] = _vlm_final(letter)
                    stats["vlm_n"] = k
                    if k % 10 == 0 or k == len(visit):
                        print("  {}/{}, {:.1f} с на изображение".format(k, len(visit),
                                                                       (time.time() - t0) / k), flush=True)
                        write_verdicts(verdicts_path, {**rows, **{i.rel: i.row for i in todo}})
                stats["vlm_s"] = time.time() - t0
    except EndpointDown as e:
        print("\nОстановлено: {}".format(e), file=sys.stderr)
        rc = 1
    except KeyboardInterrupt:
        print("\nОстановлено пользователем", file=sys.stderr)
        rc = 130

    for it in todo:
        rows[it.rel] = it.row
    for it in settled:
        rows[it.rel] = it.row
    mark_folder(dst_root)

    # the moves: only a final NO, never in a dry run
    moved_n = files_n = failed_n = 0
    if not ns.dry_run and rc == 0:
        with open(os.path.join(run_dir, MOVES_NAME), "a", encoding="utf-8") as log:
            for it in todo + settled:
                row = it.row
                if row.get("final") != "NO" or row.get("moved") in ("1", "undone"):
                    continue
                if not os.path.isfile(it.path):
                    continue
                names = move_item(root, dst_root, it.folder, it.stem)
                if names is None:
                    failed_n += 1
                    continue
                rel_dir = os.path.relpath(it.folder, root)
                log.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                                      "dir": "" if rel_dir == "." else Path(rel_dir).as_posix(),
                                      "files": names}, ensure_ascii=False) + "\n")
                row["moved"] = "1"
                moved_n += 1
                files_n += len(names)
    write_verdicts(verdicts_path, rows)

    sheets = []
    if ns.sheets:
        current = todo + settled
        groups = {"YES": [it for it in current if it.row.get("final") == "YES"],
                  "NO": [it for it in current if it.row.get("final") == "NO"],
                  "BAND": sorted([it for it in current if it.row.get("vlm_verdict") or
                                  it.row.get("reason") == R_VLM_CAP],
                                 key=lambda it: (abs((_score(it.row) or 0.5) - 0.5), it.rel))}
        sheets = render_sheets(root, dst_root, run_dir, rows, groups)

    report(ns, items, todo, settled, stats, moved_n, files_n, failed_n, root, dst_root, run_dir,
           sheets, time.time() - t_start)
    return rc


def _vlm_final(letter):
    if letter == "Y":
        return "YES", R_VLM_YES
    if letter == "N":
        return "NO", R_VLM_NO
    return "INDETERMINATE", R_VLM_OTHER


def report(ns, items, todo, settled, stats, moved_n, files_n, failed_n, root, dst_root, run_dir,
           sheets, seconds):
    cur = [it.row for it in todo + settled]
    cap = {v: sum(1 for r in cur if (r.get("caption_verdict") or "") == v)
           for v in ("YES", "NO", "INDETERMINATE", "UNANSWERED")}
    fin = {v: sum(1 for r in cur if r.get("final") == v) for v in ("YES", "NO", "INDETERMINATE")}
    vlm = {v: sum(1 for r in cur if r.get("vlm_verdict") == v) for v in ("Y", "N", "I")}
    reasons: dict[str, int] = {}
    for r in cur:
        reasons[r.get("reason") or "-"] = reasons.get(r.get("reason") or "-", 0) + 1
    print("\n==== ИТОГ ====")
    print("Изображений: {} (решено раньше: {})".format(len(items), len(settled)))
    if ns.captions:
        print("Подписи: ДА {}, НЕТ {}, НЕ ЯСНО {}{}; без подписи {}{}".format(
            cap["YES"], cap["NO"], cap["INDETERMINATE"],
            ", без ответа {}".format(cap["UNANSWERED"]) if cap["UNANSWERED"] else "",
            sum(1 for it in items if not it.caption),
            "; {:.2f} с на подпись".format(stats["caption_s"] / stats["caption_items"])
            if stats["caption_items"] else ""))
    if ns.images:
        scored = [r for r in cur if _score(r) is not None]
        print("SigLIP 2: с оценкой {}; посчитано сейчас {}{}".format(
            len(scored), stats["siglip_n"],
            ", {:.1f} изображений в секунду".format(stats["siglip_n"] / stats["siglip_s"])
            if stats["siglip_n"] and stats["siglip_s"] else ""))
        print("Полоса VLM {:.2f}..{:.2f}: {} изображений; VLM посмотрела {}{}; осталось за "
              "--vlm-max {}".format(ns.vlm_band[0], ns.vlm_band[1], stats["band"], stats["vlm_n"],
                                    ", {:.1f} с на изображение".format(stats["vlm_s"] / stats["vlm_n"])
                                    if stats["vlm_n"] else "", stats["cap_left"]))
        print("VLM всего: Y {}, N {}, I {}".format(vlm["Y"], vlm["N"], vlm["I"]))
    print("Итог: принято {}, отбраковано {}, не ясно {} (не ясно ничего не перемещает)".format(
        fin["YES"], fin["NO"], fin["INDETERMINATE"]))
    print("Причины: " + ", ".join("{} {}".format(k, v) for k, v in sorted(reasons.items())))
    if ns.dry_run:
        print("Пробный прогон: ничего не перемещено; без --dry-run отбракованные переместятся "
              "без новых вопросов к моделям.")
    else:
        print("Перемещено: {} изображений, {} файлов вместе с подписями{}".format(
            moved_n, files_n, "; НЕ перемещено (файл уже есть в отбраковке): {}".format(failed_n)
            if failed_n else ""))
    print("Отбраковка: {}".format(dst_root))
    print("Вердикты: {}".format(os.path.join(run_dir, VERDICTS_NAME)))
    if ns.images:
        print("Подсказки: {}".format(os.path.join(run_dir, PROMPTS_NAME)))
    for s in sheets:
        print("Лист: {}".format(s))
    print("Время: {:.0f} с".format(seconds))


def undo(folder) -> int:
    root, dst_root, run_dir = run_dirs(folder)
    log = os.path.join(run_dir, MOVES_NAME)
    if not os.path.isfile(log):
        raise JudgeError("Нечего возвращать: нет {}".format(log))
    with open(log, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    back = files = 0
    keep = []
    returned = set()
    for e in reversed(entries):
        rel_dir = e.get("dir") or ""
        src_dir = os.path.join(dst_root, *rel_dir.split("/")) if rel_dir else dst_root
        dst_dir = os.path.join(root, *rel_dir.split("/")) if rel_dir else root
        names = [n for n in e.get("files", []) if os.path.isfile(os.path.join(src_dir, n))]
        if any(os.path.lexists(os.path.join(dst_dir, n)) for n in names):
            print("  ПРОПУЩЕНО, на старом месте уже есть файл: {}".format(dst_dir), file=sys.stderr)
            keep.append(e)
            continue
        os.makedirs(dst_dir, exist_ok=True)
        for n in names:
            shutil.move(os.path.join(src_dir, n), os.path.join(dst_dir, n))
            if os.path.splitext(n)[1].lower() in IMAGE_EXTS:
                returned.add((Path(rel_dir) / n).as_posix() if rel_dir else n)
        back += 1
        files += len(names)
    with open(log, "w", encoding="utf-8") as f:
        for e in reversed(keep):
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    vpath = os.path.join(run_dir, VERDICTS_NAME)
    rows = read_verdicts(vpath)
    for rel in returned:
        if rel in rows:
            rows[rel]["moved"] = "undone"
    if rows:
        write_verdicts(vpath, rows)
    print("Возвращено: {} изображений, {} файлов в {}{}".format(
        back, files, root, "; не возвращено: {}".format(len(keep)) if keep else ""))
    print("Вердикты остались в {} (возвращённые помечены undone и больше не перемещаются; "
          "--rejudge судит заново).".format(vpath))
    return 0


class _RussianHelp(argparse.HelpFormatter):
    """Defaults in Russian, and only the ones that say something (no flags, no None)."""

    def _get_help_string(self, action):
        text, d = action.help or "", action.default
        if d is None or isinstance(d, bool) or d is argparse.SUPPRESS or "%(default)" in text:
            return text
        if isinstance(d, (list, tuple)):
            d = " ".join(str(x) for x in d)
        return "{} (по умолчанию: {})".format(text, d).replace("%", "%%")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="judge",
        description="Оставить в папке только изображения, которые подходят под условие, "
                    "заданное словами: подписи через локальную LLM, затем (--images) SigLIP 2 и "
                    "VLM на спорной полосе. Отбракованное переносится в <папка>_rejected. "
                    "Руководство: README.md.",
        formatter_class=_RussianHelp)
    ap.add_argument("folder", nargs="?", help="Папка датасета (обходится с подпапками)")
    ap.add_argument("--condition", help="Условие словами, на любом языке")
    ap.add_argument("--captions", action=argparse.BooleanOptionalAction, default=True,
                    help="Слой подписей через LLM (по умолчанию включён; --no-captions — выключить)")
    ap.add_argument("--images", action="store_true",
                    help="Слои изображений: SigLIP 2 по всем картинкам и VLM на спорной полосе")
    ap.add_argument("--vlm-band", nargs=2, type=float, default=list(DEFAULT_BAND), metavar=("LO", "HI"),
                    help="Полоса оценки SigLIP, которая идёт к VLM; >= HI — принять, <= LO — отбраковать")
    ap.add_argument("--vlm-max", type=int, default=DEFAULT_VLM_MAX, metavar="N",
                    help="Сколько изображений полосы показать VLM (ближайшие к 0.5 первыми); "
                         "остальные остаются «не ясно», повторный запуск продолжит")
    ap.add_argument("--positives", action="append", metavar="ТЕКСТ",
                    help="Своя положительная подсказка SigLIP (английский; можно несколько раз)")
    ap.add_argument("--negatives", action="append", metavar="ТЕКСТ",
                    help="Своя отрицательная подсказка SigLIP (английский; можно несколько раз)")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                    help="OpenAI-совместимый адрес модели (Ollama: http://127.0.0.1:11434/v1/)")
    ap.add_argument("--model", default=None,
                    help="Имя модели на сервере (по умолчанию первая из списка сервера)")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Подписей в одном запросе")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Сколько секунд ждать ответа")
    ap.add_argument("--sheets", action="store_true",
                    help="Листы accepted.jpg, rejected.jpg и band.jpg (до 24 миниатюр)")
    ap.add_argument("--limit", type=int, default=0, metavar="N",
                    help="Взять только первые N изображений (для пробы)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Все слои и вердикты, но ничего не перемещать")
    ap.add_argument("--rejudge", action="store_true",
                    help="Судить заново и то, что уже решено в verdicts.csv")
    ap.add_argument("--undo", action="store_true",
                    help="Вернуть все перемещённые файлы на место")
    ap.add_argument("--check-models", action="store_true",
                    help="Проверить модель SigLIP 2 (для install.bat)")
    return ap


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = build_parser()
    ns = ap.parse_args(argv)
    try:
        if ns.check_models:
            return check_models()
        if not ns.folder:
            ap.error("нужна папка")
        if ns.undo:
            return undo(ns.folder)
        if not ns.condition:
            ap.error("нужно --condition \"<условие>\"")
        if ns.batch < 1:
            ap.error("--batch: не меньше 1")
        return run(ns)
    except JudgeError as e:
        print("[!] {}".format(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
