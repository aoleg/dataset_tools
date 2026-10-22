"""autocaption: captions written by a vision-language model, from each image
and its existing caption (the facts line), through an OpenAI-compatible chat
endpoint (KoboldCpp, llama.cpp server, LM Studio, or a cloud API).

The client is the API core of TagGUI's auto-captioner: one chat request per
image with the image inline as base64 JPEG, the end-of-turn stop sequences,
reasoning switched off, thinking blocks stripped from the answer, a retry at
half the size when the endpoint rejects the request, and a stop when three
photos in a row fail.

The VLM caption replaces the caption file. The text the file held before goes
into the store first (caption_log), so `autocaption <target> --undo` puts the
last run back; the caption a photo had before its first VLM caption is kept as
its original, which a later --recaption sends to the VLM again.
"""

from __future__ import annotations

import base64
import hashlib
import io
import queue
import re
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageOps

import sidecars
from conf import CaptionConf

# End-of-turn tokens to stop on; the OpenAI API allows at most four (TagGUI's list).
STOP_SEQUENCES = ["<|return|>", "<|im_end|>", "<end_of_turn>", "<|eot_id|>"]
THINKING_DELIMITERS = (("<|channel>", "<channel|>"), ("<think>", "</think>"), ("<thinking>", "</thinking>"),
                       ("<reasoning>", "</reasoning>"))
HARMONY_FINAL_CHANNEL = "<|channel|>final<|message|>"
HARMONY_CHANNEL = "<|channel|>"
RESIDUAL_CONTROL_TOKEN = re.compile(r"<\|[\w-]+\|>|<\|[\w-]+>|<[\w-]+\|>|</?(?:start|end)_of_turn>")
MAX_RETRY_COUNT = 2            # a rejected request is sent again at half the size, this many times
MAX_CONSECUTIVE_FAILS = 3      # then the endpoint itself is the likely problem: stop
JPEG_QUALITY = 92


class CaptionFailed(Exception):
    """This photo could not be captioned (the next one may be)."""


class EndpointDown(Exception):
    """The endpoint is unreachable or does not answer: stop the run."""


class _Rejected(Exception):
    """The endpoint returned an error, which a smaller image may avoid."""


def strip_thinking(text: str) -> str:
    """The answer without a chain-of-thought block or leftover control tokens (TagGUI)."""
    if HARMONY_FINAL_CHANNEL in text:
        text = text.split(HARMONY_FINAL_CHANNEL)[-1]
    elif HARMONY_CHANNEL in text:
        text = ""
    for opening, closing in THINKING_DELIMITERS:
        if opening not in text:
            continue
        text = "".join(part.split(opening)[0] if opening in part else part for part in text.split(closing))
    return RESIDUAL_CONTROL_TOKEN.sub("", text)


def one_line(text: str) -> str:
    """Trainers read the whole caption file as one prompt, line breaks included: one paragraph."""
    return re.sub(r"\s+", " ", text).strip()


def fill_prompt(template: str, tags: str, cid: int | None = None, folder: str = "") -> str:
    """TagGUI's template variables: {tags} (the existing caption), {name}, {folder}; \\{ and \\} stay literal."""
    values = {"tags": tags, "name": str(cid or ""), "folder": folder, "directory": folder}

    def rep(m):
        key = m.group(0)[1:-1].lower()
        return values.get(key, m.group(0))

    text = re.sub(r"(?<!\\){[^{}]+(?<!\\)}", rep, template)
    return re.sub(r"\\([{}])", r"\1", text)


def image_data_url(image, side: int) -> str:
    """An upright RGB image (PIL or array) as a JPEG data URL, longest side at most `side`."""
    im = Image.fromarray(np.ascontiguousarray(image)) if isinstance(image, np.ndarray) else image
    im = im.convert("RGB")
    if max(im.size) > side:
        im = im.copy()
        im.thumbnail((side, side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def load_upright(path: Path) -> Image.Image:
    with Image.open(path) as im:
        im.load()
        return ImageOps.exif_transpose(im).convert("RGB")


class VLMClient:
    def __init__(self, cc: CaptionConf, client: httpx.Client | None = None):
        self.cc = cc
        self.base = cc.api_url.rstrip("/")
        for f in (cc.system_prompt, cc.prompt):
            if not Path(f).is_file():
                raise SystemExit(f"prompt file {f} does not exist; set [caption] system_prompt and prompt in the conf")
        self.system = Path(cc.system_prompt).read_text(encoding="utf-8").strip()
        self.prompt = Path(cc.prompt).read_text(encoding="utf-8").strip()
        self.http = client or httpx.Client(timeout=httpx.Timeout(connect=15.0, read=cc.timeout, write=60.0, pool=30.0))
        self.model = cc.model
        self.send_reasoning_effort = not cc.reasoning
        self.prompt_sha = hashlib.sha256((self.system + "\0" + self.prompt).encode("utf-8")).hexdigest()[:16]
        self.stats = {"requests": 0, "retries": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.cc.api_key}"} if self.cc.api_key else {}

    def check(self) -> str:
        """The model the endpoint serves; EndpointDown when it does not answer."""
        try:
            r = self.http.get(f"{self.base}/models", headers=self.headers(), timeout=15)
            r.raise_for_status()
            data = r.json().get("data") or []
        except (httpx.HTTPError, ValueError) as e:
            raise EndpointDown(f"the caption endpoint {self.base} does not answer ({e}); start the VLM server or set "
                               f"[caption] api_url") from e
        served = data[0]["id"] if data else "unknown"
        self.model = self.cc.model or served
        return served

    def payload(self, text: str, url: str) -> dict:
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": [{"type": "text", "text": text},
                                                     {"type": "image_url", "image_url": {"url": url}}]})
        p = {"model": self.model or "local-model", "messages": messages, "stop": STOP_SEQUENCES,
             "max_tokens": self.cc.max_tokens, "temperature": self.cc.temperature, "top_p": self.cc.top_p,
             "chat_template_kwargs": {"enable_thinking": bool(self.cc.reasoning)}}
        if self.send_reasoning_effort:
            p["reasoning_effort"] = "none"
        return p

    def _send(self, payload: dict) -> dict:
        try:
            r = self.http.post(f"{self.base}/chat/completions", json=payload, headers=self.headers())
        except httpx.TimeoutException as e:
            raise EndpointDown(f"no answer from {self.base} in {self.cc.timeout} s") from e
        except httpx.TransportError as e:
            raise EndpointDown(f"{self.base} is unreachable ({e})") from e
        if r.status_code == 400 and "reasoning_effort" in payload and "reasoning_effort" in r.text:
            self.send_reasoning_effort = False          # some endpoints reject it: once, then never again
            payload = dict(payload)
            del payload["reasoning_effort"]
            return self._send(payload)
        if r.status_code >= 400:
            raise _Rejected(f"HTTP {r.status_code}: {r.text[:200]}")
        try:
            return r.json()
        except ValueError as e:
            raise _Rejected(f"not JSON: {r.text[:200]}") from e

    def caption(self, image, tags: str, cid: int | None = None, folder: str = "") -> tuple[str, dict]:
        """One caption. -> (text, facts: seconds, retries, tokens, size sent)."""
        text = fill_prompt(self.prompt, tags, cid, folder)
        side = self.cc.image_side
        t0 = time.time()
        last = ""
        for attempt in range(MAX_RETRY_COUNT + 1):
            url = image_data_url(image, side)
            self.stats["requests"] += 1
            try:
                d = self._send(self.payload(text, url))
            except _Rejected as e:
                last = str(e)
                self.stats["retries"] += 1
                side = max(128, side // 2)
                continue
            choice = (d.get("choices") or [{}])[0]
            out = one_line(strip_thinking((choice.get("message") or {}).get("content") or ""))
            usage = d.get("usage") or {}
            self.stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            self.stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            if not out:
                why = " (the token limit was reached while reasoning)" if choice.get("finish_reason") == "length" else ""
                raise CaptionFailed(f"the model returned an empty caption{why}")
            return out, {"seconds": time.time() - t0, "retries": attempt, "side": side,
                         "tokens": int(usage.get("completion_tokens") or 0),
                         "finish": choice.get("finish_reason")}
        raise CaptionFailed(f"rejected {MAX_RETRY_COUNT + 1} times, last: {last}")


# --- one photo -------------------------------------------------------------------------------

@dataclass
class Target:
    cid: int
    path: Path                 # the image
    image: object = None       # the upright pixels, when the stream has them in memory
    tags: str | None = None    # the existing caption, when the caller knows it
    scope: str = ""            # the scope it belongs to (the stream's caption run per scope)
    base: Path | None = None   # the base folder, to find the photo through the store


def current_path(store, t: Target) -> Path:
    """Where the photo is now: a category that passed the split moves its
    images into 001 while a caption job may run, so the store decides."""
    st = store.status(t.cid)
    if st and st.get("path") and t.base is not None:
        return t.base / st["path"]
    return t.path


def caption_photo(store, client: VLMClient, run: int, t: Target) -> dict:
    """Caption one saved photo: the text sent as {tags} is the photo's original
    caption when it was captioned before (--recaption), else its caption file;
    the file's old text goes into the store, then the file gets the VLM caption."""
    t.path = current_path(store, t)
    txt = t.path.with_suffix(".txt")
    before = txt.read_text(encoding="utf-8") if txt.is_file() else ""
    prev = store.caption(t.cid)
    tags = t.tags if t.tags is not None else (prev["original"] if prev else before)
    if not tags:
        rec = store.record(t.cid) or {}
        tags = sidecars.caption(rec.get("record") or {})
    image = t.image if t.image is not None else load_upright(t.path)
    text, facts = client.caption(image, tags, t.cid, t.path.parent.name)
    txt = current_path(store, t).with_suffix(".txt")          # the VLM took seconds: look again
    txt.write_text(text, encoding="utf-8")
    store.save_caption(run, t.cid, before, text, client.model, client.prompt_sha)
    return facts


# --- the run over saved photos ----------------------------------------------------------------

def select(store, base: Path, scope: str, prefix: str, recaption: bool, retries: int) -> tuple[list[Target], dict]:
    """The photos of a scope under a folder prefix (relative to the base) that need a caption."""
    rows = [r for r in store.done_rows(scope) if r[1].startswith(prefix)]
    done = store.captioned_cids(scope)
    attempts = store.caption_attempts()
    counts = {"listed": len(rows), "captioned before": 0, "gave up": 0}
    out = []
    for cid, path, category, sha in rows:
        if cid in done and not recaption:
            counts["captioned before"] += 1
            continue
        if attempts.get(cid, 0) >= retries:
            counts["gave up"] += 1
            continue
        out.append(Target(cid, base / path, base=base))
    return out, counts


class _Interrupt:
    """First Ctrl+C: finish the photo being captioned, then stop. Second: stop at once."""

    def __init__(self):
        self.stop = False
        self._old = None

    def __enter__(self):
        def handler(signum, frame):
            if self.stop:
                raise KeyboardInterrupt
            self.stop = True
        try:
            self._old = signal.signal(signal.SIGINT, handler)
        except ValueError:           # not the main thread
            self._old = None
        return self

    def __exit__(self, *exc):
        if self._old is not None:
            signal.signal(signal.SIGINT, self._old)


def report(stats: dict, total: int, elapsed: float, client: VLMClient, vram, final: bool = False) -> str:
    n = stats["captioned"]
    rate = n / elapsed * 3600 if elapsed > 0 else 0.0
    mean = stats["seconds"] / n if n else 0.0
    toks = client.stats["completion_tokens"] / stats["seconds"] if stats["seconds"] else 0.0
    lines = [f"{'Finished: ' if final else ''}{n:,} of {total:,} photos captioned in {_hms(elapsed)} "
             f"({rate:,.0f} per hour, {mean:.1f} s per photo, {toks:.0f} tokens/s generated)"]
    if stats["failed"] or stats["missing"]:
        lines.append(f"  failed: {stats['failed']:,}; image missing: {stats['missing']:,}")
    if stats["retries"]:
        lines.append(f"  sent again at half size: {stats['retries']:,}")
    if vram is not None:
        lines.append("  " + vram.report())
    return "\n".join(lines)


def _hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min {s:02d} s" if m else f"{s} s"


def run(store, base: Path, scope: str, targets: list[Target], client: VLMClient, display, run_id: int,
        report_every: int = 100, vram=None) -> dict:
    """Caption the targets one after another, with the live display, report
    blocks and a graceful Ctrl+C. The store is updated per photo, so a stopped
    run continues with the rest when it is started again."""
    stats = {"captioned": 0, "failed": 0, "missing": 0, "retries": 0, "seconds": 0.0}
    fails_in_row = 0
    t_start = time.time()
    stopped = None
    with _Interrupt() as intr:
        for i, t in enumerate(targets, 1):
            if intr.stop:
                stopped = "Ctrl+C"
                break
            t.path = current_path(store, t)
            rel = t.path.relative_to(base).as_posix() if t.path.is_relative_to(base) else t.path.name
            if not t.path.is_file():
                stats["missing"] += 1
                display.advance(saved=False)
                continue
            display.set(f"{t.cid}: captioning {rel}")
            try:
                facts = caption_photo(store, client, run_id, t)
            except CaptionFailed as e:
                store.caption_failed(t.cid, str(e))
                stats["failed"] += 1
                fails_in_row += 1
                display.print(f"[caption failed] {t.cid}: {e}")
                display.line({"cid": t.cid, "state": "failed", "reason": str(e)})
                display.advance(saved=False)
                if fails_in_row >= MAX_CONSECUTIVE_FAILS:
                    stopped = f"{MAX_CONSECUTIVE_FAILS} photos in a row failed; check the endpoint"
                    break
                continue
            except EndpointDown as e:
                stopped = str(e)
                break
            fails_in_row = 0
            stats["captioned"] += 1
            if stats["captioned"] % 20 == 0:
                store.end_caption_run(run_id, stats["captioned"], stats["failed"])
            stats["seconds"] += facts["seconds"]
            stats["retries"] += facts["retries"]
            display.set(f"{t.cid}: captioned in {facts['seconds']:.1f} s")
            display.line({"cid": t.cid, "state": "captioned", "seconds": round(facts["seconds"], 2),
                          "tokens": facts["tokens"]})
            display.advance(saved=True)
            if report_every and stats["captioned"] % report_every == 0:
                display.print(report(stats, len(targets), time.time() - t_start, client, vram))
    store.end_caption_run(run_id, stats["captioned"], stats["failed"])
    display.print(report(stats, len(targets), time.time() - t_start, client, vram, final=True))
    if stopped:
        display.print(f"Stopped: {stopped}. Run the same command again to continue.")
    stats["stopped"] = stopped
    return stats


def undo(store, base: Path, scope: str) -> dict:
    """Put back the caption files of the last autocaption run of a scope. A file
    edited since that run keeps the edit."""
    last = store.last_caption_run(scope)
    if not last:
        raise SystemExit(f"{scope}: no autocaption run to undo")
    restored = kept = missing = 0
    for cid, before, after in store.caption_log_rows(last["id"]):
        st = store.status(cid)
        if not st or not st.get("path"):
            missing += 1
            continue
        txt = (base / st["path"]).with_suffix(".txt")
        now = txt.read_text(encoding="utf-8") if txt.is_file() else None
        if now is not None and now != after:
            kept += 1
            continue
        txt.write_text(before, encoding="utf-8")
        store.unsave_caption(last["id"], cid)
        restored += 1
    store.undo_caption_run(last["id"])
    return {"run": last["id"], "target": last["target"], "restored": restored, "kept": kept, "missing": missing}


# --- the worker the download stream feeds ------------------------------------------------------

class StreamCaptioner:
    """Captions saved photos in a thread while the stream downloads. The queue
    is small, so the writers wait when the VLM is behind: the downloads slow
    to the VLM's pace instead of running ahead."""

    def __init__(self, store, client: VLMClient, stats, display_ref, stop_event: threading.Event, depth: int = 4):
        self.store, self.client = store, client
        self.runs: dict[str, int] = {}             # one caption run per scope, so undo works per scope
        self.per_run: dict[int, list[int]] = {}
        self.stats, self.display_ref, self.stop_event = stats, display_ref, stop_event
        self.q: queue.Queue = queue.Queue(maxsize=max(1, depth))
        self.fails_in_row = 0
        self.captioned = self.failed = 0
        self.thread = threading.Thread(target=self._work, daemon=True, name="caption")
        self.thread.start()

    def put(self, target: Target) -> None:
        self.q.put(target)

    def run_for(self, scope: str) -> int:
        if scope not in self.runs:
            rid = self.store.start_caption_run(scope, scope, self.client.model, self.client.prompt_sha)
            self.runs[scope] = rid
            self.per_run[rid] = [0, 0]
        return self.runs[scope]

    def close(self) -> None:
        self.q.put(None)
        self.thread.join()
        for rid, (ok, bad) in self.per_run.items():
            self.store.end_caption_run(rid, ok, bad)

    def _work(self) -> None:
        while True:
            t = self.q.get()
            if t is None:
                break
            d = self.display_ref()
            if self.stop_event.is_set() and self.fails_in_row >= MAX_CONSECUTIVE_FAILS:
                continue                       # the endpoint is down: leave the rest for a later autocaption
            rid = self.run_for(t.scope)
            try:
                if d:
                    d.set(f"{t.cid}: captioning")
                facts = caption_photo(self.store, self.client, rid, t)
                self.stats.time("caption", facts["seconds"])
                self.stats.add(captioned=1)
                self.captioned += 1
                self.per_run[rid][0] += 1
                self.fails_in_row = 0
            except (CaptionFailed, EndpointDown) as e:
                self.store.caption_failed(t.cid, str(e))
                self.failed += 1
                self.per_run[rid][1] += 1
                self.fails_in_row += 1
                if d:
                    d.print(f"[caption failed] {t.cid}: {e}")
                if isinstance(e, EndpointDown) or self.fails_in_row >= MAX_CONSECUTIVE_FAILS:
                    self.fails_in_row = MAX_CONSECUTIVE_FAILS
                    if d:
                        d.print("Captioning stopped: the endpoint fails. The stream stops; run autocaption later.")
                    self.stop_event.set()
            except Exception as e:  # noqa: BLE001 - one photo never stops the stream
                self.store.caption_failed(t.cid, f"{type(e).__name__}: {e}")
                self.failed += 1
                self.per_run[rid][1] += 1
