"""The live display of a run: a status line with the current activity above a
progress bar, both redrawn in place, and report blocks printed above them
every N images, which stay in the scrollback. --quiet prints the report
blocks only; --json-progress prints one JSON line per image instead.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import Counter

from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, \
    TimeRemainingColumn
from rich.text import Text

STAGES = ["fetch", "pull", "prepare", "detect", "paint", "classify", "colour", "encode", "write", "caption"]


def hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} h {m:02d} min"
    if m:
        return f"{m} min {s:02d} s"
    return f"{s} s"


class Stats:
    """Pace and counts of this process; the store keeps the totals of the
    scope across runs, written with each photo. Beside the run's totals (c,
    seconds, calls) the same counts are kept since the last report block
    (interval()), so a block says what happened lately, not on average."""

    def __init__(self):
        self.lock = threading.Lock()
        self.c = Counter()
        self.seconds = Counter()
        self.calls = Counter()
        self.i = Counter()                 # counts since the last interval()
        self.iseconds = Counter()
        self.icalls = Counter()
        self.latencies: list[float] = []   # download seconds since the last interval()
        self.t_interval = time.time()

    def add(self, **kw) -> None:
        with self.lock:
            for k, v in kw.items():
                self.c[k] += v
                self.i[k] += v

    def time(self, stage: str, seconds: float) -> None:
        with self.lock:
            self.seconds[stage] += seconds
            self.calls[stage] += 1
            self.iseconds[stage] += seconds
            self.icalls[stage] += 1

    def latency(self, seconds: float) -> None:
        with self.lock:
            self.latencies.append(seconds)

    def interval(self) -> dict:
        """The counts, stage seconds and download latencies since the last call, and the
        wall seconds they cover; then they start again."""
        with self.lock:
            now = time.time()
            out = {"wall": now - self.t_interval, "c": dict(self.i), "seconds": dict(self.iseconds),
                   "calls": dict(self.icalls), "latencies": sorted(self.latencies)}
            self.i.clear()
            self.iseconds.clear()
            self.icalls.clear()
            self.latencies = []
            self.t_interval = now
        return out


def _pct(part: float, whole: float) -> str:
    return f"{100 * part / whole:.0f} %" if whole > 0 else "-"


def interval_block(snap: dict, live: dict) -> tuple[str, dict]:
    """The report lines about the photos since the last block, from Stats.interval()
    and the stream's live view (threads in a request and the oldest's age, storage
    threads waiting, writers). -> (text, the same numbers as a dict for a JSON line)."""
    c, sec, calls, lat, wall = snap["c"], snap["seconds"], snap["calls"], snap["latencies"], snap["wall"]
    done = int(c.get("done", 0) + c.get("stored", 0))
    stage, w_in, w_out = sec.get("stage", 0.0), sec.get("wait_input", 0.0), sec.get("wait_writers", 0.0)
    n = len(lat)
    med = lat[n // 2] if n else 0.0
    p90 = lat[min(n - 1, int(n * 0.9))] if n else 0.0
    over = sum(1 for x in lat if x > 30)
    writers = max(1, int(live.get("writers", 1)))
    busy = sum(sec.get(k, 0.0) for k in ("encode", "write", "colour")) / writers
    moves, move_s = int(c.get("chunk_moves", 0)), sec.get("chunk_move", 0.0)
    d = {"wall": round(wall, 1), "photos": done, "rate": round(done / wall, 2) if wall > 0 else 0.0,
         "main_stage": round(stage, 1), "main_wait_input": round(w_in, 1), "main_wait_writers": round(w_out, 1),
         "downloads": n, "download_median": round(med, 2), "download_p90": round(p90, 2), "download_over_30s": over,
         "retries": int(c.get("retries", 0)), "in_request": int(live.get("in_request", 0)),
         "oldest_request": round(float(live.get("oldest_request", 0.0)), 1),
         "stored_prepared": int(calls.get("pull", 0)), "store_read": round(sec.get("store_read", 0.0), 2),
         "storage_wait_put": round(sec.get("storage_wait_put", 0.0), 1), "storage_threads": int(live.get("storage_threads", 0)),
         "writers_busy": round(busy, 1), "chunk_moves": moves, "chunk_move_seconds": round(move_s, 1),
         "export_links": int(c.get("exported", 0))}
    lines = [f"  last {done:,} photo{'s' if done != 1 else ''} ({hms(wall)}): {d['rate']:.2f}/s; "
             f"main thread: stage {_pct(stage, wall)}, "
             f"waiting for input {_pct(w_in, wall)}, for the writers {_pct(w_out, wall)}"]
    if n or live.get("in_request"):
        lines.append(f"    downloads: {n:,} done, median {med:.1f} s, 90 % under {p90:.1f} s, {over} over 30 s, "
                     f"{d['retries']} retries; {d['in_request']} threads in a request, oldest {d['oldest_request']:.0f} s")
    if calls.get("pull"):
        st = int(live.get("storage_threads", 0)) or 1
        lines.append(f"    storage: {d['stored_prepared']:,} prepared, store reads {sec.get('store_read', 0.0) / max(1, calls.get('pull', 1)) * 1000:.0f} ms "
                     f"each, waiting for a free slot {_pct(sec.get('storage_wait_put', 0.0), wall * st)}")
    lines.append(f"    writers: busy {_pct(busy, wall)}"
                 + (f", {moves} chunk move{'s' if moves != 1 else ''} ({move_s:.0f} s)" if moves else "")
                 + (f", {d['export_links']:,} export links" if d["export_links"] else ""))
    return "\n".join(lines), d


def _reason_lines(c: dict) -> list[str]:
    """The skipped and failed lines of a report, by reason."""
    skipped = {k[8:]: int(v) for k, v in c.items() if k.startswith("skipped.") and v}
    failed = {k[7:]: int(v) for k, v in c.items() if k.startswith("failed.") and v}
    lines = []
    if skipped:
        lines.append("  skipped: " + ", ".join(f"{v:,} {k}" for k, v in sorted(skipped.items(), key=lambda x: -x[1])))
    if failed:
        lines.append("  failed: " + ", ".join(f"{v:,} {k}" for k, v in sorted(failed.items(), key=lambda x: -x[1])))
    return lines


def raw_report_block(totals: dict, session: Stats, counts: dict, elapsed: float, final: bool = False) -> str:
    """The report of a raw run (run --raw): photos put into the storage, nothing processed."""
    c = {k[2:]: v for k, v in totals.items() if k.startswith("c.")}
    rate = session.c["stored"] / elapsed if elapsed > 0 else 0.0
    mb = session.c["raw_bytes"] / 2**20
    lines = [f"{'Finished: ' if final else ''}{counts.get('stored', 0):,} photos in the storage, {counts['done']:,} saved "
             f"out of {counts['total']:,} ({session.c['stored']:,} stored in {hms(elapsed)} this run, {rate:.2f}/s, "
             f"{mb:,.0f} MB)"]
    if counts.get("elsewhere"):
        lines.append(f"  already saved under another scope: {counts['elsewhere']:,} (not downloaded again)")
    lines.extend(_reason_lines(c))
    with session.lock:
        pace = [f"{s} {session.seconds[s] / session.calls[s]:.2f} s" for s in ("fetch", "write") if session.calls[s]]
    if pace:
        lines.append("  pace per photo: " + ", ".join(pace))
    return "\n".join(lines)


def report_block(totals: dict, session: Stats, counts: dict, elapsed: float, final: bool = False) -> str:
    """The report printed every N images: totals of the scope from the store,
    rate and pace from this process."""
    c = {k[2:]: v for k, v in totals.items() if k.startswith("c.")}
    cats = {k[4:]: int(v) for k, v in totals.items() if k.startswith("cat.") and v}
    rate = session.c["done"] / elapsed if elapsed > 0 else 0.0
    lines = [f"{'Finished: ' if final else ''}{counts['done']:,} images saved out of {counts['total']:,} "
             f"({session.c['done']:,} in {hms(elapsed)} this run, {rate:.2f}/s)"]
    if counts.get("elsewhere"):
        lines.append(f"  already saved under another scope: {counts['elsewhere']:,} (not downloaded again)")
    lines.extend(_reason_lines(c))
    lines.append(f"  band cut: {int(c.get('band', 0)):,}; band colour unverified: {int(c.get('band_unverified', 0)):,}; "
                 f"borders cut: {int(c.get('borders', 0)):,} images (frames {int(c.get('cut.frame', 0)):,}, "
                 f"sides {int(c.get('cut.border', 0)):,}, lines {int(c.get('cut.line', 0)):,}, "
                 f"banners {int(c.get('cut.banner', 0)):,}); border review: {int(c.get('borders_review', 0)):,}")
    lines.append(f"  watermarks: {int(c.get('wm_images', 0)):,} images, {int(c.get('wm_boxes', 0)):,} boxes painted, "
                 f"{int(c.get('wm_trim', 0)):,} trimmed; {int(c.get('wm_below', 0)):,} images with detections below "
                 f"the paint threshold (recorded, not painted)")
    lines.append(f"  saved as: {int(c.get('write.verbatim', 0)):,} untouched, {int(c.get('write.lossless', 0)):,} "
                 f"lossless JPEG crops, {int(c.get('write.exact', 0)) + int(c.get('write.png', 0)):,} exact crops, "
                 f"{int(c.get('write.array', 0)):,} re-encoded after painting")
    bw, col = int(c.get("mode.bw", 0)), int(c.get("mode.color", 0))
    if bw or col:
        lines.append(f"  black-and-white {bw:,}, colour {col:,} ({col / max(1, bw + col):.1%} colour; "
                     f"{int(c.get('mode.probe', 0)):,} borderline photos decided by the probe)")
    if cats:
        unsure = sum(v for k, v in cats.items() if k.startswith("_unsure"))
        top = Counter({k: v for k, v in cats.items() if not k.startswith("_unsure")})
        parts = [f"{k} {v:,}" for k, v in sorted(top.items())]
        if unsure:
            parts.append(f"_unsure {unsure:,} ({unsure / max(1, sum(cats.values())):.1%})")
        lines.append("  categories: " + ", ".join(parts))
    with session.lock:
        pace = [f"{s} {session.seconds[s] / session.calls[s]:.2f} s" for s in STAGES if session.calls[s]]
    if pace:
        lines.append("  pace per image: " + ", ".join(pace))
    return "\n".join(lines)


class Display:
    """Status line + progress bar (rich), or plain report lines, or JSON lines."""

    def __init__(self, total: int, completed: int, mode: str = "live", vram=None):
        self.mode = mode                   # live, quiet or json
        self.status = "starting"
        self.lock = threading.Lock()
        self.console = Console(stderr=False, highlight=False)
        self.progress = Progress(TextColumn("[bold]{task.description}"), BarColumn(), MofNCompleteColumn(),
                                 TaskProgressColumn(), TextColumn("{task.fields[rate]}"), TimeElapsedColumn(),
                                 TextColumn("left"), TimeRemainingColumn(), console=self.console)
        self.task = self.progress.add_task("pastvu", total=total, completed=completed, rate="")
        self.live = None
        self.vram = vram
        self.t0 = time.time()
        self.session_done = 0

    def __enter__(self):
        if self.mode == "live":
            self.live = Live(self._render(), console=self.console, refresh_per_second=8, transient=False,
                             get_renderable=self._render)
            self.live.__enter__()
        return self

    def __exit__(self, *exc):
        if self.live:
            self.live.__exit__(*exc)
            self.live = None

    def _render(self):
        with self.lock:
            status = self.status
        if self.vram is not None and self.vram.ok:
            status = f"{status}   |   {self.vram.text()}"
        return Group(Text(status, style="cyan", overflow="ellipsis", no_wrap=True), self.progress)

    def set(self, status: str) -> None:
        with self.lock:
            self.status = status

    def advance(self, saved: bool) -> None:
        if saved:
            self.session_done += 1
        el = time.time() - self.t0
        rate = f"{self.session_done / el:.2f}/s" if el > 0 else ""
        self.progress.update(self.task, advance=1, rate=rate)

    def print(self, text: str) -> None:
        if self.mode == "json":
            sys.stdout.write(json.dumps({"report": text}, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        elif self.live:
            self.live.console.print(text, highlight=False, markup=False)
        else:
            self.console.print(text, highlight=False, markup=False)

    def line(self, obj: dict) -> None:
        if self.mode == "json":
            sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            sys.stdout.flush()
