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

STAGES = ["fetch", "detect", "paint", "classify", "encode", "write", "caption"]


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
    scope across runs, written with each photo."""

    def __init__(self):
        self.lock = threading.Lock()
        self.c = Counter()
        self.seconds = Counter()
        self.calls = Counter()

    def add(self, **kw) -> None:
        with self.lock:
            for k, v in kw.items():
                self.c[k] += v

    def time(self, stage: str, seconds: float) -> None:
        with self.lock:
            self.seconds[stage] += seconds
            self.calls[stage] += 1


def report_block(totals: dict, session: Stats, counts: dict, elapsed: float, final: bool = False) -> str:
    """The report printed every N images: totals of the scope from the store,
    rate and pace from this process."""
    c = {k[2:]: v for k, v in totals.items() if k.startswith("c.")}
    cats = {k[4:]: int(v) for k, v in totals.items() if k.startswith("cat.") and v}
    skipped = {k[8:]: int(v) for k, v in c.items() if k.startswith("skipped.") and v}
    failed = {k[7:]: int(v) for k, v in c.items() if k.startswith("failed.") and v}
    rate = session.c["done"] / elapsed if elapsed > 0 else 0.0
    lines = [f"{'Finished: ' if final else ''}{counts['done']:,} images saved out of {counts['total']:,} "
             f"({session.c['done']:,} in {hms(elapsed)} this run, {rate:.2f}/s)"]
    if counts.get("elsewhere"):
        lines.append(f"  already saved under another scope: {counts['elsewhere']:,} (not downloaded again)")
    if skipped:
        lines.append("  skipped: " + ", ".join(f"{v:,} {k}" for k, v in sorted(skipped.items(), key=lambda x: -x[1])))
    if failed:
        lines.append("  failed: " + ", ".join(f"{v:,} {k}" for k, v in sorted(failed.items(), key=lambda x: -x[1])))
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
