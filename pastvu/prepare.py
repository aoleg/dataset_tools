"""The pixel side of a photo before the GPU: decode, the band check, the picture box,
the 4 MP rule, the border plan and the models' inputs. One function of plain values,
so it runs in worker processes.

Why processes: in threads this work holds the GIL for most of its time (Pillow,
NumPy, the heap allocator), and the GPU thread gives the GIL up and takes it back on
every torch call. On a run of many hours the GPU thread spent two thirds of its time
waiting for the GIL, and the rate fell from 10 to under 4 photos per second
(2026-11-06). In a worker process this work has a GIL of its own; the main process
only sends the downloaded bytes and gets the arrays back.

The fetch and storage threads stay: they download, read the storage and check the
store, then hand the bytes to a worker and wait for the answer with the GIL released.
"""

from __future__ import annotations

import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass

import process as pr


@dataclass(frozen=True)
class Config:
    borders: bool = True
    dark_borders: bool = True
    max_pixels: int = pr.DEFAULT_MAX_PIXELS   # 0: never downscale
    det_imgsz: int = 0                        # 0: no detector input
    patches: int = 0                          # 0: no encoder input


def run(data: bytes, head: dict, rows: int, cfg: Config) -> dict:
    """What Stream.prepare puts on the job: array, band, pic_box, head, head_orig,
    rows, downscale, borders, det_input, det_scale, naflex, and the times t_total
    (all of it) and t_prepare (the models' inputs)."""
    t0 = time.perf_counter()
    array = pr.decode(data, head)
    band = pr.verify_band(array, head, rows)
    pic_box = pr.picture_box(head, rows)
    out = {"band": band, "head": head, "head_orig": None, "rows": rows, "downscale": None, "borders": None}
    pw, ph = pic_box[2] - pic_box[0], pic_box[3] - pic_box[1]
    area = cfg.max_pixels * cfg.max_pixels
    if cfg.max_pixels and pw * ph > area:
        # The 4 MP rule: the picture (the band cut as exact rows) downscaled to that area.
        # From here on the photo has no band and is saved from pixels; the original head
        # stays for the record.
        array = pr.downscale(pr.crop_array(array, pic_box), area)
        h, w = array.shape[:2]
        out.update(downscale={"from": [pw, ph], "to": [w, h], "limit": cfg.max_pixels, "band_rows": rows,
                              "filter": pr.DOWNSCALE_FILTER},
                   head_orig=head, head=dict(head, stored=[w, h], orientation=1, mcu=None), rows=0)
        pic_box = [0, 0, w, h]
    picture = pr.crop_array(array, pic_box)
    if cfg.borders:
        out["borders"] = pr.borders_plan(picture, {"dark": cfg.dark_borders})
    t1 = time.perf_counter()
    det_input, det_scale, naflex = None, 1.0, None
    if cfg.det_imgsz:
        import watermark as wm
        det_input, det_scale = wm.detector_input(picture, cfg.det_imgsz)
    if cfg.patches:
        naflex = pr.naflex_resize(picture, cfg.patches)
    t2 = time.perf_counter()
    out.update(array=array, pic_box=pic_box, det_input=det_input, det_scale=det_scale, naflex=naflex,
               t_prepare=t2 - t1, t_total=t2 - t0)
    return out


def _init_worker() -> None:
    """Ctrl+C belongs to the main process: it finishes the photos in flight, which needs
    the workers alive. OpenCV's own threads would only compete with the other workers."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:  # noqa: BLE001
        pass


class Pool:
    """Worker processes for run(); a crashed worker breaks the executor, so it is made
    again once and the photo tried again, then the photo fails."""

    def __init__(self, workers: int):
        import threading
        self.workers = max(1, workers)
        self.lock = threading.Lock()
        self.ex = self._new()
        self.closed = False
        self.fn = run                            # what the workers run (a test swaps in one that crashes)

    def _new(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(self.workers, initializer=_init_worker)

    def run(self, data: bytes, head: dict, rows: int, cfg: Config) -> dict:
        for attempt in (1, 2):
            ex = self.ex
            try:
                return ex.submit(self.fn, data, head, rows, cfg).result()
            except BrokenProcessPool:
                with self.lock:
                    if self.closed:
                        raise
                    if self.ex is ex:            # the first thread to see it makes the new one
                        ex.shutdown(wait=False, cancel_futures=True)
                        self.ex = self._new()
                if attempt == 2:
                    raise pr.Rejected("prepare: the worker process crashed", failed=True)
        raise AssertionError("unreachable")

    def close(self, wait: bool = True) -> None:
        with self.lock:
            self.closed = True
            self.ex.shutdown(wait=wait, cancel_futures=True)


def default_workers() -> int:
    """Half the logical processors, at most 8: the downloads, the storage and the GPU
    stage need the rest, and 8 prepare about 40 photos per second."""
    return max(1, min(8, (os.cpu_count() or 4) // 2))
