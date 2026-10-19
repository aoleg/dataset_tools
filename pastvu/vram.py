"""GPU memory of the whole device, as nvidia-smi reports it, through NVML.

On Windows a process's own CUDA query (torch.cuda.mem_get_info) does not see
the memory other processes hold, and the driver does not fail when the card is
full: it moves memory to system RAM, and everything slows down. Watching the
device-wide figure is the only way to see a local VLM and this tool's models
crowd each other out.
"""

from __future__ import annotations

import threading
import time

GB = 2 ** 30
LOW_FREE_GB = 1.0          # below this the driver is about to spill to system RAM


class VramMonitor:
    """Samples the device's memory every `interval` seconds in a thread;
    `text()` is a short status, `peak` the highest use seen."""

    def __init__(self, index: int = 0, interval: float = 2.0):
        self.ok = False
        self.used = self.total = self.peak = 0.0
        self.interval = interval
        self._stop = threading.Event()
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            self.ok = True
            self.sample()
        except Exception:  # noqa: BLE001 - no NVML, no monitor
            self.ok = False
        self._thread = None

    def sample(self) -> None:
        if not self.ok:
            return
        try:
            m = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
        except Exception:  # noqa: BLE001
            return
        self.used, self.total = m.used / GB, m.total / GB
        self.peak = max(self.peak, self.used)

    def start(self) -> "VramMonitor":
        if self.ok and self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="vram")
            self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()

    def stop(self) -> None:
        self._stop.set()

    @property
    def low(self) -> bool:
        return self.ok and (self.total - self.used) < LOW_FREE_GB

    def text(self) -> str:
        if not self.ok:
            return ""
        warn = "  LOW: spilling to system RAM slows everything" if self.low else ""
        return f"VRAM {self.used:.1f}/{self.total:.1f} GB{warn}"

    def report(self) -> str:
        if not self.ok:
            return "VRAM: not available (no NVML)"
        return f"VRAM: {self.used:.1f} of {self.total:.1f} GB in use now, peak {self.peak:.1f} GB"
