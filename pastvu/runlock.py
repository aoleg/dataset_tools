"""Is a process still running? For the store's run lock: a lock is held by a
process number and its start time, so a number that Windows gave to a new
process after the old one ended does not count as the holder.

os.kill(pid, 0) is no test on Windows: there it terminates the process.
"""

from __future__ import annotations

import os
import sys


def process_start(pid: int) -> float | None:
    """The process's start time (seconds since the epoch), or None when it does not run."""
    if not pid:
        return None
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        PROCESS_QUERY_LIMITED_INFORMATION, STILL_ACTIVE = 0x1000, 259
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            return None
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)) or code.value != STILL_ACTIVE:
                return None
            ft = [wintypes.FILETIME() for _ in range(4)]
            if not k32.GetProcessTimes(h, *[ctypes.byref(f) for f in ft]):
                return 0.0
            t = (ft[0].dwHighDateTime << 32) | ft[0].dwLowDateTime
            return t / 1e7 - 11644473600.0
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(int(pid), 0)
    except OSError:
        return None
    try:
        return os.stat(f"/proc/{int(pid)}").st_ctime
    except OSError:
        return 0.0


def is_running(pid, start) -> bool:
    now = process_start(pid) if pid else None
    if now is None:
        return False
    if start is None or now == 0.0:
        return True
    return abs(now - float(start)) < 2.0
