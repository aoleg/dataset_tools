"""Fetch turbojpeg.dll into bin/ for PyTurboJPEG (the in-memory lossless JPEG
crop). PyTurboJPEG does not bundle the library and libjpeg-turbo publishes its
Windows builds only as installers, so this takes the DLL out of conda-forge's
libjpeg-turbo package for win-64: a zip that holds a zstd-compressed tar. No
conda is involved. The package is pinned by version and sha256.
"""

from __future__ import annotations

import hashlib
import io
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DLL = HERE / "bin" / "turbojpeg.dll"
PACKAGE = "libjpeg-turbo-3.2.0-hfd05255_1.conda"
URL = f"https://api.anaconda.org/download/conda-forge/libjpeg-turbo/3.2.0/win-64/{PACKAGE}"
PACKAGE_SHA256 = "df78ab4c0eecb3dd9331898f96baeed8e5ca1c363346332517abeb0b614b9a53"
MEMBER = "Library/bin/turbojpeg.dll"


def check() -> bool:
    try:
        from turbojpeg import TurboJPEG
        TurboJPEG(str(DLL))
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  turbojpeg.dll does not load: {type(e).__name__}: {e}")
        return False


def main() -> int:
    if DLL.is_file() and check():
        print(f"  {DLL} is in place")
        return 0
    import zstandard
    print(f"  downloading {URL}")
    with urllib.request.urlopen(URL, timeout=120) as r:
        data = r.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != PACKAGE_SHA256:
        print(f"[ERROR] {PACKAGE}: sha256 {digest}, expected {PACKAGE_SHA256}")
        return 1
    z = zipfile.ZipFile(io.BytesIO(data))
    inner = next(n for n in z.namelist() if n.startswith("pkg-") and n.endswith(".tar.zst"))
    tar_bytes = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(z.read(inner))).read()
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as t:
        dll = t.extractfile(MEMBER).read()
    DLL.parent.mkdir(parents=True, exist_ok=True)
    DLL.write_bytes(dll)
    print(f"  {DLL} written ({len(dll):,} bytes)")
    return 0 if check() else 1


if __name__ == "__main__":
    sys.exit(main())
