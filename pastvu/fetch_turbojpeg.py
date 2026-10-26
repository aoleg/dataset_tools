"""Fetch turbojpeg.dll into bin/ for PyTurboJPEG (the in-memory lossless JPEG
crop). PyTurboJPEG does not bundle the library and libjpeg-turbo publishes its
Windows builds only as installers, so this takes the DLL out of conda-forge's
libjpeg-turbo package for win-64: a zip that holds a zstd-compressed tar. No
conda is involved. The package is pinned by version and sha256. The download
goes through httpx, which trusts certifi's bundle: the Python that made the venv
(often a Miniconda one) trusts the Windows certificate store through urllib,
and a store that has not been updated fails the handshake with "certificate has
expired".
"""

from __future__ import annotations

import hashlib
import io
import sys
import tarfile
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


def download() -> bytes | None:
    import httpx
    print(f"  downloading {URL}")
    try:
        r = httpx.get(URL, timeout=120, follow_redirects=True)
        r.raise_for_status()
    except httpx.HTTPError as e:
        print(f"[ERROR] download failed: {type(e).__name__}: {e}")
        print(f"        Download {PACKAGE} by hand from {URL} and put it next to this")
        print(f"        script as {HERE / PACKAGE}, then run install.bat again.")
        return None
    return r.content


def main() -> int:
    if DLL.is_file() and check():
        print(f"  {DLL} is in place")
        return 0
    import zstandard
    local = HERE / PACKAGE
    if local.is_file():
        print(f"  using {local}")
        data = local.read_bytes()
    else:
        data = download()
        if data is None:
            return 1
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
