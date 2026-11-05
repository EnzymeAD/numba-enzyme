"""
Fetch the `llvm-extract` binary the borrowed toolchain wheel lacks.

`hatch_build.py` and `bootstrap_dev_toolchain.py` take the LLVM 15 + Enzyme
binaries from upstream's released numba-enzyme wheel, which predates the
`llvm-extract` step in the build pipeline and so does not ship it. This
takes it from LLVM's own 15.0.6 release instead. The archive is streamed and
only the one binary is kept, so nothing close to its full size touches disk.
The binary needs nothing beyond glibc 2.15 and libstdc++, well inside the
wheel's manylinux_2_38 tag.
"""

from __future__ import annotations

import hashlib
import stat
import tarfile
import urllib.request
from pathlib import Path

_RELEASE_URL = (
    "https://github.com/llvm/llvm-project/releases/download/llvmorg-15.0.6/"
    "clang+llvm-15.0.6-x86_64-linux-gnu-ubuntu-18.04.tar.xz"
)
_MEMBER_SUFFIX = "/bin/llvm-extract"
_SHA256 = "76523bd7a4a06ffbf45247de1200f1634716e0afb3116362f258679dba4af4e5"


def fetch_llvm_extract(destination: Path, url: str = _RELEASE_URL) -> None:
    """
    Download `llvm-extract` from the LLVM release and write it to a path.

    Parameters
    ----------
    destination : pathlib.Path
        Where to write the executable.
    url : str, optional
        Release archive to take it from.

    Raises
    ------
    RuntimeError
        If the archive has no `llvm-extract`, or its checksum doesn't match.
    """
    with (
        urllib.request.urlopen(url, timeout=600) as response,
        tarfile.open(fileobj=response, mode="r|xz") as archive,
    ):
        for member in archive:
            if member.isfile() and member.name.endswith(_MEMBER_SUFFIX):
                data = archive.extractfile(member).read()
                break
        else:
            raise RuntimeError(f"{url} has no {_MEMBER_SUFFIX.lstrip('/')}")

    digest = hashlib.sha256(data).hexdigest()
    if digest != _SHA256:
        raise RuntimeError(
            f"llvm-extract from {url} has SHA-256 {digest}, expected {_SHA256}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    mode = destination.stat().st_mode
    destination.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
