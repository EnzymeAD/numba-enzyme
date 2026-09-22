"""
Vendor the pre-built LLVM/Enzyme binaries into the wheel.

The derivative pipeline shells out to `clang`/`llvm-link`/`opt` from
LLVM 15 and loads the standalone Enzyme plugin, none of which are in
the source tree. `toolchain.py` looks for them under
`src/numba_enzyme/_vendor/` first, so a wheel built without that
directory installs cleanly and then fails at the first differentiation
unless the machine happens to carry LLVM 15 and a hand-built plugin on
`PATH`.

Two things populate `_vendor/`:

- Under cibuildwheel, `packaging/cibw_before_build.sh` stages binaries
  that `cibw_before_all.sh` built from source. That path is unchanged.
- Otherwise -- a `uv add`/`pip install` straight from a git checkout,
  which never runs cibuildwheel -- this hook downloads the released
  manylinux wheel from PyPI and reuses the binaries it already carries.
  That is the same trick `packaging/bootstrap_dev_toolchain.py` uses for
  editable checkouts, and it is what makes the git URL installable on
  its own. Set `NUMBA_ENZYME_VENDOR_FROM_PYPI=0` to suppress it and
  build a toolchain-less wheel deliberately.

A static `force-include` in pyproject.toml can't express any of this --
hatchling requires force-included files to exist unconditionally -- so
the entries are added here, once `_vendor/` is known to be present.

The released wheel keeps its shared libraries in a top-level
`numba_enzyme.libs/` written by `auditwheel repair`. They are staged
into `_vendor/lib/` rather than kept at that top-level name because
`$ORIGIN/../lib` is the first entry in the binaries' own RPATH, so the
layout resolves without shipping a directory that sits outside the
package.

The wheel is tagged `py3-none-manylinux_2_38_x86_64`. Everything under
`_vendor/` is a standalone executable or shared object rather than a
CPython extension, and the package has no extension modules at all, so
nothing here is ABI-specific and a CPython tag would only narrow which
interpreters can install it. An explicit platform tag also keeps
cibuildwheel happy: it refuses to hand an apparently-pure-Python wheel
(`py3-none-any`) to `auditwheel repair`, rejecting the build outright
("Build failed because a pure Python wheel was generated") before
auditwheel can rewrite the tag from the real contents. Discovered via a
real cibuildwheel run; local `auditwheel repair` testing never
exercised that gate, since it reads the declared tag rather than the
contents.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# The release supplying the binaries. Pinned rather than tracking the
# project's own version: the LLVM 15 + Enzyme toolchain changes far more
# rarely than the Python source, and a fork whose version has moved past
# the last upload must still resolve to a real file on PyPI.
_VENDOR_WHEEL_VERSION = "0.1.3"
_PYPI_RELEASE_JSON = "https://pypi.org/pypi/numba-enzyme/{version}/json"
# manylinux rather than a bare `linux_x86_64`, which PyPI rejects outright.
# 2.38 is the highest GLIBC_ symbol version any binary under _vendor/
# references (measured over all 19 ELF files in the built wheel), which is
# what auditwheel would derive from the contents anyway.
_WHEEL_TAG = "py3-none-manylinux_2_38_x86_64"

# Everything toolchain.py resolves, and so everything the staged tree
# has to hold for the wheel to be worth shipping.
_TOOLS = ("clang", "llvm-link", "opt", "ld.lld")
_PLUGIN = "enzyme/LLVMEnzyme-15.so"


def _released_wheel_url(version: str) -> str:
    url = _PYPI_RELEASE_JSON.format(version=version)
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = json.load(response)
    candidates = [
        entry["url"]
        for entry in payload.get("urls", ())
        if entry.get("packagetype") == "bdist_wheel"
        and entry.get("filename", "").endswith("_x86_64.whl")
        and "linux" in entry["filename"]
    ]
    if not candidates:
        raise RuntimeError(
            f"numba-enzyme {version} has no linux x86_64 wheel on PyPI to take the "
            "LLVM/Enzyme toolchain from"
        )
    # Any of them will do: the binaries are identical across the CPython
    # tags, which differ only in the pure-Python payload.
    return min(candidates)


def _make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _stage_from_pypi(vendor_dir: Path, version: str) -> None:
    url = _released_wheel_url(version)
    with tempfile.TemporaryDirectory(prefix="numba-enzyme-vendor-") as temporary:
        scratch = Path(temporary)
        archive = scratch / "released.whl"
        with (
            urllib.request.urlopen(url, timeout=600) as response,
            archive.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle)

        extracted = scratch / "wheel"
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(
                extracted,
                members=[
                    name
                    for name in bundle.namelist()
                    if name.startswith(("numba_enzyme/_vendor/", "numba_enzyme.libs/"))
                ],
            )

        source = extracted / "numba_enzyme" / "_vendor"
        if not source.is_dir():
            raise RuntimeError(
                f"{url} carries no numba_enzyme/_vendor/, so it cannot supply the "
                "LLVM/Enzyme toolchain"
            )

        staged = scratch / "_vendor"
        shutil.copytree(source, staged)
        libraries = extracted / "numba_enzyme.libs"
        if libraries.is_dir():
            shutil.copytree(libraries, staged / "lib")

        # zipfile drops the mode bits, and the tools are useless without
        # them. The shared objects are only dlopen()ed, but auditwheel
        # ships them executable and matching that costs nothing.
        for name in _TOOLS:
            tool = staged / "bin" / name
            if tool.is_file():
                _make_executable(tool)
        for shared_object in staged.rglob("*.so*"):
            if shared_object.is_file():
                _make_executable(shared_object)

        vendor_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged), str(vendor_dir))


def _missing(vendor_dir: Path) -> list[str]:
    required = [f"bin/{name}" for name in _TOOLS] + [_PLUGIN, "crt"]
    return [name for name in required if not (vendor_dir / name).exists()]


class VendorBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        vendor_dir = Path(self.root) / "src" / "numba_enzyme" / "_vendor"

        if not vendor_dir.is_dir():
            if os.environ.get("NUMBA_ENZYME_VENDOR_FROM_PYPI", "1") == "0":
                return
            wheel_version = os.environ.get(
                "NUMBA_ENZYME_VENDOR_WHEEL_VERSION", _VENDOR_WHEEL_VERSION
            )
            self.app.display_info(
                f"staging the LLVM/Enzyme toolchain from numba-enzyme "
                f"{wheel_version} on PyPI"
            )
            _stage_from_pypi(vendor_dir, wheel_version)

        missing = _missing(vendor_dir)
        if missing:
            raise RuntimeError(
                f"{vendor_dir} is missing the LLVM/Enzyme toolchain:\n  - "
                + "\n  - ".join(missing)
            )

        force_include = build_data.setdefault("force_include", {})
        src_root = Path(self.root) / "src"
        for path in sorted(vendor_dir.rglob("*")):
            if path.is_file():
                force_include[str(path)] = str(path.relative_to(src_root))

        build_data["pure_python"] = False
        build_data["tag"] = _WHEEL_TAG
