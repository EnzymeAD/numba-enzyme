#!/usr/bin/env python3
"""Bootstrap the binary LLVM/Enzyme toolchain for an editable checkout."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DESTINATION = _REPO_ROOT / ".dev-toolchain"
_DEFAULT_WHEEL_VERSION = "0.1.3"
# Upstream, deliberately: this is where the prebuilt LLVM/Enzyme binaries
# come from, and this fork (published as numba-enzyme-cuda) has none of
# its own to download. Same reasoning as hatch_build.py.
_DISTRIBUTION = "numba-enzyme"
_TOOLS = ("clang", "llvm-link", "opt", "ld.lld")


def _vendor_dir(destination: Path) -> Path:
    return destination / "wheel" / "numba_enzyme" / "_vendor"


def _validate(destination: Path) -> list[str]:
    vendor_dir = _vendor_dir(destination)
    missing = []
    for name in _TOOLS:
        path = vendor_dir / "bin" / name
        if not path.is_file():
            missing.append(str(path))
        elif not os.access(path, os.X_OK):
            missing.append(f"{path} (not executable)")
    plugin = vendor_dir / "enzyme" / "LLVMEnzyme-15.so"
    if not plugin.is_file() or not os.access(plugin, os.R_OK):
        missing.append(str(plugin))
    crt_dir = vendor_dir / "crt"
    if not crt_dir.is_dir():
        missing.append(str(crt_dir))
    libraries = destination / "wheel" / "numba_enzyme.libs"
    if not libraries.is_dir():
        missing.append(str(libraries))
    return missing


def _install_with_uv(destination: Path, wheel_version: str) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "uv is required to bootstrap the development toolchain; "
            "install it from https://docs.astral.sh/uv/"
        )
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--target",
            str(destination / "wheel"),
            "--no-deps",
            "--only-binary",
            ":all:",
            f"{_DISTRIBUTION}=={wheel_version}",
        ],
        check=True,
    )


def _installed_version(destination: Path) -> str | None:
    try:
        metadata = json.loads((destination / "bootstrap.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return metadata.get("wheel_version")


def bootstrap(destination: Path, wheel_version: str, force: bool = False) -> bool:
    """Install and validate the released binary toolchain.

    Returns ``True`` when a new toolchain was installed and ``False`` when an
    existing valid installation was reused.
    """

    destination = destination.resolve()
    unsafe_destinations = {Path(destination.anchor), Path.home(), _REPO_ROOT}
    if destination in unsafe_destinations:
        raise ValueError(f"refusing to use unsafe toolchain destination: {destination}")
    if (
        not force
        and destination.is_dir()
        and not _validate(destination)
        and _installed_version(destination) == wheel_version
    ):
        return False

    if sys.platform != "linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise RuntimeError(
            "the bundled development toolchain supports Linux x86_64 only"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary) / destination.name
        staged.mkdir()
        _install_with_uv(staged, wheel_version)
        missing = _validate(staged)
        if missing:
            raise RuntimeError(
                "downloaded wheel does not contain a complete toolchain:\n  - "
                + "\n  - ".join(missing)
            )
        (staged / "bootstrap.json").write_text(
            json.dumps(
                {
                    "distribution": _DISTRIBUTION,
                    "wheel_version": wheel_version,
                    "platform": platform.platform(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        if destination.exists():
            shutil.rmtree(destination)
        staged.rename(destination)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=_DEFAULT_DESTINATION,
        help="installation directory (default: repository .dev-toolchain)",
    )
    parser.add_argument(
        "--wheel-version",
        default=_DEFAULT_WHEEL_VERSION,
        help="released numba-enzyme wheel supplying the binaries",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace an existing valid toolchain"
    )
    args = parser.parse_args()

    installed = bootstrap(args.destination, args.wheel_version, args.force)
    action = "installed" if installed else "already ready"
    print(f"Development toolchain {action}: {args.destination.resolve()}")
    print("No PATH or NUMBA_ENZYME_PLUGIN_PATH changes are required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
