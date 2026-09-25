"""
Locate the required external build tools.

Resolves the `clang`/`llvm-link`/`opt`/`llvm-extract` binaries
and the standalone Enzyme LLVM pass plugin. Prioritise
the `_vendor/` directory shipped alongside this module,
and falls back to the system-installed, ``PATH``-resolved
tools used during development.

See Also
--------
numba_enzyme.build.build : Uses these tools to compile
    a differentiated kernel.

Examples
--------
>>> from numba_enzyme.toolchain import get_toolchain
>>> toolchain = get_toolchain()  # doctest: +SKIP
>>> toolchain.clang.name  # doctest: +SKIP
'clang-15'
"""

import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PLUGIN_PATH = _REPO_ROOT / "enzyme-build" / "Enzyme" / "LLVMEnzyme-15.so"
_PLUGIN_PATH_ENV_VAR = "NUMBA_ENZYME_PLUGIN_PATH"
_VENDOR_DIR = Path(__file__).resolve().parent / "_vendor"


class ToolchainError(RuntimeError):
    """
    Raise when a required build tool cannot be located or used.

    This covers both a missing tool (not found on ``PATH``, or the
    Enzyme plugin not found at its configured path) and a tool that
    exists on disk but is not executable or readable.

    See Also
    --------
    get_toolchain : Raises this error when tool resolution fails.

    Examples
    --------
    >>> from numba_enzyme.toolchain import ToolchainError, get_toolchain
    >>> try:
    ...     get_toolchain()
    ... except ToolchainError as exc:
    ...     print(exc)  # doctest: +SKIP
    """


@dataclass(frozen=True)
class Toolchain:
    """
    Resolved and validated paths to the build tools.

    Attributes
    ----------
    clang : pathlib.Path
        Path to the `clang-15` executable.
    llvm_link : pathlib.Path
        Path to the `llvm-link-15` executable.
    opt : pathlib.Path
        Path to the ``opt-15`` executable.
    llvm_extract : pathlib.Path
        Path to the `llvm-extract-15` executable.
    enzyme_plugin : pathlib.Path
        Path to the standalone Enzyme LLVM pass plugin (`LLVMEnzyme-15.so`).

    See Also
    --------
    get_toolchain : Builds and validates a `Toolchain` instance.

    Examples
    --------
    >>> from numba_enzyme.toolchain import get_toolchain
    >>> get_toolchain().opt.name  # doctest: +SKIP
    'opt-15'
    """

    clang: Path
    llvm_link: Path
    opt: Path
    llvm_extract: Path
    enzyme_plugin: Path


def _which(name: str) -> Path | None:
    """
    Resolve an executable's absolute path in ``PATH``.

    Parameters
    ----------
    name : str
        Name of the executable to look up, e.g. ``"clang-15"``.

    Returns
    -------
    pathlib.Path or None
        The resolved absolute path, or `None` if `name` is not found in
        ``PATH``.

    Examples
    --------
    >>> from numba_enzyme.toolchain import _which
    >>> _which("arbitrary-tool") is None
    True
    """
    found = shutil.which(name)
    return Path(found) if found else None


def _resolve_vendored() -> Toolchain | None:
    """
    Resolve tools from the `_vendor/` directory shipped in a wheel.

    Returns
    -------
    Toolchain or None
        The vendored paths, or `None` if `_vendor/` doesn't exist
        (e.g. running from source rather than an installed wheel).

    See Also
    --------
    _resolve_system : The fallback used when this returns `None`.

    Examples
    --------
    >>> from numba_enzyme.toolchain import _resolve_vendored
    >>> _resolve_vendored() is None  # doctest: +SKIP
    True
    """
    vendored_clang = _VENDOR_DIR / "bin" / "clang"
    if not vendored_clang.is_file():
        return None
    return Toolchain(
        clang=vendored_clang,
        llvm_link=_VENDOR_DIR / "bin" / "llvm-link",
        opt=_VENDOR_DIR / "bin" / "opt",
        llvm_extract=_VENDOR_DIR / "bin" / "llvm-extract",
        enzyme_plugin=_VENDOR_DIR / "enzyme" / "LLVMEnzyme-15.so",
    )


def _resolve_system() -> tuple[Toolchain | None, list[str]]:
    """
    Resolve tools from the system ``PATH`` (the development-mode path).

    Returns
    -------
    Toolchain or None
        The resolved paths, or `None` if any required tool is missing.
    list of str
        A description of each missing tool, empty if none are missing.

    See Also
    --------
    _resolve_vendored : Tried first, before this fallback.

    Examples
    --------
    >>> from numba_enzyme.toolchain import _resolve_system
    >>> _resolve_system()  # doctest: +SKIP
    (Toolchain(clang=..., llvm_link=..., opt=..., enzyme_plugin=...), [])
    """
    clang = _which("clang-15")
    llvm_link = _which("llvm-link-15")
    opt = _which("opt-15")
    llvm_extract = _which("llvm-extract-15")

    plugin_override = os.environ.get(_PLUGIN_PATH_ENV_VAR)
    enzyme_plugin = Path(plugin_override) if plugin_override else _DEFAULT_PLUGIN_PATH

    missing = []
    if clang is None:
        missing.append("clang-15 (not found on PATH)")
    if llvm_link is None:
        missing.append("llvm-link-15 (not found on PATH)")
    if opt is None:
        missing.append("opt-15 (not found on PATH)")
    if llvm_extract is None:
        missing.append("llvm-extract-15 (not found on PATH)")
    if not enzyme_plugin.is_file():
        missing.append(
            f"Enzyme plugin (not found at {enzyme_plugin}; "
            f"override with the {_PLUGIN_PATH_ENV_VAR} env var)"
        )
    if missing:
        return None, missing

    return (
        Toolchain(
            clang=clang,
            llvm_link=llvm_link,
            opt=opt,
            llvm_extract=llvm_extract,
            enzyme_plugin=enzyme_plugin,
        ),
        [],
    )


@lru_cache(maxsize=1)
def get_toolchain() -> Toolchain:
    """
    Resolve and validate every build tool.

    Priorities the `_vendor/` directory shipped alongside
    this module in a built wheel; falls back to
    ``clang-15``/``llvm-link-15``/``opt-15`` on ``PATH``
    and the standalone Enzyme LLVM pass plugin at a path
    relative to the repository root otherwise (configurable
    via the ``NUMBA_ENZYME_PLUGIN_PATH`` environment variable)
    The result is cached after the first successful call.

    Returns
    -------
    Toolchain
        The resolved, validated paths of the build tools.

    Raises
    ------
    ToolchainError
        If any tool is missing, or exists but is not
        executable/readable.

    See Also
    --------
    Toolchain : The return resolved paths.

    Examples
    --------
    >>> from numba_enzyme.toolchain import get_toolchain
    >>> get_toolchain()  # doctest: +SKIP
    Toolchain(clang=..., llvm_link=..., opt=..., enzyme_plugin=...)

    Force re-resolution (e.g. in tests that change the environment)::

    >>> get_toolchain.cache_clear()
    """
    toolchain = _resolve_vendored()
    if toolchain is None:
        toolchain, missing = _resolve_system()
        if toolchain is None:
            raise ToolchainError(
                "missing required build tool(s):\n  - " + "\n  - ".join(missing)
            )

    for tool in (
        toolchain.clang,
        toolchain.llvm_link,
        toolchain.opt,
        toolchain.llvm_extract,
    ):
        if not tool.is_file():
            raise ToolchainError(f"{tool} does not exist")
        if not os.access(tool, os.X_OK):
            raise ToolchainError(f"{tool} exists but is not executable")
    if not toolchain.enzyme_plugin.is_file():
        raise ToolchainError(f"{toolchain.enzyme_plugin} does not exist")
    if not os.access(toolchain.enzyme_plugin, os.R_OK):
        raise ToolchainError(f"{toolchain.enzyme_plugin} exists but is not readable")

    return toolchain
