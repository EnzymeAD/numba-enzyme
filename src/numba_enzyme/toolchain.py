"""
Locate the native Enzyme FFI shared library (`libNumbaEnzyme`).

Resolves the in-process Enzyme LLVM pass shared library shipped alongside
this module (in Bazel runfiles, Bazel-built wheels, or `_vendor/`), with
environment-variable overrides and development-tree fallbacks.

See Also
--------
numba_enzyme.build.build : Uses this library to link and differentiate
    lowered kernels in-process.

Examples
--------
>>> from numba_enzyme.toolchain import get_toolchain
>>> toolchain = get_toolchain()  # doctest: +SKIP
>>> toolchain.enzyme_lib.name  # doctest: +SKIP
'libNumbaEnzyme.so'
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_PACKAGE_DIR = Path(os.path.abspath(__file__)).parent
_VENDOR_DIR = _PACKAGE_DIR / "_vendor"
_REPO_ROOT = Path(__file__).resolve().parents[2]

_LIB_NAMES = ("libNumbaEnzyme.so", "libNumbaEnzyme.dylib")
_LIB_PATH_ENV_VAR = "NUMBA_ENZYME_LIB_PATH"
_LEGACY_PLUGIN_ENV_VAR = "NUMBA_ENZYME_PLUGIN_PATH"


class ToolchainError(RuntimeError):
    """
    Raise when the native Enzyme shared library cannot be located or read.

    Covers both a missing shared library (not found in the package, `_vendor/`,
    or development build directories) and an unreadable file at the configured
    path.

    See Also
    --------
    get_toolchain : Raises this error when library resolution fails.

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
    Resolved and validated path to the native Enzyme shared library.

    Attributes
    ----------
    enzyme_lib : pathlib.Path
        Path to `libNumbaEnzyme.so` (or `libNumbaEnzyme.dylib`).

    See Also
    --------
    get_toolchain : Builds and validates a `Toolchain` instance.

    Examples
    --------
    >>> from numba_enzyme.toolchain import get_toolchain
    >>> get_toolchain().enzyme_lib.is_file()  # doctest: +SKIP
    True
    """

    enzyme_lib: Path

    @property
    def enzyme_plugin(self) -> Path:
        """
        Backward-compatible alias for `enzyme_lib`.

        Returns
        -------
        pathlib.Path
            Path to the resolved `libNumbaEnzyme` shared library.

        See Also
        --------
        Toolchain : Carries `enzyme_lib` as its primary attribute.

        Examples
        --------
        >>> from numba_enzyme.toolchain import get_toolchain
        >>> tc = get_toolchain()  # doctest: +SKIP
        >>> tc.enzyme_plugin == tc.enzyme_lib  # doctest: +SKIP
        True
        """
        return self.enzyme_lib


def _first_existing(directory: Path) -> Path | None:
    """
    Find the first existing `libNumbaEnzyme` file in a directory.

    Parameters
    ----------
    directory : pathlib.Path
        Directory to probe for `_LIB_NAMES`.

    Returns
    -------
    pathlib.Path or None
        Path to the first matching file, or `None` if absent.

    See Also
    --------
    get_toolchain : Calls this helper across candidate directories.

    Examples
    --------
    >>> from pathlib import Path
    >>> from numba_enzyme.toolchain import _first_existing
    >>> _first_existing(Path("/nonexistent-numba-enzyme-dir")) is None
    True
    """
    for name in _LIB_NAMES:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _resolve_packaged() -> Toolchain | None:
    """
    Resolve `libNumbaEnzyme` shipped inside the package or `_vendor/`.

    Checks `_VENDOR_DIR` first (used by Hatch wheel builds) and then
    `_PACKAGE_DIR` (used by Bazel wheels and Bazel test runfiles).

    Returns
    -------
    Toolchain or None
        The packaged library toolchain, or `None` if neither location
        contains `libNumbaEnzyme`.

    See Also
    --------
    _resolve_dev : Fallback used during local source development.

    Examples
    --------
    >>> from numba_enzyme.toolchain import _resolve_packaged
    >>> _resolve_packaged()  # doctest: +SKIP
    Toolchain(enzyme_lib=PosixPath('.../libNumbaEnzyme.so'))
    """
    for directory in (_VENDOR_DIR, _PACKAGE_DIR):
        found = _first_existing(directory)
        if found is not None:
            return Toolchain(enzyme_lib=found)
    return None


def _resolve_dev() -> tuple[Toolchain | None, list[Path]]:
    """
    Resolve `libNumbaEnzyme` from local development build directories.

    Checks `enzyme-build/` (populated by `packaging/build_enzyme_plugin.sh`)
    and `bazel-bin/src/numba_enzyme/` (populated by `bazel build`).

    Returns
    -------
    Toolchain or None
        The resolved development library, or `None` if not built yet.
    list of pathlib.Path
        Directories searched when resolution returns `None`.

    See Also
    --------
    _resolve_packaged : Checked before falling back to development paths.

    Examples
    --------
    >>> from numba_enzyme.toolchain import _resolve_dev
    >>> _resolve_dev()  # doctest: +SKIP
    (Toolchain(enzyme_lib=...), [...])
    """
    dev_dirs = [
        _REPO_ROOT / "enzyme-build",
        _REPO_ROOT / "bazel-bin" / "src" / "numba_enzyme",
    ]
    for directory in dev_dirs:
        found = _first_existing(directory)
        if found is not None:
            return Toolchain(enzyme_lib=found), dev_dirs
    return None, dev_dirs


@lru_cache(maxsize=1)
def get_toolchain() -> Toolchain:
    """
    Resolve and validate the native Enzyme shared library.

    Resolution order:

    1. Environment variable override (`NUMBA_ENZYME_LIB_PATH`, or legacy
       `NUMBA_ENZYME_PLUGIN_PATH`).
    2. Packaged library in `_vendor/` or alongside `numba_enzyme` (Bazel
       wheels and runfiles).
    3. Local development build outputs in `enzyme-build/` or
       `bazel-bin/src/numba_enzyme/`.

    The result is cached after the first successful call.

    Returns
    -------
    Toolchain
        The resolved and validated path to `libNumbaEnzyme`.

    Raises
    ------
    ToolchainError
        If `libNumbaEnzyme` cannot be found or is not readable.

    See Also
    --------
    Toolchain : Dataclass holding the resolved `enzyme_lib` path.

    Examples
    --------
    >>> from numba_enzyme.toolchain import get_toolchain
    >>> get_toolchain()  # doctest: +SKIP
    Toolchain(enzyme_lib=PosixPath('.../libNumbaEnzyme.so'))

    Force re-resolution (e.g. in tests that change the environment):

    >>> get_toolchain.cache_clear()
    """
    override = os.environ.get(_LIB_PATH_ENV_VAR) or os.environ.get(
        _LEGACY_PLUGIN_ENV_VAR
    )
    if override:
        toolchain = Toolchain(enzyme_lib=Path(override))
    else:
        toolchain = _resolve_packaged()
        if toolchain is None:
            toolchain, searched_dirs = _resolve_dev()
            if toolchain is None:
                searched = ", ".join(
                    str(d) for d in (_VENDOR_DIR, _PACKAGE_DIR, *searched_dirs)
                )
                raise ToolchainError(
                    f"libNumbaEnzyme shared library not found (searched: "
                    f"{searched}); build with `bazel build "
                    "//src/numba_enzyme:libNumbaEnzyme.so` or "
                    "`bash packaging/build_enzyme_plugin.sh enzyme-build`, or "
                    f"override via {_LIB_PATH_ENV_VAR}"
                )

    if not toolchain.enzyme_lib.is_file():
        raise ToolchainError(f"{toolchain.enzyme_lib} does not exist")
    if not os.access(toolchain.enzyme_lib, os.R_OK):
        raise ToolchainError(f"{toolchain.enzyme_lib} exists but is not readable")

    return toolchain
