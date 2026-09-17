import tomllib
from pathlib import Path

import numba_enzyme

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_import():
    pyproject = tomllib.loads(_PYPROJECT.read_text())
    assert numba_enzyme.__version__ == pyproject["project"]["version"]


def test_public_derivative_api_is_exported():
    expected = {
        "grad",
        "jacfwd",
        "jacrev",
        "jvp",
        "vjp",
    }
    assert expected <= set(numba_enzyme.__all__)
    assert all(callable(getattr(numba_enzyme, name)) for name in expected)
