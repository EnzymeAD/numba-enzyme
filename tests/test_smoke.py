import re
from pathlib import Path

import numba_enzyme

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_import():
    # A regex rather than tomllib, which only arrived in Python 3.11.
    match = re.search(r'^version = "([^"]+)"$', _PYPROJECT.read_text(), re.MULTILINE)
    assert match is not None
    assert numba_enzyme.__version__ == match.group(1)


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
