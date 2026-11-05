"""Pytest entrypoint for Bazel `py_test` targets."""

import sys

import pytest

if __name__ == "__main__":
    raise SystemExit(pytest.main(sys.argv[1:]))
