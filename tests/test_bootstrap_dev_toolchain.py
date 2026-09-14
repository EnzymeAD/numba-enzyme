"""Tests for the development-toolchain bootstrap."""

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1] / "packaging" / "bootstrap_dev_toolchain.py"
)
_SPEC = importlib.util.spec_from_file_location("bootstrap_dev_toolchain", _SCRIPT)
bootstrap_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bootstrap_module)


def _populate_fake_wheel(destination, wheel_version):
    vendor = destination / "wheel" / "numba_enzyme" / "_vendor"
    bin_dir = vendor / "bin"
    bin_dir.mkdir(parents=True)
    for name in bootstrap_module._TOOLS:
        executable = bin_dir / name
        executable.write_bytes(b"#!/bin/sh\n")
        executable.chmod(0o755)
    enzyme_dir = vendor / "enzyme"
    enzyme_dir.mkdir()
    (enzyme_dir / "LLVMEnzyme-15.so").write_bytes(b"plugin")
    (vendor / "crt").mkdir()
    (destination / "wheel" / "numba_enzyme.libs").mkdir()


def test_bootstrap_is_validated_and_idempotent(tmp_path, monkeypatch):
    destination = tmp_path / "development-toolchain"
    installs = []

    def install(staged, wheel_version):
        installs.append(wheel_version)
        _populate_fake_wheel(staged, wheel_version)

    monkeypatch.setattr(bootstrap_module, "_install_with_uv", install)

    assert bootstrap_module.bootstrap(destination, "test-version") is True
    assert bootstrap_module._validate(destination) == []
    assert (
        json.loads((destination / "bootstrap.json").read_text())["wheel_version"]
        == "test-version"
    )
    assert bootstrap_module.bootstrap(destination, "test-version") is False
    assert installs == ["test-version"]


def test_bootstrap_rejects_broad_destructive_destination():
    with pytest.raises(ValueError, match="unsafe toolchain destination"):
        bootstrap_module.bootstrap(Path("/"), "test-version", force=True)
