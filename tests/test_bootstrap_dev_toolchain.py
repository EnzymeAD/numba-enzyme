"""Tests for the development-toolchain bootstrap."""

import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1] / "packaging" / "bootstrap_dev_toolchain.py"
)
_SPEC = importlib.util.spec_from_file_location("bootstrap_dev_toolchain", _SCRIPT)
bootstrap_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bootstrap_module)

_HELPER = _SCRIPT.with_name("llvm_extract.py")
_HELPER_SPEC = importlib.util.spec_from_file_location("llvm_extract", _HELPER)
llvm_extract_module = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(llvm_extract_module)


def _populate_fake_wheel(destination, wheel_version, tools=None):
    vendor = destination / "wheel" / "numba_enzyme" / "_vendor"
    bin_dir = vendor / "bin"
    bin_dir.mkdir(parents=True)
    for name in bootstrap_module._TOOLS if tools is None else tools:
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


def test_bootstrap_fetches_llvm_extract_the_wheel_lacks(tmp_path, monkeypatch):
    destination = tmp_path / "development-toolchain"
    without = tuple(name for name in bootstrap_module._TOOLS if name != "llvm-extract")
    fetched = []

    def fetch(path):
        fetched.append(path.name)
        path.write_bytes(b"#!/bin/sh\n")
        path.chmod(0o755)

    monkeypatch.setattr(
        bootstrap_module,
        "_install_with_uv",
        lambda staged, version: _populate_fake_wheel(staged, version, without),
    )
    monkeypatch.setattr(bootstrap_module, "_fetch_llvm_extract", fetch)

    assert bootstrap_module.bootstrap(destination, "test-version") is True
    assert fetched == ["llvm-extract"]
    assert bootstrap_module._validate(destination) == []


def _release_archive(tmp_path, members):
    archive = tmp_path / "release.tar.xz"
    with tarfile.open(archive, "w:xz") as bundle:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    return archive.as_uri()


def test_fetch_llvm_extract_checks_the_binary(tmp_path, monkeypatch):
    url = _release_archive(tmp_path, {"llvm/bin/llvm-extract": b"binary"})
    destination = tmp_path / "bin" / "llvm-extract"

    with pytest.raises(RuntimeError, match="SHA-256"):
        llvm_extract_module.fetch_llvm_extract(destination, url)
    assert not destination.exists()

    monkeypatch.setattr(
        llvm_extract_module,
        "_SHA256",
        "9a3a45d01531a20e89ac6ae10b0b0beb0492acd7216a368aa062d1a5fecaf9cd",
    )
    llvm_extract_module.fetch_llvm_extract(destination, url)
    assert destination.read_bytes() == b"binary"
    assert destination.stat().st_mode & 0o111


def test_fetch_llvm_extract_requires_the_binary(tmp_path):
    url = _release_archive(tmp_path, {"llvm/bin/opt": b"other"})
    with pytest.raises(RuntimeError, match="has no bin/llvm-extract"):
        llvm_extract_module.fetch_llvm_extract(tmp_path / "llvm-extract", url)
