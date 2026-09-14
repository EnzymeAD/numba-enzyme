import os
from pathlib import Path

import pytest

from numba_enzyme.toolchain import Toolchain, ToolchainError, get_toolchain


def test_get_toolchain_resolves_existing_executables():
    get_toolchain.cache_clear()
    tc = get_toolchain()
    assert isinstance(tc, Toolchain)
    for path in (tc.clang, tc.llvm_link, tc.opt):
        assert path.is_file()
        assert os.access(path, os.X_OK)
    assert tc.enzyme_plugin.is_file()
    get_toolchain.cache_clear()


def test_missing_tool_raises_toolchain_error(monkeypatch):
    get_toolchain.cache_clear()
    monkeypatch.setattr(
        "numba_enzyme.toolchain._DEV_VENDOR_DIR", Path("/nonexistent/dev-vendor")
    )
    monkeypatch.setattr("numba_enzyme.toolchain._which", lambda name: None)
    with pytest.raises(ToolchainError, match="bootstrap_dev_toolchain.py"):
        get_toolchain()
    get_toolchain.cache_clear()


def test_plugin_path_override(monkeypatch, tmp_path):
    get_toolchain.cache_clear()
    monkeypatch.setattr(
        "numba_enzyme.toolchain._DEV_VENDOR_DIR", tmp_path / "absent-dev-vendor"
    )
    fake_plugin = tmp_path / "LLVMEnzyme-15.so"
    fake_plugin.write_bytes(b"not a real plugin, just here for a stat check")
    monkeypatch.setenv("NUMBA_ENZYME_PLUGIN_PATH", str(fake_plugin))
    system_tools = _make_fake_system_tools(tmp_path)
    monkeypatch.setattr(
        "numba_enzyme.toolchain._which", lambda name: system_tools[name]
    )
    tc = get_toolchain()
    assert tc.enzyme_plugin == fake_plugin
    get_toolchain.cache_clear()


def _make_fake_vendor_dir(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    enzyme_dir = tmp_path / "enzyme"
    enzyme_dir.mkdir()
    for name in ("clang", "llvm-link", "opt"):
        fake = bin_dir / name
        fake.write_bytes(b"#!/bin/sh\n")
        fake.chmod(0o755)
    plugin = enzyme_dir / "LLVMEnzyme-15.so"
    plugin.write_bytes(b"not a real plugin, just here for a stat check")
    return tmp_path


def _make_fake_system_tools(tmp_path):
    tools = {}
    for name in ("clang-15", "llvm-link-15", "opt-15"):
        path = tmp_path / name
        path.write_bytes(b"#!/bin/sh\n")
        path.chmod(0o755)
        tools[name] = path
    return tools


def test_vendored_toolchain_is_preferred_when_present(monkeypatch, tmp_path):
    get_toolchain.cache_clear()
    vendor_dir = _make_fake_vendor_dir(tmp_path)
    monkeypatch.setattr("numba_enzyme.toolchain._VENDOR_DIR", vendor_dir)

    tc = get_toolchain()
    assert tc.clang == vendor_dir / "bin" / "clang"
    assert tc.llvm_link == vendor_dir / "bin" / "llvm-link"
    assert tc.opt == vendor_dir / "bin" / "opt"
    assert tc.enzyme_plugin == vendor_dir / "enzyme" / "LLVMEnzyme-15.so"
    get_toolchain.cache_clear()


def test_bootstrapped_development_toolchain_is_second_choice(monkeypatch, tmp_path):
    get_toolchain.cache_clear()
    absent_vendor = tmp_path / "absent-installed-vendor"
    development_vendor = _make_fake_vendor_dir(tmp_path / "development")
    monkeypatch.setattr("numba_enzyme.toolchain._VENDOR_DIR", absent_vendor)
    monkeypatch.setattr("numba_enzyme.toolchain._DEV_VENDOR_DIR", development_vendor)
    monkeypatch.setenv("PATH", "/nonexistent")

    tc = get_toolchain()
    assert tc.clang == development_vendor / "bin" / "clang"
    assert tc.enzyme_plugin == (development_vendor / "enzyme" / "LLVMEnzyme-15.so")
    get_toolchain.cache_clear()


def test_falls_back_to_system_when_vendor_dir_absent(monkeypatch, tmp_path):
    get_toolchain.cache_clear()
    absent_dir = tmp_path / "does-not-exist"
    monkeypatch.setattr("numba_enzyme.toolchain._VENDOR_DIR", absent_dir)
    monkeypatch.setattr("numba_enzyme.toolchain._DEV_VENDOR_DIR", absent_dir)
    plugin = tmp_path / "LLVMEnzyme-15.so"
    plugin.write_bytes(b"plugin")
    monkeypatch.setenv("NUMBA_ENZYME_PLUGIN_PATH", str(plugin))
    system_tools = _make_fake_system_tools(tmp_path)
    monkeypatch.setattr(
        "numba_enzyme.toolchain._which", lambda name: system_tools[name]
    )

    tc = get_toolchain()
    assert tc.clang != absent_dir / "bin" / "clang"
    assert tc.clang.is_file()
    get_toolchain.cache_clear()
