import os

import pytest

from numba_enzyme.toolchain import Toolchain, ToolchainError, get_toolchain


def test_get_toolchain_resolves_existing_executables():
    get_toolchain.cache_clear()
    tc = get_toolchain()
    assert isinstance(tc, Toolchain)
    for path in (tc.clang, tc.llvm_link, tc.opt, tc.llvm_extract):
        assert path.is_file()
        assert os.access(path, os.X_OK)
    assert tc.enzyme_plugin.is_file()
    get_toolchain.cache_clear()


def test_missing_tool_raises_toolchain_error(monkeypatch):
    get_toolchain.cache_clear()
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(ToolchainError, match="clang-15"):
        get_toolchain()
    get_toolchain.cache_clear()


def test_plugin_path_override(monkeypatch, tmp_path):
    get_toolchain.cache_clear()
    fake_plugin = tmp_path / "LLVMEnzyme-15.so"
    fake_plugin.write_bytes(b"not a real plugin, just here for a stat check")
    monkeypatch.setenv("NUMBA_ENZYME_PLUGIN_PATH", str(fake_plugin))
    tc = get_toolchain()
    assert tc.enzyme_plugin == fake_plugin
    get_toolchain.cache_clear()


def _make_fake_vendor_dir(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    enzyme_dir = tmp_path / "enzyme"
    enzyme_dir.mkdir()
    for name in ("clang", "llvm-link", "opt", "llvm-extract"):
        fake = bin_dir / name
        fake.write_bytes(b"#!/bin/sh\n")
        fake.chmod(0o755)
    plugin = enzyme_dir / "LLVMEnzyme-15.so"
    plugin.write_bytes(b"not a real plugin, just here for a stat check")
    return tmp_path


def test_vendored_toolchain_is_preferred_when_present(monkeypatch, tmp_path):
    get_toolchain.cache_clear()
    vendor_dir = _make_fake_vendor_dir(tmp_path)
    monkeypatch.setattr("numba_enzyme.toolchain._VENDOR_DIR", vendor_dir)

    tc = get_toolchain()
    assert tc.clang == vendor_dir / "bin" / "clang"
    assert tc.llvm_link == vendor_dir / "bin" / "llvm-link"
    assert tc.opt == vendor_dir / "bin" / "opt"
    assert tc.llvm_extract == vendor_dir / "bin" / "llvm-extract"
    assert tc.enzyme_plugin == vendor_dir / "enzyme" / "LLVMEnzyme-15.so"
    get_toolchain.cache_clear()


def test_falls_back_to_system_when_vendor_dir_absent(monkeypatch, tmp_path):
    get_toolchain.cache_clear()
    absent_dir = tmp_path / "does-not-exist"
    monkeypatch.setattr("numba_enzyme.toolchain._VENDOR_DIR", absent_dir)

    tc = get_toolchain()
    assert tc.clang != absent_dir / "bin" / "clang"
    assert tc.clang.is_file()
    get_toolchain.cache_clear()
