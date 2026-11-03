import os

import pytest

from numba_enzyme.toolchain import Toolchain, ToolchainError, get_toolchain


def test_get_toolchain_resolves_existing_library():
    get_toolchain.cache_clear()
    tc = get_toolchain()
    assert isinstance(tc, Toolchain)
    assert tc.enzyme_lib.is_file()
    assert os.access(tc.enzyme_lib, os.R_OK)
    assert tc.enzyme_plugin == tc.enzyme_lib
    get_toolchain.cache_clear()


def test_missing_tool_raises_toolchain_error(monkeypatch, tmp_path):
    get_toolchain.cache_clear()
    monkeypatch.delenv("NUMBA_ENZYME_LIB_PATH", raising=False)
    monkeypatch.delenv("NUMBA_ENZYME_PLUGIN_PATH", raising=False)
    monkeypatch.setattr("numba_enzyme.toolchain._VENDOR_DIR", tmp_path / "no-vendor")
    monkeypatch.setattr("numba_enzyme.toolchain._PACKAGE_DIR", tmp_path / "no-pkg")
    monkeypatch.setattr("numba_enzyme.toolchain._REPO_ROOT", tmp_path / "no-repo")
    with pytest.raises(ToolchainError, match="libNumbaEnzyme"):
        get_toolchain()
    get_toolchain.cache_clear()


def test_lib_path_override(monkeypatch, tmp_path):
    get_toolchain.cache_clear()
    fake_lib = tmp_path / "libNumbaEnzyme.so"
    fake_lib.write_bytes(b"not a real library, just here for a stat check")
    monkeypatch.setenv("NUMBA_ENZYME_LIB_PATH", str(fake_lib))
    tc = get_toolchain()
    assert tc.enzyme_lib == fake_lib
    get_toolchain.cache_clear()


def test_plugin_path_override(monkeypatch, tmp_path):
    get_toolchain.cache_clear()
    monkeypatch.delenv("NUMBA_ENZYME_LIB_PATH", raising=False)
    fake_plugin = tmp_path / "libNumbaEnzyme.so"
    fake_plugin.write_bytes(b"not a real plugin, just here for a stat check")
    monkeypatch.setenv("NUMBA_ENZYME_PLUGIN_PATH", str(fake_plugin))
    tc = get_toolchain()
    assert tc.enzyme_lib == fake_plugin
    assert tc.enzyme_plugin == fake_plugin
    get_toolchain.cache_clear()


def test_vendored_toolchain_is_preferred_when_present(monkeypatch, tmp_path):
    get_toolchain.cache_clear()
    monkeypatch.delenv("NUMBA_ENZYME_LIB_PATH", raising=False)
    monkeypatch.delenv("NUMBA_ENZYME_PLUGIN_PATH", raising=False)
    vendor_dir = tmp_path / "_vendor"
    vendor_dir.mkdir()
    vendored_lib = vendor_dir / "libNumbaEnzyme.so"
    vendored_lib.write_bytes(b"not a real library, just here for a stat check")
    monkeypatch.setattr("numba_enzyme.toolchain._VENDOR_DIR", vendor_dir)

    tc = get_toolchain()
    assert tc.enzyme_lib == vendored_lib
    get_toolchain.cache_clear()


def test_falls_back_when_vendor_dir_absent(monkeypatch, tmp_path):
    get_toolchain.cache_clear()
    monkeypatch.delenv("NUMBA_ENZYME_LIB_PATH", raising=False)
    monkeypatch.delenv("NUMBA_ENZYME_PLUGIN_PATH", raising=False)
    absent_dir = tmp_path / "does-not-exist"
    monkeypatch.setattr("numba_enzyme.toolchain._VENDOR_DIR", absent_dir)

    tc = get_toolchain()
    assert tc.enzyme_lib != absent_dir / "libNumbaEnzyme.so"
    assert tc.enzyme_lib.is_file()
    get_toolchain.cache_clear()
