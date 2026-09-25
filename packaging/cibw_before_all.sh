#!/usr/bin/env bash
# CIBW_BEFORE_ALL: runs once per platform container, before any
# per-Python-version wheel build starts. Builds LLVMEnzyme-15.so from
# the pinned Enzyme release and stages it plus clang/opt/llvm-link/
# llvm-extract/ld.lld into a location that persists for the rest of this
# cibuildwheel session
# (packaging/cibw_before_build.sh copies from here into
# src/numba_enzyme/_vendor/ before each per-Python-version build).
#
# UNVERIFIED AS WRITTEN: authored without access to a real
# cibuildwheel/Docker run (see packaging/Dockerfile.manylinux's header).
# The cmake invocation itself is NOT a guess -- it's the exact one
# already proven working in numba_autodiff/enzyme-build/ on the dev box.
#
# Usage: cibw_before_all.sh
# (no project-dir argument needed -- Enzyme is cloned fresh from
# GitHub, nothing from the project tree is read here)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGING_DIR="/tmp/numba_enzyme_vendor_staging"

rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR/bin" "$STAGING_DIR/enzyme" "$STAGING_DIR/crt"

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

# Pinned Enzyme tag + CMake flags live in build_enzyme_plugin.sh, shared
# with .github/workflows/tests.yml so the two can't silently drift.
bash "$SCRIPT_DIR/build_enzyme_plugin.sh" "$WORK_DIR/enzyme-build"

echo "== staging vendored binaries =="
cp "$WORK_DIR/enzyme-build/Enzyme/LLVMEnzyme-15.so" "$STAGING_DIR/enzyme/LLVMEnzyme-15.so"
cp /usr/lib/llvm-15/bin/clang "$STAGING_DIR/bin/clang"
cp /usr/lib/llvm-15/bin/opt "$STAGING_DIR/bin/opt"
cp /usr/lib/llvm-15/bin/llvm-link "$STAGING_DIR/bin/llvm-link"
cp /usr/lib/llvm-15/bin/llvm-extract "$STAGING_DIR/bin/llvm-extract"
# `clang -shared` shells out to a separate linker executable for the
# final link step -- vendoring lld (LLVM's own linker) as `ld.lld`
# right next to clang keeps this self-contained with no system linker
# dependency. `-L` dereferences the real binary rather than copying the
# `ld.lld -> lld` symlink itself, since only the dispatch-on-argv[0]
# name (`ld.lld`) matters, not how it got there. Confirmed working with
# a fully stripped environment before wiring this in -- see build.py's
# `-fuse-ld=lld` flag, which is what actually makes clang use this
# instead of looking for a system `ld`.
cp -L /usr/lib/llvm-15/bin/ld.lld "$STAGING_DIR/bin/ld.lld"

# The final `clang -shared` link step needs the C runtime startup
# objects and libgcc that a system C toolchain (gcc/libc6-dev) normally
# provides -- confirmed missing on a bare-minimum target
# (python:3.11-slim has none of these: "ld.lld: error: cannot open
# crti.o", "unable to find library -lgcc"). Unlike libc/libm
# themselves (which build.py locates on the *target* machine at
# runtime via /proc/self/maps, since that's the one machine guaranteed
# to already have them loaded), these come from the *build* image's
# own gcc installation and are baked into the wheel here. `libgcc.a`
# (the static archive, not `libgcc_s.so`) is deliberately used so the
# resulting .so has no runtime dependency on libgcc_s.so.1 either --
# confirmed via `ldd` on a real linked test .so showing only
# libc.so.6/libm.so.6 as NEEDED entries. `gcc -print-file-name=` finds
# the exact paths portably regardless of the installed gcc version.
cp "$(gcc -print-file-name=crti.o)" "$STAGING_DIR/crt/crti.o"
cp "$(gcc -print-file-name=crtn.o)" "$STAGING_DIR/crt/crtn.o"
cp "$(gcc -print-file-name=crtbeginS.o)" "$STAGING_DIR/crt/crtbeginS.o"
cp "$(gcc -print-file-name=crtendS.o)" "$STAGING_DIR/crt/crtendS.o"
cp "$(gcc -print-file-name=libgcc.a)" "$STAGING_DIR/crt/libgcc.a"

echo "== staged contents =="
find "$STAGING_DIR" -type f -exec ls -lh {} \;
