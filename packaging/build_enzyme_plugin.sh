#!/usr/bin/env bash
# Clones the pinned Enzyme commit and builds `libNumbaEnzyme.{so,dylib}`
# via CMake against a local LLVM 22 installation (for non-Bazel local dev).
#
# Usage: build_enzyme_plugin.sh <output-dir>
# Produces <output-dir>/libNumbaEnzyme.so (or .dylib on macOS).

set -euo pipefail

OUTPUT_DIR="${1:?usage: build_enzyme_plugin.sh <output-dir>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENZYME_COMMIT="ccce9cf45c6e1c1db4b64b419776b94ecf707c93"

if [[ -z "${LLVM_DIR:-}" ]]; then
    if [[ "$(uname -s)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
        if brew --prefix llvm@22 >/dev/null 2>&1; then
            LLVM_DIR="$(brew --prefix llvm@22)/lib/cmake/llvm"
        elif brew --prefix llvm >/dev/null 2>&1; then
            LLVM_DIR="$(brew --prefix llvm)/lib/cmake/llvm"
        fi
    elif [[ -d "/usr/lib/llvm-22/lib/cmake/llvm" ]]; then
        LLVM_DIR="/usr/lib/llvm-22/lib/cmake/llvm"
    fi
fi

if [[ -z "${LLVM_DIR:-}" ]]; then
    echo "error: LLVM_DIR is not set and LLVM 22 was not found in standard paths." >&2
    echo "Set LLVM_DIR=/path/to/lib/cmake/llvm or build with Bazel instead:" >&2
    echo "  bazel build //src/numba_enzyme:libNumbaEnzyme.so" >&2
    exit 1
fi

if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
elif command -v sysctl >/dev/null 2>&1; then
    JOBS="$(sysctl -n hw.ncpu)"
else
    JOBS=4
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

echo "== fetching Enzyme @ $ENZYME_COMMIT =="
git init "$WORK_DIR/enzyme-src"
git -C "$WORK_DIR/enzyme-src" remote add origin https://github.com/EnzymeAD/Enzyme.git
git -C "$WORK_DIR/enzyme-src" fetch --depth 1 origin "$ENZYME_COMMIT"
git -C "$WORK_DIR/enzyme-src" checkout FETCH_HEAD

echo "== configuring libNumbaEnzyme against $LLVM_DIR =="
cmake -S "$REPO_ROOT/src/numba_enzyme/csrc" -B "$OUTPUT_DIR" \
    -DLLVM_DIR="$LLVM_DIR" \
    -DENZYME_SOURCE_DIR="$WORK_DIR/enzyme-src" \
    -DCMAKE_BUILD_TYPE=Release

echo "== building libNumbaEnzyme =="
cmake --build "$OUTPUT_DIR" --target NumbaEnzyme -j"$JOBS"
