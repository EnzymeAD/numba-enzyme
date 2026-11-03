# Match llvmlite 0.50.x's LLVM 22.1 release series so textual LLVM IR parsed
# and emitted by libNumbaEnzyme matches llvmlite.binding.parse_assembly().
LLVM_VERSION = "22.1.8"
LLVM_SHA256 = "922f1817a0df7b1489272d18134ee0087a8b068828f87ac63b9861b1a9965888"

# Enzyme v0.0.304
ENZYME_COMMIT = "ccce9cf45c6e1c1db4b64b419776b94ecf707c93"
ENZYME_SHA256 = "e303de90e9b962ba791733910995d866249f15061c4ddea9f8cfa219c021947c"

# If non-empty, use a local Enzyme directory (pointing to its `enzyme/` subdir
# containing the Bazel BUILD file) instead of downloading ENZYME_COMMIT.
OVERRIDE_ENZYME_PATH = ""

LLVM_TARGETS = ["X86", "AArch64"]
