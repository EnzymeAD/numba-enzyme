load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

exports_files([
    "csrc/numba_enzyme.exp",
    "csrc/numba_enzyme.lds",
    "csrc/numba_enzyme_ffi.cpp",
])

cc_binary(
    name = "libNumbaEnzyme.so",
    srcs = ["csrc/numba_enzyme_ffi.cpp"],
    additional_linker_inputs = select({
        "@platforms//os:macos": ["csrc/numba_enzyme.exp"],
        "//conditions:default": ["csrc/numba_enzyme.lds"],
    }),
    linkopts = select({
        "@platforms//os:macos": [
            "-Wl,-dead_strip",
            "-Wl,-exported_symbols_list,$(location csrc/numba_enzyme.exp)",
        ],
        "//conditions:default": [
            "-Wl,--gc-sections",
            "-Wl,--version-script,$(location csrc/numba_enzyme.lds)",
        ],
    }),
    linkshared = 1,
    linkstatic = 1,
    deps = [
        "@enzyme//:EnzymeStatic",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:IRPrinter",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
    ],
)

py_library(
    name = "numba_enzyme_py",
    srcs = glob(["*.py"]),
    imports = [".."],
)
