load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")
load("@rules_python//python:defs.bzl", "py_library")
load("@rules_python//python:packaging.bzl", "py_wheel")
load(":package.bzl", "py_package")

licenses(["notice"])

exports_files([
    "LICENSE",
    "NOTICE",
    "README.md",
    "pyproject.toml",
])

py_package(
    name = "numba_enzyme_data",
    packages = [
        "//src/numba_enzyme:libNumbaEnzyme.so",
        "@//src/numba_enzyme:libNumbaEnzyme.so",
    ],
    deps = [
        "//src/numba_enzyme:libNumbaEnzyme.so",
    ],
)

py_library(
    name = "numba_enzyme",
    data = [
        ":numba_enzyme_data",
    ],
    imports = ["src"],
    visibility = ["//visibility:public"],
    deps = [
        "//src/numba_enzyme:numba_enzyme_py",
        "@pypi_llvmlite//:pkg",
        "@pypi_numba//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_wheel(
    name = "wheel",
    abi = "none",
    author = "Yousof Mardoukhi, Enzyme Authors",
    description_file = "README.md",
    distribution = "numba_enzyme",
    homepage = "https://github.com/EnzymeAD/numba-enzyme",
    license = "Apache-2.0 WITH LLVM-exception",
    platform = select({
        "@bazel_tools//src/conditions:darwin_arm64": "macosx_12_0_arm64",
        "@bazel_tools//src/conditions:darwin_x86_64": "macosx_12_0_x86_64",
        "@bazel_tools//src/conditions:linux_aarch64": "manylinux_2_35_aarch64",
        "@bazel_tools//src/conditions:linux_x86_64": "manylinux_2_35_x86_64",
    }),
    project_urls = {
        "Repository": "https://github.com/EnzymeAD/numba-enzyme",
    },
    python_requires = ">=3.11",
    python_tag = "py3",
    requires = [
        "llvmlite>=0.50.0,<0.51",
        "numba>=0.68.0",
    ],
    strip_path_prefixes = ["src/"],
    summary = "Differentiate Numba-compiled Python functions with Enzyme.",
    version = "0.1.3",
    deps = [
        ":numba_enzyme_data",
        "//src/numba_enzyme:numba_enzyme_py",
    ],
)

refresh_compile_commands(
    name = "refresh_compile_commands",
    targets = ["//src/numba_enzyme:libNumbaEnzyme.so"],
)
