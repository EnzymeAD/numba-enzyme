workspace(name = "numba_enzyme")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

http_archive(
    name = "platforms",
    sha256 = "218efe8ee736d26a3572663b374a253c012b716d8af0c07e842e82f238a0a7ee",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/0.0.10/platforms-0.0.10.tar.gz",
        "https://github.com/bazelbuild/platforms/releases/download/0.0.10/platforms-0.0.10.tar.gz",
    ],
)

load("@platforms//host:extension.bzl", "host_platform_repo")

maybe(
    host_platform_repo,
    name = "host_platform",
)

http_archive(
    name = "bazel_skylib",
    sha256 = "3b5b49006181f5f8ff626ef8ddceaa95e9bb8ad294f7b5d7b11ea9f7ddaf8c59",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.9.0/bazel-skylib-1.9.0.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.9.0/bazel-skylib-1.9.0.tar.gz",
    ],
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

http_archive(
    name = "bazel_features",
    sha256 = "07271d0f6b12633777b69020c4cb1eb67b1939c0cf84bb3944dc85cc250c0c01",
    strip_prefix = "bazel_features-1.38.0",
    url = "https://github.com/bazel-contrib/bazel_features/releases/download/v1.38.0/bazel_features-v1.38.0.tar.gz",
)

load("@bazel_features//:deps.bzl", "bazel_features_deps")

bazel_features_deps()

http_archive(
    name = "rules_cc",
    sha256 = "b8b918a85f9144c01f6cfe0f45e4f2838c7413961a8ff23bc0c6cdf8bb07a3b6",
    strip_prefix = "rules_cc-0.1.5",
    urls = ["https://github.com/bazelbuild/rules_cc/releases/download/0.1.5/rules_cc-0.1.5.tar.gz"],
)

http_archive(
    name = "rules_shell",
    sha256 = "e6b87c89bd0b27039e3af2c5da01147452f240f75d505f5b6880874f31036307",
    strip_prefix = "rules_shell-0.6.1",
    url = "https://github.com/bazelbuild/rules_shell/releases/download/v0.6.1/rules_shell-v0.6.1.tar.gz",
)

load("@rules_shell//shell:repositories.bzl", "rules_shell_dependencies", "rules_shell_toolchains")

rules_shell_dependencies()

rules_shell_toolchains()

http_archive(
    name = "build_bazel_apple_support",
    sha256 = "a270f034007ab6111645d27a8ed88b47ce2a75b608af633eeca889cb5e07ae0d",
    url = "https://github.com/bazelbuild/apple_support/releases/download/1.24.1/apple_support.1.24.1.tar.gz",
)

load("@build_bazel_apple_support//lib:repositories.bzl", "apple_support_dependencies")

apple_support_dependencies()

http_archive(
    name = "rules_python",
    sha256 = "4765b9a4a9c6e0fb89f465aac5b5b0ac06589c9477144340859310777ad801a7",
    strip_prefix = "rules_python-2.4.6",
    url = "https://github.com/bazel-contrib/rules_python/releases/download/2.4.6/rules_python-2.4.6.tar.gz",
)

load("@rules_python//python:repositories.bzl", "py_repositories", "python_register_toolchains")

py_repositories()

http_archive(
    name = "rules_multitool",
    sha256 = "6d0d9fb231eb450ffb3acec879a250e9dea69c7acfc94f138d53103aa5123f08",
    strip_prefix = "rules_multitool-1.11.1",
    url = "https://github.com/bazel-contrib/rules_multitool/releases/download/v1.11.1/rules_multitool-1.11.1.tar.gz",
)

load("@rules_multitool//multitool:multitool.bzl", "multitool")

multitool(
    name = "multitool",
    lockfile = "//:multitool.lock.json",
)

load("@multitool//:tools.bzl", "register_tools")

register_tools()

http_archive(
    name = "rules_uv",
    sha256 = "3aff4af8ca72a7bb5978f7f501b407f1909dd1952c4e6eedac066f4911f5dcf8",
    strip_prefix = "rules_uv-0.89.2",
    url = "https://github.com/bazel-contrib/rules_uv/releases/download/v0.89.2/rules_uv-0.89.2.tar.gz",
)

load("//third_party/py:python_repo.bzl", "python_init_repositories")

python_init_repositories(
    requirements = {
        "3.11": "//builddeps:requirements_lock_3_11.txt",
        "3.12": "//builddeps:requirements_lock_3_12.txt",
        "3.13": "//builddeps:requirements_lock_3_13.txt",
        "3.14": "//builddeps:requirements_lock_3_14.txt",
        "3.15": "//builddeps:requirements_lock_3_15.txt",
    },
)

load(
    "@python_version_repo//:py_version.bzl",
    "HERMETIC_PYTHON_VERSION",
    "PYTHON_TOOLCHAIN_NAME",
    "REQUIREMENTS",
)

python_register_toolchains(
    name = PYTHON_TOOLCHAIN_NAME,
    ignore_root_user_error = True,
    python_version = HERMETIC_PYTHON_VERSION,
)

load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "pypi",
    envsubst = ["RULES_PYTHON_ENABLE_PIPSTAR"],
    python_interpreter_target = "@{}_host//:python".format(PYTHON_TOOLCHAIN_NAME),
    requirements_lock = REQUIREMENTS,
)

load("@pypi//:requirements.bzl", "install_deps")

install_deps()

# LLVM 22.1 (matching llvmlite 0.50.x)
load("//:workspace.bzl", "LLVM_SHA256", "LLVM_TARGETS", "LLVM_VERSION")

http_archive(
    name = "llvm-raw",
    build_file_content = "# empty",
    patch_args = ["-p1"],
    patches = ["//patches:llvm_config.patch"],
    sha256 = LLVM_SHA256,
    strip_prefix = "llvm-project-" + LLVM_VERSION + ".src",
    urls = [
        "https://github.com/llvm/llvm-project/releases/download/llvmorg-{v}/llvm-project-{v}.src.tar.xz".format(v = LLVM_VERSION),
    ],
)

maybe(
    http_archive,
    name = "llvm_zlib",
    build_file = "@llvm-raw//utils/bazel/third_party_build:zlib-ng.BUILD",
    sha256 = "e36bb346c00472a1f9ff2a0a4643e590a254be6379da7cddd9daeb9a7f296731",
    strip_prefix = "zlib-ng-2.0.7",
    urls = [
        "https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.0.7.zip",
    ],
)

maybe(
    http_archive,
    name = "llvm_zstd",
    build_file = "@llvm-raw//utils/bazel/third_party_build:zstd.BUILD",
    sha256 = "7c42d56fac126929a6a85dbc73ff1db2411d04f104fae9bdea51305663a83fd0",
    strip_prefix = "zstd-1.5.2",
    urls = [
        "https://github.com/facebook/zstd/releases/download/v1.5.2/zstd-1.5.2.tar.gz",
    ],
)

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")

llvm_configure(
    name = "llvm-project",
    targets = LLVM_TARGETS,
)

load("//third_party/enzyme:workspace.bzl", enzyme_workspace = "repo")

enzyme_workspace()

load(
    "@enzyme//third_party/hedron_compile_commands:workspace.bzl",
    hedron_compile_commands_workspace = "repo",
)

hedron_compile_commands_workspace()

load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")
load("@hedron_compile_commands//:workspace_setup_transitive.bzl", "hedron_compile_commands_setup_transitive")
load("@hedron_compile_commands//:workspace_setup_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive")
load("@hedron_compile_commands//:workspace_setup_transitive_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive_transitive")

hedron_compile_commands_setup()

hedron_compile_commands_setup_transitive()

hedron_compile_commands_setup_transitive_transitive()

hedron_compile_commands_setup_transitive_transitive_transitive()
