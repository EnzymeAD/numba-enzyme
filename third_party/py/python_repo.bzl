"""Repository rule to configure hermetic Python version and lockfile."""

DEFAULT_PYTHON_VERSION = "3.11"

def _python_version_repo_impl(ctx):
    version = ctx.os.environ.get("HERMETIC_PYTHON_VERSION", "")
    if not version:
        version = ctx.attr.default_python_version

    requirements = None
    for i, ver in enumerate(ctx.attr.requirements_versions):
        if ver == version:
            requirements = ctx.attr.requirements_locks[i]
            break

    if not requirements:
        fail(
            "Unsupported HERMETIC_PYTHON_VERSION={version!r}; supported versions: {supported}".format(
                version = version,
                supported = ", ".join(ctx.attr.requirements_versions),
            ),
        )

    toolchain_name = "python_" + version.replace(".", "_")
    ctx.file("BUILD", "")
    ctx.file(
        "py_version.bzl",
        """\
HERMETIC_PYTHON_VERSION = "{version}"
PYTHON_TOOLCHAIN_NAME = "{toolchain_name}"
REQUIREMENTS = "{requirements}"
""".format(
            version = version,
            toolchain_name = toolchain_name,
            requirements = str(requirements),
        ),
    )

python_version_repository = repository_rule(
    implementation = _python_version_repo_impl,
    attrs = {
        "default_python_version": attr.string(default = DEFAULT_PYTHON_VERSION),
        "requirements_locks": attr.label_list(default = []),
        "requirements_versions": attr.string_list(default = []),
    },
    environ = ["HERMETIC_PYTHON_VERSION"],
    local = True,
)

def python_init_repositories(requirements = {}, default_python_version = DEFAULT_PYTHON_VERSION):
    python_version_repository(
        name = "python_version_repo",
        default_python_version = default_python_version,
        requirements_locks = requirements.values(),
        requirements_versions = requirements.keys(),
    )
