load("@python_version_repo//:py_version.bzl", "REQUIREMENTS")
load("@rules_uv//uv:pip.bzl", "pip_compile")

exports_files([
    "requirements.in",
    "requirements_lock_3_11.txt",
    "requirements_lock_3_12.txt",
    "requirements_lock_3_13.txt",
    "requirements_lock_3_14.txt",
    "requirements_lock_3_15.txt",
])

pip_compile(
    name = "requirements",
    extra_args = [
        "--universal",
        "--resolution=highest",
    ],
    requirements_in = "requirements.in",
    requirements_txt = REQUIREMENTS,
)
