load("@rules_python//python:defs.bzl", "py_test")

TEST_SRCS = glob(
    ["test_*.py"],
)

[
    py_test(
        name = src[:-3],
        srcs = [
            "pytest_main.py",
            src,
        ],
        args = [
            "-v",
            "$(location %s)" % src,
        ],
        data = ["//:pyproject.toml"] if src == "test_smoke.py" else [],
        main = "pytest_main.py",
        size = "small",
        deps = [
            "//:numba_enzyme",
            "@pypi_numpy//:pkg",
            "@pypi_pytest//:pkg",
        ],
    )
    for src in TEST_SRCS
]

test_suite(
    name = "python_tests",
    tests = [":%s" % src[:-3] for src in TEST_SRCS],
)
