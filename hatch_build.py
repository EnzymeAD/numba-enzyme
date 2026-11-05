"""
Conditionally bundle pre-built `libNumbaEnzyme` into Hatch-built wheels.

Release wheels are built with Bazel (`bazel build //:wheel`). For local
Hatch/uv wheel builds, if `src/numba_enzyme/_vendor/` contains a pre-built
`libNumbaEnzyme` shared library, this hook includes it dynamically and marks
the wheel as platform-specific while leaving editable source installs pure.
"""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class VendorBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        vendor_dir = Path(self.root) / "src" / "numba_enzyme" / "_vendor"
        if not vendor_dir.is_dir():
            return

        files = [p for p in sorted(vendor_dir.rglob("*")) if p.is_file()]
        if not files:
            return

        force_include = build_data.setdefault("force_include", {})
        src_root = Path(self.root) / "src"
        for path in files:
            force_include[str(path)] = str(path.relative_to(src_root))

        build_data["pure_python"] = False
        build_data["infer_tag"] = True
