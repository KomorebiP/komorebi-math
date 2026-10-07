"""Deterministic source-tree audit for the Block 16 freeze candidate."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PackageAuditCheck:
    """One deterministic package-audit result."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class PackageAuditReport:
    """Portable source-tree audit for the Block 16 release candidate."""

    package_version: str
    release_stage: str
    checks: tuple[PackageAuditCheck, ...]
    diagnostics: dict[str, Any]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": "komorebi-math.block16-package-audit",
            "schema_version": 1,
            "package_version": self.package_version,
            "release_stage": self.release_stage,
            "passed": self.passed,
            "checks": [asdict(check) for check in self.checks],
            "diagnostics": self.diagnostics,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        payload["sha256"] = sha256(canonical.encode("utf-8")).hexdigest()
        return payload

    def to_markdown(self) -> str:
        state = "passed" if self.passed else "failed"
        lines = [
            "# Block 16 package audit",
            "",
            f"Status: **{state}**",
            "",
            f"Package version: `{self.package_version}`  ",
            f"Release stage: `{self.release_stage}`",
            "",
            "## Checks",
            "",
        ]
        for check in self.checks:
            mark = "PASS" if check.passed else "FAIL"
            lines.append(f"- **{mark} — {check.name}:** {check.detail}")
        lines.extend(["", "## Inventory", ""])
        for key, value in self.diagnostics.items():
            rendered = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
            lines.append(f"- `{key}`: {rendered}")
        lines.append("")
        return "\n".join(lines)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _version_from_pyproject(path: Path) -> str:
    match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', path.read_text(encoding="utf-8"), re.MULTILINE)
    if match is None:
        raise ValueError("project version is missing from pyproject.toml")
    return match.group(1)


def _capture(name: str, action) -> PackageAuditCheck:
    try:
        detail = str(action())
    except Exception as exc:  # release audit must report rather than hide a defect
        return PackageAuditCheck(name, False, f"{type(exc).__name__}: {exc}")
    return PackageAuditCheck(name, True, detail)


def run_audit(root: Path | None = None) -> PackageAuditReport:
    """Audit the current source tree without depending on pre-existing release logs."""

    project = (root or Path(__file__).resolve().parents[1]).resolve()
    source = project / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

    import komorebi_math
    from komorebi_math.stable_contract import check_stable_contract

    data = source / "komorebi_math" / "data"
    research = _load(data / "public_api_0_16.json")
    beta = _load(data / "public_api_beta_0_16.json")
    stable = _load(data / "public_api_stable_0_16.json")
    semantics = _load(data / "stable_semantics_0_16.json")
    persistence = _load(data / "portable_codec_contract_0_16.json")
    support = _load(data / "support_matrix_0_16.json")
    release_contract = _load(data / "release_contract_0_16.json")
    crosswalk = _load(project / "verification" / "COMPANION_C_BLOCK16_CROSSWALK.json")

    required_paths = (
        ".github/workflows/ci.yml",
        "README_BLOCK16.md",
        "Run_Block16_Source_Validation.ps1",
        "Install_And_Verify_Block16.ps1",
        "Run_GPU_Validation.ps1",
        "Build_Block16_Distributions.ps1",
        "run_block16_source_validation.sh",
        "install_and_verify_block16.sh",
        "run_gpu_validation.sh",
        "build_block16_distributions.sh",
        "docs/ANALYSIS_SYNTHESIS_SYSTEMS.md",
        "docs/BLOCK_16_RELEASE.md",
        "docs/MIGRATION_TO_1_0.md",
        "docs/RELEASE_QUALIFICATION.md",
        "docs/STABLE_API_1_0_CANDIDATE.md",
        "docs/SUPPORTED_PLATFORMS.md",
        "examples/block16_analysis_synthesis.py",
        "verification/run_isolated_test_batches.py",
        "verification/block16_distribution_audit.py",
        "verification/block16_static_audit.py",
        "src/komorebi_math/stable_contract.py",
        "src/komorebi_math/persistence_contract.py",
        "src/komorebi_math/stable_semantics.py",
        "src/komorebi_math/support.py",
        "src/komorebi_math/qualification.py",
    )

    def versions() -> str:
        pyproject = _version_from_pyproject(project / "pyproject.toml")
        versions = {
            komorebi_math.__version__,
            pyproject,
            research["package_version"],
            beta["package_version"],
            stable["package_version"],
            persistence["package_version"],
            release_contract["package_version"],
            crosswalk["package_version"],
        }
        if versions != {"0.16.0a2"}:
            raise AssertionError(f"inconsistent versions: {sorted(versions)}")
        return "all source and contract versions are 0.16.0a2"

    def api_counts() -> str:
        observed = {
            "research": len(research["symbols"]),
            "beta": len(beta["symbols"]),
            "stable": len(stable["symbols"]),
        }
        expected = {name: int(spec["count"]) for name, spec in release_contract["api"].items() if name in observed}
        if observed != expected:
            raise AssertionError(f"API counts differ: observed={observed}, expected={expected}")
        if observed != {"research": 1179, "beta": 83, "stable": 131}:
            raise AssertionError(f"unexpected Block 16 API counts: {observed}")
        return f"research={observed['research']}, beta={observed['beta']}, stable={observed['stable']}"

    def semantic_and_codec_counts() -> str:
        observed = {"semantics": len(semantics["cases"]), "portable_codecs": len(persistence["entries"])}
        expected = {
            "semantics": int(release_contract["semantics"]["count"]),
            "portable_codecs": int(release_contract["persistence"]["count"]),
        }
        if observed != expected:
            raise AssertionError(f"contract count mismatch: observed={observed}, expected={expected}")
        return f"semantic cases={observed['semantics']}, portable codecs={observed['portable_codecs']}"

    def required_layout() -> str:
        missing = [relative for relative in required_paths if not (project / relative).is_file()]
        if missing:
            raise AssertionError(f"missing Block 16 files: {missing}")
        return f"{len(required_paths)} required files present"

    def current_helpers() -> str:
        paths = (
            project / "examples" / "Test-KomorebiEverything.ps1",
            project / "README_BLOCK16.md",
            project / "Run_Block16_Source_Validation.ps1",
            project / "Install_And_Verify_Block16.ps1",
            project / "Build_Block16_Distributions.ps1",
        )
        forbidden = (
            "komorebi_math-0.15.0a2-py3-none-any.whl",
            "komorebi_math-0.15.0a2.tar.gz",
            "run_block15_test_batches.py",
            'package_version = "0.15.0a2"',
        )
        failures: list[str] = []
        for path in paths:
            text = path.read_text(encoding="utf-8")
            failures.extend(f"{path.relative_to(project)}: {term}" for term in forbidden if term in text)
        if failures:
            raise AssertionError(f"stale current-release references: {failures}")
        return f"{len(paths)} current helper files are version-consistent"

    def aliases() -> str:
        from komorebi_math import stable, v1
        from komorebi_math.api import v1 as api_v1

        if stable.__all__ != v1.__all__ or stable.__all__ != api_v1.__all__:
            raise AssertionError("versioned stable aliases expose different names")
        for name in stable.__all__:
            if getattr(stable, name) is not getattr(v1, name) or getattr(stable, name) is not getattr(api_v1, name):
                raise AssertionError(f"stable alias identity differs for {name}")
        return f"three aliases share {len(stable.__all__)} object-identical exports"

    def contract() -> str:
        report = check_stable_contract()
        report.certificate.require()
        return "structural, semantic, and persistence layers passed"

    def support_shape() -> str:
        if support["python"]["declared"] != ["3.11", "3.12", "3.13"]:
            raise AssertionError("declared Python matrix changed")
        if not support["validated_environments"]:
            raise AssertionError("support matrix lacks recorded validation evidence")
        if "macOS" not in support["unqualified_platforms"]:
            raise AssertionError("unqualified platforms must remain explicit")
        return "Python 3.11-3.13 declared; qualified and unqualified environments are explicit"

    def companion() -> str:
        source_archive = crosswalk["source_archive"]
        digest = str(source_archive["sha256"])
        if len(digest) != 64 or len(crosswalk["entries"]) != 8:
            raise AssertionError("Companion C crosswalk is incomplete")
        return f"8 reviewed overlaps tied to source digest {digest[:12]}…"

    checks = (
        _capture("version consistency", versions),
        _capture("API inventory consistency", api_counts),
        _capture("semantic and persistence inventory", semantic_and_codec_counts),
        _capture("Block 16 release layout", required_layout),
        _capture("current helper version hygiene", current_helpers),
        _capture("versioned stable alias identity", aliases),
        _capture("complete stable contract", contract),
        _capture("support policy shape", support_shape),
        _capture("Companion C crosswalk", companion),
    )

    diagnostics = {
        "production_python_files": len(list((source / "komorebi_math").rglob("*.py"))),
        "test_files": len(list((project / "tests").glob("test_*.py"))),
        "documentation_markdown_files": len(list((project / "docs").glob("*.md"))),
        "example_python_files": len(list((project / "examples").glob("*.py"))),
        "verification_python_files": len(list((project / "verification").glob("*.py"))),
        "public_api_symbol_counts": {
            "research": len(research["symbols"]),
            "beta": len(beta["symbols"]),
            "stable": len(stable["symbols"]),
        },
        "stable_semantic_cases": len(semantics["cases"]),
        "portable_codec_entries": len(persistence["entries"]),
        "declared_python": support["python"]["declared"],
        "validated_environment_ids": [item["identifier"] for item in support["validated_environments"]],
    }
    return PackageAuditReport(
        package_version=komorebi_math.__version__,
        release_stage="1.0 freeze candidate",
        checks=checks,
        diagnostics=diagnostics,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args(argv)
    report = run_audit(args.root)
    payload = report.to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    print(text, end="")
    if args.output_json:
        args.output_json.write_text(text, encoding="utf-8")
    if args.output_markdown:
        args.output_markdown.write_text(report.to_markdown(), encoding="utf-8")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
