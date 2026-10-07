"""Deterministic source-tree audit for the Komorebi Math 1.0 public beta."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class PackageAuditCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class PackageAuditReport:
    package_version: str
    release_stage: str
    checks: tuple[PackageAuditCheck, ...]
    diagnostics: dict[str, Any]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": "komorebi-math.block17-package-audit",
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
            "# Block 17 package audit",
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
            marker = "PASS" if check.passed else "FAIL"
            lines.append(f"- **{marker} — {check.name}:** {check.detail}")
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


def _capture(name: str, action: Callable[[], str]) -> PackageAuditCheck:
    try:
        detail = str(action())
    except Exception as exc:
        return PackageAuditCheck(name, False, f"{type(exc).__name__}: {exc}")
    return PackageAuditCheck(name, True, detail)


def _symbol_identity(payload: dict[str, Any]) -> dict[tuple[str, str], tuple[Any, ...]]:
    return {
        (str(symbol["namespace"]), str(symbol["name"])): (
            symbol["kind"],
            symbol["implementation_module"],
            symbol["qualname"],
            symbol["signature"],
        )
        for symbol in payload["symbols"]
    }


def run_audit(root: Path | None = None) -> PackageAuditReport:
    project = (root or Path(__file__).resolve().parents[1]).resolve()
    source = project / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

    import komorebi_math
    from komorebi_math.release import (
        check_beta_readiness,
        check_release_contract,
        check_stable_documentation,
        check_upgrade_contract,
        check_validation_evidence,
    )

    data = source / "komorebi_math" / "data"
    research = _load(data / "public_api_1_0.json")
    beta = _load(data / "public_api_beta_1_0.json")
    stable = _load(data / "public_api_stable_1_0.json")
    predecessor_stable = _load(data / "public_api_stable_0_16.json")
    semantics = _load(data / "stable_semantics_1_0.json")
    persistence = _load(data / "portable_codec_contract_1_0.json")
    upgrade = _load(data / "upgrade_contract_1_0.json")
    evidence = _load(data / "validation_evidence_1_0.json")
    documentation = _load(data / "stable_documentation_1_0.json")
    support = _load(data / "support_matrix_1_0.json")
    release_contract = _load(data / "release_contract_1_0.json")

    required_paths = (
        "README_BLOCK17.md",
        "Run_Block17_Source_Validation.ps1",
        "Run_Block17_Upgrade_Validation.ps1",
        "Run_Block17_GPU_Validation.ps1",
        "Install_And_Verify_Block17.ps1",
        "Build_Block17_Distributions.ps1",
        "run_block17_source_validation.sh",
        "run_block17_upgrade_validation.sh",
        "run_block17_gpu_validation.sh",
        "install_and_verify_block17.sh",
        "build_block17_distributions.sh",
        "docs/BLOCK_17_RELEASE.md",
        "docs/UPGRADE_CONTRACT.md",
        "docs/BETA_TO_RC.md",
        "verification/block17_static_audit.py",
        "verification/block17_distribution_audit.py",
        "verification/build_block17_distributions.py",
        "verification/run_block17_upgrade_validation.py",
        "src/komorebi_math/release/upgrade.py",
        "src/komorebi_math/release/evidence.py",
        "src/komorebi_math/release/documentation.py",
        "src/komorebi_math/release/readiness.py",
    )

    def versions() -> str:
        versions = {
            komorebi_math.__version__,
            _version_from_pyproject(project / "pyproject.toml"),
            research["package_version"],
            beta["package_version"],
            stable["package_version"],
            persistence["package_version"],
            release_contract["package_version"],
            upgrade["target_version"],
            evidence["target_version"],
            documentation["package_version"],
        }
        if versions != {komorebi_math.__version__}:
            raise AssertionError(f"inconsistent current versions: {sorted(versions)}")
        return f"all current source and contract identities are {komorebi_math.__version__}"

    def inventories() -> str:
        observed = {
            "research": len(research["symbols"]),
            "beta": len(beta["symbols"]),
            "stable": len(stable["symbols"]),
            "semantics": len(semantics["cases"]),
            "portable_codecs": len(persistence["entries"]),
            "upgrade_fixtures": len(upgrade["fixtures"]),
            "evidence_records": len(evidence["records"]),
            "documentation_entries": len(documentation["symbols"]),
        }
        expected = {
            "research": len(research["symbols"]),
            "beta": 83,
            "stable": 131,
            "semantics": 10,
            "portable_codecs": 51,
            "upgrade_fixtures": 7,
            "evidence_records": 2,
            "documentation_entries": 131,
        }
        if observed != expected:
            raise AssertionError(f"inventory differs: observed={observed}, expected={expected}")
        return ", ".join(f"{key}={value}" for key, value in observed.items())

    def stable_freeze() -> str:
        old = _symbol_identity(predecessor_stable)
        new = _symbol_identity(stable)
        if old != new:
            added = sorted(set(new) - set(old))
            removed = sorted(set(old) - set(new))
            changed = sorted(key for key in set(old) & set(new) if old[key] != new[key])
            raise AssertionError(f"stable surface changed: added={added}, removed={removed}, changed={changed}")
        return "131 stable exports are exactly unchanged from 0.16.0a2"

    def runtime_contracts() -> str:
        check_release_contract().certificate.require()
        check_upgrade_contract().certificate.require()
        check_validation_evidence().certificate.require()
        check_stable_documentation().certificate.require()
        check_beta_readiness().certificate.require()
        return "release, upgrade, evidence, documentation, and beta readiness passed"

    def beta_rc_boundary() -> str:
        beta_report = check_validation_evidence()
        rc_report = check_validation_evidence(require_current_all_platforms=True)
        if not beta_report.passed:
            raise AssertionError("beta evidence gate failed")
        if rc_report.passed or rc_report.missing_current_platforms != ("windows-x86_64",):
            raise AssertionError("release-candidate boundary is not explicit")
        return f"beta evidence passes; exact Windows {komorebi_math.__version__} evidence remains required for RC"

    def layout() -> str:
        missing = [relative for relative in required_paths if not (project / relative).is_file()]
        if missing:
            raise AssertionError(f"missing Block 17 files: {missing}")
        return f"{len(required_paths)} required Block 17 files present"

    def support_policy() -> str:
        if support["release_line"] != "1.0.0b1 to 1.0.0rc1":
            raise AssertionError("support release line differs")
        if support["python"]["declared"] != ["3.11", "3.12", "3.13"]:
            raise AssertionError("declared Python matrix differs")
        if len(support["validated_environments"]) != 2:
            raise AssertionError("support matrix must name current Linux and inherited Windows evidence")
        return "Python 3.11-3.13 declared with explicit qualified and unqualified platforms"

    checks = (
        _capture("version identity", versions),
        _capture("contract inventories", inventories),
        _capture("stable API freeze", stable_freeze),
        _capture("runtime contracts", runtime_contracts),
        _capture("beta to RC boundary", beta_rc_boundary),
        _capture("required release layout", layout),
        _capture("support policy", support_policy),
    )
    diagnostics = {
        "public_api_symbol_counts": {
            "research": len(research["symbols"]),
            "beta": len(beta["symbols"]),
            "stable": len(stable["symbols"]),
        },
        "stable_semantic_cases": len(semantics["cases"]),
        "portable_codec_entries": len(persistence["entries"]),
        "upgrade_fixtures": len(upgrade["fixtures"]),
        "validation_evidence_records": len(evidence["records"]),
        "stable_documentation_entries": len(documentation["symbols"]),
        "release_contract_sha256": sha256((data / "release_contract_1_0.json").read_bytes()).hexdigest(),
    }
    return PackageAuditReport(
        package_version=komorebi_math.__version__,
        release_stage="public-beta",
        checks=checks,
        diagnostics=diagnostics,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json")
    parser.add_argument("--markdown")
    args = parser.parse_args(argv)
    report = run_audit()
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.json:
        Path(args.json).write_text(payload, encoding="utf-8", newline="\n")
    else:
        print(payload, end="")
    if args.markdown:
        Path(args.markdown).write_text(report.to_markdown(), encoding="utf-8", newline="\n")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
