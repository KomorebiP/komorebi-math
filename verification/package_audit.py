"""Deterministic package-contract and source-layout audit for Komorebi Math."""

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
            "schema": "komorebi-math.package-audit",
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
            "# Package audit",
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
        check_release_contract,
        check_stable_documentation,
        check_upgrade_contract,
        check_validation_evidence,
    )

    from komorebi_math.qualification import run_release_qualification
    from komorebi_math.release.evidence import _check_historical_validation_evidence
    from verification.source_layout import source_files

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
        "README.md", "LICENSE.md", "pyproject.toml", "MANIFEST.in", "mkdocs.yml",
        "docs/USER_GUIDE.md", "docs/STABLE_API.md", "docs/RELEASE_PROCESS.md",
        "verification/build_distributions.py", "verification/build_source_release.py",
        "verification/check_release.py", "verification/source_audit.py",
        "verification/package_audit.py", "verification/source_layout.py",
        "src/komorebi_math/release/contract.py", "src/komorebi_math/release/evidence.py",
        "src/komorebi_math/release/documentation.py", "src/komorebi_math/release/upgrade.py",
    )
    publication_diagnostics: dict[str, Any] = {}


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
            "evidence_records": len(evidence["records"]),
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
        _check_historical_validation_evidence().certificate.require()
        check_stable_documentation().certificate.require()
        functional = run_release_qualification()
        if not functional.passed:
            failed = [check.name for check in functional.verification.checks if not check.passed]
            raise AssertionError(f"installed functionality failed: {failed}")
        return "release, upgrade, historical-record integrity, documentation and installed functionality passed"

    def publication_boundary() -> str:
        current = check_validation_evidence(require_current_all_platforms=True)
        registry_is_current = evidence["target_version"] == komorebi_math.__version__
        if current.passed and not registry_is_current:
            raise AssertionError("historical platform records were accepted as current qualification")
        publication_diagnostics.update({
            "registry_target_version": evidence["target_version"],
            "current_platform_evidence_passed": current.passed,
            "external_promotion_review_required": True,
            "gpu_qualification_established_by_this_audit": False,
        })
        return "historical evidence remains distinct from current platform, hardware and tagged promotion review"

    def layout() -> str:
        selected = source_files(project)
        missing = [relative for relative in required_paths if relative not in selected]
        if missing:
            raise AssertionError(f"missing current source files: {missing}")
        historical = [name for name in selected if name.startswith(("release/", "development/"))]
        if historical:
            raise AssertionError(f"historical or generated files selected for distribution: {historical}")
        return f"{len(required_paths)} required source files selected; generated history excluded"

    def support_policy() -> str:
        if support["release_line"] != release_contract["release_line"]:
            raise AssertionError("support and release-contract lines differ")
        if support["python"]["declared"] != ["3.11", "3.12", "3.13"]:
            raise AssertionError("declared Python matrix differs")
        if not support["validated_environments"]:
            raise AssertionError("support matrix must retain its explicitly scoped execution records")
        return "Python 3.11-3.13 declared; recorded environments retain their own versions and scope"

    checks = (
        _capture("version identity", versions),
        _capture("contract inventories", inventories),
        _capture("stable API freeze", stable_freeze),
        _capture("runtime contracts", runtime_contracts),
        _capture("publication evidence boundary", publication_boundary),
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
        "publication_evidence": publication_diagnostics,
        "release_contract_sha256": sha256((data / "release_contract_1_0.json").read_bytes()).hexdigest(),
    }
    return PackageAuditReport(
        package_version=komorebi_math.__version__,
        release_stage=str(release_contract["stage"]),
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
