"""Dependency-light static audit for the Komorebi Math 1.0 public beta."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verification.block16_static_audit import run_audit as run_foundation_audit


@dataclass(frozen=True, slots=True)
class AuditCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class StaticAuditReport:
    package_version: str
    root: str
    checks: tuple[AuditCheck, ...]
    diagnostics: dict[str, Any]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["passed"] = self.passed
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        payload["sha256"] = sha256(canonical.encode("utf-8")).hexdigest()
        return payload


def _capture(name: str, action) -> AuditCheck:
    try:
        detail = str(action())
    except Exception as exc:
        return AuditCheck(name, False, f"{type(exc).__name__}: {exc}")
    return AuditCheck(name, True, detail)


def run_audit(root: Path | None = None) -> StaticAuditReport:
    project = (root or Path(__file__).resolve().parents[1]).resolve()
    foundation = run_foundation_audit(project)
    data = project / "src" / "komorebi_math" / "data"

    pyproject_text = (project / "pyproject.toml").read_text(encoding="utf-8")
    version_match = re.search(r'^version\s*=\s*"([^"]+)"$', pyproject_text, re.MULTILINE)
    if version_match is None:
        raise AssertionError("pyproject version is missing")
    current_version = version_match.group(1)

    def foundation_gate() -> str:
        if not foundation.passed:
            failures = [check.name for check in foundation.checks if not check.passed]
            raise AssertionError(f"foundation static audit failed: {failures}")
        return f"{len(foundation.checks)} foundation checks passed"

    def current_defaults() -> str:
        required = {
            project / "src/komorebi_math/compatibility.py": (
                "public_api_1_0.json",
                "public_api_beta_1_0.json",
                "public_api_stable_1_0.json",
            ),
            project / "src/komorebi_math/persistence_contract.py": (
                "portable_codec_contract_1_0.json",
            ),
            project / "src/komorebi_math/stable_semantics.py": (
                "stable_semantics_1_0.json",
            ),
            project / "src/komorebi_math/support.py": ("support_matrix_1_0.json",),
            project / "src/komorebi_math/release/contract.py": ("release_contract_1_0.json",),
        }
        missing: list[str] = []
        for path, terms in required.items():
            text = path.read_text(encoding="utf-8")
            missing.extend(f"{path.name}: {term}" for term in terms if term not in text)
        if missing:
            raise AssertionError(f"missing current contract defaults: {missing}")
        return "all current default contract paths target the 1.0 public-beta artifacts"

    def beta_version_literals() -> str:
        pyproject = pyproject_text
        version_py = (project / "src/komorebi_math/_version.py").read_text(encoding="utf-8")
        if not re.fullmatch(r"1\.0\.0b[1-9]\d*", current_version):
            raise AssertionError(f"pyproject version is not a 1.0 beta: {current_version}")
        if f'__version__ = "{current_version}"' not in version_py:
            raise AssertionError("runtime and build versions differ")
        if "Development Status :: 4 - Beta" not in pyproject:
            raise AssertionError("beta classifier is absent")
        return f"runtime, build metadata, and classifier identify {current_version}"

    def contract_json() -> str:
        names = (
            "public_api_1_0.json",
            "public_api_beta_1_0.json",
            "public_api_stable_1_0.json",
            "stable_semantics_1_0.json",
            "portable_codec_contract_1_0.json",
            "upgrade_contract_1_0.json",
            "validation_evidence_1_0.json",
            "stable_documentation_1_0.json",
            "support_matrix_1_0.json",
            "release_contract_1_0.json",
        )
        for name in names:
            json.loads((data / name).read_text(encoding="utf-8"))
        return f"{len(names)} Block 17 contract JSON files parsed"

    checks = (
        _capture("foundation static audit", foundation_gate),
        _capture("current contract defaults", current_defaults),
        _capture("public-beta version literals", beta_version_literals),
        _capture("Block 17 contract JSON", contract_json),
    )
    diagnostics = {
        "foundation": foundation.to_dict(),
        "block17_contract_files": 10,
    }
    return StaticAuditReport(
        package_version=current_version,
        root=str(project),
        checks=checks,
        diagnostics=diagnostics,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    report = run_audit()
    text = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8", newline="\n")
    else:
        print(text, end="")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
