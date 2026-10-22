"""External reproducibility and publication gate for Komorebi Math.

This gate complements the domain verification suites.  It checks the live
release contracts, executes a representative public capability tour, validates
repository-facing metadata, and can run the complete source test suite and a
strict MkDocs build.  The output is a self-digesting JSON evidence record plus
an optional Markdown report.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.parse import unquote

SCHEMA = "komorebi-math.publication-gate"
SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class GateCheck:
    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PublicationGateReport:
    package_version: str
    root: str
    mode: str
    checks: tuple[GateCheck, ...]
    warnings: tuple[str, ...]
    environment: dict[str, Any]
    capability_record: dict[str, Any]
    full_test_record: dict[str, Any] | None
    generated_at_utc: str

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "package_version": self.package_version,
            "root": self.root,
            "mode": self.mode,
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
            "warnings": list(self.warnings),
            "environment": self.environment,
            "capability_record": self.capability_record,
            "full_test_record": self.full_test_record,
            "generated_at_utc": self.generated_at_utc,
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        payload["report_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return payload

    def to_markdown(self) -> str:
        state = "PASSED" if self.passed else "FAILED"
        lines = [
            "# Release Verification",
            "",
            f"Status: **{state}**",
            "",
            f"Package version: `{self.package_version}`  ",
            f"Mode: `{self.mode}`  ",
            f"Generated: `{self.generated_at_utc}`",
            "",
            "## Checks",
            "",
            "| Check | Result | Detail |",
            "|---|---:|---|",
        ]
        for check in self.checks:
            marker = "PASS" if check.passed else "FAIL"
            detail = check.detail.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {check.name} | **{marker}** | {detail} |")
        if self.warnings:
            lines.extend(["", "## Non-gating warnings", ""])
            lines.extend(f"- {warning}" for warning in self.warnings)
        lines.extend(
            [
                "",
                "## Public capability tour",
                "",
                f"Record digest: `{self.capability_record.get('record_sha256', '')}`",
                "",
                "The tour covers polyphase refinement, redundant analysis and synthesis, "
                "Euler–Fibonacci coordinates, exact finite integration, typed expressions, "
                "portable persistence, and the frozen stable API manifest.",
                "",
            ]
        )
        if self.full_test_record is not None:
            lines.extend(
                [
                    "## Full source tests",
                    "",
                    f"- Return code: `{self.full_test_record['returncode']}`",
                    f"- Summary: `{self.full_test_record['summary']}`",
                    "",
                ]
            )
        lines.append(f"Report SHA-256: `{self.to_dict()['report_sha256']}`")
        lines.append("")
        return "\n".join(lines)


def _capture(name: str, action: Callable[[], str]) -> GateCheck:
    try:
        detail = str(action())
    except Exception as exc:
        return GateCheck(name, False, f"{type(exc).__name__}: {exc}")
    return GateCheck(name, True, detail)


def _project_version(root: Path) -> str:
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', text, re.MULTILINE)
    if match is None:
        raise AssertionError("project version is absent from pyproject.toml")
    return match.group(1)


def _local_markdown_link_failures(root: Path) -> list[str]:
    pattern = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
    failures: list[str] = []
    for source in sorted(root.rglob("*.md")):
        if any(part in {".git", "build", "site", ".venv", "development", "qualification", "dist"} for part in source.parts):
            continue
        text = source.read_text(encoding="utf-8", errors="replace")
        for match in pattern.finditer(text):
            raw = match.group(1).strip()
            if raw.startswith("<") and ">" in raw:
                raw = raw[1 : raw.index(">")]
            else:
                raw = raw.split()[0]
            raw = unquote(raw)
            if not raw or raw.startswith(("#", "https://", "http://", "mailto:", "data:")):
                continue
            target = raw.split("#", 1)[0]
            if not target:
                continue
            # Mathematical notation such as ``F[R](z)`` and packet-jet
            # signatures inside fenced examples can resemble Markdown links.
            # Repository-local links in this project are path-like: they carry
            # an extension or a directory component.  Ignore bare symbolic
            # arguments rather than reporting them as missing files.
            destination = (source.parent / target).resolve()
            known_suffixes = {
                ".md", ".rst", ".txt", ".png", ".jpg", ".jpeg", ".gif",
                ".svg", ".pdf", ".html", ".htm", ".json", ".toml",
                ".yaml", ".yml", ".py", ".ps1", ".sh", ".whl",
                ".gz", ".zip", ".cff",
            }
            suffix = PurePosixPath(target).suffix.lower()
            if (
                "/" not in target
                and "\\" not in target
                and suffix not in known_suffixes
                and not destination.exists()
            ):
                continue
            if not destination.exists():
                line = text.count("\n", 0, match.start()) + 1
                failures.append(f"{source.relative_to(root)}:{line} -> {raw}")
    return failures


def _archive_member_is_safe(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and "\\" not in name


def _environment() -> dict[str, Any]:
    import numpy as np

    return {
        "python": platform.python_version(),
        "python_executable": Path(sys.executable).name,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
    }


def _run_full_tests(root: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.pop("PYTHONOPTIMIZE", None)
    environment.pop("PYTEST_ADDOPTS", None)
    environment["PYTHONPATH"] = str(root / "src")
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    from verification.stress_release_candidate import _pytest_counts
    with tempfile.TemporaryDirectory(prefix="komorebi-full-source-") as directory:
        junit_path = Path(directory) / "source.xml"
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=short", f"--junitxml={junit_path}", "tests"],
            cwd=root,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=900,
        )
        junit: dict[str, Any] = {"passed": False}
        try:
            raw_xml = junit_path.read_bytes()
            xml = raw_xml.decode("utf-8")
            # Preserve the exact producer bytes even if concrete-case parsing
            # rejects their format. A failed receipt must remain inspectable.
            junit.update(xml=xml, sha256=hashlib.sha256(raw_xml).hexdigest())
            counts = _pytest_counts(junit_path)
            junit.update(passed=counts["tests"] > 0 and counts["errors"] == counts["failures"] == 0,
                         counts=counts)
        except Exception as exc:
            junit.update(passed=False, error=f"{type(exc).__name__}: {exc}")
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    summary = lines[-1] if lines else "no pytest output"
    return {
        "passed": completed.returncode == 0 and junit["passed"],
        "returncode": completed.returncode,
        "summary": summary,
        "output": completed.stdout,
        "output_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
        "junit": junit,
    }


def _run_mkdocs(root: Path) -> str:
    if importlib.util.find_spec("mkdocs") is None:
        raise RuntimeError("mkdocs is not installed")
    with tempfile.TemporaryDirectory(prefix="komorebi-block21-mkdocs-") as directory:
        completed = subprocess.run(
            [sys.executable, "-m", "mkdocs", "build", "--strict", "--site-dir", directory],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=180,
        )
    if completed.returncode:
        tail = "\n".join(completed.stdout.splitlines()[-20:])
        raise AssertionError(f"strict MkDocs build failed:\n{tail}")
    return "strict MkDocs build passed"


def run_gate(
    root: str | Path | None = None,
    *,
    full_tests: bool = False,
    require_mkdocs: bool = False,
    skip_mkdocs: bool = False,
    installed_artifacts: bool = False,
) -> PublicationGateReport:
    if require_mkdocs and skip_mkdocs:
        raise ValueError("require_mkdocs and skip_mkdocs are mutually exclusive")
    if sys.flags.optimize:
        raise ValueError("qualification requires Python assertions enabled")
    project = (Path(root) if root is not None else Path(__file__).resolve().parents[1]).resolve()
    source = project / "src"
    examples = project / "examples"
    for path in (str(source), str(examples), str(project)):
        if path not in sys.path:
            sys.path.insert(0, path)

    import komorebi_math
    from library_tour import build_record
    from komorebi_math.compatibility import check_beta_api, check_public_api, check_stable_api
    from komorebi_math.release import (
        check_beta_readiness,
        check_release_contract,
        check_upgrade_contract,
        check_validation_evidence,
    )
    from komorebi_math.verify import run_block20_verification, run_stable_verification
    from komorebi_math.release.contract import _version_stage
    from komorebi_math.release.evidence import _check_historical_validation_evidence

    from verification.release_integrity import release_binding
    binding = release_binding(project)
    checks: list[GateCheck] = []
    warnings: list[str] = []
    version = _project_version(project)
    promoted_stage = _version_stage(version) in {"release-candidate", "final"}
    if promoted_stage:
        warnings.append("This source/publication gate checks historical platform-record integrity only. Current candidate CPU/GPU qualification requires release_promotion.py.")

    def metadata_identity() -> str:
        runtime_text = (source / "komorebi_math" / "_version.py").read_text(encoding="utf-8")
        citation_text = (project / "CITATION.cff").read_text(encoding="utf-8")
        assert komorebi_math.__version__ == version
        assert f'__version__ = "{version}"' in runtime_text
        assert re.search(rf"^version:\s*{re.escape(version)}\s*$", citation_text, re.MULTILINE)
        return f"pyproject, runtime, and citation identify {version}"

    checks.append(_capture("version and citation identity", metadata_identity))

    def repository_files() -> str:
        required = (
            ".gitignore",
            "README.md",
            "LICENSE.md",
            "CITATION.cff",
            "SECURITY.md",
            "CONTRIBUTING.md",
            "REPRODUCIBILITY.md",
            "pyproject.toml",
            ".github/workflows/ci.yml",
            ".github/workflows/release.yml",
            "examples/library_tour.py",
            "verification/check_release.py",
            "verification/build_source_release.py",
            "verification/build_distributions.py",
            "verification/source_audit.py",
            "verification/package_audit.py",
        )
        missing = [name for name in required if not (project / name).is_file()]
        if missing:
            raise AssertionError(f"missing publication files: {missing}")
        ignore = (project / ".gitignore").read_text(encoding="utf-8")
        for pattern in ("__pycache__/", "*.py[cod]", ".pytest_cache/", "build/", "dist/", ".venv"):
            if pattern not in ignore:
                raise AssertionError(f".gitignore is missing {pattern!r}")
        return f"{len(required)} publication-facing files present"

    checks.append(_capture("repository publication layout", repository_files))

    def archive_safety() -> str:
        unsafe = [
            str(path.relative_to(project))
            for path in project.rglob("*")
            if not _archive_member_is_safe(path.relative_to(project).as_posix())
        ]
        if unsafe:
            raise AssertionError(f"unsafe relative paths: {unsafe[:10]}")
        return "all source-tree paths are archive safe"

    checks.append(_capture("archive path safety", archive_safety))

    checks.append(
        _capture(
            "strict research API contract",
            lambda: (check_public_api(strict_signatures=True).certificate.require(), "strict API passed")[1],
        )
    )
    checks.append(
        _capture(
            "strict expanded API contract",
            lambda: (check_beta_api().certificate.require(), "expanded API passed")[1],
        )
    )
    checks.append(
        _capture(
            "strict stable API contract",
            lambda: (check_stable_api().certificate.require(), "stable API passed")[1],
        )
    )
    def verification_detail(runner: Callable[[], Any]) -> str:
        report = runner()
        report.certificate().require()
        return f"{report.passed_count}/{report.total_count}"

    checks.append(
        _capture(
            "stable verification suite",
            lambda: verification_detail(run_stable_verification),
        )
    )
    checks.append(
        _capture(
            "typed-expression verification suite",
            lambda: verification_detail(run_block20_verification),
        )
    )
    checks.append(
        _capture(
            "current release contract" if promoted_stage else "current beta release contract",
            lambda: (check_release_contract().certificate.require(), "release contract passed")[1],
        )
    )
    checks.append(
        _capture(
            "cross-version upgrade contract",
            lambda: (check_upgrade_contract().certificate.require(), "upgrade contract passed")[1],
        )
    )
    checks.append(
        _capture(
            "historical platform record integrity (not current qualification)" if promoted_stage else "executed platform evidence",
            lambda: ((_check_historical_validation_evidence() if promoted_stage else check_validation_evidence()).certificate.require(),
                     "historical record integrity passed; external promotion required" if promoted_stage else "beta evidence passed")[1],
        )
    )
    checks.append(
        _capture(
            "source-stage contract readiness (external promotion required)" if promoted_stage else "public beta readiness",
            lambda: ((check_release_contract() if promoted_stage else check_beta_readiness()).certificate.require(),
                     "source contract passed; external promotion required" if promoted_stage else "beta readiness passed")[1],
        )
    )

    capability_record: dict[str, Any] = {}

    def capability_tour() -> str:
        nonlocal capability_record
        capability_record = build_record()
        assert capability_record["passed"] is True
        assert len(capability_record["record_sha256"]) == 64
        return f"capability record {capability_record['record_sha256']}"

    checks.append(_capture("deterministic public capability tour", capability_tour))

    def local_links() -> str:
        failures = _local_markdown_link_failures(project)
        if failures:
            raise AssertionError("broken local Markdown links: " + "; ".join(failures[:20]))
        return "all local Markdown links resolve"

    checks.append(_capture("local documentation links", local_links))

    def artifact_integrity() -> str:
        from verification.release_integrity import verify_release
        result = verify_release(project)
        if not result["passed"]:
            raise AssertionError(json.dumps(result, sort_keys=True))
        return (f"{result['release_id']}: {result['wheel_package_parity']['source_files']} source files "
                "match both distributions, wheel RECORD and shell modes pass")

    checks.append(_capture("current distribution parity and executable modes", artifact_integrity))
    if installed_artifacts:
        def installed_behavior() -> str:
            from verification.release_artifact_gate import qualify
            result = qualify(project, runtime_tests=True)
            if not result["passed"]:
                raise AssertionError(json.dumps(result, sort_keys=True))
            return ("3/3 original regressions plus the named mathematical profile pass in each installed artifact; "
                    "package-isolated virtual environments with inherited dependencies")
        checks.append(_capture("installed wheel and sdist regression probes", installed_behavior))
        warnings.append("Installed-artifact probes inherit dependencies; they are not dependency-isolated qualification.")
    else:
        warnings.append("Installed-artifact behavior was not executed; use --installed-artifacts or release_artifact_gate.py.")

    if skip_mkdocs:
        warnings.append("Strict MkDocs build was explicitly skipped.")
    elif require_mkdocs:
        checks.append(_capture("strict MkDocs build", lambda: _run_mkdocs(project)))
    elif importlib.util.find_spec("mkdocs") is not None:
        checks.append(_capture("strict MkDocs build", lambda: _run_mkdocs(project)))
    else:
        warnings.append("MkDocs is not installed, so the strict documentation build was not executed.")

    full_test_record: dict[str, Any] | None = None
    if full_tests:
        full_test_record = _run_full_tests(project)
        checks.append(
            GateCheck(
                "complete source test suite",
                full_test_record.get("passed") is True,
                full_test_record["summary"],
            )
        )
    else:
        warnings.append("The complete source test suite was not requested in this gate invocation.")

    checks.append(GateCheck("source and artifact snapshot unchanged during qualification",
                            release_binding(project) == binding,
                            binding["source_sha256"]))
    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return PublicationGateReport(
        package_version=version,
        root=".",
        mode="full" if full_tests else "quick",
        checks=tuple(checks),
        warnings=tuple(warnings),
        environment={**_environment(), "release_binding": binding},
        capability_record=capability_record,
        full_test_record=full_test_record,
        generated_at_utc=generated,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--output", default="qualification/source.json")
    parser.add_argument("--markdown", default="qualification/source.md")
    parser.add_argument("--full-tests", action="store_true")
    parser.add_argument("--require-mkdocs", action="store_true")
    parser.add_argument("--skip-mkdocs", action="store_true")
    parser.add_argument("--installed-artifacts", action="store_true")
    args = parser.parse_args(argv)
    if args.require_mkdocs and args.skip_mkdocs:
        parser.error("--require-mkdocs and --skip-mkdocs are mutually exclusive")
    root = (Path(args.root) if args.root else Path(__file__).resolve().parents[1]).resolve()
    report = run_gate(
        root,
        full_tests=args.full_tests,
        require_mkdocs=args.require_mkdocs,
        skip_mkdocs=args.skip_mkdocs,
        installed_artifacts=args.installed_artifacts,
    )
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    markdown = Path(args.markdown)
    if not markdown.is_absolute():
        markdown = root / markdown
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8", newline="\n")
    markdown.write_text(report.to_markdown(), encoding="utf-8", newline="\n")
    print(payload, end="")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
