"""Static source checks and dependency-boundary negative controls."""

from __future__ import annotations

from pathlib import Path

from verification.source_audit import _parse_python, _shell_mode_diagnostics, run_audit


def test_current_static_source_audit_passes() -> None:
    report = run_audit()
    assert report.passed, report.to_dict()
    assert len(report.checks) == 8
    assert len(report.diagnostics["production_modules_imported"]) >= 260


def test_production_imports_cannot_depend_on_tests_or_verification(tmp_path: Path) -> None:
    module = tmp_path / "src/komorebi_math/probe.py"
    module.parent.mkdir(parents=True)
    module.write_text("from tests.private import helper\nimport verification.build_distributions\n", encoding="utf-8")
    syntax, forbidden = _parse_python([module])
    assert syntax == []
    assert len(forbidden) == 2
    assert any("tests.private" in item for item in forbidden)
    assert any("verification.build_distributions" in item for item in forbidden)


def test_invalid_python_is_reported_instead_of_skipped(tmp_path: Path) -> None:
    module = tmp_path / "invalid.py"
    module.write_text("def incomplete(\n", encoding="utf-8")
    syntax, forbidden = _parse_python([module])
    assert len(syntax) == 1
    assert "invalid.py" in syntax[0]
    assert forbidden == []


def test_posix_mode_gate_is_not_applied_on_windows(tmp_path: Path) -> None:
    script = tmp_path / "runner.sh"
    script.write_text("#!/usr/bin/env sh\n", encoding="utf-8")
    script.chmod(0o644)
    scripts, failures, applicable = _shell_mode_diagnostics(
        tmp_path,
        operating_system="nt",
    )
    assert [path.name for path in scripts] == ["runner.sh"]
    assert failures == []
    assert applicable is False
