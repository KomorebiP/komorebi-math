"""Dependency-free Block 16 static release checks."""

from __future__ import annotations

from pathlib import Path

from verification.block16_static_audit import _shell_mode_diagnostics, run_audit


def test_block16_static_audit_passes() -> None:
    report = run_audit()
    assert report.passed, report.to_dict()
    assert len(report.checks) == 8
    assert len(report.diagnostics["production_modules_imported"]) >= 260


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
