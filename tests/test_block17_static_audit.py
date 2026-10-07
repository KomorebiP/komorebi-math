"""Static source audit for the 1.0 public beta."""

from __future__ import annotations

from pathlib import Path

from verification.block17_static_audit import run_audit

ROOT = Path(__file__).resolve().parents[1]


def test_block17_static_audit_passes() -> None:
    report = run_audit(ROOT)
    assert report.passed, report.to_dict()
    assert report.package_version == "1.0.0b4"
    assert len(report.checks) == 4
    assert report.diagnostics["block17_contract_files"] == 10
