"""Deterministic package audit for the 1.0 public beta."""

from __future__ import annotations

import json
from pathlib import Path

from verification.block17_package_audit import run_audit

ROOT = Path(__file__).resolve().parents[1]


def test_block17_package_audit_passes() -> None:
    report = run_audit(ROOT)
    assert report.passed, report.to_dict()
    assert report.package_version == "1.0.0b4"
    assert report.release_stage == "public-beta"
    assert len(report.checks) == 7
    assert report.diagnostics["public_api_symbol_counts"] == {
        "research": 1312,
        "beta": 83,
        "stable": 131,
    }
    assert report.diagnostics["upgrade_fixtures"] == 7
    assert report.diagnostics["stable_documentation_entries"] == 131
