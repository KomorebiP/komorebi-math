"""Historical Block 16 package-audit record retained in the beta handoff."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_block16_package_audit_remains_valid_historical_evidence() -> None:
    payload = json.loads((ROOT / "PACKAGE_AUDIT.json").read_text(encoding="utf-8"))
    assert payload["schema"] == "komorebi-math.block16-package-audit"
    assert payload["package_version"] == "0.16.0a2"
    assert payload["passed"] is True
    assert len(payload["checks"]) == 9
    assert payload["diagnostics"]["public_api_symbol_counts"] == {
        "research": 1179,
        "beta": 83,
        "stable": 131,
    }
    assert payload["diagnostics"]["stable_semantic_cases"] == 10
    assert payload["diagnostics"]["portable_codec_entries"] == 51


def test_checked_in_block16_package_audit_markdown_is_labeled_historical() -> None:
    markdown = (ROOT / "PACKAGE_AUDIT.md").read_text(encoding="utf-8")
    assert "Status: **passed**" in markdown
    assert "Package version: `0.16.0a2`" in markdown
