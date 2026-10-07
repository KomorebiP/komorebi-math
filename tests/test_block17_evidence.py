"""Executed-platform evidence and beta-to-RC boundary tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from komorebi_math.exceptions import ValidationError
from komorebi_math.release import (
    ValidationEvidence,
    check_beta_readiness,
    check_validation_evidence,
    load_validation_evidence,
)


def test_beta_platform_evidence_passes_with_authorized_inheritance() -> None:
    report = check_validation_evidence()
    report.certificate.require()
    assert report.covered_platforms == ("linux-x86_64", "windows-x86_64")
    assert report.current_platforms == ("linux-x86_64",)
    assert report.missing_platforms == ()
    assert report.missing_current_platforms == ("windows-x86_64",)
    assert report.inherited_records == ("windows-0.16.0a2-user-qualification",)
    assert not report.release_candidate_ready


def test_release_candidate_evidence_gate_remains_closed_until_exact_windows_beta_run() -> None:
    report = check_validation_evidence(require_current_all_platforms=True)
    assert not report.passed
    assert report.missing_current_platforms == ("windows-x86_64",)
    readiness = check_beta_readiness(require_current_platforms=True)
    assert not readiness.passed
    assert any(check.name == "executed platform evidence" and not check.passed for check in readiness.checks)


def test_evidence_sources_are_digest_checked() -> None:
    evidence = load_validation_evidence()
    data = Path(__file__).resolve().parents[1] / "src" / "komorebi_math" / "data"
    for record in evidence.records:
        assert (data / record.source).is_file()
        assert len(record.source_sha256) == 64


def test_validation_evidence_roundtrip_is_deterministic() -> None:
    evidence = load_validation_evidence()
    restored = ValidationEvidence.from_dict(evidence.to_dict())
    assert restored == evidence
    assert restored.digest == evidence.digest
    assert evidence.target_version == "1.0.0b4"


def test_validation_evidence_rejects_duplicate_platform_requirements() -> None:
    payload = load_validation_evidence().to_dict()
    payload["required_platforms"].append(payload["required_platforms"][0])
    with pytest.raises(ValidationError, match="unique"):
        ValidationEvidence.from_dict(payload)


def test_tampered_evidence_source_is_reported(tmp_path: Path) -> None:
    payload = load_validation_evidence().to_dict()
    payload["records"][0]["source_sha256"] = "f" * 64
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = check_validation_evidence(path)
    assert not report.passed
    assert report.certificate.residuals["invalid_sources"] == 1.0
