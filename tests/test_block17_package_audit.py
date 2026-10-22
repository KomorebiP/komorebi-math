"""Current package audit preserves functionality and publication boundaries."""
import json
from pathlib import Path
from komorebi_math import __version__
from komorebi_math.release import load_release_contract
from verification import block17_package_audit
from verification.package_audit import run_audit
ROOT = Path(__file__).resolve().parents[1]


def test_current_package_audit_passes_without_claiming_hardware_qualification() -> None:
    report = run_audit(ROOT)
    assert report.passed, report.to_dict()
    assert report.package_version == __version__
    assert report.release_stage == load_release_contract().stage
    assert len(report.checks) == 7
    assert report.diagnostics["public_api_symbol_counts"]["stable"] == 131
    assert report.diagnostics["public_api_symbol_counts"]["beta"] == 83
    assert report.diagnostics["upgrade_fixtures"] == 7
    assert report.diagnostics["stable_documentation_entries"] == 131
    publication = report.diagnostics["publication_evidence"]
    registry = json.loads((ROOT / "src/komorebi_math/data/validation_evidence_1_0.json").read_text(encoding="utf-8"))
    assert publication["registry_target_version"] == registry["target_version"]
    if registry["target_version"] != __version__:
        assert publication["current_platform_evidence_passed"] is False
    assert publication["external_promotion_review_required"] is True
    assert publication["gpu_qualification_established_by_this_audit"] is False


def test_previous_package_audit_import_uses_current_checks() -> None:
    assert block17_package_audit.run_audit is run_audit
