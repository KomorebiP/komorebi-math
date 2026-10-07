"""Aggregate public-beta contract and stable surface closure."""

from __future__ import annotations

from komorebi_math.release import check_beta_readiness, check_release_contract, load_release_contract
from komorebi_math.verify import run_block17_verification


def test_block17_release_contract_passes_all_layers() -> None:
    report = check_release_contract()
    report.certificate.require()
    assert report.passed
    assert len(report.checks) == 21
    assert report.contract.stage == "beta"
    assert report.contract.package_version == "1.0.0b4"


def test_public_beta_readiness_passes_without_overclaiming_rc_status() -> None:
    report = check_beta_readiness()
    report.certificate.require()
    assert report.passed
    assert report.mode == "beta"
    assert report.gpu is None
    assert "release candidate requires current-version evidence" in " ".join(
        report.certificate.assumptions_unchecked
    )


def test_cumulative_block17_installed_verifier_passes() -> None:
    report = run_block17_verification()
    assert report.passed
    assert report.total_count > 189
    assert not [check for check in report.checks if not check.passed]


def test_release_contract_binds_every_new_beta_artifact() -> None:
    contract = load_release_contract()
    assert contract.upgrade.count == 7
    assert contract.evidence.count == 2
    assert contract.documentation.count == 131
    assert len(contract.upgrade.sha256) == 64
    assert len(contract.evidence.sha256) == 64
    assert len(contract.documentation.sha256) == 64
