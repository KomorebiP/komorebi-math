"""Aggregate public-beta contract and stable surface closure."""

from __future__ import annotations

from komorebi_math import __version__

from komorebi_math.release import check_beta_readiness, check_release_contract, load_release_contract
from komorebi_math.verify import run_block17_verification


def test_block17_release_contract_passes_all_layers() -> None:
    report = check_release_contract()
    report.certificate.require()
    assert report.passed
    assert len(report.checks) == 21
    assert report.contract.stage == ("final" if __version__ == "1.0.0" else "release-candidate")
    assert report.contract.package_version == __version__


def test_final_readiness_does_not_reuse_historical_execution() -> None:
    report = check_beta_readiness()
    assert not report.passed
    assert report.mode == ("final" if __version__ == "1.0.0" else "release-candidate")
    assert report.gpu is not None
    assert any(check.name == "executed platform evidence" and not check.passed for check in report.checks)


def test_legacy_cumulative_readiness_verifier_remains_fail_closed() -> None:
    report = run_block17_verification()
    assert not report.passed
    assert report.total_count > 189
    assert [check for check in report.checks if not check.passed]


def test_release_contract_binds_every_new_beta_artifact() -> None:
    contract = load_release_contract()
    assert contract.upgrade.count == 7
    assert contract.evidence.count == 2
    assert contract.documentation.count == 131
    assert len(contract.upgrade.sha256) == 64
    assert len(contract.evidence.sha256) == 64
    assert len(contract.documentation.sha256) == 64
