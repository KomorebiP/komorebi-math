"""Cross-version API, persistence, and portable-record upgrade closure."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from komorebi_math.compatibility import (
    compare_public_api,
    create_stable_api_manifest,
    load_public_api_manifest,
)
from komorebi_math.exceptions import ValidationError
from komorebi_math.export.persistence import load_portable
from komorebi_math.release import (
    UpgradeContract,
    check_upgrade_contract,
    load_upgrade_contract,
)
from komorebi_math.refinement import PolyphaseRefinement, PolyphaseRefinementTower

DATA = Path(__file__).resolve().parents[1] / "src" / "komorebi_math" / "data"


def test_upgrade_contract_is_closed_and_deterministic() -> None:
    contract = load_upgrade_contract()
    assert contract == UpgradeContract.from_dict(contract.to_dict())
    assert contract.target_version == "1.0.0b4"
    assert contract.predecessor_versions == (
        "0.15.0a2",
        "0.16.0a2",
        "1.0.0b1",
        "1.0.0b2",
        "1.0.0b3",
    )
    assert len(contract.stable_api) == 5
    assert len(contract.fixtures) == 7
    assert len(contract.digest) == 64


def test_frozen_0_16_stable_api_is_exactly_unchanged() -> None:
    reference = load_public_api_manifest(DATA / "public_api_stable_0_16.json")
    current = create_stable_api_manifest()
    report = compare_public_api(reference, current, strict_signatures=True)
    report.certificate.require()
    assert report.added == ()
    assert report.removed == ()
    assert report.kind_changed == ()
    assert report.signature_changed == ()
    assert len(current.symbols) == 131


def test_complete_upgrade_contract_passes_and_loads_every_fixture() -> None:
    report = check_upgrade_contract()
    report.certificate.require()
    assert report.passed
    assert len(report.checks) == 15
    assert len(report.loaded_fixtures) == 7
    assert not [check for check in report.checks if not check.passed]


def test_0_15_refinement_records_load_under_the_beta() -> None:
    refinement = load_portable(DATA / "upgrade_0_15_0a2_polyphase_refinement.json")
    tower = load_portable(DATA / "upgrade_0_15_0a2_polyphase_tower.json")
    assert refinement == PolyphaseRefinement(4, 12)
    assert tower == PolyphaseRefinementTower((2, 6, 24))
    assert refinement.children(1) == (1, 5, 9)
    assert tower.orders == (2, 6, 24)


def test_upgrade_contract_detects_a_tampered_fixture_digest(tmp_path: Path) -> None:
    payload = load_upgrade_contract().to_dict()
    payload["fixtures"][0]["sha256"] = "0" * 64
    path = tmp_path / "upgrade.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = check_upgrade_contract(path)
    assert not report.passed
    assert any(not check.passed for check in report.checks)


def test_upgrade_contract_rejects_unknown_root_fields() -> None:
    payload = load_upgrade_contract().to_dict()
    payload["unknown"] = True
    with pytest.raises(ValidationError, match="fields do not match"):
        UpgradeContract.from_dict(payload)


def test_every_committed_upgrade_source_has_the_recorded_file_digest() -> None:
    contract = load_upgrade_contract()
    for fixture in contract.fixtures:
        source = DATA / fixture.filename
        assert source.is_file()
        assert sha256(source.read_bytes()).hexdigest() == fixture.sha256
