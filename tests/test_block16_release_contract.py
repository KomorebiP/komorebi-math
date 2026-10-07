"""Machine-readable 1.0 public-beta release contract."""

from __future__ import annotations

import json

import pytest

import komorebi_math
import komorebi_math.api.v1 as api_v1
import komorebi_math.stable as stable
import komorebi_math.v1 as v1
from komorebi_math.cli.contracts import API_CONTRACT_MODES, VERIFICATION_SUITE_NAMES
from komorebi_math.exceptions import ValidationError
from komorebi_math.release import (
    ApiTier,
    ReleaseContract,
    check_release_contract,
    current_runtime_record,
    load_release_contract,
)


def test_release_tier_names() -> None:
    assert komorebi_math.__version__ == "1.0.0b4"
    assert tuple(tier.value for tier in ApiTier) == (
        "stable",
        "beta",
        "research",
        "internal",
    )


def test_release_contract_is_deterministic_and_complete() -> None:
    left = load_release_contract()
    right = load_release_contract()
    assert left == right
    assert left.digest == right.digest
    assert left.package_version == komorebi_math.__version__
    assert left.predecessor_version == "1.0.0b3"
    assert left.release_line == "1.0.0b1 to 1.0.0rc1"
    assert left.stage == "beta"
    assert left.declared_python == ("3.11", "3.12", "3.13")
    assert left.stable_api.count == 131
    assert left.beta_api.count == 83
    assert left.research_api.count == 1312
    assert left.semantics.count == 10
    assert left.persistence.count == 51
    assert left.upgrade.count == 7
    assert left.evidence.count == 2
    assert left.documentation.count == 131


def test_release_contract_roundtrip_from_mapping() -> None:
    contract = load_release_contract()
    restored = ReleaseContract.from_dict(contract.to_dict())
    assert restored == contract
    assert restored.digest == contract.digest


def test_release_contract_rejects_unknown_fields() -> None:
    value = load_release_contract().to_dict()
    value["unexpected"] = True
    with pytest.raises(ValidationError, match="fields do not match"):
        ReleaseContract.from_dict(value)


def test_release_contract_checks_every_layer() -> None:
    report = check_release_contract()
    report.certificate.require()
    assert report.passed
    assert len(report.checks) == 21
    assert not [check for check in report.checks if not check.passed]


def test_versioned_aliases_are_identity_preserving() -> None:
    assert stable.__all__ == v1.__all__ == api_v1.__all__
    for name in stable.__all__:
        assert getattr(v1, name) is getattr(stable, name)
        assert getattr(api_v1, name) is getattr(stable, name)


def test_cli_contract_names_are_frozen() -> None:
    assert API_CONTRACT_MODES == ("research", "beta", "stable")
    required = set(load_release_contract().required_cli_suites)
    assert required <= set(VERIFICATION_SUITE_NAMES)
    assert {
        "block15",
        "stable",
        "release",
        "gpu",
        "block16",
        "upgrade",
        "evidence",
        "documentation",
        "beta",
        "block17",
        "all",
    } <= required


def test_runtime_record_is_json_serializable() -> None:
    record = current_runtime_record()
    assert record["package_version"] == "1.0.0b4"
    assert record["python"]
    assert record["platform"]
    json.dumps(record, sort_keys=True)
