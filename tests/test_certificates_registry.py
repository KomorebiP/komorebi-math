from __future__ import annotations

import json

import pytest

import komorebi_math as km
from komorebi_math.certificates import Certificate, ConditionalResult, Exactness
from komorebi_math.config import Settings, configure, get_settings, settings_context
from komorebi_math.exceptions import ValidationError
from komorebi_math.registry import FeatureStatus


def test_certificate_serialization_and_require() -> None:
    certificate = Certificate(
        valid=True,
        exactness=Exactness.EXACT,
        diagnostics={"value": 3},
    )
    assert certificate.require() is certificate
    parsed = json.loads(certificate.to_json())
    assert parsed["valid"] is True
    assert parsed["exactness"] == "exact"


def test_invalid_certificate_raises() -> None:
    certificate = Certificate(False, Exactness.NUMERICAL, message="bad object")
    with pytest.raises(ValidationError, match="bad object"):
        certificate.require()


def test_conditional_result_requires_conditional_certificate() -> None:
    certificate = Certificate(True, Exactness.CONDITIONAL, assumptions_unchecked=("existence",))
    result = ConditionalResult(4, certificate)
    assert result.value == 4
    with pytest.raises(ValueError):
        ConditionalResult(4, Certificate(True, Exactness.EXACT))


def test_settings_context_restores_previous_settings() -> None:
    configure(backend="numpy", tolerance=1e-9, validate_on_construction=True)
    before = get_settings()
    assert isinstance(before, Settings)
    with settings_context(tolerance=1e-5) as active:
        assert active.tolerance == 1e-5
        assert get_settings().tolerance == 1e-5
    assert get_settings() == before


def test_feature_registry_has_all_major_domains() -> None:
    keys = {feature.key for feature in km.features()}
    assert {
        "core",
        "refinement",
        "nonlinear",
        "analytic",
        "signals",
        "dynamics",
        "quantum",
        "algorithms",
        "arithmetic",
        "sequences",
        "deformation",
        "langlands.local",
        "langlands.global",
        "langlands.finite_image",
    }.issubset(keys)
    assert km.feature("core").status is FeatureStatus.IMPLEMENTED
    assert km.feature("refinement").status is FeatureStatus.IMPLEMENTED
    assert km.feature("nonlinear").status is FeatureStatus.IMPLEMENTED


def test_implemented_and_planned_namespaces_report_status() -> None:
    from komorebi_math import analytic, arithmetic, quantum, refinement, sequences

    assert refinement.status().key == "refinement"
    assert refinement.require() is None
    assert analytic.require() is None
    assert quantum.require() is None
    assert arithmetic.status().key == "arithmetic"
    assert arithmetic.require() is None
    assert sequences.status().key == "sequences"
    assert sequences.require() is None


def test_provenance_describe() -> None:
    record = km.describe(km.PacketSystem)
    assert record is not None
    assert record.identifier == "PSC-I-T20"
