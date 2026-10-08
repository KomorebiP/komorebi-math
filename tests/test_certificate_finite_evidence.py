"""Live certificates cannot approve nonfinite or subsequently changed evidence."""

import json

import numpy as np
import pytest

from komorebi_math.certificates import Certificate, ComputationResult, Exactness
from komorebi_math.exceptions import ValidationError
from komorebi_math.export import dumps_portable, loads_portable


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), complex(0, 1), True])
def test_nonfinite_or_nonreal_residual_cannot_be_valid(value):
    certificate = Certificate(True, Exactness.NUMERICAL, residuals={"identity": value})
    assert certificate.valid is False
    assert not ComputationResult(1, certificate).valid
    with pytest.raises(ValidationError, match="residual.*finite"):
        certificate.require()


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), -1, True])
def test_invalid_tolerance_cannot_approve_even_zero_residual(tolerance):
    certificate = Certificate(True, Exactness.NUMERICAL, {"identity": 0.0}, tolerance)
    assert not certificate.valid
    with pytest.raises(ValidationError, match="tolerance"):
        certificate.require()


def test_residuals_are_owned_and_immutable():
    residuals = {"identity": 0.0}
    certificate = Certificate(np.bool_(True), Exactness.NUMERICAL, residuals, 1e-10)
    residuals["identity"] = float("nan")
    assert certificate.residuals["identity"] == 0.0
    with pytest.raises(TypeError):
        certificate.residuals["identity"] = 1.0
    assert certificate.require() is certificate
    assert json.loads(certificate.to_json())["residuals"] == {"identity": 0.0}


def test_immutable_residuals_preserve_portable_historical_semantics():
    original = Certificate(True, Exactness.EXACT, {"identity": 0})
    restored = loads_portable(dumps_portable(original))
    assert restored == original
    assert restored.recorded_valid
    assert not restored.current_valid
    with pytest.raises(ValidationError, match="revalidation"):
        restored.require()


def test_invalid_record_keeps_diagnostic_infinity_and_unbounded_integers_are_finite():
    invalid = Certificate(False, Exactness.NUMERICAL, {"failed_identity": float("inf")})
    assert not invalid.valid
    assert np.isinf(invalid.residuals["failed_identity"])
    assert Certificate(True, Exactness.EXACT, {"exact_integer": 10**1000}).valid
