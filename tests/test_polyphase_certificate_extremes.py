"""Reconstruction certificates retain discrepancies at extreme scales."""

import numpy as np
import pytest

from komorebi_math.signals.polyphase import lattice_polyphase_analyze, polyphase_analyze


@pytest.fixture(params=["axis", "lattice"])
def analyze(request):
    if request.param == "axis":
        return lambda value: polyphase_analyze(value, 2)
    return lambda value: lattice_polyphase_analyze(value, (2,))


@pytest.mark.parametrize("delta", [1e-200, 1e200, complex(0.0, 1e-200)])
def test_nonzero_finite_discrepancy_cannot_underflow_to_zero(analyze, delta):
    expected = np.array([delta, 0])
    coordinates = analyze(np.zeros_like(expected))
    certificate = coordinates.certificate(expected, tolerance=0.0)
    residual, = certificate.residuals.values()
    assert not certificate.valid
    assert np.isfinite(residual)
    assert residual == pytest.approx(abs(delta), rel=1e-14, abs=0.0)


@pytest.mark.parametrize("dtype", [np.uint64, np.int64])
def test_fixed_width_subtraction_does_not_hide_large_discrepancy(analyze, dtype):
    info = np.iinfo(dtype)
    expected = np.array([info.max, info.min], dtype=dtype)
    coordinates = analyze(np.array([info.min, info.max], dtype=dtype))
    certificate = coordinates.certificate(expected)
    residual, = certificate.residuals.values()
    assert not certificate.valid
    assert np.isfinite(residual)
    assert residual > float(info.max)


def test_mixed_integer_float_difference_is_computed_before_rounding(analyze):
    coordinates = analyze(np.array([2**63 + 1, 0], dtype=np.uint64))
    certificate = coordinates.certificate(np.array([float(2**63), 0.0]), tolerance=0.0)
    residual, = certificate.residuals.values()
    assert not certificate.valid
    assert residual == 1.0


@pytest.mark.parametrize("imaginary", [0.0, 2.0])
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_integer_complex_components_are_subtracted_before_rounding(analyze, imaginary, reverse):
    integer = np.array([2**63 + 1, 0], dtype=np.uint64)
    complex_value = np.array([complex(2**63, imaginary), 0j])
    source, expected = (complex_value, integer) if reverse else (integer, complex_value)
    certificate = analyze(source).certificate(expected, tolerance=0.0)
    residual, = certificate.residuals.values()
    assert not certificate.valid
    assert residual == pytest.approx(np.hypot(1.0, imaginary), rel=1e-14, abs=0.0)


@pytest.mark.parametrize("signal", [np.array([1e-200, 0.0]), np.array([1e200, 0.0]),
                                    np.array([0, 2**64 - 1], dtype=np.uint64)])
def test_identical_extreme_signal_retains_valid_certificate(analyze, signal):
    certificate = analyze(signal).certificate(signal, tolerance=0.0)
    certificate.require()
    assert set(certificate.residuals.values()) == {0.0}


def test_nonfinite_expected_array_cannot_receive_valid_certificate(analyze):
    certificate = analyze(np.array([0, 0], dtype=np.uint64)).certificate(np.array([np.inf, 0.0]))
    assert not certificate.valid
