from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.exceptions import DimensionError, ValidationError
from komorebi_math.signals import (
    RationalResamplingPlan,
    decimate,
    design_resample_filter,
    downsample,
    fir_filter,
    interpolate,
    lms_filter,
    nlms_filter,
    resample_poly,
    resample_poly_result,
    resampled_length,
    rls_filter,
    upsample,
)


def test_upsampling_and_downsampling_residue_conventions() -> None:
    signal = np.array([1.0, 2.0, 3.0])
    expanded = upsample(signal, 3)
    assert np.array_equal(expanded, np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0]))
    assert np.array_equal(downsample(expanded, 3), signal)
    assert np.array_equal(downsample(np.arange(10), 3, phase=1), np.array([1, 4, 7]))


def test_resampled_length_reduces_ratio_and_uses_ceiling() -> None:
    assert resampled_length(11, 6, 4) == 17
    assert resampled_length(10, 1, 3) == 4
    assert resampled_length(10, 3, 1) == 30


def test_resampling_filter_has_declared_dc_gain_and_odd_length() -> None:
    taps = design_resample_filter(5, 3)
    assert taps.size % 2 == 1
    assert np.sum(taps) == pytest.approx(5.0, abs=1e-12)
    assert np.allclose(taps, taps[::-1], atol=1e-15)


@pytest.mark.parametrize("up,down", [(2, 1), (3, 2), (2, 5), (7, 4), (6, 4)])
def test_polyphase_plan_matches_explicit_reference(up: int, down: int) -> None:
    rng = np.random.default_rng(12200 + up * 10 + down)
    signal = rng.normal(size=43)
    result = resample_poly_result(signal, up, down)
    result.certificate.require()
    assert result.signal.size == resampled_length(signal.size, up, down)
    assert np.allclose(result.signal, result.plan.explicit_reference(signal), atol=2e-14)


def test_polyphase_plan_supports_complex_signals() -> None:
    rng = np.random.default_rng(12201)
    signal = rng.normal(size=31) + 1j * rng.normal(size=31)
    taps = design_resample_filter(4, 3)
    plan = RationalResamplingPlan(4, 3, taps)
    plan.certificate(signal).require()
    assert np.allclose(plan.apply(signal), plan.explicit_reference(signal), atol=2e-14)


def test_identity_ratio_is_numerically_identity() -> None:
    rng = np.random.default_rng(12202)
    signal = rng.normal(size=77)
    output = resample_poly(signal, 9, 9)
    assert output.shape == signal.shape
    assert np.allclose(output, signal, atol=2e-15)


def test_custom_single_tap_resampling_exposes_zero_insertion_convention() -> None:
    signal = np.array([1.0, 2.0, 3.0])
    output = resample_poly(signal, 2, 1, taps=np.array([1.0]))
    assert np.array_equal(output, np.array([1.0, 0.0, 2.0, 0.0, 3.0, 0.0]))


def test_integer_decimation_and_interpolation_wrappers_match_rational_calls() -> None:
    rng = np.random.default_rng(12208)
    signal = rng.normal(size=51)
    assert np.array_equal(decimate(signal, 4), resample_poly(signal, 1, 4))
    assert np.array_equal(interpolate(signal, 3), resample_poly(signal, 3, 1))


def test_complex_resampling_taps_with_real_dc_gain_are_certifiable() -> None:
    signal = np.array([1.0 + 1.0j, -2.0j, 0.5])
    taps = np.array([0.25j, 2.0, -0.25j])
    plan = RationalResamplingPlan(2, 1, taps)
    plan.certificate(signal).require()
    assert np.allclose(plan.apply(signal), plan.explicit_reference(signal))


def test_resampling_validation() -> None:
    with pytest.raises(ValidationError):
        design_resample_filter(2, 3, num_taps=20)
    with pytest.raises(ValidationError):
        downsample(np.arange(10), 3, phase=3)
    with pytest.raises(ValidationError):
        downsample(np.arange(10), 3, phase=0.5)
    with pytest.raises(DimensionError):
        resample_poly([], 2, 1)


def _identification_problem(seed: int, length: int = 4000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    source = rng.normal(size=length)
    target = np.array([0.7, -0.4, 0.2])
    desired = fir_filter(source, target, mode="causal")
    return source, desired, target


def test_lms_identifies_short_real_fir() -> None:
    source, desired, target = _identification_problem(12203)
    result = lms_filter(source, desired, 3, step_size=0.01)
    result.certificate.require()
    assert np.allclose(result.final_weights, target, atol=1e-9)
    assert result.tail_mean_squared_error(0.2) < 1e-16
    assert "persistent excitation" in result.certificate.assumptions_unchecked


def test_nlms_identifies_short_real_fir_under_input_scaling() -> None:
    source, desired, target = _identification_problem(12204)
    source *= 25.0
    desired *= 25.0
    result = nlms_filter(source, desired, 3, step_size=0.8)
    result.certificate.require()
    assert np.allclose(result.final_weights, target, atol=1e-10)


def test_rls_identifies_short_real_fir() -> None:
    source, desired, target = _identification_problem(12205, length=2500)
    result = rls_filter(source, desired, 3, forgetting_factor=0.999, delta=10.0)
    result.certificate.require()
    assert np.allclose(result.final_weights, target, atol=3e-4)
    assert result.tail_mean_squared_error(0.2) < 1e-6


def test_complex_nlms_identification() -> None:
    rng = np.random.default_rng(12206)
    source = rng.normal(size=5000) + 1j * rng.normal(size=5000)
    target = np.array([0.4 + 0.2j, -0.1 + 0.3j])
    desired = fir_filter(source, np.conjugate(target), mode="causal")
    result = nlms_filter(source, desired, 2, step_size=0.6)
    result.certificate.require()
    assert np.allclose(result.final_weights, target, atol=1e-10)


def test_adaptive_weight_history_and_error_identity() -> None:
    source, desired, _ = _identification_problem(12207, length=100)
    result = nlms_filter(source, desired, 3)
    assert result.weight_history.shape == (101, 3)
    assert np.allclose(result.error, desired - result.output)
    assert result.mean_squared_error >= 0.0


def test_adaptive_validation() -> None:
    with pytest.raises(DimensionError):
        lms_filter(np.ones(10), np.ones(11), 2)
    with pytest.raises(ValidationError):
        nlms_filter(np.ones(10), np.ones(10), 2, epsilon=0.0)
    with pytest.raises(ValidationError):
        rls_filter(np.ones(10), np.ones(10), 2, forgetting_factor=1.1)
    with pytest.raises(DimensionError):
        lms_filter(np.ones(10), np.ones(10), 3, initial_weights=np.ones(2))
