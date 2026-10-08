"""Independent finite-model tests for Fourier jump recovery."""

import numpy as np
import pytest

from komorebi_math.exceptions import AssumptionError, DimensionError, ValidationError
from komorebi_math.signals.fourier_recovery import (
    FourierJumpRecovery, PeriodicJumpPacket, edge_moments_from_fourier,
    hankel_moment_matrix, recover_fourier_jumps,
)


def measured_coefficients(locations, amplitudes, period, degree, bound):
    # Analytic integration by parts, independent of the implementation's packet.
    output = {}
    for index in range(degree + 1, degree + 1 + 2 * bound):
        value = sum(amplitude * np.exp(-2j * np.pi * index * location / period)
                    for location, amplitude in zip(locations, amplitudes, strict=True)) / (2j * np.pi * index)
        output[index] = value
        output[-index] = value.conjugate()
    return output


@pytest.mark.parametrize("period", [2 * np.pi, 1, 5.0])
@pytest.mark.parametrize("degree,bound", [(0, 3), (3, 5)])
@pytest.mark.parametrize("known", [None, 3])
def test_recovers_independently_integrated_three_jump_model(period, degree, bound, known):
    locations = np.array([.11, .37, .72]) * period
    amplitudes = np.array([2.0, -.75, -1.25])
    result = recover_fourier_jumps(measured_coefficients(locations, amplitudes, period, degree, bound),
                                  background_degree=degree, maximum_jumps=bound,
                                  known_order=known, period=period)
    assert isinstance(result, FourierJumpRecovery)
    assert result.order == 3
    assert result.certificate.valid
    np.testing.assert_allclose(result.packet.locations, locations, atol=1e-11)
    np.testing.assert_allclose(result.packet.amplitudes, amplitudes, atol=1e-11)
    assert result.packet.fourier(0) == 0


@pytest.mark.parametrize("period", [1e-309, 1e308])
def test_period_normalization_avoids_intermediate_overflow(period):
    ratios = np.array([.1, .4, .7])
    locations, amplitudes = ratios * period, np.array([2.0, -1.0, -1.0])
    measurements = {index: sum(amplitudes * np.exp(-2j * np.pi * index * ratios)) / (2j * np.pi * index)
                    for index in range(1, 9)}
    result = recover_fourier_jumps(measurements, background_degree=0, maximum_jumps=4, period=period)
    assert result.certificate.valid
    np.testing.assert_allclose(result.packet.locations / period, ratios, atol=1e-11)


def test_mean_zero_periodic_integration_and_right_continuous_representative():
    packet = PeriodicJumpPacket([1.0, 3.0], [2, -2], period=4)
    np.testing.assert_array_equal(packet.evaluate([0, 1, 2, 3, 4, 5]), [-1, 1, 1, -1, -1, 1])
    # Constant segment lengths are both two, giving exact integral zero.
    assert 2 * packet.evaluate(0) + 2 * packet.evaluate(2) == 0
    for index in (1, 2, 3, -1):
        omega = 2 * np.pi * index / 4
        # Direct integration of -1 on [0,1), +1 on [1,3), -1 on [3,4).
        coefficient = sum(level * (np.exp(-1j * omega * right) - np.exp(-1j * omega * left))
                          / (-1j * omega) for left, right, level in ((0, 1, -1), (1, 3, 1), (3, 4, -1))) / 4
        assert packet.fourier(index) == pytest.approx(coefficient, abs=1e-14)


def test_zero_edge_branch_and_declared_zero_order():
    coefficients = {index: 0j for index in range(1, 9)}
    for known in (None, 0):
        result = recover_fourier_jumps(coefficients, background_degree=0, maximum_jumps=4, known_order=known)
        assert result.order == 0
        assert result.certificate.valid
        np.testing.assert_array_equal(result.packet.evaluate([0, 1, 2]), [0, 0, 0])
    with pytest.raises(AssumptionError, match="rank"):
        recover_fourier_jumps(coefficients, background_degree=0, maximum_jumps=4, known_order=2)


def test_declared_zero_order_does_not_certify_nonzero_moments():
    coefficients = measured_coefficients([.2, 2.4], [1.0, -1.0], 2 * np.pi, 0, 3)
    result = recover_fourier_jumps(coefficients, background_degree=0, maximum_jumps=3, known_order=0)
    assert not result.certificate.valid


def test_aliased_measurement_subset_cannot_replace_consecutive_samples():
    coefficients = measured_coefficients([.2, 2.4], [1.0, -1.0], 2 * np.pi, 0, 4)
    decimated = {key: value for key, value in coefficients.items() if key % 2 == 0}
    with pytest.raises(ValidationError, match="missing Fourier measurement"):
        recover_fourier_jumps(decimated, background_degree=0, maximum_jumps=4)


def test_observed_conjugacy_violation_is_rejected():
    coefficients = measured_coefficients([.2, 2.4], [1.0, -1.0], 2 * np.pi, 0, 3)
    coefficients[-1] += 1j
    with pytest.raises(AssumptionError, match="conjugacy"):
        recover_fourier_jumps(coefficients, background_degree=0, maximum_jumps=3)


def test_nonunit_exponential_nodes_fail_periodic_model_certificate():
    nodes = np.array([.8 * np.exp(-.2j), .7 * np.exp(-2.4j)])
    coefficients = {index: sum(np.array([1, -1]) * nodes ** index) / (2j * np.pi * index)
                    for index in range(1, 7)}
    result = recover_fourier_jumps(coefficients, background_degree=0, maximum_jumps=3, known_order=2)
    assert not result.certificate.valid
    assert result.certificate.residuals["node_circle"] > .1


def test_hankel_matrix_has_independent_expected_entries_and_shape():
    matrix = hankel_moment_matrix(np.array([1, 2, 3, 4], np.uint8), 2, 3)
    np.testing.assert_array_equal(matrix, [[1, 2, 3], [2, 3, 4]])
    with pytest.raises(DimensionError, match="not enough"):
        hankel_moment_matrix([1, 2], 2, 2)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), complex(0, float("inf"))])
def test_nonfinite_fourier_data_cannot_produce_a_valid_certificate(bad):
    coefficients = {index: 0j for index in range(1, 7)}
    coefficients[1] = bad
    with pytest.raises(ValidationError, match="finite"):
        recover_fourier_jumps(coefficients, background_degree=0, maximum_jumps=3)


@pytest.mark.parametrize("kwargs", [dict(background_degree=-1, maximum_jumps=2),
                                    dict(background_degree=0, maximum_jumps=0),
                                    dict(background_degree=True, maximum_jumps=2),
                                    dict(background_degree=2.0, maximum_jumps=2),
                                    dict(background_degree=0, maximum_jumps=2.0),
                                    dict(background_degree=0, maximum_jumps=2.5),
                                    dict(background_degree=0, maximum_jumps=2, known_order=3),
                                    dict(background_degree=0, maximum_jumps=2, rank_tolerance=float("nan")),
                                    dict(background_degree=0, maximum_jumps=2, rank_tolerance=1)])
def test_parameter_gate_rejects_invalid_finite_model_requests(kwargs):
    with pytest.raises(ValidationError):
        recover_fourier_jumps({}, **kwargs)


@pytest.mark.parametrize("locations,amplitudes", [([1, 1], [1, -1]), ([1, 5], [1, -1]),
                                                  ([1, 3], [1, 2]), ([1, 3], [0, 0])])
def test_periodic_packet_rejects_invalid_active_jump_geometry(locations, amplitudes):
    with pytest.raises(AssumptionError):
        PeriodicJumpPacket(locations, amplitudes, period=4)


def test_packet_inputs_and_result_evidence_are_copied_and_readonly():
    locations, amplitudes = np.array([1.0, 3.0]), np.array([1.0, -1.0])
    packet = PeriodicJumpPacket(locations, amplitudes, period=4)
    locations[:] = 0
    amplitudes[:] = 0
    np.testing.assert_array_equal(packet.locations, [1, 3])
    assert not packet.locations.flags.writeable
    assert not packet.amplitudes.flags.writeable


def test_fourier_derivative_normalization_and_invalid_zero_index():
    coefficients = {1: 1 / (2j * np.pi), -1: 1 / (-2j * np.pi)}
    np.testing.assert_allclose(edge_moments_from_fourier(coefficients, [1, -1]), [1, 1])
    with pytest.raises(ValidationError, match="nonzero"):
        edge_moments_from_fourier({0: 0j}, [0])


@pytest.mark.parametrize("value", [True, np.bool_(True), 2.0, np.float64(2.0)])
def test_fourier_indices_require_actual_integer_types(value):
    with pytest.raises(ValidationError, match="indices"):
        edge_moments_from_fourier({2: 1j}, [value])


@pytest.mark.parametrize("amplitudes", [[True, -1], np.array([True, False]), np.array([True, -1], dtype=object)])
def test_boolean_values_are_not_silently_promoted_to_numeric_jump_amplitudes(amplitudes):
    with pytest.raises(ValidationError, match="booleans"):
        PeriodicJumpPacket([1, 3], amplitudes, period=4)


def test_hankel_row_count_requires_integer_type():
    with pytest.raises(ValidationError, match="rows"):
        hankel_moment_matrix([1, 2, 3], 2.0, 2)

