from __future__ import annotations

from math import exp, sqrt

import numpy as np
import pytest

from komorebi_math.dynamics import (
    HiddenMarkovFilter,
    MatrixSemigroup,
    PhaseTypeSurvival,
    approximate_likelihood_partition,
    coarsest_sufficient_partition,
    matrix_exponential,
)
from komorebi_math.exceptions import ValidationError
from komorebi_math.signals import (
    TruncatedPowerTerm,
    convolve_signal,
    packet_product,
    polyphase_analyze,
    polyphase_filter_kernels,
    recover_vector_jumps_subgaussian,
)


def _reconstruct_compact(channels: tuple[np.ndarray, ...], order: int) -> np.ndarray:
    length = max(
        (order * (channel.size - 1) + residue + 1 for residue, channel in enumerate(channels)),
        default=0,
    )
    result = np.zeros(length, dtype=np.result_type(*(channel.dtype for channel in channels), float))
    for residue, channel in enumerate(channels):
        result[residue : residue + order * channel.size : order] = channel
    return result


def test_taylor_matrix_exponential_is_independently_executable() -> None:
    matrix = np.array([[0.3, 0.7], [0.0, 0.3]])
    expected = exp(0.3) * np.array([[1.0, 0.7], [0.0, 1.0]])
    actual = matrix_exponential(matrix, method="taylor")
    assert np.allclose(actual, expected, atol=1e-13, rtol=1e-13)


def test_matrix_semigroup_rejects_negative_time() -> None:
    semigroup = MatrixSemigroup(np.array([[-1.0]]))
    with pytest.raises(ValidationError):
        semigroup.at(-0.1)


def test_compact_packet_product_propagates_radix_carry() -> None:
    left = (np.array([1.0, 2.0]), np.array([3.0, -1.0]))
    right = (np.array([2.0, 1.0]), np.array([4.0]))
    result = packet_product(left, right, order=2).require()
    direct = np.convolve(_reconstruct_compact(left, 2), _reconstruct_compact(right, 2))
    assert np.allclose(_reconstruct_compact(result, 2)[: direct.size], direct)


def test_infinite_polyphase_kernel_law_includes_leading_zero_lags() -> None:
    rng = np.random.default_rng(91)
    for order in range(2, 6):
        for _ in range(25):
            signal = rng.normal(size=13)
            kernel = rng.normal(size=8)
            source = polyphase_analyze(signal, order).channels
            kernels = polyphase_filter_kernels(kernel, order)
            target_channels: list[np.ndarray] = []
            for target in range(order):
                terms = [
                    np.convolve(source[source_index], kernels[target][source_index].coefficients)
                    for source_index in range(order)
                ]
                length = max(term.size for term in terms)
                combined = np.zeros(length)
                for term in terms:
                    combined[: term.size] += term
                target_channels.append(combined)
            reconstructed = _reconstruct_compact(tuple(target_channels), order)
            direct = np.convolve(signal, kernel)
            assert np.allclose(reconstructed[: direct.size], direct)


def test_finite_convolution_modes_use_declared_signal_first_convention() -> None:
    signal = np.array([1.0, 2.0])
    kernel = np.array([1.0, -1.0, 2.0, 3.0])
    full = np.convolve(signal, kernel, mode="full")
    start = (kernel.size - 1) // 2
    assert np.allclose(convolve_signal(signal, kernel, mode="same"), full[start : start + signal.size])
    assert convolve_signal(signal, kernel, mode="valid").size == 0
    assert convolve_signal(signal, kernel, mode="circular").shape == signal.shape


def test_vector_subgaussian_recovery_uses_dimension_aware_bound() -> None:
    samples = np.zeros((20, 3))
    samples[8:] += np.array([3.0, 4.0, 0.0])
    result = recover_vector_jumps_subgaussian(
        samples,
        spacing=1.0,
        lipschitz_bound=0.0,
        sigma=0.0,
        minimum_jump=4.0,
        failure_probability=0.05,
    )
    result.certificate.require()
    assert result.cells == (7,)
    assert np.allclose(result.amplitudes[0], [3.0, 4.0, 0.0])
    assert result.certificate.residuals["coordinate_noise_bound"] == 0.0


def test_order_zero_location_bound_respects_finite_interval() -> None:
    term = TruncatedPowerTerm(location=-1.0, amplitude=2.0, order=0)
    assert term.location_bound(-0.5, interval=(0.0, 1.0), p=2.0) == 0.0
    assert np.isclose(term.location_bound(0.5, interval=(0.0, 1.0), p=2.0), 2.0 * sqrt(0.5))


def test_phase_type_model_rejects_nontransient_generator() -> None:
    with pytest.raises(Exception):
        PhaseTypeSurvival(np.zeros((1, 1)), [1.0])
    unchecked = PhaseTypeSurvival(np.zeros((1, 1)), [1.0], validate=False)
    assert not unchecked.certificate().valid


def test_exact_sufficient_partition_rejects_approximate_tolerance() -> None:
    likelihood = np.array([0.0, 0.09, 0.18])
    with pytest.raises(ValidationError):
        coarsest_sufficient_partition([likelihood], tolerance=0.1)
    approximate = approximate_likelihood_partition([likelihood], tolerance=0.1)
    assert approximate.labels == (0, 0, 0)
    assert not approximate.is_fixed_sufficient([likelihood])


def test_hidden_markov_filter_uses_transition_lumping_directly() -> None:
    coarse = np.array([[0.8, 0.3], [0.2, 0.7]])
    transition = np.kron(coarse, np.eye(2))
    filter_model = HiddenMarkovFilter(transition)
    prior = np.array([0.1, 0.2, 0.3, 0.4])
    likelihood = np.array([2.0, 2.0, 0.5, 0.5])
    result = filter_model.coarse_step(prior, likelihood, [0, 0, 1, 1])
    result.certificate.require()
