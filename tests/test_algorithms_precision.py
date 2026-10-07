from __future__ import annotations

import itertools

import numpy as np
import pytest

from komorebi_math.algorithms import (
    DTypeChoice,
    assign_discrete_precisions,
    clipped_precision_allocation,
    optimal_precision_tolerances,
    packet_error_bound,
    precision_certificate,
    quantization_error,
    quantize_array,
    quantize_by_packet,
)
from komorebi_math.exceptions import ValidationError


def test_packet_error_bound_and_certificate() -> None:
    bound = packet_error_bound([0.1, 0.2], [3.0, 4.0])
    assert np.isclose(bound, 1.1)
    precision_certificate([0.1, 0.2], [3.0, 4.0], 1.0).require()


def test_optimal_precision_tolerances() -> None:
    sensitivities = np.array([1.0, 2.0, 4.0])
    costs = np.array([2.0, 1.0, 3.0])
    tolerance = 0.3
    allocation = optimal_precision_tolerances(sensitivities, costs, tolerance)
    assert np.isclose(np.dot(sensitivities, allocation), tolerance)
    ratios = sensitivities * allocation / costs
    assert np.allclose(ratios, ratios[0])


def test_clipped_precision_allocation() -> None:
    sensitivities = [1.0, 2.0, 3.0]
    costs = [1.0, 4.0, 2.0]
    maxima = [0.2, 0.1, 0.4]
    allocation = clipped_precision_allocation(sensitivities, costs, 0.35, maxima)
    assert np.dot(sensitivities, allocation) <= 0.35 + 1e-12
    assert np.all(allocation <= np.asarray(maxima) + 1e-12)
    assert np.allclose(clipped_precision_allocation(sensitivities, costs, 10.0, maxima), maxima)


def test_discrete_precision_assignment_matches_bruteforce() -> None:
    options = [
        [DTypeChoice("a16", 1.0, 4), DTypeChoice("a32", 2.0, 1)],
        [DTypeChoice("b16", 1.5, 3), DTypeChoice("b64", 4.0, 0)],
        [DTypeChoice("c32", 2.5, 2), DTypeChoice("c64", 5.0, 0)],
    ]
    result = assign_discrete_precisions(options, budget=4)
    assert result is not None
    result.certificate.require()
    feasible = []
    for choices in itertools.product(*options):
        error = sum(choice.error_units for choice in choices)
        if error <= 4:
            feasible.append((sum(choice.cost for choice in choices), error, tuple(choice.name for choice in choices)))
    brute = min(feasible)
    assert (result.total_cost, result.error_units, result.names) == brute


def test_precision_input_validation() -> None:
    with pytest.raises(ValidationError):
        DTypeChoice("bad", float("nan"), 1)
    with pytest.raises(ValidationError):
        DTypeChoice("bad", 1.0, 1.5)
    with pytest.raises(ValidationError):
        assign_discrete_precisions([], budget=1)
    with pytest.raises(ValidationError):
        assign_discrete_precisions([[DTypeChoice("a", 1.0, 0)]], budget=1.5)
    with pytest.raises(ValidationError):
        optimal_precision_tolerances([1.0], [float("inf")], 1.0)


def test_quantization_real_and_complex() -> None:
    real = np.array([1.234567, 2.345678])
    quantized = quantize_array(real, np.float32)
    assert quantized.dtype == np.float32
    complex_value = np.array([1.0 + 2.0j])
    complex_quantized = quantize_array(complex_value, np.float32)
    assert complex_quantized.dtype == np.complex64
    assert quantization_error(real, np.float16) >= 0
    packets = quantize_by_packet([real, complex_value], [np.float32, np.float32])
    assert packets[1].dtype == np.complex64

def test_precision_choice_and_quantization_validation() -> None:
    choice = DTypeChoice("finite", 1.25, 0, dtype="float32")
    assert choice.cost == 1.25
    with pytest.raises(ValidationError):
        DTypeChoice("bad", 1.0, 0, dtype="not-a-dtype")
    with pytest.raises(ValidationError):
        quantize_array([1.0], "not-a-dtype")
    with pytest.raises(ValidationError):
        packet_error_bound([float("inf")], [1.0])

