from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.signals import (
    detect_piecewise_polynomial_breaks,
    finite_difference,
    haar_lipschitz_bound,
    haar_step_coefficient,
    recover_jumps_subgaussian,
    recover_separated_jumps,
)


def test_deterministic_jump_recovery() -> None:
    n = 512
    h = 1.0 / n
    grid = np.arange(n + 1) * h
    locations = np.array([0.173, 0.461, 0.802])
    amplitudes = np.array([0.45, -0.62, 0.31])
    smooth = 0.09 * np.sin(2 * np.pi * grid) + 0.03 * grid
    lipschitz = 0.09 * 2 * np.pi + 0.03
    signal = smooth.copy()
    for location, amplitude in zip(locations, amplitudes, strict=True):
        signal += amplitude * (grid >= location)
    rng = np.random.default_rng(61)
    epsilon = 0.001
    samples = signal + rng.uniform(-epsilon, epsilon, size=grid.size)
    result = recover_separated_jumps(
        samples,
        spacing=h,
        lipschitz_bound=lipschitz,
        noise_bound=epsilon,
        minimum_jump=float(np.min(np.abs(amplitudes))),
        minimum_separation=0.2,
    )
    result.certificate.require()
    true_cells = tuple(int(location / h) for location in locations)
    assert result.cells == true_cells
    assert np.max(np.abs(result.locations - locations)) <= result.location_error_bound
    assert np.max(np.abs(result.amplitudes - amplitudes)) <= result.amplitude_error_bound + 1e-12


def test_subgaussian_recovery_declares_conditional_model() -> None:
    samples = np.array([0.0, 0.0, 1.0, 1.0])
    result = recover_jumps_subgaussian(
        samples,
        spacing=1.0,
        lipschitz_bound=0.0,
        sigma=0.01,
        delta=0.1,
        minimum_jump=1.0,
        minimum_separation=1.0,
    )
    assert result.cells == (1,)
    assert result.certificate.assumptions_unchecked


def test_piecewise_polynomial_break_band() -> None:
    left = np.arange(10, dtype=float) ** 2
    right_x = np.arange(10, 20, dtype=float)
    right = right_x**2 + 20.0
    values = np.concatenate((left, right))
    result = detect_piecewise_polynomial_breaks(values, degree=2, noise_bound=0.0, threshold=1e-12)
    result.certificate.require()
    assert result.indices
    assert finite_difference(np.arange(10, dtype=float) ** 2, order=3).size == 7


def test_haar_bounds() -> None:
    assert np.isclose(haar_step_coefficient(2.0, 0.25, 0.3), 2.0 * 0.5 * 0.3)
    assert np.isclose(haar_lipschitz_bound(4.0, 0.25), 4.0 * 0.25**1.5 / 4.0)


def test_jump_recovery_rejects_closed_gap() -> None:
    with pytest.raises(Exception):
        recover_separated_jumps(
            [0.0, 1.0],
            spacing=1.0,
            lipschitz_bound=1.0,
            noise_bound=0.0,
            minimum_jump=1.0,
        )


def test_vector_jump_recovery_and_scalar_packet_boundary() -> None:
    from komorebi_math.signals import recover_vector_jumps
    from komorebi_math.exceptions import ValidationError

    values = np.zeros((30, 2))
    values[10:] += np.array([3.0, 4.0])
    values[22:] += np.array([-1.0, -2.0])
    result = recover_vector_jumps(
        values,
        spacing=0.1,
        lipschitz_bound=0.0,
        noise_bound=0.0,
        minimum_jump=2.0,
    )
    result.certificate.require()
    assert result.cells == (9, 21)
    assert result.amplitudes.shape == (2, 2)
    with pytest.raises(ValidationError):
        result.packet()
