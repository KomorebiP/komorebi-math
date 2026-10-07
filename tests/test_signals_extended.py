from __future__ import annotations

import numpy as np

from komorebi_math.certificates import Exactness
from komorebi_math.signals import (
    PeriodicFilterBank,
    estimate_positive_packet_1d,
    gamma_from_impedance,
    impedance_from_gamma,
    lattice_polyphase_analyze,
    nonlinear_shift_filter,
    packet_product,
    phase_mobius,
    polyphase_filter_kernels,
    residue_project,
    scalar_region_lp_error,
    smith_sensitivity,
)


def test_lattice_polyphase_and_residue_projectors() -> None:
    values = np.arange(6 * 7).reshape(6, 7)
    coordinates = lattice_polyphase_analyze(values, (2, 3))
    coordinates.certificate(values).require()
    assert np.array_equal(coordinates.reconstruct(), values)
    projected = sum((residue_project(values, 3, residue, axis=1) for residue in range(3)), np.zeros_like(values))
    assert np.array_equal(projected, values)


def test_polyphase_filter_kernel_addresses() -> None:
    kernels = polyphase_filter_kernels([1.0, 2.0, 3.0, 4.0], 2)
    assert len(kernels) == 2
    assert all(len(row) == 2 for row in kernels)
    assert kernels[0][0].target == 0
    assert kernels[1][0].source == 0


def test_periodic_identity_filter_bank() -> None:
    bank = PeriodicFilterBank(([1.0],), ([1.0],), 1)
    bank.perfect_reconstruction_certificate(17).require()
    values = np.arange(17, dtype=float)
    assert np.allclose(bank.synthesize(bank.analyze(values)), values)


def test_nonlinear_packet_product_and_shift() -> None:
    first = [np.array([1.0, 2.0]), np.array([3.0])]
    second = [np.array([4.0]), np.array([5.0, 6.0])]
    product = packet_product(first, second, order=2).require()
    assert len(product) == 2
    values = np.arange(8, dtype=float)
    result = nonlinear_shift_filter(values, shifts=[0, 1], coefficients=[1.0, 0.5], alpha=0.1)
    assert result.shape == values.shape


def test_smith_coordinates_roundtrip_and_phase() -> None:
    impedance = np.array([0.5, 1.0, 2.0, 1.0 + 0.2j])
    gamma = gamma_from_impedance(impedance)
    assert np.allclose(impedance_from_gamma(gamma), impedance)
    assert np.all(smith_sensitivity(gamma, regularization=1e-6) > 0)
    assert np.allclose(phase_mobius(impedance, 5, 0), impedance)


def test_heuristic_packet_estimator_is_labeled() -> None:
    values = np.concatenate((np.zeros(30), np.ones(20)))
    result = estimate_positive_packet_1d(values)
    assert result.certificate.exactness is Exactness.HEURISTIC
    assert result.mask.sum() == 20


def test_scalar_region_formula_direct_value() -> None:
    value = scalar_region_lp_error(1.0, -1.0, 4.0, 3.0, 2.0, p=2)
    expected = np.sqrt(4.0 * 4.0 + 3.0 + 2.0)
    assert np.isclose(value, expected)
