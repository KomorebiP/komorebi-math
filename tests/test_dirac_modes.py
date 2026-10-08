"""Independent finite matrix references for the Physics Dirac-mode helpers."""
from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from komorebi_math.exceptions import DimensionError, ValidationError
from komorebi_math.quantum.dirac import (
    central_mode_momentum, dirac_matrices, dirac_propagator, dirac_symbol,
)


def test_exact_clifford_coordinate_actions_and_anticommutators() -> None:
    matrices = dirac_matrices()
    identity = np.eye(4)
    basis = np.eye(4)
    assert np.array_equal(matrices[0] @ basis[:, 0], basis[:, 3])
    assert np.array_equal(matrices[1] @ basis[:, 0], 1j * basis[:, 3])
    assert np.array_equal(matrices[2] @ basis[:, 1], -basis[:, 3])
    assert np.array_equal(matrices[3] @ basis[:, 3], -basis[:, 3])
    for index, matrix in enumerate(matrices):
        assert np.array_equal(matrix, matrix.conj().T)
        assert np.array_equal(matrix @ matrix, identity)
        assert not matrix.flags.writeable
        for other in matrices[index + 1:]:
            assert np.array_equal(matrix @ other + other @ matrix, np.zeros((4, 4)))


def test_dirac_matrices_are_independent_input_snapshots() -> None:
    first = dirac_matrices()
    first[0].setflags(write=True)
    first[0][:] = 0
    assert dirac_matrices()[0][0, 3] == 1
    momentum = np.array([1., 2., 3.])
    symbol = dirac_symbol(momentum)
    momentum[:] = np.nan
    assert np.all(np.isfinite(symbol))


@pytest.mark.parametrize("seed", range(8))
def test_closed_propagator_matches_independent_eigh_and_group_law(seed: int) -> None:
    rng = np.random.default_rng(seed)
    momentum = rng.normal(size=3) * (seed + 1)
    for mass, time in product((-1., 0., 2.5), (-.7, 0., .3)):
        symbol = dirac_symbol(momentum, mass)
        values, vectors = np.linalg.eigh(symbol)
        reference = (vectors * np.exp(-1j * time * values)) @ vectors.conj().T
        actual = dirac_propagator(momentum, time, mass)
        assert np.allclose(actual, reference, atol=2e-13)
        assert np.allclose(actual.conj().T @ actual, np.eye(4), atol=2e-13)
        assert np.allclose(symbol @ symbol, (mass * mass + momentum @ momentum) * np.eye(4), atol=2e-12)
        assert np.allclose(dirac_propagator(momentum, time + .19, mass), actual @ dirac_propagator(momentum, .19, mass), atol=2e-13)


@pytest.mark.parametrize("mesh", [0., 1e-100, -.125, .125, 1/32])
def test_mode_momentum_zero_mesh_removable_value_and_scalar_reference(mesh: float) -> None:
    mode = np.array([0, -2, 3], dtype=np.int64)
    p = 2 * np.pi * mode
    expected = p if mesh == 0 else np.sin(mesh * p) / mesh
    assert np.allclose(central_mode_momentum(mode, mesh), expected, atol=2e-14)
    assert np.allclose(central_mode_momentum(mode, mesh), central_mode_momentum(mode, -mesh))


def test_zero_energy_and_tiny_energy_are_stable() -> None:
    assert np.array_equal(dirac_propagator([0., 0., 0.], 3., 0.), np.eye(4))
    assert np.array_equal(dirac_propagator([0., 0., 0.], 1e308, 0.), np.eye(4))
    time = 1e308
    mass = 1e-308
    expected = np.diag([np.exp(-1j), np.exp(-1j), np.exp(1j), np.exp(1j)])
    assert np.allclose(dirac_propagator([0., 0., 0.], time, mass), expected)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, 1j])
def test_invalid_real_scalars_are_rejected(value) -> None:
    with pytest.raises(ValidationError):
        dirac_symbol([0., 0., 0.], value)
    with pytest.raises(ValidationError):
        dirac_propagator([0., 0., 0.], value)
    with pytest.raises(ValidationError):
        central_mode_momentum([0, 0, 0], value)


@pytest.mark.parametrize("momentum,error", [([1, 2], DimensionError), ([[1, 2, 3]], DimensionError),
                                         ([True, 0, 0], ValidationError), ([np.nan, 0, 0], ValidationError),
                                         ([1j, 0, 0], ValidationError)])
def test_invalid_momentum_domains_are_rejected(momentum, error) -> None:
    with pytest.raises(error):
        dirac_symbol(momentum)
    with pytest.raises(error):
        dirac_propagator(momentum, 1.)


@pytest.mark.parametrize("mode,error", [([1, 2], DimensionError), ([1., 2., 3.], ValidationError),
                                     ([True, 0, 0], ValidationError), ([np.inf, 0, 0], ValidationError)])
def test_mode_labels_are_strict_integer_triples(mode, error) -> None:
    with pytest.raises(error):
        central_mode_momentum(mode)


def test_unrepresentable_energy_and_phase_reject_instead_of_returning_nonfinite() -> None:
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValidationError, match="energy"):
            dirac_propagator([1e308, 1e308, 1e308], 1., 1e308)
        with pytest.raises(ValidationError, match="phase"):
            dirac_propagator([1., 0., 0.], 1e308, 2.)
        with pytest.raises(ValidationError, match="phase"):
            central_mode_momentum([1, 0, 0], 1e308)
