"""Independent references for quantum numerical contracts found in the deep audit."""
from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.dynamics import RationalMemoryRealization, matrix_exponential
from komorebi_math.exceptions import ValidationError
from komorebi_math.quantum import (
    LindbladGenerator,
    QuantumFeshbachReduction,
    constraint_closure,
    spectral_dimension,
)
from komorebi_math.quantum._validation import density_matrix


@pytest.mark.parametrize("time", [0.0, 0.4, 1.7])
def test_quantum_derivative_memory_reconstructs_schrodinger_trajectory(time: float) -> None:
    hamiltonian = np.array([[0., 1.], [1., 0.]], dtype=complex)
    reduction = QuantumFeshbachReduction(hamiltonian, (0,))
    kernel = reduction.memory_kernel()
    realization = RationalMemoryRealization(kernel.output, kernel.hidden, kernel.input)
    initial = np.array([1., 0.], dtype=complex)
    retained = realization.retained_trajectory(-1j * reduction.reduction.retained_block, initial, time)
    assert retained[0] == pytest.approx(np.cos(time))
    assert np.allclose(retained, (matrix_exponential(-1j * time * hamiltonian) @ initial)[:1])
    assert kernel.at(0)[0, 0] == pytest.approx(-1.)


@pytest.mark.parametrize("seed", range(6))
def test_complex_quantum_memory_matches_full_independent_generator(seed: int) -> None:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    hamiltonian = (matrix + matrix.conj().T) / 2
    retained_indices = (3, 0)
    reduction = QuantumFeshbachReduction(hamiltonian, retained_indices)
    kernel = reduction.memory_kernel()
    realization = RationalMemoryRealization(kernel.output, kernel.hidden, kernel.input)
    permutation = retained_indices + reduction.reduction.hidden
    initial = rng.normal(size=5) + 1j * rng.normal(size=5)
    ordered = initial[list(permutation)]
    actual = realization.retained_trajectory(-1j * reduction.reduction.retained_block, ordered, .37)
    expected = (matrix_exponential(-1j * .37 * hamiltonian) @ initial)[list(retained_indices)]
    assert np.allclose(actual, expected, atol=1e-11)


@pytest.mark.parametrize("time", [1., 1000., 1e6])
def test_spectral_dimension_survives_heat_trace_underflow(time: float) -> None:
    assert spectral_dimension(np.eye(3), time) == pytest.approx(2 * time)
    assert spectral_dimension(np.diag([1., 2.]), time) == pytest.approx(
        2 * time * (1. + np.exp(-time) / (1. + np.exp(-time)))
    )


def test_spectral_dimension_with_kernel_has_zero_large_time_limit() -> None:
    assert spectral_dimension(np.diag([0., 1., 2.]), 1000.) == pytest.approx(0.)
    assert spectral_dimension(np.diag([0., 1., 2.]), 1e308) == pytest.approx(0.)


@pytest.mark.parametrize("scale", [1e-100, 1e-6, 1., 1e6, 1e100])
def test_constraint_nonclosure_depends_on_span_not_generator_scale(scale: float) -> None:
    x = np.array([[0., 1.], [1., 0.]])
    z = np.diag([1., -1.])
    report = constraint_closure((scale * x, scale * z))
    assert not report.certificate.valid
    assert report.certificate.residuals["closure_obstruction"] == pytest.approx(2.)


@pytest.mark.parametrize("mixing", [np.eye(2), np.array([[1., 2.], [3., 4.]])])
def test_constraint_quotient_obstruction_is_generator_basis_invariant(mixing: np.ndarray) -> None:
    x = np.array([[0., 1.], [1., 0.]])
    z = np.diag([1., -1.])
    constraints = tuple(row[0] * x + row[1] * z for row in mixing)
    assert constraint_closure(constraints).certificate.residuals["closure_obstruction"] == pytest.approx(2.)


def test_complete_constraint_algebra_still_closes_at_small_scale() -> None:
    x = np.array([[0., 1.], [1., 0.]])
    y = np.array([[0., -1j], [1j, 0.]])
    z = np.diag([1., -1.])
    constraint_closure((1e-20 * x, 1e-20 * y, 1e-20 * z)).certificate.require()


def test_lindblad_input_arrays_are_snapshotted_before_caller_mutation() -> None:
    hamiltonian = np.zeros((2, 2), dtype=complex)
    jump = np.array([[0., 1.], [0., 0.]], dtype=complex)
    generator = LindbladGenerator(hamiltonian, (jump,))
    reference = generator.apply(np.eye(2))
    hamiltonian[:] = np.nan
    jump[:] = np.inf
    assert np.allclose(generator.apply(np.eye(2)), reference)
    generator.trace_preservation_certificate().require()
    assert not generator.hamiltonian.flags.writeable
    assert not generator.jumps[0].flags.writeable


def test_lindblad_certificate_rejects_reassigned_nonfinite_hamiltonian() -> None:
    generator = LindbladGenerator(np.zeros((2, 2)), ())
    generator.hamiltonian = np.full((2, 2), np.nan)
    with pytest.raises(ValidationError, match="finite"):
        generator.trace_preservation_certificate()


def test_density_normalization_rejects_overflowing_trace_instead_of_returning_zero() -> None:
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValidationError, match="trace"):
            density_matrix(np.eye(2) * 1e308, normalize=True)


@pytest.mark.parametrize("tolerance", [float("inf"), float("nan"), -1.])
def test_quantum_validation_rejects_invalid_tolerances(tolerance: float) -> None:
    with pytest.raises(ValidationError, match="tolerance"):
        LindbladGenerator(np.eye(2), (), tolerance=tolerance)
    with pytest.raises(ValidationError, match="tolerance"):
        constraint_closure((np.eye(2),), tolerance=tolerance)
