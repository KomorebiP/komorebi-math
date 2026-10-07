from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.algorithms import (
    PacketLayout,
    block_jacobi_certificate,
    block_jacobi_preconditioner,
    conjugate_gradient,
    krylov_basis,
    lanczos_tridiagonal,
    power_perturbation_bound,
    preconditioned_condition_bound,
)
from komorebi_math.exceptions import AssumptionError


def random_spd(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n, n))
    return matrix.T @ matrix + n * np.eye(n)


def test_krylov_basis() -> None:
    matrix = np.diag([1.0, 2.0, 3.0, 4.0])
    result = krylov_basis(lambda x: matrix @ x, np.ones(4), 4)
    result.certificate.require()
    assert result.dimension == 4
    assert np.allclose(result.basis.T.conj() @ result.basis, np.eye(4))


def test_conjugate_gradient_real_and_preconditioned() -> None:
    matrix = random_spd(11, 8)
    b = np.arange(1.0, 9.0)
    result = conjugate_gradient(lambda x: matrix @ x, b, relative_tolerance=1e-12, max_iterations=20)
    assert result.converged
    assert np.allclose(result.solution, np.linalg.solve(matrix, b), atol=1e-9)

    layout = PacketLayout((4, 4))
    certificate = block_jacobi_certificate(matrix, layout)
    precondition = block_jacobi_preconditioner(certificate.diagonal)
    result2 = conjugate_gradient(
        lambda x: matrix @ x,
        b,
        precondition=precondition,
        relative_tolerance=1e-12,
        max_iterations=20,
    )
    assert result2.converged


def test_conjugate_gradient_accepts_integer_rhs() -> None:
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    right = np.array([1, 2], dtype=int)
    result = conjugate_gradient(lambda x: matrix @ x, right, relative_tolerance=1e-12)
    assert result.converged
    assert np.allclose(result.solution, np.linalg.solve(matrix, right))


def test_conjugate_gradient_complex_hermitian() -> None:
    matrix = np.array([[4.0, 1.0j], [-1.0j, 3.0]], dtype=complex)
    b = np.array([1.0 + 2.0j, 3.0 - 1.0j])
    result = conjugate_gradient(lambda x: matrix @ x, b, relative_tolerance=1e-13, max_iterations=5)
    assert result.converged
    assert np.allclose(result.solution, np.linalg.solve(matrix, b))


def test_conjugate_gradient_rejects_negative_curvature() -> None:
    matrix = np.diag([1.0, -1.0])
    with pytest.raises(AssumptionError):
        conjugate_gradient(lambda x: matrix @ x, np.array([0.0, 1.0]), max_iterations=3)


def test_lanczos_tridiagonal() -> None:
    matrix = random_spd(12, 7)
    result = lanczos_tridiagonal(lambda x: matrix @ x, 7, 7, seed=4)
    result.certificate.require()
    projected = result.basis.conj().T @ matrix @ result.basis
    assert np.allclose(projected, result.tridiagonal, atol=1e-9)


def test_block_jacobi_bound() -> None:
    matrix = np.array(
        [
            [3.0, 0.0, 0.1, 0.0],
            [0.0, 2.0, 0.0, 0.1],
            [0.1, 0.0, 4.0, 0.0],
            [0.0, 0.1, 0.0, 5.0],
        ]
    )
    result = block_jacobi_certificate(matrix, PacketLayout((2, 2)))
    result.certificate.require()
    assert result.eta < 1
    assert result.condition_bound is not None
    assert result.condition_bound >= 1


def test_condition_and_power_bounds() -> None:
    assert np.isclose(preconditioned_condition_bound(0.2), 1.5)
    bound = power_perturbation_bound(2.0, 3.0, 0.1, 4)
    assert np.isclose(bound, 4 * 3**3 * 0.1)

def test_krylov_strict_integer_and_finite_validation() -> None:
    from komorebi_math.exceptions import ValidationError

    matrix = np.eye(2)
    with pytest.raises(ValidationError):
        krylov_basis(lambda x: matrix @ x, np.ones(2), 2.5)
    with pytest.raises(ValidationError):
        lanczos_tridiagonal(lambda x: matrix @ x, 2.0, 2)
    with pytest.raises(ValidationError):
        conjugate_gradient(lambda x: matrix @ x, np.ones(2), relative_tolerance=float("nan"))
    with pytest.raises(ValidationError):
        power_perturbation_bound(1.0, 1.0, 0.1, True)


def test_block_jacobi_reports_failed_coupling_hypothesis() -> None:
    matrix = np.array([[1.0, 2.0], [2.0, 1.0]])
    result = block_jacobi_certificate(matrix, PacketLayout((1, 1)))
    assert result.eta >= 1
    assert result.condition_bound is None
    assert not result.certificate.valid
    assert "normalized coupling norm below one" in result.certificate.assumptions_unchecked

