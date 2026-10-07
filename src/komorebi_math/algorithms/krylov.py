"""Krylov methods and packet block preconditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..certificates import Certificate, Exactness
from ..exceptions import AssumptionError, DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._validation import finite_float, strict_integer
from .layout import PacketLayout
from .structured import BlockDiagonalOperator, DenseOperator

Array = np.ndarray
Apply = Callable[[Array], Array]


@dataclass(frozen=True, slots=True)
class KrylovBasisResult:
    """Orthonormal Krylov basis and its numerical certificate."""

    basis: Array
    dimension: int
    breakdown: bool
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="PSC-VIII-T20",
        title="Finite Krylov packet decomposition",
        status=ClaimStatus.PROVED,
        source="Paper VIII",
        assumptions=("finite linear operator", "declared start vector"),
        verification_suite="algorithms.krylov",
    )
)
def krylov_basis(
    apply: Apply,
    start: Any,
    steps: int,
    *,
    tolerance: float = 1e-12,
) -> KrylovBasisResult:
    """Construct a modified Gram Schmidt Krylov basis."""

    vector = np.asarray(start)
    if vector.ndim != 1 or vector.size == 0:
        raise DimensionError("start must be a nonempty vector")
    steps_value = strict_integer(steps, "steps", minimum=1, maximum=int(vector.size))
    tolerance_value = finite_float(tolerance, "tolerance", minimum=0.0)
    norm = float(np.linalg.norm(vector))
    if norm <= tolerance_value:
        raise ValidationError("start vector must be nonzero")
    basis: list[Array] = []
    current = vector.astype(np.result_type(vector, complex), copy=True) / norm
    breakdown = False
    for _ in range(steps_value):
        for previous in basis:
            current -= np.vdot(previous, current) * previous
        current_norm = float(np.linalg.norm(current))
        if current_norm <= tolerance_value:
            breakdown = True
            break
        current /= current_norm
        basis.append(current.copy())
        current = np.asarray(apply(current))
        if current.shape != vector.shape:
            raise DimensionError("operator application changed the vector dimension")
    matrix = np.column_stack(basis) if basis else np.zeros((vector.size, 0), dtype=complex)
    orthogonality = float(np.linalg.norm(matrix.conj().T @ matrix - np.eye(matrix.shape[1])))
    certificate = Certificate(
        valid=orthogonality <= 100 * tolerance_value,
        exactness=Exactness.NUMERICAL,
        residuals={"orthogonality": orthogonality},
        tolerance=100 * tolerance_value,
        assumptions_checked=("finite dimension", "modified Gram Schmidt"),
        diagnostics={"requested_steps": steps_value, "computed_dimension": matrix.shape[1]},
        theorem_id="PSC-VIII-T20",
        message="valid Krylov basis" if orthogonality <= 100 * tolerance_value else "Krylov basis lost orthogonality",
    )
    return KrylovBasisResult(matrix, matrix.shape[1], breakdown, certificate)


@dataclass(frozen=True, slots=True)
class ConjugateGradientResult:
    """Conjugate gradient iterate, residual history, and certificate."""

    solution: Array
    residual_norms: tuple[float, ...]
    iterations: int
    converged: bool
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="PSC-VIII-T22",
        title="Preconditioned conjugate gradient certificate",
        status=ClaimStatus.CLASSICAL,
        source="Paper VIII",
        assumptions=("Hermitian positive definite operator", "positive definite preconditioner"),
        verification_suite="algorithms.krylov",
    )
)
def conjugate_gradient(
    apply: Apply,
    right_hand_side: Any,
    *,
    precondition: Apply | None = None,
    initial: Any | None = None,
    relative_tolerance: float = 1e-10,
    absolute_tolerance: float = 0.0,
    max_iterations: int | None = None,
    curvature_tolerance: float = 1e-14,
) -> ConjugateGradientResult:
    """Solve a declared Hermitian positive definite system by CG."""

    b = np.asarray(right_hand_side)
    if b.ndim != 1 or b.size == 0:
        raise DimensionError("right_hand_side must be a nonempty vector")
    dtype = np.result_type(b, complex)
    x = np.zeros(b.shape, dtype=dtype) if initial is None else np.asarray(initial, dtype=dtype).copy()
    if x.shape != b.shape:
        raise DimensionError("initial vector has an incompatible shape")
    relative_value = finite_float(relative_tolerance, "relative_tolerance", minimum=0.0)
    absolute_value = finite_float(absolute_tolerance, "absolute_tolerance", minimum=0.0)
    curvature_value = finite_float(curvature_tolerance, "curvature_tolerance", minimum=0.0)
    limit = (
        10 * b.size
        if max_iterations is None
        else strict_integer(max_iterations, "max_iterations", minimum=1)
    )

    def _apply(vector: Array) -> Array:
        value = np.asarray(apply(vector), dtype=dtype)
        if value.shape != b.shape:
            raise DimensionError("operator application changed the vector dimension")
        return value

    def _precondition(vector: Array) -> Array:
        value = vector.copy() if precondition is None else np.asarray(precondition(vector), dtype=dtype)
        if value.shape != b.shape:
            raise DimensionError("preconditioner application changed the vector dimension")
        return value

    def _real_pairing(
        left: Array,
        right: Array,
        *,
        hermitian_message: str,
        positivity_message: str,
        allow_numerical_zero: bool,
        residual_norm: float,
        numerical_stop: float,
    ) -> float | None:
        pairing = np.vdot(left, right)
        scale = float(np.linalg.norm(left) * np.linalg.norm(right))
        comparison_scale = max(scale, np.finfo(float).tiny)
        if abs(pairing.imag) > curvature_value * comparison_scale:
            raise AssumptionError(hermitian_message)
        real = float(pairing.real)
        negative_boundary = -curvature_value * comparison_scale
        if real < negative_boundary:
            raise AssumptionError(positivity_message)
        if real <= 0.0:
            if allow_numerical_zero and residual_norm <= numerical_stop:
                return None
            raise AssumptionError(positivity_message)
        return real

    b_norm = float(np.linalg.norm(b))
    threshold = max(absolute_value, relative_value * b_norm)
    epsilon = np.finfo(np.empty((), dtype=dtype).real.dtype).eps
    numerical_stop = max(threshold, 100.0 * epsilon * max(1.0, b_norm))

    residual = b.astype(dtype, copy=False) - _apply(x)
    residual_norms = [float(np.linalg.norm(residual))]
    if residual_norms[-1] <= threshold:
        certificate = Certificate(
            valid=True,
            exactness=Exactness.NUMERICAL,
            residuals={"final_residual": residual_norms[-1]},
            tolerance=threshold,
            assumptions_checked=("initial residual tolerance",),
            theorem_id="PSC-VIII-T22",
            message="conjugate gradient converged at the initial iterate",
        )
        return ConjugateGradientResult(x, tuple(residual_norms), 0, True, certificate)

    z = _precondition(residual)
    rz_real = _real_pairing(
        residual,
        z,
        hermitian_message="preconditioner is not Hermitian in the initial residual direction",
        positivity_message="preconditioner is not positive in the initial residual direction",
        allow_numerical_zero=False,
        residual_norm=residual_norms[-1],
        numerical_stop=numerical_stop,
    )
    assert rz_real is not None
    direction = z.copy()

    iterations = 0
    stopped_at_numerical_floor = False
    for iteration in range(1, limit + 1):
        current_residual_norm = residual_norms[-1]
        if current_residual_norm <= threshold:
            break
        applied = _apply(direction)
        curvature_real = _real_pairing(
            direction,
            applied,
            hermitian_message="operator is not Hermitian in the generated direction",
            positivity_message="operator is not positive definite in the generated direction",
            allow_numerical_zero=True,
            residual_norm=current_residual_norm,
            numerical_stop=numerical_stop,
        )
        if curvature_real is None:
            stopped_at_numerical_floor = True
            break

        alpha = rz_real / curvature_real
        if not np.isfinite(alpha):
            raise AssumptionError("conjugate gradient produced a nonfinite step length")
        x += alpha * direction
        residual -= alpha * applied
        residual_norms.append(float(np.linalg.norm(residual)))
        iterations = iteration
        if residual_norms[-1] <= threshold:
            break

        z = _precondition(residual)
        rz_new_real = _real_pairing(
            residual,
            z,
            hermitian_message="preconditioner is not Hermitian in a generated residual direction",
            positivity_message="preconditioner is not positive in a generated residual direction",
            allow_numerical_zero=True,
            residual_norm=residual_norms[-1],
            numerical_stop=numerical_stop,
        )
        if rz_new_real is None:
            stopped_at_numerical_floor = True
            break
        beta = rz_new_real / rz_real
        if not np.isfinite(beta):
            raise AssumptionError("conjugate gradient produced a nonfinite recurrence coefficient")
        direction = z + beta * direction
        rz_real = rz_new_real

    final_residual = float(np.linalg.norm(b.astype(dtype, copy=False) - _apply(x)))
    converged = final_residual <= threshold
    valid = converged
    certificate = Certificate(
        valid=valid,
        exactness=Exactness.NUMERICAL,
        residuals={
            "final_residual": final_residual,
            "recursive_final_residual": residual_norms[-1],
        },
        tolerance=threshold,
        assumptions_checked=("positive curvature on generated directions",),
        assumptions_unchecked=("global Hermitian positive definiteness",),
        diagnostics={
            "iterations": iterations,
            "converged": converged,
            "stopped_at_numerical_floor": stopped_at_numerical_floor,
            "numerical_stop": numerical_stop,
        },
        theorem_id="PSC-VIII-T22",
        message="conjugate gradient converged" if converged else "conjugate gradient did not meet the requested tolerance",
    )
    return ConjugateGradientResult(x, tuple(residual_norms), iterations, converged, certificate)


@dataclass(frozen=True, slots=True)
class LanczosResult:
    """Lanczos basis and projected tridiagonal operator."""

    tridiagonal: Array
    basis: Array
    alpha: Array
    beta: Array
    breakdown: bool
    certificate: Certificate


def lanczos_tridiagonal(
    apply: Apply,
    dimension: int,
    steps: int,
    *,
    start: Any | None = None,
    seed: int = 0,
    tolerance: float = 1e-12,
) -> LanczosResult:
    """Run fully reorthogonalized finite Lanczos tridiagonalization."""

    dimension_value = strict_integer(dimension, "dimension", minimum=1)
    steps_value = strict_integer(steps, "steps", minimum=1, maximum=dimension_value)
    tolerance_value = finite_float(tolerance, "tolerance", minimum=0.0)
    if start is None:
        seed_value = strict_integer(seed, "seed")
        rng = np.random.default_rng(seed_value)
        vector = rng.normal(size=dimension_value)
    else:
        vector = np.asarray(start)
    if vector.ndim != 1 or vector.size != dimension_value:
        raise DimensionError("start vector has an incompatible shape")
    vector = vector.astype(np.result_type(vector, complex), copy=True)
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm <= tolerance_value:
        raise ValidationError("start vector must be nonzero")
    q = vector / vector_norm
    previous = np.zeros_like(q)
    beta_previous = 0.0
    basis: list[Array] = []
    alpha_values: list[float] = []
    beta_values: list[float] = []
    breakdown = False
    for index in range(steps_value):
        basis.append(q.copy())
        z = np.asarray(apply(q), dtype=q.dtype) - beta_previous * previous
        if z.shape != q.shape:
            raise DimensionError("operator application changed the vector dimension")
        alpha_complex = np.vdot(q, z)
        if abs(alpha_complex.imag) > 100 * tolerance_value * max(1.0, abs(alpha_complex.real)):
            raise AssumptionError("operator is not Hermitian on the Lanczos basis")
        alpha = float(alpha_complex.real)
        alpha_values.append(alpha)
        z -= alpha * q
        for basis_vector in basis:
            z -= np.vdot(basis_vector, z) * basis_vector
        beta = float(np.linalg.norm(z))
        if index < steps_value - 1:
            beta_values.append(beta)
        if beta <= tolerance_value:
            breakdown = True
            break
        previous, q = q, z / beta
        beta_previous = beta

    count = len(alpha_values)
    tridiagonal = np.diag(np.asarray(alpha_values, dtype=float))
    if count > 1:
        off_diagonal = np.asarray(beta_values[: count - 1], dtype=float)
        tridiagonal += np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    basis_matrix = np.column_stack(basis[:count])
    orthogonality = float(
        np.linalg.norm(basis_matrix.conj().T @ basis_matrix - np.eye(count))
    )
    projected = basis_matrix.conj().T @ np.column_stack(
        [np.asarray(apply(basis_matrix[:, index])) for index in range(count)]
    )
    projection_residual = float(np.linalg.norm(projected - tridiagonal))
    valid = max(orthogonality, projection_residual) <= 1000 * tolerance_value
    certificate = Certificate(
        valid=valid,
        exactness=Exactness.NUMERICAL,
        residuals={
            "orthogonality": orthogonality,
            "projected_tridiagonal": projection_residual,
        },
        tolerance=1000 * tolerance_value,
        assumptions_checked=("Hermitian behavior on generated basis", "full reorthogonalization"),
        assumptions_unchecked=("global Hermitian operator",),
        diagnostics={"computed_steps": count, "breakdown": breakdown},
        theorem_id="PSC-VIII-T23",
        message="valid Lanczos tridiagonalization" if valid else "Lanczos certificate failed",
    )
    return LanczosResult(
        tridiagonal,
        basis_matrix,
        np.asarray(alpha_values),
        np.asarray(beta_values[: max(0, count - 1)]),
        breakdown,
        certificate,
    )


def preconditioned_condition_bound(eta: float) -> float:
    """Return the block Jacobi condition bound from a coupling norm."""

    eta_value = finite_float(eta, "eta", minimum=0.0)
    if eta_value >= 1:
        raise ValidationError("eta must be less than one")
    return (1.0 + eta_value) / (1.0 - eta_value)


@dataclass(frozen=True, slots=True)
class BlockJacobiCertificate:
    """Packet diagonal, normalized coupling, and condition certificate."""

    diagonal: BlockDiagonalOperator
    normalized_coupling: Array
    eta: float
    condition_bound: float | None
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="PSC-VIII-T24",
        title="Packet block Jacobi condition bound",
        status=ClaimStatus.PROVED,
        source="Paper VIII",
        assumptions=("Hermitian positive definite packet diagonal", "normalized coupling norm below one"),
        verification_suite="algorithms.krylov",
    )
)
def block_jacobi_certificate(
    matrix: Any,
    layout: PacketLayout,
    *,
    tolerance: float = 1e-10,
) -> BlockJacobiCertificate:
    """Certify the packet block Jacobi condition estimate."""

    tolerance_value = finite_float(tolerance, "tolerance", minimum=0.0)
    array = np.asarray(matrix)
    if array.ndim != 2 or array.shape != (layout.dimension, layout.dimension):
        raise DimensionError("matrix shape does not match the packet layout")
    hermitian_residual = float(np.linalg.norm(array - array.conj().T, 2))
    if hermitian_residual > tolerance_value:
        raise AssumptionError("matrix must be Hermitian")
    blocks: list[DenseOperator] = []
    inverse_square_roots: list[Array] = []
    minimum_eigenvalue = float("inf")
    for packet_slice in layout.slices:
        block = array[packet_slice, packet_slice]
        eigenvalues, eigenvectors = np.linalg.eigh(block)
        minimum_eigenvalue = min(minimum_eigenvalue, float(np.min(eigenvalues)))
        if np.min(eigenvalues) <= tolerance_value:
            raise AssumptionError("every packet diagonal block must be positive definite")
        blocks.append(DenseOperator(block))
        inverse_square_roots.append(
            (eigenvectors * (1.0 / np.sqrt(eigenvalues))[None, :]) @ eigenvectors.conj().T
        )
    diagonal = BlockDiagonalOperator(tuple(blocks), layout.names)
    inverse_root = BlockDiagonalOperator(
        tuple(DenseOperator(block) for block in inverse_square_roots),
        layout.names,
    ).materialize()
    off_diagonal = array - diagonal.materialize()
    normalized = inverse_root @ off_diagonal @ inverse_root
    eta = float(np.linalg.norm(normalized, 2))
    condition_bound = preconditioned_condition_bound(eta) if eta < 1 else None
    preconditioned = np.eye(layout.dimension, dtype=array.dtype) + normalized
    actual_condition = float(np.linalg.cond(preconditioned))
    bound_valid = condition_bound is not None and actual_condition <= condition_bound + 100 * tolerance_value
    certificate = Certificate(
        valid=bound_valid,
        exactness=Exactness.NUMERICAL,
        residuals={
            "hermitian": hermitian_residual,
            "normalized_coupling": eta,
            "actual_preconditioned_condition": actual_condition,
        },
        tolerance=tolerance_value,
        assumptions_checked=("Hermitian matrix", "positive definite packet diagonal"),
        assumptions_unchecked=(() if eta < 1 else ("normalized coupling norm below one",)),
        diagnostics={
            "minimum_packet_eigenvalue": minimum_eigenvalue,
            "condition_bound": condition_bound,
        },
        theorem_id="PSC-VIII-T24",
        message=(
            "valid block Jacobi condition certificate"
            if bound_valid
            else "normalized coupling is not below one or the condition bound failed"
        ),
    )
    return BlockJacobiCertificate(diagonal, normalized, eta, condition_bound, certificate)


def block_jacobi_preconditioner(diagonal: BlockDiagonalOperator) -> Apply:
    """Return the exact block diagonal solve as a preconditioner."""

    return lambda value: diagonal.solve(value)


def power_perturbation_bound(
    left_norm: float,
    right_norm: float,
    difference_norm: float,
    exponent: int,
) -> float:
    """Return the standard telescoping bound for two operator powers."""

    left_value = finite_float(left_norm, "left_norm", minimum=0.0)
    right_value = finite_float(right_norm, "right_norm", minimum=0.0)
    difference_value = finite_float(difference_norm, "difference_norm", minimum=0.0)
    exponent_value = strict_integer(exponent, "exponent", minimum=0)
    if exponent_value == 0:
        return 0.0
    maximum = max(left_value, right_value)
    return float(exponent_value) * maximum ** (exponent_value - 1) * difference_value


__all__ = [
    "BlockJacobiCertificate",
    "ConjugateGradientResult",
    "KrylovBasisResult",
    "LanczosResult",
    "block_jacobi_certificate",
    "block_jacobi_preconditioner",
    "conjugate_gradient",
    "krylov_basis",
    "lanczos_tridiagonal",
    "power_perturbation_bound",
    "preconditioned_condition_bound",
]
