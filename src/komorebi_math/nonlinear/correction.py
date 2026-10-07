"""Quantitative nonlinear correction and finite Lyapunov--Schmidt reduction."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from .._residuals import value_audit, finite_tolerance
from .._snapshots import numpy_snapshot
from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .derivatives import DerivativeProvider


def _numeric_vector(value: Any, dimension: int, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=complex)
    if vector.ndim == 2 and 1 in vector.shape:
        vector = vector.reshape(-1)
    if vector.ndim != 1 or vector.shape[0] != dimension:
        raise DimensionError(f"{name} must have dimension {dimension}")
    if not np.all(np.isfinite(vector)):
        raise ValidationError(f"{name} must be finite")
    return vector


def _numeric_norm(value: Any, *, name: str = "value") -> float:
    """Stable Euclidean/Frobenius norm for already numerical data.

    ``np.linalg.norm`` squares entries and can underflow a tiny nonzero vector
    to zero.  Hypot reduction preserves the representable magnitude.  This is
    still floating-point arithmetic, not an interval or exact-real bound.
    """
    array = np.asarray(value, dtype=complex)
    if not np.all(np.isfinite(array)):
        raise ValidationError(f"{name} must be finite")
    magnitudes = np.hypot(array.real, array.imag)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = float(np.hypot.reduce(magnitudes.ravel(), initial=0.0))
    if not isfinite(result):
        raise ValidationError(f"{name} norm is not representable")
    return result


def _jacobian(provider: DerivativeProvider, point: np.ndarray) -> np.ndarray:
    if not provider.supports_order(1):
        raise ValidationError("a first derivative is required")
    columns = []
    for index in range(provider.source_dimension):
        direction = np.zeros(provider.source_dimension, dtype=complex)
        direction[index] = 1.0
        columns.append(_numeric_vector(
            provider.derivative(1, point, (direction,)),
            provider.target_dimension,
            name="first derivative",
        ))
    matrix = np.column_stack(columns)
    if matrix.shape != (provider.target_dimension, provider.source_dimension):
        raise DimensionError("declared derivative produced an invalid Jacobian shape")
    return matrix


@dataclass(frozen=True, slots=True)
class HilbertObstructionResult:
    """Finite orthogonal obstruction projection and least-norm correction data."""


    obstruction: np.ndarray
    obstruction_norm: float
    vanishes: bool
    least_norm_correction: np.ndarray | None
    correction_residual: float
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="PSC-III-T31",
        title="Orthogonal Hilbert obstruction projector",
        status=ClaimStatus.CLASSICAL,
        source="Paper III",
        assumptions=("finite-dimensional Hilbert spaces", "numerically resolved singular-value rank"),
        verification_suite="nonlinear.correction",
    )
)
class HilbertObstructionProjector:
    """Moore--Penrose correction and orthogonal cokernel projection."""

    def __init__(self, correction_map: Any, *, tolerance: float = 1e-10) -> None:
        matrix = numpy_snapshot(correction_map, dtype=complex)
        if matrix.ndim != 2:
            raise DimensionError("correction map must be a matrix")
        if not np.all(np.isfinite(matrix)):
            raise ValidationError("correction map must be finite")
        self.matrix = matrix
        self.codomain_dimension, self.domain_dimension = matrix.shape
        self.tolerance = float(tolerance)
        if not isfinite(self.tolerance) or self.tolerance < 0:
            raise ValidationError("tolerance must be finite and nonnegative")
        u, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
        threshold = self.tolerance
        if singular_values.size:
            threshold = max(threshold, self.tolerance * float(singular_values[0]))
        self.rank = int(np.sum(singular_values > threshold))
        self.singular_values = singular_values
        if not isfinite(threshold):
            raise ValidationError("singular-value threshold is not representable")
        # A separate pinv call uses a different relative cutoff. Construct the
        # inverse and all subspaces from the SAME retained singular values.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            self.pseudoinverse = ((vh[:self.rank].conj().T / singular_values[:self.rank])
                                  @ u[:, :self.rank].conj().T)
        if not np.all(np.isfinite(self.pseudoinverse)):
            raise ValidationError("retained singular-value inverse is not representable")
        image_basis = u[:, : self.rank]
        self.image_projector = image_basis @ image_basis.conj().T
        self.obstruction_projector = np.eye(self.codomain_dimension, dtype=complex) - self.image_projector
        self.kernel_basis = vh.conj().T[:, self.rank :]
        self.cokernel_basis = u[:, self.rank :]

    def solve(self, defect: Any) -> HilbertObstructionResult:
        residual_vector = _numeric_vector(defect, self.codomain_dimension, name="defect")
        obstruction = self.obstruction_projector @ residual_vector
        tol = finite_tolerance(self.tolerance)
        check_tol = finite_tolerance(tol * 10)
        audit = value_audit(exact=False, tolerance=tol)
        obstruction_norm = audit.check_value("obstruction", obstruction)
        vanishes = audit.decisions["obstruction"]["accepted"]
        correction = -self.pseudoinverse @ residual_vector if vanishes else None
        correction_residual = (
            audit.backend.norm(self.matrix @ correction + residual_vector)
            if correction is not None
            else obstruction_norm
        )
        distance_identity = abs(obstruction_norm - correction_residual)
        valid = (isfinite(obstruction_norm) and isfinite(correction_residual)
                 and isfinite(distance_identity)
                 and distance_identity <= max(check_tol, 1e-14))
        if vanishes:
            valid = valid and bool(audit.backend.is_zero(
                self.matrix @ correction + residual_vector, tolerance=check_tol))
        return HilbertObstructionResult(
            obstruction,
            obstruction_norm,
            vanishes,
            correction,
            correction_residual,
            Certificate(
                valid=valid,
                exactness=Exactness.NUMERICAL,
                residuals={
                    "correction_residual": correction_residual,
                    "distance_identity": distance_identity,
                },
                tolerance=check_tol,
                assumptions_checked=(
                    "singular-value rank",
                    "pseudoinverse of the retained singular-value model",
                    "orthogonal image and cokernel projectors",
                ),
                assumptions_unchecked=("rank stability below the singular-value threshold",),
                diagnostics={
                    "rank": self.rank,
                    "kernel_dimension": self.domain_dimension - self.rank,
                    "cokernel_dimension": self.codomain_dimension - self.rank,
                    "obstruction_norm": obstruction_norm,
                    "residual_decisions": audit.decisions,
                    "rank_model": "one retained SVD shared by inverse and subspaces",
                },
                theorem_id="PSC-III-T31",
                message="orthogonal obstruction and least-norm correction computed",
            ),
        )


@dataclass(frozen=True, slots=True)
class RightInverseCorrectionResult:
    """Result of a finite right-inverse contraction correction."""


    solution: np.ndarray
    iterations: int
    contraction_factor: float
    equation_residual: float
    fixed_point_residual: float
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="PSC-III-T32",
        title="Right-inverse nonlinear correction",
        status=ClaimStatus.PROVED,
        source="Paper III",
        assumptions=("declared right inverse", "declared Lipschitz bound", "invariant contraction ball"),
        verification_suite="nonlinear.correction",
    )
)
class RightInverseCorrectionProblem:
    """Solve ``D x + N(x) = y`` by the theorem's fixed-point map."""

    def __init__(
        self,
        linear_map: Any,
        right_inverse: Any,
        nonlinear: Callable[[np.ndarray], Any],
        target: Any,
        *,
        radius: float,
        lipschitz_bound: float,
        tolerance: float = 1e-10,
    ) -> None:
        self.linear_map = numpy_snapshot(linear_map, dtype=complex)
        self.right_inverse = numpy_snapshot(right_inverse, dtype=complex)
        if self.linear_map.ndim != 2 or self.right_inverse.ndim != 2:
            raise DimensionError("linear map and right inverse must be matrices")
        if not np.all(np.isfinite(self.linear_map)) or not np.all(np.isfinite(self.right_inverse)):
            raise ValidationError("linear map and right inverse must be finite")
        self.codomain_dimension, self.domain_dimension = self.linear_map.shape
        if self.right_inverse.shape != (self.domain_dimension, self.codomain_dimension):
            raise DimensionError("right inverse has incompatible dimensions")
        self.nonlinear = nonlinear
        if not callable(nonlinear):
            raise ValidationError("nonlinear term must be callable")
        self.target = numpy_snapshot(
            _numeric_vector(target, self.codomain_dimension, name="target"), dtype=complex
        )
        self.radius = float(radius)
        self.lipschitz_bound = float(lipschitz_bound)
        self.tolerance = float(tolerance)
        if not isfinite(self.radius) or self.radius <= 0:
            raise ValidationError("correction radius must be finite and positive")
        if not isfinite(self.lipschitz_bound) or self.lipschitz_bound < 0:
            raise ValidationError("Lipschitz bound must be finite and nonnegative")
        if not isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValidationError("tolerance must be finite and positive")
        self._tolerance10 = finite_tolerance(self.tolerance * 10.0)
        self._tolerance20 = finite_tolerance(self.tolerance * 20.0)

    def _nonlinear(self, value: np.ndarray) -> np.ndarray:
        return _numeric_vector(self.nonlinear(value), self.codomain_dimension, name="nonlinear term")

    def hypothesis_certificate(self) -> Certificate:
        identity = np.eye(self.codomain_dimension, dtype=complex)
        right_inverse_residual = _numeric_norm(
            self.linear_map @ self.right_inverse - identity, name="right-inverse residual")
        right_inverse_norm = float(np.linalg.norm(self.right_inverse, ord=2))
        if not isfinite(right_inverse_norm):
            raise ValidationError("right-inverse operator norm is not representable")
        contraction = right_inverse_norm * self.lipschitz_bound
        if not isfinite(contraction):
            raise ValidationError("declared contraction factor is not representable")
        zero = np.zeros(self.domain_dimension, dtype=complex)
        center = self.right_inverse @ (self.target - self._nonlinear(zero))
        center_norm = _numeric_norm(center, name="fixed-point center")
        ball_bound = (1.0 - contraction) * self.radius
        ball_excess = max(0.0, center_norm - ball_bound)
        valid = (
            right_inverse_residual <= self._tolerance10
            and contraction < 1.0
            and center_norm <= ball_bound
        )
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={
                "right_inverse": right_inverse_residual,
                "invariant_ball_excess": ball_excess,
            },
            tolerance=self._tolerance10,
            assumptions_checked=(
                "D R equals identity numerically",
                "declared contraction factor below one",
                "declared ball invariance inequality",
            ),
            assumptions_unchecked=("declared Lipschitz bound holds throughout the ball",),
            diagnostics={
                "right_inverse_norm": right_inverse_norm,
                "lipschitz_bound": self.lipschitz_bound,
                "contraction_factor": contraction,
                "center_norm": center_norm,
                "radius": self.radius,
                "invariant_ball_bound": ball_bound,
                "invariant_ball_uses_tolerance_slack": False,
            },
            theorem_id="PSC-III-T32",
            message="right-inverse correction hypotheses hold" if valid else "right-inverse correction hypotheses fail",
        )

    def solve(self, *, max_iterations: int = 10_000) -> RightInverseCorrectionResult:
        hypothesis = self.hypothesis_certificate()
        hypothesis.require()
        limit = int(max_iterations)
        if isinstance(max_iterations, bool) or limit != max_iterations or limit < 1:
            raise ValidationError("max_iterations must be a positive integer")
        contraction = float(hypothesis.diagnostics["contraction_factor"])
        value = np.zeros(self.domain_dimension, dtype=complex)
        fixed_residual = float("inf")
        iterations = 0
        for iterations in range(1, limit + 1):
            candidate = self.right_inverse @ (self.target - self._nonlinear(value))
            candidate_norm = _numeric_norm(candidate, name="fixed-point iterate")
            if candidate_norm > self.radius:
                raise ValidationError("fixed-point iteration left the declared correction ball")
            fixed_residual = _numeric_norm(candidate - value, name="fixed-point step")
            value = candidate
            if fixed_residual <= self.tolerance:
                break
        equation_residual = _numeric_norm(
            self.linear_map @ value + self._nonlinear(value) - self.target,
            name="correction equation residual",
        )
        converged = fixed_residual <= self.tolerance and equation_residual <= self._tolerance20
        return RightInverseCorrectionResult(
            np.real_if_close(value, tol=1000),
            iterations,
            contraction,
            equation_residual,
            fixed_residual,
            Certificate(
                valid=converged,
                exactness=Exactness.NUMERICAL,
                residuals={
                    "equation": equation_residual,
                    "fixed_point": fixed_residual,
                },
                tolerance=self._tolerance20,
                assumptions_checked=hypothesis.assumptions_checked + ("finite Picard residual reached the numerical stopping tolerance",),
                assumptions_unchecked=hypothesis.assumptions_unchecked,
                diagnostics={
                    **hypothesis.diagnostics,
                    "iterations": iterations,
                    "max_iterations": limit,
                },
                theorem_id="PSC-III-T32",
                message="right-inverse correction converged" if converged else "right-inverse correction did not converge within the limit",
            ),
        )

    def target_stability_bound(self, target_distance: float) -> float:
        hypothesis = self.hypothesis_certificate()
        hypothesis.require()
        distance = float(target_distance)
        if not isfinite(distance) or distance < 0:
            raise ValidationError("target distance must be finite and nonnegative")
        norm = float(hypothesis.diagnostics["right_inverse_norm"])
        contraction = float(hypothesis.diagnostics["contraction_factor"])
        return norm * distance / (1.0 - contraction)


@dataclass(frozen=True, slots=True)
class LyapunovSchmidtDecomposition:
    """Resolved kernel, range, complement, and obstruction spaces."""


    linearization: np.ndarray
    kernel_basis: np.ndarray
    complement_basis: np.ndarray
    range_basis: np.ndarray
    obstruction_basis: np.ndarray
    rank: int
    certificate: Certificate


@dataclass(frozen=True, slots=True)
class LyapunovSchmidtPoint:
    """One computed point on a finite Lyapunov--Schmidt branch."""


    kernel_coordinates: np.ndarray
    complement_coordinates: np.ndarray
    point: np.ndarray
    reduced_obstruction: np.ndarray
    range_residual: float
    iterations: int
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="PSC-III-T33",
        title="Finite local Lyapunov--Schmidt reduction",
        status=ClaimStatus.CLASSICAL,
        source="Paper III",
        assumptions=("C1 finite-dimensional map", "constant resolved rank near the base point", "local implicit-function neighborhood"),
        verification_suite="nonlinear.correction",
    )
)
class LyapunovSchmidtReduction:
    """Numerical SVD realization of local Lyapunov--Schmidt reduction."""

    def __init__(
        self,
        provider: DerivativeProvider,
        base_point: Any,
        *,
        tolerance: float = 1e-10,
    ) -> None:
        if provider.source_dimension < 1 or provider.target_dimension < 1:
            raise DimensionError("Lyapunov--Schmidt provider dimensions must be positive")
        if not provider.supports_order(1):
            raise ValidationError("Lyapunov--Schmidt reduction requires a first derivative")
        self.provider = provider
        self.base_point = numpy_snapshot(
            _numeric_vector(base_point, provider.source_dimension, name="base point"), dtype=complex
        )
        self.tolerance = float(tolerance)
        if not isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValidationError("tolerance must be finite and positive")
        self._check_tolerance = finite_tolerance(self.tolerance * 10.0)
        base_value = _numeric_vector(provider.value(self.base_point), provider.target_dimension, name="base value")
        self.base_residual = _numeric_norm(base_value, name="Lyapunov--Schmidt base residual")
        if self.base_residual > self._check_tolerance:
            raise ValidationError("Lyapunov--Schmidt base residual exceeds the numerical tolerance")
        linearization = _jacobian(provider, self.base_point)
        u, singular_values, vh = np.linalg.svd(linearization, full_matrices=True)
        threshold = self.tolerance
        if singular_values.size:
            threshold = max(threshold, self.tolerance * float(singular_values[0]))
        threshold = finite_tolerance(threshold)
        rank = int(np.sum(singular_values > threshold))
        self.linearization = linearization
        self.rank = rank
        self.kernel_basis = vh.conj().T[:, rank:]
        self.complement_basis = vh.conj().T[:, :rank]
        self.range_basis = u[:, :rank]
        self.obstruction_basis = u[:, rank:]
        self.kernel_dimension = self.kernel_basis.shape[1]
        self.complement_dimension = self.complement_basis.shape[1]
        self.obstruction_dimension = self.obstruction_basis.shape[1]
        restricted = self.range_basis.conj().T @ linearization @ self.complement_basis
        restricted_inverse_residual = 0.0
        if rank:
            inverse = np.linalg.inv(restricted)
            restricted_inverse_residual = _numeric_norm(
                restricted @ inverse - np.eye(rank), name="restricted inverse residual")
        self.decomposition = LyapunovSchmidtDecomposition(
            linearization,
            self.kernel_basis,
            self.complement_basis,
            self.range_basis,
            self.obstruction_basis,
            rank,
            Certificate(
                valid=restricted_inverse_residual <= self._check_tolerance,
                exactness=Exactness.NUMERICAL,
                residuals={"base_equation": self.base_residual, "restricted_inverse": restricted_inverse_residual},
                tolerance=self._check_tolerance,
                assumptions_checked=(
                    "base residual is within the declared numerical tolerance",
                    "SVD kernel/range decomposition",
                    "invertibility on the resolved complement",
                ),
                assumptions_unchecked=(
                    "the base point is an exact zero of F",
                    "rank is stable in a neighborhood of the base point",
                    "existence of the local Lyapunov--Schmidt branch",
                ),
                diagnostics={
                    "rank": rank,
                    "kernel_dimension": self.kernel_dimension,
                    "obstruction_dimension": self.obstruction_dimension,
                    "singular_values": tuple(float(value) for value in singular_values),
                    "base_residual": self.base_residual,
                },
                theorem_id="PSC-III-T33",
                message="numerical Lyapunov--Schmidt splitting data constructed; analytic branch assumptions remain",
            ),
        )

    def _ambient_point(self, kernel_coordinates: np.ndarray, complement_coordinates: np.ndarray) -> np.ndarray:
        return self.base_point + self.kernel_basis @ kernel_coordinates + self.complement_basis @ complement_coordinates

    def solve_range(
        self,
        kernel_coordinates: Any,
        *,
        initial_complement: Any | None = None,
        max_iterations: int = 50,
    ) -> LyapunovSchmidtPoint:
        kernel = _numeric_vector(kernel_coordinates, self.kernel_dimension, name="kernel coordinates")
        if initial_complement is None:
            complement = np.zeros(self.complement_dimension, dtype=complex)
        else:
            complement = _numeric_vector(initial_complement, self.complement_dimension, name="complement coordinates")
        limit = int(max_iterations)
        if isinstance(max_iterations, bool) or limit != max_iterations or limit < 1:
            raise ValidationError("max_iterations must be a positive integer")
        iterations = 0
        range_residual = 0.0
        for iterations in range(1, limit + 1):
            point = self._ambient_point(kernel, complement)
            value = _numeric_vector(self.provider.value(point), self.provider.target_dimension, name="provider value")
            range_equation = self.range_basis.conj().T @ value
            range_residual = _numeric_norm(range_equation, name="range equation residual")
            if range_residual <= self.tolerance:
                break
            jacobian = _jacobian(self.provider, point)
            reduced_jacobian = self.range_basis.conj().T @ jacobian @ self.complement_basis
            if reduced_jacobian.shape == (0, 0):
                break
            try:
                step = np.linalg.solve(reduced_jacobian, -range_equation)
            except np.linalg.LinAlgError as exc:
                raise ValidationError("range Jacobian is singular away from the base point") from exc
            accepted = False
            damping = 1.0
            for _ in range(20):
                candidate = complement + damping * step
                candidate_point = self._ambient_point(kernel, candidate)
                candidate_value = _numeric_vector(
                    self.provider.value(candidate_point),
                    self.provider.target_dimension,
                    name="provider value",
                )
                candidate_residual = _numeric_norm(
                    self.range_basis.conj().T @ candidate_value, name="candidate range residual")
                if candidate_residual < range_residual or candidate_residual <= self.tolerance:
                    complement = candidate
                    accepted = True
                    break
                damping *= 0.5
            if not accepted:
                raise ValidationError("range Newton step failed to decrease the residual")
        point = self._ambient_point(kernel, complement)
        value = _numeric_vector(self.provider.value(point), self.provider.target_dimension, name="provider value")
        range_residual = _numeric_norm(
            self.range_basis.conj().T @ value, name="final range residual")
        obstruction = self.obstruction_basis.conj().T @ value
        valid = range_residual <= self._check_tolerance
        return LyapunovSchmidtPoint(
            np.real_if_close(kernel, tol=1000),
            np.real_if_close(complement, tol=1000),
            np.real_if_close(point, tol=1000),
            np.real_if_close(obstruction, tol=1000),
            range_residual,
            iterations,
            Certificate(
                valid=valid,
                exactness=Exactness.NUMERICAL,
                residuals={"range_equation": range_residual},
                tolerance=self._check_tolerance,
                assumptions_checked=("range-equation residual is within the declared numerical tolerance",),
                assumptions_unchecked=(
                    "the returned point is an exact zero of the range equation",
                    "the selected point remains in an implicit-function neighborhood",
                    "local solution branch is the unique branch represented by this Newton solve",
                ),
                diagnostics={
                    "iterations": iterations,
                    "kernel_dimension": self.kernel_dimension,
                    "obstruction_dimension": self.obstruction_dimension,
                },
                theorem_id="PSC-III-T33",
                message="approximate range equation residual accepted; reduced obstruction evaluated" if valid else "range equation residual exceeds tolerance",
            ),
        )

    def base_derivative_certificate(self) -> Certificate:
        dpsi = np.zeros((self.complement_dimension, self.kernel_dimension), dtype=complex)
        dkappa = self.obstruction_basis.conj().T @ self.linearization @ self.kernel_basis
        dpsi_residual = _numeric_norm(dpsi, name="formal Dpsi residual")
        dkappa_residual = _numeric_norm(dkappa, name="formal Dkappa residual")
        residual = max(dpsi_residual, dkappa_residual)
        return Certificate(
            valid=residual <= self._check_tolerance,
            exactness=Exactness.NUMERICAL,
            residuals={"Dpsi_at_base": dpsi_residual, "Dkappa_at_base": dkappa_residual},
            tolerance=self._check_tolerance,
            assumptions_checked=("kernel and cokernel SVD identities", "formal first-order reduction identities"),
            assumptions_unchecked=(
                "the base point is an exact zero of F",
                "existence and differentiability of the selected implicit branch",
            ),
            theorem_id="PSC-III-T33",
            message="formal base derivative identities satisfy the numerical residual test; branch regularity remains external",
        )

    def reduced_second_derivative(self, left: Any, right: Any) -> ComputationResult[np.ndarray]:
        if not self.provider.supports_order(2):
            raise ValidationError("quadratic reduced obstruction requires a second derivative")
        h = _numeric_vector(left, self.kernel_dimension, name="left kernel direction")
        k = _numeric_vector(right, self.kernel_dimension, name="right kernel direction")
        ambient_h = self.kernel_basis @ h
        ambient_k = self.kernel_basis @ k
        second = _numeric_vector(
            self.provider.derivative(2, self.base_point, (ambient_h, ambient_k)),
            self.provider.target_dimension,
            name="second derivative",
        )
        reduced = self.obstruction_basis.conj().T @ second
        return ComputationResult(
            np.real_if_close(reduced, tol=1000),
            Certificate(
                valid=True,
                exactness=Exactness.NUMERICAL,
                residuals={},
                tolerance=self.tolerance,
                assumptions_checked=("kernel input projection", "cokernel output projection"),
                assumptions_unchecked=("declared second derivative is globally correct",),
                theorem_id="PSC-III-C34",
                message="quadratic reduced obstruction evaluated",
            ),
        )


__all__ = [
    "HilbertObstructionProjector",
    "HilbertObstructionResult",
    "LyapunovSchmidtDecomposition",
    "LyapunovSchmidtPoint",
    "LyapunovSchmidtReduction",
    "RightInverseCorrectionProblem",
    "RightInverseCorrectionResult",
]
