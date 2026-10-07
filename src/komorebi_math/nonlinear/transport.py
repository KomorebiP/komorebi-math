"""Smooth transport for finite moving orthogonal packet systems.

This module is domain independent.  The quantum namespace re-exports the same
objects for backward compatibility, while the nonlinear namespace owns the
shared implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..dynamics._linalg import matrix_exponential
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented


def _finite_square_matrix(value: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise DimensionError(f"{name} must be a finite square matrix")
    if matrix.size and not np.all(np.isfinite(matrix)):
        raise ValidationError(f"{name} must contain finite values")
    return matrix


def _orthogonal_packet_family(
    projectors: Iterable[Any],
    *,
    tolerance: float,
) -> tuple[np.ndarray, ...]:
    values = tuple(
        _finite_square_matrix(projector, name=f"packet projector {index}")
        for index, projector in enumerate(projectors)
    )
    if not values:
        raise ValidationError("at least one packet projector is required")
    tol = float(tolerance)
    if not np.isfinite(tol) or tol <= 0:
        raise ValidationError("packet tolerance must be positive and finite")
    dimension = values[0].shape[0]
    if any(projector.shape != (dimension, dimension) for projector in values):
        raise DimensionError("all packet projectors must act on one common space")
    zero = np.zeros((dimension, dimension), dtype=complex)
    for left_index, left in enumerate(values):
        if not np.allclose(left, left.conj().T, atol=tol, rtol=0.0):
            raise ValidationError("packet projectors must be self adjoint")
        if not np.allclose(left @ left, left, atol=tol, rtol=0.0):
            raise ValidationError("packet projectors must be idempotent")
        for right_index, right in enumerate(values):
            expected = left if left_index == right_index else zero
            if not np.allclose(left @ right, expected, atol=tol, rtol=0.0):
                raise ValidationError("packet projectors must be pairwise orthogonal")
    if not np.allclose(sum(values), np.eye(dimension, dtype=complex), atol=tol, rtol=0.0):
        raise ValidationError("packet projectors must sum to the identity")
    return values


@dataclass(frozen=True, slots=True)
class DirectRotationResult:
    """Polar direct rotation between two nearby packet systems."""

    unitary: np.ndarray
    minimum_singular_value: float
    certificate: Certificate


@dataclass(frozen=True, slots=True)
class PacketDerivativeResult:
    """Finite-difference validation of declared moving-packet derivatives."""

    time: float
    step: float
    maximum_derivative_residual: float
    tangent_residual: float
    partition_derivative_residual: float
    certificate: Certificate


@dataclass(frozen=True, slots=True)
class TransportRefinementResult:
    """A posteriori comparison of two midpoint transport resolutions."""

    coarse_steps: int
    fine_steps: int
    coarse: np.ndarray
    fine: np.ndarray
    difference: float
    endpoint_intertwining: float
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="PSC-III-T55",
        title="Direct rotation of finite packet systems",
        status=ClaimStatus.PROVED,
        source="Paper III shared runtime",
        assumptions=("two complete orthogonal packet systems", "invertible overlap sum"),
        verification_suite="nonlinear.transport",
    )
)
def direct_packet_rotation(
    source: Iterable[Any],
    target: Iterable[Any],
    *,
    tolerance: float = 1e-10,
) -> DirectRotationResult:
    """Return the polar unitary of ``sum target[a] source[a]``."""

    left = _orthogonal_packet_family(source, tolerance=tolerance)
    right = _orthogonal_packet_family(target, tolerance=tolerance)
    if len(left) != len(right) or left[0].shape != right[0].shape:
        raise DimensionError("source and target packet systems must have matching dimensions and counts")
    overlap = sum(q @ p for p, q in zip(left, right, strict=True))
    u, singular_values, vh = np.linalg.svd(overlap)
    minimum = float(np.min(singular_values))
    if minimum <= float(tolerance):
        raise ValidationError("packet overlap sum is singular or numerically unresolved")
    unitary = u @ vh
    identity = np.eye(unitary.shape[0], dtype=complex)
    unitary_residual = float(np.linalg.norm(unitary.conj().T @ unitary - identity, ord=2))
    intertwining = max(
        float(np.linalg.norm(q @ unitary - unitary @ p, ord=2))
        for p, q in zip(left, right, strict=True)
    )
    threshold = float(tolerance) * 20
    valid = max(unitary_residual, intertwining) <= threshold
    return DirectRotationResult(
        unitary,
        minimum,
        Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"unitarity": unitary_residual, "intertwining": intertwining},
            tolerance=threshold,
            assumptions_checked=("complete orthogonal packets", "invertible overlap sum"),
            theorem_id="PSC-III-T55",
            message="direct packet rotation is unitary and intertwines every packet",
        ),
    )


@documented(
    ProvenanceRecord(
        identifier="PSC-III-D56",
        title="Smooth finite packet system",
        status=ClaimStatus.DEFINITION,
        source="Paper III shared runtime",
        assumptions=("declared differentiable complete orthogonal packet family",),
        verification_suite="nonlinear.transport",
    )
)
class MovingPacketSystem:
    """A differentiable finite family of complete orthogonal projectors."""

    def __init__(
        self,
        projectors: Callable[[float], Iterable[Any]],
        derivatives: Callable[[float], Iterable[Any]],
        *,
        tolerance: float = 1e-9,
        reference_time: float = 0.0,
    ) -> None:
        if not callable(projectors) or not callable(derivatives):
            raise ValidationError("projectors and derivatives must be callable")
        tol = float(tolerance)
        reference = float(reference_time)
        if not np.isfinite(tol) or tol <= 0:
            raise ValidationError("moving packet tolerance must be positive and finite")
        if not np.isfinite(reference):
            raise ValidationError("reference_time must be finite")
        self._projectors = projectors
        self._derivatives = derivatives
        self.tolerance = tol
        self.reference_time = reference
        initial = self.packets(reference)
        self.dimension = initial[0].shape[0]
        self.count = len(initial)

    @staticmethod
    def _time(value: float, *, name: str = "time") -> float:
        time = float(value)
        if not np.isfinite(time):
            raise ValidationError(f"{name} must be finite")
        return time

    def packets(self, time: float) -> tuple[np.ndarray, ...]:
        """Evaluate and validate the packet family."""

        t = self._time(time)
        values = _orthogonal_packet_family(self._projectors(t), tolerance=self.tolerance)
        if hasattr(self, "count") and (len(values) != self.count or values[0].shape[0] != self.dimension):
            raise DimensionError("moving packet dimension or packet count changed")
        return values

    def packet_derivatives(self, time: float) -> tuple[np.ndarray, ...]:
        """Evaluate the declared packet derivatives."""

        t = self._time(time)
        values = tuple(
            _finite_square_matrix(value, name=f"packet derivative {index}")
            for index, value in enumerate(self._derivatives(t))
        )
        if len(values) != self.count or any(value.shape != (self.dimension, self.dimension) for value in values):
            raise DimensionError("packet derivatives must match the moving packet family")
        return values

    def derivative_certificate(self, time: float, *, step: float = 1e-6) -> PacketDerivativeResult:
        """Compare declared derivatives with centered finite differences.

        The residual is numerical evidence, not a proof of differentiability.
        Algebraic tangent and partition-of-unity identities are checked at the
        evaluation point independently of the finite difference.
        """

        t = self._time(time)
        h = float(step)
        if not np.isfinite(h) or h <= 0:
            raise ValidationError("derivative validation step must be positive and finite")
        current = self.packets(t)
        declared = self.packet_derivatives(t)
        forward = self.packets(t + h)
        backward = self.packets(t - h)
        finite_differences = tuple((right - left) / (2 * h) for right, left in zip(forward, backward, strict=True))
        derivative_residual = max(
            float(np.linalg.norm(exact - approximate, ord=2))
            for exact, approximate in zip(declared, finite_differences, strict=True)
        )
        tangent_residual = max(
            float(np.linalg.norm(derivative - derivative @ projector - projector @ derivative, ord=2))
            for derivative, projector in zip(declared, current, strict=True)
        )
        partition_residual = float(np.linalg.norm(sum(declared), ord=2))
        threshold = max(self.tolerance * 50, h * h * 50)
        valid = max(derivative_residual, tangent_residual, partition_residual) <= threshold
        return PacketDerivativeResult(
            t,
            h,
            derivative_residual,
            tangent_residual,
            partition_residual,
            Certificate(
                valid=valid,
                exactness=Exactness.NUMERICAL,
                residuals={
                    "finite_difference": derivative_residual,
                    "projector_tangent": tangent_residual,
                    "partition_derivative": partition_residual,
                },
                tolerance=threshold,
                assumptions_checked=(
                    "centered finite-difference comparison",
                    "differentiated projector identity",
                    "differentiated partition of unity",
                ),
                assumptions_unchecked=("analytic differentiability beyond the sampled point",),
                theorem_id="PSC-III-D56",
                message="declared packet derivatives satisfy sampled smoothness identities",
            ),
        )

    def canonical_generator(self, time: float) -> np.ndarray:
        """Return the canonical skew generator ``sum dot(P_a) P_a``."""

        packets = self.packets(time)
        derivatives = self.packet_derivatives(time)
        return sum(
            derivative @ projector
            for derivative, projector in zip(derivatives, packets, strict=True)
        )

    def generator_certificate(self, time: float) -> Certificate:
        """Check skew adjointness and the projector commutator identity."""

        packets = self.packets(time)
        derivatives = self.packet_derivatives(time)
        generator = self.canonical_generator(time)
        skew = float(np.linalg.norm(generator + generator.conj().T, ord=2))
        commutator = max(
            float(np.linalg.norm(derivative - (generator @ projector - projector @ generator), ord=2))
            for derivative, projector in zip(derivatives, packets, strict=True)
        )
        threshold = self.tolerance * 20
        return Certificate(
            valid=max(skew, commutator) <= threshold,
            exactness=Exactness.NUMERICAL,
            residuals={"skew_adjoint": skew, "projector_commutator": commutator},
            tolerance=threshold,
            assumptions_checked=("packet axioms at the sampled time", "sampled skew-adjoint generator", "sampled projector commutator identities"),
            assumptions_unchecked=("declared packet derivatives are actual time derivatives in a neighborhood of the sampled time",),
            theorem_id="PSC-III-T57",
            message="the canonical generator transports the moving packets",
        )


@documented(
    ProvenanceRecord(
        identifier="PSC-III-T58",
        title="Finite smooth packet transport",
        status=ClaimStatus.PROVED,
        source="Paper III shared runtime",
        assumptions=("differentiable complete orthogonal packet family",),
        verification_suite="nonlinear.transport",
    )
)
class KatoTransport:
    """Numerical finite Kato transport for a moving packet system."""

    def __init__(self, system: MovingPacketSystem) -> None:
        if not isinstance(system, MovingPacketSystem):
            raise ValidationError("KatoTransport requires a MovingPacketSystem")
        self.system = system

    @staticmethod
    def _steps(value: int) -> int:
        count = int(value)
        if isinstance(value, bool) or count != value or count < 1:
            raise ValidationError("steps must be a positive integer")
        return count

    def evolve(self, start: float, stop: float, *, steps: int = 256) -> ComputationResult[np.ndarray]:
        """Integrate the canonical generator by midpoint exponential steps."""

        count = self._steps(steps)
        t0 = self.system._time(start, name="start")
        t1 = self.system._time(stop, name="stop")
        delta = (t1 - t0) / count
        unitary = np.eye(self.system.dimension, dtype=complex)
        generator_failures = 0
        for index in range(count):
            midpoint = t0 + (index + 0.5) * delta
            generator = self.system.canonical_generator(midpoint)
            if np.linalg.norm(generator + generator.conj().T, ord=2) > self.system.tolerance * 20:
                generator_failures += 1
            unitary = matrix_exponential(delta * generator) @ unitary
        source = self.system.packets(t0)
        target = self.system.packets(t1)
        intertwining = max(
            float(np.linalg.norm(q @ unitary - unitary @ p, ord=2))
            for p, q in zip(source, target, strict=True)
        )
        unitarity = float(
            np.linalg.norm(
                unitary.conj().T @ unitary - np.eye(self.system.dimension, dtype=complex),
                ord=2,
            )
        )
        tolerance = max(self.system.tolerance * 50, abs(delta) ** 2 * 20)
        valid = generator_failures == 0 and max(intertwining, unitarity) <= tolerance
        return ComputationResult(
            unitary,
            Certificate(
                valid=valid,
                exactness=Exactness.NUMERICAL,
                residuals={
                    "intertwining": intertwining,
                    "unitarity": unitarity,
                    "generator_failures": float(generator_failures),
                },
                tolerance=tolerance,
                assumptions_checked=("midpoint integration of the sampled canonical generator",),
                assumptions_unchecked=(
                    "declared packet derivatives are actual time derivatives throughout the integration interval",
                    "global higher-order integration error beyond residual checks",
                ),
                diagnostics={"steps": count, "step_size": delta},
                theorem_id="PSC-III-T58",
                message="finite smooth transport intertwines the endpoint packet systems",
            ),
        )

    def direct_rotation_product(self, times: Iterable[float]) -> ComputationResult[np.ndarray]:
        """Multiply direct rotations along a finite increasing time partition."""

        values = tuple(self.system._time(time) for time in times)
        if len(values) < 2 or any(values[index] >= values[index + 1] for index in range(len(values) - 1)):
            raise ValidationError("times must be a strictly increasing sequence with at least two points")
        unitary = np.eye(self.system.dimension, dtype=complex)
        minimum = float("inf")
        for left_time, right_time in zip(values[:-1], values[1:], strict=True):
            step = direct_packet_rotation(
                self.system.packets(left_time),
                self.system.packets(right_time),
                tolerance=self.system.tolerance,
            )
            step.certificate.require()
            minimum = min(minimum, step.minimum_singular_value)
            unitary = step.unitary @ unitary
        source = self.system.packets(values[0])
        target = self.system.packets(values[-1])
        residual = max(
            float(np.linalg.norm(q @ unitary - unitary @ p, ord=2))
            for p, q in zip(source, target, strict=True)
        )
        threshold = self.system.tolerance * 50
        return ComputationResult(
            unitary,
            Certificate(
                valid=residual <= threshold,
                exactness=Exactness.NUMERICAL,
                residuals={"endpoint_intertwining": residual},
                tolerance=threshold,
                assumptions_checked=("invertible overlap at every partition step",),
                diagnostics={"minimum_step_singular_value": minimum, "steps": len(values) - 1},
                theorem_id="PSC-III-T59",
                message="the direct-rotation product intertwines the endpoint packets",
            ),
        )

    def refinement_certificate(
        self,
        start: float,
        stop: float,
        *,
        coarse_steps: int = 64,
        refinement_factor: int = 2,
        tolerance: float | None = None,
    ) -> TransportRefinementResult:
        """Compare two midpoint discretizations and certify sampled convergence."""

        coarse_count = self._steps(coarse_steps)
        factor = int(refinement_factor)
        if isinstance(refinement_factor, bool) or factor != refinement_factor or factor < 2:
            raise ValidationError("refinement_factor must be an integer at least two")
        fine_count = coarse_count * factor
        coarse = self.evolve(start, stop, steps=coarse_count)
        fine = self.evolve(start, stop, steps=fine_count)
        difference = float(np.linalg.norm(fine.value - coarse.value, ord=2))
        source = self.system.packets(float(start))
        target = self.system.packets(float(stop))
        endpoint = max(
            float(np.linalg.norm(q @ fine.value - fine.value @ p, ord=2))
            for p, q in zip(source, target, strict=True)
        )
        threshold = (
            max(coarse.certificate.tolerance or 0.0, fine.certificate.tolerance or 0.0) * 2
            if tolerance is None
            else float(tolerance)
        )
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValidationError("refinement tolerance must be positive and finite")
        valid = coarse.certificate.valid and fine.certificate.valid and difference <= threshold and endpoint <= threshold
        return TransportRefinementResult(
            coarse_count,
            fine_count,
            coarse.value,
            fine.value,
            difference,
            endpoint,
            Certificate(
                valid=valid,
                exactness=Exactness.NUMERICAL,
                residuals={"coarse_fine_difference": difference, "endpoint_intertwining": endpoint},
                tolerance=threshold,
                assumptions_checked=("nested midpoint discretizations", "endpoint residual comparison"),
                assumptions_unchecked=tuple(dict.fromkeys(
                    coarse.certificate.assumptions_unchecked
                    + fine.certificate.assumptions_unchecked
                    + ("asymptotic convergence outside the two sampled resolutions",)
                )),
                diagnostics={"refinement_factor": factor},
                theorem_id="PSC-III-C60",
                message="nested smooth-transport discretizations agree within tolerance",
            ),
        )


__all__ = [
    "DirectRotationResult",
    "KatoTransport",
    "MovingPacketSystem",
    "PacketDerivativeResult",
    "TransportRefinementResult",
    "direct_packet_rotation",
]
