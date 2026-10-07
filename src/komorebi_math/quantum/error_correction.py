"""Finite quantum error correction through Knill Laflamme compression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._validation import as_square_matrix, density_matrix, require_projector
from .channels import QuantumChannel


@dataclass(frozen=True, slots=True)
class KnillLaflammeReport:
    """Compressed error matrix and residuals for one finite code."""

    coefficient_matrix: np.ndarray
    residual_matrix: np.ndarray
    maximum_residual: float
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="PSC-VII-T42",
        title="Knill Laflamme compression criterion",
        status=ClaimStatus.CLASSICAL,
        source="Paper VII",
        assumptions=("finite orthogonal code projector", "finite error family"),
        verification_suite="quantum.error_correction",
    )
)
class QuantumCode:
    """A finite code projector with a declared family of error operators."""

    def __init__(self, projector: Any, errors: Iterable[Any], *, tolerance: float = 1e-10) -> None:
        p = require_projector(projector, tolerance=tolerance, name="code projector")
        operators = tuple(as_square_matrix(value, name="error operator") for value in errors)
        if not operators:
            raise ValidationError("a quantum code requires at least one error operator")
        if any(operator.shape != p.shape for operator in operators):
            raise DimensionError("every error must act on the code Hilbert space")
        rank = int(round(float(np.trace(p).real)))
        if rank < 1:
            raise ValidationError("code projector must have positive rank")
        self.projector = p
        self.errors = operators
        self.dimension = p.shape[0]
        self.code_dimension = rank
        self.tolerance = float(tolerance)

    def _orthonormal_error_span(self) -> tuple[tuple[np.ndarray, ...], np.ndarray, float]:
        """Return a Hilbert--Schmidt orthonormal basis of the declared error span.

        The numerical rank threshold scales with the largest singular value, so
        uniform rescaling of every declared generator does not change the
        resolved span.  Extremely ill-conditioned generating families remain a
        finite-precision boundary and their singular spectrum is exposed in the
        Knill--Laflamme certificate diagnostics.
        """

        vectors = np.column_stack([error.reshape(-1) for error in self.errors])
        left_vectors, singular_values, _ = np.linalg.svd(vectors, full_matrices=False)
        if singular_values.size == 0 or float(singular_values[0]) == 0.0:
            return (), singular_values, 0.0
        threshold = (
            np.finfo(float).eps
            * max(vectors.shape)
            * float(singular_values[0])
        )
        rank = int(np.sum(singular_values > threshold))
        basis = tuple(
            left_vectors[:, index].reshape(self.dimension, self.dimension)
            for index in range(rank)
        )
        return basis, singular_values, threshold

    def knill_laflamme_report(self) -> KnillLaflammeReport:
        """Return Knill--Laflamme data with an error-span invariant certificate.

        ``coefficient_matrix``, ``residual_matrix``, and ``maximum_residual`` retain
        the declared error-family coordinates for diagnostics.  Certificate validity
        is intentionally *not* based on the largest declared-coordinate residual:
        that quantity changes under harmless unitary mixing of Kraus/error labels.

        Instead, the declared operators are first replaced by an orthonormal
        Hilbert--Schmidt basis of their linear span.  The residual operators in that
        basis are assembled into one block operator.  Its spectral norm is invariant
        under unitary changes of orthonormal error-span basis, so the numerical
        obstruction is a property of the represented error space rather than of one
        arbitrary decomposition.
        """

        count = len(self.errors)
        coefficients = np.zeros((count, count), dtype=complex)
        residuals = np.zeros((count, count), dtype=float)
        for left, error_left in enumerate(self.errors):
            for right, error_right in enumerate(self.errors):
                compressed = self.projector @ error_left.conj().T @ error_right @ self.projector
                scalar = np.trace(compressed) / self.code_dimension
                coefficients[left, right] = scalar
                residuals[left, right] = np.linalg.norm(compressed - scalar * self.projector, ord=2)
        maximum = float(np.max(residuals))

        # Correctability depends on the linear span of the declared errors, not on
        # a particular generating family.  Resolve that span with a scale-covariant
        # numerical-rank threshold and use an orthonormal Hilbert--Schmidt basis.
        basis, singular_values, rank_threshold = self._orthonormal_error_span()
        span_rank = len(basis)

        if span_rank == 0:
            invariant_obstruction = 0.0
            invariant_scale = 1.0
        else:
            span_coefficients = np.zeros((span_rank, span_rank), dtype=complex)
            obstruction = np.zeros(
                (span_rank * self.dimension, span_rank * self.dimension),
                dtype=complex,
            )
            for left, error_left in enumerate(basis):
                for right, error_right in enumerate(basis):
                    compressed = self.projector @ error_left.conj().T @ error_right @ self.projector
                    scalar = np.trace(compressed) / self.code_dimension
                    span_coefficients[left, right] = scalar
                    residual = compressed - scalar * self.projector
                    row = slice(left * self.dimension, (left + 1) * self.dimension)
                    column = slice(right * self.dimension, (right + 1) * self.dimension)
                    obstruction[row, column] = residual
            invariant_obstruction = float(np.linalg.norm(obstruction, ord=2))
            invariant_scale = max(1.0, float(np.linalg.norm(span_coefficients, ord=2)))

        return KnillLaflammeReport(
            coefficients,
            residuals,
            maximum,
            Certificate(
                valid=invariant_obstruction <= self.tolerance * invariant_scale,
                exactness=Exactness.NUMERICAL,
                residuals={
                    "compression_obstruction": invariant_obstruction,
                    "declared_basis_max_residual": maximum,
                },
                tolerance=self.tolerance * invariant_scale,
                assumptions_checked=(
                    "orthogonal code projector",
                    "finite error family",
                    "Hilbert Schmidt error span resolved by numerical SVD",
                ),
                diagnostics={
                    "declared_error_count": count,
                    "error_span_rank": span_rank,
                    "error_span_rank_threshold": rank_threshold,
                    "error_span_singular_values": tuple(float(value) for value in singular_values),
                    "obstruction_norm": "spectral norm of the full Knill-Laflamme residual block operator",
                    "declared_basis_max_residual_is_gauge_invariant": False,
                },
                theorem_id="PSC-VII-T42",
                message="error products compress to scalars on the code error span",
            ),
        )

    def recovery_channel(self) -> QuantumChannel:
        """Construct a trace preserving recovery channel when the criterion holds."""

        report = self.knill_laflamme_report()
        report.certificate.require()
        basis, _, _ = self._orthonormal_error_span()
        span_rank = len(basis)
        span_coefficients = np.zeros((span_rank, span_rank), dtype=complex)
        for left, error_left in enumerate(basis):
            for right, error_right in enumerate(basis):
                compressed = self.projector @ error_left.conj().T @ error_right @ self.projector
                span_coefficients[left, right] = np.trace(compressed) / self.code_dimension
        hermitian = (span_coefficients + span_coefficients.conj().T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
        if eigenvalues.size == 0 or float(np.max(eigenvalues)) <= 0.0:
            active_threshold = 0.0
        else:
            active_threshold = (
                np.finfo(float).eps
                * max(1, span_rank)
                * float(np.max(eigenvalues))
            )
        active = [index for index, value in enumerate(eigenvalues) if value > active_threshold]
        recovery_kraus: list[np.ndarray] = []
        syndrome_projector = np.zeros_like(self.projector)
        for index in active:
            coefficients = eigenvectors[:, index]
            transformed = sum(basis[error] * coefficients[error] for error in range(span_rank))
            isometry = transformed @ self.projector / np.sqrt(eigenvalues[index])
            syndrome = isometry @ isometry.conj().T
            syndrome_projector += syndrome
            recovery_kraus.append(isometry.conj().T)
        complement = (np.eye(self.dimension, dtype=complex) - syndrome_projector)
        complement = (complement + complement.conj().T) / 2
        values, vectors = np.linalg.eigh(complement)
        code_values, code_vectors = np.linalg.eigh(self.projector)
        code_vector = code_vectors[:, int(np.argmax(code_values))]
        for value, vector in zip(values, vectors.T, strict=True):
            if value > self.tolerance:
                recovery_kraus.append(np.sqrt(value) * np.outer(code_vector, vector.conj()))
        return QuantumChannel(recovery_kraus, tolerance=max(self.tolerance, 1e-9))

    def recovered_channel_certificate(self) -> Certificate:
        """Check exact recovery of the equal weight error channel on code matrix units."""

        recovery = self.recovery_channel()
        noise = QuantumChannel(
            tuple(error / np.sqrt(len(self.errors)) for error in self.errors),
            tolerance=max(self.tolerance, 1e-9),
        )
        noise_certificate = noise.certificate()
        if not noise_certificate.valid:
            trace_residual = float(noise_certificate.residuals.get("trace_preservation", float("inf")))
            return Certificate(
                valid=False,
                exactness=Exactness.NUMERICAL,
                residuals={"noise_trace_preservation": trace_residual},
                tolerance=noise_certificate.tolerance,
                assumptions_checked=("Knill Laflamme compression",),
                assumptions_unchecked=("equal weight error family is trace preserving",),
                theorem_id="PSC-VII-T44",
                message="the declared equal weight error family does not define a quantum channel",
            )
        combined = recovery.compose(noise)
        code_values, code_vectors = np.linalg.eigh(self.projector)
        basis = code_vectors[:, code_values > 0.5]
        residual = 0.0
        for row in range(self.code_dimension):
            for column in range(self.code_dimension):
                unit = np.outer(basis[:, row], basis[:, column].conj())
                residual = max(residual, float(np.linalg.norm(combined.apply(unit) - unit, ord=2)))
        return Certificate(
            valid=residual <= max(self.tolerance * 100, 1e-8),
            exactness=Exactness.NUMERICAL,
            residuals={"recovered_code_matrix_units": residual},
            tolerance=max(self.tolerance * 100, 1e-8),
            assumptions_checked=("Knill Laflamme compression", "equal weight error channel"),
            theorem_id="PSC-VII-T44",
            message="constructed recovery restores every code matrix unit",
        )

    def encode_density(self, logical_state: Any) -> np.ndarray:
        """Embed a logical density matrix into the code subspace."""

        logical = density_matrix(logical_state, tolerance=self.tolerance)
        if logical.shape != (self.code_dimension, self.code_dimension):
            raise DimensionError("logical state dimension does not match the code rank")
        values, vectors = np.linalg.eigh(self.projector)
        basis = vectors[:, values > 0.5]
        return basis @ logical @ basis.conj().T


__all__ = ["KnillLaflammeReport", "QuantumCode"]
