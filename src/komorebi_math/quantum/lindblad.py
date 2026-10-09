"""Finite Lindblad generators and cyclic coherence packets."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .._residuals import finite_tolerance
from ..certificates import Certificate, Exactness
from ..dynamics._linalg import matrix_exponential
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._validation import as_square_matrix, density_matrix, require_hermitian, require_unitary


@documented(
    ProvenanceRecord(
        identifier="PSC-VII-T38",
        title="Lindblad coherence packet preservation",
        status=ClaimStatus.PROVED,
        source="Paper VII",
        assumptions=("finite Lindblad generator", "cyclic Hamiltonian symmetry", "charged jump operators"),
        verification_suite="quantum.lindblad",
    )
)
class LindbladGenerator:
    """A finite Gorini Kossakowski Sudarshan Lindblad generator."""

    def __init__(self, hamiltonian: Any, jumps: Iterable[Any], *, tolerance: float = 1e-10) -> None:
        tolerance = finite_tolerance(tolerance)
        h = require_hermitian(hamiltonian, tolerance=tolerance, name="Hamiltonian")
        operators = tuple(as_square_matrix(value, name="Lindblad jump") for value in jumps)
        if any(operator.shape != h.shape for operator in operators):
            raise DimensionError("every jump operator must act on the Hamiltonian space")
        # Validated state must not change when the caller edits its input arrays.
        self.hamiltonian = np.array(h, copy=True)
        self.jumps = tuple(np.array(operator, copy=True) for operator in operators)
        self.hamiltonian.setflags(write=False)
        for operator in self.jumps:
            operator.setflags(write=False)
        self.dimension = h.shape[0]
        self.tolerance = float(tolerance)

    def _validate_form(self) -> None:
        """Recheck the represented form, including publicly reassigned attributes."""

        h = require_hermitian(self.hamiltonian, tolerance=self.tolerance, name="Hamiltonian")
        if h.shape != (self.dimension, self.dimension):
            raise DimensionError("Hamiltonian has changed dimension")
        for jump in self.jumps:
            operator = as_square_matrix(jump, name="Lindblad jump")
            if operator.shape != h.shape:
                raise DimensionError("every jump operator must act on the Hamiltonian space")

    def apply(self, operator: Any) -> np.ndarray:
        """Apply the Lindblad generator to one operator."""

        self._validate_form()
        value = as_square_matrix(operator, name="Lindblad input")
        if value.shape != (self.dimension, self.dimension):
            raise DimensionError("Lindblad input has the wrong dimension")
        result = -1j * (self.hamiltonian @ value - value @ self.hamiltonian)
        for jump in self.jumps:
            effect = jump.conj().T @ jump
            result += jump @ value @ jump.conj().T - 0.5 * (effect @ value + value @ effect)
        if not np.all(np.isfinite(result)):
            raise ValidationError("Lindblad application overflowed finite floating-point range")
        return result

    def superoperator(self) -> np.ndarray:
        """Return the column vectorized generator matrix."""

        self._validate_form()
        identity = np.eye(self.dimension, dtype=complex)
        result = -1j * (
            np.kron(identity, self.hamiltonian)
            - np.kron(self.hamiltonian.T, identity)
        )
        for jump in self.jumps:
            effect = jump.conj().T @ jump
            result += np.kron(jump.conj(), jump)
            result -= 0.5 * np.kron(identity, effect)
            result -= 0.5 * np.kron(effect.T, identity)
        if not np.all(np.isfinite(result)):
            raise ValidationError("Lindblad superoperator overflowed finite floating-point range")
        return result

    def evolve(self, state: Any, time: float) -> np.ndarray:
        """Evolve a density matrix for one finite time."""

        rho = density_matrix(state, tolerance=self.tolerance)
        if rho.shape != (self.dimension, self.dimension):
            raise DimensionError("Lindblad state has the wrong dimension")
        duration = float(time)
        if not np.isfinite(duration):
            raise ValidationError("Lindblad evolution time must be finite")
        vector = rho.reshape(-1, order="F")
        evolved = matrix_exponential(duration * self.superoperator()) @ vector
        result = evolved.reshape((self.dimension, self.dimension), order="F")
        return density_matrix(result, tolerance=max(self.tolerance, 1e-8), normalize=True)

    def trace_preservation_certificate(self) -> Certificate:
        """Check that the generator has zero trace on every matrix unit."""

        residual = 0.0
        for row in range(self.dimension):
            for column in range(self.dimension):
                unit = np.zeros((self.dimension, self.dimension), dtype=complex)
                unit[row, column] = 1
                derivative = complex(np.trace(self.apply(unit)))
                if not np.isfinite(derivative):
                    raise ValidationError("Lindblad trace derivative became nonfinite")
                residual = max(residual, abs(derivative))
        return Certificate(
            valid=residual <= self.tolerance,
            exactness=Exactness.NUMERICAL,
            residuals={"trace_derivative": float(residual)},
            tolerance=self.tolerance,
            assumptions_checked=("finite Lindblad form",),
            theorem_id="PSC-VII-T37",
            message="Lindblad generator preserves trace",
        )

    def coherence_preservation_certificate(
        self,
        symmetry: Any,
        order: int,
    ) -> Certificate:
        """Check preservation of every cyclic coherence order."""

        unitary = require_unitary(symmetry, tolerance=self.tolerance, name="cyclic symmetry")
        if unitary.shape != (self.dimension, self.dimension):
            raise DimensionError("symmetry and Lindblad dimensions do not agree")
        n = int(order)
        if isinstance(order, bool) or n != order or n < 1:
            raise ValidationError("order must be a positive integer")
        identity = np.eye(self.dimension, dtype=complex)
        closure_residual = float(np.linalg.norm(np.linalg.matrix_power(unitary, n) - identity, ord=2))
        closure_tolerance = self.tolerance * max(1.0, float(n))

        conjugation = np.kron(unitary.conj(), unitary)
        generator = self.superoperator()
        generator_scale = max(1.0, float(np.linalg.norm(generator, ord=2)))
        commutator_tolerance = self.tolerance * generator_scale
        residual = float(np.linalg.norm(conjugation @ generator - generator @ conjugation, ord=2))
        return Certificate(
            valid=(
                closure_residual <= closure_tolerance
                and residual <= commutator_tolerance
            ),
            exactness=Exactness.NUMERICAL,
            residuals={
                "cyclic_closure": closure_residual,
                "coherence_commutator": residual,
            },
            tolerance=max(closure_tolerance, commutator_tolerance),
            assumptions_checked=("unitary symmetry",),
            assumptions_unchecked=() if closure_residual <= closure_tolerance else ("U^order = I cyclic closure",),
            diagnostics={
                "order": n,
                "cyclic_closure_tolerance": closure_tolerance,
                "coherence_commutator_tolerance": commutator_tolerance,
            },
            theorem_id="PSC-VII-T38",
            message=(
                "Lindblad generator preserves cyclic coherence order"
                if closure_residual <= closure_tolerance
                else "declared symmetry does not close at the requested cyclic order"
            ),
        )


__all__ = ["LindbladGenerator"]
