"""Finite Schur Feshbach reduction and exact hidden state memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._linalg import matrix_exponential
from .memory import MemoryKernel


def _indices(values: Iterable[int], dimension: int) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValidationError("retained indices must be nonempty and unique")
    if any(value < 0 or value >= dimension for value in result):
        raise ValidationError("retained index lies outside the matrix")
    return result


@documented(
    ProvenanceRecord(
        identifier="PSC-VI-D27",
        title="Finite Feshbach decomposition",
        status=ClaimStatus.DEFINITION,
        source="Paper VI",
        assumptions=("finite matrix", "declared retained coordinates"),
        verification_suite="dynamics.feshbach",
    )
)
@dataclass(frozen=True, slots=True)
class FeshbachReduction:
    """Retained and hidden block reduction of a finite matrix."""

    matrix: np.ndarray
    retained: tuple[int, ...]
    hidden: tuple[int, ...]

    def __init__(self, matrix: Any, retained: Iterable[int]) -> None:
        value = np.asarray(matrix)
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise DimensionError("Feshbach matrix must be square")
        if value.size and not np.all(np.isfinite(value)):
            raise ValidationError("Feshbach matrix must contain finite values")
        retained_indices = _indices(retained, value.shape[0])
        hidden_indices = tuple(index for index in range(value.shape[0]) if index not in retained_indices)
        if not hidden_indices:
            raise ValidationError("Feshbach reduction requires at least one hidden coordinate")
        object.__setattr__(self, "matrix", value)
        object.__setattr__(self, "retained", retained_indices)
        object.__setattr__(self, "hidden", hidden_indices)

    def _block(self, rows: tuple[int, ...], columns: tuple[int, ...]) -> np.ndarray:
        return self.matrix[np.ix_(rows, columns)]

    @property
    def retained_block(self) -> np.ndarray:
        return self._block(self.retained, self.retained)

    @property
    def hidden_block(self) -> np.ndarray:
        return self._block(self.hidden, self.hidden)

    @property
    def retained_to_hidden(self) -> np.ndarray:
        """Return the top right coupling ``A_PQ``.

        The historic public name is retained even though the matrix maps a
        hidden coordinate vector into the retained equation.
        """

        return self._block(self.retained, self.hidden)

    @property
    def hidden_to_retained(self) -> np.ndarray:
        """Return the bottom left coupling ``A_QP``."""

        return self._block(self.hidden, self.retained)

    @property
    def a_pq(self) -> np.ndarray:
        return self.retained_to_hidden

    @property
    def a_qp(self) -> np.ndarray:
        return self.hidden_to_retained

    def self_energy(self, spectral_parameter: complex) -> np.ndarray:
        z = complex(spectral_parameter)
        identity = np.eye(len(self.hidden), dtype=np.result_type(self.matrix, z))
        resolvent = np.linalg.inv(z * identity - self.hidden_block)
        return self.a_pq @ resolvent @ self.a_qp

    def effective_operator(self, spectral_parameter: complex) -> np.ndarray:
        return self.retained_block + self.self_energy(spectral_parameter)

    def effective_resolvent(self, spectral_parameter: complex) -> np.ndarray:
        z = complex(spectral_parameter)
        identity = np.eye(len(self.retained), dtype=np.result_type(self.matrix, z))
        return np.linalg.inv(z * identity - self.effective_operator(z))

    def direct_projected_resolvent(self, spectral_parameter: complex) -> np.ndarray:
        z = complex(spectral_parameter)
        identity = np.eye(self.matrix.shape[0], dtype=np.result_type(self.matrix, z))
        full = np.linalg.inv(z * identity - self.matrix)
        return full[np.ix_(self.retained, self.retained)]

    @documented(
        ProvenanceRecord(
            identifier="PSC-VI-T28",
            title="Finite Schur Feshbach resolvent",
            status=ClaimStatus.PROVED,
            source="Paper VI",
            assumptions=("full and hidden resolvents exist",),
            verification_suite="dynamics.feshbach",
        )
    )
    def resolvent_certificate(
        self,
        spectral_parameter: complex,
        *,
        tolerance: float = 1e-10,
    ) -> ComputationResult[np.ndarray]:
        effective = self.effective_resolvent(spectral_parameter)
        direct = self.direct_projected_resolvent(spectral_parameter)
        residual = float(np.linalg.norm(effective - direct, ord=2))
        scale = max(1.0, float(np.linalg.norm(direct, ord=2)))
        valid = residual <= tolerance * scale
        return ComputationResult(
            effective,
            Certificate(
                valid=valid,
                exactness=Exactness.NUMERICAL,
                residuals={"feshbach_resolvent": residual},
                tolerance=tolerance,
                assumptions_checked=("hidden resolvent exists", "full resolvent exists"),
                theorem_id="PSC-VI-T28",
                message="Feshbach resolvent verified" if valid else "Feshbach resolvent identity failed",
            ),
        )

    def memory_kernel(self) -> MemoryKernel:
        """Return the exact retained memory kernel ``A_PQ exp(tA_QQ) A_QP``."""

        return MemoryKernel(self.a_pq, self.hidden_block, self.a_qp)

    def memory_kernel_at(self, time: float) -> np.ndarray:
        return self.memory_kernel().at(time)

    def hidden_source(self, hidden_initial: Any, time: float) -> np.ndarray:
        value = np.asarray(hidden_initial)
        if value.ndim != 1 or value.size != len(self.hidden):
            raise DimensionError("hidden initial state has the wrong dimension")
        return self.a_pq @ matrix_exponential(float(time) * self.hidden_block) @ value

    def closure_moments(self) -> tuple[np.ndarray, ...]:
        """Return the finite Cayley Hamilton moment list for exact local closure."""

        moments: list[np.ndarray] = []
        power = np.eye(len(self.hidden), dtype=np.result_type(self.matrix.dtype, float))
        for _ in range(len(self.hidden)):
            moments.append(self.a_pq @ power @ self.a_qp)
            power = power @ self.hidden_block
        return tuple(moments)

    @documented(
        ProvenanceRecord(
            identifier="PSC-VI-T29",
            title="Finite exact local closure criterion",
            status=ClaimStatus.PROVED,
            source="Paper VI",
            assumptions=("finite hidden state space",),
            verification_suite="dynamics.feshbach",
        )
    )
    def local_closure_certificate(self, *, tolerance: float = 1e-10) -> Certificate:
        norms = tuple(float(np.linalg.norm(moment, ord=2)) for moment in self.closure_moments())
        residual = max(norms, default=0.0)
        valid = residual <= tolerance
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={f"moment_{index}": value for index, value in enumerate(norms)},
            tolerance=tolerance,
            assumptions_checked=("finite hidden dimension", "Cayley Hamilton moment test"),
            theorem_id="PSC-VI-T29",
            message="hidden feedback vanishes and the retained equation closes locally" if valid else "hidden feedback produces memory",
        )


__all__ = ["FeshbachReduction"]
