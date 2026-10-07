"""Finite quantum channels, operator packets, Choi matrices, and covariance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..core import PacketSystem
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._validation import as_matrix, as_square_matrix, density_matrix, require_unitary


def operator_packets(operator: Any, packets: PacketSystem) -> tuple[tuple[np.ndarray, ...], ...]:
    """Resolve an operator into left and right packet addresses."""

    value = as_square_matrix(operator, name="operator packet input")
    if value.shape != (packets.dimension, packets.dimension):
        raise DimensionError("operator and packet dimensions do not agree")
    return tuple(
        tuple(left @ value @ right for right in packets.projectors)
        for left in packets.projectors
    )


def reconstruct_operator_packets(blocks: Iterable[Iterable[Any]]) -> np.ndarray:
    """Reconstruct an operator from a rectangular family of packet blocks."""

    rows = tuple(tuple(as_matrix(value, name="operator packet block") for value in row) for row in blocks)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValidationError("operator packet blocks must form a nonempty rectangle")
    shape = rows[0][0].shape
    if any(value.shape != shape for row in rows for value in row):
        raise DimensionError("all operator packet blocks must have one common ambient shape")
    return sum((value for row in rows for value in row), np.zeros(shape, dtype=complex))


def dephase(operator: Any, packets: PacketSystem) -> np.ndarray:
    """Apply the packet diagonal conditional expectation."""

    value = as_square_matrix(operator, name="dephasing input")
    if value.shape != (packets.dimension, packets.dimension):
        raise DimensionError("operator and packet dimensions do not agree")
    return sum(projector @ value @ projector for projector in packets.projectors)


@dataclass(frozen=True, slots=True)
class ChannelPacketBlock:
    """One input and output packet superblock of a quantum channel."""

    output_left: int
    output_right: int
    input_left: int
    input_right: int
    superoperator: np.ndarray


@documented(
    ProvenanceRecord(
        identifier="PSC-VII-T31",
        title="Finite Kraus representation",
        status=ClaimStatus.CLASSICAL,
        source="Paper VII",
        assumptions=("finite Kraus family",),
        verification_suite="quantum.channels",
    )
)
class QuantumChannel:
    """A finite completely positive map represented by Kraus operators."""

    def __init__(self, kraus: Iterable[Any], *, tolerance: float = 1e-10) -> None:
        operators = tuple(as_matrix(value, name="Kraus operator") for value in kraus)
        if not operators:
            raise ValidationError("a quantum channel requires at least one Kraus operator")
        output_dimension, input_dimension = operators[0].shape
        if any(operator.shape != (output_dimension, input_dimension) for operator in operators):
            raise DimensionError("all Kraus operators must have one common rectangular shape")
        self.kraus = operators
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.tolerance = float(tolerance)

    @classmethod
    def identity(cls, dimension: int) -> "QuantumChannel":
        """Return the identity channel."""

        n = int(dimension)
        if isinstance(dimension, bool) or n != dimension or n < 1:
            raise ValidationError("dimension must be a positive integer")
        return cls((np.eye(n, dtype=complex),))

    def apply(self, operator: Any) -> np.ndarray:
        """Apply the channel to one input operator."""

        value = as_square_matrix(operator, name="channel input")
        if value.shape != (self.input_dimension, self.input_dimension):
            raise DimensionError("channel input has the wrong dimension")
        return sum(kraus @ value @ kraus.conj().T for kraus in self.kraus)

    def apply_density(self, state: Any) -> np.ndarray:
        """Apply the channel to a validated density matrix."""

        rho = density_matrix(state, tolerance=self.tolerance)
        output = self.apply(rho)
        if self.is_trace_preserving():
            return density_matrix(output, tolerance=self.tolerance)
        return output

    def adjoint(self, operator: Any) -> np.ndarray:
        """Apply the Hilbert Schmidt adjoint channel."""

        value = as_square_matrix(operator, name="adjoint channel input")
        if value.shape != (self.output_dimension, self.output_dimension):
            raise DimensionError("adjoint channel input has the wrong dimension")
        return sum(kraus.conj().T @ value @ kraus for kraus in self.kraus)

    def is_trace_preserving(self) -> bool:
        """Return whether the Kraus family is trace preserving."""

        total = sum(kraus.conj().T @ kraus for kraus in self.kraus)
        return bool(np.allclose(total, np.eye(self.input_dimension), atol=self.tolerance, rtol=0.0))

    def is_unital(self) -> bool:
        """Return whether the channel maps the identity to the identity."""

        total = sum(kraus @ kraus.conj().T for kraus in self.kraus)
        return bool(np.allclose(total, np.eye(self.output_dimension), atol=self.tolerance, rtol=0.0))

    def certificate(self) -> Certificate:
        """Return the complete positivity and trace preservation certificate."""

        tp = sum(kraus.conj().T @ kraus for kraus in self.kraus)
        residual = float(np.linalg.norm(tp - np.eye(self.input_dimension), ord=2))
        choi_eigenvalue = float(np.min(np.linalg.eigvalsh((self.choi_matrix() + self.choi_matrix().conj().T) / 2)))
        return Certificate(
            valid=residual <= self.tolerance and choi_eigenvalue >= -self.tolerance,
            exactness=Exactness.NUMERICAL,
            residuals={"trace_preservation": residual, "negative_choi_part": max(0.0, -choi_eigenvalue)},
            tolerance=self.tolerance,
            assumptions_checked=("finite Kraus representation",),
            theorem_id="PSC-VII-T31",
            message="Kraus family defines a completely positive trace preserving map",
        )

    def superoperator(self) -> np.ndarray:
        """Return the column vectorized superoperator matrix."""

        return sum(np.kron(kraus.conj(), kraus) for kraus in self.kraus)

    def choi_matrix(self) -> np.ndarray:
        """Return the Choi matrix in output tensor input ordering."""

        result = np.zeros(
            (self.output_dimension * self.input_dimension, self.output_dimension * self.input_dimension),
            dtype=complex,
        )
        for column in range(self.input_dimension):
            for row in range(self.input_dimension):
                matrix_unit = np.zeros((self.input_dimension, self.input_dimension), dtype=complex)
                matrix_unit[row, column] = 1
                result += np.kron(self.apply(matrix_unit), matrix_unit)
        return result

    def compose(self, previous: "QuantumChannel") -> "QuantumChannel":
        """Return ``self`` after ``previous``."""

        if previous.output_dimension != self.input_dimension:
            raise DimensionError("channel dimensions do not compose")
        return QuantumChannel(
            tuple(left @ right for left in self.kraus for right in previous.kraus),
            tolerance=max(self.tolerance, previous.tolerance),
        )

    def tensor(self, other: "QuantumChannel") -> "QuantumChannel":
        """Return the tensor product channel."""

        return QuantumChannel(
            tuple(np.kron(left, right) for left in self.kraus for right in other.kraus),
            tolerance=max(self.tolerance, other.tolerance),
        )

    def packet_blocks(
        self,
        input_packets: PacketSystem,
        output_packets: PacketSystem,
    ) -> tuple[ChannelPacketBlock, ...]:
        """Return every operator packet superblock of the channel."""

        if input_packets.dimension != self.input_dimension or output_packets.dimension != self.output_dimension:
            raise DimensionError("channel and packet dimensions do not agree")
        input_super = tuple(
            np.kron(right.T, left)
            for left in input_packets.projectors
            for right in input_packets.projectors
        )
        output_super = tuple(
            np.kron(right.T, left)
            for left in output_packets.projectors
            for right in output_packets.projectors
        )
        matrix = self.superoperator()
        results: list[ChannelPacketBlock] = []
        input_count = len(input_packets)
        output_count = len(output_packets)
        for output_index, q in enumerate(output_super):
            for input_index, p in enumerate(input_super):
                results.append(
                    ChannelPacketBlock(
                        output_index // output_count,
                        output_index % output_count,
                        input_index // input_count,
                        input_index % input_count,
                        q @ matrix @ p,
                    )
                )
        return tuple(results)

    def covariance_certificate(
        self,
        input_symmetry: Any,
        output_symmetry: Any | None = None,
    ) -> Certificate:
        """Check covariance under declared input and output unitaries."""

        u_in = require_unitary(input_symmetry, tolerance=self.tolerance, name="input symmetry")
        u_out = u_in if output_symmetry is None else require_unitary(output_symmetry, tolerance=self.tolerance, name="output symmetry")
        if u_in.shape[0] != self.input_dimension or u_out.shape[0] != self.output_dimension:
            raise DimensionError("symmetry and channel dimensions do not agree")
        residual = 0.0
        for row in range(self.input_dimension):
            for column in range(self.input_dimension):
                unit = np.zeros((self.input_dimension, self.input_dimension), dtype=complex)
                unit[row, column] = 1
                lhs = self.apply(u_in @ unit @ u_in.conj().T)
                rhs = u_out @ self.apply(unit) @ u_out.conj().T
                residual = max(residual, float(np.linalg.norm(lhs - rhs, ord=2)))
        return Certificate(
            valid=residual <= self.tolerance,
            exactness=Exactness.NUMERICAL,
            residuals={"covariance": residual},
            tolerance=self.tolerance,
            assumptions_checked=("unitary input and output symmetries",),
            theorem_id="PSC-VII-T35",
            message="channel is covariant under the declared symmetries",
        )


def amplitude_damping(gamma: float) -> QuantumChannel:
    """Return the qubit amplitude damping channel."""

    value = float(gamma)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValidationError("gamma must lie between zero and one")
    k0 = np.diag([1.0, np.sqrt(1 - value)]).astype(complex)
    k1 = np.array([[0.0, np.sqrt(value)], [0.0, 0.0]], dtype=complex)
    return QuantumChannel((k0, k1))


__all__ = [
    "ChannelPacketBlock",
    "QuantumChannel",
    "amplitude_damping",
    "dephase",
    "operator_packets",
    "reconstruct_operator_packets",
]
