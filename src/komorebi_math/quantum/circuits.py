"""Exact finite quantum circuit packet path expansions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..core import PacketSystem
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._validation import as_square_matrix, as_vector, require_unitary


@dataclass(frozen=True, slots=True)
class CircuitPathAmplitude:
    """One packet address path and its operator block."""

    path: tuple[int, ...]
    block: np.ndarray


@dataclass(frozen=True, slots=True)
class InterferenceReport:
    """Coherent path probability decomposition for one source vector."""

    target_probability: float
    incoherent_sum: float
    interference: float
    reconstructed_probability: float
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="PSC-VII-T14",
        title="Circuit packet path expansion",
        status=ClaimStatus.PROVED,
        source="Paper VII",
        assumptions=("finite unitary circuit", "complete packet system at every circuit depth"),
        verification_suite="quantum.circuits",
    )
)
class CircuitPacketExpansion:
    """Insert packet resolutions between every gate of a finite circuit."""

    def __init__(
        self,
        gates: Iterable[Any],
        packet_systems: Iterable[PacketSystem],
        *,
        tolerance: float = 1e-10,
        max_paths: int = 1_000_000,
    ) -> None:
        values = tuple(require_unitary(gate, tolerance=tolerance, name="circuit gate") for gate in gates)
        packets = tuple(packet_systems)
        if not values:
            raise ValidationError("a circuit requires at least one gate")
        if len(packets) != len(values) + 1:
            raise ValidationError("packet_systems must contain one family before and after every gate")
        dimension = values[0].shape[0]
        if any(gate.shape != (dimension, dimension) for gate in values):
            raise DimensionError("all circuit gates must have one common dimension")
        if any(packet.dimension != dimension for packet in packets):
            raise DimensionError("every packet system must act on the circuit Hilbert space")
        internal_paths = 1
        for packet in packets[1:-1]:
            internal_paths *= len(packet)
        if internal_paths > max_paths:
            raise ValidationError("circuit packet expansion exceeds the declared path safety limit")
        self.gates = values
        self.packet_systems = packets
        self.tolerance = float(tolerance)
        self.max_paths = int(max_paths)

    @property
    def dimension(self) -> int:
        """Return the circuit Hilbert space dimension."""

        return self.gates[0].shape[0]

    def unitary(self) -> np.ndarray:
        """Return the full circuit unitary in chronological gate order."""

        result = np.eye(self.dimension, dtype=complex)
        for gate in self.gates:
            result = gate @ result
        return result

    def path_blocks(self, source: int, target: int) -> tuple[CircuitPathAmplitude, ...]:
        """Return every packet path block from one source to one target."""

        source_index = int(source)
        target_index = int(target)
        if not 0 <= source_index < len(self.packet_systems[0]):
            raise ValidationError("source packet lies outside the first packet family")
        if not 0 <= target_index < len(self.packet_systems[-1]):
            raise ValidationError("target packet lies outside the final packet family")
        internal_ranges = [range(len(packet)) for packet in self.packet_systems[1:-1]]
        results: list[CircuitPathAmplitude] = []
        for internal in product(*internal_ranges):
            path = (source_index, *internal, target_index)
            block = self.packet_systems[0].projectors[path[0]]
            for level, gate in enumerate(self.gates):
                block = self.packet_systems[level + 1].projectors[path[level + 1]] @ gate @ block
            results.append(CircuitPathAmplitude(path, block))
        return tuple(results)

    def block(self, source: int, target: int) -> np.ndarray:
        """Sum every packet path from one source address to one target address."""

        paths = self.path_blocks(source, target)
        return sum((path.block for path in paths), np.zeros((self.dimension, self.dimension), dtype=complex))

    def reconstruction_certificate(self) -> ComputationResult[np.ndarray]:
        """Reconstruct the complete circuit from all packet paths."""

        reconstructed = np.zeros((self.dimension, self.dimension), dtype=complex)
        for source in range(len(self.packet_systems[0])):
            for target in range(len(self.packet_systems[-1])):
                reconstructed += self.block(source, target)
        direct = self.unitary()
        residual = float(np.linalg.norm(reconstructed - direct, ord=2))
        return ComputationResult(
            reconstructed,
            Certificate(
                valid=residual <= self.tolerance,
                exactness=Exactness.NUMERICAL,
                residuals={"circuit_reconstruction": residual},
                tolerance=self.tolerance,
                assumptions_checked=("complete packet resolutions", "unitary circuit gates"),
                theorem_id="PSC-VII-T14",
                message="packet paths reconstruct the finite circuit",
            ),
        )

    def interference(self, state: Any, source: int, target: int) -> InterferenceReport:
        """Resolve one target probability into incoherent and interference terms."""

        vector = as_vector(state, dimension=self.dimension, name="circuit input state")
        source_projector = self.packet_systems[0].projectors[int(source)]
        source_vector = source_projector @ vector
        amplitudes = tuple(path.block @ source_vector for path in self.path_blocks(source, target))
        target_vector = self.packet_systems[-1].projectors[int(target)] @ self.unitary() @ source_vector
        probability = float(np.vdot(target_vector, target_vector).real)
        incoherent = float(sum(np.vdot(value, value).real for value in amplitudes))
        interference = 0.0
        for left in range(len(amplitudes)):
            for right in range(left + 1, len(amplitudes)):
                interference += 2 * float(np.vdot(amplitudes[left], amplitudes[right]).real)
        reconstructed = incoherent + interference
        residual = abs(probability - reconstructed)
        return InterferenceReport(
            probability,
            incoherent,
            interference,
            reconstructed,
            Certificate(
                valid=residual <= self.tolerance * max(1.0, probability),
                exactness=Exactness.NUMERICAL,
                residuals={"interference_identity": residual},
                tolerance=self.tolerance * max(1.0, probability),
                assumptions_checked=("source packet projection", "coherent path amplitudes"),
                theorem_id="PSC-VII-C17",
                message="path interference reconstructs the target probability",
            ),
        )

    def block_unitarity_certificate(self) -> Certificate:
        """Check the packet block row and column unitarity identities."""

        unitary = self.unitary()
        source_packets = self.packet_systems[0]
        target_packets = self.packet_systems[-1]
        residual = 0.0
        for source in range(len(source_packets)):
            for other in range(len(source_packets)):
                lhs = sum(
                    (target_packets.projectors[target] @ unitary @ source_packets.projectors[source]).conj().T
                    @ (target_packets.projectors[target] @ unitary @ source_packets.projectors[other])
                    for target in range(len(target_packets))
                )
                expected = source_packets.projectors[source] if source == other else np.zeros_like(lhs)
                residual = max(residual, float(np.linalg.norm(lhs - expected, ord=2)))
        return Certificate(
            valid=residual <= self.tolerance,
            exactness=Exactness.NUMERICAL,
            residuals={"column_unitarity": residual},
            tolerance=self.tolerance,
            assumptions_checked=("unitary circuit", "complete source and target packets"),
            theorem_id="PSC-VII-T15",
            message="circuit packet blocks satisfy column unitarity",
        )


__all__ = ["CircuitPacketExpansion", "CircuitPathAmplitude", "InterferenceReport"]
