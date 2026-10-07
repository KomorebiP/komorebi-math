"""Finite Hecke packet decompositions and support transport models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from ...certificates import Certificate, Exactness
from ...exceptions import DimensionError, ValidationError


@dataclass(frozen=True, slots=True)
class HeckePacket:
    """One central idempotent packet in a finite Hecke carrier."""

    name: str
    projector: Any
    dimension: int


class FiniteHeckePackets:
    """A validated finite family of central idempotents on one carrier."""

    def __init__(
        self,
        projectors: Mapping[str, Any],
        *,
        commuting_operators: Iterable[Any] = (),
        tolerance: float = 1e-10,
    ) -> None:
        if not projectors:
            raise ValidationError("finite Hecke packets require projectors")
        self.names = tuple(projectors)
        self.projectors = {name: np.asarray(matrix, dtype=complex) for name, matrix in projectors.items()}
        self.tolerance = float(tolerance)
        shape = next(iter(self.projectors.values())).shape
        if len(shape) != 2 or shape[0] != shape[1] or any(matrix.shape != shape for matrix in self.projectors.values()):
            raise DimensionError("Hecke projectors must be square matrices of one carrier dimension")
        self.dimension = shape[0]
        total = np.zeros(shape, dtype=complex)
        for name, matrix in self.projectors.items():
            if np.linalg.norm(matrix @ matrix - matrix) > self.tolerance:
                raise ValidationError(f"Hecke packet {name!r} is not idempotent")
            for earlier_name, earlier in self.projectors.items():
                if earlier_name == name:
                    break
                if np.linalg.norm(matrix @ earlier) > self.tolerance or np.linalg.norm(earlier @ matrix) > self.tolerance:
                    raise ValidationError("Hecke packets are not orthogonal")
            total += matrix
        if np.linalg.norm(total - np.eye(self.dimension)) > self.tolerance:
            raise ValidationError("Hecke packets are not complete")
        self.commuting_operators = tuple(np.asarray(operator, dtype=complex) for operator in commuting_operators)
        for operator in self.commuting_operators:
            if operator.shape != shape:
                raise DimensionError("Hecke operator has the wrong carrier dimension")
            for matrix in self.projectors.values():
                if np.linalg.norm(operator @ matrix - matrix @ operator) > self.tolerance:
                    raise ValidationError("Hecke projector is not central for the declared operators")

    def packets(self) -> tuple[HeckePacket, ...]:
        """Return packet ranks in the finite carrier."""

        return tuple(
            HeckePacket(name, matrix.copy(), int(np.linalg.matrix_rank(matrix, tol=self.tolerance)))
            for name, matrix in self.projectors.items()
        )

    def occupied(self, name: str, carrier_basis: Any | None = None) -> bool:
        """Return whether one packet is occupied in a declared subcarrier."""

        if name not in self.projectors:
            raise ValidationError("unknown Hecke packet")
        if carrier_basis is None:
            return np.linalg.matrix_rank(self.projectors[name], tol=self.tolerance) > 0
        basis = np.asarray(carrier_basis, dtype=complex)
        if basis.ndim != 2 or basis.shape[0] != self.dimension:
            raise DimensionError("carrier basis has the wrong ambient dimension")
        image = self.projectors[name] @ basis
        return np.linalg.matrix_rank(image, tol=self.tolerance) > 0

    def certificate(self) -> Certificate:
        """Return a finite packet decomposition certificate."""

        ranks = {name: int(np.linalg.matrix_rank(matrix, tol=self.tolerance)) for name, matrix in self.projectors.items()}
        return Certificate(
            valid=sum(ranks.values()) == self.dimension,
            exactness=Exactness.NUMERICAL,
            tolerance=self.tolerance,
            assumptions_checked=("finite central idempotents", "complete orthogonal decomposition"),
            diagnostics={"ranks": ranks},
            theorem_id="PSC-XII-HECKE-PACKETS",
            message="finite Hecke packet decomposition verified",
        )


@dataclass(frozen=True, slots=True)
class SupportComparison:
    """A support-preserving or support-losing comparison diagnostic."""

    source_components: frozenset[str]
    target_components: frozenset[str]
    image_components: frozenset[str]
    certificate: Certificate

    @classmethod
    def construct(
        cls,
        source_components: Iterable[str],
        target_components: Iterable[str],
        image_components: Iterable[str],
    ) -> "SupportComparison":
        """Construct one finite support comparison."""

        source = frozenset(source_components)
        target = frozenset(target_components)
        image = frozenset(image_components)
        if not image <= target:
            raise ValidationError("comparison image contains an unknown target component")
        valid = source <= image
        return cls(
            source,
            target,
            image,
            Certificate(
                valid=valid,
                exactness=Exactness.EXACT,
                assumptions_checked=("finite declared component sets",),
                diagnostics={"missing_components": tuple(sorted(source - image))},
                theorem_id="PSC-XII-SUPPORT-COMPARISON",
                message="comparison covers every required source component" if valid else "comparison loses required components",
            ),
        )


__all__ = ["FiniteHeckePackets", "HeckePacket", "SupportComparison"]
