"""Finite packet bridge and holonomy diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ...certificates import Certificate, Exactness
from ...nonlinear import TransportEdge, TransportGraph
from .symplectic import AlternatingForm


@dataclass(frozen=True, slots=True)
class SymplecticBridge:
    """A declared finite symplectic comparison between two envelopes."""

    source: AlternatingForm
    target: AlternatingForm
    matrix: Any
    certificate: Certificate

    @classmethod
    def construct(
        cls,
        source: AlternatingForm,
        target: AlternatingForm,
        matrix: Any,
    ) -> "SymplecticBridge":
        """Validate one equivariant-free finite symplectic isometry."""

        import numpy as np

        if source.prime != target.prime or source.dimension != target.dimension:
            from ...exceptions import DimensionError

            raise DimensionError("symplectic bridge forms must use one prime and dimension")
        p = source.prime
        value = np.asarray(matrix, dtype=object)
        if value.shape != (source.dimension, source.dimension):
            from ...exceptions import DimensionError

            raise DimensionError("symplectic bridge matrix has the wrong dimension")
        integer = np.vectorize(int)(value) % p
        residual = (integer.T @ target.matrix @ integer - source.matrix) % p
        from ...deformation.complexes import row_rank_mod

        rank = row_rank_mod(integer, p)
        invertible = rank == source.dimension
        valid = invertible and not np.any(residual)
        certificate = Certificate(
            valid=valid,
            exactness=Exactness.EXACT,
            residuals={"symplectic_mismatch": float(np.max(np.abs(residual), initial=0))},
            assumptions_checked=("finite prime field symplectic forms",),
            diagnostics={"rank_mod_prime": rank},
            theorem_id="PSC-XI-RESTRICTED-BRIDGE",
            message="finite symplectic bridge verified" if valid else "finite symplectic bridge failed",
        )
        return cls(source, target, integer, certificate)


class LocalPacketBridge:
    """A finite transport graph used to test packetwise gluing."""

    def __init__(self, vertices: Iterable[str], edges: Iterable[TransportEdge], *, tolerance: float = 1e-10) -> None:
        self.transport = TransportGraph(vertices, edges, tolerance=tolerance)

    def holonomy_certificate(self) -> Certificate:
        """Return the fundamental-cycle gluing certificate."""

        return self.transport.gluing_certificate()


__all__ = ["LocalPacketBridge", "SymplecticBridge"]
