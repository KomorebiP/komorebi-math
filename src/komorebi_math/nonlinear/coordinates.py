"""Exact coordinate conjugation for nonlinear maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .._residuals import ResidualAudit
from .._prerequisites import packet_certificate_dependencies
from ..certificates import Certificate, ComputationResult, Exactness
from ..core.packets import PacketCoordinates, PacketSystem
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented


@documented(
    ProvenanceRecord(
        identifier="PSC-III-D01",
        title="Nonlinear packet coordinate map",
        status=ClaimStatus.DEFINITION,
        source="Paper III",
        assumptions=("finite source and target packet systems",),
        verification_suite="nonlinear.coordinates",
    )
)
@dataclass(slots=True)
class NonlinearPacketMap:
    """A nonlinear map evaluated through exact packet analysis and synthesis."""

    function: Callable[[Any], Any]
    source: PacketSystem
    target: PacketSystem
    name: str = "nonlinear packet map"

    def __post_init__(self) -> None:
        if self.source.backend.name != self.target.backend.name:
            raise ValidationError("source and target packets must use one backend")

    def __call__(self, coordinates: PacketCoordinates) -> PacketCoordinates:
        if not coordinates.system.equivalent(self.source):
            raise ValidationError("coordinates belong to a different source packet system")
        input_value = coordinates.reconstruct()
        output_value = self.target.backend.as_vector(self.function(input_value))
        if self.target.backend.shape(output_value)[0] != self.target.dimension:
            raise DimensionError("nonlinear map output does not match the target packet dimension")
        return self.target.analyze(output_value)

    def apply(self, vector: Any) -> Any:
        value = self.source.backend.as_vector(vector)
        if self.source.backend.shape(value)[0] != self.source.dimension:
            raise DimensionError("input vector does not match the source packet dimension")
        output = self.target.backend.as_vector(self.function(value))
        if self.target.backend.shape(output)[0] != self.target.dimension:
            raise DimensionError("nonlinear map output does not match the target packet dimension")
        return output

    @documented(
        ProvenanceRecord(
            identifier="PSC-III-T02",
            title="Exact nonlinear packet coordinate theorem",
            status=ClaimStatus.PROVED,
            source="Paper III",
            assumptions=("exact packet analysis and synthesis",),
            verification_suite="nonlinear.coordinates",
        )
    )
    def certificate(self, vector: Any) -> Certificate:
        value = self.source.backend.as_vector(vector)
        direct = self.apply(value)
        packet_value = self(self.source.analyze(value)).reconstruct()
        tolerance = max(self.source.tolerance, self.target.tolerance)
        audit = ResidualAudit(self.target.backend, tolerance)
        residual = audit.check("coordinate_conjugacy", self.target.backend.subtract(direct, packet_value))
        parents_valid, parent_checked, parent_unchecked, parent_diagnostics = packet_certificate_dependencies(
            source=self.source, target=self.target
        )
        valid = parents_valid and audit.valid
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT if self.source.backend.exact else Exactness.NUMERICAL,
            residuals={"coordinate_conjugacy": residual},
            tolerance=None if self.source.backend.exact else tolerance,
            assumptions_checked=parent_checked + ("source reconstruction", "target analysis", "function evaluation"),
            assumptions_unchecked=parent_unchecked + audit.unresolved + ("callback is consistent across the compared evaluations",),
            diagnostics={**parent_diagnostics, "residual_decisions": audit.decisions},
            theorem_id="PSC-III-T02",
            message="nonlinear packet coordinates reproduce the ambient map" if valid else "nonlinear coordinate conjugacy failed",
        )

    def iterate(self, coordinates: PacketCoordinates, count: int) -> PacketCoordinates:
        if count < 0:
            raise ValueError("iteration count must be nonnegative")
        if not self.source.equivalent(self.target):
            raise ValidationError("iteration requires equivalent source and target packet systems")
        current = coordinates
        for _ in range(count):
            current = self(current)
        return current

    def fixed_point_residual(self, coordinates: PacketCoordinates) -> ComputationResult[Any]:
        if not self.source.equivalent(self.target):
            raise ValidationError("a fixed point residual requires equivalent source and target packet systems")
        output = self(coordinates)
        residual = self.source.backend.subtract(output.reconstruct(), coordinates.reconstruct())
        parents_valid, parent_checked, parent_unchecked, parent_diagnostics = packet_certificate_dependencies(
            source=self.source, target=self.target
        )
        return ComputationResult(
            residual,
            Certificate(
                valid=parents_valid,
                exactness=Exactness.EXACT if self.source.backend.exact else Exactness.NUMERICAL,
                residuals={"fixed_point_residual_norm": self.source.backend.norm(residual)},
                assumptions_checked=parent_checked + ("source and target packet equivalence",),
                assumptions_unchecked=parent_unchecked,
                diagnostics=parent_diagnostics,
                theorem_id="PSC-III-C03",
                message="fixed point residual computed in ambient and packet coordinates",
            ),
        )


__all__ = ["NonlinearPacketMap"]
