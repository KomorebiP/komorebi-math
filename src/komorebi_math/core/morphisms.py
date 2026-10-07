"""Intertwining and packet compatible morphisms."""

from __future__ import annotations

from .._residuals import ResidualAudit, finite_tolerance

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..backends import LinearAlgebraBackend, get_backend
from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .packets import PacketSystem
from .seeds import SeedSystem


@documented(
    ProvenanceRecord(
        identifier="PSC-I-T50",
        title="Naturality of seed hulls and packet blocks",
        status=ClaimStatus.PROVED,
        source="Paper I",
        assumptions=("declared intertwining equations",),
        verification_suite="core.morphisms",
    )
)
@dataclass(frozen=True, slots=True)
class OperatorMorphism:
    source_operators: Mapping[str, Any]
    target_operators: Mapping[str, Any]
    linear_map: Any
    operator_map: Mapping[str, str]
    backend: Any
    tolerance: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "tolerance", finite_tolerance(self.tolerance))

    @classmethod
    def create(
        cls,
        source_operators: Mapping[str, Any],
        target_operators: Mapping[str, Any],
        linear_map: Any,
        operator_map: Mapping[str, str] | None = None,
        *,
        backend: str | LinearAlgebraBackend | None = None,
        tolerance: float = 1e-10,
    ) -> "OperatorMorphism":
        active = get_backend(backend)
        source = {name: active.as_matrix(value) for name, value in source_operators.items()}
        target = {name: active.as_matrix(value) for name, value in target_operators.items()}
        mapping = dict(operator_map or {name: name for name in source if name in target})
        phi = active.as_matrix(linear_map)
        for source_name, target_name in mapping.items():
            if source_name not in source or target_name not in target:
                raise ValidationError("operator map references an unknown operator")
            source_dimension = active.shape(source[source_name])[0]
            target_dimension = active.shape(target[target_name])[0]
            if active.shape(phi) != (target_dimension, source_dimension):
                raise DimensionError("linear map has incompatible source or target dimension")
        return cls(source, target, phi, mapping, active, float(tolerance))

    def certificate(self) -> Certificate:
        audit = ResidualAudit(self.backend, self.tolerance)
        residuals: dict[str, float] = {}
        for source_name, target_name in self.operator_map.items():
            residual = self.backend.subtract(
                self.backend.matmul(self.linear_map, self.source_operators[source_name]),
                self.backend.matmul(self.target_operators[target_name], self.linear_map))
            residuals[f"{source_name}->{target_name}"] = audit.check(f"{source_name}->{target_name}", residual)
        valid = audit.valid
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT if self.backend.exact else Exactness.NUMERICAL,
            residuals=residuals,
            tolerance=None if self.backend.exact else self.tolerance,
            assumptions_checked=("operator dimensions", "declared operator map"),
            assumptions_unchecked=audit.unresolved,
            diagnostics={"residual_decisions": audit.decisions},
            theorem_id="PSC-I-T50",
            message="valid operator intertwiner" if valid else "intertwining checks failed or are unresolved",
        )


@dataclass(frozen=True, slots=True)
class PacketMorphism:
    source: PacketSystem
    target: PacketSystem
    linear_map: Any
    packet_map: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        if self.source.backend.name != self.target.backend.name:
            raise ValidationError("packet morphism requires one backend")
        phi = self.source.backend.as_matrix(self.linear_map)
        if self.source.backend.shape(phi) != (self.target.dimension, self.source.dimension):
            raise DimensionError("packet morphism has incompatible shape")
        object.__setattr__(self, "linear_map", phi)
        if set(self.packet_map) != set(self.source.names):
            raise ValidationError("packet map must specify every source packet")
        target_names = set(self.target.names)
        if any(not set(values).issubset(target_names) for values in self.packet_map.values()):
            raise ValidationError("packet map references an unknown target packet")

    def certificate(self) -> Certificate:
        backend = self.source.backend
        source_certificate = self.source.certificate()
        target_certificate = self.target.certificate()
        tolerance = max(finite_tolerance(self.source.tolerance), finite_tolerance(self.target.tolerance))
        audit = ResidualAudit(backend, tolerance)
        residuals: dict[str, float] = {}
        for source_name, targets in self.packet_map.items():
            source_projector = self.source.projectors[self.source.index(source_name)]
            total = backend.zeros(self.target.dimension, self.target.dimension)
            for target_name in targets:
                total = backend.add(total, self.target.projectors[self.target.index(target_name)])
            source_image = backend.matmul(self.linear_map, source_projector)
            residual = backend.subtract(source_image, backend.matmul(total, source_image))
            residuals[source_name] = audit.check(source_name, residual)
        valid = source_certificate.valid and target_certificate.valid and audit.valid
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT if backend.exact else Exactness.NUMERICAL,
            residuals=residuals,
            tolerance=None if backend.exact else tolerance,
            assumptions_checked=("source packets", "target packets", "declared aggregation map"),
            assumptions_unchecked=source_certificate.assumptions_unchecked + target_certificate.assumptions_unchecked + audit.unresolved,
            diagnostics={"source_certificate": source_certificate.to_dict(),
                         "target_certificate": target_certificate.to_dict(), "residual_decisions": audit.decisions},
            theorem_id="PSC-I-T52",
            message="packet compatible morphism" if valid else "parent packet or compatibility checks failed or are unresolved",
        )


@dataclass(frozen=True, slots=True)
class SeedMorphism:
    source: SeedSystem
    target: SeedSystem
    linear_map: Any
    operator_map: Mapping[str, str]
    seed_images: Sequence[Any] | None = None

    def certificate(self) -> Certificate:
        if self.source.backend.name != self.target.backend.name:
            raise ValidationError("seed morphism requires one backend")
        backend = self.source.backend
        phi = backend.as_matrix(self.linear_map)
        if backend.shape(phi) != (self.target.dimension, self.source.dimension):
            raise DimensionError("seed morphism has incompatible shape")
        tolerance = max(finite_tolerance(self.source.tolerance), finite_tolerance(self.target.tolerance))
        operator_morphism = OperatorMorphism.create(
            self.source.operators, self.target.operators, phi, self.operator_map,
            backend=backend, tolerance=tolerance)
        operator_certificate = operator_morphism.certificate()
        audit = ResidualAudit(backend, tolerance)
        residuals = dict(operator_certificate.residuals)
        if self.seed_images is not None:
            if len(self.seed_images) != len(self.source.seeds):
                raise ValidationError("one expected image is required for every source seed")
            for index, (seed, expected) in enumerate(zip(self.source.seeds, self.seed_images, strict=True)):
                residual = backend.subtract(backend.matmul(phi, seed), backend.as_vector(expected))
                residuals[f"seed_{index}"] = audit.check(f"seed_{index}", residual)
        valid = operator_certificate.valid and audit.valid
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT if backend.exact else Exactness.NUMERICAL,
            residuals=residuals,
            tolerance=None if backend.exact else tolerance,
            assumptions_checked=("operator intertwining", "declared seed images"),
            assumptions_unchecked=operator_certificate.assumptions_unchecked + audit.unresolved,
            diagnostics={"operator_certificate": operator_certificate.to_dict(), "residual_decisions": audit.decisions},
            theorem_id="PSC-I-T50",
            message="valid seed morphism" if valid else "seed morphism validation failed or is unresolved",
        )
