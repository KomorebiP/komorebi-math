"""The common executable language used across Komorebi Math domains."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from .certificates import Certificate, Exactness
from .exceptions import DimensionError, ValidationError
from .core.grammar import OperatorGrammar
from .core.operators import PacketOperator
from .core.packets import PacketCoordinates, PacketSystem
from .core.realization import OperatorRealization
from .core.seeds import SeedSystem
from .nonlinear.obstruction import LinearObstructionProblem
from .refinement.towers import RefinementTower


class UnifiedStage(StrEnum):
    SEED = "seed"
    ADDRESS = "address"
    OPERATE = "operate"
    PROJECT = "project"
    TRANSFER = "transfer"
    REFINE = "refine"
    CONSTRAIN = "constrain"
    COMPLETE = "complete"
    RECONSTRUCT = "reconstruct"


@dataclass(frozen=True, slots=True)
class StageSpec:
    stage: UnifiedStage
    meaning: str
    implemented_in_alpha: bool
    module: str


STAGES: tuple[StageSpec, ...] = (
    StageSpec(UnifiedStage.SEED, "Choose finite generating data", True, "komorebi_math.core.seeds"),
    StageSpec(UnifiedStage.ADDRESS, "Name legal operator paths and packet coordinates", True, "komorebi_math.core.grammar"),
    StageSpec(UnifiedStage.OPERATE, "Apply realized operators", True, "komorebi_math.core.realization"),
    StageSpec(UnifiedStage.PROJECT, "Resolve objects into exact packets", True, "komorebi_math.core.packets"),
    StageSpec(UnifiedStage.TRANSFER, "Track source to target packet blocks", True, "komorebi_math.core.operators"),
    StageSpec(UnifiedStage.REFINE, "Pass between scales or residue levels", True, "komorebi_math.refinement"),
    StageSpec(UnifiedStage.CONSTRAIN, "Form defects and obstruction classes", True, "komorebi_math.nonlinear"),
    StageSpec(UnifiedStage.COMPLETE, "Construct compatible inverse limits", False, "komorebi_math.deformation"),
    StageSpec(UnifiedStage.RECONSTRUCT, "Return to the ambient object", True, "komorebi_math.core.packets"),
)


@dataclass(slots=True)
class UnifiedSystem:
    """One finite system expressed in the common seed and packet language."""

    grammar: OperatorGrammar | None = None
    realization: OperatorRealization | None = None
    seed_system: SeedSystem | None = None
    packet_system: PacketSystem | None = None
    operators: Mapping[str, Any] = field(default_factory=dict)
    refinement_tower: RefinementTower | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> Certificate:
        residuals: dict[str, float] = {}
        checked: list[str] = []
        exact = True
        valid = True

        if self.realization is not None:
            if self.grammar is not None and self.realization.grammar is not self.grammar:
                raise ValidationError("the realization does not belong to the declared grammar")
            relation_certificate = self.realization.relation_certificate()
            residuals.update({f"relation:{key}": value for key, value in relation_certificate.residuals.items()})
            checked.append("grammar realization")
            valid &= relation_certificate.valid
            exact &= relation_certificate.exactness is Exactness.EXACT

        if self.seed_system is not None:
            seed_certificate = self.seed_system.generation_certificate()
            residuals["seed_hull_codimension"] = float(
                self.seed_system.dimension - seed_certificate.diagnostics["hull_dimension"]
            )
            checked.append("seed hull")
            valid &= seed_certificate.valid
            exact &= seed_certificate.exactness is Exactness.EXACT

        if self.packet_system is not None:
            packet_certificate = self.packet_system.certificate()
            residuals.update({f"packet:{key}": value for key, value in packet_certificate.residuals.items()})
            checked.append("packet system")
            valid &= packet_certificate.valid
            exact &= packet_certificate.exactness is Exactness.EXACT

        dimensions = {
            dimension
            for dimension in (
                None if self.seed_system is None else self.seed_system.dimension,
                None if self.packet_system is None else self.packet_system.dimension,
            )
            if dimension is not None
        }
        if len(dimensions) > 1:
            raise DimensionError("seed and packet systems must have one ambient dimension")

        if self.operators and self.packet_system is not None:
            for name, operator in self.operators.items():
                matrix = self.packet_system.backend.as_matrix(operator)
                expected = (self.packet_system.dimension, self.packet_system.dimension)
                if self.packet_system.backend.shape(matrix) != expected:
                    raise DimensionError(f"operator {name!r} has incompatible dimension")
            checked.append("named ambient operators")

        if self.refinement_tower is not None:
            tower_certificate = self.refinement_tower.certificate()
            residuals.update({f"refinement:{key}": value for key, value in tower_certificate.residuals.items()})
            valid &= tower_certificate.valid
            exact &= tower_certificate.exactness is Exactness.EXACT
            if self.packet_system is not None and not any(
                self.packet_system.equivalent(level) for level in self.refinement_tower.levels
            ):
                raise DimensionError("the declared packet system is not a level of the refinement tower")
            checked.append("refinement tower")

        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT if exact else Exactness.NUMERICAL,
            residuals=residuals,
            tolerance=(
                None
                if exact
                else max(
                    1e-10,
                    0.0 if self.seed_system is None else self.seed_system.tolerance,
                    0.0 if self.packet_system is None else self.packet_system.tolerance,
                )
            ),
            assumptions_checked=tuple(checked),
            diagnostics={
                "has_grammar": self.grammar is not None,
                "has_realization": self.realization is not None,
                "has_seed_system": self.seed_system is not None,
                "has_packet_system": self.packet_system is not None,
                "operator_names": tuple(self.operators),
            },
            message="valid unified finite system" if valid else "unified system validation failed",
        )

    def analyze(self, vector: Any) -> PacketCoordinates:
        if self.packet_system is None:
            raise ValidationError("the system has no packet coordinates")
        return self.packet_system.analyze(vector)

    def block_operator(self, name: str) -> PacketOperator:
        if self.packet_system is None:
            raise ValidationError("the system has no packet coordinates")
        try:
            operator = self.operators[name]
        except KeyError as exc:
            raise KeyError(f"unknown named operator: {name}") from exc
        return PacketOperator.from_operator(operator, self.packet_system)

    def refine_coordinates(self, coordinates: PacketCoordinates, *, source_level: int, target_level: int) -> PacketCoordinates:
        if self.refinement_tower is None:
            raise ValidationError("the system has no refinement tower")
        return self.refinement_tower.refine(
            coordinates,
            source_level=source_level,
            target_level=target_level,
        )

    def coarsen_coordinates(self, coordinates: PacketCoordinates, *, source_level: int, target_level: int) -> PacketCoordinates:
        if self.refinement_tower is None:
            raise ValidationError("the system has no refinement tower")
        return self.refinement_tower.coarsen(
            coordinates,
            source_level=source_level,
            target_level=target_level,
        )

    def obstruction(self, correction_map: Any, defect: Any, *, backend: str | None = None) -> LinearObstructionProblem:
        chosen_backend = backend
        if chosen_backend is None and self.packet_system is not None:
            chosen_backend = self.packet_system.backend
        return LinearObstructionProblem(correction_map, defect, backend=chosen_backend)

    def summary(self) -> dict[str, Any]:
        certificate = self.validate()
        return {
            "valid": certificate.valid,
            "exactness": certificate.exactness.value,
            "grammar_vertices": () if self.grammar is None else self.grammar.vertices,
            "seed_dimension": None if self.seed_system is None else self.seed_system.dimension,
            "packet_names": () if self.packet_system is None else self.packet_system.names,
            "operators": tuple(self.operators),
            "refinement_levels": 0 if self.refinement_tower is None else len(self.refinement_tower.levels),
            "metadata": dict(self.metadata),
        }
