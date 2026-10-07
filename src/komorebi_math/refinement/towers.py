"""Exact packet refinements and finite refinement towers."""

from __future__ import annotations

from .._residuals import ResidualAudit, finite_tolerance

from dataclasses import dataclass
from typing import Any

from ..certificates import Certificate, Exactness
from ..core.operators import PacketOperator
from ..core.packets import PacketCoordinates, PacketSystem
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented


@documented(
    ProvenanceRecord(
        identifier="PSC-II-D02",
        title="Packet refinement",
        status=ClaimStatus.DEFINITION,
        source="Paper II",
        assumptions=("finite packet systems on one ambient space",),
        verification_suite="refinement.towers",
    )
)
@dataclass(frozen=True, slots=True)
class PacketRefinement:
    """One exact refinement from a coarse packet system to a fine system."""

    coarse: PacketSystem
    fine: PacketSystem
    parent_of_fine: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.coarse.backend.name != self.fine.backend.name:
            raise ValidationError("coarse and fine packet systems must use one backend")
        if self.coarse.dimension != self.fine.dimension:
            raise DimensionError("coarse and fine packets must share an ambient dimension")
        if len(self.parent_of_fine) != len(self.fine):
            raise DimensionError("every fine packet must have one parent")
        if any(parent < 0 or parent >= len(self.coarse) for parent in self.parent_of_fine):
            raise ValueError("a fine packet parent is outside the coarse packet range")
        if set(self.parent_of_fine) != set(range(len(self.coarse))):
            raise ValidationError("every coarse packet must have at least one fine child")
        self.certificate().require()

    @property
    def backend(self):
        return self.coarse.backend

    def children(self, coarse_index: int | str) -> tuple[int, ...]:
        index = self.coarse.index(coarse_index) if isinstance(coarse_index, str) else coarse_index
        return tuple(child for child, parent in enumerate(self.parent_of_fine) if parent == index)

    def parent(self, fine_index: int | str) -> int:
        index = self.fine.index(fine_index) if isinstance(fine_index, str) else fine_index
        return self.parent_of_fine[index]

    def certificate(self) -> Certificate:
        coarse_certificate = self.coarse.certificate()
        fine_certificate = self.fine.certificate()
        tolerance = max(finite_tolerance(self.coarse.tolerance), finite_tolerance(self.fine.tolerance))
        audit = ResidualAudit(self.backend, tolerance)
        residuals: dict[str, float] = {}
        for coarse_index, projector in enumerate(self.coarse.projectors):
            total = self.backend.zeros(self.coarse.dimension, self.coarse.dimension)
            for child in self.children(coarse_index):
                total = self.backend.add(total, self.fine.projectors[child])
            residuals[self.coarse.names[coarse_index]] = audit.check(
                self.coarse.names[coarse_index], self.backend.subtract(projector, total))
        valid = coarse_certificate.valid and fine_certificate.valid and audit.valid
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT if self.backend.exact else Exactness.NUMERICAL,
            residuals=residuals,
            tolerance=None if self.backend.exact else tolerance,
            assumptions_checked=("packet axioms", "one parent per fine packet", "coarse projector equals child sum"),
            assumptions_unchecked=coarse_certificate.assumptions_unchecked + fine_certificate.assumptions_unchecked + audit.unresolved,
            diagnostics={"coarse_certificate": coarse_certificate.to_dict(),
                         "fine_certificate": fine_certificate.to_dict(), "residual_decisions": audit.decisions},
            theorem_id="PSC-II-T03",
            message="valid packet refinement" if valid else "parent packet or refinement checks failed or are unresolved",
        )

    @documented(
        ProvenanceRecord(
            identifier="PSC-II-T04",
            title="Exact coordinate change across one refinement",
            status=ClaimStatus.PROVED,
            source="Paper II",
            assumptions=("exact packet refinement",),
            verification_suite="refinement.towers",
        )
    )
    def coarsen(self, coordinates: PacketCoordinates) -> PacketCoordinates:
        """Sum fine coordinates inside every coarse parent."""

        if not coordinates.system.equivalent(self.fine):
            raise ValidationError("coordinates do not belong to the fine packet system")
        values: list[Any] = []
        for coarse_index in range(len(self.coarse)):
            children = self.children(coarse_index)
            total = self.backend.scale(0, coordinates.values[children[0]])
            for child in children:
                total = self.backend.add(total, coordinates.values[child])
            values.append(total)
        return PacketCoordinates(self.coarse, tuple(values))

    def refine(self, coordinates: PacketCoordinates) -> PacketCoordinates:
        """Resolve coarse coordinates into their fine descendants."""

        if not coordinates.system.equivalent(self.coarse):
            raise ValidationError("coordinates do not belong to the coarse packet system")
        vector = coordinates.reconstruct()
        return self.fine.analyze(vector)

    def refine_operator(self, operator: PacketOperator | Any) -> PacketOperator:
        """Resolve one ambient operator in fine packet coordinates."""

        matrix = operator.reconstruct() if isinstance(operator, PacketOperator) else operator
        return PacketOperator.from_operator(matrix, self.fine)

    def coarsen_operator(self, operator: PacketOperator | Any) -> PacketOperator:
        """Resolve one ambient operator in coarse packet coordinates."""

        matrix = operator.reconstruct() if isinstance(operator, PacketOperator) else operator
        return PacketOperator.from_operator(matrix, self.coarse)

    def support_reflection(self, coordinates: PacketCoordinates) -> dict[str, tuple[str, ...]]:
        """Return the fine support lying under every active coarse packet."""

        fine_coordinates = coordinates if coordinates.system.equivalent(self.fine) else self.refine(coordinates)
        result: dict[str, tuple[str, ...]] = {}
        for coarse_index, coarse_name in enumerate(self.coarse.names):
            active = tuple(
                self.fine.names[child]
                for child in self.children(coarse_index)
                if not self.backend.is_zero(fine_coordinates.values[child], tolerance=self.fine.tolerance)
            )
            if active:
                result[coarse_name] = active
        return result


@documented(
    ProvenanceRecord(
        identifier="PSC-II-T05",
        title="Coherence of finite refinement towers",
        status=ClaimStatus.PROVED,
        source="Paper II",
        assumptions=("adjacent exact packet refinements",),
        verification_suite="refinement.towers",
    )
)
class RefinementTower:
    """A finite sequence of exact packet refinements."""

    def __init__(self, refinements: tuple[PacketRefinement, ...] | list[PacketRefinement]) -> None:
        self.refinements = tuple(refinements)
        if not self.refinements:
            raise ValueError("a refinement tower requires at least one refinement")
        for left, right in zip(self.refinements, self.refinements[1:]):
            if not left.fine.equivalent(right.coarse):
                raise ValidationError("adjacent refinement levels do not agree")

    @property
    def levels(self) -> tuple[PacketSystem, ...]:
        return (self.refinements[0].coarse,) + tuple(refinement.fine for refinement in self.refinements)

    def refine(self, coordinates: PacketCoordinates, *, source_level: int, target_level: int) -> PacketCoordinates:
        if target_level < source_level:
            raise ValueError("target level must not be coarser than source level")
        if not 0 <= source_level < len(self.levels) or not 0 <= target_level < len(self.levels):
            raise IndexError("refinement level is outside the tower")
        current = coordinates
        if not current.system.equivalent(self.levels[source_level]):
            raise ValidationError("coordinates do not belong to the declared source level")
        for level in range(source_level, target_level):
            current = self.refinements[level].refine(current)
        return current

    def coarsen(self, coordinates: PacketCoordinates, *, source_level: int, target_level: int) -> PacketCoordinates:
        if target_level > source_level:
            raise ValueError("target level must not be finer than source level")
        if not 0 <= source_level < len(self.levels) or not 0 <= target_level < len(self.levels):
            raise IndexError("refinement level is outside the tower")
        current = coordinates
        if not current.system.equivalent(self.levels[source_level]):
            raise ValidationError("coordinates do not belong to the declared source level")
        for level in range(source_level - 1, target_level - 1, -1):
            current = self.refinements[level].coarsen(current)
        return current

    def descendant_address(self, coarse_index: int, fine_index: int, *, coarse_level: int, fine_level: int) -> bool:
        if fine_level <= coarse_level:
            raise ValueError("fine level must be strictly greater than coarse level")
        current = fine_index
        for level in range(fine_level - 1, coarse_level - 1, -1):
            current = self.refinements[level].parent(current)
        return current == coarse_index

    def certificate(self) -> Certificate:
        certificates = tuple(refinement.certificate() for refinement in self.refinements)
        return Certificate(
            valid=all(certificate.valid for certificate in certificates),
            exactness=(
                Exactness.EXACT
                if all(certificate.exactness is Exactness.EXACT for certificate in certificates)
                else Exactness.NUMERICAL
            ),
            residuals={
                f"level_{level}:{name}": residual
                for level, certificate in enumerate(certificates)
                for name, residual in certificate.residuals.items()
            },
            tolerance=None if all(certificate.exactness is Exactness.EXACT for certificate in certificates) else max(
                refinement.coarse.tolerance for refinement in self.refinements
            ),
            assumptions_checked=("all adjacent refinements", "level coherence"),
            theorem_id="PSC-II-T05",
            message="valid refinement tower" if all(certificate.valid for certificate in certificates) else "invalid refinement tower",
        )


def common_refinement(left: PacketSystem, right: PacketSystem, *, names: tuple[str, ...] | None = None) -> tuple[PacketSystem, PacketRefinement, PacketRefinement]:
    """Construct the common refinement of two commuting packet systems."""

    if left.backend.name != right.backend.name or left.dimension != right.dimension:
        raise ValidationError("packet systems must share one backend and ambient dimension")
    backend = left.backend
    tolerance = max(left.tolerance, right.tolerance)
    for left_projector in left.projectors:
        for right_projector in right.projectors:
            commutator = backend.subtract(
                backend.matmul(left_projector, right_projector),
                backend.matmul(right_projector, left_projector),
            )
            if not backend.is_zero(commutator, tolerance=tolerance):
                raise ValidationError("common refinement requires commuting packet systems")

    projectors: list[Any] = []
    left_parents: list[int] = []
    right_parents: list[int] = []
    generated_names: list[str] = []
    for left_index, left_projector in enumerate(left.projectors):
        for right_index, right_projector in enumerate(right.projectors):
            product = backend.matmul(left_projector, right_projector)
            if backend.is_zero(product, tolerance=tolerance):
                continue
            projectors.append(product)
            left_parents.append(left_index)
            right_parents.append(right_index)
            generated_names.append(f"{left.names[left_index]} & {right.names[right_index]}")
    if not projectors:
        raise ValidationError("commuting packet systems have no nonzero common packets")
    common = PacketSystem(
        projectors,
        names=tuple(generated_names) if names is None else names,
        backend=backend,
        tolerance=tolerance,
    )
    return (
        common,
        PacketRefinement(left, common, tuple(left_parents)),
        PacketRefinement(right, common, tuple(right_parents)),
    )


__all__ = ["PacketRefinement", "RefinementTower", "common_refinement"]
