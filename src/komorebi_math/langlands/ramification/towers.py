"""Finite ramification packet and Clifford tower diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..._number_theory import require_integer, require_prime
from ...certificates import Certificate, Exactness
from ...exceptions import ValidationError
from ..local import AlternatingForm, QuadraticRefinement


@dataclass(frozen=True, slots=True)
class ProjectiveMultiplicityPacket:
    """A finite projective multiplicity packet at one ramification stage."""

    prime: int
    quotient_rank: int
    form: AlternatingForm
    phase_label: str | None = None
    quadratic_refinement: QuadraticRefinement | None = None

    def __post_init__(self) -> None:
        p = require_prime(self.prime)
        rank = require_integer(self.quotient_rank, name="quotient_rank")
        if rank < 0 or self.form.prime != p or self.form.dimension != rank:
            raise ValidationError("projective packet prime, quotient rank, and form dimension must agree")
        if p == 2 and self.quadratic_refinement is not None and self.quadratic_refinement.form != self.form:
            raise ValidationError("dyadic quadratic refinement belongs to a different alternating form")
        object.__setattr__(self, "prime", p)
        object.__setattr__(self, "quotient_rank", rank)

    @property
    def radical_dimension(self) -> int:
        """Return the projective commutator radical dimension."""

        return self.form.radical_dimension

    @property
    def symplectic_rank(self) -> int:
        """Return the rank of the radical-free symplectic quotient."""

        return self.form.rank

    @property
    def heisenberg_dimension(self) -> int:
        """Return the irreducible Heisenberg dimension of the nondegenerate quotient."""

        if self.symplectic_rank % 2:
            raise ArithmeticError("alternating form rank must be even")
        return self.prime ** (self.symplectic_rank // 2)

    def phase_diagnostic(self) -> dict[str, object]:
        """Return the available finite phase data without inventing missing normalization."""

        data: dict[str, object] = {
            "phase_label": self.phase_label,
            "prime": self.prime,
            "symplectic_rank": self.symplectic_rank,
        }
        if self.prime == 2 and self.quadratic_refinement is not None and self.form.nondegenerate:
            data["arf_invariant"] = self.quadratic_refinement.arf_invariant()
            data["gauss_sign"] = self.quadratic_refinement.normalized_gauss_sign()
        return data


@dataclass(frozen=True, slots=True)
class RamificationStage:
    """One finite stage in a declared ramification-adapted tower."""

    break_number: int
    parent_order: int
    child_order: int
    orbit_size: int
    multiplicity_dimension: int
    packet: ProjectiveMultiplicityPacket | None = None
    label: str = ""

    def __post_init__(self) -> None:
        break_number = require_integer(self.break_number, name="break_number")
        parent = require_integer(self.parent_order, name="parent_order")
        child = require_integer(self.child_order, name="child_order")
        orbit = require_integer(self.orbit_size, name="orbit_size")
        multiplicity = require_integer(self.multiplicity_dimension, name="multiplicity_dimension")
        if break_number < 0 or min(parent, child, orbit, multiplicity) < 1:
            raise ValidationError("ramification stage data must be nonnegative or positive as appropriate")
        if parent % child or parent // child != orbit:
            raise ValidationError("ramification orbit size must equal the subgroup index")
        object.__setattr__(self, "break_number", break_number)
        object.__setattr__(self, "parent_order", parent)
        object.__setattr__(self, "child_order", child)
        object.__setattr__(self, "orbit_size", orbit)
        object.__setattr__(self, "multiplicity_dimension", multiplicity)

    @property
    def reconstructed_dimension(self) -> int:
        """Return the dimension contributed after induction from this stage."""

        return self.orbit_size * self.multiplicity_dimension


@dataclass(frozen=True, slots=True)
class RamificationTowerReport:
    """Dimension, finite packet, and phase data for a declared tower."""

    breaks: tuple[int, ...]
    dimensions: tuple[int, ...]
    unresolved_phase_stages: tuple[int, ...]
    certificate: Certificate


class RamificationCliffordTower:
    """A finite ramification-adapted chain of Clifford packet stages."""

    def __init__(self, stages: Iterable[RamificationStage]) -> None:
        self.stages = tuple(stages)
        if not self.stages:
            raise ValidationError("a ramification tower requires at least one stage")
        if any(self.stages[index].break_number >= self.stages[index + 1].break_number for index in range(len(self.stages) - 1)):
            raise ValidationError("ramification break numbers must be strictly increasing")
        for index in range(len(self.stages) - 1):
            if self.stages[index].child_order != self.stages[index + 1].parent_order:
                raise ValidationError("ramification group orders do not form a chain")

    def report(self) -> RamificationTowerReport:
        """Return dimensions and finite phase gaps in the tower."""

        unresolved = tuple(
            index
            for index, stage in enumerate(self.stages)
            if stage.packet is not None
            and stage.packet.phase_label is None
            and not (stage.packet.prime == 2 and stage.packet.quadratic_refinement is not None)
        )
        dimensions = tuple(stage.reconstructed_dimension for stage in self.stages)
        return RamificationTowerReport(
            breaks=tuple(stage.break_number for stage in self.stages),
            dimensions=dimensions,
            unresolved_phase_stages=unresolved,
            certificate=Certificate(
                valid=True,
                exactness=Exactness.CONDITIONAL,
                assumptions_checked=("finite ramification order chain", "declared Clifford orbit dimensions"),
                assumptions_unchecked=tuple(f"canonical phase at ramification stage {index}" for index in unresolved),
                diagnostics={"stage_count": len(self.stages), "unresolved_phase_stages": unresolved},
                theorem_id="PSC-LANG-RAMIFICATION-TOWER",
                message="finite ramification tower assembled",
            ),
        )

    def tensor_packet(self, other: "RamificationCliffordTower") -> "RamificationCliffordTower":
        """Form a stagewise tensor diagnostic for aligned break towers."""

        if len(self.stages) != len(other.stages) or any(a.break_number != b.break_number for a, b in zip(self.stages, other.stages, strict=True)):
            raise ValidationError("ramification towers must have aligned break numbers")
        stages: list[RamificationStage] = []
        for left, right in zip(self.stages, other.stages, strict=True):
            packet = None
            if left.packet is not None and right.packet is not None and left.packet.prime == right.packet.prime:
                p = left.packet.prime
                block = np.block([
                    [left.packet.form.matrix, np.zeros((left.packet.form.dimension, right.packet.form.dimension), dtype=int)],
                    [np.zeros((right.packet.form.dimension, left.packet.form.dimension), dtype=int), right.packet.form.matrix],
                ])
                packet = ProjectiveMultiplicityPacket(p, block.shape[0], AlternatingForm(block, p), phase_label="tensor product phase requires multiplicativity")
            stages.append(
                RamificationStage(
                    left.break_number,
                    left.parent_order * right.parent_order,
                    left.child_order * right.child_order,
                    left.orbit_size * right.orbit_size,
                    left.multiplicity_dimension * right.multiplicity_dimension,
                    packet,
                    label=f"{left.label} tensor {right.label}".strip(),
                )
            )
        return RamificationCliffordTower(stages)


__all__ = [
    "ProjectiveMultiplicityPacket",
    "RamificationCliffordTower",
    "RamificationStage",
    "RamificationTowerReport",
]
