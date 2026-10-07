"""Symbolic automorphic induction plans for finite monomial parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Hashable, TypeVar

from ...certificates import Certificate, ConditionalResult, Exactness
from .characters import InducedRepresentation

Element = TypeVar("Element", bound=Hashable)


@dataclass(frozen=True, slots=True)
class AutomorphicInductionPlan(Generic[Element]):
    """A conditional symbolic plan attached to one induced character parameter."""

    representation: InducedRepresentation[Element]
    base_field: str
    inducing_field: str
    inducing_character: str
    cyclic_tower: tuple[str, ...] = ()

    def operations(self) -> ConditionalResult[tuple[str, ...]]:
        """Return the formal class field and automorphic induction operations."""

        steps = [f"class field theory for {self.inducing_character} over {self.inducing_field}"]
        if self.cyclic_tower:
            steps.extend(f"automorphic induction along {extension}" for extension in self.cyclic_tower)
        else:
            steps.append(f"automorphic induction from {self.inducing_field} to {self.base_field}")
        certificate = Certificate(
            valid=True,
            exactness=Exactness.CONDITIONAL,
            assumptions_checked=("finite induced representation", "declared field and character labels"),
            assumptions_unchecked=(
                "the declared inducing character is algebraic when required",
                "the declared field tower satisfies the external automorphic induction hypotheses",
            ),
            diagnostics={"representation_dimension": self.representation.dimension},
            theorem_id="PSC-LANG-AUTOMORPHIC-INDUCTION-PLAN",
            message="symbolic automorphic induction plan constructed",
        )
        return ConditionalResult(tuple(steps), certificate)


__all__ = ["AutomorphicInductionPlan"]
