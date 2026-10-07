"""Linear characters and induced finite group representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Hashable, Mapping, TypeVar

import numpy as np

from ...certificates import Certificate, Exactness
from ...exceptions import ValidationError
from .groups import FiniteGroup, FiniteSubgroup

Element = TypeVar("Element", bound=Hashable)


@dataclass(frozen=True, slots=True)
class LinearCharacter(Generic[Element]):
    """A one dimensional complex character of an explicit finite subgroup."""

    subgroup: FiniteSubgroup[Element]
    values: Mapping[Element, complex]
    tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if set(self.values) != set(self.subgroup.elements):
            raise ValidationError("a linear character value is required on every subgroup element")
        if abs(self.values[self.subgroup.parent.identity] - 1.0) > self.tolerance:
            raise ValidationError("linear character does not send the identity to one")
        for left in self.subgroup.elements:
            for right in self.subgroup.elements:
                product_value = self.subgroup.parent.multiply(left, right)
                if abs(self.values[product_value] - self.values[left] * self.values[right]) > self.tolerance:
                    raise ValidationError("linear character is not multiplicative")

    def __call__(self, element: Element) -> complex:
        """Evaluate the character."""

        try:
            return complex(self.values[element])
        except KeyError as exc:
            raise ValidationError("character evaluated outside its subgroup") from exc

    def conjugate(self) -> "LinearCharacter[Element]":
        """Return the complex conjugate character."""

        return LinearCharacter(self.subgroup, {element: np.conjugate(value) for element, value in self.values.items()}, self.tolerance)


class InducedRepresentation(Generic[Element]):
    """The explicit monomial representation induced from a linear character."""

    def __init__(self, group: FiniteGroup[Element], character: LinearCharacter[Element]) -> None:
        character.subgroup.require_parent(group)
        self.group = group
        self.character = character
        self.subgroup = character.subgroup
        self.coset_representatives = group.left_coset_representatives(self.subgroup)
        self.dimension = len(self.coset_representatives)
        self._coset_lookup: dict[Element, tuple[int, Element]] = {}
        for index, representative in enumerate(self.coset_representatives):
            for h in self.subgroup.elements:
                element = self.group.multiply(representative, h)
                self._coset_lookup[element] = (index, h)
        if len(self._coset_lookup) != self.group.order:
            raise ArithmeticError("left cosets do not partition the finite group")

    def basis_action(self, element: Element) -> tuple[tuple[int, ...], tuple[complex, ...]]:
        """Return target basis indices and monomial scalars."""

        if element not in self.group.element_set:
            raise ValidationError("representation evaluated outside the finite group")
        targets: list[int] = []
        scalars: list[complex] = []
        for representative in self.coset_representatives:
            product_value = self.group.multiply(element, representative)
            target, h = self._coset_lookup[product_value]
            targets.append(target)
            scalars.append(self.character(h))
        return tuple(targets), tuple(scalars)

    def matrix(self, element: Element) -> np.ndarray:
        """Return the induced monomial matrix."""

        targets, scalars = self.basis_action(element)
        matrix = np.zeros((self.dimension, self.dimension), dtype=complex)
        for source, (target, scalar) in enumerate(zip(targets, scalars, strict=True)):
            matrix[target, source] = scalar
        return matrix

    def trace(self, element: Element) -> complex:
        """Return the induced character value."""

        return complex(np.trace(self.matrix(element)))

    def certificate(self, *, tolerance: float = 1e-10) -> Certificate:
        """Exhaustively check the representation law."""

        maximum = 0.0
        for left in self.group.elements:
            for right in self.group.elements:
                product_value = self.group.multiply(left, right)
                maximum = max(maximum, float(np.linalg.norm(self.matrix(left) @ self.matrix(right) - self.matrix(product_value))))
        return Certificate(
            valid=maximum <= tolerance,
            exactness=Exactness.NUMERICAL,
            residuals={"representation_law": maximum},
            tolerance=tolerance,
            assumptions_checked=("finite explicit group", "multiplicative inducing character"),
            diagnostics={"dimension": self.dimension, "index": self.subgroup.index},
            theorem_id="KM-INDUCED-REPRESENTATION",
            message="induced representation law verified" if maximum <= tolerance else "induced representation law failed",
        )


__all__ = ["InducedRepresentation", "LinearCharacter"]
