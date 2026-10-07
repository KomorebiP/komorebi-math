"""Orbit packet decompositions for monomial tensor operations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from math import comb
from typing import Generic, Hashable, Iterable, TypeVar

from ..._number_theory import require_integer
from ...certificates import Certificate, Exactness
from ...exceptions import ValidationError
from .characters import InducedRepresentation

Element = TypeVar("Element", bound=Hashable)
BasisState = tuple[int, ...]


def weak_compositions(total: int, length: int) -> tuple[tuple[int, ...], ...]:
    """Return all weak compositions of ``total`` into ``length`` parts."""

    m = require_integer(total, name="total")
    n = require_integer(length, name="length")
    if m < 0 or n < 1:
        raise ValidationError("weak composition parameters are invalid")
    if n == 1:
        return ((m,),)
    output: list[tuple[int, ...]] = []
    for first in range(m + 1):
        for tail in weak_compositions(m - first, n - 1):
            output.append((first, *tail))
    return tuple(output)


def _inversion_sign(values: tuple[int, ...]) -> int:
    inversions = sum(1 for i in range(len(values)) for j in range(i + 1, len(values)) if values[i] > values[j])
    return -1 if inversions % 2 else 1


@dataclass(frozen=True, slots=True)
class OrbitPacket(Generic[Element]):
    """One basis orbit, stabilizer, and stabilizer character."""

    representative: BasisState
    orbit: tuple[BasisState, ...]
    stabilizer: tuple[Element, ...]
    character_values: tuple[complex, ...]

    @property
    def dimension(self) -> int:
        """Return the orbit dimension."""

        return len(self.orbit)


class MonomialAction(Generic[Element]):
    """A finite monomial action described by permutations and basis scalars."""

    def __init__(self, representation: InducedRepresentation[Element]) -> None:
        self.representation = representation
        self.group = representation.group
        self.dimension = representation.dimension

    def _symmetric_action(self, element: Element, state: BasisState) -> tuple[BasisState, complex]:
        targets, scalars = self.representation.basis_action(element)
        output = [0] * self.dimension
        scalar = 1.0 + 0.0j
        for source, multiplicity in enumerate(state):
            output[targets[source]] += multiplicity
            scalar *= scalars[source] ** multiplicity
        return tuple(output), scalar

    def _exterior_action(self, element: Element, state: BasisState) -> tuple[BasisState, complex]:
        targets, scalars = self.representation.basis_action(element)
        images = tuple(targets[index] for index in state)
        if len(set(images)) != len(images):
            raise ArithmeticError("permutation action collapsed an exterior basis subset")
        ordered = tuple(sorted(images))
        scalar = complex(_inversion_sign(images))
        for index in state:
            scalar *= scalars[index]
        return ordered, scalar

    def _orbit_packets(
        self,
        states: tuple[BasisState, ...],
        action,
    ) -> tuple[OrbitPacket[Element], ...]:
        remaining = set(states)
        output: list[OrbitPacket[Element]] = []
        while remaining:
            representative = min(remaining)
            orbit_map: dict[BasisState, complex] = {}
            stabilizer: list[Element] = []
            character_values: list[complex] = []
            for element in self.group.elements:
                target, scalar = action(element, representative)
                orbit_map.setdefault(target, scalar)
                if target == representative:
                    stabilizer.append(element)
                    character_values.append(scalar)
            orbit = tuple(sorted(orbit_map))
            remaining.difference_update(orbit)
            output.append(OrbitPacket(representative, orbit, tuple(stabilizer), tuple(character_values)))
        return tuple(output)

    def symmetric_power_packets(self, degree: int) -> tuple[OrbitPacket[Element], ...]:
        """Return the weak-composition orbit packets of a symmetric power."""

        m = require_integer(degree, name="degree")
        if m < 0:
            raise ValidationError("symmetric degree must be nonnegative")
        states = weak_compositions(m, self.dimension)
        packets = self._orbit_packets(states, self._symmetric_action)
        expected = comb(self.dimension + m - 1, m)
        if sum(packet.dimension for packet in packets) != expected:
            raise ArithmeticError("symmetric orbit packets do not conserve dimension")
        return packets

    def exterior_power_packets(self, degree: int) -> tuple[OrbitPacket[Element], ...]:
        """Return the subset orbit packets of an exterior power."""

        r = require_integer(degree, name="degree")
        if not 0 <= r <= self.dimension:
            raise ValidationError("exterior degree lies outside the representation dimension")
        states = tuple(tuple(state) for state in combinations(range(self.dimension), r))
        packets = self._orbit_packets(states, self._exterior_action)
        if sum(packet.dimension for packet in packets) != comb(self.dimension, r):
            raise ArithmeticError("exterior orbit packets do not conserve dimension")
        return packets

    def tensor_packets(self, other: "MonomialAction[Element]") -> tuple[OrbitPacket[Element], ...]:
        """Return basis orbit packets for a tensor product over the same group."""

        if other.group is not self.group:
            raise ValidationError("tensor monomial actions must use the same finite group object")
        states = tuple((left, right) for left in range(self.dimension) for right in range(other.dimension))

        def action(element: Element, state: BasisState) -> tuple[BasisState, complex]:
            left_targets, left_scalars = self.representation.basis_action(element)
            right_targets, right_scalars = other.representation.basis_action(element)
            left, right = state
            return (left_targets[left], right_targets[right]), left_scalars[left] * right_scalars[right]

        packets = self._orbit_packets(states, action)
        if sum(packet.dimension for packet in packets) != self.dimension * other.dimension:
            raise ArithmeticError("tensor orbit packets do not conserve dimension")
        return packets

    def packet_certificate(self, packets: Iterable[OrbitPacket[Element]], *, expected_dimension: int) -> Certificate:
        """Check orbit disjointness, stabilizers, and dimension conservation."""

        packet_tuple = tuple(packets)
        states = [state for packet in packet_tuple for state in packet.orbit]
        valid = len(states) == len(set(states)) == expected_dimension
        valid = valid and all(len(packet.stabilizer) * packet.dimension == self.group.order for packet in packet_tuple)
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT,
            assumptions_checked=("finite monomial basis action", "orbit stabilizer theorem"),
            diagnostics={"packet_count": len(packet_tuple), "dimension": len(states)},
            theorem_id="PSC-LANG-MONOMIAL-ORBITS",
            message="monomial orbit packet decomposition verified" if valid else "monomial orbit packet decomposition failed",
        )


__all__ = ["MonomialAction", "OrbitPacket", "weak_compositions"]
