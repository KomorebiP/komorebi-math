"""Finite Mackey double-coset packet data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Hashable, TypeVar

from ...certificates import Certificate, Exactness
from .characters import LinearCharacter
from .groups import FiniteGroup, FiniteSubgroup

Element = TypeVar("Element", bound=Hashable)


@dataclass(frozen=True, slots=True)
class MackeyPacket(Generic[Element]):
    """One double-coset packet in a tensor product of induced characters."""

    representative: Element
    intersection: FiniteSubgroup[Element]
    character_values: tuple[complex, ...]
    induced_dimension: int


@dataclass(frozen=True, slots=True)
class MackeyDecomposition(Generic[Element]):
    """Finite double-coset packet decomposition data."""

    packets: tuple[MackeyPacket[Element], ...]
    total_dimension: int
    certificate: Certificate


def mackey_tensor_packets(
    group: FiniteGroup[Element],
    left: LinearCharacter[Element],
    right: LinearCharacter[Element],
) -> MackeyDecomposition[Element]:
    """Return the Mackey packets for two character-induced representations."""

    left.subgroup.require_parent(group)
    right.subgroup.require_parent(group)
    packets: list[MackeyPacket[Element]] = []
    total = 0
    for representative in group.double_coset_representatives(left.subgroup, right.subgroup):
        conjugate = right.subgroup.conjugate(representative, name="gKg^-1")
        intersection = left.subgroup.intersection(conjugate, name="H_cap_gKg")
        inverse = group.inverse(representative)
        values: list[complex] = []
        for element in intersection.elements:
            pulled = group.multiply(group.multiply(inverse, element), representative)
            values.append(left(element) * right(pulled))
        dimension = group.order // intersection.order
        packets.append(MackeyPacket(representative, intersection, tuple(values), dimension))
        total += dimension
    expected = left.subgroup.index * right.subgroup.index
    return MackeyDecomposition(
        packets=tuple(packets),
        total_dimension=total,
        certificate=Certificate(
            valid=total == expected,
            exactness=Exactness.EXACT,
            assumptions_checked=("finite group", "linear inducing characters", "double coset representatives"),
            diagnostics={"packet_count": len(packets), "dimension": total, "expected": expected},
            theorem_id="PSC-LANG-MACKEY-TENSOR",
            message="Mackey tensor dimensions are conserved" if total == expected else "Mackey tensor dimension mismatch",
        ),
    )


__all__ = ["MackeyDecomposition", "MackeyPacket", "mackey_tensor_packets"]
