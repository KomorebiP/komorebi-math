"""Structural support algebra for packet operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ..exceptions import ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._validation import strict_integer

Address = tuple[int, int]


def _checked_support(vertex_count: int, support: Iterable[Address]) -> frozenset[Address]:
    count = strict_integer(vertex_count, "vertex_count", minimum=1)
    checked: set[Address] = set()
    for address in support:
        try:
            target, source = address
        except (TypeError, ValueError) as exc:
            raise ValidationError("support addresses must be target and source pairs") from exc
        target_index = strict_integer(target, "target vertex", minimum=0)
        source_index = strict_integer(source, "source vertex", minimum=0)
        if not 0 <= target_index < count or not 0 <= source_index < count:
            raise ValidationError("support address is outside the vertex range")
        checked.add((target_index, source_index))
    return frozenset(checked)


def _normalized_support(support: Iterable[Address]) -> frozenset[Address]:
    checked: set[Address] = set()
    for address in support:
        try:
            target, source = address
        except (TypeError, ValueError) as exc:
            raise ValidationError("support addresses must be target and source pairs") from exc
        checked.add(
            (
                strict_integer(target, "target vertex", minimum=0),
                strict_integer(source, "source vertex", minimum=0),
            )
        )
    return frozenset(checked)


@documented(
    ProvenanceRecord(
        identifier="PSC-VIII-T15",
        title="Packet support multiplication",
        status=ClaimStatus.PROVED,
        source="Paper VIII",
        assumptions=("finite packet support relations",),
        verification_suite="algorithms.support",
    )
)
def support_product(left: Iterable[Address], right: Iterable[Address]) -> frozenset[Address]:
    """Return the structural support of ``left`` after ``right``.

    Addresses use ``(target, source)`` order.
    """

    left_set = _normalized_support(left)
    right_set = _normalized_support(right)
    return frozenset(
        (target, source)
        for target, middle in left_set
        for middle_right, source in right_set
        if middle == middle_right
    )


def support_power(support: Iterable[Address], exponent: int) -> frozenset[Address]:
    """Return the structural support of a positive operator power."""

    exponent_value = strict_integer(exponent, "exponent", minimum=0)
    base = _normalized_support(support)
    if exponent_value == 0:
        vertices = {index for edge in base for index in edge}
        return frozenset((index, index) for index in vertices)
    result = base
    for _ in range(1, exponent_value):
        result = support_product(base, result)
    return result


def support_polynomial(
    support: Iterable[Address],
    coefficients: Sequence[complex],
    *,
    vertex_count: int,
) -> frozenset[Address]:
    """Return the structural support inclusion for a polynomial in an operator."""

    base = _checked_support(vertex_count, support)
    result: set[Address] = set()
    if coefficients and coefficients[0] != 0:
        result.update((index, index) for index in range(vertex_count))
    power = frozenset((index, index) for index in range(vertex_count))
    for degree, coefficient in enumerate(coefficients[1:], start=1):
        power = base if degree == 1 else support_product(base, power)
        if coefficient != 0:
            result.update(power)
    return frozenset(result)


def transitive_closure(
    vertex_count: int,
    support: Iterable[Address],
    *,
    include_identity: bool = True,
) -> frozenset[Address]:
    """Return directed reachability as a packet support relation."""

    checked = _checked_support(vertex_count, support)
    reach = [[False] * vertex_count for _ in range(vertex_count)]
    if include_identity:
        for index in range(vertex_count):
            reach[index][index] = True
    for target, source in checked:
        reach[source][target] = True
    for middle in range(vertex_count):
        for source in range(vertex_count):
            if not reach[source][middle]:
                continue
            for target in range(vertex_count):
                reach[source][target] = reach[source][target] or reach[middle][target]
    return frozenset(
        (target, source)
        for source in range(vertex_count)
        for target in range(vertex_count)
        if reach[source][target]
    )


def inverse_support_if_acyclic(vertex_count: int, support: Iterable[Address]) -> frozenset[Address]:
    """Support inclusion for the inverse of an acyclic unit diagonal operator."""

    graph = SupportGraph(vertex_count, support)
    if not graph.is_acyclic:
        raise ValidationError("acyclic inverse support requires an acyclic strict graph")
    return transitive_closure(vertex_count, graph.support, include_identity=True)


@documented(
    ProvenanceRecord(
        identifier="PSC-VIII-T18",
        title="Schur complement fill support",
        status=ClaimStatus.PROVED,
        source="Paper VIII",
        assumptions=("finite retained and hidden packet partition",),
        limitations=("support is an inclusion because algebraic cancellation may remove permitted blocks",),
        verification_suite="algorithms.support",
    )
)
def schur_fill_support(
    retained: Sequence[int],
    hidden: Sequence[int],
    support: Iterable[Address],
) -> frozenset[Address]:
    """Return support permitted by retained to hidden to retained paths."""

    retained_set = {strict_integer(index, "retained packet", minimum=0) for index in retained}
    hidden_set = {strict_integer(index, "hidden packet", minimum=0) for index in hidden}
    if not retained_set:
        raise ValidationError("at least one retained packet is required")
    if retained_set & hidden_set:
        raise ValidationError("retained and hidden packets must be disjoint")
    vertices = retained_set | hidden_set
    if vertices != set(range(max(vertices) + 1)):
        raise ValidationError("retained and hidden packets must form a zero based contiguous partition")
    checked = _checked_support(len(vertices), support)
    result = {
        (target, source)
        for target, source in checked
        if target in retained_set and source in retained_set
    }
    hidden_adjacency: dict[int, set[int]] = {index: set() for index in hidden_set}
    for target, source in checked:
        if source in hidden_set and target in hidden_set:
            hidden_adjacency[source].add(target)

    for source in retained_set:
        frontier = {
            target
            for target, edge_source in checked
            if edge_source == source and target in hidden_set
        }
        reached = set(frontier)
        while frontier:
            current = frontier.pop()
            for target in hidden_adjacency[current]:
                if target not in reached:
                    reached.add(target)
                    frontier.add(target)
        for hidden_vertex in reached:
            for target, edge_source in checked:
                if edge_source == hidden_vertex and target in retained_set:
                    result.add((target, source))
    return frozenset(result)


def topological_levels(
    vertex_count: int,
    support: Iterable[Address],
) -> tuple[tuple[int, ...], ...]:
    """Partition an acyclic support graph into executable levels."""

    graph = SupportGraph(vertex_count, support)
    if not graph.is_acyclic:
        raise ValidationError("topological levels require an acyclic graph")
    indegree = [0] * vertex_count
    adjacency = {index: [] for index in range(vertex_count)}
    for target, source in graph.support:
        if target == source:
            continue
        adjacency[source].append(target)
        indegree[target] += 1
    current = sorted(index for index, degree in enumerate(indegree) if degree == 0)
    levels: list[tuple[int, ...]] = []
    seen = 0
    while current:
        levels.append(tuple(current))
        next_level: list[int] = []
        for source in current:
            seen += 1
            for target in sorted(adjacency[source]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    next_level.append(target)
        current = sorted(next_level)
    if seen != vertex_count:
        raise ValidationError("graph contains a directed cycle")
    return tuple(levels)


def longest_path_vertices(vertex_count: int, support: Iterable[Address]) -> int:
    """Return the number of vertices on a longest directed path."""

    levels = topological_levels(vertex_count, support)
    graph = SupportGraph(vertex_count, support)
    order = tuple(vertex for level in levels for vertex in level)
    distance = [1] * vertex_count
    for source in order:
        for target in graph.targets(source):
            if target != source:
                distance[target] = max(distance[target], distance[source] + 1)
    return max(distance)


@dataclass(frozen=True, slots=True)
class SupportGraph:
    """Finite directed graph encoded by target and source addresses."""

    vertex_count: int
    support: frozenset[Address]

    def __init__(self, vertex_count: int, support: Iterable[Address]) -> None:
        count = strict_integer(vertex_count, "vertex_count", minimum=1)
        object.__setattr__(self, "vertex_count", count)
        object.__setattr__(self, "support", _checked_support(count, support))

    def targets(self, source: int) -> tuple[int, ...]:
        source_index = strict_integer(source, "source vertex", minimum=0)
        if not 0 <= source_index < self.vertex_count:
            raise IndexError("source vertex is outside the graph")
        return tuple(sorted(target for target, edge_source in self.support if edge_source == source_index))

    def sources(self, target: int) -> tuple[int, ...]:
        target_index = strict_integer(target, "target vertex", minimum=0)
        if not 0 <= target_index < self.vertex_count:
            raise IndexError("target vertex is outside the graph")
        return tuple(sorted(source for edge_target, source in self.support if edge_target == target_index))

    def reachable(self, sources: Iterable[int]) -> frozenset[int]:
        reached = {strict_integer(source, "source vertex", minimum=0) for source in sources}
        if any(not 0 <= source < self.vertex_count for source in reached):
            raise ValidationError("source vertex is outside the graph")
        frontier = list(reached)
        while frontier:
            source = frontier.pop()
            for target in self.targets(source):
                if target not in reached:
                    reached.add(target)
                    frontier.append(target)
        return frozenset(reached)

    @property
    def is_acyclic(self) -> bool:
        strict = frozenset(edge for edge in self.support if edge[0] != edge[1])
        color = [0] * self.vertex_count
        adjacency = {index: [] for index in range(self.vertex_count)}
        for target, source in strict:
            adjacency[source].append(target)

        def visit(vertex: int) -> bool:
            color[vertex] = 1
            for target in adjacency[vertex]:
                if color[target] == 1:
                    return False
                if color[target] == 0 and not visit(target):
                    return False
            color[vertex] = 2
            return True

        return all(color[vertex] != 0 or visit(vertex) for vertex in range(self.vertex_count))

    def closure(self, *, include_identity: bool = True) -> frozenset[Address]:
        return transitive_closure(
            self.vertex_count,
            self.support,
            include_identity=include_identity,
        )


__all__ = [
    "Address",
    "SupportGraph",
    "inverse_support_if_acyclic",
    "longest_path_vertices",
    "schur_fill_support",
    "support_polynomial",
    "support_power",
    "support_product",
    "topological_levels",
    "transitive_closure",
]
