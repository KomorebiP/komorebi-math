"""Transfer graphs, strongly connected components, and support preorders."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

from ..certificates import Certificate, ConditionalResult, Exactness
from ..exceptions import ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented


@documented(
    ProvenanceRecord(
        identifier="PSC-I-T40",
        title="Packet transfer graph",
        status=ClaimStatus.DEFINITION,
        source="Paper I",
        verification_suite="core.graphs",
    )
)
class TransferGraph:
    """A finite directed graph with adjacency stored as source by target."""

    def __init__(self, adjacency: Sequence[Sequence[bool | int]], *, names: Sequence[str] | None = None) -> None:
        rows = tuple(tuple(bool(value) for value in row) for row in adjacency)
        if not rows or any(len(row) != len(rows) for row in rows):
            raise ValidationError("a transfer graph requires a nonempty square adjacency matrix")
        self.adjacency = rows
        self.order = len(rows)
        if names is None:
            self.names = tuple(str(index) for index in range(self.order))
        else:
            self.names = tuple(str(name) for name in names)
            if len(self.names) != self.order or len(set(self.names)) != self.order:
                raise ValidationError("graph names must be unique and match the graph order")

    @classmethod
    def from_packet_operator(cls, operator, *, tolerance: float | None = None) -> "TransferGraph":
        if not operator.is_endomorphism():
            raise ValidationError("a transfer graph requires a packet endomorphism")
        mask = operator.support_mask(tolerance=tolerance)
        adjacency = [
            [mask[target][source] for target in range(len(operator.target))]
            for source in range(len(operator.source))
        ]
        return cls(adjacency, names=operator.source.names)

    @classmethod
    def from_operator_family(cls, operators: Iterable, *, tolerance: float | None = None) -> "TransferGraph":
        family = tuple(operators)
        if not family:
            raise ValidationError("operator family cannot be empty")
        first = family[0]
        if not first.is_endomorphism():
            raise ValidationError("every transfer operator must be an endomorphism")
        order = len(first.source)
        adjacency = [[False] * order for _ in range(order)]
        for operator in family:
            if not operator.source.equivalent(first.source) or not operator.target.equivalent(first.target):
                raise ValidationError("operator family must use one packet system")
            mask = operator.support_mask(tolerance=tolerance)
            for source in range(order):
                for target in range(order):
                    adjacency[source][target] |= mask[target][source]
        return cls(adjacency, names=first.source.names)

    def successors(self, vertex: int) -> tuple[int, ...]:
        self._check_vertex(vertex)
        return tuple(index for index, active in enumerate(self.adjacency[vertex]) if active)

    def predecessors(self, vertex: int) -> tuple[int, ...]:
        self._check_vertex(vertex)
        return tuple(index for index in range(self.order) if self.adjacency[index][vertex])

    def reachable(self, starts: Iterable[int]) -> frozenset[int]:
        seen = {int(start) for start in starts}
        for vertex in seen:
            self._check_vertex(vertex)
        queue = deque(seen)
        while queue:
            current = queue.popleft()
            for target in self.successors(current):
                if target not in seen:
                    seen.add(target)
                    queue.append(target)
        return frozenset(seen)

    def paths(self, source: int, target: int, *, length: int) -> tuple[tuple[int, ...], ...]:
        self._check_vertex(source)
        self._check_vertex(target)
        if length < 0:
            raise ValueError("path length must be nonnegative")
        if length == 0:
            return ((source,),) if source == target else ()
        current = [(source,)]
        for _ in range(length):
            next_paths: list[tuple[int, ...]] = []
            for path in current:
                for successor in self.successors(path[-1]):
                    next_paths.append((*path, successor))
            current = next_paths
        return tuple(path for path in current if path[-1] == target)

    def strongly_connected_components(self) -> tuple[tuple[int, ...], ...]:
        index = 0
        indices = [-1] * self.order
        lowlink = [0] * self.order
        stack: list[int] = []
        on_stack = [False] * self.order
        components: list[tuple[int, ...]] = []

        def visit(vertex: int) -> None:
            nonlocal index
            indices[vertex] = index
            lowlink[vertex] = index
            index += 1
            stack.append(vertex)
            on_stack[vertex] = True
            for target in self.successors(vertex):
                if indices[target] == -1:
                    visit(target)
                    lowlink[vertex] = min(lowlink[vertex], lowlink[target])
                elif on_stack[target]:
                    lowlink[vertex] = min(lowlink[vertex], indices[target])
            if lowlink[vertex] == indices[vertex]:
                component: list[int] = []
                while True:
                    member = stack.pop()
                    on_stack[member] = False
                    component.append(member)
                    if member == vertex:
                        break
                components.append(tuple(sorted(component)))

        for vertex in range(self.order):
            if indices[vertex] == -1:
                visit(vertex)
        return tuple(sorted(components, key=lambda component: component[0]))

    def condensation(self) -> tuple["TransferGraph", tuple[tuple[int, ...], ...]]:
        components = self.strongly_connected_components()
        component_of = {
            vertex: component_index
            for component_index, component in enumerate(components)
            for vertex in component
        }
        adjacency = [[False] * len(components) for _ in components]
        for source in range(self.order):
            for target in self.successors(source):
                left = component_of[source]
                right = component_of[target]
                if left != right:
                    adjacency[left][right] = True
        names = ["{" + ",".join(self.names[index] for index in component) + "}" for component in components]
        return TransferGraph(adjacency, names=names), components

    def source_components(self) -> tuple[tuple[int, ...], ...]:
        condensation, components = self.condensation()
        return tuple(
            components[index]
            for index in range(condensation.order)
            if not condensation.predecessors(index)
        )

    def is_acyclic(self) -> bool:
        return all(len(component) == 1 and not self.adjacency[component[0]][component[0]] for component in self.strongly_connected_components())

    def topological_order(self) -> tuple[int, ...]:
        indegree = [len(self.predecessors(vertex)) for vertex in range(self.order)]
        queue = deque(vertex for vertex, degree in enumerate(indegree) if degree == 0)
        order: list[int] = []
        while queue:
            vertex = queue.popleft()
            order.append(vertex)
            for target in self.successors(vertex):
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        if len(order) != self.order:
            raise ValidationError("topological order is defined only for acyclic graphs")
        return tuple(order)

    def longest_path_vertices(self) -> int:
        order = self.topological_order()
        length = [1] * self.order
        for source in order:
            for target in self.successors(source):
                length[target] = max(length[target], length[source] + 1)
        return max(length, default=0)

    def source_component_certificate(self) -> ConditionalResult[int]:
        count = len(self.source_components())
        certificate = Certificate(
            valid=True,
            exactness=Exactness.CONDITIONAL,
            assumptions_checked=("finite transfer graph", "source strongly connected components"),
            assumptions_unchecked=(
                "edge maps are surjective on the declared packet spaces",
                "each source component is generated by one packet pure seed",
            ),
            diagnostics={"source_component_count": count},
            theorem_id="PSC-I-T44",
            message="source component count is a seed rank only under the listed propagation assumptions",
        )
        return ConditionalResult(count, certificate)

    def _check_vertex(self, vertex: int) -> None:
        if vertex < 0 or vertex >= self.order:
            raise IndexError(vertex)

    def edges(self) -> Iterator[tuple[int, int]]:
        for source in range(self.order):
            for target in self.successors(source):
                yield source, target


@dataclass(frozen=True, slots=True)
class SupportPreorder:
    """A reflexive transitive support relation on packet labels."""

    relation: tuple[tuple[bool, ...], ...]

    def __init__(self, relation: Sequence[Sequence[bool | int]]) -> None:
        rows = tuple(tuple(bool(value) for value in row) for row in relation)
        if not rows or any(len(row) != len(rows) for row in rows):
            raise ValidationError("a support relation must be nonempty and square")
        object.__setattr__(self, "relation", rows)
        if not self.is_reflexive() or not self.is_transitive():
            raise ValidationError("a support preorder must be reflexive and transitive")

    @property
    def order(self) -> int:
        return len(self.relation)

    def is_reflexive(self) -> bool:
        return all(self.relation[index][index] for index in range(self.order))

    def is_transitive(self) -> bool:
        for left in range(self.order):
            for middle in range(self.order):
                if not self.relation[left][middle]:
                    continue
                for right in range(self.order):
                    if self.relation[middle][right] and not self.relation[left][right]:
                        return False
        return True

    def equivalence_classes(self) -> tuple[tuple[int, ...], ...]:
        unseen = set(range(self.order))
        classes: list[tuple[int, ...]] = []
        while unseen:
            seed = min(unseen)
            equivalent = tuple(
                index
                for index in range(self.order)
                if self.relation[seed][index] and self.relation[index][seed]
            )
            classes.append(equivalent)
            unseen.difference_update(equivalent)
        return tuple(classes)

    def strict_class_graph(self) -> TransferGraph:
        classes = self.equivalence_classes()
        class_of = {
            vertex: class_index
            for class_index, members in enumerate(classes)
            for vertex in members
        }
        adjacency = [[False] * len(classes) for _ in classes]
        for source in range(self.order):
            for target in range(self.order):
                left = class_of[source]
                right = class_of[target]
                if left != right and self.relation[source][target]:
                    adjacency[left][right] = True
        return TransferGraph(adjacency)

    def strict_ideal_nilpotency_bound(self) -> int:
        """Return the class chain bound ``I^c = 0`` for strict transfers."""
        return self.strict_class_graph().longest_path_vertices()
