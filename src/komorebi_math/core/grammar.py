"""Typed operator grammars and formal path relations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..exceptions import RelationError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented


@dataclass(frozen=True, slots=True)
class Arrow:
    name: str
    source: str
    target: str


@dataclass(frozen=True, slots=True)
class OperatorWord:
    """A typed path written in the order in which arrows act."""

    arrows: tuple[str, ...]
    source: str
    target: str

    @property
    def length(self) -> int:
        return len(self.arrows)

    @property
    def is_identity(self) -> bool:
        return not self.arrows and self.source == self.target

    def __str__(self) -> str:
        return f"id_{self.source}" if self.is_identity else " · ".join(self.arrows)


@dataclass(frozen=True, slots=True)
class OperatorRelation:
    """A linear relation among parallel operator words."""

    name: str
    terms: tuple[tuple[Any, OperatorWord], ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise RelationError("a relation requires a name")
        if not self.terms:
            raise RelationError("a relation requires at least one term")
        source = self.terms[0][1].source
        target = self.terms[0][1].target
        if any(word.source != source or word.target != target for _, word in self.terms):
            raise RelationError("all words in a relation must have the same source and target")

    @property
    def source(self) -> str:
        return self.terms[0][1].source

    @property
    def target(self) -> str:
        return self.terms[0][1].target


@documented(
    ProvenanceRecord(
        identifier="PSC-I-D01",
        title="Typed operator grammar",
        status=ClaimStatus.DEFINITION,
        source="Paper I",
        assumptions=("finite vertex set", "typed arrows"),
        limitations=("relations are finite linear combinations of typed paths",),
        verification_suite="core.grammar",
    )
)
class OperatorGrammar:
    """A finite quiver with optional linear path relations.

    Paths are stored in application order. If ``a`` maps ``X`` to ``Y`` and
    ``b`` maps ``Y`` to ``Z``, then ``grammar.word("a", "b")`` evaluates as
    the matrix product ``B @ A``.
    """

    def __init__(
        self,
        vertices: Iterable[str],
        arrows: Mapping[str, tuple[str, str]] | Iterable[Arrow],
        relations: Iterable[OperatorRelation] = (),
    ) -> None:
        vertex_tuple = tuple(dict.fromkeys(str(vertex) for vertex in vertices))
        if not vertex_tuple:
            raise ValidationError("an operator grammar requires at least one vertex")
        if any(not vertex for vertex in vertex_tuple):
            raise ValidationError("vertex names must be nonempty")
        self._vertices = vertex_tuple

        if isinstance(arrows, Mapping):
            arrow_values = tuple(
                Arrow(str(name), str(pair[0]), str(pair[1])) for name, pair in arrows.items()
            )
        else:
            arrow_values = tuple(arrows)
        names = [arrow.name for arrow in arrow_values]
        if len(set(names)) != len(names):
            raise ValidationError("arrow names must be unique")
        vertex_set = set(vertex_tuple)
        for arrow in arrow_values:
            if not arrow.name:
                raise ValidationError("arrow names must be nonempty")
            if arrow.source not in vertex_set or arrow.target not in vertex_set:
                raise ValidationError(f"arrow {arrow.name!r} uses an unknown vertex")
        self._arrows = {arrow.name: arrow for arrow in arrow_values}

        self._relations: dict[str, OperatorRelation] = {}
        for relation in relations:
            self.add_relation(relation)

    @property
    def vertices(self) -> tuple[str, ...]:
        return self._vertices

    @property
    def arrows(self) -> tuple[Arrow, ...]:
        return tuple(self._arrows.values())

    @property
    def relations(self) -> tuple[OperatorRelation, ...]:
        return tuple(self._relations.values())

    def arrow(self, name: str) -> Arrow:
        try:
            return self._arrows[name]
        except KeyError as exc:
            raise ValidationError(f"unknown arrow: {name}") from exc

    def identity(self, vertex: str) -> OperatorWord:
        if vertex not in self._vertices:
            raise ValidationError(f"unknown vertex: {vertex}")
        return OperatorWord((), vertex, vertex)

    def word(self, *arrow_names: str, vertex: str | None = None) -> OperatorWord:
        if not arrow_names:
            if vertex is None:
                raise ValidationError("an empty word requires its identity vertex")
            return self.identity(vertex)
        arrows = [self.arrow(name) for name in arrow_names]
        for left, right in zip(arrows, arrows[1:], strict=False):
            if left.target != right.source:
                raise ValidationError(
                    f"path is not composable at {left.name!r}, {right.name!r}: "
                    f"{left.target!r} != {right.source!r}"
                )
        return OperatorWord(tuple(arrow_names), arrows[0].source, arrows[-1].target)

    def relation(
        self,
        name: str,
        terms: Sequence[tuple[Any, OperatorWord]],
        *,
        add: bool = True,
    ) -> OperatorRelation:
        relation = OperatorRelation(name, tuple(terms))
        if add:
            self.add_relation(relation)
        return relation

    def add_relation(self, relation: OperatorRelation) -> None:
        if relation.name in self._relations:
            raise RelationError(f"relation name already exists: {relation.name}")
        for _, word in relation.terms:
            if word.is_identity:
                if word.source not in self._vertices:
                    raise RelationError("relation contains an unknown identity vertex")
                continue
            reconstructed = self.word(*word.arrows)
            if reconstructed.source != word.source or reconstructed.target != word.target:
                raise RelationError("relation contains a word with inconsistent endpoints")
        self._relations[relation.name] = relation

    def paths(
        self,
        *,
        source: str | None = None,
        target: str | None = None,
        max_length: int = 1,
        include_identities: bool = True,
    ) -> tuple[OperatorWord, ...]:
        if max_length < 0:
            raise ValueError("max_length must be nonnegative")
        for vertex in (source, target):
            if vertex is not None and vertex not in self._vertices:
                raise ValidationError(f"unknown vertex: {vertex}")

        found: list[OperatorWord] = []
        frontier: list[OperatorWord] = []
        if include_identities:
            for vertex in self._vertices:
                identity = self.identity(vertex)
                if (source is None or identity.source == source) and (
                    target is None or identity.target == target
                ):
                    found.append(identity)
        if max_length == 0:
            return tuple(found)

        for arrow in self.arrows:
            word = self.word(arrow.name)
            frontier.append(word)
            if (source is None or word.source == source) and (target is None or word.target == target):
                found.append(word)

        for _ in range(2, max_length + 1):
            next_frontier: list[OperatorWord] = []
            for word in frontier:
                for arrow in self.arrows:
                    if word.target != arrow.source:
                        continue
                    extended = self.word(*word.arrows, arrow.name)
                    next_frontier.append(extended)
                    if (source is None or extended.source == source) and (
                        target is None or extended.target == target
                    ):
                        found.append(extended)
            frontier = next_frontier
        return tuple(found)

    def __repr__(self) -> str:
        return (
            f"OperatorGrammar(vertices={self.vertices!r}, arrows={tuple(self._arrows)!r}, "
            f"relations={tuple(self._relations)!r})"
        )
