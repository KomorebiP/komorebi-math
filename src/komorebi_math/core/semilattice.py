"""Exact finite join congruences adapted from Stratified Universality.

This module computes finite tables and finite derivation witnesses. It does not
certify the manuscript's universal matroid, colimit or doctrine statements.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

from .._number_theory import require_integer
from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> bool:
        left, right = self.find(left), self.find(right)
        if left == right:
            return False
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1
        return True

    def partition(self) -> tuple[int, ...]:
        minima: dict[int, int] = {}
        for value in range(len(self.parent)):
            root = self.find(value)
            minima[root] = min(value, minima.get(root, value))
        return tuple(minima[self.find(value)] for value in range(len(self.parent)))


@dataclass(frozen=True, slots=True)
class FiniteJoinSemilattice:
    """A nonempty finite join table on labels ``0,...,size-1``.

    Validation uses O(size^3) exact comparisons and O(size^2) storage.
    """

    join_table: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        try:
            rows = tuple(tuple(row) for row in self.join_table)
        except TypeError as exc:
            raise DimensionError("join table must be a nonempty square table") from exc
        size = len(rows)
        if not size or any(len(row) != size for row in rows):
            raise DimensionError("join table must be a nonempty square table")
        table = tuple(tuple(require_integer(value, name="join entry") for value in row) for row in rows)
        if any(value < 0 or value >= size for row in table for value in row):
            raise ValidationError("join entries must be valid element labels")
        for left in range(size):
            if table[left][left] != left:
                raise ValidationError("join table must be idempotent")
            for right in range(size):
                if table[left][right] != table[right][left]:
                    raise ValidationError("join table must be commutative")
                for third in range(size):
                    if table[table[left][right]][third] != table[left][table[right][third]]:
                        raise ValidationError("join table must be associative")
        object.__setattr__(self, "join_table", table)

    @property
    def size(self) -> int:
        return len(self.join_table)

    def _element(self, value: Any) -> int:
        label = require_integer(value, name="element")
        if not 0 <= label < self.size:
            raise ValidationError("element label lies outside the join table")
        return label

    def join(self, left: int, right: int) -> int:
        return self.join_table[self._element(left)][self._element(right)]

    def generated_congruence(self, relations: Iterable[tuple[int, int]]) -> "SemilatticeQuotient":
        """Compute the least join congruence containing the supplied relations.

        Each successful merge stores either an input relation or a join context
        of an earlier merge. The result certificate independently checks that
        finite derivation and the quotient homomorphism.
        """

        normalized: list[tuple[int, int]] = []
        for relation in relations:
            try:
                pair = tuple(relation)
            except TypeError as exc:
                raise DimensionError("relations must contain pairs of element labels") from exc
            if len(pair) != 2:
                raise DimensionError("relations must contain pairs of element labels")
            normalized.append((self._element(pair[0]), self._element(pair[1])))
        requested = tuple(normalized)
        queue = deque((left, right, -1, index) for index, (left, right) in enumerate(requested))
        union = _UnionFind(self.size)
        witnesses: list[tuple[int, int, int, int]] = []
        while queue:
            left, right, parent, context = queue.popleft()
            if not union.union(left, right):
                continue
            witness_index = len(witnesses)
            witnesses.append((left, right, parent, context))
            for element in range(self.size):
                queue.append((self.join_table[left][element], self.join_table[right][element], witness_index, element))
        result = SemilatticeQuotient(self, union.partition(), requested, tuple(witnesses))
        result.certificate().require()
        return result

    def quotient(self, partition: Iterable[int]) -> "SemilatticeQuotient":
        """Validate and construct the quotient of a supplied equivalence partition."""

        labels = tuple(require_integer(value, name="partition label") for value in partition)
        if len(labels) != self.size:
            raise DimensionError("partition must contain one label per source element")
        minima: dict[int, int] = {}
        for element, label in enumerate(labels):
            minima[label] = min(element, minima.get(label, element))
        canonical = tuple(minima[label] for label in labels)
        result = SemilatticeQuotient(self, canonical)
        result.certificate().require()
        return result


@dataclass(frozen=True, slots=True)
class SemilatticeQuotient:
    """A canonical finite quotient and optional derivation of its least congruence."""

    source: FiniteJoinSemilattice
    partition: tuple[int, ...]
    relations: tuple[tuple[int, int], ...] = ()
    derivations: tuple[tuple[int, int, int, int], ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, FiniteJoinSemilattice):
            raise TypeError("source must be a FiniteJoinSemilattice")
        object.__setattr__(self, "partition", tuple(require_integer(value, name="partition label") for value in self.partition))
        object.__setattr__(self, "relations", tuple(tuple(require_integer(value, name="relation label") for value in pair)
                                                  for pair in self.relations))
        if self.derivations is not None:
            object.__setattr__(self, "derivations", tuple(tuple(require_integer(value, name="derivation entry") for value in row)
                                                        for row in self.derivations))

    @property
    def classes(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(element for element, label in enumerate(self.partition) if label == root)
                     for root in sorted(set(self.partition)))

    @property
    def projection(self) -> tuple[int, ...]:
        indices = {root: index for index, root in enumerate(sorted(set(self.partition)))}
        return tuple(indices[label] for label in self.partition)

    @property
    def quotient(self) -> FiniteJoinSemilattice:
        self.certificate().require()
        projection = self.projection
        representatives = tuple(group[0] for group in self.classes)
        return FiniteJoinSemilattice(tuple(tuple(projection[self.source.join_table[left][right]]
                                                for right in representatives) for left in representatives))

    def certificate(self) -> Certificate:
        """Check the exact table homomorphism and, when present, leastness witness."""

        size = self.source.size
        partition_valid = (
            len(self.partition) == size
            and all(isinstance(label, int) and not isinstance(label, bool) and 0 <= label < size
                    for label in self.partition)
            and all(self.partition[label] == label for label in self.partition)
            and all(label <= element for element, label in enumerate(self.partition))
        )
        homomorphism = False
        requested = False
        derivation_valid = self.derivations is None
        if partition_valid:
            projection = self.projection
            representatives = tuple(group[0] for group in self.classes)
            qtable = tuple(tuple(projection[self.source.join_table[left][right]] for right in representatives)
                           for left in representatives)
            homomorphism = all(
                projection[self.source.join_table[left][right]] == qtable[projection[left]][projection[right]]
                for left in range(size) for right in range(size)
            )
            requested = all(
                len(pair) == 2 and all(isinstance(value, int) and not isinstance(value, bool) and 0 <= value < size
                                      for value in pair) and self.partition[pair[0]] == self.partition[pair[1]]
                for pair in self.relations
            )
            if self.derivations is not None:
                replay = _UnionFind(size)
                derivation_valid = True
                for index, witness in enumerate(self.derivations):
                    if len(witness) != 4 or any(not isinstance(value, int) or isinstance(value, bool) for value in witness):
                        derivation_valid = False
                        break
                    left, right, parent, context = witness
                    if not (0 <= left < size and 0 <= right < size):
                        derivation_valid = False
                        break
                    if parent == -1:
                        expected = self.relations[context] if 0 <= context < len(self.relations) else None
                    elif 0 <= parent < index and 0 <= context < size:
                        earlier = self.derivations[parent]
                        expected = (self.source.join_table[earlier[0]][context], self.source.join_table[earlier[1]][context])
                    else:
                        expected = None
                    if expected != (left, right):
                        derivation_valid = False
                        break
                    replay.union(left, right)
                derivation_valid = derivation_valid and replay.partition() == self.partition
        valid = partition_valid and homomorphism and requested and derivation_valid
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT,
            residuals={"finite_quotient_axioms": 0.0 if valid else 1.0},
            assumptions_checked=tuple(name for name, checked in (
                ("validated finite join-semilattice", True),
                ("canonical equivalence partition", partition_valid),
                ("quotient join homomorphism", homomorphism),
                ("requested relations identified", requested),
                ("finite least-congruence derivation", derivation_valid and self.derivations is not None),
            ) if checked),
            diagnostics={
                "source_size": size,
                "quotient_size": len(set(self.partition)) if partition_valid else None,
                "input_relations": len(self.relations),
                "merge_witnesses": len(self.derivations) if self.derivations is not None else None,
                "source": "Stratified Universality finite join-congruence algorithm",
                "scope": "finite table quotient; no universal matroid or doctrine theorem",
            },
            theorem_id="STRATIFIED.B26.FINITE.JOIN.CONGRUENCE",
            message="finite join quotient verified" if valid else "finite join quotient verification failed",
        )


__all__ = ["FiniteJoinSemilattice", "SemilatticeQuotient"]
