"""Finite residual lift forests and stable inverse-system cores."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Generic, Hashable, Iterable, Mapping, TypeVar

from ..certificates import Certificate, Exactness
from ..exceptions import ValidationError
from .._number_theory import require_integer

Node = TypeVar("Node", bound=Hashable)


@dataclass(frozen=True, slots=True)
class LiftBranch(Generic[Node]):
    """One compatible branch through a finite lift forest."""

    nodes: tuple[Node, ...]

    @property
    def depth(self) -> int:
        """Return the deepest represented level."""

        return len(self.nodes)


@dataclass(frozen=True, slots=True)
class LiftForestReport(Generic[Node]):
    """Validation and branch data for a finite lift forest."""

    levels: tuple[tuple[Node, ...], ...]
    stable_cores: tuple[tuple[Node, ...], ...]
    branches: tuple[LiftBranch[Node], ...]
    certificate: Certificate


class FiniteLiftForest(Generic[Node]):
    """A finite inverse system of lift levels with explicit reduction maps.

    Levels are indexed from one.  The reduction map at depth ``m`` sends a
    node of level ``m`` to a node of level ``m - 1``.
    """

    def __init__(
        self,
        levels: Mapping[int, Iterable[Node]],
        reductions: Mapping[int, Callable[[Node], Node] | Mapping[Node, Node]],
    ) -> None:
        if not levels:
            raise ValidationError("a lift forest requires at least one level")
        normalized: dict[int, tuple[Node, ...]] = {}
        for depth, values in levels.items():
            d = require_integer(depth, name="depth")
            if d < 1:
                raise ValidationError("lift depths must be positive")
            nodes = tuple(dict.fromkeys(values))
            if not nodes:
                raise ValidationError(f"lift level {d} is empty")
            normalized[d] = nodes
        expected = set(range(1, max(normalized) + 1))
        if set(normalized) != expected:
            raise ValidationError("lift levels must be consecutive starting at one")
        self._levels = normalized
        self._reductions: dict[int, Callable[[Node], Node]] = {}
        for depth in range(2, self.max_depth + 1):
            if depth not in reductions:
                raise ValidationError(f"missing reduction map at depth {depth}")
            raw = reductions[depth]
            if callable(raw):
                function = raw
            else:
                table = dict(raw)
                function = table.__getitem__
            self._reductions[depth] = function
            previous = set(self._levels[depth - 1])
            for node in self._levels[depth]:
                try:
                    image = function(node)
                except Exception as exc:
                    raise ValidationError(f"reduction failed at depth {depth}") from exc
                if image not in previous:
                    raise ValidationError(
                        f"reduction at depth {depth} leaves the preceding level"
                    )

    @property
    def max_depth(self) -> int:
        """Return the deepest finite level."""

        return max(self._levels)

    def level(self, depth: int) -> tuple[Node, ...]:
        """Return one finite lift level."""

        d = require_integer(depth, name="depth")
        try:
            return self._levels[d]
        except KeyError as exc:
            raise ValidationError("depth is outside the finite lift forest") from exc

    def reduce(self, node: Node, *, from_depth: int, to_depth: int = 1) -> Node:
        """Reduce one node to a shallower level."""

        source = require_integer(from_depth, name="from_depth")
        target = require_integer(to_depth, name="to_depth")
        if not 1 <= target <= source <= self.max_depth:
            raise ValidationError("reduction depths are not ordered inside the forest")
        if node not in set(self.level(source)):
            raise ValidationError("node does not belong to the declared source level")
        current = node
        for depth in range(source, target, -1):
            current = self._reductions[depth](current)
        return current

    def children(self, node: Node, *, depth: int) -> tuple[Node, ...]:
        """Return the children of a node at the next level."""

        d = require_integer(depth, name="depth")
        if d < 1 or d >= self.max_depth:
            return ()
        if node not in set(self.level(d)):
            raise ValidationError("node does not belong to the declared level")
        reduction = self._reductions[d + 1]
        return tuple(child for child in self.level(d + 1) if reduction(child) == node)

    def image(self, source_depth: int, target_depth: int) -> frozenset[Node]:
        """Return the image of one level in a shallower level."""

        source = require_integer(source_depth, name="source_depth")
        target = require_integer(target_depth, name="target_depth")
        if not 1 <= target <= source <= self.max_depth:
            raise ValidationError("image depths are not ordered inside the forest")
        return frozenset(
            self.reduce(node, from_depth=source, to_depth=target)
            for node in self.level(source)
        )

    def stable_core(self, depth: int) -> tuple[Node, ...]:
        """Return nodes surviving to every deeper represented level."""

        d = require_integer(depth, name="depth")
        if not 1 <= d <= self.max_depth:
            raise ValidationError("depth is outside the finite lift forest")
        core = set(self.level(d))
        for source in range(d + 1, self.max_depth + 1):
            core.intersection_update(self.image(source, d))
        return tuple(node for node in self.level(d) if node in core)

    def branches(self, *, limit: int = 100000) -> tuple[LiftBranch[Node], ...]:
        """Enumerate all branches through the represented finite levels."""

        maximum = require_integer(limit, name="limit")
        if maximum < 1:
            raise ValidationError("branch limit must be positive")
        partial: list[tuple[Node, ...]] = [(root,) for root in self.level(1)]
        for depth in range(1, self.max_depth):
            extended: list[tuple[Node, ...]] = []
            for branch in partial:
                for child in self.children(branch[-1], depth=depth):
                    extended.append((*branch, child))
                    if len(extended) > maximum:
                        raise ValidationError(
                            f"branch count exceeds the safety limit {maximum}"
                        )
            partial = extended
            if not partial:
                break
        return tuple(LiftBranch(branch) for branch in partial)

    def report(self, *, branch_limit: int = 100000) -> LiftForestReport[Node]:
        """Return stable cores, complete finite branches, and a certificate."""

        cores = tuple(self.stable_core(depth) for depth in range(1, self.max_depth + 1))
        branches = self.branches(limit=branch_limit)
        valid = bool(branches) == all(bool(core) for core in cores)
        return LiftForestReport(
            levels=tuple(self.level(depth) for depth in range(1, self.max_depth + 1)),
            stable_cores=cores,
            branches=branches,
            certificate=Certificate(
                valid=valid,
                exactness=Exactness.EXACT,
                assumptions_checked=(
                    "finite consecutive levels",
                    "compatible reduction maps",
                    "finite stable core calculation",
                ),
                diagnostics={
                    "depth": self.max_depth,
                    "level_sizes": tuple(len(self.level(d)) for d in range(1, self.max_depth + 1)),
                    "stable_core_sizes": tuple(len(core) for core in cores),
                    "branch_count": len(branches),
                },
                theorem_id="PSC-X-LIFT-FOREST",
                message="finite lift forest is compatible" if valid else "stable core and branch data disagree",
            ),
        )


def residue_forest(
    values_by_depth: Mapping[int, Iterable[int]],
    *,
    prime: int,
) -> FiniteLiftForest[int]:
    """Build a finite prime power residue forest from explicit residues."""

    from .._number_theory import require_prime

    p = require_prime(prime)
    levels = {depth: tuple(value % (p**depth) for value in values) for depth, values in values_by_depth.items()}
    reductions = {
        depth: (lambda value, modulus=p ** (depth - 1): value % modulus)
        for depth in levels
        if depth > 1
    }
    return FiniteLiftForest(levels, reductions)


__all__ = [
    "FiniteLiftForest",
    "LiftBranch",
    "LiftForestReport",
    "residue_forest",
]
