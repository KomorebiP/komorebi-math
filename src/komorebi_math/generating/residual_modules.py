"""Bounded exact residual modules over ``Z/p**r Z``.

Cartier Compilers Chapter 24 works with modules, not vector spaces.  This
implementation enumerates the reachable module and its complete future-output
equivalence classes.  Quotient elements are class indices with explicit module
operations; no free basis or deterministic-automaton minimum is presumed.

Matrices act on column states.  Words are read left to right: appending digit
``d`` applies ``T_d`` after the previous word.  All computations use integers.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from math import isqrt
from types import MappingProxyType
from typing import Any

from .._number_theory import require_integer, require_prime
from ..exceptions import DimensionError, ResourceLimitError, ValidationError

State = tuple[int, ...]
Matrix = tuple[State, ...]


@dataclass(frozen=True, slots=True)
class ResidualModuleBudget:
    """Hard finite bounds, checked before the corresponding allocations.

    ``max_work`` bounds a conservative number of scalar/table operations, not
    elapsed time or bit complexity.  Ambient enumeration and the quadratic
    operation tables are deliberately suitable only for small finite models.
    """

    max_carrier_size: int = 4096
    max_scalar_count: int = 4096
    max_dimension: int = 32
    max_symbols: int = 16
    max_exponent: int = 64
    max_integer_bits: int = 256
    max_operation_entries: int = 1_000_000
    max_work: int = 10_000_000
    max_word_length: int = 100_000

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = require_integer(getattr(self, name), name=name)
            if value < 1:
                raise ValidationError(f"{name} must be positive")
            object.__setattr__(self, name, value)

    def require(self, value: int, maximum: int, name: str) -> None:
        if value > maximum:
            raise ResourceLimitError(f"{name} {value} exceeds budget {maximum}")


DEFAULT_RESIDUAL_BUDGET = ResidualModuleBudget()


def _bounded_power(base: int, exponent: int, cap: int, name: str) -> int:
    result = 1
    for _ in range(exponent):
        if result > cap // base:
            raise ResourceLimitError(f"{name} exceeds budget {cap}")
        result *= base
    return result


def _tuple_integers(values: Iterable[int], cap: int, budget: ResidualModuleBudget, name: str) -> State:
    result = []
    for value in values:
        budget.require(len(result) + 1, cap, f"{name} length")
        integer = require_integer(value, name=name)
        budget.require(abs(integer).bit_length(), budget.max_integer_bits, f"{name} integer bits")
        result.append(integer)
    return tuple(result)


def _state_add(left: State, right: State, modulus: int) -> State:
    return tuple((x + y) % modulus for x, y in zip(left, right, strict=True))


def _state_scale(scalar: int, state: State, modulus: int) -> State:
    return tuple(scalar * x % modulus for x in state)


def _log_prime_cardinality(cardinality: int, prime: int) -> int:
    result = 0
    while cardinality > 1 and cardinality % prime == 0:
        cardinality //= prime
        result += 1
    if cardinality != 1:
        raise ArithmeticError("carrier cardinality is not a power of the stated prime")
    return result


@dataclass(frozen=True, slots=True, init=False)
class RingLinearRealization:
    """An exact free ambient realization over one fixed prime-power ring.

    The ambient carrier is ``(Z/p**r Z)**n``.  Empty dimensions and an empty
    alphabet are allowed.  Its observable quotient can be nonfree.
    """

    prime: int
    exponent: int
    modulus: int
    matrices: tuple[Matrix, ...]
    seed: State
    output_row: State
    dimension: int
    carrier_cardinality: int
    budget: ResidualModuleBudget

    def __init__(
        self,
        prime: int,
        exponent: int,
        matrices: Iterable[Iterable[Iterable[int]]],
        seed: Iterable[int],
        output: Iterable[int],
        *,
        budget: ResidualModuleBudget = DEFAULT_RESIDUAL_BUDGET,
    ) -> None:
        if not isinstance(budget, ResidualModuleBudget):
            raise TypeError("budget must be a ResidualModuleBudget")
        p = require_integer(prime, name="prime")
        budget.require(abs(p).bit_length(), budget.max_integer_bits, "prime bits")
        budget.require(p, budget.max_scalar_count, "prime")
        if p < 2:
            raise ValidationError("prime must be a positive prime integer")
        budget.require(isqrt(p), budget.max_work, "primality trial bound")
        p = require_prime(p)
        r = require_integer(exponent, name="exponent")
        if r < 1:
            raise ValidationError("exponent must be positive")
        budget.require(r, budget.max_exponent, "exponent")
        modulus = _bounded_power(p, r, budget.max_scalar_count, "scalar count")
        v = _tuple_integers(seed, budget.max_dimension, budget, "seed")
        dimension = len(v)
        cardinality = _bounded_power(modulus, dimension, budget.max_carrier_size, "ambient cardinality")
        o = _tuple_integers(output, dimension, budget, "output")
        if len(o) != dimension:
            raise DimensionError("output and seed must have equal dimension")
        operators = []
        for raw_matrix in matrices:
            budget.require(len(operators) + 1, budget.max_symbols, "alphabet size")
            rows = []
            for raw_row in raw_matrix:
                budget.require(len(rows) + 1, dimension, "matrix row count")
                row = _tuple_integers(raw_row, dimension, budget, "matrix row")
                if len(row) != dimension:
                    raise DimensionError("transition matrix must be square of seed dimension")
                rows.append(tuple(x % modulus for x in row))
            if len(rows) != dimension:
                raise DimensionError("transition matrix must be square of seed dimension")
            operators.append(tuple(rows))
        for name, value in (
            ("prime", p),
            ("exponent", r),
            ("modulus", modulus),
            ("matrices", tuple(operators)),
            ("seed", tuple(x % modulus for x in v)),
            ("output_row", tuple(x % modulus for x in o)),
            ("dimension", dimension),
            ("carrier_cardinality", cardinality),
            ("budget", budget),
        ):
            object.__setattr__(self, name, value)

    @property
    def alphabet(self) -> tuple[int, ...]:
        return tuple(range(len(self.matrices)))

    @property
    def zero(self) -> State:
        return (0,) * self.dimension

    def normalize_state(self, state: Iterable[int]) -> State:
        values = _tuple_integers(state, self.dimension, self.budget, "state")
        if len(values) != self.dimension:
            raise DimensionError("state has wrong dimension")
        return tuple(x % self.modulus for x in values)

    def transition(self, state: Iterable[int], digit: int) -> State:
        values = self.normalize_state(state)
        d = require_integer(digit, name="digit")
        if d < 0 or d >= len(self.matrices):
            raise ValidationError("digit is outside the finite alphabet")
        return self._transition(values, d)

    def _transition(self, state: State, digit: int) -> State:
        return tuple(
            sum(a * x for a, x in zip(row, state, strict=True)) % self.modulus for row in self.matrices[digit]
        )

    def output(self, state: Iterable[int]) -> int:
        values = self.normalize_state(state)
        return self._output(values)

    def _output(self, state: State) -> int:
        return sum(a * x for a, x in zip(self.output_row, state, strict=True)) % self.modulus

    def evaluate(self, word: Iterable[int]) -> int:
        state = self.seed
        for length, digit in enumerate(word, 1):
            self.budget.require(length, self.budget.max_word_length, "word length")
            state = self.transition(state, digit)
        return self._output(state)


@dataclass(frozen=True, slots=True)
class ModuleLawReport:
    valid: bool
    checks: tuple[tuple[str, bool], ...]
    checked_operations_bound: int
    cardinality: int
    modulus: int


@dataclass(frozen=True, slots=True, init=False)
class CanonicalResidualModule:
    """The actual finite reachable/unobservable quotient, with operation tables.

    ``classes`` retains every reachable ambient vector, so ``project`` describes
    the quotient map explicitly.  The kernel is ``unobservable_states``.
    Class zero is always index 0; other indices are deterministic local labels,
    not a claimed canonical basis or coordinate identification across systems.
    """

    source: RingLinearRealization
    prime: int
    exponent: int
    modulus: int
    ambient_cardinality: int
    orbit_states: tuple[State, ...]
    reachable_states: tuple[State, ...]
    unobservable_states: tuple[State, ...]
    classes: tuple[tuple[State, ...], ...]
    seed: int
    addition: tuple[tuple[int, ...], ...]
    scalars: tuple[tuple[int, ...], ...]
    actions: tuple[tuple[int, ...], ...]
    outputs: tuple[int, ...]
    budget: ResidualModuleBudget
    _projection: Any = field(repr=False, compare=False)

    @property
    def carrier_cardinality(self) -> int:
        return len(self.classes)

    @property
    def alphabet(self) -> tuple[int, ...]:
        return self.source.alphabet

    @property
    def zero(self) -> int:
        return 0

    def _element(self, value: int) -> int:
        index = require_integer(value, name="quotient element")
        if index < 0 or index >= self.carrier_cardinality:
            raise ValidationError("quotient element is outside the carrier")
        return index

    def project(self, source_state: Iterable[int]) -> int:
        state = self.source.normalize_state(source_state)
        if state not in self._projection:
            raise ValidationError("quotient map is defined only on the reachable submodule")
        return self._projection[state]

    def add(self, left: int, right: int) -> int:
        return self.addition[self._element(left)][self._element(right)]

    def scale(self, scalar: int, element: int) -> int:
        value = require_integer(scalar, name="scalar")
        self.budget.require(abs(value).bit_length(), self.budget.max_integer_bits, "scalar bits")
        return self.scalars[value % self.modulus][self._element(element)]

    def transition(self, element: int, digit: int) -> int:
        d = require_integer(digit, name="digit")
        if d < 0 or d >= len(self.actions):
            raise ValidationError("digit is outside the finite alphabet")
        return self._transition(self._element(element), d)

    def _transition(self, element: int, digit: int) -> int:
        return self.actions[digit][element]

    def output(self, element: int) -> int:
        return self._output(self._element(element))

    def _output(self, element: int) -> int:
        return self.outputs[element]

    def evaluate(self, word: Iterable[int]) -> int:
        state = self.seed
        for length, digit in enumerate(word, 1):
            self.budget.require(length, self.budget.max_word_length, "word length")
            state = self.transition(state, digit)
        return self._output(state)

    def additive_p_exponents(self) -> tuple[int, ...]:
        """Return the cyclic factors ``Z/p**e Z`` of this finite module.

        Exact p-torsion cardinalities determine the elementary divisors.  In
        particular ``(1,)`` over Z/4 is nonfree, whereas ``(2,)`` is free rank 1.
        The zero module returns the empty tuple.
        """
        previous = 0
        widths = []
        scalar = 1
        self.budget.require(self.exponent * self.carrier_cardinality, self.budget.max_work, "p-torsion work")
        for _ in range(self.exponent):
            scalar *= self.prime
            torsion = sum(
                self.scalars[scalar % self.modulus][x] == 0 for x in range(self.carrier_cardinality)
            )
            current = _log_prime_cardinality(torsion, self.prime)
            widths.append(current - previous)
            previous = current
        widths.append(0)
        if any(widths[j] < widths[j + 1] for j in range(self.exponent)):
            raise ArithmeticError("p-torsion cardinalities violate the finite module law")
        return tuple(e for e in range(1, self.exponent + 1) for _ in range(widths[e - 1] - widths[e]))

    def verify_laws(self, *, budget: ResidualModuleBudget | None = None) -> ModuleLawReport:
        """Exhaustively check additive, scalar, digit-action and output laws.

        Cubic associativity and scalar-ring laws have their own explicit work
        preflight.  A resource refusal never changes a failure into a pass.
        """
        limits = self.budget if budget is None else budget
        if not isinstance(limits, ResidualModuleBudget):
            raise TypeError("budget must be a ResidualModuleBudget")
        n, m, k = self.carrier_cardinality, self.modulus, len(self.actions)
        work = n**3 + 2 * m * m * n + (m + k + 2) * n * n + (k + 1) * m * n + 5 * n
        limits.require(work, limits.max_work, "module-law work")
        add, scale = self.addition, self.scalars
        checks = (
            ("additive_identity", all(add[0][x] == x == add[x][0] for x in range(n))),
            ("additive_inverse", all(add[x][scale[m - 1][x]] == 0 for x in range(n))),
            ("additive_commutativity", all(add[x][y] == add[y][x] for x in range(n) for y in range(n))),
            (
                "additive_associativity",
                all(
                    add[add[x][y]][z] == add[x][add[y][z]]
                    for x in range(n)
                    for y in range(n)
                    for z in range(n)
                ),
            ),
            ("scalar_zero_and_one", all(scale[0][x] == 0 and scale[1][x] == x for x in range(n))),
            (
                "scalar_addition",
                all(
                    scale[(a + b) % m][x] == add[scale[a][x]][scale[b][x]]
                    for a in range(m)
                    for b in range(m)
                    for x in range(n)
                ),
            ),
            (
                "scalar_multiplication",
                all(
                    scale[a * b % m][x] == scale[a][scale[b][x]]
                    for a in range(m)
                    for b in range(m)
                    for x in range(n)
                ),
            ),
            (
                "scalar_distributivity",
                all(
                    scale[a][add[x][y]] == add[scale[a][x]][scale[a][y]]
                    for a in range(m)
                    for x in range(n)
                    for y in range(n)
                ),
            ),
            (
                "digit_additivity",
                all(
                    action[add[x][y]] == add[action[x]][action[y]]
                    for action in self.actions
                    for x in range(n)
                    for y in range(n)
                ),
            ),
            (
                "digit_scalar_linearity",
                all(
                    action[scale[a][x]] == scale[a][action[x]]
                    for action in self.actions
                    for a in range(m)
                    for x in range(n)
                ),
            ),
            (
                "output_additivity",
                all(
                    self.outputs[add[x][y]] == (self.outputs[x] + self.outputs[y]) % m
                    for x in range(n)
                    for y in range(n)
                ),
            ),
            (
                "output_scalar_linearity",
                all(self.outputs[scale[a][x]] == a * self.outputs[x] % m for a in range(m) for x in range(n)),
            ),
        )
        return ModuleLawReport(all(ok for _, ok in checks), checks, work, n, m)


def canonical_residual_module(
    realization: RingLinearRealization, *, budget: ResidualModuleBudget | None = None
) -> CanonicalResidualModule:
    """Compute ``V_reach/(V_reach intersect N)`` by complete finite semantics.

    First enumerate seed iterates and their ring span.  Stable output partition
    refinement then identifies states with equal outputs after *every* word.
    Addition, scalar multiplication and digit actions descend to explicit
    quotient tables.  Theorem 24.1 identifies this quotient with prefix
    residuals; the implementation does not choose a free basis.
    """
    if not isinstance(realization, RingLinearRealization):
        raise TypeError("realization must be a RingLinearRealization")
    limits = realization.budget if budget is None else budget
    if not isinstance(limits, ResidualModuleBudget):
        raise TypeError("budget must be a ResidualModuleBudget")
    n, m, k, dimension = (
        realization.carrier_cardinality,
        realization.modulus,
        len(realization.matrices),
        realization.dimension,
    )
    limits.require(n, limits.max_carrier_size, "ambient cardinality")
    limits.require(m, limits.max_scalar_count, "scalar count")
    composition_length = realization.exponent * dimension
    # At most composition_length subgroup extensions, and at most n strict
    # partition refinements.  These bounds precede orbit/module allocations.
    work = (
        n * m * max(1, composition_length) * max(1, dimension)
        + n * k * max(1, dimension * dimension)
        + n * n * (k + 1)
        + n * dimension
    )
    limits.require(work, limits.max_work, "residual-module construction work")
    orbit = {realization.seed}
    pending = deque([realization.seed])
    while pending:
        state = pending.popleft()
        for digit in realization.alphabet:
            target = realization._transition(state, digit)
            if target not in orbit:
                orbit.add(target)
                pending.append(target)
    reachable = {realization.zero}
    for generator in sorted(orbit):
        if generator in reachable:
            continue
        old = tuple(reachable)
        multiples = tuple(_state_scale(a, generator, m) for a in range(m))
        reachable.update(_state_add(x, y, m) for x in old for y in multiples)
    states = tuple(sorted(reachable))
    indices = {state: index for index, state in enumerate(states)}
    transitions = tuple(
        tuple(indices[realization._transition(state, digit)] for state in states)
        for digit in realization.alphabet
    )
    values = tuple(realization._output(state) for state in states)
    labels = tuple(values)
    while True:
        signatures = tuple(
            (values[i], *(labels[action[i]] for action in transitions)) for i in range(len(states))
        )
        distinct = {signature: index for index, signature in enumerate(sorted(set(signatures)))}
        refined = tuple(distinct[signature] for signature in signatures)
        # Compare equivalence relations, not arbitrary numeric class labels.
        old_to_new: dict[int, int] = {}
        stable = True
        for old_label, new_label in zip(labels, refined, strict=True):
            if old_label in old_to_new and old_to_new[old_label] != new_label:
                stable = False
            old_to_new[old_label] = new_label
        labels = refined
        if stable:
            break
    groups: dict[int, list[State]] = {}
    for state, label in zip(states, labels, strict=True):
        groups.setdefault(label, []).append(state)
    classes = tuple(sorted((tuple(group) for group in groups.values()), key=lambda group: group[0]))
    projection = {state: index for index, group in enumerate(classes) for state in group}
    representatives = tuple(group[0] for group in classes)
    q = len(classes)
    entries = q * q + (m + k + 1) * q
    limits.require(entries, limits.max_operation_entries, "quotient operation entries")
    limits.require(entries * max(1, dimension), limits.max_work, "quotient table work")
    addition = tuple(tuple(projection[_state_add(x, y, m)] for y in representatives) for x in representatives)
    scalars = tuple(tuple(projection[_state_scale(a, x, m)] for x in representatives) for a in range(m))
    actions = tuple(
        tuple(projection[realization._transition(x, d)] for x in representatives)
        for d in realization.alphabet
    )
    result = object.__new__(CanonicalResidualModule)
    for name, value in (
        ("source", realization),
        ("prime", realization.prime),
        ("exponent", realization.exponent),
        ("modulus", m),
        ("ambient_cardinality", n),
        ("orbit_states", tuple(sorted(orbit))),
        ("reachable_states", states),
        ("unobservable_states", classes[0]),
        ("classes", classes),
        ("seed", projection[realization.seed]),
        ("addition", addition),
        ("scalars", scalars),
        ("actions", actions),
        ("outputs", tuple(realization._output(x) for x in representatives)),
        ("budget", limits),
        ("_projection", MappingProxyType(projection)),
    ):
        object.__setattr__(result, name, value)
    return result


@dataclass(frozen=True, slots=True)
class WordComparison:
    equivalent: bool
    witness: tuple[int, ...] | None
    left_output: int | None
    right_output: int | None
    length_horizon: int
    visited_pairs: int
    algorithm: str = "complete product-orbit BFS"


def finite_word_comparison(
    left: RingLinearRealization | CanonicalResidualModule,
    right: RingLinearRealization | CanonicalResidualModule,
    *,
    budget: ResidualModuleBudget | None = None,
) -> WordComparison:
    """Decide complete word behavior, or return a shortest distinguishing word.

    The models must use the same prime-power ring and ordered finite alphabet.
    If carrier sizes are p**b and p**c, Theorem 24.5 gives the strict horizon
    ``len(word) < b+c``.  Product-orbit BFS checks all distinct state pairs and
    avoids materializing the exponentially many repeated short words.  The
    horizon includes noncanonical digit strings and the empty word.
    """
    kinds = (RingLinearRealization, CanonicalResidualModule)
    if not isinstance(left, kinds) or not isinstance(right, kinds):
        raise TypeError("comparison requires exact finite ring realizations")
    if (left.prime, left.exponent, left.alphabet) != (right.prime, right.exponent, right.alphabet):
        raise ValidationError("comparison requires the same ring and finite alphabet")
    limits = left.budget if budget is None else budget
    if not isinstance(limits, ResidualModuleBudget):
        raise TypeError("budget must be a ResidualModuleBudget")
    product_size = left.carrier_cardinality * right.carrier_cardinality
    limits.require(product_size, limits.max_carrier_size, "comparison product cardinality")
    horizon = _log_prime_cardinality(left.carrier_cardinality, left.prime) + _log_prime_cardinality(
        right.carrier_cardinality, right.prime
    )
    limits.require(horizon, limits.max_word_length, "comparison word horizon")
    transition_work = sum(
        model.dimension**2 if isinstance(model, RingLinearRealization) else 1 for model in (left, right)
    )
    output_work = sum(
        model.dimension if isinstance(model, RingLinearRealization) else 1 for model in (left, right)
    )
    limits.require(
        product_size * (1 + output_work + len(left.alphabet) * max(1, transition_work)),
        limits.max_work,
        "comparison work",
    )
    start = (left.seed, right.seed)
    parents = {start: None}
    queue = deque([start])
    while queue:
        pair = queue.popleft()
        left_value, right_value = left._output(pair[0]), right._output(pair[1])
        if left_value != right_value:
            digits = []
            cursor = pair
            while parents[cursor] is not None:
                cursor, digit = parents[cursor]
                digits.append(digit)
            witness = tuple(reversed(digits))
            if len(witness) >= horizon:
                raise ArithmeticError("linear p-group witness violates the finite-length theorem")
            return WordComparison(False, witness, left_value, right_value, horizon, len(parents))
        for digit in left.alphabet:
            target = (left._transition(pair[0], digit), right._transition(pair[1], digit))
            if target not in parents:
                parents[target] = (pair, digit)
                queue.append(target)
    return WordComparison(True, None, None, None, horizon, len(parents))


__all__ = [
    "DEFAULT_RESIDUAL_BUDGET",
    "CanonicalResidualModule",
    "ModuleLawReport",
    "ResidualModuleBudget",
    "RingLinearRealization",
    "WordComparison",
    "canonical_residual_module",
    "finite_word_comparison",
]
