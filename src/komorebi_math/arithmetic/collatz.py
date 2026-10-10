"""Exact finite Collatz words and their integer realization cylinders.

The formulas implement Sections 2, 3 and 6 of the Collatz reader edition 1.1.
Finite words and successful finite descents do not establish universal descent.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Integral

from ..certificates import Certificate, Exactness
from ..exceptions import ResourceLimitError, ValidationError


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValidationError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValidationError(f"{name} must be at least {minimum}")
    return result


def _odd_seed(value: object) -> int:
    result = _integer(value, "seed", 1)
    if result % 2 == 0:
        raise ValidationError("seed must be positive and odd")
    return result


def odd_collatz_step(seed: int) -> tuple[int, int]:
    """Return the next odd state and exact valuation of ``3*seed+1``."""
    value = 3 * _odd_seed(seed) + 1
    valuation = (value & -value).bit_length() - 1
    return value >> valuation, valuation


@dataclass(frozen=True, slots=True)
class CollatzWord:
    """A bounded, owned valuation word with exact affine and cylinder data.

    The empty word is identity on positive odd integers. Resource limits are
    construction limits, not mathematical assertions about the infinite tree.
    """

    valuations: tuple[int, ...] | Iterable[int]
    max_steps: int = field(default=10_000, repr=False, compare=False)
    max_depth: int = field(default=100_000, repr=False, compare=False)
    depth: int = field(init=False)
    offset: int = field(init=False)

    def __post_init__(self) -> None:
        steps = _integer(self.max_steps, "max_steps", 0)
        limit = _integer(self.max_depth, "max_depth", 0)
        values: list[int] = []
        depth = offset = 0
        try:
            iterator = iter(self.valuations)
        except TypeError as exc:
            raise ValidationError("valuations must be an iterable of positive integers") from exc
        for value in iterator:
            if len(values) >= steps:
                raise ResourceLimitError("valuation word exceeds max_steps")
            item = _integer(value, "valuation", 1)
            if depth + item > limit:
                raise ResourceLimitError("valuation word exceeds max_depth")
            offset = 3 * offset + (1 << depth)
            depth += item
            values.append(item)
        object.__setattr__(self, "valuations", tuple(values))
        object.__setattr__(self, "max_steps", steps)
        object.__setattr__(self, "max_depth", limit)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "offset", offset)

    @property
    def length(self) -> int:
        return len(self.valuations)

    @property
    def multiplier(self) -> int:
        return 3**self.length

    @property
    def denominator(self) -> int:
        return 1 << self.depth

    @property
    def slope(self) -> Fraction:
        return Fraction(self.multiplier, self.denominator)

    @property
    def affine_offset(self) -> Fraction:
        return Fraction(self.offset, self.denominator)

    @property
    def modulus(self) -> int:
        """The extra terminal parity bit is retained: ``2**(depth+1)``."""
        return 1 << (self.depth + 1)

    @property
    def residue(self) -> int:
        return ((self.denominator - self.offset) * pow(self.multiplier, -1, self.modulus)) % self.modulus

    @property
    def coefficient_gap(self) -> int:
        return self.denominator - self.multiplier

    @property
    def descent_threshold(self) -> int | None:
        """Strict descent requires a realizing seed above this integer floor."""
        return self.offset // self.coefficient_gap if self.coefficient_gap > 0 else None

    @property
    def uniform_descent_start(self) -> int | None:
        """First positive cylinder member above the strict descent threshold."""
        floor = self.descent_threshold
        if floor is None:
            return None
        first = max(1, floor + 1)
        return first + (self.residue - first) % self.modulus

    def realizes(self, seed: int) -> bool:
        return _odd_seed(seed) % self.modulus == self.residue

    def affine_value(self, seed: int) -> Fraction:
        """Evaluate the expression without asserting that the word is realized."""
        return Fraction(self.multiplier * _odd_seed(seed) + self.offset, self.denominator)

    def apply(self, seed: int) -> int:
        """Evaluate an actual word endpoint, rejecting a nonrealizing seed."""
        if not self.realizes(seed):
            raise ValidationError("seed does not realize this exact valuation word")
        return (self.multiplier * int(seed) + self.offset) // self.denominator

    def orbit(self, seed: int) -> tuple[int, ...]:
        """Replay every step and independently check its recorded valuation."""
        value = _odd_seed(seed)
        if not self.realizes(value):
            raise ValidationError("seed does not realize this exact valuation word")
        states = [value]
        for expected in self.valuations:
            value, actual = odd_collatz_step(value)
            if actual != expected:
                raise ValidationError("valuation replay disagrees with cylinder")
            states.append(value)
        if value != self.apply(seed):
            raise ValidationError("affine endpoint disagrees with direct orbit")
        return tuple(states)

    def descends(self, seed: int) -> bool:
        if not self.realizes(seed):
            raise ValidationError("descent requires a realizing seed")
        return self.coefficient_gap * int(seed) > self.offset

    def concatenate(self, following: CollatzWord) -> CollatzWord:
        """Compose ordered words, keeping this object's construction limits."""
        if not isinstance(following, CollatzWord):
            raise ValidationError("following must be a CollatzWord")
        return CollatzWord(
            self.valuations + following.valuations, max_steps=self.max_steps, max_depth=self.max_depth
        )

    def realization_certificate(self, seed: int) -> Certificate:
        states = self.orbit(seed)
        return Certificate(
            valid=True,
            exactness=Exactness.EXACT,
            assumptions_checked=("positive odd seed", "exact valuations and affine endpoint replayed"),
            diagnostics={
                "word": self.valuations,
                "states": states,
                "modulus": self.modulus,
                "residue": self.residue,
            },
            message="Exact finite valuation word realization; no universal descent claim.",
        )


@dataclass(frozen=True, slots=True)
class FirstDescentResult:
    seed: int
    states: tuple[int, ...]
    word: CollatzWord
    status: str

    def __post_init__(self) -> None:
        seed = _odd_seed(self.seed)
        states = tuple(_odd_seed(x) for x in self.states)
        if not isinstance(self.word, CollatzWord) or states != self.word.orbit(seed):
            raise ValidationError("descent record must replay its exact word")
        if self.status not in {"descent", "terminal", "inconclusive"}:
            raise ValidationError("unknown descent status")
        if self.status == "terminal" and (seed != 1 or states != (1,)):
            raise ValidationError("terminal status requires initial seed one")
        if self.status == "descent" and (
            len(states) < 2 or states[-1] >= seed or any(x < seed for x in states[1:-1])
        ):
            raise ValidationError("first descent must compare every prefix to the initial seed")
        if self.status == "inconclusive" and any(x < seed for x in states[1:]):
            raise ValidationError("an observed descent cannot be labeled inconclusive")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "states", states)

    @property
    def certificate(self) -> Certificate:
        return Certificate(
            valid=self.status in {"descent", "terminal"},
            exactness=Exactness.EXACT,
            assumptions_checked=("finite odd orbit replay", "comparison with original seed"),
            diagnostics={
                "seed": self.seed,
                "status": self.status,
                "steps": self.word.length,
                "depth": self.word.depth,
            },
            message=(
                "Exact first descent (or initial terminal state)."
                if self.status != "inconclusive"
                else "The finite search bound was exhausted; universal descent is not established."
            ),
        )


def first_odd_descent(
    seed: int, *, max_steps: int = 1_000, max_depth: int = 100_000, max_bits: int = 100_000
) -> FirstDescentResult:
    """Search a bounded exact orbit; exhaustion returns an inconclusive record."""
    initial = _odd_seed(seed)
    steps = _integer(max_steps, "max_steps", 0)
    depth_limit = _integer(max_depth, "max_depth", 0)
    bit_limit = _integer(max_bits, "max_bits", 1)
    if initial.bit_length() > bit_limit:
        raise ResourceLimitError("initial seed exceeds max_bits")
    if initial == 1:
        return FirstDescentResult(initial, (1,), CollatzWord(()), "terminal")
    value = initial
    states = [value]
    valuations: list[int] = []
    depth = 0
    status = "inconclusive"
    for _ in range(steps):
        value, valuation = odd_collatz_step(value)
        if value.bit_length() > bit_limit or depth + valuation > depth_limit:
            raise ResourceLimitError("orbit exceeds max_bits or max_depth")
        depth += valuation
        states.append(value)
        valuations.append(valuation)
        if value < initial:
            status = "descent"
            break
    word = CollatzWord(valuations, max_steps=steps, max_depth=depth_limit)
    return FirstDescentResult(initial, tuple(states), word, status)


__all__ = ["CollatzWord", "FirstDescentResult", "first_odd_descent", "odd_collatz_step"]
