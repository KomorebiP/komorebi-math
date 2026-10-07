"""Finite automata with output for exact automatic sequence execution."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import Iterable

from .._number_theory import require_integer
from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import DimensionError, ValidationError
from .base import Sequence


def base_digits(index: int, base: int) -> tuple[int, ...]:
    """Return the canonical most significant first base expansion of an index."""

    n = require_integer(index, name="index")
    radix = require_integer(base, name="base")
    if n < 0:
        raise ValidationError("index must be nonnegative")
    if radix < 2:
        raise ValidationError("base must be at least two")
    if n == 0:
        return (0,)
    digits: list[int] = []
    while n:
        digits.append(n % radix)
        n //= radix
    return tuple(reversed(digits))


@dataclass(frozen=True, slots=True)
class AutomaticSequence:
    """A deterministic finite automaton with numeric output.

    Digits are read from most significant to least significant. The transition
    table has one row per state and one column per base digit.
    """

    base: int
    transitions: tuple[tuple[int, ...], ...]
    outputs: tuple[Number, ...]
    initial_state: int = 0
    name: str = "automatic sequence"

    def __init__(
        self,
        base: int,
        transitions: Iterable[Iterable[int]],
        outputs: Iterable[Number],
        *,
        initial_state: int = 0,
        name: str = "automatic sequence",
    ) -> None:
        radix = require_integer(base, name="base")
        if radix < 2:
            raise ValidationError("base must be at least two")
        table = tuple(tuple(require_integer(target, name="transition target") for target in row) for row in transitions)
        values = tuple(outputs)
        if not table:
            raise ValidationError("at least one automaton state is required")
        if any(len(row) != radix for row in table):
            raise DimensionError("every transition row must contain one target per base digit")
        if len(values) != len(table):
            raise DimensionError("output count must equal the number of states")
        if any(isinstance(value, bool) or not isinstance(value, Number) for value in values):
            raise ValidationError("automatic sequence outputs must be numeric and cannot be boolean")
        if any(target < 0 or target >= len(table) for row in table for target in row):
            raise ValidationError("every transition target must be a valid state index")
        initial = require_integer(initial_state, name="initial_state")
        if initial < 0 or initial >= len(table):
            raise ValidationError("initial_state must be a valid state index")
        if table[initial][0] != initial:
            raise ValidationError("the initial state must have a zero digit loop for canonical leading zero invariance")
        object.__setattr__(self, "base", radix)
        object.__setattr__(self, "transitions", table)
        object.__setattr__(self, "outputs", values)
        object.__setattr__(self, "initial_state", initial)
        object.__setattr__(self, "name", str(name))

    @property
    def state_count(self) -> int:
        """Return the number of automaton states."""

        return len(self.transitions)

    def state(self, index: int) -> int:
        """Return the terminal state reached by the base digits of ``index``."""

        current = self.initial_state
        for digit in base_digits(index, self.base):
            current = self.transitions[current][digit]
        return current

    def term(self, index: int) -> Number:
        """Return one automatic sequence term."""

        return self.outputs[self.state(index)]

    def first(self, count: int) -> tuple[Number, ...]:
        """Return a finite prefix."""

        total = require_integer(count, name="count")
        if total < 0:
            raise ValidationError("count must be nonnegative")
        return tuple(self.term(index) for index in range(total))

    def residue_model(self, power: int, address: int) -> "AutomaticSequence":
        """Return an automaton for ``a[base**power*n + address]``.

        The fixed low digits of the address are applied after the input digits.
        """

        exponent = require_integer(power, name="power")
        residue = require_integer(address, name="address")
        if exponent < 0:
            raise ValidationError("power must be nonnegative")
        modulus = self.base**exponent
        if residue < 0 or residue >= modulus:
            raise ValidationError("address must lie in the declared residue range")
        low_digits = () if exponent == 0 else tuple(
            (residue // self.base**position) % self.base for position in reversed(range(exponent))
        )
        outputs: list[Number] = []
        for state in range(self.state_count):
            current = state
            for digit in low_digits:
                current = self.transitions[current][digit]
            outputs.append(self.outputs[current])
        return AutomaticSequence(
            self.base,
            self.transitions,
            outputs,
            initial_state=self.initial_state,
            name=f"{self.name}[{residue} mod {modulus}]",
        )

    def to_sequence(self) -> Sequence:
        """Return the common lazy sequence wrapper."""

        return Sequence(
            self.term,
            name=self.name,
            exact=True,
            metadata={"automatic_model": self, "automatic_certified": True},
        )

    def certificate(self, count: int = 256) -> ComputationResult[tuple[Number, ...]]:
        """Validate transition closure and finite evaluation consistency."""

        total = require_integer(count, name="count")
        if total < 1:
            raise ValidationError("count must be positive")
        values = self.first(total)
        valid = all(self.term(index) == self.outputs[self.state(index)] for index in range(total))
        return ComputationResult(
            values,
            Certificate(
                valid=valid,
                exactness=Exactness.EXACT,
                assumptions_checked=("finite deterministic transition table", "numeric output table"),
                diagnostics={"base": self.base, "states": self.state_count, "count": total},
                theorem_id="PSC-SEQ-AUTOMATIC",
                message="the automatic sequence agrees with its declared transition and output tables",
            ),
        )


def thue_morse_automaton() -> AutomaticSequence:
    """Return the standard binary parity automaton for Thue Morse."""

    return AutomaticSequence(
        2,
        ((0, 1), (1, 0)),
        (0, 1),
        name="Thue Morse",
    )


__all__ = ["AutomaticSequence", "base_digits", "thue_morse_automaton"]
