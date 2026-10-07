"""Lazy exact sequence objects and common sequence views."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from fractions import Fraction
from numbers import Integral, Number, Rational
from typing import Any

from .._number_theory import require_integer
from ..exceptions import ValidationError


class Sequence:
    """A cached lazy sequence indexed by nonnegative integers."""

    def __init__(
        self,
        evaluator: Callable[[int], Number],
        *,
        name: str = "sequence",
        exact: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not callable(evaluator):
            raise ValidationError("evaluator must be callable")
        self._evaluator = evaluator
        self.name = str(name)
        self.exact = bool(exact)
        self.metadata = dict(metadata or {})
        self._cache: dict[int, Number] = {}

    def term(self, index: int) -> Number:
        """Return one term, evaluating and caching it when necessary."""

        n = require_integer(index, name="sequence index")
        if n < 0:
            raise ValidationError("sequence index must be a nonnegative integer")
        if n not in self._cache:
            value = self._evaluator(n)
            if not isinstance(value, Number):
                raise ValidationError("sequence evaluator must return a numeric value")
            self._cache[n] = value
        return self._cache[n]

    def first(self, count: int) -> tuple[Number, ...]:
        """Return the first ``count`` terms."""

        total = require_integer(count, name="count")
        if total < 0:
            raise ValidationError("count must be a nonnegative integer")
        return tuple(self.term(index) for index in range(total))

    def residue(self, modulus: int, address: int) -> "Sequence":
        """Return the residue subsequence ``a[modulus*k + address]``."""

        step = require_integer(modulus, name="modulus")
        residue = require_integer(address, name="address")
        if step < 1 or residue < 0 or residue >= step:
            raise ValidationError("modulus must be positive and address must lie in its residue range")
        return Sequence(
            lambda k: self.term(step * k + residue),
            name=f"{self.name}[{residue} mod {step}]",
            exact=self.exact,
            metadata={**self.metadata, "parent": self.name, "modulus": step, "address": residue},
        )

    def modulo(self, modulus: int) -> "Sequence":
        """Return the termwise residue sequence modulo ``modulus``."""

        mod = require_integer(modulus, name="modulus")
        if mod < 2:
            raise ValidationError("modulus must be at least two")

        def reduce_term(index: int) -> int:
            value = self.term(index)
            if isinstance(value, bool) or not isinstance(value, Rational):
                raise ValidationError("modular sequence reduction requires rational terms")
            rational = Fraction(value)
            if rational.denominator != 1:
                raise ValidationError("modular sequence reduction requires integral terms")
            return rational.numerator % mod

        return Sequence(
            reduce_term,
            name=f"{self.name} modulo {mod}",
            exact=True,
            metadata={**self.metadata, "modulus": mod},
        )

    def map(self, function: Callable[[Number], Number], *, name: str | None = None, exact: bool | None = None) -> "Sequence":
        """Apply a scalar function term by term."""

        if not callable(function):
            raise ValidationError("function must be callable")
        return Sequence(
            lambda n: function(self.term(n)),
            name=name or f"map({self.name})",
            exact=self.exact if exact is None else bool(exact),
            metadata={"parent": self.name, "transform": "map"},
        )

    def __getitem__(self, item: int | slice) -> Number | tuple[Number, ...]:
        if isinstance(item, slice):
            start = 0 if item.start is None else require_integer(item.start, name="slice start")
            stop = item.stop
            step = 1 if item.step is None else require_integer(item.step, name="slice step")
            if stop is None:
                raise ValidationError("sequence slices require a finite stop")
            stop_value = require_integer(stop, name="slice stop")
            if start < 0 or stop_value < 0 or step < 1:
                raise ValidationError("sequence slices require nonnegative bounds and positive step")
            return tuple(self.term(index) for index in range(start, stop_value, step))
        return self.term(item)

    def __iter__(self) -> Iterator[Number]:
        index = 0
        while True:
            yield self.term(index)
            index += 1

    def __repr__(self) -> str:
        return f"Sequence(name={self.name!r}, exact={self.exact!r})"


__all__ = ["Sequence"]
