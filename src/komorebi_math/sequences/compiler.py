"""Finite sequence analysis and executable model compilation."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral, Number, Rational
from typing import Iterable

from .._number_theory import require_integer
from ..arithmetic.recurrence import LinearRecurrence
from ..certificates import Certificate, Exactness
from ..exceptions import ValidationError
from .automatic import AutomaticSequence
from .base import Sequence
from .holonomic import HolonomicRecurrence


def _fraction_values(values: Iterable[Rational]) -> tuple[Fraction, ...]:
    result: list[Fraction] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Rational):
            raise ValidationError("exact recurrence detection requires integer or rational values")
        result.append(Fraction(value))
    return tuple(result)


def _solve_rational_system(matrix: list[list[Fraction]], target: list[Fraction]) -> tuple[Fraction, ...] | None:
    if len(matrix) != len(target):
        raise ValidationError("matrix and target row counts must agree")
    if not matrix:
        return ()
    columns = len(matrix[0])
    if any(len(row) != columns for row in matrix):
        raise ValidationError("linear system rows must have one common length")
    augmented = [row[:] + [target_value] for row, target_value in zip(matrix, target, strict=True)]
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(columns):
        found = next((row for row in range(pivot_row, len(augmented)) if augmented[row][column] != 0), None)
        if found is None:
            continue
        augmented[pivot_row], augmented[found] = augmented[found], augmented[pivot_row]
        pivot = augmented[pivot_row][column]
        augmented[pivot_row] = [value / pivot for value in augmented[pivot_row]]
        for row in range(len(augmented)):
            if row == pivot_row:
                continue
            factor = augmented[row][column]
            if factor != 0:
                augmented[row] = [
                    augmented[row][index] - factor * augmented[pivot_row][index]
                    for index in range(columns + 1)
                ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(augmented):
            break
    for row in augmented:
        if all(value == 0 for value in row[:columns]) and row[columns] != 0:
            return None
    solution = [Fraction(0) for _ in range(columns)]
    for row, column in enumerate(pivot_columns):
        solution[column] = augmented[row][columns]
    return tuple(solution)


def detect_linear_recurrence(
    values: Iterable[Rational],
    *,
    max_order: int | None = None,
) -> LinearRecurrence | None:
    """Detect the minimum exact constant coefficient recurrence supported by a prefix."""

    data = _fraction_values(values)
    if len(data) < 3:
        raise ValidationError("at least three values are required for recurrence detection")
    if max_order is None:
        limit = len(data) // 2
    else:
        requested = require_integer(max_order, name="max_order")
        limit = min(requested, len(data) // 2)
    if limit < 1:
        raise ValidationError("max_order must be positive")
    for order in range(1, limit + 1):
        matrix = [list(data[n : n + order]) for n in range(len(data) - order)]
        target = [data[n + order] for n in range(len(data) - order)]
        solution = _solve_rational_system(matrix, target)
        if solution is None:
            continue
        if all(
            data[n + order] == sum(solution[offset] * data[n + offset] for offset in range(order))
            for n in range(len(data) - order)
        ):
            return LinearRecurrence(solution, data[:order], name="detected recurrence")
    return None


def recurrence_generating_function(recurrence: LinearRecurrence) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    """Return numerator and denominator coefficients of the ordinary generating function.

    Coefficients are in ascending powers of the generating variable.
    """

    d = recurrence.order
    coefficients = tuple(Fraction(value) for value in recurrence.coefficients)
    initial = tuple(Fraction(value) for value in recurrence.initial)
    denominator = [Fraction(1)]
    denominator.extend(-coefficients[d - k] for k in range(1, d + 1))
    numerator: list[Fraction] = []
    for n in range(d):
        value = initial[n]
        for k in range(1, n + 1):
            value += denominator[k] * initial[n - k]
        numerator.append(value)
    while len(numerator) > 1 and numerator[-1] == 0:
        numerator.pop()
    return tuple(numerator), tuple(denominator)


@dataclass(frozen=True, slots=True)
class CompiledSequence:
    """An executable sequence strategy with an explicit validation certificate."""

    sequence: Sequence
    strategy: str
    certificate: Certificate
    recurrence: LinearRecurrence | None = None
    automatic: AutomaticSequence | None = None
    holonomic: HolonomicRecurrence | None = None

    def term(self, index: int) -> Number:
        """Return one term through the selected execution strategy."""

        if self.recurrence is not None:
            return self.recurrence.term(index)
        if self.automatic is not None:
            return self.automatic.term(index)
        if self.holonomic is not None:
            return self.holonomic.term(index)
        return self.sequence.term(index)

    def first(self, count: int) -> tuple[Number, ...]:
        """Return a finite prefix through the selected strategy."""

        if self.recurrence is not None:
            return self.recurrence.first(count)
        if self.automatic is not None:
            return self.automatic.first(count)
        if self.holonomic is not None:
            return self.holonomic.first(count)
        return self.sequence.first(count)

    def modulo(self, index: int, modulus: int) -> int:
        """Return one integral term modulo an integer."""

        n = require_integer(index, name="index")
        mod = require_integer(modulus, name="modulus")
        if n < 0:
            raise ValidationError("index must be nonnegative")
        if mod < 2:
            raise ValidationError("modulus must be at least two")
        if self.recurrence is not None:
            return self.recurrence.term_mod(n, mod)
        value = self.term(n)
        if isinstance(value, bool) or not isinstance(value, Rational):
            raise ValidationError("modular execution requires an exact rational sequence term")
        rational = Fraction(value)
        if rational.denominator != 1:
            raise ValidationError("modular execution requires an integral sequence term")
        return rational.numerator % mod

    def generating_function(self) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
        """Return a rational ordinary generating function when recurrence compiled."""

        if self.recurrence is None:
            raise ValidationError("the selected strategy has no certified rational generating function")
        return recurrence_generating_function(self.recurrence)


def _model_certificate(
    *,
    valid: bool,
    exact: bool,
    model_name: str,
    sample_count: int,
    assumptions_unchecked: tuple[str, ...],
    diagnostics: dict[str, object],
) -> Certificate:
    return Certificate(
        valid=valid,
        exactness=Exactness.EXACT if exact else Exactness.CONDITIONAL,
        assumptions_checked=(model_name, "finite prefix agreement"),
        assumptions_unchecked=assumptions_unchecked,
        diagnostics={"sample_count": sample_count, **diagnostics},
        message=f"the declared {model_name} agrees with the sequence on the certified prefix",
    )


def compile_sequence(
    sequence: Sequence,
    *,
    sample_count: int = 24,
    max_order: int | None = 8,
) -> CompiledSequence:
    """Compile a sequence to a declared or detected finite execution model."""

    count_requested = require_integer(sample_count, name="sample_count")
    if count_requested < 3:
        raise ValidationError("sample_count must be at least three")
    if max_order is not None and require_integer(max_order, name="max_order") < 1:
        raise ValidationError("max_order must be positive")

    known = sequence.metadata.get("recurrence")
    if isinstance(known, LinearRecurrence):
        count = max(count_requested, 2 * known.order + 2)
        valid = known.first(count) == sequence.first(count)
        certified = sequence.metadata.get("recurrence_certified") is True
        certificate = _model_certificate(
            valid=valid,
            exact=certified,
            model_name="linear recurrence",
            sample_count=count,
            assumptions_unchecked=() if certified else ("finite agreement does not prove a user supplied recurrence for all terms",),
            diagnostics={"order": known.order},
        )
        return CompiledSequence(sequence, "declared linear recurrence", certificate, recurrence=known)

    automatic = sequence.metadata.get("automatic_model")
    if isinstance(automatic, AutomaticSequence):
        valid = automatic.first(count_requested) == sequence.first(count_requested)
        certified = sequence.metadata.get("automatic_certified") is True
        certificate = _model_certificate(
            valid=valid,
            exact=certified,
            model_name="finite automaton",
            sample_count=count_requested,
            assumptions_unchecked=() if certified else ("finite agreement does not identify an arbitrary infinite automatic sequence",),
            diagnostics={"base": automatic.base, "states": automatic.state_count},
        )
        return CompiledSequence(sequence, "declared finite automaton", certificate, automatic=automatic)

    holonomic = sequence.metadata.get("holonomic_recurrence")
    if isinstance(holonomic, HolonomicRecurrence):
        count = max(count_requested, 2 * holonomic.order + 2)
        valid = holonomic.first(count) == sequence.first(count)
        certified = sequence.metadata.get("holonomic_certified") is True
        certificate = _model_certificate(
            valid=valid,
            exact=certified,
            model_name="holonomic recurrence",
            sample_count=count,
            assumptions_unchecked=() if certified else ("finite agreement does not prove a user supplied holonomic recurrence for all terms",),
            diagnostics={"order": holonomic.order},
        )
        return CompiledSequence(sequence, "declared holonomic recurrence", certificate, holonomic=holonomic)

    values = sequence.first(count_requested)
    if all(isinstance(value, Rational) and not isinstance(value, bool) for value in values):
        recurrence = detect_linear_recurrence(values, max_order=max_order)
        if recurrence is not None:
            return CompiledSequence(
                sequence,
                "detected linear recurrence",
                Certificate(
                    valid=True,
                    exactness=Exactness.CONDITIONAL,
                    assumptions_checked=("exact finite prefix recurrence",),
                    assumptions_unchecked=("finite prefix detection does not prove the recurrence for all future terms",),
                    diagnostics={"sample_count": count_requested, "order": recurrence.order},
                    message="an exact recurrence fits the complete sampled prefix",
                ),
                recurrence=recurrence,
            )
    return CompiledSequence(
        sequence,
        "direct evaluator",
        Certificate(
            valid=True,
            exactness=Exactness.EXACT if sequence.exact else Exactness.NUMERICAL,
            assumptions_checked=("callable sequence evaluator",),
            diagnostics={"sample_count": count_requested},
            message="no stronger finite execution strategy was certified",
        ),
    )


__all__ = [
    "CompiledSequence",
    "compile_sequence",
    "detect_linear_recurrence",
    "recurrence_generating_function",
]
