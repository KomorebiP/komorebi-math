"""Universal defect coordinates for finite companion recurrences."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb
from typing import Iterable, Sequence

from ..arithmetic.recurrence import LinearRecurrence
from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from .budget import DEFAULT_COORDINATE_BUDGET, CoordinateBudget
from .exact import (
    Matrix,
    Scalar,
    Vector,
    as_vector,
    matrix_multiply,
    matrix_vector_multiply,
)
from .normalized import apply_lowering_polynomial
from .transform import CoordinateTransform

RecurrenceLike = Iterable[Scalar] | LinearRecurrence


def _coefficients(values: RecurrenceLike) -> Vector:
    if isinstance(values, LinearRecurrence):
        data = as_vector(values.coefficients)
    else:
        data = as_vector(values)
    if not data:
        raise ValidationError("a recurrence needs at least one coefficient")
    return data


def recurrence_polynomial(coefficients: RecurrenceLike) -> Vector:
    r"""Return ascending coefficients of ``x^d-sum_i c_i x^i``."""

    recurrence = _coefficients(coefficients)
    return tuple(-value for value in recurrence) + (Fraction(1),)


def companion_matrix(coefficients: RecurrenceLike) -> Matrix:
    r"""Return the exact companion matrix for consecutive sequence states.

    The recurrence convention is
    ``a_(n+d)=sum_(i=0)^(d-1) c_i a_(n+i)``.
    """

    recurrence = _coefficients(coefficients)
    order = len(recurrence)
    rows: list[list[Fraction]] = [
        [Fraction(0) for _ in range(order)] for _ in range(order)
    ]
    for index in range(order - 1):
        rows[index][index + 1] = Fraction(1)
    rows[-1] = list(recurrence)
    return tuple(tuple(row) for row in rows)


def apply_recurrence_defect(
    values: Sequence[Scalar],
    coefficients: RecurrenceLike,
) -> Vector:
    """Return the unique recurrence defect stream of a finite prefix."""

    return apply_lowering_polynomial(values, recurrence_polynomial(coefficients))


def recurrence_analysis_transform(
    coefficients: RecurrenceLike,
    length: int,
    *,
    name: str | None = None,
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> CoordinateTransform:
    r"""Map raw values to initial state followed by recurrence defects.

    The matrix is unit lower triangular. It is therefore invertible over
    every coefficient field containing the recurrence coefficients.
    """

    recurrence = _coefficients(coefficients)
    order = len(recurrence)
    if isinstance(length, bool) or not isinstance(length, int) or length < order:
        raise DimensionError("length must be at least the recurrence order")
    budget.require_matrix(length, length, operation="recurrence coordinate analysis")
    rows: list[list[Fraction]] = [
        [Fraction(0) for _ in range(length)] for _ in range(length)
    ]
    for row in range(order):
        rows[row][row] = Fraction(1)
    for defect_index in range(length - order):
        row = order + defect_index
        for offset, coefficient in enumerate(recurrence):
            rows[row][defect_index + offset] = -coefficient
        rows[row][defect_index + order] = Fraction(1)
    label = name or f"order {order} recurrence"
    return CoordinateTransform(
        f"raw prefix length {length}",
        f"{label} defect coordinates length {length}",
        tuple(tuple(row) for row in rows),
        "CC.R1",
        "initial state and recurrence defect analysis",
    )


def recurrence_synthesis_transform(
    coefficients: RecurrenceLike,
    length: int,
    *,
    name: str | None = None,
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> CoordinateTransform:
    """Return the exact inverse of recurrence defect analysis."""

    return recurrence_analysis_transform(
        coefficients,
        length,
        name=name,
        budget=budget,
    ).inverse(theorem="CC.R1")


@dataclass(frozen=True, slots=True)
class RecurrenceCoordinateRecord:
    """Initial state and forcing coordinates for one recurrence model."""

    name: str
    coefficients: Vector
    initial: Vector
    defects: Vector

    def __post_init__(self) -> None:
        coefficients = as_vector(self.coefficients)
        initial = as_vector(self.initial)
        defects = as_vector(self.defects)
        if not coefficients:
            raise ValidationError("a recurrence coordinate record needs coefficients")
        if len(initial) != len(coefficients):
            raise DimensionError("initial state length must equal recurrence order")
        if not self.name.strip():
            raise ValidationError("recurrence coordinate name must be nonempty")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "defects", defects)

    @property
    def order(self) -> int:
        """Return the recurrence order."""

        return len(self.coefficients)

    @property
    def length(self) -> int:
        """Return the represented prefix length."""

        return len(self.initial) + len(self.defects)

    @property
    def homogeneous(self) -> bool:
        """Return whether every defect coordinate vanishes."""

        return all(value == 0 for value in self.defects)

    def coordinates(self) -> Vector:
        """Return the concatenated coordinate vector."""

        return self.initial + self.defects

    def reconstruct(self) -> Vector:
        """Reconstruct the exact finite sequence prefix."""

        return synthesize_recurrence_coordinates(self)

    def certificate(self) -> Certificate:
        """Return an exact round-trip certificate."""

        values = self.reconstruct()
        repeated = analyze_recurrence_coordinates(
            values,
            self.coefficients,
            name=self.name,
        )
        valid = repeated.initial == self.initial and repeated.defects == self.defects
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT,
            assumptions_checked=(
                "finite sequence prefix",
                "constant coefficient recurrence",
            ),
            diagnostics={
                "length": self.length,
                "order": self.order,
                "homogeneous": self.homogeneous,
            },
            theorem_id="CC.R1",
            message="recurrence defect coordinates reconstruct exactly",
        )

    def as_linear_recurrence(self) -> LinearRecurrence:
        """Return the homogeneous recurrence determined by the initial state."""

        return LinearRecurrence(self.coefficients, self.initial, name=self.name)


def analyze_recurrence_coordinates(
    values: Sequence[Scalar],
    coefficients: RecurrenceLike,
    *,
    name: str = "recurrence",
) -> RecurrenceCoordinateRecord:
    """Analyze a finite sequence prefix in recurrence defect coordinates."""

    data = as_vector(values)
    recurrence = _coefficients(coefficients)
    order = len(recurrence)
    if len(data) < order:
        raise DimensionError("the sequence prefix is shorter than the recurrence order")
    return RecurrenceCoordinateRecord(
        name,
        recurrence,
        data[:order],
        apply_recurrence_defect(data, recurrence),
    )


def synthesize_recurrence_coordinates(record: RecurrenceCoordinateRecord) -> Vector:
    """Reconstruct a prefix from initial state and recurrence defects."""

    if not isinstance(record, RecurrenceCoordinateRecord):
        raise ValidationError("record must be a RecurrenceCoordinateRecord")
    values = list(record.initial)
    order = record.order
    for defect_index, defect in enumerate(record.defects):
        next_value = sum(
            (
                record.coefficients[offset] * values[defect_index + offset]
                for offset in range(order)
            ),
            Fraction(0),
        ) + defect
        values.append(next_value)
    return tuple(values)


def coordinate_change_between_recurrences(
    source_coefficients: RecurrenceLike,
    target_coefficients: RecurrenceLike,
    length: int,
    *,
    source_name: str = "source recurrence",
    target_name: str = "target recurrence",
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> CoordinateTransform:
    """Return the exact finite coordinate map between recurrence models."""

    source_synthesis = recurrence_synthesis_transform(
        source_coefficients,
        length,
        name=source_name,
        budget=budget,
    )
    target_analysis = recurrence_analysis_transform(
        target_coefficients,
        length,
        name=target_name,
        budget=budget,
    )
    matrix = matrix_multiply(target_analysis.matrix, source_synthesis.matrix)
    return CoordinateTransform(
        source_synthesis.source,
        target_analysis.target,
        matrix,
        "CC.R2",
        "finite change between recurrence defect coordinate systems",
    )


def defect_green_weights(coefficients: RecurrenceLike, count: int) -> Vector:
    r"""Return ``g_l=e_1^T C^(l-1)e_d`` for ``1 <= l <= count``."""

    recurrence = _coefficients(coefficients)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValidationError("count must be a nonnegative integer")
    order = len(recurrence)
    companion = companion_matrix(recurrence)
    injection = tuple(Fraction(int(index == order - 1)) for index in range(order))
    output: list[Fraction] = []
    state = injection
    for _lag in range(1, count + 1):
        output.append(state[0])
        state = matrix_vector_multiply(companion, state)
    return tuple(output)


def _homogeneous_prefix(initial: Vector, coefficients: Vector, length: int) -> Vector:
    record = RecurrenceCoordinateRecord(
        "homogeneous",
        coefficients,
        initial,
        tuple(Fraction(0) for _ in range(length - len(initial))),
    )
    return synthesize_recurrence_coordinates(record)


def variation_of_constants_prefix(record: RecurrenceCoordinateRecord) -> Vector:
    """Reconstruct a prefix through the companion Green-kernel formula."""

    if not isinstance(record, RecurrenceCoordinateRecord):
        raise ValidationError("record must be a RecurrenceCoordinateRecord")
    length = record.length
    homogeneous = _homogeneous_prefix(record.initial, record.coefficients, length)
    green = defect_green_weights(record.coefficients, length)
    values = list(homogeneous)
    for index in range(length):
        correction = Fraction(0)
        for defect_index, defect in enumerate(record.defects):
            lag = index - defect_index
            if lag >= 1:
                correction += green[lag - 1] * defect
        values[index] += correction
    return tuple(values)


def fibonacci_numbers(count: int) -> tuple[int, ...]:
    """Return Fibonacci numbers beginning with zero."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValidationError("count must be a nonnegative integer")
    result: list[int] = []
    a, b = 0, 1
    for _ in range(count):
        result.append(a)
        a, b = b, a + b
    return tuple(result)


def signed_secant_euler_numbers(count: int) -> tuple[int, ...]:
    r"""Return ``E_n`` from ``sech(z)=sum E_n z^n/n!``."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValidationError("count must be a nonnegative integer")
    if count == 0:
        return ()
    values = [0 for _ in range(count)]
    values[0] = 1
    for n in range(1, count):
        values[n] = -sum(comb(n, k) * values[n - k] for k in range(2, n + 1, 2))
    return tuple(values)


def euler_fibonacci_coordinates(count: int) -> RecurrenceCoordinateRecord:
    """Return exact Fibonacci companion coordinates of a signed Euler prefix."""

    values = signed_secant_euler_numbers(count)
    if count < 2:
        raise DimensionError("the Euler to Fibonacci record needs at least two terms")
    return analyze_recurrence_coordinates(
        values,
        (1, 1),
        name="Fibonacci companion",
    )


__all__ = [
    "RecurrenceCoordinateRecord",
    "RecurrenceLike",
    "analyze_recurrence_coordinates",
    "apply_recurrence_defect",
    "companion_matrix",
    "coordinate_change_between_recurrences",
    "defect_green_weights",
    "euler_fibonacci_coordinates",
    "fibonacci_numbers",
    "recurrence_analysis_transform",
    "recurrence_polynomial",
    "recurrence_synthesis_transform",
    "signed_secant_euler_numbers",
    "synthesize_recurrence_coordinates",
    "variation_of_constants_prefix",
]
