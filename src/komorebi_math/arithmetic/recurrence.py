"""Exact linear recurrences, residue packets, and modular orbit diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from numbers import Integral, Number, Rational
from typing import Iterable

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import BackendUnavailableError, DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented


def _integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValidationError(f"{name} must be an integer")
    return int(value)


def _native_number(value: Number) -> Number:
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Rational):
        return Fraction(int(value.numerator), int(value.denominator))
    return value


def _object_matrix(matrix: Iterable[Iterable[Number]]) -> np.ndarray:
    # Object arrays can still contain fixed-width NumPy integer scalars. Lift
    # them before arithmetic so an "exact" product cannot overflow internally.
    array = np.asarray(
        tuple(tuple(_native_number(value) for value in row) for row in matrix),
        dtype=object,
    )
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise DimensionError("matrix must be square")
    return array


def matrix_power_exact(matrix: np.ndarray, exponent: int) -> np.ndarray:
    """Return a square object matrix power by binary exponentiation."""

    array = _object_matrix(matrix)
    power = _integer(exponent, name="exponent")
    if power < 0:
        raise ValidationError("exponent must be nonnegative")
    result = np.eye(array.shape[0], dtype=object)
    base = array.copy()
    while power:
        if power & 1:
            result = result @ base
        base = base @ base
        power //= 2
    return result


def matrix_power_mod(matrix: np.ndarray, exponent: int, modulus: int) -> np.ndarray:
    """Return a square matrix power modulo ``modulus``."""

    array = _object_matrix(matrix)
    power = _integer(exponent, name="exponent")
    mod = _integer(modulus, name="modulus")
    if power < 0:
        raise ValidationError("exponent must be nonnegative")
    if mod < 2:
        raise ValidationError("modulus must be at least two")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in array.flat):
        raise ValidationError("modular matrix powers require integral matrix entries")
    result = np.eye(array.shape[0], dtype=object)
    base = np.vectorize(lambda value: int(value) % mod, otypes=[object])(array)
    while power:
        if power & 1:
            result = np.vectorize(lambda value: int(value) % mod, otypes=[object])(result @ base)
        base = np.vectorize(lambda value: int(value) % mod, otypes=[object])(base @ base)
        power //= 2
    return result


def determinant_integer(matrix: np.ndarray) -> int:
    """Return an exact integer determinant using fraction elimination."""

    array = _object_matrix(matrix)
    n = array.shape[0]
    work = [[Fraction(array[i, j]) for j in range(n)] for i in range(n)]
    determinant = Fraction(1)
    sign = 1
    for column in range(n):
        pivot = next((row for row in range(column, n) if work[row][column] != 0), None)
        if pivot is None:
            return 0
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            sign *= -1
        pivot_value = work[column][column]
        determinant *= pivot_value
        for row in range(column + 1, n):
            factor = work[row][column] / pivot_value
            for index in range(column, n):
                work[row][index] -= factor * work[column][index]
    result = determinant * sign
    if result.denominator != 1:
        raise ValidationError("determinant is not integral")
    return result.numerator


@dataclass(frozen=True, slots=True)
class OrbitPeriod:
    """Preperiod and period of a finite modular recurrence orbit."""

    modulus: int
    preperiod: int
    period: int
    states_checked: int
    purely_periodic: bool
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="PSC-IX-T43",
        title="Companion jump recurrence",
        status=ClaimStatus.PROVED,
        source="Paper IX",
        assumptions=("finite order linear recurrence", "positive residue modulus"),
        verification_suite="arithmetic.recurrence",
    )
)
class LinearRecurrence:
    """An exact constant coefficient recurrence.

    The convention is
    ``x[n + d] = sum(coefficients[i] * x[n + i] for i in range(d))``.
    """

    def __init__(self, coefficients: Iterable[Number], initial: Iterable[Number], *, name: str | None = None) -> None:
        recurrence = tuple(_native_number(value) for value in coefficients)
        seeds = tuple(_native_number(value) for value in initial)
        if not recurrence:
            raise ValidationError("at least one recurrence coefficient is required")
        if len(seeds) != len(recurrence):
            raise DimensionError("initial value count must equal the recurrence order")
        if any(isinstance(value, bool) or not isinstance(value, Number) for value in (*recurrence, *seeds)):
            raise ValidationError("recurrence data must be numeric and cannot be boolean")
        self.coefficients = recurrence
        self.initial = seeds
        self.name = name or "linear recurrence"
        self._cache: list[Number] = list(seeds)

    @property
    def order(self) -> int:
        """Return the recurrence order."""

        return len(self.coefficients)

    @property
    def companion_matrix(self) -> np.ndarray:
        """Return the state transition matrix for ``(x_n, ..., x_{n+d-1})``."""

        d = self.order
        matrix = np.zeros((d, d), dtype=object)
        for index in range(d - 1):
            matrix[index, index + 1] = 1
        matrix[d - 1, :] = np.asarray(self.coefficients, dtype=object)
        return matrix

    @property
    def initial_state(self) -> np.ndarray:
        """Return the initial state column as a one dimensional object array."""

        return np.asarray(self.initial, dtype=object)

    def term(self, index: int) -> Number:
        """Return one exact term."""

        n = _integer(index, name="index")
        if n < 0:
            raise ValidationError("index must be nonnegative")
        while len(self._cache) <= n:
            start = len(self._cache) - self.order
            value = sum(self.coefficients[offset] * self._cache[start + offset] for offset in range(self.order))
            self._cache.append(value)
        return self._cache[n]

    def _integral_data(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return integral recurrence data for modular execution."""

        data = (*self.coefficients, *self.initial)
        if any(
            isinstance(value, bool)
            or not isinstance(value, Rational)
            or Fraction(value).denominator != 1
            for value in data
        ):
            raise ValidationError("modular recurrence execution requires integral coefficients and initial values")
        return tuple(int(value) for value in self.coefficients), tuple(int(value) for value in self.initial)

    def term_mod(self, index: int, modulus: int) -> int:
        """Return one recurrence term modulo ``modulus`` in logarithmic matrix time."""

        n = _integer(index, name="index")
        mod = _integer(modulus, name="modulus")
        if n < 0:
            raise ValidationError("index must be nonnegative")
        if mod < 2:
            raise ValidationError("modulus must be at least two")
        coefficients, initial = self._integral_data()
        matrix = self.companion_matrix
        matrix[-1, :] = coefficients
        jump = matrix_power_mod(matrix, n, mod)
        state = np.asarray(tuple(value % mod for value in initial), dtype=object)
        return int((jump @ state)[0]) % mod

    def first(self, count: int) -> tuple[Number, ...]:
        """Return the first ``count`` terms."""

        total = _integer(count, name="count")
        if total < 0:
            raise ValidationError("count must be nonnegative")
        return tuple(self.term(index) for index in range(total))

    def state(self, index: int) -> np.ndarray:
        """Return ``(x_index, ..., x_(index+order-1))``."""

        n = _integer(index, name="index")
        if n < 0:
            raise ValidationError("index must be nonnegative")
        return np.asarray(tuple(self.term(n + offset) for offset in range(self.order)), dtype=object)

    def jump_matrix(self, modulus: int) -> np.ndarray:
        """Return the exact jump matrix for every ``modulus``th state."""

        step = _integer(modulus, name="modulus")
        if step < 1:
            raise ValidationError("modulus must be positive")
        return matrix_power_exact(self.companion_matrix, step)

    def residue_packet(self, modulus: int, address: int) -> "RecurrencePacket":
        """Return one residue state packet."""

        step = _integer(modulus, name="modulus")
        residue = _integer(address, name="address")
        if step < 1 or residue < 0 or residue >= step:
            raise ValidationError("modulus must be positive and address must lie in its residue range")
        return RecurrencePacket(self, step, residue)

    def period_mod(self, modulus: int, *, max_steps: int | None = None) -> OrbitPeriod:
        """Return the eventual period of the initial state modulo ``modulus``."""

        mod = _integer(modulus, name="modulus")
        if mod < 2:
            raise ValidationError("modulus must be at least two")
        limit = mod**self.order + 1 if max_steps is None else _integer(max_steps, name="max_steps")
        if limit < 1:
            raise ValidationError("max_steps must be positive")
        self._integral_data()
        matrix = np.vectorize(lambda value: int(value) % mod, otypes=[object])(self.companion_matrix)
        state = tuple(int(value) % mod for value in self.initial_state)
        seen: dict[tuple[int, ...], int] = {}
        current = state
        for step in range(limit + 1):
            if current in seen:
                preperiod = seen[current]
                period = step - preperiod
                determinant = determinant_integer(self.companion_matrix)
                invertible = gcd(determinant, mod) == 1
                valid = period > 0 and (not invertible or preperiod == 0)
                return OrbitPeriod(
                    modulus=mod,
                    preperiod=preperiod,
                    period=period,
                    states_checked=step,
                    purely_periodic=preperiod == 0,
                    certificate=Certificate(
                        valid=valid,
                        exactness=Exactness.EXACT,
                        assumptions_checked=("finite modular state space",),
                        diagnostics={
                            "companion_determinant": determinant,
                            "invertible_modulus": invertible,
                            "preperiod": preperiod,
                            "period": period,
                        },
                        theorem_id="PSC-IX-T47",
                        message="finite modular recurrence orbit was classified",
                    ),
                )
            seen[current] = step
            vector = np.asarray(current, dtype=object)
            vector = np.vectorize(lambda value: int(value) % mod, otypes=[object])(matrix @ vector)
            current = tuple(int(value) for value in vector)
        raise ValidationError("period search exceeded max_steps before a state repeated")

    def characteristic_polynomial(self) -> tuple[Number, ...]:
        """Return ascending coefficients of the recurrence characteristic polynomial."""

        return tuple(-coefficient for coefficient in self.coefficients) + (1,)

    def jump_characteristic_polynomial(self, modulus: int) -> tuple[Number, ...]:
        """Return the exact characteristic polynomial of the jump matrix using SymPy."""

        try:
            import sympy as sp
        except ImportError as exc:
            raise BackendUnavailableError("jump characteristic polynomials require the exact SymPy extra") from exc
        jump = self.jump_matrix(modulus)
        symbol = sp.Symbol("x")
        polynomial = sp.Poly(sp.Matrix(jump.tolist()).charpoly(symbol).as_expr(), symbol)
        coefficients_descending = polynomial.all_coeffs()
        return tuple(reversed(tuple(int(value) if value.q == 1 else Fraction(int(value.p), int(value.q)) for value in coefficients_descending)))

    def certificate(self, count: int = 20) -> ComputationResult[tuple[Number, ...]]:
        """Validate the recurrence and companion state evolution."""

        total = _integer(count, name="count")
        if total < self.order:
            raise ValidationError("count must be at least the recurrence order")
        values = self.first(total)
        recurrence_valid = True
        state_valid = True
        matrix = self.companion_matrix
        for n in range(total - self.order):
            expected = sum(self.coefficients[offset] * values[n + offset] for offset in range(self.order))
            recurrence_valid = recurrence_valid and values[n + self.order] == expected
        for n in range(total - self.order):
            state_valid = state_valid and np.array_equal(matrix @ self.state(n), self.state(n + 1))
        exact = all(isinstance(value, Rational) and not isinstance(value, bool) for value in (*self.coefficients, *values))
        return ComputationResult(
            values,
            Certificate(
                valid=recurrence_valid and state_valid,
                exactness=Exactness.EXACT if exact else Exactness.NUMERICAL,
                assumptions_checked=("finite recurrence prefix",),
                diagnostics={"order": self.order, "count": total},
                theorem_id="PSC-IX-T43",
                message="scalar recurrence and companion state evolution agree",
            ),
        )


@dataclass(frozen=True, slots=True)
class RecurrencePacket:
    """One residue state packet of a linear recurrence."""

    recurrence: LinearRecurrence
    modulus: int
    address: int

    @property
    def jump_matrix(self) -> np.ndarray:
        """Return the common residue jump matrix."""

        return self.recurrence.jump_matrix(self.modulus)

    @property
    def initial_state(self) -> np.ndarray:
        """Return the state at the residue address."""

        return self.recurrence.state(self.address)

    def state(self, packet_index: int) -> np.ndarray:
        """Return the packet state at one coarse index."""

        k = _integer(packet_index, name="packet_index")
        if k < 0:
            raise ValidationError("packet_index must be nonnegative")
        return matrix_power_exact(self.jump_matrix, k) @ self.initial_state

    def term(self, packet_index: int) -> Number:
        """Return the scalar residue subsequence term."""

        return self.recurrence.term(self.modulus * _integer(packet_index, name="packet_index") + self.address)

    def term_mod(self, packet_index: int, modulus: int) -> int:
        """Return one residue packet term modulo ``modulus``."""

        k = _integer(packet_index, name="packet_index")
        if k < 0:
            raise ValidationError("packet_index must be nonnegative")
        return self.recurrence.term_mod(self.modulus * k + self.address, modulus)

    def first(self, count: int) -> tuple[Number, ...]:
        """Return the first scalar terms in this packet."""

        total = _integer(count, name="count")
        if total < 0:
            raise ValidationError("count must be nonnegative")
        return tuple(self.term(index) for index in range(total))

    def certificate(self, count: int = 10) -> ComputationResult[tuple[Number, ...]]:
        """Validate the jump matrix representation of the packet."""

        total = _integer(count, name="count")
        if total < 1:
            raise ValidationError("count must be positive")
        values = self.first(total)
        valid = all(self.state(index)[0] == values[index] for index in range(total))
        exact = all(isinstance(value, Rational) and not isinstance(value, bool) for value in values)
        return ComputationResult(
            values,
            Certificate(
                valid=valid,
                exactness=Exactness.EXACT if exact else Exactness.NUMERICAL,
                assumptions_checked=("finite residue packet",),
                diagnostics={"modulus": self.modulus, "address": self.address, "count": total},
                theorem_id="PSC-IX-T43",
                message="residue packet terms agree with jump matrix evolution",
            ),
        )


def second_order_trace(first: Number, determinant: Number, index: int) -> Number:
    """Return the trace sequence ``V_index`` for a second order recurrence."""

    n = _integer(index, name="index")
    if n < 0:
        raise ValidationError("index must be nonnegative")
    if n == 0:
        return 2
    if n == 1:
        return first
    left: Number = 2
    right: Number = first
    for _ in range(2, n + 1):
        left, right = right, first * right - determinant * left
    return right


def second_order_jump(first: Number, determinant: Number, modulus: int) -> tuple[Number, Number]:
    """Return coefficients ``(V_modulus, determinant**modulus)`` of a residue jump."""

    step = _integer(modulus, name="modulus")
    if step < 1:
        raise ValidationError("modulus must be positive")
    return second_order_trace(first, determinant, step), determinant**step


def fibonacci_recurrence() -> LinearRecurrence:
    """Return the Fibonacci recurrence."""

    return LinearRecurrence((1, 1), (0, 1), name="Fibonacci")


def lucas_recurrence() -> LinearRecurrence:
    """Return the Lucas recurrence."""

    return LinearRecurrence((1, 1), (2, 1), name="Lucas")


__all__ = [
    "LinearRecurrence",
    "OrbitPeriod",
    "RecurrencePacket",
    "determinant_integer",
    "fibonacci_recurrence",
    "lucas_recurrence",
    "matrix_power_exact",
    "matrix_power_mod",
    "second_order_jump",
    "second_order_trace",
]
