"""Factorial-power Bell recurrences and digit-sum compilers."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb, isqrt
from typing import Iterable, Sequence

from ..certificates import Certificate, Exactness
from ..exceptions import ResourceLimitError, ValidationError
from .budget import DEFAULT_GENERATING_BUDGET, GeneratingBudget
from .congruences import base_digit_sum
from .sequences import (
    bell_numbers,
    generalized_bell_polynomials,
    weighted_generalized_bell_polynomials,
)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor <= isqrt(value):
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def _require_prime(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not _is_prime(value):
        raise ValidationError("prime must be prime")
    return value


def _require_index(value: int, name: str = "index") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{name} must be a nonnegative integer")
    return value


def bell_period_bound(prime: int) -> int:
    r"""Return ``(p**p-1)/(p-1)``, a period bound modulo ``p``."""

    p = _require_prime(prime)
    return (p**p - 1) // (p - 1)


def _bell_residues_mod_prime(prime: int, count: int) -> tuple[int, ...]:
    """Return Bell residues using Touchard's linear recurrence.

    Exact Bell integers become very large.  Modulo a prime, Touchard's
    congruence ``B_(n+p)=B_(n+1)+B_n`` avoids constructing them and gives a
    linear-time finite-state compiler after the first ``p`` values.
    """

    p = _require_prime(prime)
    size = _require_index(count, "count")
    if size == 0:
        return ()
    initial = [value % p for value in bell_numbers(min(size, p))]
    if size <= p:
        return tuple(initial)
    values = initial + [0] * (size - p)
    for n in range(size - p):
        values[n + p] = (values[n + 1] + values[n]) % p
    return tuple(values)


def pointed_subindex_polynomial(index: int, prime: int) -> tuple[int, ...]:
    r"""Return the grouped left side of the pointed subindex identity.

    The coefficient of ``u**j`` is

    ``sum binomial(n-1,m-1)``

    modulo ``prime``, over positive digitwise subindices ``m`` with digit sum
    ``j+1``.
    """

    n = _require_index(index)
    p = _require_prime(prime)
    if n == 0:
        raise ValidationError("the pointed subindex identity requires a positive index")
    digits: list[int] = []
    remaining = n
    while remaining:
        remaining, digit = divmod(remaining, p)
        digits.append(digit)
    least = next(position for position, digit in enumerate(digits) if digit)
    polynomial = [1]
    # The least nonzero position contributes (1+u)^(n_j-1).
    local = [comb(digits[least] - 1, j) % p for j in range(digits[least])]
    polynomial = _convolve_mod(polynomial, local, p)
    # Higher positions contribute (1+u)^n_i.
    for digit in digits[least + 1 :]:
        local = [comb(digit, j) % p for j in range(digit + 1)]
        polynomial = _convolve_mod(polynomial, local, p)
    return tuple(polynomial)


def _convolve_mod(left: Sequence[int], right: Sequence[int], modulus: int) -> list[int]:
    output = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            output[i + j] = (output[i + j] + a * b) % modulus
    return output


@dataclass(frozen=True, slots=True)
class PointedSubindexCertificate:
    """Exact finite certificate for the pointed digit identity."""

    index: int
    prime: int
    left: tuple[int, ...]
    right: tuple[int, ...]

    @property
    def valid(self) -> bool:
        return self.left == self.right

    def require(self) -> None:
        if not self.valid:
            raise ValidationError("pointed subindex identity failed")

    def certificate(self) -> Certificate:
        return Certificate(
            valid=self.valid,
            exactness=Exactness.EXACT,
            assumptions_checked=("prime modulus", "positive index"),
            diagnostics={
                "index": self.index,
                "prime": self.prime,
                "left": self.left,
                "right": self.right,
            },
            theorem_id="NGF.POINTED_SUBINDEX",
            message="pointed subindex identity verified" if self.valid else "identity failed",
        )


def certify_pointed_subindex_identity(index: int, prime: int) -> PointedSubindexCertificate:
    """Certify the pointed subindex polynomial against ``(1+u)^(s_p(n)-1)``."""

    n = _require_index(index)
    p = _require_prime(prime)
    if n == 0:
        raise ValidationError("the pointed subindex identity requires a positive index")
    left = pointed_subindex_polynomial(n, p)
    digit_sum = base_digit_sum(n, p)
    right = tuple(comb(digit_sum - 1, j) % p for j in range(digit_sum))
    result = PointedSubindexCertificate(n, p, left, right)
    result.require()
    return result


@dataclass(frozen=True, slots=True)
class BellDigitReduction:
    """One coefficientwise factorial-power Bell digit-sum reduction."""

    index: int
    prime: int
    factorial_power: int
    digit_sum: int
    high_polynomial: tuple[int, ...]
    reduced_polynomial: tuple[int, ...]

    @property
    def valid(self) -> bool:
        degree = max(len(self.high_polynomial), len(self.reduced_polynomial))
        return all(
            (self.high_polynomial[k] if k < len(self.high_polynomial) else 0)
            == (self.reduced_polynomial[k] if k < len(self.reduced_polynomial) else 0)
            for k in range(degree)
        )

    def require(self) -> None:
        if not self.valid:
            raise ValidationError("Bell digit-sum reduction failed")

    def certificate(self) -> Certificate:
        return Certificate(
            valid=self.valid,
            exactness=Exactness.EXACT,
            assumptions_checked=(
                "prime modulus",
                "factorial_power >= 2",
                "prime-1 divides factorial_power-1",
            ),
            diagnostics={
                "index": self.index,
                "prime": self.prime,
                "factorial_power": self.factorial_power,
                "digit_sum": self.digit_sum,
            },
            theorem_id="NGF.BELL_DIGIT_SUM_REDUCTION",
            message="Bell digit-sum reduction verified" if self.valid else "reduction failed",
        )


def generalized_bell_digit_reduction(
    index: int,
    prime: int,
    factorial_power: int,
    *,
    budget: GeneratingBudget = DEFAULT_GENERATING_BUDGET,
) -> BellDigitReduction:
    """Compute and certify the unweighted Bell digit theorem at one index."""

    n = budget.require_degree(_require_index(index))
    p = _require_prime(prime)
    if (
        isinstance(factorial_power, bool)
        or not isinstance(factorial_power, int)
        or factorial_power < 2
        or (factorial_power - 1) % (p - 1)
    ):
        raise ValidationError(
            "factorial_power must be at least two and congruent to one modulo prime-1"
        )
    high = generalized_bell_polynomials(factorial_power, n + 1)[n]
    digit_sum = base_digit_sum(n, p)
    reduced = generalized_bell_polynomials(1, digit_sum + 1)[digit_sum]
    result = BellDigitReduction(
        n,
        p,
        factorial_power,
        digit_sum,
        tuple(int(value) % p for value in high),
        tuple(int(value) % p for value in reduced),
    )
    result.require()
    return result


def _normalize_digit_weights(values: Iterable[int], required: int) -> tuple[int, ...]:
    table = tuple(int(value) for value in values)
    if len(table) <= required:
        raise ValidationError("digit-weight table is too short")
    return table


def weighted_bell_digit_reduction(
    index: int,
    prime: int,
    factorial_power: int,
    digit_weights: Iterable[int],
    *,
    budget: GeneratingBudget = DEFAULT_GENERATING_BUDGET,
) -> BellDigitReduction:
    """Compute the weighted Bell digit-sum reduction at one index."""

    n = budget.require_degree(_require_index(index))
    p = _require_prime(prime)
    if (
        isinstance(factorial_power, bool)
        or not isinstance(factorial_power, int)
        or factorial_power < 2
        or (factorial_power - 1) % (p - 1)
    ):
        raise ValidationError(
            "factorial_power must be at least two and congruent to one modulo prime-1"
        )
    maximum_digit_sum = max(base_digit_sum(k, p) for k in range(n + 1))
    table = _normalize_digit_weights(digit_weights, maximum_digit_sum)
    high_rows = weighted_generalized_bell_polynomials(
        factorial_power,
        n + 1,
        lambda m: table[base_digit_sum(m, p)],
    )
    digit_sum = base_digit_sum(n, p)
    low_rows = weighted_generalized_bell_polynomials(1, digit_sum + 1, table)
    high = high_rows[n]
    low = low_rows[digit_sum]
    result = BellDigitReduction(
        n,
        p,
        factorial_power,
        digit_sum,
        tuple(int(Fraction(value).numerator * pow(Fraction(value).denominator, -1, p)) % p for value in high),
        tuple(int(Fraction(value).numerator * pow(Fraction(value).denominator, -1, p)) % p for value in low),
    )
    result.require()
    return result


def generalized_bell_parity(index: int, factorial_power: int) -> int:
    """Return the exact parity predicted by the population-count automaton."""

    n = _require_index(index)
    if (
        isinstance(factorial_power, bool)
        or not isinstance(factorial_power, int)
        or factorial_power < 2
    ):
        raise ValidationError("factorial_power must be at least two")
    return 0 if n.bit_count() % 3 == 2 else 1


@dataclass(frozen=True, slots=True)
class BellParityAutomaton:
    """The three-state binary automaton for generalized Bell parity."""

    outputs: tuple[int, int, int] = (1, 1, 0)

    def transition(self, state: int, digit: int) -> int:
        if state not in (0, 1, 2) or digit not in (0, 1):
            raise ValidationError("invalid parity automaton state or digit")
        return state if digit == 0 else (state + 1) % 3

    def evaluate(self, index: int) -> int:
        n = _require_index(index)
        state = 0
        while n:
            n, digit = divmod(n, 2)
            state = self.transition(state, digit)
        return self.outputs[state]


@dataclass(frozen=True, slots=True)
class BellResidueAutomaton:
    """A finite digit-sum automaton for factorial-power Bell residues."""

    prime: int
    factorial_power: int
    period_bound: int
    outputs: tuple[int, ...]

    @classmethod
    def compile(
        cls,
        prime: int,
        factorial_power: int,
        *,
        budget: GeneratingBudget = DEFAULT_GENERATING_BUDGET,
    ) -> "BellResidueAutomaton":
        p = _require_prime(prime)
        if (
            isinstance(factorial_power, bool)
            or not isinstance(factorial_power, int)
            or factorial_power < 2
            or (factorial_power - 1) % (p - 1)
        ):
            raise ValidationError(
                "factorial_power must be at least two and congruent to one modulo prime-1"
            )
        period = bell_period_bound(p)
        budget.require_states(period)
        outputs = _bell_residues_mod_prime(p, period)
        return cls(p, factorial_power, period, outputs)

    def transition(self, state: int, digit: int) -> int:
        if not 0 <= state < self.period_bound:
            raise ValidationError("state lies outside the Bell period bound")
        if not 0 <= digit < self.prime:
            raise ValidationError("digit lies outside the prime radix")
        return (state + digit) % self.period_bound

    def evaluate(self, index: int) -> int:
        n = _require_index(index)
        state = 0
        while n:
            n, digit = divmod(n, self.prime)
            state = self.transition(state, digit)
        return self.outputs[state]

    def certificate(self, count: int) -> Certificate:
        size = _require_index(count, "count")
        direct = generalized_bell_polynomials(self.factorial_power, size)
        failures = tuple(
            n
            for n, row in enumerate(direct)
            if sum(row) % self.prime != self.evaluate(n)
        )
        return Certificate(
            valid=not failures,
            exactness=Exactness.EXACT,
            assumptions_checked=(
                "prime modulus",
                "factorial_power >= 2",
                "prime-1 divides factorial_power-1",
            ),
            diagnostics={
                "count": size,
                "states": self.period_bound,
                "failures": failures,
            },
            theorem_id="NGF.BELL_RESIDUE_AUTOMATON",
            message="Bell residue automaton verified" if not failures else "automaton mismatch",
        )


__all__ = [
    "BellDigitReduction",
    "BellParityAutomaton",
    "BellResidueAutomaton",
    "PointedSubindexCertificate",
    "bell_period_bound",
    "certify_pointed_subindex_identity",
    "generalized_bell_digit_reduction",
    "generalized_bell_parity",
    "pointed_subindex_polynomial",
    "weighted_bell_digit_reduction",
]
