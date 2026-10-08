"""Carry-linear valuation profiles and digit-local unit twists."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, gcd, isqrt
from typing import Callable, Sequence

from .._number_theory import require_prime as _require_prime
from ..exceptions import ValidationError
from .padic_units import binomial_valuation, factorial_valuation


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


@dataclass(frozen=True)
class CarryLinearValuation:
    """Profile ``nu(n)=a*n+s*v_p(n!)`` with defect equal to weighted carries."""

    prime: int
    linear_slope: int
    carry_weight: int

    def __post_init__(self) -> None:
        if not _is_prime(self.prime):
            raise ValueError("prime must be prime")
        if self.carry_weight < 0:
            raise ValueError("carry_weight must be nonnegative")

    def value(self, index: int) -> int:
        """Evaluate the profile."""

        if index < 0:
            raise ValueError("index must be nonnegative")
        return self.linear_slope * index + self.carry_weight * factorial_valuation(
            index, self.prime
        )

    def defect(self, index: int, subindex: int) -> int:
        """Return ``nu(n)-nu(k)-nu(n-k)``."""

        return self.value(index) - self.value(subindex) - self.value(index - subindex)


@dataclass(frozen=True)
class CarryLinearCertificate:
    """Finite audit of the carry-linear valuation classification."""

    prime: int
    linear_slope: int
    carry_weight: int
    checked_through: int
    failures: tuple[tuple[int, int, int, int], ...]

    @property
    def valid(self) -> bool:
        """Whether every finite comparison passed."""

        return not self.failures

    def require(self) -> None:
        """Raise on the first failed comparison."""

        if self.failures:
            raise ValidationError(f"carry-linear mismatch: {self.failures[0]}")


def classify_carry_linear_valuation(
    valuation: Callable[[int], int], prime: int, checked_through: int
) -> CarryLinearCertificate:
    """Recover the profile parameters and audit a finite range.

    The mathematical classification is unbounded. This routine is explicitly
    a finite certificate for a supplied callback.
    """

    if not _is_prime(prime):
        raise ValueError("prime must be prime")
    if checked_through < prime:
        raise ValueError("checked_through must be at least prime")
    if valuation(0) != 0:
        raise ValueError("valuation(0) must be zero")
    slope = int(valuation(1))
    carry_weight = int(valuation(prime) - prime * slope)
    failures = []
    for index in range(checked_through + 1):
        observed_value = int(valuation(index))
        expected_value = slope * index + carry_weight * factorial_valuation(index, prime)
        if observed_value != expected_value:
            failures.append((index, -1, observed_value, expected_value))
        for subindex in range(index + 1):
            observed = observed_value - int(valuation(subindex)) - int(
                valuation(index - subindex)
            )
            expected = carry_weight * binomial_valuation(index, subindex, prime)
            if observed != expected:
                failures.append((index, subindex, observed, expected))
    return CarryLinearCertificate(
        prime, slope, carry_weight, checked_through, tuple(failures)
    )


def base_digits(index: int, radix: int) -> tuple[int, ...]:
    """Return least-significant-first radix digits."""

    if index < 0:
        raise ValueError("index must be nonnegative")
    if radix < 2:
        raise ValueError("radix must be at least two")
    if index == 0:
        return (0,)
    digits = []
    remaining = index
    while remaining:
        remaining, digit = divmod(remaining, radix)
        digits.append(digit)
    return tuple(digits)


def digit_unit(index: int, prime: int, weights: Sequence[int]) -> int:
    """Evaluate a digit-multiplicative p-adic unit profile modulo ``prime``."""

    if not _is_prime(prime):
        raise ValueError("prime must be prime")
    normalized = tuple(int(value) % prime for value in weights)
    if len(normalized) != prime:
        raise ValueError("weights must contain one value per digit")
    if any(gcd(value, prime) != 1 for value in normalized):
        raise ValueError("every digit weight must be nonzero modulo prime")
    result = 1
    for digit in base_digits(index, prime):
        result = result * normalized[digit] % prime
    return result


def local_normalized_kernel(
    total_digit: int,
    selected_digit: int,
    prime: int,
    factorial_power: int,
    unit_weights: Sequence[int],
) -> int:
    """Return the no-carry local kernel for a digit-twisted factorial basis."""

    prime = _require_prime(prime)
    if not 0 <= selected_digit <= total_digit < prime:
        return 0
    if factorial_power < 1:
        raise ValueError("factorial_power must be positive")
    weights = tuple(int(value) % prime for value in unit_weights)
    if len(weights) != prime or any(gcd(value, prime) != 1 for value in weights):
        raise ValueError("unit_weights must be nonzero residues indexed by digits")
    denominator = weights[selected_digit] * weights[total_digit - selected_digit] % prime
    return (
        pow(comb(total_digit, selected_digit), factorial_power, prime)
        * weights[total_digit]
        * pow(denominator, -1, prime)
    ) % prime


def digit_factorized_normalized_kernel(
    index: int,
    subindex: int,
    prime: int,
    factorial_power: int,
    unit_weights: Sequence[int],
) -> int:
    """Multiply the local normalized kernels over all base-p digits."""

    prime = _require_prime(prime)
    if subindex < 0 or subindex > index:
        return 0
    n_digits = list(base_digits(index, prime))
    k_digits = list(base_digits(subindex, prime))
    length = max(len(n_digits), len(k_digits))
    n_digits.extend([0] * (length - len(n_digits)))
    k_digits.extend([0] * (length - len(k_digits)))
    result = 1
    for total_digit, selected_digit in zip(n_digits, k_digits):
        local = local_normalized_kernel(
            total_digit, selected_digit, prime, factorial_power, unit_weights
        )
        if local == 0:
            return 0
        result = result * local % prime
    return result
