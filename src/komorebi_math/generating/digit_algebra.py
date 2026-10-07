"""Finite digit-kernel convolution laws for modular generating-function products.

The module isolates the mechanism behind Lucas-compatible custom bases.  It
works entirely with finite residue tables and therefore makes no claim that a
finite audit proves an unbounded congruence unless a separate theorem supplies
the digit factorization.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Iterable, Sequence


def base_digits(index: int, radix: int) -> tuple[int, ...]:
    """Return least-significant-first digits, with zero represented by ``(0,)``."""

    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("index must be a nonnegative integer")
    if isinstance(radix, bool) or not isinstance(radix, int) or radix < 2:
        raise ValueError("radix must be at least two")
    if index == 0:
        return (0,)
    digits: list[int] = []
    value = index
    while value:
        value, digit = divmod(value, radix)
        digits.append(digit)
    return tuple(digits)


@dataclass(frozen=True, slots=True)
class DigitKernel:
    r"""A lower-triangular local convolution table modulo an integer.

    For digit ``d`` and subdigit ``e``, ``table[d][e]`` is the local weight.
    The global no-borrow weight is the product of these local weights across
    radix digits.  Entries above the diagonal are represented implicitly by
    zero.
    """

    radix: int
    modulus: int
    table: tuple[tuple[int, ...], ...]
    name: str = "digit kernel"

    def __post_init__(self) -> None:
        if self.radix < 2:
            raise ValueError("radix must be at least two")
        if self.modulus < 2:
            raise ValueError("modulus must be at least two")
        if len(self.table) != self.radix:
            raise ValueError("digit-kernel table must have one row per digit")
        for digit, row in enumerate(self.table):
            if len(row) != digit + 1:
                raise ValueError("digit-kernel row d must contain d+1 entries")

    def local_weight(self, digit: int, subdigit: int) -> int:
        """Return one lower-triangular local weight."""

        if not 0 <= digit < self.radix:
            raise ValueError("digit is outside the kernel radix")
        if not 0 <= subdigit <= digit:
            return 0
        return self.table[digit][subdigit] % self.modulus

    def global_weight(self, index: int, subindex: int) -> int:
        """Lift the local table digitwise, returning zero on a borrow."""

        if subindex < 0 or subindex > index:
            return 0
        n_digits = base_digits(index, self.radix)
        k_digits = base_digits(subindex, self.radix)
        width = max(len(n_digits), len(k_digits))
        result = 1
        for position in range(width):
            digit = n_digits[position] if position < len(n_digits) else 0
            subdigit = k_digits[position] if position < len(k_digits) else 0
            if subdigit > digit:
                return 0
            result = result * self.local_weight(digit, subdigit) % self.modulus
        return result

    def lift(self, digit_values: Sequence[int], index: int) -> int:
        """Lift a local digit table to a digit-multiplicative sequence."""

        if len(digit_values) != self.radix:
            raise ValueError("digit value table must have one entry per radix digit")
        result = 1
        for digit in base_digits(index, self.radix):
            result = result * int(digit_values[digit]) % self.modulus
        return result

    def local_multiply(self, left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
        """Apply the finite local bilinear product to two digit tables."""

        if len(left) != self.radix or len(right) != self.radix:
            raise ValueError("local factors must each have one entry per digit")
        return tuple(
            sum(
                self.local_weight(digit, subdigit)
                * int(left[subdigit])
                * int(right[digit - subdigit])
                for subdigit in range(digit + 1)
            )
            % self.modulus
            for digit in range(self.radix)
        )

    def global_multiply(
        self,
        left: Sequence[int],
        right: Sequence[int],
        *,
        count: int,
    ) -> tuple[int, ...]:
        """Compute a finite global convolution using the lifted kernel."""

        if count < 0 or len(left) < count or len(right) < count:
            raise ValueError("global factors must cover the requested range")
        return tuple(
            sum(
                self.global_weight(index, subindex)
                * int(left[subindex])
                * int(right[index - subindex])
                for subindex in range(index + 1)
            )
            % self.modulus
            for index in range(count)
        )


def factorial_power_digit_kernel(power: int, prime: int) -> DigitKernel:
    r"""Return the Lucas table for ``binom(n,k)^power`` modulo ``prime``."""

    if isinstance(power, bool) or not isinstance(power, int) or power < 1:
        raise ValueError("power must be a positive integer")
    if isinstance(prime, bool) or not isinstance(prime, int) or prime < 2:
        raise ValueError("prime must be at least two")
    table = tuple(
        tuple(pow(comb(digit, subdigit), power, prime) for subdigit in range(digit + 1))
        for digit in range(prime)
    )
    return DigitKernel(prime, prime, table, f"factorial-power-{power} mod {prime}")


def lift_digit_sequence(
    digit_values: Sequence[int],
    *,
    radix: int,
    modulus: int,
    count: int,
) -> tuple[int, ...]:
    """Lift a local table without coupling it to a particular kernel."""

    identity = DigitKernel(
        radix,
        modulus,
        tuple(tuple(1 for _ in range(digit + 1)) for digit in range(radix)),
        "identity lift",
    )
    return tuple(identity.lift(digit_values, index) for index in range(count))


@dataclass(frozen=True, slots=True)
class DigitAlgebraAudit:
    """Finite evidence for digitwise factorization and convolution closure."""

    radix: int
    modulus: int
    power: int
    tested_weights: int
    weight_failures: int
    tested_coefficients: int
    closure_failures: int
    local_product: tuple[int, ...]


def audit_factorial_power_digit_algebra(
    *,
    prime: int,
    power: int,
    left_digits: Iterable[int],
    right_digits: Iterable[int],
    max_digits: int = 3,
) -> DigitAlgebraAudit:
    r"""Audit Lucas-compatible closure below ``prime^max_digits``.

    The theorem being tested is

    ``(a star_s b)_n = prod_i (a star_s b)_(n_i) (mod p)``

    whenever ``a_n`` and ``b_n`` are lifted products of their base-``p``
    digit values and ``star_s`` uses ``binom(n,k)^s``.
    """

    if max_digits < 1:
        raise ValueError("max_digits must be positive")
    kernel = factorial_power_digit_kernel(power, prime)
    left_local = tuple(int(value) % prime for value in left_digits)
    right_local = tuple(int(value) % prime for value in right_digits)
    if len(left_local) != prime or len(right_local) != prime:
        raise ValueError("local factors must have exactly prime entries")
    if left_local[0] != 1 or right_local[0] != 1:
        raise ValueError("unbounded digit-multiplicative sequences require zeroth digit value one")
    count = prime**max_digits
    left = tuple(kernel.lift(left_local, index) for index in range(count))
    right = tuple(kernel.lift(right_local, index) for index in range(count))

    weight_failures = 0
    tested_weights = 0
    for index in range(count):
        for subindex in range(index + 1):
            expected = pow(comb(index, subindex), power, prime)
            if kernel.global_weight(index, subindex) != expected:
                weight_failures += 1
            tested_weights += 1

    local_product = kernel.local_multiply(left_local, right_local)
    global_product = kernel.global_multiply(left, right, count=count)
    closure_failures = sum(
        global_product[index] != kernel.lift(local_product, index)
        for index in range(count)
    )
    return DigitAlgebraAudit(
        prime,
        prime,
        power,
        tested_weights,
        weight_failures,
        count,
        closure_failures,
        local_product,
    )


__all__ = [
    "DigitAlgebraAudit",
    "DigitKernel",
    "audit_factorial_power_digit_algebra",
    "base_digits",
    "factorial_power_digit_kernel",
    "lift_digit_sequence",
]
