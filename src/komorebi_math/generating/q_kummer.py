"""Mixed-radix Kummer formulas for integer Gaussian binomial coefficients.

The compiler covers every prime and every integer base greater than one.
Bases divisible by the selected prime give valuation zero. Odd primes use the
multiplicative-order block, while the prime two uses a parity block for odd
bases.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd, isqrt

from .padic_units import binomial_valuation, integer_valuation


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


def multiplicative_order_mod_prime(value: int, prime: int) -> int:
    """Return the multiplicative order of ``value`` modulo ``prime``."""

    if not _is_prime(prime):
        raise ValueError("prime must be prime")
    residue = value % prime
    if gcd(residue, prime) != 1:
        raise ValueError("value must be coprime to prime")
    running = 1
    for order in range(1, prime):
        running = running * residue % prime
        if running == 1:
            return order
    raise ArithmeticError("multiplicative order was not found")


def q_integer(index: int, base: int) -> int:
    """Return the integer q-analogue ``[index]_base``."""

    if index < 0:
        raise ValueError("index must be nonnegative")
    if base <= 1:
        raise ValueError("base must exceed one")
    if index == 0:
        return 0
    return (base**index - 1) // (base - 1)


def q_factorial(index: int, base: int) -> int:
    """Return the integer q-factorial."""

    if index < 0:
        raise ValueError("index must be nonnegative")
    result = 1
    for value in range(1, index + 1):
        result *= q_integer(value, base)
    return result


def gaussian_binomial(index: int, subindex: int, base: int) -> int:
    """Return the integer Gaussian binomial evaluated at ``base``."""

    if subindex < 0 or subindex > index:
        return 0
    selected = min(subindex, index - subindex)
    result = 1
    for step in range(1, selected + 1):
        result *= q_integer(index - selected + step, base)
        divisor = q_integer(step, base)
        if result % divisor:
            raise ArithmeticError("Gaussian binomial divisibility invariant failed")
        result //= divisor
    return result


@dataclass(frozen=True)
class QKummerDecomposition:
    """Exact mixed-radix valuation decomposition for a Gaussian binomial."""

    index: int
    subindex: int
    base: int
    prime: int
    order: int
    lift_constant: int
    quotient_index: int
    quotient_subindex: int
    residue_carry: int
    quotient_carries: int
    endpoint_valuation: int
    valuation: int

    def survives(self, exponent: int, power: int = 1) -> bool:
        """Whether the powered Gaussian binomial survives modulo ``p**r``."""

        if exponent < 1 or power < 1:
            raise ValueError("exponent and power must be positive")
        return power * self.valuation < exponent


def q_binomial_valuation_decomposition(
    index: int, subindex: int, base: int, prime: int
) -> QKummerDecomposition:
    """Return the exact mixed-radix valuation decomposition.

    If ``prime`` divides ``base``, every Gaussian binomial is a unit modulo
    ``prime`` and the valuation is zero. For an odd prime coprime to ``base``,
    the block length is ``ord_prime(base)`` and the usual order-block formula
    applies. For ``prime == 2`` and odd ``base``, the block length is two and
    the lift constant is ``v_2(base + 1)``.
    """

    if not _is_prime(prime):
        raise ValueError("prime must be prime")
    if base <= 1:
        raise ValueError("base must exceed one")
    if subindex < 0 or subindex > index:
        raise ValueError("subindex must lie between zero and index")

    if base % prime == 0:
        return QKummerDecomposition(
            index,
            subindex,
            base,
            prime,
            1,
            0,
            index,
            subindex,
            0,
            0,
            0,
            0,
        )

    if prime == 2:
        order = 2
        lift_constant = integer_valuation(base + 1, 2)
    else:
        order = multiplicative_order_mod_prime(base, prime)
        lift_constant = integer_valuation(base**order - 1, prime)
        if (base - 1) % prime == 0:
            lift_constant -= integer_valuation(base - 1, prime)

    quotient_index, index_residue = divmod(index, order)
    quotient_subindex, subindex_residue = divmod(subindex, order)
    residue_carry = int(subindex_residue > index_residue)
    quotient_carries = binomial_valuation(
        quotient_index, quotient_subindex, prime
    )
    endpoint_valuation = 0
    if residue_carry:
        endpoint_valuation = integer_valuation(
            quotient_index - quotient_subindex, prime
        )
    valuation = quotient_carries + residue_carry * (
        lift_constant + endpoint_valuation
    )
    return QKummerDecomposition(
        index,
        subindex,
        base,
        prime,
        order,
        lift_constant,
        quotient_index,
        quotient_subindex,
        residue_carry,
        quotient_carries,
        endpoint_valuation,
        valuation,
    )


def q_binomial_valuation_direct(
    index: int, subindex: int, base: int, prime: int
) -> int:
    """Compute the valuation directly from the integer Gaussian binomial."""

    value = gaussian_binomial(index, subindex, base)
    if value == 0:
        raise ValueError("direct valuation is undefined for zero")
    return integer_valuation(value, prime)


def q_binomial_support(
    index: int,
    base: int,
    prime: int,
    exponent: int,
    power: int = 1,
) -> tuple[int, ...]:
    """Return subindices surviving in ``GaussianBinomial**power mod p**exponent``."""

    if exponent < 1 or power < 1:
        raise ValueError("exponent and power must be positive")
    return tuple(
        subindex
        for subindex in range(index + 1)
        if q_binomial_valuation_decomposition(index, subindex, base, prime).survives(
            exponent, power
        )
    )
