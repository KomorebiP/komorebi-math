"""Exact benchmark and custom sequence families."""

from __future__ import annotations

from fractions import Fraction
from math import comb, factorial
from typing import Callable, Iterable


def fibonacci_numbers(count: int) -> tuple[int, ...]:
    """Return the first ``count`` Fibonacci numbers beginning with zero."""

    if count < 0:
        raise ValueError("count must be nonnegative")
    values: list[int] = []
    a, b = 0, 1
    for _ in range(count):
        values.append(a)
        a, b = b, a + b
    return tuple(values)


def central_binomial_numbers(count: int) -> tuple[int, ...]:
    r"""Return ``binomial(2*n, n)`` for ``0 <= n < count``."""

    return tuple(comb(2 * n, n) for n in range(count))


def catalan_numbers(count: int) -> tuple[int, ...]:
    """Return the first ``count`` Catalan numbers."""

    return tuple(comb(2 * n, n) // (n + 1) for n in range(count))


def bell_numbers(count: int) -> tuple[int, ...]:
    """Return the first ``count`` ordinary Bell numbers by their recurrence."""

    if count < 0:
        raise ValueError("count must be nonnegative")
    if count == 0:
        return ()
    values = [1]
    while len(values) < count:
        n = len(values) - 1
        values.append(sum(comb(n, k) * values[k] for k in range(n + 1)))
    return tuple(values)


def partition_numbers(count: int) -> tuple[int, ...]:
    """Return partition numbers through Euler's pentagonal recurrence."""

    if count < 0:
        raise ValueError("count must be nonnegative")
    if count == 0:
        return ()
    values = [1]
    for n in range(1, count):
        total = 0
        k = 1
        while True:
            g1 = k * (3 * k - 1) // 2
            g2 = k * (3 * k + 1) // 2
            if g1 > n:
                break
            sign = 1 if k % 2 else -1
            total += sign * values[n - g1]
            if g2 <= n:
                total += sign * values[n - g2]
            k += 1
        values.append(total)
    return tuple(values)


def apery_numbers(count: int) -> tuple[int, ...]:
    """Return the Apéry numbers for zeta(3) by their finite binomial sum."""

    if count < 0:
        raise ValueError("count must be nonnegative")
    return tuple(sum(comb(n, k) ** 2 * comb(n + k, k) ** 2 for k in range(n + 1)) for n in range(count))


def signed_euler_numbers(count: int) -> tuple[int, ...]:
    r"""Return coefficients ``E_n`` of ``sech(z)=sum E_n z^n/n!``."""

    if count < 0:
        raise ValueError("count must be nonnegative")
    values = [0 for _ in range(count)]
    if count == 0:
        return ()
    values[0] = 1
    # cosh(z) * sech(z) = 1.  Odd coefficients vanish.
    for n in range(1, count):
        values[n] = -sum(
            comb(n, k) * values[n - k]
            for k in range(2, n + 1, 2)
        )
    return tuple(values)


def bernoulli_numbers(count: int) -> tuple[Fraction, ...]:
    """Return Bernoulli numbers with ``B_1=-1/2``."""

    if count < 0:
        raise ValueError("count must be nonnegative")
    result: list[Fraction] = []
    for m in range(count):
        row = [Fraction(1, k + 1) for k in range(m + 1)]
        for j in range(m, 0, -1):
            for k in range(j):
                row[k] = k + 1 and Fraction(k + 1) * (row[k] - row[k + 1])
        value = row[0]
        if m == 1:
            value = -value
        result.append(value)
    return tuple(result)


def generalized_bell_numbers(power: int, count: int) -> tuple[int, ...]:
    r"""Return the factorial-power Bell family.

    The defining generating function is

    ``exp(sum_(m>=1) z^m/(m!)^power) = sum B_n^(power) z^n/(n!)^power``.

    Exact recurrence:

    ``B_n = sum_(m=1)^n C(n-1,m-1) C(n,m)^(power-1) B_(n-m)``.
    """

    if isinstance(power, bool) or not isinstance(power, int) or power < 1:
        raise ValueError("power must be a positive integer")
    if count < 0:
        raise ValueError("count must be nonnegative")
    if count == 0:
        return ()
    values = [1]
    for n in range(1, count):
        values.append(
            sum(
                comb(n - 1, m - 1) * comb(n, m) ** (power - 1) * values[n - m]
                for m in range(1, n + 1)
            )
        )
    return tuple(values)


def generalized_bell_polynomials(power: int, count: int) -> tuple[tuple[int, ...], ...]:
    r"""Return coefficient rows of factorial-power Bell polynomials.

    ``B_n^(power)(x)=sum_k S_power(n,k) x^k`` and

    ``exp(x*sum_(m>=1) z^m/(m!)^power)
      =sum_n B_n^(power)(x) z^n/(n!)^power``.
    """

    if isinstance(power, bool) or not isinstance(power, int) or power < 1:
        raise ValueError("power must be a positive integer")
    if count < 0:
        raise ValueError("count must be nonnegative")
    if count == 0:
        return ()
    rows: list[list[int]] = [[1]]
    for n in range(1, count):
        row = [0 for _ in range(n + 1)]
        for m in range(1, n + 1):
            weight = comb(n - 1, m - 1) * comb(n, m) ** (power - 1)
            previous = rows[n - m]
            for k, value in enumerate(previous):
                row[k + 1] += weight * value
        rows.append(row)
    return tuple(tuple(row) for row in rows)



def weighted_generalized_bell_polynomials(
    power: int,
    count: int,
    weights: Iterable[int | Fraction] | Callable[[int], int | Fraction],
) -> tuple[tuple[Fraction, ...], ...]:
    r"""Return factorial-power Bell polynomials with custom block weights.

    The defining series is

    ``exp(x*sum_(m>=1) c_m z^m/(m!)^power)``

    and the normalized coefficient rows satisfy

    ``P_n(x)=x*sum_(m=1)^n c_m C(n-1,m-1)
       C(n,m)^(power-1) P_(n-m)(x)``.

    ``weights`` may be a callable on positive indices or an iterable whose
    zeroth entry is ignored.  Rational weights are supported exactly.
    """

    if isinstance(power, bool) or not isinstance(power, int) or power < 1:
        raise ValueError("power must be a positive integer")
    if count < 0:
        raise ValueError("count must be nonnegative")
    if count == 0:
        return ()
    if callable(weights):
        weight = lambda index: Fraction(weights(index))
    else:
        table = tuple(Fraction(value) for value in weights)

        def weight(index: int) -> Fraction:
            if index >= len(table):
                raise ValueError("weight table is too short for the requested coefficient range")
            return table[index]

    rows: list[list[Fraction]] = [[Fraction(1)]]
    for n in range(1, count):
        row = [Fraction(0) for _ in range(n + 1)]
        for m in range(1, n + 1):
            coefficient = weight(m)
            if coefficient == 0:
                continue
            kernel = coefficient * comb(n - 1, m - 1) * comb(n, m) ** (power - 1)
            for k, value in enumerate(rows[n - m]):
                row[k + 1] += kernel * value
        rows.append(row)
    return tuple(tuple(row) for row in rows)


def weighted_generalized_bell_numbers(
    power: int,
    count: int,
    weights: Iterable[int | Fraction] | Callable[[int], int | Fraction],
) -> tuple[Fraction, ...]:
    """Evaluate the custom factorial-power Bell polynomials at ``x=1``."""

    return tuple(sum(row, Fraction(0)) for row in weighted_generalized_bell_polynomials(power, count, weights))

def generalized_bell_hypergeometric_inner(power: int, degree: int) -> tuple[Fraction, ...]:
    r"""Return ``H(z)=sum_(m>=1) z^m/(m!)^power`` through ``degree``."""

    if power < 1 or degree < 0:
        raise ValueError("invalid power or degree")
    return tuple(Fraction(0) if n == 0 else Fraction(1, factorial(n) ** power) for n in range(degree + 1))


def finite_differences(values: Iterable[int | Fraction]) -> tuple[tuple[Fraction, ...], ...]:
    """Return the complete exact forward-difference table."""

    row = tuple(Fraction(value) for value in values)
    table = [row]
    while len(row) > 1:
        row = tuple(row[index + 1] - row[index] for index in range(len(row) - 1))
        table.append(row)
    return tuple(table)


__all__ = [
    "apery_numbers",
    "bell_numbers",
    "bernoulli_numbers",
    "catalan_numbers",
    "central_binomial_numbers",
    "fibonacci_numbers",
    "finite_differences",
    "generalized_bell_hypergeometric_inner",
    "generalized_bell_numbers",
    "generalized_bell_polynomials",
    "weighted_generalized_bell_numbers",
    "weighted_generalized_bell_polynomials",
    "partition_numbers",
    "signed_euler_numbers",
]
