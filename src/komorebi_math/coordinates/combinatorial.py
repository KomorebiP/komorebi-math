"""Exact combinatorial coordinate transforms for finite sequences."""

from __future__ import annotations

from fractions import Fraction
from math import comb, factorial
from collections.abc import Callable, Sequence

from ..exceptions import DimensionError, ValidationError
from .budget import DEFAULT_COORDINATE_BUDGET, CoordinateBudget
from .exact import Matrix
from .transform import CoordinateTransform


def _degree(value: int, *, name: str = "degree") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{name} must be a nonnegative integer")
    return value


def _size(value: int, *, name: str = "size") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DimensionError(f"{name} must be a positive integer")
    return value


def stirling_second(n: int, k: int) -> int:
    """Return the Stirling number of the second kind ``S(n,k)``."""

    _degree(n, name="n")
    _degree(k, name="k")
    if k > n:
        return 0
    row = [0 for _ in range(k + 1)]
    row[0] = 1
    for degree in range(1, n + 1):
        upper = min(degree, k)
        updated = [0 for _ in range(k + 1)]
        for blocks in range(1, upper + 1):
            updated[blocks] = row[blocks - 1] + blocks * row[blocks]
        row = updated
    return row[k]


def stirling_first_signed(n: int, k: int) -> int:
    """Return the signed Stirling number of the first kind ``s(n,k)``."""

    _degree(n, name="n")
    _degree(k, name="k")
    if k > n:
        return 0
    row = [0 for _ in range(k + 1)]
    row[0] = 1
    for degree in range(1, n + 1):
        upper = min(degree, k)
        updated = [0 for _ in range(k + 1)]
        for cycles in range(0, upper + 1):
            inherited = row[cycles - 1] if cycles > 0 else 0
            updated[cycles] = inherited - (degree - 1) * row[cycles]
        row = updated
    return row[k]


def stirling_first_unsigned(n: int, k: int) -> int:
    """Return the unsigned Stirling number of the first kind."""

    return (-1) ** (n - k) * stirling_first_signed(n, k)


def lah_number(n: int, k: int) -> int:
    """Return the unsigned Lah number ``L(n,k)``."""

    _degree(n, name="n")
    _degree(k, name="k")
    if n == 0 and k == 0:
        return 1
    if k == 0 or k > n:
        return 0
    return comb(n - 1, k - 1) * factorial(n) // factorial(k)


def _connection_matrix(
    degree: int,
    coefficient: Callable[[int, int], int],
) -> Matrix:
    return tuple(
        tuple(Fraction(coefficient(column, row)) for column in range(degree + 1))
        for row in range(degree + 1)
    )


def monomial_to_falling_factorial_transform(
    degree: int,
    *,
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> CoordinateTransform:
    """Return monomial coefficients in the falling-factorial basis."""

    bound = _degree(degree)
    budget.require_matrix(bound + 1, bound + 1, operation="Stirling second transform")
    return CoordinateTransform(
        f"monomial polynomial coefficients through degree {bound}",
        f"falling-factorial coefficients through degree {bound}",
        _connection_matrix(bound, stirling_second),
        "CC.C1",
        "Stirling transform from powers to falling factorials",
    )


def falling_factorial_to_monomial_transform(
    degree: int,
    *,
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> CoordinateTransform:
    """Return falling-factorial coefficients in the monomial basis."""

    bound = _degree(degree)
    budget.require_matrix(bound + 1, bound + 1, operation="signed Stirling first transform")
    return CoordinateTransform(
        f"falling-factorial coefficients through degree {bound}",
        f"monomial polynomial coefficients through degree {bound}",
        _connection_matrix(bound, stirling_first_signed),
        "CC.C2",
        "signed Stirling transform from falling factorials to powers",
    )


def monomial_to_rising_factorial_transform(
    degree: int,
    *,
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> CoordinateTransform:
    """Return monomial coefficients in the rising-factorial basis."""

    bound = _degree(degree)
    budget.require_matrix(bound + 1, bound + 1, operation="rising Stirling transform")
    matrix = _connection_matrix(
        bound,
        lambda n, k: (-1) ** (n - k) * stirling_second(n, k),
    )
    return CoordinateTransform(
        f"monomial polynomial coefficients through degree {bound}",
        f"rising-factorial coefficients through degree {bound}",
        matrix,
        "CC.C3",
        "signed Stirling transform from powers to rising factorials",
    )


def rising_factorial_to_monomial_transform(
    degree: int,
    *,
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> CoordinateTransform:
    """Return rising-factorial coefficients in the monomial basis."""

    bound = _degree(degree)
    budget.require_matrix(bound + 1, bound + 1, operation="unsigned Stirling transform")
    return CoordinateTransform(
        f"rising-factorial coefficients through degree {bound}",
        f"monomial polynomial coefficients through degree {bound}",
        _connection_matrix(bound, stirling_first_unsigned),
        "CC.C4",
        "unsigned Stirling transform from rising factorials to powers",
    )


def rising_to_falling_factorial_transform(
    degree: int,
    *,
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> CoordinateTransform:
    """Return rising-factorial coordinates in the falling-factorial basis."""

    bound = _degree(degree)
    budget.require_matrix(bound + 1, bound + 1, operation="Lah transform")
    return CoordinateTransform(
        f"rising-factorial coefficients through degree {bound}",
        f"falling-factorial coefficients through degree {bound}",
        _connection_matrix(bound, lah_number),
        "CC.C5",
        "Lah transform between rising and falling factorials",
    )


def binomial_transform(
    size: int,
    *,
    inverse: bool = False,
    source: str = "sequence values",
    target: str = "binomial transform",
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> CoordinateTransform:
    r"""Return the finite binomial transform or its inverse.

    The forward convention is ``b_n=sum_(k<=n) binom(n,k) a_k``.
    """

    count = _size(size)
    budget.require_matrix(count, count, operation="binomial transform")
    matrix = tuple(
        tuple(
            Fraction(((-1) ** (row - column) if inverse else 1) * comb(row, column))
            if column <= row
            else Fraction(0)
            for column in range(count)
        )
        for row in range(count)
    )
    if inverse:
        return CoordinateTransform(target, source, matrix, "CC.C7", "inverse finite binomial transform")
    return CoordinateTransform(source, target, matrix, "CC.C6", "finite binomial transform")


def forward_difference_transform(
    size: int,
    *,
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> CoordinateTransform:
    r"""Map sampled values ``f(0),...,f(N)`` to forward differences at zero."""

    count = _size(size)
    budget.require_matrix(count, count, operation="forward difference transform")
    matrix = tuple(
        tuple(
            Fraction((-1) ** (row - column) * comb(row, column))
            if column <= row
            else Fraction(0)
            for column in range(count)
        )
        for row in range(count)
    )
    return CoordinateTransform(
        f"samples at 0 through {count - 1}",
        f"forward differences at zero through order {count - 1}",
        matrix,
        "CC.C8",
        "Newton forward-difference analysis",
    )


def newton_evaluation_transform(
    size: int,
    *,
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> CoordinateTransform:
    """Map forward differences at zero to values at consecutive integers."""

    count = _size(size)
    transform = forward_difference_transform(count, budget=budget)
    return transform.inverse(theorem="CC.C9")


def mobius_integer(value: int) -> int:
    """Return the classical Möbius function of a positive integer."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError("Möbius arguments must be positive integers")
    n = value
    prime_count = 0
    prime = 2
    while prime * prime <= n:
        if n % prime == 0:
            n //= prime
            prime_count += 1
            if n % prime == 0:
                return 0
            while n % prime == 0:
                n //= prime
        prime += 1 if prime == 2 else 2
    if n > 1:
        prime_count += 1
    return -1 if prime_count % 2 else 1


def divisor_zeta_transform(
    size: int,
    *,
    inverse: bool = False,
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> CoordinateTransform:
    r"""Return zeta or Möbius coordinates on the divisibility poset.

    The forward map is ``b_n=sum_(d|n) a_d`` for indices ``1<=n<=size``.
    """

    count = _size(size)
    budget.require_matrix(count, count, operation="divisor zeta transform")
    matrix = tuple(
        tuple(
            Fraction(mobius_integer((row + 1) // (column + 1)) if inverse else 1)
            if (row + 1) % (column + 1) == 0
            else Fraction(0)
            for column in range(count)
        )
        for row in range(count)
    )
    if inverse:
        return CoordinateTransform(
            f"divisor sums through {count}",
            f"arithmetic coefficients through {count}",
            matrix,
            "CC.C11",
            "Möbius inversion on the divisibility poset",
        )
    return CoordinateTransform(
        f"arithmetic coefficients through {count}",
        f"divisor sums through {count}",
        matrix,
        "CC.C10",
        "zeta transform on the divisibility poset",
    )


def subset_zeta_transform(
    rank: int,
    *,
    inverse: bool = False,
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> CoordinateTransform:
    """Return the subset zeta or Möbius transform on a Boolean lattice."""

    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValidationError("Boolean-lattice rank must be a nonnegative integer")
    count = 1 << rank
    budget.require_matrix(count, count, operation="subset zeta transform")
    matrix = tuple(
        tuple(
            Fraction((-1) ** ((row ^ column).bit_count()) if inverse else 1)
            if column & ~row == 0
            else Fraction(0)
            for column in range(count)
        )
        for row in range(count)
    )
    if inverse:
        return CoordinateTransform(
            f"subset sums on rank {rank}",
            f"subset coefficients on rank {rank}",
            matrix,
            "CC.C13",
            "Boolean-lattice Möbius inversion",
        )
    return CoordinateTransform(
        f"subset coefficients on rank {rank}",
        f"subset sums on rank {rank}",
        matrix,
        "CC.C12",
        "Boolean-lattice zeta transform",
    )


def transform_sequence(
    values: Sequence[int | Fraction],
    transform: CoordinateTransform,
) -> tuple[Fraction, ...]:
    """Apply a coordinate transform to a finite exact sequence."""

    if len(values) != transform.source_dimension:
        raise DimensionError("sequence length does not match the transform source dimension")
    return transform.apply(values)


__all__ = [
    "binomial_transform",
    "divisor_zeta_transform",
    "falling_factorial_to_monomial_transform",
    "forward_difference_transform",
    "lah_number",
    "mobius_integer",
    "monomial_to_falling_factorial_transform",
    "monomial_to_rising_factorial_transform",
    "newton_evaluation_transform",
    "rising_factorial_to_monomial_transform",
    "rising_to_falling_factorial_transform",
    "stirling_first_signed",
    "stirling_first_unsigned",
    "stirling_second",
    "subset_zeta_transform",
    "transform_sequence",
]
