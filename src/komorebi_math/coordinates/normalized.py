"""Coordinate changes for normalized monomial generating functions."""

from __future__ import annotations

from fractions import Fraction
from typing import Iterable, Sequence

from ..exceptions import ValidationError
from ..generating import (
    WeightedMonomialBasis,
    exponential_basis,
    fibonacci_factorial_basis,
    ordinary_basis,
)
from .budget import DEFAULT_COORDINATE_BUDGET, CoordinateBudget
from .exact import Scalar, Vector, as_vector
from .transform import CoordinateTransform

NormalizedBasis = WeightedMonomialBasis


def normalization_change_transform(
    source: WeightedMonomialBasis,
    target: WeightedMonomialBasis,
    degree: int,
    *,
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> CoordinateTransform:
    r"""Return the diagonal map preserving the underlying formal series.

    If ``F=sum a_n z^n/gamma_n=sum b_n z^n/eta_n``, then
    ``b_n=(eta_n/gamma_n)a_n``.
    """

    if not isinstance(source, WeightedMonomialBasis) or not isinstance(
        target, WeightedMonomialBasis
    ):
        raise ValidationError("source and target must be normalized monomial bases")
    budget.require_degree(degree, operation="normalization coordinate change")
    budget.require_matrix(
        degree + 1,
        degree + 1,
        operation="normalization coordinate change",
    )
    matrix = tuple(
        tuple(
            target.gamma(row) / source.gamma(row)
            if row == column
            else Fraction(0)
            for column in range(degree + 1)
        )
        for row in range(degree + 1)
    )
    return CoordinateTransform(
        f"{source.name} coefficients through degree {degree}",
        f"{target.name} coefficients through degree {degree}",
        matrix,
        "CC.N1",
        "diagonal normalized basis change",
    )


def convert_normalized_coefficients(
    coefficients: Sequence[Scalar],
    source: WeightedMonomialBasis,
    target: WeightedMonomialBasis,
    *,
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> Vector:
    """Convert a finite coefficient vector between normalized bases."""

    data = as_vector(coefficients)
    if not data:
        return ()
    return normalization_change_transform(
        source,
        target,
        len(data) - 1,
        budget=budget,
    ).apply(data)


def apply_lowering_polynomial(
    coefficients: Sequence[Scalar],
    polynomial: Iterable[Scalar],
) -> Vector:
    r"""Apply a polynomial in the canonical lowering operator.

    If ``q(x)=sum_j q_j x^j``, the output coefficient at index ``n`` is
    ``sum_j q_j a_(n+j)`` whenever all terms lie in the finite prefix.
    """

    data = as_vector(coefficients)
    operator = as_vector(polynomial)
    if not operator:
        raise ValidationError("the lowering polynomial cannot be empty")
    degree = len(operator) - 1
    if len(data) <= degree:
        return ()
    return tuple(
        sum(
            (
                operator[shift] * data[index + shift]
                for shift in range(degree + 1)
            ),
            Fraction(0),
        )
        for index in range(len(data) - degree)
    )


__all__ = [
    "NormalizedBasis",
    "apply_lowering_polynomial",
    "convert_normalized_coefficients",
    "exponential_basis",
    "fibonacci_factorial_basis",
    "normalization_change_transform",
    "ordinary_basis",
]
