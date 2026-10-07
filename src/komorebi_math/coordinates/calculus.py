"""Operator matrices in arbitrary polynomial coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Sequence

from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from .budget import DEFAULT_COORDINATE_BUDGET, CoordinateBudget
from .classical import operator_matrix_between_families, polynomial_compose_affine, polynomial_subtract
from .exact import Matrix, Scalar, Vector, as_matrix, as_vector, matrix_multiply, matrix_shape, matrix_vector_multiply
from .polynomials import (
    Polynomial,
    PolynomialFamily,
    connection_transform,
    polynomial_derivative,
    polynomial_multiply,
    polynomial_scale,
)
from .tensor import transport_rectangular_operator


def polynomial_integral(polynomial: Sequence[Scalar], constant: Scalar = 0) -> Polynomial:
    """Return one exact antiderivative."""

    values = as_vector(polynomial)
    return (Fraction(constant),) + tuple(values[index] / Fraction(index + 1) for index in range(len(values)))


@dataclass(frozen=True, slots=True)
class PolynomialCoordinateOperator:
    """A finite polynomial operator matrix between graded family truncations."""

    name: str
    source_family: str
    target_family: str
    source_degree: int
    target_degree: int
    matrix: Matrix
    theorem: str = "CC.O1"

    def __post_init__(self) -> None:
        matrix = as_matrix(self.matrix)
        rows, columns = matrix_shape(matrix)
        if columns != self.source_degree + 1 or rows != self.target_degree + 1:
            raise DimensionError("operator matrix shape does not match the declared degrees")
        if not str(self.name).strip():
            raise ValidationError("coordinate-operator names must be nonempty")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "source_family", str(self.source_family))
        object.__setattr__(self, "target_family", str(self.target_family))
        object.__setattr__(self, "theorem", str(self.theorem))

    def apply(self, coefficients: Sequence[Scalar]) -> Vector:
        """Apply the operator to one coordinate vector."""

        return matrix_vector_multiply(self.matrix, coefficients)

    @property
    def square(self) -> bool:
        """Return whether source and target truncations have equal dimension."""

        return self.source_degree == self.target_degree

    def compose(self, following: "PolynomialCoordinateOperator") -> "PolynomialCoordinateOperator":
        """Compose this operator with a following operator."""

        if not isinstance(following, PolynomialCoordinateOperator):
            raise ValidationError("following must be a PolynomialCoordinateOperator")
        if self.target_family != following.source_family:
            raise ValidationError("operator family names do not compose")
        if self.target_degree != following.source_degree:
            raise DimensionError("operator truncation degrees do not compose")
        return PolynomialCoordinateOperator(
            f"{following.name} after {self.name}",
            self.source_family,
            following.target_family,
            self.source_degree,
            following.target_degree,
            matrix_multiply(following.matrix, self.matrix),
            f"{self.theorem} then {following.theorem}",
        )

    def certificate(self) -> Certificate:
        """Return an exact finite shape certificate."""

        rows, columns = matrix_shape(self.matrix)
        return Certificate(
            valid=rows == self.target_degree + 1 and columns == self.source_degree + 1,
            exactness=Exactness.EXACT,
            assumptions_checked=("finite graded truncations", "exact rational coefficients"),
            diagnostics={
                "source_degree": self.source_degree,
                "target_degree": self.target_degree,
                "shape": (rows, columns),
            },
            theorem_id=self.theorem,
            message="polynomial coordinate operator validated",
        )


def polynomial_operator(
    source: PolynomialFamily,
    target: PolynomialFamily,
    source_degree: int,
    target_degree: int,
    rule: Callable[[Polynomial], Polynomial],
    *,
    name: str,
    theorem: str = "CC.O1",
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> PolynomialCoordinateOperator:
    """Compile a polynomial rule into coordinates of two families."""

    matrix = operator_matrix_between_families(
        source,
        target,
        source_degree,
        target_degree,
        rule,
        name=name,
        budget=budget,
    )
    return PolynomialCoordinateOperator(
        name,
        source.name,
        target.name,
        source_degree,
        target_degree,
        matrix,
        theorem,
    )


def derivative_coordinate_operator(
    family: PolynomialFamily,
    degree: int,
    *,
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> PolynomialCoordinateOperator:
    """Return differentiation in one graded polynomial family."""

    return polynomial_operator(
        family,
        family,
        degree,
        degree,
        polynomial_derivative,
        name="derivative",
        theorem="CC.O2",
        budget=budget,
    )


def integral_coordinate_operator(
    family: PolynomialFamily,
    degree: int,
    *,
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> PolynomialCoordinateOperator:
    """Return zero-constant integration in one graded polynomial family."""

    return polynomial_operator(
        family,
        family,
        degree,
        degree + 1,
        polynomial_integral,
        name="zero-constant integration",
        theorem="CC.O3",
        budget=budget,
    )


def multiplication_coordinate_operator(
    family: PolynomialFamily,
    degree: int,
    multiplier: Sequence[Scalar],
    *,
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> PolynomialCoordinateOperator:
    """Return multiplication by a fixed exact polynomial."""

    polynomial = as_vector(multiplier)
    if not polynomial:
        raise ValidationError("multiplier polynomial must be nonempty")
    multiplier_degree = len(polynomial) - 1
    return polynomial_operator(
        family,
        family,
        degree,
        degree + multiplier_degree,
        lambda value: polynomial_multiply(value, polynomial),
        name=f"multiplication by {tuple(polynomial)}",
        theorem="CC.O4",
        budget=budget,
    )


def affine_substitution_coordinate_operator(
    family: PolynomialFamily,
    degree: int,
    *,
    scale: Scalar = 1,
    shift: Scalar = 0,
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> PolynomialCoordinateOperator:
    """Return the degree-preserving substitution ``p(x)->p(scale*x+shift)``."""

    return polynomial_operator(
        family,
        family,
        degree,
        degree,
        lambda value: polynomial_compose_affine(value, scale=scale, shift=shift),
        name=f"affine substitution x -> {scale}x+{shift}",
        theorem="CC.O5",
        budget=budget,
    )


def finite_difference_coordinate_operator(
    family: PolynomialFamily,
    degree: int,
    *,
    step: Scalar = 1,
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> PolynomialCoordinateOperator:
    r"""Return ``Delta_h p(x)=p(x+h)-p(x)`` in one family."""

    h = Fraction(step)
    if h == 0:
        raise ValidationError("finite-difference step must be nonzero")
    return polynomial_operator(
        family,
        family,
        degree,
        degree,
        lambda value: polynomial_subtract(
            polynomial_compose_affine(value, scale=1, shift=h),
            value,
        ),
        name=f"forward difference with step {h}",
        theorem="CC.O6",
        budget=budget,
    )


def transport_polynomial_operator(
    operator: PolynomialCoordinateOperator,
    source_family: PolynomialFamily,
    target_family: PolynomialFamily,
    *,
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> PolynomialCoordinateOperator:
    """Transport an operator between two graded polynomial-family coordinates."""

    if operator.source_family != source_family.name or operator.target_family != source_family.name:
        raise ValidationError("operator must act within the declared source family")
    domain_change = connection_transform(
        source_family,
        target_family,
        operator.source_degree,
        budget=budget,
    )
    codomain_change = connection_transform(
        source_family,
        target_family,
        operator.target_degree,
        budget=budget,
    )
    matrix = transport_rectangular_operator(operator.matrix, domain_change, codomain_change)
    return PolynomialCoordinateOperator(
        f"{operator.name} in {target_family.name}",
        target_family.name,
        target_family.name,
        operator.source_degree,
        operator.target_degree,
        matrix,
        "CC.O7",
    )


def operator_commutator(left: Matrix, right: Matrix) -> Matrix:
    """Return the exact commutator ``left*right-right*left``."""

    if matrix_shape(left) != matrix_shape(right):
        raise DimensionError("commutator matrices must have equal shape")
    rows, columns = matrix_shape(left)
    if rows != columns:
        raise DimensionError("commutators require square matrices")
    left_right = matrix_multiply(left, right)
    right_left = matrix_multiply(right, left)
    return tuple(
        tuple(a - b for a, b in zip(left_row, right_row, strict=True))
        for left_row, right_row in zip(left_right, right_left, strict=True)
    )


__all__ = [
    "PolynomialCoordinateOperator",
    "affine_substitution_coordinate_operator",
    "derivative_coordinate_operator",
    "finite_difference_coordinate_operator",
    "integral_coordinate_operator",
    "multiplication_coordinate_operator",
    "operator_commutator",
    "polynomial_integral",
    "polynomial_operator",
    "transport_polynomial_operator",
]
