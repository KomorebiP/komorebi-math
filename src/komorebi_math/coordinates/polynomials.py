"""Connection coefficients for graded polynomial families."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb
from typing import Callable, Sequence

from ..exceptions import DimensionError, ValidationError
from .budget import DEFAULT_COORDINATE_BUDGET, CoordinateBudget
from .exact import (
    Matrix,
    Scalar,
    Vector,
    as_vector,
    inverse_matrix,
    matrix_from_columns,
    matrix_multiply,
    matrix_shape,
)
from .transform import CoordinateTransform

Polynomial = tuple[Fraction, ...]


def _trim(polynomial: Sequence[Scalar]) -> Polynomial:
    values = list(as_vector(polynomial))
    if not values:
        return (Fraction(0),)
    while len(values) > 1 and values[-1] == 0:
        values.pop()
    return tuple(values)


def polynomial_add(left: Sequence[Scalar], right: Sequence[Scalar]) -> Polynomial:
    """Add exact polynomials stored in ascending degree order."""

    a = as_vector(left)
    b = as_vector(right)
    size = max(len(a), len(b))
    return _trim(
        tuple(
            (a[index] if index < len(a) else 0)
            + (b[index] if index < len(b) else 0)
            for index in range(size)
        )
    )


def polynomial_scale(polynomial: Sequence[Scalar], scalar: Scalar) -> Polynomial:
    """Scale an exact polynomial."""

    factor = Fraction(scalar)
    return _trim(tuple(factor * value for value in as_vector(polynomial)))


def polynomial_shift(polynomial: Sequence[Scalar], steps: int = 1) -> Polynomial:
    """Multiply a polynomial by ``x^steps``."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValidationError("steps must be a nonnegative integer")
    return tuple(Fraction(0) for _ in range(steps)) + _trim(polynomial)


def polynomial_multiply(left: Sequence[Scalar], right: Sequence[Scalar]) -> Polynomial:
    """Multiply exact polynomials."""

    a = as_vector(left)
    b = as_vector(right)
    output = [Fraction(0) for _ in range(len(a) + len(b) - 1)]
    for i, left_value in enumerate(a):
        for j, right_value in enumerate(b):
            output[i + j] += left_value * right_value
    return _trim(output)


def polynomial_evaluate(polynomial: Sequence[Scalar], value: Scalar) -> Fraction:
    """Evaluate an exact polynomial by Horner's method."""

    x = Fraction(value)
    result = Fraction(0)
    for coefficient in reversed(as_vector(polynomial)):
        result = result * x + coefficient
    return result


def polynomial_derivative(polynomial: Sequence[Scalar]) -> Polynomial:
    """Differentiate an exact polynomial."""

    data = as_vector(polynomial)
    if len(data) <= 1:
        return (Fraction(0),)
    return _trim(tuple(index * data[index] for index in range(1, len(data))))


@dataclass(frozen=True, slots=True)
class PolynomialFamily:
    """A graded polynomial basis with one nonzero leading term per degree."""

    name: str
    rule: Callable[[int], Polynomial]

    def polynomial(self, degree: int) -> Polynomial:
        """Return the family polynomial of one degree."""

        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
            raise ValidationError("degree must be a nonnegative integer")
        polynomial = _trim(self.rule(degree))
        if len(polynomial) != degree + 1 or polynomial[-1] == 0:
            raise ValidationError("family polynomials must have exact graded degree")
        return polynomial

    def coefficient_matrix(self, degree: int) -> Matrix:
        r"""Return columns of basis polynomials in monomial coordinates."""

        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
            raise ValidationError("degree must be a nonnegative integer")
        columns: list[Vector] = []
        for basis_degree in range(degree + 1):
            polynomial = self.polynomial(basis_degree)
            columns.append(polynomial + tuple(Fraction(0) for _ in range(degree + 1 - len(polynomial))))
        return matrix_from_columns(columns)


def monomial_polynomial_family() -> PolynomialFamily:
    """Return the monomial family ``1,x,x^2,...``."""

    return PolynomialFamily(
        "monomials",
        lambda degree: tuple(Fraction(int(index == degree)) for index in range(degree + 1)),
    )


def fibonacci_polynomial(index: int) -> Polynomial:
    r"""Return the standard Fibonacci polynomial ``F_index(x)``.

    ``F_0=0``, ``F_1=1``, and ``F_(n+1)=x F_n+F_(n-1)``.
    """

    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValidationError("index must be a nonnegative integer")
    if index == 0:
        return (Fraction(0),)
    if index == 1:
        return (Fraction(1),)
    previous = (Fraction(0),)
    current = (Fraction(1),)
    for _ in range(1, index):
        previous, current = current, polynomial_add(polynomial_shift(current), previous)
    return current


def fibonacci_polynomial_family() -> PolynomialFamily:
    r"""Return the graded basis ``G_n(x)=F_(n+1)(x)``."""

    return PolynomialFamily("Fibonacci polynomials", lambda degree: fibonacci_polynomial(degree + 1))


def euler_polynomial_at_zero(index: int) -> Fraction:
    r"""Return ``E_index(0)`` for ``2e^(xt)/(e^t+1)``."""

    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValidationError("index must be a nonnegative integer")
    values = [Fraction(1)]
    for n in range(1, index + 1):
        values.append(
            -sum((Fraction(comb(n, k)) * values[k] for k in range(n)), Fraction(0)) / 2
        )
    return values[index]


def euler_polynomial(degree: int) -> Polynomial:
    r"""Return the Euler polynomial from ``2e^(xt)/(e^t+1)``."""

    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
        raise ValidationError("degree must be a nonnegative integer")
    return tuple(
        Fraction(comb(degree, power)) * euler_polynomial_at_zero(degree - power)
        for power in range(degree + 1)
    )


def euler_polynomial_family() -> PolynomialFamily:
    """Return the graded Euler polynomial family."""

    return PolynomialFamily("Euler polynomials", euler_polynomial)


def connection_transform(
    source: PolynomialFamily,
    target: PolynomialFamily,
    degree: int,
    *,
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> CoordinateTransform:
    r"""Return target coordinates of source basis vectors through one degree.

    If ``M_source`` and ``M_target`` contain basis polynomials as monomial
    coordinate columns, then ``C=M_target^(-1) M_source``.
    """

    if not isinstance(source, PolynomialFamily) or not isinstance(target, PolynomialFamily):
        raise ValidationError("source and target must be PolynomialFamily objects")
    budget.require_degree(degree, operation="polynomial family connection")
    budget.require_matrix(degree + 1, degree + 1, operation="polynomial family connection")
    source_matrix = source.coefficient_matrix(degree)
    target_matrix = target.coefficient_matrix(degree)
    connection = matrix_multiply(inverse_matrix(target_matrix), source_matrix)
    return CoordinateTransform(
        f"{source.name} coordinates through degree {degree}",
        f"{target.name} coordinates through degree {degree}",
        connection,
        "CC.P1",
        "graded polynomial family connection",
    )


def monomial_to_fibonacci_coefficient(degree: int, fibonacci_degree: int) -> int:
    r"""Return the coefficient of ``F_(k+1)`` in ``x^degree``."""

    n = degree
    k = fibonacci_degree
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValidationError("degree must be a nonnegative integer")
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValidationError("fibonacci degree must be a nonnegative integer")
    if k > n or (n - k) % 2:
        return 0
    j = (n - k) // 2
    previous = comb(n, j - 1) if j > 0 else 0
    return (-1) ** j * (comb(n, j) - previous)


def euler_to_fibonacci_coefficient(euler_degree: int, fibonacci_degree: int) -> Fraction:
    r"""Return the exact coefficient in ``E_n=sum_k c_(k,n)F_(k+1)``."""

    n = euler_degree
    k = fibonacci_degree
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValidationError("Euler degree must be a nonnegative integer")
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValidationError("Fibonacci degree must be a nonnegative integer")
    if k > n:
        return Fraction(0)
    total = Fraction(0)
    for j in range((n - k) // 2 + 1):
        monomial_degree = k + 2 * j
        total += (
            Fraction(comb(n, monomial_degree))
            * euler_polynomial_at_zero(n - monomial_degree)
            * monomial_to_fibonacci_coefficient(monomial_degree, k)
        )
    return total


def euler_to_fibonacci_transform(
    degree: int,
    *,
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> CoordinateTransform:
    """Return the Euler polynomial to Fibonacci polynomial connection."""

    transform = connection_transform(
        euler_polynomial_family(),
        fibonacci_polynomial_family(),
        degree,
        budget=budget,
    )
    explicit = tuple(
        tuple(euler_to_fibonacci_coefficient(column, row) for column in range(degree + 1))
        for row in range(degree + 1)
    )
    if explicit != transform.matrix:
        raise ValidationError("explicit Euler to Fibonacci formula failed verification")
    return CoordinateTransform(
        transform.source,
        transform.target,
        transform.matrix,
        "CC.P3",
        "Euler polynomial coordinates in the Fibonacci companion basis",
    )


def fibonacci_companion_power(power: int) -> tuple[tuple[Polynomial, Polynomial], tuple[Polynomial, Polynomial]]:
    r"""Return ``[[x,1],[1,0]]^power`` as exact polynomial entries."""

    if isinstance(power, bool) or not isinstance(power, int) or power < 0:
        raise ValidationError("power must be a nonnegative integer")
    zero = (Fraction(0),)
    one = (Fraction(1),)
    x = (Fraction(0), Fraction(1))
    result = ((one, zero), (zero, one))
    base = ((x, one), (one, zero))

    def multiply(left, right):
        return tuple(
            tuple(
                polynomial_add(
                    polynomial_multiply(left[row][0], right[0][column]),
                    polynomial_multiply(left[row][1], right[1][column]),
                )
                for column in range(2)
            )
            for row in range(2)
        )

    exponent = power
    while exponent:
        if exponent & 1:
            result = multiply(result, base)
        base = multiply(base, base)
        exponent >>= 1
    return result


def derivative_operator_in_family(family: PolynomialFamily, degree: int) -> Matrix:
    """Return the derivative operator in one graded polynomial basis."""

    basis = family.coefficient_matrix(degree)
    derivative = [
        [Fraction(0) for _ in range(degree + 1)]
        for _ in range(degree + 1)
    ]
    for column in range(1, degree + 1):
        derivative[column - 1][column] = Fraction(column)
    return matrix_multiply(
        inverse_matrix(basis),
        matrix_multiply(tuple(tuple(row) for row in derivative), basis),
    )


def operator_change_of_coordinates(
    operator_in_source: Matrix,
    source_to_target: CoordinateTransform,
) -> Matrix:
    r"""Conjugate an operator from source coordinates into target coordinates."""

    rows, columns = matrix_shape(operator_in_source)
    if rows != columns or rows != source_to_target.source_dimension:
        raise DimensionError("operator dimension does not match the source coordinates")
    inverse = inverse_matrix(source_to_target.matrix)
    return matrix_multiply(
        source_to_target.matrix,
        matrix_multiply(operator_in_source, inverse),
    )


def secant_euler_from_fibonacci_coordinates(index: int) -> Fraction:
    r"""Recover the signed secant Euler number from Fibonacci coordinates.

    The identity uses ``mathcal E_n=2^n E_n(1/2)``.
    """

    transform = euler_to_fibonacci_transform(index)
    column = tuple(transform.matrix[row][index] for row in range(index + 1))
    value = sum(
        (
            column[k] * polynomial_evaluate(fibonacci_polynomial(k + 1), Fraction(1, 2))
            for k in range(index + 1)
        ),
        Fraction(0),
    )
    return Fraction(2**index) * value


__all__ = [
    "Polynomial",
    "PolynomialFamily",
    "connection_transform",
    "derivative_operator_in_family",
    "euler_polynomial",
    "euler_polynomial_at_zero",
    "euler_to_fibonacci_coefficient",
    "euler_to_fibonacci_transform",
    "fibonacci_companion_power",
    "fibonacci_polynomial",
    "fibonacci_polynomial_family",
    "monomial_polynomial_family",
    "monomial_to_fibonacci_coefficient",
    "operator_change_of_coordinates",
    "polynomial_add",
    "polynomial_derivative",
    "polynomial_evaluate",
    "polynomial_multiply",
    "polynomial_scale",
    "polynomial_shift",
    "secant_euler_from_fibonacci_coordinates",
]
