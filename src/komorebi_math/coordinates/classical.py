"""Classical polynomial families and interpolation coordinate systems."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb, factorial
from typing import Callable, Iterable, Sequence

from ..exceptions import DimensionError, ValidationError
from .basis import FiniteBasis
from .budget import DEFAULT_COORDINATE_BUDGET, CoordinateBudget
from .exact import Matrix, Scalar, Vector, as_vector, inverse_matrix, matrix_from_columns
from .polynomials import (
    Polynomial,
    PolynomialFamily,
    euler_polynomial_family,
    fibonacci_polynomial_family,
    monomial_polynomial_family,
    polynomial_add,
    polynomial_evaluate,
    polynomial_multiply,
    polynomial_scale,
    polynomial_shift,
)
from .transform import CoordinateTransform


def _index(value: int, *, name: str = "index") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{name} must be a nonnegative integer")
    return value


def _pad(polynomial: Sequence[Scalar], size: int) -> Vector:
    values = as_vector(polynomial)
    if len(values) > size:
        raise DimensionError("polynomial degree exceeds the declared ambient space")
    return values + tuple(Fraction(0) for _ in range(size - len(values)))


def constant_polynomial(value: Scalar) -> Polynomial:
    """Return one exact constant polynomial."""

    return (Fraction(value),)


def polynomial_subtract(left: Sequence[Scalar], right: Sequence[Scalar]) -> Polynomial:
    """Subtract exact polynomials."""

    return polynomial_add(left, polynomial_scale(right, -1))


def affine_polynomial(scale: Scalar, shift: Scalar) -> Polynomial:
    """Return the affine polynomial ``shift + scale*x``."""

    return (Fraction(shift), Fraction(scale))


def polynomial_compose_affine(
    polynomial: Sequence[Scalar],
    *,
    scale: Scalar = 1,
    shift: Scalar = 0,
) -> Polynomial:
    """Return ``polynomial(scale*x+shift)`` exactly."""

    affine = affine_polynomial(scale, shift)
    result: Polynomial = (Fraction(0),)
    for coefficient in reversed(as_vector(polynomial)):
        result = polynomial_add(polynomial_multiply(result, affine), (coefficient,))
    return result


def falling_factorial_polynomial(degree: int) -> Polynomial:
    r"""Return ``x^(underline degree)``."""

    n = _index(degree, name="degree")
    result: Polynomial = (Fraction(1),)
    for root in range(n):
        result = polynomial_multiply(result, (Fraction(-root), Fraction(1)))
    return result


def rising_factorial_polynomial(degree: int) -> Polynomial:
    r"""Return ``x^(overline degree)``."""

    n = _index(degree, name="degree")
    result: Polynomial = (Fraction(1),)
    for root in range(n):
        result = polynomial_multiply(result, (Fraction(root), Fraction(1)))
    return result


def binomial_polynomial(degree: int) -> Polynomial:
    r"""Return the binomial polynomial ``binom(x,degree)``."""

    n = _index(degree, name="degree")
    return polynomial_scale(falling_factorial_polynomial(n), Fraction(1, factorial(n)))


def falling_factorial_polynomial_family() -> PolynomialFamily:
    """Return the graded falling-factorial family."""

    return PolynomialFamily("falling factorials", falling_factorial_polynomial)


def rising_factorial_polynomial_family() -> PolynomialFamily:
    """Return the graded rising-factorial family."""

    return PolynomialFamily("rising factorials", rising_factorial_polynomial)


def binomial_polynomial_family() -> PolynomialFamily:
    """Return the graded binomial-polynomial family."""

    return PolynomialFamily("binomial polynomials", binomial_polynomial)


def bernoulli_number(index: int) -> Fraction:
    """Return the Bernoulli number with convention ``B_1=-1/2``."""

    n = _index(index)
    values = [Fraction(1)]
    for degree in range(1, n + 1):
        values.append(
            -sum(
                (Fraction(comb(degree + 1, k)) * values[k] for k in range(degree)),
                Fraction(0),
            )
            / Fraction(degree + 1)
        )
    return values[n]


def bernoulli_polynomial(degree: int) -> Polynomial:
    """Return the Bernoulli polynomial ``B_degree(x)``."""

    n = _index(degree, name="degree")
    return tuple(
        Fraction(comb(n, power)) * bernoulli_number(n - power)
        for power in range(n + 1)
    )


def bernoulli_polynomial_family() -> PolynomialFamily:
    """Return the graded Bernoulli polynomial family."""

    return PolynomialFamily("Bernoulli polynomials", bernoulli_polynomial)


def lucas_polynomial(index: int) -> Polynomial:
    r"""Return ``L_index(x)`` with ``L_0=2``, ``L_1=x``."""

    n = _index(index)
    if n == 0:
        return (Fraction(2),)
    if n == 1:
        return (Fraction(0), Fraction(1))
    previous: Polynomial = (Fraction(2),)
    current: Polynomial = (Fraction(0), Fraction(1))
    for _ in range(1, n):
        previous, current = current, polynomial_add(polynomial_shift(current), previous)
    return current


def lucas_polynomial_family() -> PolynomialFamily:
    """Return the graded Lucas polynomial family."""

    return PolynomialFamily("Lucas polynomials", lucas_polynomial)


def pell_polynomial(index: int) -> Polynomial:
    r"""Return ``P_index(x)`` with ``P_0=0``, ``P_1=1`` and ``P_n=2xP_(n-1)+P_(n-2)``."""

    n = _index(index)
    if n == 0:
        return (Fraction(0),)
    if n == 1:
        return (Fraction(1),)
    previous: Polynomial = (Fraction(0),)
    current: Polynomial = (Fraction(1),)
    for _ in range(1, n):
        previous, current = current, polynomial_add(polynomial_scale(polynomial_shift(current), 2), previous)
    return current


def pell_polynomial_family() -> PolynomialFamily:
    """Return the graded basis ``P_(n+1)(x)`` of Pell polynomials."""

    return PolynomialFamily("Pell polynomials", lambda degree: pell_polynomial(degree + 1))


def chebyshev_t_polynomial(degree: int) -> Polynomial:
    """Return the Chebyshev polynomial ``T_degree``."""

    n = _index(degree, name="degree")
    if n == 0:
        return (Fraction(1),)
    if n == 1:
        return (Fraction(0), Fraction(1))
    previous: Polynomial = (Fraction(1),)
    current: Polynomial = (Fraction(0), Fraction(1))
    for _ in range(1, n):
        following = polynomial_subtract(polynomial_scale(polynomial_shift(current), 2), previous)
        previous, current = current, following
    return current


def chebyshev_u_polynomial(degree: int) -> Polynomial:
    """Return the Chebyshev polynomial ``U_degree``."""

    n = _index(degree, name="degree")
    if n == 0:
        return (Fraction(1),)
    if n == 1:
        return (Fraction(0), Fraction(2))
    previous: Polynomial = (Fraction(1),)
    current: Polynomial = (Fraction(0), Fraction(2))
    for _ in range(1, n):
        following = polynomial_subtract(polynomial_scale(polynomial_shift(current), 2), previous)
        previous, current = current, following
    return current


def chebyshev_t_polynomial_family() -> PolynomialFamily:
    """Return the graded Chebyshev ``T`` family."""

    return PolynomialFamily("Chebyshev T polynomials", chebyshev_t_polynomial)


def chebyshev_u_polynomial_family() -> PolynomialFamily:
    """Return the graded Chebyshev ``U`` family."""

    return PolynomialFamily("Chebyshev U polynomials", chebyshev_u_polynomial)


def legendre_polynomial(degree: int) -> Polynomial:
    """Return the Legendre polynomial ``P_degree``."""

    n = _index(degree, name="degree")
    if n == 0:
        return (Fraction(1),)
    if n == 1:
        return (Fraction(0), Fraction(1))
    previous: Polynomial = (Fraction(1),)
    current: Polynomial = (Fraction(0), Fraction(1))
    for k in range(1, n):
        following = polynomial_scale(
            polynomial_subtract(
                polynomial_scale(polynomial_shift(current), 2 * k + 1),
                polynomial_scale(previous, k),
            ),
            Fraction(1, k + 1),
        )
        previous, current = current, following
    return current


def legendre_polynomial_family() -> PolynomialFamily:
    """Return the graded Legendre polynomial family."""

    return PolynomialFamily("Legendre polynomials", legendre_polynomial)


def hermite_probabilists_polynomial(degree: int) -> Polynomial:
    """Return the probabilists' Hermite polynomial ``He_degree``."""

    n = _index(degree, name="degree")
    if n == 0:
        return (Fraction(1),)
    if n == 1:
        return (Fraction(0), Fraction(1))
    previous: Polynomial = (Fraction(1),)
    current: Polynomial = (Fraction(0), Fraction(1))
    for k in range(1, n):
        following = polynomial_subtract(polynomial_shift(current), polynomial_scale(previous, k))
        previous, current = current, following
    return current


def hermite_physicists_polynomial(degree: int) -> Polynomial:
    """Return the physicists' Hermite polynomial ``H_degree``."""

    n = _index(degree, name="degree")
    if n == 0:
        return (Fraction(1),)
    if n == 1:
        return (Fraction(0), Fraction(2))
    previous: Polynomial = (Fraction(1),)
    current: Polynomial = (Fraction(0), Fraction(2))
    for k in range(1, n):
        following = polynomial_subtract(
            polynomial_scale(polynomial_shift(current), 2),
            polynomial_scale(previous, 2 * k),
        )
        previous, current = current, following
    return current


def hermite_probabilists_polynomial_family() -> PolynomialFamily:
    """Return the probabilists' Hermite family."""

    return PolynomialFamily("probabilists Hermite polynomials", hermite_probabilists_polynomial)


def hermite_physicists_polynomial_family() -> PolynomialFamily:
    """Return the physicists' Hermite family."""

    return PolynomialFamily("physicists Hermite polynomials", hermite_physicists_polynomial)


def laguerre_polynomial(degree: int) -> Polynomial:
    """Return the ordinary Laguerre polynomial ``L_degree``."""

    n = _index(degree, name="degree")
    return tuple(
        Fraction((-1) ** power * comb(n, power), factorial(power))
        for power in range(n + 1)
    )


def laguerre_polynomial_family() -> PolynomialFamily:
    """Return the graded ordinary Laguerre family."""

    return PolynomialFamily("Laguerre polynomials", laguerre_polynomial)


def appell_polynomial(
    degree: int,
    values_at_zero: Sequence[Scalar],
) -> Polynomial:
    r"""Return one Appell polynomial from the values ``P_n(0)``.

    The defining identity is
    ``P_n(x)=sum_(k=0)^n binom(n,k) P_(n-k)(0) x^k``.
    """

    n = _index(degree, name="degree")
    values = as_vector(values_at_zero)
    if len(values) <= n:
        raise DimensionError("Appell seed must include values through the requested degree")
    if values[0] == 0:
        raise ValidationError("an Appell family needs a nonzero degree-zero polynomial")
    return tuple(Fraction(comb(n, power)) * values[n - power] for power in range(n + 1))


def appell_polynomial_family(
    values_at_zero: Sequence[Scalar],
    *,
    name: str = "custom Appell polynomials",
) -> PolynomialFamily:
    """Return a finite-seed Appell family."""

    values = as_vector(values_at_zero)
    if not values or values[0] == 0:
        raise ValidationError("Appell seeds must start with a nonzero value")

    def rule(degree: int) -> Polynomial:
        if degree >= len(values):
            raise DimensionError("Appell seed is too short for the requested degree")
        return appell_polynomial(degree, values)

    return PolynomialFamily(name, rule)


def bernstein_polynomial(degree: int, index: int) -> Polynomial:
    r"""Return ``binom(degree,index)x^index(1-x)^(degree-index)``."""

    n = _index(degree, name="degree")
    k = _index(index, name="index")
    if k > n:
        raise DimensionError("Bernstein index cannot exceed its degree")
    result: Polynomial = (Fraction(comb(n, k)),)
    result = polynomial_shift(result, k)
    for _ in range(n - k):
        result = polynomial_multiply(result, (Fraction(1), Fraction(-1)))
    return result


def bernstein_basis(degree: int) -> FiniteBasis:
    """Return the fixed-degree Bernstein basis of polynomials through ``degree``."""

    n = _index(degree, name="degree")
    vectors = tuple(_pad(bernstein_polynomial(n, k), n + 1) for k in range(n + 1))
    return FiniteBasis(
        f"Bernstein basis of degree {n}",
        vectors,
        tuple(f"B({n},{k})" for k in range(n + 1)),
        "fixed-degree Bernstein polynomial coordinates",
    )


def newton_interpolation_basis(nodes: Sequence[Scalar]) -> FiniteBasis:
    """Return the Newton interpolation basis associated with distinct nodes."""

    points = as_vector(nodes)
    if not points:
        raise DimensionError("interpolation nodes must be nonempty")
    if len(set(points)) != len(points):
        raise ValidationError("interpolation nodes must be distinct")
    size = len(points)
    polynomials: list[Polynomial] = [(Fraction(1),)]
    current: Polynomial = (Fraction(1),)
    for node in points[:-1]:
        current = polynomial_multiply(current, (-node, Fraction(1)))
        polynomials.append(current)
    return FiniteBasis(
        f"Newton basis at {tuple(points)}",
        tuple(_pad(polynomial, size) for polynomial in polynomials),
        tuple(f"N{index}" for index in range(size)),
        "Newton interpolation coordinates",
    )


def lagrange_interpolation_basis(nodes: Sequence[Scalar]) -> FiniteBasis:
    """Return the cardinal Lagrange basis at distinct exact nodes."""

    points = as_vector(nodes)
    if not points:
        raise DimensionError("interpolation nodes must be nonempty")
    if len(set(points)) != len(points):
        raise ValidationError("interpolation nodes must be distinct")
    size = len(points)
    polynomials: list[Vector] = []
    for i, node in enumerate(points):
        polynomial: Polynomial = (Fraction(1),)
        denominator = Fraction(1)
        for j, other in enumerate(points):
            if i == j:
                continue
            polynomial = polynomial_multiply(polynomial, (-other, Fraction(1)))
            denominator *= node - other
        polynomials.append(_pad(polynomial_scale(polynomial, 1 / denominator), size))
    return FiniteBasis(
        f"Lagrange basis at {tuple(points)}",
        tuple(polynomials),
        tuple(f"L{index}" for index in range(size)),
        "cardinal interpolation coordinates",
    )


def evaluation_transform(
    family: PolynomialFamily,
    nodes: Sequence[Scalar],
    *,
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> CoordinateTransform:
    """Map polynomial-family coefficients to values at exact nodes."""

    points = as_vector(nodes)
    if not points:
        raise DimensionError("evaluation nodes must be nonempty")
    degree = len(points) - 1
    budget.require_matrix(len(points), len(points), operation="polynomial evaluation transform")
    matrix = tuple(
        tuple(polynomial_evaluate(family.polynomial(column), node) for column in range(degree + 1))
        for node in points
    )
    transform = CoordinateTransform(
        f"{family.name} coefficients through degree {degree}",
        f"values at {tuple(points)}",
        matrix,
        "CC.P4",
        "exact polynomial evaluation map",
    )
    if transform.determinant == 0:
        raise ValidationError("evaluation nodes do not determine the declared polynomial space")
    return transform


def interpolation_transform(
    family: PolynomialFamily,
    nodes: Sequence[Scalar],
    *,
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> CoordinateTransform:
    """Map exact node values to coefficients in a polynomial family."""

    return evaluation_transform(family, nodes, budget=budget).inverse(theorem="CC.P5")


def operator_matrix_between_families(
    source: PolynomialFamily,
    target: PolynomialFamily,
    source_degree: int,
    target_degree: int,
    operator: Callable[[Polynomial], Polynomial],
    *,
    name: str = "polynomial operator",
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> Matrix:
    """Return an exact matrix for a polynomial operator between two families."""

    source_bound = _index(source_degree, name="source degree")
    target_bound = _index(target_degree, name="target degree")
    budget.require_matrix(
        target_bound + 1,
        source_bound + 1,
        operation=name,
    )
    target_matrix = target.coefficient_matrix(target_bound)
    target_inverse = inverse_matrix(target_matrix)
    columns: list[Vector] = []
    for degree in range(source_bound + 1):
        image = operator(source.polynomial(degree))
        padded = _pad(image, target_bound + 1)
        columns.append(tuple(sum(target_inverse[row][k] * padded[k] for k in range(target_bound + 1)) for row in range(target_bound + 1)))
    return matrix_from_columns(columns)


def affine_substitution_transform(
    family: PolynomialFamily,
    degree: int,
    *,
    scale: Scalar = 1,
    shift: Scalar = 0,
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> CoordinateTransform:
    """Return the coordinates of ``p(scale*x+shift)`` in one family."""

    bound = _index(degree, name="degree")
    matrix = operator_matrix_between_families(
        family,
        family,
        bound,
        bound,
        lambda polynomial: polynomial_compose_affine(polynomial, scale=scale, shift=shift),
        name="affine polynomial substitution",
        budget=budget,
    )
    return CoordinateTransform(
        f"{family.name} coefficients through degree {bound}",
        f"{family.name} coefficients after x -> {scale}x+{shift}",
        matrix,
        "CC.P6",
        "exact affine substitution in a graded polynomial family",
    )


@dataclass(frozen=True, slots=True)
class PolynomialFamilyCatalog:
    """Named constructors for the built-in graded polynomial families."""

    families: tuple[PolynomialFamily, ...]

    def names(self) -> tuple[str, ...]:
        """Return available family names."""

        return tuple(family.name for family in self.families)

    def get(self, name: str) -> PolynomialFamily:
        """Return one family by exact name or a normalized short alias."""

        query = _family_key(name)
        aliases = {
            "monomial": "monomials",
            "euler": "eulerpolynomials",
            "fibonacci": "fibonaccipolynomials",
            "falling": "fallingfactorials",
            "rising": "risingfactorials",
            "binomial": "binomialpolynomials",
            "bernoulli": "bernoullipolynomials",
            "lucas": "lucaspolynomials",
            "pell": "pellpolynomials",
            "chebyshevt": "chebyshevtpolynomials",
            "chebyshevu": "chebyshevupolynomials",
            "legendre": "legendrepolynomials",
            "hermiteprobabilists": "probabilistshermitepolynomials",
            "hermitephysicists": "physicistshermitepolynomials",
            "laguerre": "laguerrepolynomials",
        }
        query = aliases.get(query, query)
        for family in self.families:
            if _family_key(family.name) == query:
                return family
        raise ValidationError(f"unknown polynomial family {name!r}")



def _family_key(name: str) -> str:
    return "".join(character for character in str(name).lower() if character.isalnum())


def get_polynomial_family(name: str) -> PolynomialFamily:
    """Return one built-in graded polynomial family by name or short alias."""

    return classical_polynomial_catalog().get(name)

def classical_polynomial_catalog() -> PolynomialFamilyCatalog:
    """Return the built-in exact graded polynomial-family catalog."""

    return PolynomialFamilyCatalog(
        (
            monomial_polynomial_family(),
            euler_polynomial_family(),
            fibonacci_polynomial_family(),
            falling_factorial_polynomial_family(),
            rising_factorial_polynomial_family(),
            binomial_polynomial_family(),
            bernoulli_polynomial_family(),
            lucas_polynomial_family(),
            pell_polynomial_family(),
            chebyshev_t_polynomial_family(),
            chebyshev_u_polynomial_family(),
            legendre_polynomial_family(),
            hermite_probabilists_polynomial_family(),
            hermite_physicists_polynomial_family(),
            laguerre_polynomial_family(),
        )
    )


__all__ = [
    "PolynomialFamilyCatalog",
    "affine_polynomial",
    "affine_substitution_transform",
    "appell_polynomial",
    "appell_polynomial_family",
    "bernoulli_number",
    "bernoulli_polynomial",
    "bernoulli_polynomial_family",
    "bernstein_basis",
    "bernstein_polynomial",
    "binomial_polynomial",
    "binomial_polynomial_family",
    "chebyshev_t_polynomial",
    "chebyshev_t_polynomial_family",
    "chebyshev_u_polynomial",
    "chebyshev_u_polynomial_family",
    "classical_polynomial_catalog",
    "constant_polynomial",
    "evaluation_transform",
    "falling_factorial_polynomial",
    "falling_factorial_polynomial_family",
    "get_polynomial_family",
    "hermite_physicists_polynomial",
    "hermite_physicists_polynomial_family",
    "hermite_probabilists_polynomial",
    "hermite_probabilists_polynomial_family",
    "interpolation_transform",
    "lagrange_interpolation_basis",
    "laguerre_polynomial",
    "laguerre_polynomial_family",
    "legendre_polynomial",
    "legendre_polynomial_family",
    "lucas_polynomial",
    "lucas_polynomial_family",
    "newton_interpolation_basis",
    "operator_matrix_between_families",
    "pell_polynomial",
    "pell_polynomial_family",
    "polynomial_compose_affine",
    "polynomial_subtract",
    "rising_factorial_polynomial",
    "rising_factorial_polynomial_family",
]
