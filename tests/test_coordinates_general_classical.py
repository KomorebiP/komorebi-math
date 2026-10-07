from fractions import Fraction

from komorebi_math.coordinates import connection_transform, monomial_polynomial_family
from komorebi_math.coordinates.classical import (
    affine_substitution_transform,
    appell_polynomial_family,
    bernoulli_polynomial,
    bernstein_basis,
    chebyshev_t_polynomial,
    classical_polynomial_catalog,
    hermite_probabilists_polynomial,
    lagrange_interpolation_basis,
    laguerre_polynomial,
    legendre_polynomial,
    newton_interpolation_basis,
)
from komorebi_math.coordinates.exact import identity_matrix, matrix_multiply
from komorebi_math.coordinates.polynomials import polynomial_derivative, polynomial_evaluate


def test_classical_polynomial_formulas() -> None:
    assert bernoulli_polynomial(2) == (Fraction(1, 6), Fraction(-1), Fraction(1))
    assert chebyshev_t_polynomial(3) == (0, -3, 0, 4)
    assert legendre_polynomial(3) == (0, Fraction(-3, 2), 0, Fraction(5, 2))
    assert hermite_probabilists_polynomial(4) == (3, 0, -6, 0, 1)
    assert laguerre_polynomial(3) == (1, -3, Fraction(3, 2), Fraction(-1, 6))


def test_every_classical_family_connects_to_monomials() -> None:
    monomials = monomial_polynomial_family()
    for family in classical_polynomial_catalog().families:
        transform = connection_transform(family, monomials, 8)
        assert transform.roundtrip_valid()
        assert matrix_multiply(transform.inverse().matrix, transform.matrix) == identity_matrix(9)


def test_bernstein_newton_and_lagrange_bases() -> None:
    bernstein = bernstein_basis(5)
    assert bernstein.square
    summed = tuple(sum(vector[index] for vector in bernstein.vectors) for index in range(6))
    assert summed == (1, 0, 0, 0, 0, 0)

    nodes = (Fraction(-1), Fraction(0), Fraction(2), Fraction(5))
    newton = newton_interpolation_basis(nodes)
    lagrange = lagrange_interpolation_basis(nodes)
    connection = newton.connection_to(lagrange)
    assert connection.roundtrip_valid()
    for index, polynomial in enumerate(lagrange.vectors):
        assert tuple(polynomial_evaluate(polynomial, node) for node in nodes) == tuple(
            Fraction(int(index == other)) for other in range(len(nodes))
        )


def test_appell_and_affine_substitution_are_generic() -> None:
    seed = (1, Fraction(-1, 2), Fraction(1, 6), 0, Fraction(-1, 30), 0)
    family = appell_polynomial_family(seed, name="Bernoulli via Appell")
    for degree in range(1, 6):
        assert polynomial_derivative(family.polynomial(degree)) == tuple(
            degree * coefficient for coefficient in family.polynomial(degree - 1)
        )
    transform = affine_substitution_transform(monomial_polynomial_family(), 5, scale=2, shift=3)
    vector = (1, 2, 3, 4, 5, 6)
    recovered = transform.inverse().apply(transform.apply(vector))
    assert recovered == tuple(Fraction(value) for value in vector)
