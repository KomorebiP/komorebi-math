"""Connect classical polynomial families and transport differentiation."""

from fractions import Fraction

from komorebi_math.coordinates import (
    connection_transform,
    derivative_coordinate_operator,
    get_polynomial_family,
    transport_polynomial_operator,
)

legendre = get_polynomial_family("legendre")
chebyshev = get_polynomial_family("chebyshev-t")
monomials = get_polynomial_family("monomial")

connection = connection_transform(legendre, chebyshev, 8)
connection.certificate().require()

coefficients = tuple(Fraction(index + 1) for index in range(9))
chebyshev_coordinates = connection.apply(coefficients)
assert connection.inverse().apply(chebyshev_coordinates) == coefficients

monomial_derivative = derivative_coordinate_operator(monomials, 8)
legendre_derivative = transport_polynomial_operator(
    monomial_derivative,
    monomials,
    legendre,
)
assert legendre_derivative.matrix == derivative_coordinate_operator(legendre, 8).matrix

print("Legendre to Chebyshev determinant:", connection.determinant)
print("Chebyshev coordinates:", chebyshev_coordinates)
