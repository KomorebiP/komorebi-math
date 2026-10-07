from fractions import Fraction

from komorebi_math.coordinates.affine import (
    BarycentricCoordinateSystem,
    affine_basis_transform,
    translation_transform,
)
from komorebi_math.coordinates.exact import identity_matrix, matrix_multiply
from komorebi_math.coordinates.formal import FormalCoordinateChange, identity_formal_coordinate


def test_affine_chart_translation_and_barycentric_roundtrips() -> None:
    chart = affine_basis_transform(
        (3, -2),
        ((1, 1), (1, -1)),
        local_name="diagonal local",
        ambient_name="Cartesian",
    )
    point = (Fraction(5, 2), Fraction(-3, 2))
    assert chart.inverse().apply(chart.apply(point)) == point
    chart.certificate().require()

    translation = translation_transform((2, 7), source="Cartesian", target="shifted")
    composed = chart.then(translation)
    assert composed.inverse().apply(composed.apply(point)) == point

    simplex = BarycentricCoordinateSystem("triangle", ((0, 0), (2, 0), (0, 3)))
    weights = (Fraction(1, 6), Fraction(1, 3), Fraction(1, 2))
    cartesian = simplex.to_cartesian(weights)
    assert simplex.from_cartesian(cartesian) == weights
    simplex.certificate().require()


def test_formal_coordinate_composition_inverse_and_substitution() -> None:
    phi = FormalCoordinateChange(
        "quadratic coordinate",
        (0, 2, 1, -1, 3, 0, 2, -2, 1),
        source="u",
        target="z",
    )
    inverse = phi.inverse()
    phi.certificate().require()
    identity = identity_formal_coordinate(8, coordinate="u")
    assert inverse.compose(phi, degree=8).coefficients == identity.coefficients
    transform = phi.substitution_transform(8)
    inverse_transform = inverse.substitution_transform(8)
    assert matrix_multiply(inverse_transform.matrix, transform.matrix) == identity_matrix(9)


def test_formal_pullback_and_pushforward_roundtrip() -> None:
    phi = FormalCoordinateChange("near identity", (0, 1, 1, -2, 3, 1), source="u", target="z")
    polynomial = (3, 4, -1, 2, 5, -3)
    pulled = phi.pullback_coefficients(polynomial, degree=5)
    recovered = phi.pushforward_coefficients(pulled, degree=5)
    assert recovered == tuple(Fraction(value) for value in polynomial)
