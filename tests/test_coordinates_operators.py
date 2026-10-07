from fractions import Fraction
import random

from komorebi_math.coordinates import (
    ShiftPolynomial,
    bezout_defect_identity,
    polynomial_extended_gcd,
)
from komorebi_math.coordinates.polynomials import polynomial_add, polynomial_multiply


def test_extended_gcd_identity() -> None:
    left = (-1, -1, 1)
    right = (-1, 0, 1)
    gcd, u, v = polynomial_extended_gcd(left, right)
    assert gcd == (1,)
    combined = polynomial_add(polynomial_multiply(u, left), polynomial_multiply(v, right))
    assert combined == (Fraction(1),)


def test_bezout_defect_reconstruction() -> None:
    rng = random.Random(917)
    left = ShiftPolynomial((-1, -1, 1), "Fibonacci")
    right = ShiftPolynomial((-1, 0, 1), "second")
    for length in range(8, 30):
        values = tuple(rng.randint(-100, 100) for _ in range(length))
        result = bezout_defect_identity(values, left.coefficients, right.coefficients)
        assert result.valid
        assert result.reconstructed == result.expected
