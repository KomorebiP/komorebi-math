from __future__ import annotations

from fractions import Fraction

from komorebi_math.analytic import (
    CyclicDenominator,
    cyclic_projection_certificate,
    reciprocal_coefficients,
    theta_polynomial,
)

coefficients = list(range(15))
projection = cyclic_projection_certificate(coefficients, 4)
projection.certificate.require()

secant = reciprocal_coefficients(2, Fraction(1), 5)
assert secant[:4] == (Fraction(1), Fraction(1), Fraction(5), Fraction(61))
assert theta_polynomial(4, 3, 0) == 0
assert theta_polynomial(4, 3, -1) == 0

zeros = CyclicDenominator(4).positive_zeros(4)
zeros.certificate.require()
print("cyclic analytic example passed", zeros.zeros)
