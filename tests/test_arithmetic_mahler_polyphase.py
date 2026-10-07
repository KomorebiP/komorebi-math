from fractions import Fraction
from math import comb

import numpy as np

from komorebi_math.arithmetic import (
    MahlerSeries,
    UnitModeMahler,
    affine_kernel_composition_certificate,
    affine_mahler_kernel,
    evaluate_matrix_mahler,
    mahler_polyphase,
    matrix_mahler_coefficients,
    mixed_radix_address,
    mixed_radix_digits,
    polyphase_analyze,
    polyphase_synthesize,
    prime_power_kernel_certificate,
)


def test_finite_mahler_reconstruction_and_differences() -> None:
    values = [n**4 - 3 * n**2 + 7 for n in range(9)]
    series = MahlerSeries.from_values(values)
    series.reconstruction_certificate().certificate.require()
    assert series.first(12) == tuple(Fraction(n**4 - 3 * n**2 + 7) for n in range(12))
    first_difference = series.difference()
    assert first_difference.first(8) == tuple(Fraction(values[n + 1] - values[n]) for n in range(8))


def test_affine_mahler_transfer() -> None:
    series = MahlerSeries.from_values([n**5 + 2 * n + 1 for n in range(10)])
    for multiplier in range(1, 6):
        for shift in range(multiplier):
            transformed = series.affine_subsequence(multiplier, shift)
            assert transformed.first(10) == tuple(series.evaluate(multiplier * q + shift) for q in range(10))
    affine_kernel_composition_certificate(3, 2, 4, 1, 12).require()


def test_affine_kernel_binomial_identity() -> None:
    for multiplier in range(1, 6):
        for shift in range(multiplier):
            for q in range(8):
                for r in range(15):
                    left = sum(affine_mahler_kernel(multiplier, shift, r, k) * comb(q, k) for k in range(r + 1))
                    right = comb(multiplier * q + shift, r) if r <= multiplier * q + shift else 0
                    assert left == right


def test_prime_power_sparse_kernel() -> None:
    for prime in (2, 3, 5):
        for exponent in (1, 2):
            for shift in range(min(prime**exponent, 5)):
                prime_power_kernel_certificate(prime, exponent, shift, 12).require()


def test_unit_mode_mahler_coordinates() -> None:
    mode = UnitModeMahler(base=2, jump=4, residue=3)
    mode.certificate(12, degree=12).require()
    series = MahlerSeries(mode.coefficients(12))
    assert series.first(13) == tuple(Fraction(2 ** (4 * x + 3)) for x in range(13))
    assert mode.valuation_slope(5) == 1


def test_matrix_mahler_coordinates() -> None:
    matrix = np.array([[1, 1], [1, 0]], dtype=object)
    coefficients = matrix_mahler_coefficients(matrix, jump=6, residue=2, degree=8)
    for x in range(9):
        assert np.array_equal(evaluate_matrix_mahler(coefficients, x), np.linalg.matrix_power(matrix, 6 * x + 2))


def test_exact_arithmetic_polyphase_and_mixed_radix() -> None:
    values = tuple(range(31))
    coordinates = polyphase_analyze(values, 5)
    assert coordinates.reconstruct() == values
    assert polyphase_synthesize(coordinates.channels) == values
    address = mixed_radix_address((2, 1, 3), (4, 3, 5))
    assert mixed_radix_digits(address, (4, 3, 5)) == (2, 1, 3)
    series = MahlerSeries.from_values([n**3 + 1 for n in range(8)])
    assert mahler_polyphase(series, 4, 3).first(8) == tuple(series.evaluate(4 * n + 3) for n in range(8))
