from fractions import Fraction

from komorebi_math.arithmetic import (
    EulerLocalCoordinates,
    bernoulli_numbers,
    dyadic_beta,
    dyadic_kernel_valuation,
    dyadic_row_weight,
    dyadic_source,
    euler_zigzag_numbers,
    padic_valuation,
    signed_euler_numbers,
)


def test_classical_euler_zigzag_and_bernoulli_prefixes() -> None:
    assert signed_euler_numbers(12) == (1, 0, -1, 0, 5, 0, -61, 0, 1385, 0, -50521, 0, 2702765)
    assert euler_zigzag_numbers(10) == (1, 1, 1, 2, 5, 16, 61, 272, 1385, 7936, 50521)
    assert bernoulli_numbers(10) == (
        Fraction(1),
        Fraction(-1, 2),
        Fraction(1, 6),
        Fraction(0),
        Fraction(-1, 30),
        Fraction(0),
        Fraction(1, 42),
        Fraction(0),
        Fraction(-1, 30),
        Fraction(0),
        Fraction(5, 66),
    )


def test_odd_and_dyadic_euler_coordinates() -> None:
    expected = signed_euler_numbers(24)
    for prime in (2, 3, 5, 7, 11):
        coordinates = EulerLocalCoordinates(prime)
        coordinates.certificate(24).certificate.require()
        for n in range(0, 25, 2):
            assert coordinates.reconstruct(n) == expected[n]
            assert coordinates.inverse_reconstruct_source(n) == coordinates.source(n)


def test_dyadic_kernel_and_row_weights_are_exact() -> None:
    for r in range(1, 12):
        assert padic_valuation(dyadic_beta(r), 2) == r.bit_count() - 1
        assert dyadic_kernel_valuation(r) == 4 * r - 1
    for n in range(0, 42, 2):
        assert padic_valuation(dyadic_source(n), 2) == 0
        for r in range(n // 2 + 1):
            coordinates = EulerLocalCoordinates(2)
            assert coordinates.row_weight(n, r) == coordinates.formula_row_weight(n, r)
            assert int(coordinates.formula_row_weight(n, r)) == dyadic_row_weight(n, r)


def test_local_truncation_congruences() -> None:
    for prime in (2, 3, 5, 7):
        coordinates = EulerLocalCoordinates(prime)
        for n in range(0, 26, 2):
            for depth in range(n // 2 + 1):
                result = coordinates.truncate(n, depth)
                result.certificate.require()
                if result.modulus is not None:
                    assert result.error.denominator % prime != 0
                    reduced = result.error.numerator * pow(result.error.denominator, -1, result.modulus) % result.modulus
                    assert reduced == 0
