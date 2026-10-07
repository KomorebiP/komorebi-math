from fractions import Fraction

import numpy as np
import pytest

from komorebi_math.arithmetic import (
    LinearRecurrence,
    fibonacci_recurrence,
    lucas_recurrence,
    matrix_power_exact,
    matrix_power_mod,
    second_order_jump,
)
from komorebi_math.exceptions import ValidationError


def test_fibonacci_and_lucas_recurrences() -> None:
    fibonacci = fibonacci_recurrence()
    lucas = lucas_recurrence()
    assert fibonacci.first(12) == (0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89)
    assert lucas.first(12) == (2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199)
    fibonacci.certificate(30).certificate.require()
    lucas.certificate(30).certificate.require()


def test_residue_packets_use_the_jump_matrix() -> None:
    fibonacci = fibonacci_recurrence()
    for modulus in range(1, 9):
        for address in range(modulus):
            packet = fibonacci.residue_packet(modulus, address)
            packet.certificate(12).certificate.require()
            expected = tuple(fibonacci.term(modulus * k + address) for k in range(12))
            assert packet.first(12) == expected


def test_second_order_jump_formula() -> None:
    fibonacci = fibonacci_recurrence()
    for modulus in range(1, 10):
        trace, determinant = second_order_jump(1, -1, modulus)
        packet = fibonacci.residue_packet(modulus, 0).first(15)
        for k in range(13):
            assert packet[k + 2] == trace * packet[k + 1] - determinant * packet[k]


def test_modular_periodicity() -> None:
    fibonacci = fibonacci_recurrence()
    for modulus in range(2, 20):
        result = fibonacci.period_mod(modulus)
        result.certificate.require()
        assert result.preperiod == 0
        assert result.period > 0
    singular = LinearRecurrence((0,), (1,))
    result = singular.period_mod(8)
    result.certificate.require()
    assert result.preperiod == 1
    assert result.period == 1


def test_exact_and_modular_matrix_powers() -> None:
    matrix = np.array([[1, 1], [1, 0]], dtype=object)
    assert np.array_equal(matrix_power_exact(matrix, 12), np.array([[233, 144], [144, 89]], dtype=object))
    assert np.array_equal(matrix_power_mod(matrix, 12, 8), np.array([[1, 0], [0, 1]], dtype=object))


def test_recurrence_validation() -> None:
    with pytest.raises(ValidationError):
        LinearRecurrence((), ())
    with pytest.raises(ValidationError):
        fibonacci_recurrence().period_mod(1)


def test_fast_modular_terms_and_packet_terms() -> None:
    fibonacci = fibonacci_recurrence()
    for modulus in (2, 3, 5, 8, 11, 25):
        for index in range(100):
            assert fibonacci.term_mod(index, modulus) == fibonacci.term(index) % modulus
        packet = fibonacci.residue_packet(7, 3)
        for index in range(50):
            assert packet.term_mod(index, modulus) == packet.term(index) % modulus


def test_modular_recurrence_requires_integral_data() -> None:
    recurrence = LinearRecurrence((Fraction(1, 2),), (Fraction(1),))
    with pytest.raises(ValidationError, match="integral"):
        recurrence.term_mod(10, 7)
    with pytest.raises(ValidationError, match="integral"):
        recurrence.period_mod(7)
