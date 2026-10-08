"""Independent exact controls for recovered sequence-family normalizations."""
from __future__ import annotations

from fractions import Fraction
from math import factorial
import numpy as np
import pytest

from komorebi_math.arithmetic.hurwitz import hurwitz_inverse, hurwitz_product
from komorebi_math.sequences import (
    derangement_sequence, fubini_sequence, kbonacci_closed_form,
    kbonacci_sequence, reciprocal_coefficients, reciprocal_sequence,
    universal_denominator_coefficients,
)
from komorebi_math.exceptions import ValidationError


@pytest.mark.parametrize("spacing", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("power", [0, 1, 2, 3])
def test_compressed_reciprocal_powers_match_dense_hurwitz_series(spacing, power) -> None:
    count = 9
    degree = spacing * (count - 1)
    denominator = universal_denominator_coefficients(spacing, degree)
    inverse = hurwitz_inverse(denominator, degree)
    dense = (Fraction(1),) + (Fraction(0),) * degree
    for _ in range(power):
        dense = hurwitz_product(dense, inverse, degree=degree)
    compressed = reciprocal_coefficients(spacing, count, power=power)
    assert compressed == dense[::spacing]
    assert all(isinstance(value, int) for value in compressed)
    assert all(value == 0 for n, value in enumerate(dense) if n % spacing)


def test_reciprocal_normalization_closed_terms_and_empty_count() -> None:
    assert reciprocal_coefficients(1, 8) == (1,) * 8
    assert reciprocal_coefficients(2, 6) == (1, 1, 5, 61, 1385, 50521)
    assert reciprocal_coefficients(3, 0) == ()
    assert reciprocal_coefficients(2, 5, power=0) == (1, 0, 0, 0, 0)
    assert reciprocal_sequence(2).residue(2, 1).first(3) == (1, 61, 50521)


@pytest.mark.parametrize("k", range(2, 9))
def test_kbonacci_closed_binomial_form_matches_recurrence_and_jump(k) -> None:
    sequence = kbonacci_sequence(k)
    recurrence = sequence.metadata["recurrence"]
    values = sequence.first(61)
    assert values[:k] == (0,) * (k - 1) + (1,)
    assert values == tuple(kbonacci_closed_form(k, n) for n in range(61))
    for n in (0, k - 1, k, 25, 60):
        assert recurrence.term_mod(n, 1009) == values[n] % 1009


def test_ordered_bell_against_independent_stirling_partition_formula() -> None:
    table = [[0] * 13 for _ in range(13)]
    table[0][0] = 1
    for n in range(1, 13):
        for k in range(1, n + 1):
            table[n][k] = table[n - 1][k - 1] + k * table[n - 1][k]
    expected = tuple(sum(factorial(k) * table[n][k] for k in range(n + 1)) for n in range(13))
    assert fubini_sequence().first(13) == expected


def test_derangements_against_inclusion_exclusion() -> None:
    expected = tuple(sum((-1)**k * factorial(n) // factorial(k) for k in range(n + 1)) for n in range(16))
    assert derangement_sequence().first(16) == expected


@pytest.mark.parametrize("bad", [True, 2.5, -1])
def test_family_sizes_reject_boolean_fractional_or_negative_inputs(bad) -> None:
    with pytest.raises(ValidationError):
        reciprocal_coefficients(2, bad)
    with pytest.raises(ValidationError):
        reciprocal_sequence(2, power=bad)
    with pytest.raises(ValidationError):
        universal_denominator_coefficients(2, bad)
    with pytest.raises(ValidationError):
        kbonacci_closed_form(3, bad)


@pytest.mark.parametrize("bad", [False, 0, 2.5])
def test_invalid_spacing_and_order_are_rejected(bad) -> None:
    with pytest.raises(ValidationError):
        reciprocal_sequence(bad)
    with pytest.raises(ValidationError):
        kbonacci_sequence(bad)


def test_numpy_integer_parameters_are_normalized_to_python_arithmetic() -> None:
    assert reciprocal_coefficients(np.int64(2), np.int64(6), power=np.int64(1))[-1] == 50521
    assert kbonacci_sequence(np.int64(3)).first(7) == (0, 0, 1, 1, 2, 4, 7)
