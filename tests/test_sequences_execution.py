from fractions import Fraction

from komorebi_math.arithmetic import LinearRecurrence
from komorebi_math.sequences import (
    Sequence,
    apery_sequence,
    bell_sequence,
    binomial_transform_values,
    catalan_sequence,
    compile_sequence,
    detect_linear_recurrence,
    fibonacci_sequence,
    finite_difference,
    get_sequence,
    inverse_binomial_transform_values,
    motzkin_sequence,
    partition_sequence,
    thue_morse_sequence,
)


def test_lazy_sequence_and_residue_views() -> None:
    squares = Sequence(lambda n: n * n, name="squares")
    assert squares.first(8) == (0, 1, 4, 9, 16, 25, 36, 49)
    assert squares.residue(3, 1).first(6) == (1, 16, 49, 100, 169, 256)
    assert squares.modulo(5).first(8) == (0, 1, 4, 4, 1, 0, 1, 4)


def test_sequence_catalog_prefixes() -> None:
    assert bell_sequence().first(10) == (1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147)
    assert catalan_sequence().first(10) == (1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862)
    assert motzkin_sequence().first(11) == (1, 1, 2, 4, 9, 21, 51, 127, 323, 835, 2188)
    assert partition_sequence().first(16) == (1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77, 101, 135, 176)
    assert apery_sequence().first(8) == (1, 5, 73, 1445, 33001, 819005, 21460825, 584307365)
    assert get_sequence("thue_morse").first(8) == (0, 1, 1, 0, 1, 0, 0, 1)


def test_binomial_transforms_are_inverse() -> None:
    values = tuple(range(-4, 12))
    transformed = binomial_transform_values(values)
    assert inverse_binomial_transform_values(transformed) == values


def test_thue_morse_automatic_relations() -> None:
    sequence = thue_morse_sequence()
    for n in range(1000):
        assert sequence.term(2 * n) == sequence.term(n)
        assert sequence.term(2 * n + 1) == 1 - sequence.term(n)


def test_recurrence_detection_and_compilation() -> None:
    fibonacci = fibonacci_sequence()
    compiled = compile_sequence(fibonacci)
    compiled.certificate.require()
    assert compiled.strategy == "declared linear recurrence"
    assert compiled.first(20) == fibonacci.first(20)
    numerator, denominator = compiled.generating_function()
    assert numerator == (Fraction(0), Fraction(1))
    assert denominator == (Fraction(1), Fraction(-1), Fraction(-1))

    values = fibonacci.first(20)
    detected = detect_linear_recurrence(values, max_order=6)
    assert isinstance(detected, LinearRecurrence)
    assert detected.order == 2
    assert detected.first(30) == fibonacci.first(30)


def test_finite_difference_sequence() -> None:
    polynomial = Sequence(lambda n: n**4 + 2 * n + 1, name="quartic")
    fifth = finite_difference(polynomial, order=5)
    assert fifth.first(20) == (0,) * 20


def test_sequence_modulo_rejects_nonintegral_terms() -> None:
    rational = Sequence(lambda n: Fraction(1, n + 1), name="rational")
    residues = rational.modulo(5)
    assert residues.term(0) == 1
    import pytest
    from komorebi_math.exceptions import ValidationError
    with pytest.raises(ValidationError, match="integral"):
        residues.term(1)


def test_sequence_indices_reject_boolean_and_float_coercion() -> None:
    sequence = Sequence(lambda n: n)
    import pytest
    from komorebi_math.exceptions import ValidationError
    with pytest.raises(ValidationError):
        sequence.term(True)
    with pytest.raises(ValidationError):
        sequence.term(2.0)
