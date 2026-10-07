from fractions import Fraction
import random

from komorebi_math.coordinates import (
    analyze_recurrence_coordinates,
    companion_matrix,
    coordinate_change_between_recurrences,
    defect_green_weights,
    euler_fibonacci_coordinates,
    fibonacci_numbers,
    recurrence_analysis_transform,
    recurrence_synthesis_transform,
    signed_secant_euler_numbers,
    variation_of_constants_prefix,
)
from komorebi_math.coordinates.exact import identity_matrix, matrix_multiply


def test_recurrence_analysis_is_unimodular() -> None:
    for coefficients in ((1, 1), (2, -3, 1), (Fraction(1, 2), Fraction(-2, 3))):
        for length in range(len(coefficients), len(coefficients) + 8):
            analysis = recurrence_analysis_transform(coefficients, length)
            synthesis = recurrence_synthesis_transform(coefficients, length)
            assert analysis.determinant == 1
            assert matrix_multiply(analysis.matrix, synthesis.matrix) == identity_matrix(length)
            assert matrix_multiply(synthesis.matrix, analysis.matrix) == identity_matrix(length)


def test_random_prefix_roundtrips() -> None:
    rng = random.Random(1907)
    for order in range(1, 5):
        for length in range(order, 16):
            for _ in range(30):
                coefficients = tuple(rng.randint(-3, 3) for _ in range(order))
                values = tuple(rng.randint(-50, 50) for _ in range(length))
                record = analyze_recurrence_coordinates(values, coefficients)
                assert record.reconstruct() == values
                assert variation_of_constants_prefix(record) == values
                assert record.certificate().valid


def test_coordinate_change_composes_through_raw_values() -> None:
    rng = random.Random(412)
    systems = ((1, 1), (1, 0, 1), (-1, 2), (2, -1, 3, 1))
    length = 12
    values = tuple(rng.randint(-20, 20) for _ in range(length))
    for source in systems:
        source_record = analyze_recurrence_coordinates(values, source, name="source")
        for target in systems:
            transform = coordinate_change_between_recurrences(source, target, length)
            target_record = analyze_recurrence_coordinates(values, target, name="target")
            assert transform.apply(source_record.coordinates()) == target_record.coordinates()
            assert transform.roundtrip_valid()


def test_euler_to_fibonacci_defect_coordinates() -> None:
    values = signed_secant_euler_numbers(20)
    record = euler_fibonacci_coordinates(20)
    assert record.initial == (1, 0)
    assert record.defects[:8] == (-2, 1, 6, -5, -66, 61, 1446, -1385)
    assert record.reconstruct() == values
    assert variation_of_constants_prefix(record) == values
    assert not record.homogeneous


def test_fibonacci_green_kernel() -> None:
    weights = defect_green_weights((1, 1), 12)
    fibonacci = fibonacci_numbers(13)
    assert weights == tuple(Fraction(fibonacci[lag - 1]) for lag in range(1, 13))


def test_fibonacci_companion_matrix() -> None:
    assert companion_matrix((1, 1)) == ((0, 1), (1, 1))
