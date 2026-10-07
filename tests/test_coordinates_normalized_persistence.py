from fractions import Fraction

import pytest

from komorebi_math.coordinates import (
    convert_normalized_coefficients,
    euler_fibonacci_coordinates,
    exponential_basis,
    fibonacci_factorial_basis,
    load_coordinate_record,
    load_transform,
    normalization_change_transform,
    save_coordinate_record,
    save_transform,
    signed_secant_euler_numbers,
)


def test_normalized_euler_to_fibonacci_factorial() -> None:
    values = signed_secant_euler_numbers(10)
    converted = convert_normalized_coefficients(values, exponential_basis(), fibonacci_factorial_basis())
    assert converted[:8] == (
        Fraction(1),
        Fraction(0),
        Fraction(-1, 2),
        Fraction(0),
        Fraction(5, 4),
        Fraction(0),
        Fraction(-61, 3),
        Fraction(0),
    )
    inverse = normalization_change_transform(fibonacci_factorial_basis(), exponential_basis(), 9)
    assert inverse.apply(converted) == tuple(Fraction(value) for value in values)


def test_record_roundtrips_and_tamper_rejection(tmp_path) -> None:
    transform = normalization_change_transform(exponential_basis(), fibonacci_factorial_basis(), 8)
    transform_path = tmp_path / "transform.json"
    save_transform(transform_path, transform)
    assert load_transform(transform_path) == transform

    record = euler_fibonacci_coordinates(12)
    record_path = tmp_path / "record.json"
    save_coordinate_record(record_path, record)
    assert load_coordinate_record(record_path) == record

    text = transform_path.read_text(encoding="utf-8").replace('"numerator": 1', '"numerator": 2', 1)
    transform_path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256|integrity"):
        load_transform(transform_path)
