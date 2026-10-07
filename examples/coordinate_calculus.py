from __future__ import annotations

from komorebi_math.coordinates import (
    analyze_recurrence_coordinates,
    coordinate_change_between_recurrences,
    derivative_operator_in_family,
    euler_polynomial_family,
    euler_to_fibonacci_transform,
    fibonacci_polynomial_family,
    operator_change_of_coordinates,
)

values = (1, 2, 3, 4, 5, 6, 7, 8)
source = analyze_recurrence_coordinates(values, (1, 1), name="Fibonacci")
target = analyze_recurrence_coordinates(values, (2, -1, 3), name="target")
change = coordinate_change_between_recurrences((1, 1), (2, -1, 3), len(values))
assert change.apply(source.coordinates()) == target.coordinates()

polynomial_change = euler_to_fibonacci_transform(10)
euler_derivative = derivative_operator_in_family(euler_polynomial_family(), 10)
fibonacci_derivative = derivative_operator_in_family(fibonacci_polynomial_family(), 10)
assert operator_change_of_coordinates(euler_derivative, polynomial_change) == fibonacci_derivative
print("recurrence determinant", change.determinant)
print("polynomial determinant", polynomial_change.determinant)
