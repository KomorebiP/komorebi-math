"""Deterministic validation ledger for the coordinate calculus."""

from __future__ import annotations

import json
import random
from fractions import Fraction
from pathlib import Path

from komorebi_math.coordinates import (
    LinearRealization,
    analyze_recurrence_coordinates,
    bezout_defect_identity,
    companion_matrix,
    connection_transform,
    convert_normalized_coefficients,
    coordinate_change_between_recurrences,
    cyclic_krylov_matrix,
    derivative_operator_in_family,
    euler_fibonacci_coordinates,
    euler_polynomial,
    euler_to_fibonacci_coefficient,
    euler_to_fibonacci_transform,
    exponential_basis,
    fibonacci_companion_power,
    fibonacci_factorial_basis,
    fibonacci_polynomial,
    fibonacci_polynomial_family,
    intertwiner_space,
    monomial_polynomial_family,
    monomial_to_fibonacci_coefficient,
    normalization_change_transform,
    operator_change_of_coordinates,
    polynomial_extended_gcd,
    realization_defect,
    recover_initial_state,
    secant_euler_from_fibonacci_coordinates,
    signed_secant_euler_numbers,
    similarity_from_cyclic_vectors,
    variation_of_constants_prefix,
)
from komorebi_math.coordinates import (
    identity_matrix,
    inverse_matrix,
    matrix_multiply,
    matrix_vector_multiply,
)
from komorebi_math.coordinates import (
    euler_polynomial_family,
    polynomial_add,
    polynomial_multiply,
    polynomial_scale,
)


class Ledger:
    def __init__(self) -> None:
        self.categories: dict[str, dict[str, int]] = {}

    def check(self, category: str, condition: bool) -> None:
        bucket = self.categories.setdefault(category, {"checks": 0, "failures": 0})
        bucket["checks"] += 1
        if not condition:
            bucket["failures"] += 1
            raise AssertionError(f"validation failure in {category}")

    def payload(self) -> dict[str, object]:
        total = sum(item["checks"] for item in self.categories.values())
        failures = sum(item["failures"] for item in self.categories.values())
        return {
            "schema": "komorebi-math.block14-reference-audit",
            "package_version": __import__("komorebi_math").__version__,
            "total_checks": total,
            "total_failures": failures,
            "categories": self.categories,
        }


def add_polynomials(left, right):
    size = max(len(left), len(right))
    return tuple(
        (left[index] if index < len(left) else 0)
        + (right[index] if index < len(right) else 0)
        for index in range(size)
    )


def main() -> None:
    ledger = Ledger()
    rng = random.Random(20260817)

    for order in range(1, 6):
        for length in range(order, 26):
            for _ in range(45):
                coefficients = tuple(rng.randint(-4, 4) for _ in range(order))
                values = tuple(rng.randint(-1000, 1000) for _ in range(length))
                record = analyze_recurrence_coordinates(values, coefficients)
                reconstructed = record.reconstruct()
                green = variation_of_constants_prefix(record)
                for expected, actual in zip(values, reconstructed, strict=True):
                    ledger.check("recurrence coordinate roundtrip", expected == actual)
                for expected, actual in zip(values, green, strict=True):
                    ledger.check("companion variation of constants", expected == actual)
                ledger.check("unit triangular recurrence certificate", record.certificate().valid is True)

    systems = ((1,), (1, 1), (-1, 2), (1, 0, 1), (2, -1, 3), (-2, 1, 0, 1))
    for length in range(6, 24):
        values = tuple(rng.randint(-200, 200) for _ in range(length))
        records = {
            coefficients: analyze_recurrence_coordinates(values, coefficients, name=str(coefficients))
            for coefficients in systems
            if len(coefficients) <= length
        }
        for source, source_record in records.items():
            for target, target_record in records.items():
                transform = coordinate_change_between_recurrences(source, target, length)
                actual = transform.apply(source_record.coordinates())
                for expected, value in zip(target_record.coordinates(), actual, strict=True):
                    ledger.check("recurrence system coordinate changes", expected == value)
                ledger.check("recurrence transform determinant", transform.determinant in (1, -1))
                ledger.check("recurrence transform inversion", transform.roundtrip_valid())

    euler = signed_secant_euler_numbers(42)
    record = euler_fibonacci_coordinates(42)
    for expected, actual in zip(euler, record.reconstruct(), strict=True):
        ledger.check("Euler Fibonacci defect reconstruction", expected == actual)
    for n, defect in enumerate(record.defects):
        expected = euler[n + 2] - euler[n + 1] - euler[n]
        ledger.check("Euler Fibonacci defect formula", defect == expected)
    ledger.check("Euler is not homogeneous Fibonacci", not record.homogeneous)

    converted = convert_normalized_coefficients(euler, exponential_basis(), fibonacci_factorial_basis())
    inverse = normalization_change_transform(fibonacci_factorial_basis(), exponential_basis(), len(euler) - 1)
    recovered = inverse.apply(converted)
    for expected, actual in zip(euler, recovered, strict=True):
        ledger.check("normalized basis diagonal roundtrip", Fraction(expected) == actual)

    for degree in range(31):
        power = fibonacci_companion_power(degree)
        if degree >= 1:
            ledger.check("Fibonacci polynomial companion power", power[0][0] == fibonacci_polynomial(degree + 1))
            ledger.check("Fibonacci polynomial companion power", power[0][1] == fibonacci_polynomial(degree))
            ledger.check("Fibonacci polynomial companion power", power[1][0] == fibonacci_polynomial(degree))
            ledger.check("Fibonacci polynomial companion power", power[1][1] == fibonacci_polynomial(degree - 1))
        reconstructed = (Fraction(0),)
        for k in range(degree + 1):
            reconstructed = add_polynomials(
                reconstructed,
                polynomial_scale(fibonacci_polynomial(k + 1), monomial_to_fibonacci_coefficient(degree, k)),
            )
        expected = tuple(Fraction(int(index == degree)) for index in range(degree + 1))
        for left, right in zip(reconstructed, expected, strict=True):
            ledger.check("monomial Fibonacci inverse formula", left == right)

    for degree in range(25):
        transform = euler_to_fibonacci_transform(degree)
        for n in range(degree + 1):
            reconstructed = (Fraction(0),)
            for k in range(n + 1):
                coefficient = transform.matrix[k][n]
                ledger.check("Euler Fibonacci explicit connection", coefficient == euler_to_fibonacci_coefficient(n, k))
                reconstructed = add_polynomials(
                    reconstructed,
                    polynomial_scale(fibonacci_polynomial(k + 1), coefficient),
                )
            expected = euler_polynomial(n)
            for left, right in zip(reconstructed, expected, strict=True):
                ledger.check("Euler Fibonacci polynomial reconstruction", left == right)
        ledger.check(
            "secant Euler Fibonacci specialization",
            secant_euler_from_fibonacci_coordinates(degree) == euler[degree],
        )

    for degree in range(1, 20):
        euler_family = euler_polynomial_family()
        fibonacci_family = fibonacci_polynomial_family()
        monomial_family = monomial_polynomial_family()
        euler_to_monomial = connection_transform(euler_family, monomial_family, degree)
        monomial_to_fibonacci = connection_transform(monomial_family, fibonacci_family, degree)
        direct = connection_transform(euler_family, fibonacci_family, degree)
        ledger.check(
            "graded connection composition",
            euler_to_monomial.then(monomial_to_fibonacci).matrix == direct.matrix,
        )
        derivative_euler = derivative_operator_in_family(euler_family, degree)
        derivative_fibonacci = derivative_operator_in_family(fibonacci_family, degree)
        ledger.check(
            "operator conjugation",
            operator_change_of_coordinates(derivative_euler, direct) == derivative_fibonacci,
        )

    fibonacci_companion = companion_matrix((1, 1))
    for horizon in range(2, 30):
        recovery = recover_initial_state(euler[:horizon], fibonacci_companion, (1, 0))
        ledger.check("Euler unforced Fibonacci obstruction", recovery.exact == (horizon <= 2))
        ledger.check("Fibonacci state remains initial Euler pair", recovery.state == (1, 0))

    source = fibonacci_companion
    changes = (
        ((1, 2), (3, 5)),
        ((2, 1), (1, 1)),
        ((-1, 3), (2, -5)),
    )
    for change in changes:
        inverse_change = inverse_matrix(tuple(tuple(Fraction(value) for value in row) for row in change))
        target = matrix_multiply(change, matrix_multiply(source, inverse_change))
        space = intertwiner_space(source, target)
        ledger.check("intertwiner space dimension", space.dimension == 2)
        target_vector = matrix_vector_multiply(change, (1, 0))
        recovered_change = similarity_from_cyclic_vectors(source, target, (1, 0), target_vector)
        for expected_row, actual_row in zip(change, recovered_change, strict=True):
            for expected, actual in zip(expected_row, actual_row, strict=True):
                ledger.check("cyclic similarity recovery", Fraction(expected) == actual)

    recurrence = companion_matrix((2, -1, 3))
    realization = LinearRealization(recurrence, (1, 2, 4), (1, 0, 0), "source")
    defect_realization = realization_defect(realization, (1, 1))
    values = realization.prefix(100)
    direct = tuple(values[n + 2] - values[n + 1] - values[n] for n in range(98))
    for expected, actual in zip(direct, defect_realization.prefix(98), strict=True):
        ledger.check("source realization defect realization", expected == actual)

    polynomial_pairs = (
        ((-1, -1, 1), (-1, 0, 1)),
        ((1, 0, 1), (1, 1)),
        ((-2, 1), (1, 1, 1)),
    )
    for left, right in polynomial_pairs:
        gcd, u, v = polynomial_extended_gcd(left, right)
        identity = polynomial_add(polynomial_multiply(u, left), polynomial_multiply(v, right))
        ledger.check("Bezout polynomial identity", gcd == (1,) and identity == (1,))
        for length in range(12, 45):
            values = tuple(rng.randint(-300, 300) for _ in range(length))
            result = bezout_defect_identity(values, left, right)
            ledger.check("Bezout defect reconstruction", result.valid)
            for expected, actual in zip(result.expected, result.reconstructed, strict=True):
                ledger.check("Bezout defect reconstruction coefficients", expected == actual)

    payload = ledger.payload()
    payload["Euler_Fibonacci_record"] = {
        "initial": [str(value) for value in record.initial],
        "first_defects": [str(value) for value in record.defects[:16]],
        "defect_EGF": "sech(t)*tanh(t)-2*sech(t)^3",
    }
    payload["current_wheel_audit"] = {
        "normalized_basis_change_present": True,
        "Fibonacci_companion_present": True,
        "general_recurrence_coordinate_change_present_before_this_research": False,
        "Euler_to_Fibonacci_forced_companion_present_before_this_research": False,
    }
    output = Path(__file__).resolve().parent / "BLOCK14_REFERENCE_AUDIT.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **payload}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
