"""Independent reference checks for the coordinate calculus."""

from __future__ import annotations

import json
import random
from fractions import Fraction
from math import comb, factorial
from pathlib import Path

import komorebi_math.coordinates as cc


class Ledger:
    def __init__(self) -> None:
        self.categories: dict[str, dict[str, int]] = {}

    def check(self, category: str, condition: bool) -> None:
        bucket = self.categories.setdefault(category, {"checks": 0, "failures": 0})
        bucket["checks"] += 1
        if not condition:
            bucket["failures"] += 1
            raise AssertionError(category)

    def payload(self) -> dict[str, object]:
        return {
            "schema": "komorebi-math.block14-deep-audit",
            "package_version": __import__("komorebi_math").__version__,
            "categories": self.categories,
            "total_checks": sum(v["checks"] for v in self.categories.values()),
            "total_failures": sum(v["failures"] for v in self.categories.values()),
        }


def direct_defects(values: tuple[Fraction, ...], coefficients: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    order = len(coefficients)
    return tuple(
        values[n + order]
        - sum(coefficients[j] * values[n + j] for j in range(order))
        for n in range(len(values) - order)
    )


def direct_synthesis(initial: tuple[Fraction, ...], defects: tuple[Fraction, ...], coefficients: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    values = list(initial)
    order = len(coefficients)
    for n, defect in enumerate(defects):
        values.append(sum(coefficients[j] * values[n + j] for j in range(order)) + defect)
    return tuple(values)


def fibonacci(count: int) -> tuple[int, ...]:
    result: list[int] = []
    a, b = 0, 1
    for _ in range(count):
        result.append(a)
        a, b = b, a + b
    return tuple(result)


def add_poly(a: tuple[Fraction, ...], b: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    size = max(len(a), len(b))
    values = [
        (a[i] if i < len(a) else Fraction(0))
        + (b[i] if i < len(b) else Fraction(0))
        for i in range(size)
    ]
    while len(values) > 1 and values[-1] == 0:
        values.pop()
    return tuple(values)


def scale_poly(a: tuple[Fraction, ...], scalar: Fraction) -> tuple[Fraction, ...]:
    return tuple(scalar * value for value in a)


def independent_fibonacci_polynomial(index: int) -> tuple[Fraction, ...]:
    if index == 0:
        return (Fraction(0),)
    previous = (Fraction(0),)
    current = (Fraction(1),)
    for _ in range(1, index):
        shifted = (Fraction(0),) + current
        previous, current = current, add_poly(shifted, previous)
    return current


def independent_euler_zero(count: int) -> tuple[Fraction, ...]:
    values = [Fraction(1)]
    for n in range(1, count):
        values.append(-sum(Fraction(comb(n, k)) * values[k] for k in range(n)) / 2)
    return tuple(values)


def independent_euler_polynomial(n: int) -> tuple[Fraction, ...]:
    zeros = independent_euler_zero(n + 1)
    return tuple(Fraction(comb(n, k)) * zeros[n - k] for k in range(n + 1))


def triangular_coordinates(polynomial: tuple[Fraction, ...], family: tuple[tuple[Fraction, ...], ...]) -> tuple[Fraction, ...]:
    residual = list(polynomial)
    coordinates = [Fraction(0) for _ in family]
    for degree in range(len(family) - 1, -1, -1):
        coefficient = residual[degree]
        coordinates[degree] = coefficient
        basis = family[degree]
        for index, value in enumerate(basis):
            residual[index] -= coefficient * value
    if any(residual):
        raise ArithmeticError("triangular resolution failed")
    return tuple(coordinates)


def main() -> None:
    ledger = Ledger()
    rng = random.Random(2026081703)

    for order in range(1, 6):
        for length in range(order, 34):
            for _ in range(35):
                coefficients = tuple(Fraction(rng.randint(-5, 5)) for _ in range(order))
                values = tuple(Fraction(rng.randint(-10000, 10000)) for _ in range(length))
                expected_defects = direct_defects(values, coefficients)
                record = cc.analyze_recurrence_coordinates(values, coefficients)
                for expected, actual in zip(expected_defects, record.defects, strict=True):
                    ledger.check("independent recurrence analysis", expected == actual)
                expected_values = direct_synthesis(values[:order], expected_defects, coefficients)
                for expected, actual in zip(expected_values, record.reconstruct(), strict=True):
                    ledger.check("independent recurrence synthesis", expected == actual)

    for length in range(4, 28):
        for _ in range(40):
            values = tuple(Fraction(rng.randint(-5000, 5000)) for _ in range(length))
            source = tuple(Fraction(rng.randint(-3, 3)) for _ in range(2))
            target = tuple(Fraction(rng.randint(-3, 3)) for _ in range(2))
            source_record = cc.analyze_recurrence_coordinates(values, source)
            expected_target = values[:2] + direct_defects(values, target)
            transform = cc.coordinate_change_between_recurrences(source, target, length)
            actual_target = transform.apply(source_record.coordinates())
            for expected, actual in zip(expected_target, actual_target, strict=True):
                ledger.check("independent recurrence coordinate change", expected == actual)

    euler = cc.signed_secant_euler_numbers(80)
    fib = fibonacci(82)
    record = cc.euler_fibonacci_coordinates(80)
    for n, defect in enumerate(record.defects):
        ledger.check("independent Euler defect", defect == euler[n + 2] - euler[n + 1] - euler[n])
    for n in range(1, 80):
        reconstructed = Fraction(fib[n - 1]) + sum(
            Fraction(fib[n - 1 - j]) * record.defects[j]
            for j in range(n - 1)
        )
        ledger.check("independent Euler Green reconstruction", reconstructed == euler[n])

    family = tuple(independent_fibonacci_polynomial(k + 1) for k in range(31))
    for n in range(31):
        independent = independent_fibonacci_polynomial(n + 1)
        for expected, actual in zip(independent, cc.fibonacci_polynomial(n + 1), strict=True):
            ledger.check("independent Fibonacci polynomials", expected == actual)
        polynomial = independent_euler_polynomial(n)
        coordinates = triangular_coordinates(polynomial, family[: n + 1])
        for k, expected in enumerate(coordinates):
            ledger.check(
                "independent Euler Fibonacci polynomial coordinates",
                expected == cc.euler_to_fibonacci_coefficient(n, k),
            )

    for degree in range(0, 45):
        source_weight = Fraction(factorial(degree))
        fibs = fibonacci(degree + 2)
        target_weight = Fraction(1)
        for index in range(1, degree + 1):
            target_weight *= fibs[index]
        coefficient = Fraction(rng.randint(-100000, 100000))
        expected = target_weight / source_weight * coefficient
        actual = cc.convert_normalized_coefficients(
            tuple(Fraction(0) for _ in range(degree)) + (coefficient,),
            cc.exponential_basis(),
            cc.fibonacci_factorial_basis(),
        )[-1]
        ledger.check("independent normalized basis scaling", expected == actual)

    payload = ledger.payload()
    output = Path(__file__).resolve().parent / "BLOCK14_DEEP_AUDIT.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **payload}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
