"""Independent deterministic audit of the general coordinate atlas."""

from __future__ import annotations

import json
import random
from fractions import Fraction
from pathlib import Path

import numpy as np

import komorebi_math
import komorebi_math.coordinates as cc


class Ledger:
    """Accumulate exact comparison counts by category."""

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
            "schema": "komorebi-math.block14-general-coordinate-audit.v1",
            "package_version": komorebi_math.__version__,
            "categories": self.categories,
            "total_checks": sum(item["checks"] for item in self.categories.values()),
            "total_failures": sum(item["failures"] for item in self.categories.values()),
        }


def _random_invertible_matrix(rng: random.Random, dimension: int) -> cc.Matrix:
    while True:
        matrix = tuple(
            tuple(Fraction(rng.randint(-4, 4)) for _ in range(dimension))
            for _ in range(dimension)
        )
        if cc.determinant(matrix) != 0:
            return matrix


def _direct_matrix_vector(matrix: cc.Matrix, vector: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    return tuple(sum(row[index] * vector[index] for index in range(len(vector))) for row in matrix)


def _direct_defects(
    values: tuple[Fraction, ...],
    coefficients: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    order = len(coefficients)
    return tuple(
        values[index + order]
        - sum(coefficients[offset] * values[index + offset] for offset in range(order))
        for index in range(len(values) - order)
    )


def _direct_polynomial_value(polynomial: tuple[Fraction, ...], point: Fraction) -> Fraction:
    value = Fraction(0)
    power = Fraction(1)
    for coefficient in polynomial:
        value += coefficient * power
        power *= point
    return value


def main() -> None:
    ledger = Ledger()
    rng = random.Random(2026081801)

    for dimension in range(1, 7):
        standard = cc.standard_basis(dimension, name=f"standard-{dimension}")
        for trial in range(120):
            matrix = _random_invertible_matrix(rng, dimension)
            target = cc.basis_from_matrix(f"basis-{dimension}-{trial}", matrix)
            transform = standard.connection_to(target)
            expected = cc.inverse_matrix(matrix)
            for expected_row, actual_row in zip(expected, transform.matrix, strict=True):
                for left, right in zip(expected_row, actual_row, strict=True):
                    ledger.check("arbitrary basis connection", left == right)
            vector = tuple(Fraction(rng.randint(-100, 100)) for _ in range(dimension))
            target_coordinates = transform.apply(vector)
            recovered = transform.inverse().apply(target_coordinates)
            for left, right in zip(vector, recovered, strict=True):
                ledger.check("arbitrary basis roundtrip", left == right)

    for dimension in range(2, 7):
        for trial in range(100):
            ab = cc.CoordinateTransform("A", "B", _random_invertible_matrix(rng, dimension), "audit")
            bc = cc.CoordinateTransform("B", "C", _random_invertible_matrix(rng, dimension), "audit")
            cd = cc.CoordinateTransform("C", "D", _random_invertible_matrix(rng, dimension), "audit")
            atlas = cc.CoordinateAtlas(f"atlas-{dimension}-{trial}")
            for name in "ABCD":
                atlas.register_system(cc.CoordinateSystem(name, dimension))
            for edge in (ab, bc, cd):
                atlas.register_transform(edge, include_inverse=True)
            direct = ab.then(bc).then(cd)
            atlas.register_transform(direct)
            resolved = atlas.compose_path(("A", "B", "C", "D")).transform
            for expected_row, actual_row in zip(direct.matrix, resolved.matrix, strict=True):
                for left, right in zip(expected_row, actual_row, strict=True):
                    ledger.check("atlas path composition", left == right)
            ledger.check("atlas commuting diagram", atlas.compare_paths(("A", "D"), ("A", "B", "C", "D")).valid)

    for size in range(1, 26):
        transforms = (
            cc.falling_factorial_to_monomial_transform(size),
            cc.rising_factorial_to_monomial_transform(size),
            cc.binomial_transform(size),
            cc.forward_difference_transform(size),
        )
        for transform in transforms:
            identity = cc.identity_matrix(transform.source_dimension)
            product = cc.matrix_multiply(transform.inverse().matrix, transform.matrix)
            for expected_row, actual_row in zip(identity, product, strict=True):
                for left, right in zip(expected_row, actual_row, strict=True):
                    ledger.check("combinatorial inverse identity", left == right)

    catalog = cc.classical_polynomial_catalog()
    points = tuple(Fraction(value, 3) for value in range(-6, 7))
    for family in catalog.families:
        transform = cc.connection_transform(family, cc.monomial_polynomial_family(), 10)
        inverse = transform.inverse()
        for degree in range(11):
            unit = tuple(Fraction(int(index == degree)) for index in range(11))
            monomial_coordinates = transform.apply(unit)
            recovered = inverse.apply(monomial_coordinates)
            for left, right in zip(unit, recovered, strict=True):
                ledger.check("classical family coordinate roundtrip", left == right)
            polynomial = family.polynomial(degree)
            for point in points:
                expected = _direct_polynomial_value(polynomial, point)
                actual = sum(monomial_coordinates[index] * point**index for index in range(11))
                ledger.check("classical family evaluation", expected == actual)

    recurrence_models = tuple(cc.recurrence_model_catalog())
    for length in range(8, 31):
        values = tuple(Fraction(rng.randint(-1000, 1000)) for _ in range(length))
        for source in recurrence_models:
            if source.order > length:
                continue
            source_record = source.analyze(values)
            for target in recurrence_models:
                if target.order > length:
                    continue
                transform = source.coordinate_transform_to(target, length)
                actual = transform.apply(source_record.coordinates())
                expected = values[: target.order] + _direct_defects(values, target.coefficients)
                for left, right in zip(expected, actual, strict=True):
                    ledger.check("general recurrence coordinates", left == right)

    for dimension in range(1, 6):
        for _ in range(120):
            matrix = _random_invertible_matrix(rng, dimension)
            offset = tuple(Fraction(rng.randint(-20, 20)) for _ in range(dimension))
            transform = cc.AffineCoordinateTransform("x", "y", matrix, offset, "audit")
            vector = tuple(Fraction(rng.randint(-50, 50)) for _ in range(dimension))
            recovered = transform.inverse().apply(transform.apply(vector))
            for left, right in zip(vector, recovered, strict=True):
                ledger.check("affine coordinate roundtrip", left == right)

    for degree in range(1, 13):
        for _ in range(100):
            coefficients = [Fraction(0), Fraction(rng.choice((-3, -2, -1, 1, 2, 3)))]
            coefficients.extend(Fraction(rng.randint(-4, 4)) for _ in range(degree - 1))
            change = cc.FormalCoordinateChange("phi", tuple(coefficients), "u", "z")
            inverse = change.inverse()
            identity = cc.identity_formal_coordinate(degree, coordinate="u")
            composed = inverse.compose(change, degree=degree)
            for left, right in zip(identity.coefficients, composed.coefficients, strict=True):
                ledger.check("formal coordinate inverse", left == right)

    for dimension in range(1, 6):
        for _ in range(120):
            transform = cc.CoordinateTransform("old", "new", _random_invertible_matrix(rng, dimension), "audit")
            vector = tuple(Fraction(rng.randint(-30, 30)) for _ in range(dimension))
            covector = tuple(Fraction(rng.randint(-30, 30)) for _ in range(dimension))
            ledger.check("vector covector pairing", cc.pairing_invariant(covector, vector, transform))
            operator = tuple(
                tuple(Fraction(rng.randint(-5, 5)) for _ in range(dimension))
                for _ in range(dimension)
            )
            transported = cc.transport_operator(operator, transform)
            left = transform.apply(_direct_matrix_vector(operator, vector))
            right = _direct_matrix_vector(transported, transform.apply(vector))
            for expected, actual in zip(left, right, strict=True):
                ledger.check("operator transport", expected == actual)

    for prime in (2, 3, 5, 7, 11):
        for dimension in range(1, 7):
            for _ in range(80):
                while True:
                    matrix = tuple(
                        tuple(rng.randrange(prime) for _ in range(dimension))
                        for _ in range(dimension)
                    )
                    transform = cc.FiniteFieldCoordinateTransform("a", "b", matrix, prime, "audit")
                    if transform.certificate().valid:
                        break
                values = tuple(rng.randrange(prime) for _ in range(dimension))
                recovered = transform.inverse().apply(transform.apply(values))
                for left, right in zip(values, recovered, strict=True):
                    ledger.check("finite field coordinate roundtrip", left % prime == right)

    numerical_rng = np.random.default_rng(2026081801)
    for size in (1, 2, 4, 8, 16, 32):
        transforms = [cc.discrete_fourier_transform(size), cc.discrete_cosine_transform(size)]
        if size & (size - 1) == 0:
            transforms.extend((cc.walsh_hadamard_transform(size), cc.haar_wavelet_transform(size)))
        for transform in transforms:
            for _ in range(60):
                values = numerical_rng.normal(size=size)
                recovered = transform.inverse().apply(transform.apply(values))
                ledger.check("spectral numerical roundtrip", bool(np.allclose(values, recovered, atol=1e-10, rtol=1e-10)))

    for size in range(2, 65):
        values = tuple(range(size))
        for modulus in range(2, min(size, 9) + 1):
            transform = cc.residue_class_transform(size, modulus)
            ledger.check("residue packet permutation", transform.inverse().apply(transform.apply(values)) == values)
        if size & (size - 1) == 0:
            transform = cc.digit_reversal_transform(size)
            ledger.check("digit reversal permutation", transform.inverse().apply(transform.apply(values)) == values)

    payload = ledger.payload()
    output = Path(__file__).resolve().parent / "BLOCK14A2_GENERAL_COORDINATE_AUDIT.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **payload}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
