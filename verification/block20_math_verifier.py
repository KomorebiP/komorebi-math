"""Independent exact-mathematics verifier for Komorebi Math Block 20.

This verifier is intentionally independent of the production expression and
polynomial-family implementations.  It uses only standard-library rational
arithmetic, local polynomial formulas, and small local matrix routines.

Examples
--------
python verification/block20_math_verifier.py
python verification/block20_math_verifier.py --output report.json
python verification/block20_math_verifier.py --quiet --output report.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from hashlib import sha256
from itertools import product
from math import comb
from pathlib import Path

Scalar = Fraction
Vector = tuple[Scalar, ...]
Matrix = tuple[tuple[Scalar, ...], ...]
Polynomial = tuple[Scalar, ...]


class SingularMatrixError(ValueError):
    """Raised by the local exact inverse when a matrix is singular."""


def _fraction(value: int | Fraction) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _matrix(rows: Iterable[Iterable[int | Fraction]]) -> Matrix:
    value = tuple(tuple(_fraction(entry) for entry in row) for row in rows)
    if not value or not value[0]:
        raise ValueError("a local verification matrix must be nonempty")
    width = len(value[0])
    if any(len(row) != width for row in value):
        raise ValueError("a local verification matrix must be rectangular")
    return value


def _shape(value: Matrix) -> tuple[int, int]:
    return len(value), len(value[0])


def _zeros(rows: int, columns: int) -> Matrix:
    if rows < 1 or columns < 1:
        raise ValueError("local verification matrices require positive dimensions")
    return tuple(tuple(Fraction(0) for _ in range(columns)) for _ in range(rows))


def _identity(size: int) -> Matrix:
    if size < 1:
        raise ValueError("identity size must be positive")
    return tuple(
        tuple(Fraction(int(row == column)) for column in range(size))
        for row in range(size)
    )


def _transpose(value: Matrix) -> Matrix:
    rows, columns = _shape(value)
    return tuple(tuple(value[row][column] for row in range(rows)) for column in range(columns))


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    left_rows, left_columns = _shape(left)
    right_rows, right_columns = _shape(right)
    if left_columns != right_rows:
        raise ValueError(
            f"matrix dimensions do not compose: {left_rows}x{left_columns} and "
            f"{right_rows}x{right_columns}"
        )
    return tuple(
        tuple(
            sum(
                (left[row][middle] * right[middle][column] for middle in range(left_columns)),
                Fraction(0),
            )
            for column in range(right_columns)
        )
        for row in range(left_rows)
    )


def _matrix_add(left: Matrix, right: Matrix) -> Matrix:
    if _shape(left) != _shape(right):
        raise ValueError("matrix dimensions must agree")
    return tuple(
        tuple(a + b for a, b in zip(left_row, right_row, strict=True))
        for left_row, right_row in zip(left, right, strict=True)
    )


def _matrix_subtract(left: Matrix, right: Matrix) -> Matrix:
    if _shape(left) != _shape(right):
        raise ValueError("matrix dimensions must agree")
    return tuple(
        tuple(a - b for a, b in zip(left_row, right_row, strict=True))
        for left_row, right_row in zip(left, right, strict=True)
    )


def _matrix_scale(value: Matrix, scalar: int | Fraction) -> Matrix:
    factor = _fraction(scalar)
    return tuple(tuple(factor * entry for entry in row) for row in value)


def _matvec(value: Matrix, vector: Sequence[int | Fraction]) -> Vector:
    data = tuple(_fraction(entry) for entry in vector)
    rows, columns = _shape(value)
    if len(data) != columns:
        raise ValueError("matrix and vector dimensions do not agree")
    return tuple(
        sum((value[row][column] * data[column] for column in range(columns)), Fraction(0))
        for row in range(rows)
    )


def _row_times_matrix(row: Sequence[int | Fraction], value: Matrix) -> Vector:
    data = tuple(_fraction(entry) for entry in row)
    rows, columns = _shape(value)
    if len(data) != rows:
        raise ValueError("row and matrix dimensions do not agree")
    return tuple(
        sum((data[index] * value[index][column] for index in range(rows)), Fraction(0))
        for column in range(columns)
    )


def _dot(left: Sequence[int | Fraction], right: Sequence[int | Fraction]) -> Fraction:
    a = tuple(_fraction(entry) for entry in left)
    b = tuple(_fraction(entry) for entry in right)
    if len(a) != len(b):
        raise ValueError("vector dimensions do not agree")
    return sum((x * y for x, y in zip(a, b, strict=True)), Fraction(0))


def _outer(column: Sequence[int | Fraction], row: Sequence[int | Fraction]) -> Matrix:
    left = tuple(_fraction(entry) for entry in column)
    right = tuple(_fraction(entry) for entry in row)
    if not left or not right:
        raise ValueError("outer-product vectors must be nonempty")
    return tuple(tuple(a * b for b in right) for a in left)


def _inverse(value: Matrix) -> Matrix:
    rows, columns = _shape(value)
    if rows != columns:
        raise ValueError("only square matrices can be inverted")
    augmented = [list(value[row]) + list(_identity(rows)[row]) for row in range(rows)]
    for pivot_column in range(columns):
        pivot_row = next(
            (row for row in range(pivot_column, rows) if augmented[row][pivot_column] != 0),
            None,
        )
        if pivot_row is None:
            raise SingularMatrixError("matrix is singular")
        if pivot_row != pivot_column:
            augmented[pivot_column], augmented[pivot_row] = (
                augmented[pivot_row],
                augmented[pivot_column],
            )
        pivot = augmented[pivot_column][pivot_column]
        augmented[pivot_column] = [entry / pivot for entry in augmented[pivot_column]]
        for row in range(rows):
            if row == pivot_column:
                continue
            factor = augmented[row][pivot_column]
            if factor == 0:
                continue
            augmented[row] = [
                entry - factor * pivot_entry
                for entry, pivot_entry in zip(
                    augmented[row], augmented[pivot_column], strict=True
                )
            ]
    return tuple(tuple(row[columns:]) for row in augmented)


def _polynomial_add(left: Polynomial, right: Polynomial) -> Polynomial:
    size = max(len(left), len(right))
    output = tuple(
        (left[index] if index < len(left) else Fraction(0))
        + (right[index] if index < len(right) else Fraction(0))
        for index in range(size)
    )
    return _trim(output)


def _polynomial_scale(value: Polynomial, scalar: int | Fraction) -> Polynomial:
    factor = _fraction(scalar)
    return _trim(tuple(factor * entry for entry in value))


def _polynomial_multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    output = [Fraction(0) for _ in range(len(left) + len(right) - 1)]
    for left_degree, left_value in enumerate(left):
        for right_degree, right_value in enumerate(right):
            output[left_degree + right_degree] += left_value * right_value
    return _trim(tuple(output))


def _trim(value: Sequence[int | Fraction]) -> Polynomial:
    output = [_fraction(entry) for entry in value]
    if not output:
        return (Fraction(0),)
    while len(output) > 1 and output[-1] == 0:
        output.pop()
    return tuple(output)


def _polynomial_evaluate(value: Polynomial, point: int | Fraction) -> Fraction:
    x = _fraction(point)
    output = Fraction(0)
    for coefficient in reversed(value):
        output = output * x + coefficient
    return output


def _polynomial_compose_affine(
    value: Polynomial,
    *,
    scale: int | Fraction,
    shift: int | Fraction,
) -> Polynomial:
    affine = (_fraction(shift), _fraction(scale))
    output = (Fraction(0),)
    for coefficient in reversed(value):
        output = _polynomial_add(_polynomial_multiply(output, affine), (coefficient,))
    return output


def _monomial_polynomial(degree: int) -> Polynomial:
    return tuple(Fraction(int(power == degree)) for power in range(degree + 1))


@cache
def _euler_at_zero(index: int) -> Fraction:
    if index == 0:
        return Fraction(1)
    return -sum(
        (Fraction(comb(index, lower)) * _euler_at_zero(lower) for lower in range(index)),
        Fraction(0),
    ) / 2


def _euler_polynomial(degree: int) -> Polynomial:
    return tuple(
        Fraction(comb(degree, power)) * _euler_at_zero(degree - power)
        for power in range(degree + 1)
    )


@cache
def _fibonacci_polynomial(index: int) -> Polynomial:
    if index == 0:
        return (Fraction(0),)
    if index == 1:
        return (Fraction(1),)
    previous = _fibonacci_polynomial(index - 2)
    current = _fibonacci_polynomial(index - 1)
    shifted = (Fraction(0),) + current
    return _polynomial_add(shifted, previous)


def _fibonacci_basis_polynomial(degree: int) -> Polynomial:
    # The graded basis is G_n(x) = F_(n+1)(x).
    return _fibonacci_polynomial(degree + 1)


@dataclass(frozen=True, slots=True)
class _Family:
    key: str
    display_name: str
    polynomial: Callable[[int], Polynomial]


MONOMIAL = _Family("M", "monomials", _monomial_polynomial)
EULER = _Family("E", "Euler polynomials", _euler_polynomial)
FIBONACCI = _Family("F", "Fibonacci polynomials", _fibonacci_basis_polynomial)
FAMILIES = (MONOMIAL, EULER, FIBONACCI)


def _basis_matrix(family: _Family, degree: int) -> Matrix:
    columns: list[Vector] = []
    for basis_degree in range(degree + 1):
        polynomial = family.polynomial(basis_degree)
        _require(
            len(polynomial) == basis_degree + 1 and polynomial[-1] != 0,
            f"{family.key}_{basis_degree} is not a graded basis polynomial",
        )
        columns.append(
            polynomial + tuple(Fraction(0) for _ in range(degree + 1 - len(polynomial)))
        )
    return _transpose(tuple(columns))


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _basis_fingerprint(display_name: str, value: Matrix) -> str:
    rows, columns = _shape(value)
    payload = {
        "schema": "komorebi-math.independent-polynomial-basis-fingerprint",
        "schema_version": 1,
        "display_name": str(display_name),
        "rows": rows,
        "columns": columns,
        "coefficients": [[_fraction_text(entry) for entry in row] for row in value],
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _change(source: _Family, target: _Family, degree: int) -> Matrix:
    # Coordinates transform as c_target = B_target^(-1) B_source c_source.
    return _matmul(_inverse(_basis_matrix(target, degree)), _basis_matrix(source, degree))


def _monomial_derivative(degree: int) -> Matrix:
    if degree < 1:
        raise ValueError("the derivative test degree must be positive")
    return tuple(
        tuple(
            Fraction(column) if row == column - 1 else Fraction(0)
            for column in range(degree + 1)
        )
        for row in range(degree)
    )


def _derivative(family: _Family, degree: int) -> Matrix:
    return _matmul(
        _inverse(_basis_matrix(family, degree - 1)),
        _matmul(_monomial_derivative(degree), _basis_matrix(family, degree)),
    )


def _monomial_basepoint_integral(degree: int, basepoint: int | Fraction) -> Matrix:
    """Map degree <= degree-1 to its degree <= degree primitive vanishing at a."""

    if degree < 1:
        raise ValueError("the integration test degree must be positive")
    a = _fraction(basepoint)
    output = [[Fraction(0) for _ in range(degree)] for _ in range(degree + 1)]
    for column in range(degree):
        divisor = Fraction(column + 1)
        output[0][column] = -(a ** (column + 1)) / divisor
        output[column + 1][column] = Fraction(1, column + 1)
    return tuple(tuple(row) for row in output)


def _basepoint_integral(
    family: _Family,
    degree: int,
    basepoint: int | Fraction,
) -> Matrix:
    return _matmul(
        _inverse(_basis_matrix(family, degree)),
        _matmul(
            _monomial_basepoint_integral(degree, basepoint),
            _basis_matrix(family, degree - 1),
        ),
    )


def _evaluation_row(family: _Family, degree: int, point: int | Fraction) -> Vector:
    x = _fraction(point)
    monomial_row = tuple(x**power for power in range(degree + 1))
    return _row_times_matrix(monomial_row, _basis_matrix(family, degree))


def _integration_row(
    family: _Family,
    degree: int,
    lower: int | Fraction,
    upper: int | Fraction,
    *,
    density: int | Fraction = 1,
) -> Vector:
    a = _fraction(lower)
    b = _fraction(upper)
    rho = _fraction(density)
    monomial_row = tuple(
        rho * (b ** (power + 1) - a ** (power + 1)) / Fraction(power + 1)
        for power in range(degree + 1)
    )
    return _row_times_matrix(monomial_row, _basis_matrix(family, degree))


def _constant_coordinates(family: _Family, degree: int) -> Vector:
    monomial_constant = (Fraction(1),) + tuple(Fraction(0) for _ in range(degree))
    return _matvec(_inverse(_basis_matrix(family, degree)), monomial_constant)


def _matrix_chain_dynamic(dimensions: Sequence[int]) -> tuple[int, tuple[tuple[int, ...], ...]]:
    values = tuple(int(value) for value in dimensions)
    if len(values) < 2 or any(value < 1 for value in values):
        raise ValueError("a matrix chain requires positive dimensions")
    count = len(values) - 1
    costs = [[0 for _ in range(count)] for _ in range(count)]
    splits = [[-1 for _ in range(count)] for _ in range(count)]
    for length in range(2, count + 1):
        for left in range(count - length + 1):
            right = left + length - 1
            candidates = []
            for split in range(left, right):
                cost = (
                    costs[left][split]
                    + costs[split + 1][right]
                    + values[left] * values[split + 1] * values[right + 1]
                )
                candidates.append((cost, split))
            costs[left][right], splits[left][right] = min(candidates)
    return costs[0][count - 1], tuple(tuple(row) for row in splits)


def _matrix_chain_exhaustive(dimensions: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in dimensions)
    if len(values) < 2 or any(value < 1 for value in values):
        raise ValueError("a matrix chain requires positive dimensions")
    count = len(values) - 1

    @cache
    def enumerate_costs(left: int, right: int) -> tuple[int, ...]:
        if left == right:
            return (0,)
        output: list[int] = []
        for split in range(left, right):
            multiplication = values[left] * values[split + 1] * values[right + 1]
            for left_cost, right_cost in product(
                enumerate_costs(left, split), enumerate_costs(split + 1, right)
            ):
                output.append(left_cost + right_cost + multiplication)
        return tuple(output)

    return enumerate_costs(0, count - 1)


def _check_structural_fingerprints() -> None:
    first = _matrix(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    second = _matrix(((1, 0, 0), (0, 1, 1), (0, 0, 1)))
    first_digest = _basis_fingerprint("same visible name", first)
    repeated_digest = _basis_fingerprint("same visible name", first)
    second_digest = _basis_fingerprint("same visible name", second)
    _require(first_digest == repeated_digest, "basis fingerprints are not deterministic")
    _require(first_digest != second_digest, "same-name structural bases collided")


def _check_change_cocycle_and_inverses() -> None:
    for degree in range(5):
        for source, middle, target in product(FAMILIES, repeat=3):
            source_to_middle = _change(source, middle, degree)
            middle_to_target = _change(middle, target, degree)
            source_to_target = _change(source, target, degree)
            _require(
                _matmul(middle_to_target, source_to_middle) == source_to_target,
                f"change cocycle failed through degree {degree}: "
                f"{source.key}->{middle.key}->{target.key}",
            )
        for source, target in product(FAMILIES, repeat=2):
            forward = _change(source, target, degree)
            backward = _change(target, source, degree)
            size = degree + 1
            _require(
                _matmul(backward, forward) == _identity(size)
                and _matmul(forward, backward) == _identity(size),
                f"change inverse failed through degree {degree}: {source.key}<->{target.key}",
            )


def _check_derivative_after_integral() -> None:
    for family in FAMILIES:
        for degree in range(1, 5):
            derivative = _derivative(family, degree)
            for basepoint in (Fraction(-2), Fraction(0), Fraction(3, 2)):
                primitive = _basepoint_integral(family, degree, basepoint)
                _require(
                    _matmul(derivative, primitive) == _identity(degree),
                    f"D J_a != I for {family.key}, degree={degree}, a={basepoint}",
                )


def _check_integral_after_derivative() -> None:
    for family in FAMILIES:
        for degree in range(1, 5):
            derivative = _derivative(family, degree)
            unit = _constant_coordinates(family, degree)
            for basepoint in (Fraction(-2), Fraction(0), Fraction(3, 2)):
                primitive = _basepoint_integral(family, degree, basepoint)
                evaluation = _evaluation_row(family, degree, basepoint)
                expected = _matrix_subtract(_identity(degree + 1), _outer(unit, evaluation))
                _require(
                    _matmul(primitive, derivative) == expected,
                    f"J_a D != I-u e_a for {family.key}, degree={degree}, a={basepoint}",
                )


def _check_derivative_naturality() -> None:
    for degree in range(1, 5):
        for source, target in product(FAMILIES, repeat=2):
            left = _matmul(_derivative(target, degree), _change(source, target, degree))
            right = _matmul(_change(source, target, degree - 1), _derivative(source, degree))
            _require(
                left == right,
                f"derivative naturality failed: {source.key}->{target.key}, degree={degree}",
            )


def _check_basepoint_integral_naturality() -> None:
    for degree in range(1, 5):
        for basepoint in (Fraction(-1), Fraction(0), Fraction(2, 3)):
            for source, target in product(FAMILIES, repeat=2):
                left = _matmul(
                    _basepoint_integral(target, degree, basepoint),
                    _change(source, target, degree - 1),
                )
                right = _matmul(
                    _change(source, target, degree),
                    _basepoint_integral(source, degree, basepoint),
                )
                _require(
                    left == right,
                    f"integral naturality failed: {source.key}->{target.key}, "
                    f"degree={degree}, a={basepoint}",
                )


def _check_evaluation_naturality() -> None:
    for degree in range(5):
        for point in (Fraction(-3, 2), Fraction(0), Fraction(5, 3)):
            for source, target in product(FAMILIES, repeat=2):
                transported = _row_times_matrix(
                    _evaluation_row(target, degree, point),
                    _change(source, target, degree),
                )
                _require(
                    transported == _evaluation_row(source, degree, point),
                    f"evaluation naturality failed: {source.key}->{target.key}, "
                    f"degree={degree}, x={point}",
                )


def _check_integration_naturality() -> None:
    intervals = (
        (Fraction(0), Fraction(1), Fraction(1)),
        (Fraction(-2), Fraction(3, 2), Fraction(5, 3)),
    )
    for degree in range(5):
        for lower, upper, density in intervals:
            for source, target in product(FAMILIES, repeat=2):
                transported = _row_times_matrix(
                    _integration_row(target, degree, lower, upper, density=density),
                    _change(source, target, degree),
                )
                _require(
                    transported
                    == _integration_row(source, degree, lower, upper, density=density),
                    f"integration naturality failed: {source.key}->{target.key}, "
                    f"degree={degree}",
                )


def _check_fundamental_theorem_row() -> None:
    intervals = (
        (Fraction(0), Fraction(1), Fraction(1)),
        (Fraction(-2), Fraction(3, 2), Fraction(5, 3)),
    )
    for family in FAMILIES:
        for degree in range(1, 5):
            derivative = _derivative(family, degree)
            for lower, upper, density in intervals:
                left = _row_times_matrix(
                    _integration_row(
                        family,
                        degree - 1,
                        lower,
                        upper,
                        density=density,
                    ),
                    derivative,
                )
                endpoint_difference = tuple(
                    density * (right - left_value)
                    for left_value, right in zip(
                        _evaluation_row(family, degree, lower),
                        _evaluation_row(family, degree, upper),
                        strict=True,
                    )
                )
                _require(
                    left == endpoint_difference,
                    f"ell D = rho(e_b-e_a) failed for {family.key}, degree={degree}",
                )


def _check_matrix_chain_optimality() -> None:
    cases = {
        (10, 30, 5, 60): 4500,
        (40, 20, 30, 10, 30): 26000,
        (5, 10, 3, 12, 5, 50, 6): 2010,
        (30, 35, 15, 5, 10, 20, 25): 15125,
        (7, 11): 0,
    }
    for dimensions, expected in cases.items():
        dynamic, _ = _matrix_chain_dynamic(dimensions)
        exhaustive = _matrix_chain_exhaustive(dimensions)
        _require(dynamic == min(exhaustive), f"DP was not optimal for {dimensions}")
        _require(dynamic == expected, f"unexpected exact matrix-chain cost for {dimensions}")


def _check_fibonacci_pipeline() -> None:
    # F_4(x)=x^3+2x is the degree-three member of the graded Fibonacci basis.
    j0_f3_to_f4 = _basepoint_integral(FIBONACCI, 4, Fraction(0))
    f4_to_e4 = _change(FIBONACCI, EULER, 4)
    d_e4_to_e3 = _derivative(EULER, 4)
    e3_to_f3 = _change(EULER, FIBONACCI, 3)
    full_operator = _matmul(
        e3_to_f3,
        _matmul(d_e4_to_e3, _matmul(f4_to_e4, j0_f3_to_f4)),
    )
    _require(full_operator == _identity(4), "the full Fibonacci pipeline is not I_4")
    fibonacci_degree_three = (Fraction(0), Fraction(0), Fraction(0), Fraction(1))
    recovered = _matvec(full_operator, fibonacci_degree_three)
    value = _dot(_integration_row(FIBONACCI, 3, 0, 1), recovered)
    _require(value == Fraction(5, 4), f"Fibonacci pipeline returned {value}, expected 5/4")


def _check_affine_integration_workflow() -> None:
    polynomial_y = (Fraction(1), Fraction(-2), Fraction(0), Fraction(0), Fraction(1))
    pulled_back = _polynomial_compose_affine(polynomial_y, scale=2, shift=1)
    expected_pullback = (
        Fraction(0),
        Fraction(4),
        Fraction(24),
        Fraction(32),
        Fraction(16),
    )
    _require(pulled_back == expected_pullback, "affine polynomial pullback is incorrect")
    base_integral = _dot(
        tuple(Fraction(1, power + 1) for power in range(5)),
        pulled_back,
    )
    direct_pushforward = _dot(
        tuple(
            Fraction(1, 2)
            * (Fraction(3) ** (power + 1) - Fraction(1) ** (power + 1))
            / Fraction(power + 1)
            for power in range(5)
        ),
        polynomial_y,
    )
    _require(base_integral == Fraction(106, 5), "affine pullback integral is not 106/5")
    _require(direct_pushforward == Fraction(106, 5), "pushforward integral is not 106/5")
    _require(base_integral == direct_pushforward, "affine integration paths disagree")


def _check_false_jd_identity_counterexample() -> None:
    degree = 4
    derivative = _derivative(MONOMIAL, degree)
    primitive = _basepoint_integral(MONOMIAL, degree, Fraction(0))
    jd = _matmul(primitive, derivative)
    constant = (Fraction(1), Fraction(0), Fraction(0), Fraction(0), Fraction(0))
    _require(jd != _identity(degree + 1), "the false identity J D = I was accepted")
    _require(_matvec(jd, constant) == tuple(Fraction(0) for _ in range(degree + 1)), "J D did not annihilate the constant counterexample")
    _require(_matvec(_identity(degree + 1), constant) == constant, "identity control failed")


def _check_density_omission_adversary() -> None:
    polynomial_y = (Fraction(1), Fraction(-2), Fraction(0), Fraction(0), Fraction(1))
    correct = _dot(_integration_row(MONOMIAL, 4, 1, 3, density=Fraction(1, 2)), polynomial_y)
    omitted = _dot(_integration_row(MONOMIAL, 4, 1, 3, density=1), polynomial_y)
    _require(correct == Fraction(106, 5), "density-aware control is not 106/5")
    _require(omitted == Fraction(212, 5), "density-omission adversary did not expose doubling")
    _require(omitted != correct, "omitting pushforward density went undetected")


def _check_singular_basis_inverse_adversary() -> None:
    singular = _matrix(((1, 1, 0), (0, 0, 0), (0, 0, 1)))
    try:
        _inverse(singular)
    except SingularMatrixError:
        pass
    else:
        raise AssertionError("singular basis was assigned an exact inverse")
    valid = _matrix(((1, 1, 0), (0, 1, 1), (0, 0, 1)))
    inverse = _inverse(valid)
    _require(_matmul(inverse, valid) == _identity(3), "nonsingular inverse control failed")


@dataclass(frozen=True, slots=True)
class _Check:
    name: str
    function: Callable[[], None]


CHECKS = (
    _Check("same-name structural basis fingerprints", _check_structural_fingerprints),
    _Check("monomial Euler Fibonacci change cocycle and inverses", _check_change_cocycle_and_inverses),
    _Check("tight derivative-after-integral law", _check_derivative_after_integral),
    _Check("basepoint integral-after-derivative defect law", _check_integral_after_derivative),
    _Check("derivative coordinate naturality", _check_derivative_naturality),
    _Check("basepoint integration coordinate naturality", _check_basepoint_integral_naturality),
    _Check("evaluation coordinate naturality", _check_evaluation_naturality),
    _Check("definite integration coordinate naturality", _check_integration_naturality),
    _Check("density-weighted fundamental theorem row", _check_fundamental_theorem_row),
    _Check("matrix-chain dynamic-programming optimality", _check_matrix_chain_optimality),
    _Check("Fibonacci Euler derivative integration pipeline", _check_fibonacci_pipeline),
    _Check("affine pushforward integration workflow", _check_affine_integration_workflow),
    _Check("false J D identity counterexample", _check_false_jd_identity_counterexample),
    _Check("pushforward density omission adversary", _check_density_omission_adversary),
    _Check("singular basis inverse adversary", _check_singular_basis_inverse_adversary),
)


def _run_check(check: _Check) -> dict[str, object]:
    try:
        check.function()
    except Exception as exc:
        return {
            "name": check.name,
            "passed": False,
            "detail": f"{type(exc).__name__}: {exc}",
        }
    return {"name": check.name, "passed": True, "detail": ""}


def run() -> dict[str, object]:
    """Run all independent checks and continue after individual failures."""

    results = [_run_check(check) for check in CHECKS]
    passed = sum(bool(result["passed"]) for result in results)
    total = len(results)
    return {
        "schema": "komorebi-math.block20-independent-math-verification",
        "schema_version": 1,
        "package_version": "1.0.0b4",
        "verifier": "verification/block20_math_verifier.py",
        "arithmetic": "stdlib fractions.Fraction",
        "production_expression_imports": False,
        "success": passed == total,
        "passed": passed,
        "failed": total - passed,
        "total": total,
        "worked_values": {
            "fibonacci_pipeline": "5/4",
            "affine_pushforward_integral": "106/5",
            "density_omission": "212/5",
        },
        "checks": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write the JSON report to this path")
    parser.add_argument("--quiet", action="store_true", help="suppress standard output")
    arguments = parser.parse_args(argv)
    report = run()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8", newline="\n")
    if not arguments.quiet:
        print(payload, end="")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
