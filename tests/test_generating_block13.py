from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from math import comb

import numpy as np
import pytest

from komorebi_math.exceptions import BackendUnavailableError, ValidationError
from komorebi_math.export import dumps_portable, loads_portable
from komorebi_math.generating import (
    FiniteMaskCarryPlan,
    FormalSeries,
    GeneratingBudget,
    compile_cartier_linear_representation,
    compile_digit_cauchy_product,
    compile_finite_mask_carry_plan,
    exponential_basis,
    factorial_power_bell_function,
    factorial_power_convolution_layers,
    factorial_power_support_count,
    factorial_power_support_indices,
    fir_taps_to_series,
    first_frobenius_lift,
    freeze_custom_function,
    frobenius_defect_mask,
    kummer_layer_distribution,
    mahler_mask_inverse,
    mahler_mask_product,
    one_borrow_count,
    ordinary_basis,
    polynomial_mask_coboundary,
    series_to_fir_adapter,
    series_to_fir_taps,
    zero_borrow_count,
)
from komorebi_math.generating.padic_units import borrow_count


def test_mahler_mask_inverse_and_functional_equation():
    mask = (1, 2, -1, 3)
    series = mahler_mask_product(mask, 3, 90)
    recovered = mahler_mask_inverse(series, 3, 90)
    assert recovered[: len(mask)] == tuple(Fraction(value) for value in mask)
    assert all(value == 0 for value in recovered[len(mask) :])


def test_finite_mask_plan_matches_direct_product():
    for radix in (2, 3, 5):
        plan = compile_finite_mask_carry_plan((1, -2, 3, 1), radix)
        plan.certificate(300).require()


def test_overfull_binary_mask_recovers_stern_shift():
    plan = compile_finite_mask_carry_plan((1, 1, 1), 2)

    def stern(index: int) -> int:
        values = [0, 1]
        for n in range(2, index + 1):
            values.append(values[n // 2] if n % 2 == 0 else values[(n - 1) // 2] + values[(n + 1) // 2])
        return values[index]

    assert plan.sequence(100) == tuple(Fraction(stern(index + 1)) for index in range(100))


def test_digit_cauchy_product_matches_direct_convolution():
    radix = 3
    weights = ((1, 2, -1), (1, 3, 4), (1, -2, 5))
    plan = compile_digit_cauchy_product(weights, radix)

    def digit_value(table: tuple[int, ...], index: int) -> int:
        result = 1
        while index:
            index, digit = divmod(index, radix)
            result *= table[digit]
        return result

    factors = [tuple(digit_value(table, n) for n in range(100)) for table in weights]
    first_two = tuple(sum(factors[0][k] * factors[1][n - k] for k in range(n + 1)) for n in range(100))
    expected = tuple(sum(first_two[k] * factors[2][n - k] for k in range(n + 1)) for n in range(80))
    assert plan.sequence(80) == tuple(Fraction(value) for value in expected)


@pytest.mark.parametrize("prime", [2, 3, 5, 7])
def test_kummer_layer_distribution_matches_borrow_counts(prime):
    for index in range(80):
        distribution = kummer_layer_distribution(index, prime)
        brute = [0] * max(1, len(distribution.counts))
        for subindex in range(index + 1):
            layer = borrow_count(index, subindex, prime)
            if layer >= len(brute):
                brute.extend([0] * (layer + 1 - len(brute)))
            brute[layer] += 1
        while len(brute) > 1 and brute[-1] == 0:
            brute.pop()
        assert distribution.counts == tuple(brute)
        assert distribution.count(0) == zero_borrow_count(index, prime)
        assert distribution.count(1) == one_borrow_count(index, prime)


@pytest.mark.parametrize("prime", [2, 3, 5])
@pytest.mark.parametrize("power", [1, 2, 3])
@pytest.mark.parametrize("exponent", [1, 2, 3, 4])
def test_factorial_power_support_matches_direct_residues(prime, power, exponent):
    modulus = prime**exponent
    for index in range(30):
        expected = tuple(
            subindex
            for subindex in range(index + 1)
            if pow(comb(index, subindex), power, modulus) != 0
        )
        assert factorial_power_support_indices(index, prime, power, exponent) == expected
        assert factorial_power_support_count(index, prime, power, exponent) == len(expected)


def test_factorial_power_convolution_layer_sum():
    left = tuple(index * index - 2 * index + 3 for index in range(60))
    right = tuple(2 * index + 5 for index in range(60))
    result = factorial_power_convolution_layers(left, right, 45, 3, 2, 4)
    expected = sum(
        pow(comb(45, k), 2, result.modulus) * left[k] * right[45 - k]
        for k in range(46)
    ) % result.modulus
    assert result.residue == expected


def test_frobenius_defect_and_first_lift():
    for prime in (2, 3, 5, 7):
        defect = frobenius_defect_mask((1, 1, 2), prime, 60)
        assert defect[0] == 1
        lift = first_frobenius_lift((1, 1, 2), prime, 60)
        assert lift[0] == 0


def test_polynomial_coboundary_telescopes():
    candidate = (1, -1, 2, 3)
    mask = polynomial_mask_coboundary(candidate, 3, 80)
    reconstructed = mahler_mask_product(mask, 3, 80)
    expected = tuple(Fraction(value) for value in candidate) + (Fraction(0),) * 77
    assert reconstructed == expected


def test_budget_guards_expansive_requests():
    budget = GeneratingBudget(maximum_degree=8, maximum_local_combinations=16, maximum_matrix_dimension=2)
    with pytest.raises(ValidationError):
        mahler_mask_product((1, 1), 2, 9, budget=budget)
    with pytest.raises(ValidationError):
        compile_finite_mask_carry_plan((1, 1, 1, 1, 1), 2, budget=budget)
    with pytest.raises(ValidationError):
        compile_digit_cauchy_product(((1, 1),) * 5, 2, budget=budget)


def test_formal_series_portable_round_trip():
    source = FormalSeries(exponential_basis(), (1, 2, -3, 5, Fraction(7, 3)))
    restored = loads_portable(dumps_portable(source))
    assert isinstance(restored, FormalSeries)
    assert restored.coefficients == source.coefficients
    assert restored.ordinary_coefficients() == source.ordinary_coefficients()


def test_frozen_custom_function_portable_round_trip():
    frozen = freeze_custom_function(factorial_power_bell_function(3), 18)
    restored = loads_portable(dumps_portable(frozen))
    assert restored == frozen
    assert restored.series().coefficients == frozen.coefficients
    with pytest.raises(ValidationError):
        restored.series(19)


def test_mask_plan_and_cartier_portable_round_trip():
    plan = compile_finite_mask_carry_plan((1, 2, 1), 3)
    restored_plan = loads_portable(dumps_portable(plan))
    assert isinstance(restored_plan, FiniteMaskCarryPlan)
    assert restored_plan.sequence(100) == plan.sequence(100)

    representation = compile_cartier_linear_representation((1, 1), (1, -1), 3)
    restored_representation = loads_portable(dumps_portable(representation))
    assert restored_representation.sequence(200) == representation.sequence(200)


def test_portable_digest_detects_tampering():
    text = dumps_portable(compile_finite_mask_carry_plan((1, 1, 1), 2))
    tampered = text.replace('"numerator": 1', '"numerator": 7', 1)
    assert tampered != text
    with pytest.raises(ValidationError, match="SHA256 mismatch"):
        loads_portable(tampered)


def test_signal_bridge_round_trip_and_response():
    source = FormalSeries(ordinary_basis(), (1, Fraction(1, 2), Fraction(-1, 4), 2))
    taps = series_to_fir_taps(source)
    assert np.allclose(taps, [1.0, 0.5, -0.25, 2.0])
    restored = fir_taps_to_series(taps, max_denominator=1000)
    assert restored.coefficients == source.coefficients
    adapter = series_to_fir_adapter(source)
    adapter.certificate().require()
    assert adapter.response(points=32).response.shape == (32,)


def test_sympy_is_lazy_optional_dependency():
    script = r'''
import sys
sys.modules["sympy"] = None
import komorebi_math.generating as gf
assert gf.ordinary_basis().name == "ordinary"
try:
    gf.guess_linear_recurrence([0, 1, 1, 2, 3, 5, 8, 13])
except Exception as exc:
    assert type(exc).__name__ == "BackendUnavailableError", type(exc).__name__
else:
    raise AssertionError("symbolic call did not report the missing backend")
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src")},
    )
    assert result.returncode == 0, result.stderr


def test_analytic_mahler_product_tail_bound_contains_deeper_product():
    from komorebi_math.generating import analytic_mahler_product

    shallow = analytic_mahler_product((1, Fraction(1, 2), -1), 0.31 + 0.07j, 3, 4)
    deep = analytic_mahler_product((1, Fraction(1, 2), -1), 0.31 + 0.07j, 3, 10)
    shallow.certificate().require()
    assert abs(deep.value - shallow.value) <= shallow.absolute_error_bound


def test_rational_mask_algebraic_relation_certificate():
    from komorebi_math.generating import certify_rational_mask_algebraic_relation

    for prime in (2, 3, 5, 7):
        certificate = certify_rational_mask_algebraic_relation(
            (1, 1, 2), (1, -1, 1), prime, 120
        )
        certificate.require()


def test_bounded_polynomial_coboundary_solver_positive_and_negative():
    from komorebi_math.generating import solve_polynomial_coboundary

    positive = solve_polynomial_coboundary((1, 2), (1, 0, 2), 2, 4)
    assert positive.require() == (Fraction(1), Fraction(2))

    negative = solve_polynomial_coboundary((1, 1), (1, -1), 2, 5)
    assert not negative.found
    assert not negative.certificate().valid


def test_euler_product_orbit_collapse_and_obstruction():
    from komorebi_math.generating import euler_product_collapse

    collapse = euler_product_collapse(3, {1: -2, 3: 2, 5: 1, 15: -1})
    collapse.require()
    assert dict(collapse.closed_form_exponents) == {1: -2, 5: 1}

    obstructed = euler_product_collapse(3, {1: -1, 3: 2})
    assert not obstructed.collapses
    assert obstructed.obstruction_orbits == (1,)
