from __future__ import annotations

from fractions import Fraction
from math import comb

import pytest

from komorebi_math.generating import (
    FormalSeries,
    FunctionRegistry,
    apery_numbers,
    audit_apery_supercongruence,
    audit_factorial_power_packet_convolution,
    audit_generalized_bell_digit_sum,
    audit_generalized_lucas,
    audit_partition_lane,
    audit_q_lucas,
    audit_section_cycle,
    audit_touchard,
    bell_numbers,
    catalan_numbers,
    central_binomial_numbers,
    characteristic_closed_form,
    dirichlet_convolution,
    euler_product_function,
    exponential_basis,
    factorial_power_basis,
    factorial_power_bell_function,
    fibonacci_factorial_basis,
    fibonacci_numbers,
    generalized_bell_numbers,
    generalized_bell_polynomials,
    guess_algebraic_gf,
    guess_hypergeometric_ratio,
    guess_linear_recurrence,
    guess_p_recursive,
    ordinary_basis,
    partition_numbers,
    pochhammer_basis,
    q_factorial_basis,
    rational_gf_from_recurrence,
)


@pytest.mark.parametrize(
    "basis",
    [
        ordinary_basis(),
        exponential_basis(),
        factorial_power_basis(2),
        q_factorial_basis(2),
        fibonacci_factorial_basis(),
    ],
)
def test_weighted_product_matches_ordinary_product(basis):
    left = FormalSeries(basis, [1, 2, -3, 5, 7, -11])
    right = FormalSeries(basis, [0, 3, 4, -2, 1, 6])
    product = left.multiply(right, degree=8)
    a = left.ordinary_coefficients(8)
    b = right.ordinary_coefficients(8)
    expected = tuple(sum(a[k] * b[n - k] for k in range(n + 1)) for n in range(9))
    assert product.ordinary_coefficients() == expected


@pytest.mark.parametrize(
    "basis",
    [ordinary_basis(), exponential_basis(), factorial_power_basis(3), q_factorial_basis(3)],
)
def test_packet_identity(basis):
    series = FormalSeries(basis, range(1, 35))
    for modulus in range(2, 7):
        for residue in range(modulus):
            assert series.packet_identity(modulus, residue, target_basis=ordinary_basis()).valid


def test_exp_log_round_trip():
    basis = factorial_power_basis(2)
    source = FormalSeries(basis, [0, 1, 2, -3, 5, 7, 11, -13])
    assert source.exp(degree=7).log(degree=7).coefficients == source.coefficients


@pytest.mark.parametrize(
    "basis",
    [ordinary_basis(), exponential_basis(), factorial_power_basis(2), q_factorial_basis(2), fibonacci_factorial_basis()],
)
def test_custom_section_cycle(basis):
    for signed in (False, True):
        audit = audit_section_cycle(basis, modulus=5, degree=44, signed=signed)
        assert audit.one_step_failures == 0
        assert audit.full_cycle_failures == 0
        assert audit.root_filter_residual < 1e-12


def test_factorial_power_structure_constants():
    basis = factorial_power_basis(4)
    for n in range(14):
        for k in range(n + 1):
            assert basis.structure_constant(n, k) == comb(n, k) ** 4


def test_built_in_basis_integrality():
    for basis in [exponential_basis(), factorial_power_basis(3), q_factorial_basis(2), fibonacci_factorial_basis()]:
        assert basis.finite_certificate(16).integral_structure_constants
    nonintegral = pochhammer_basis(2).finite_certificate(18)
    assert not nonintegral.integral_structure_constants
    assert nonintegral.maximum_denominator == 9


def test_fibonacci_closed_form_pipeline():
    values = fibonacci_numbers(40)
    recurrence = guess_linear_recurrence(values, max_order=6)
    assert recurrence is not None
    assert recurrence.order == 2
    assert recurrence.coefficients == (Fraction(1), Fraction(1))
    numerator, denominator = rational_gf_from_recurrence(recurrence, values)
    assert numerator == (Fraction(0), Fraction(1))
    assert denominator == (Fraction(1), Fraction(-1), Fraction(-1))
    expression = characteristic_closed_form(recurrence, values)
    assert expression is not None


def test_hypergeometric_detectors():
    central = guess_hypergeometric_ratio(central_binomial_numbers(50))
    catalan = guess_hypergeometric_ratio(catalan_numbers(50))
    assert central is not None and str(central.ratio_expression()) == "(4*n + 2)/(n + 1)"
    assert catalan is not None and str(catalan.ratio_expression()) == "(4*n + 2)/(n + 2)"


def test_catalan_algebraic_guess():
    guess = guess_algebraic_gf(catalan_numbers(50), max_x_degree=2, max_y_degree=3)
    assert guess is not None
    x, y = __import__("sympy").symbols("x y")
    assert __import__("sympy").expand(guess.expression() - (1 - y + x * y**2)) == 0


def test_apery_p_recursive_guess():
    guess = guess_p_recursive(apery_numbers(45), max_order=3, max_degree=4)
    assert guess is not None
    assert guess.order == 2
    assert guess.degree == 3


def test_nonholonomic_low_complexity_boundaries():
    assert guess_p_recursive(bell_numbers(55), max_order=3, max_degree=4) is None
    assert guess_algebraic_gf(bell_numbers(55), max_x_degree=3, max_y_degree=3) is None
    assert guess_p_recursive(partition_numbers(80), max_order=3, max_degree=4) is None
    assert guess_p_recursive(generalized_bell_numbers(2, 45), max_order=3, max_degree=4) is None


def test_euler_product_partition_builder():
    function = euler_product_function("partition", lambda _m: 1)
    values = function.series(80).coefficients
    assert tuple(int(value) for value in values) == partition_numbers(81)


def test_custom_function_registry():
    registry = FunctionRegistry()
    function = factorial_power_bell_function(2)
    registry.register(function)
    assert registry.get(function.name).series(10).coefficients == tuple(Fraction(v) for v in generalized_bell_numbers(2, 11))
    with pytest.raises(KeyError):
        registry.register(function)


def test_generalized_bell_polynomial_integrality():
    rows = generalized_bell_polynomials(4, 25)
    assert all(isinstance(value, int) for row in rows for value in row)
    assert tuple(sum(row) for row in rows) == generalized_bell_numbers(4, 25)


@pytest.mark.parametrize("prime", [2, 3, 5, 7, 11])
def test_generalized_bell_digit_sum_theorem(prime):
    power = 1 + 2 * (prime - 1)
    audit = audit_generalized_bell_digit_sum(prime=prime, power=power, count=180)
    assert audit.number_failures == 0
    assert audit.polynomial_failures == 0


def test_generalized_bell_parity_corollary():
    values = generalized_bell_numbers(6, 300)
    for n, value in enumerate(values):
        expected = 0 if n.bit_count() % 3 == 2 else 1
        assert value % 2 == expected


@pytest.mark.parametrize("power", [1, 2, 3, 4])
@pytest.mark.parametrize("prime", [2, 3, 5, 7])
def test_factorial_power_lucas(power, prime):
    audit = audit_generalized_lucas(factorial_power_basis(power), prime=prime, high_limit=6)
    assert audit.failures == 0


@pytest.mark.parametrize("q,prime", [(2, 3), (2, 5), (2, 7), (2, 31), (3, 2), (3, 5), (3, 13)])
def test_q_lucas(q, prime):
    audit = audit_q_lucas(q=q, prime=prime, high_limit=6)
    assert audit.failures == 0


def test_naive_lucas_boundary_for_ordinary_basis():
    assert not audit_generalized_lucas(ordinary_basis(), prime=5, high_limit=5).lucas_compatible


def test_factorial_power_packet_convolution():
    left = [n * n - 3 * n + 7 for n in range(60)]
    right = [2**n + n for n in range(60)]
    for power in (1, 2, 3):
        for prime in (2, 3, 5, 7):
            audit = audit_factorial_power_packet_convolution(left, right, power=power, prime=prime)
            assert audit.failures == 0


def test_classical_congruence_benchmarks():
    bell = bell_numbers(350)
    for prime in (2, 3, 5, 7, 11, 13):
        assert audit_touchard(bell, prime).failures == 0
    partitions = partition_numbers(1200)
    for prime, residue in ((5, 4), (7, 5), (11, 6)):
        assert audit_partition_lane(partitions, prime, residue).failures == 0
    apery = apery_numbers(220)
    for prime in (5, 7, 11, 13, 17, 19):
        assert audit_apery_supercongruence(apery, prime, 3).failures == 0


def test_dirichlet_convolution():
    ones = [1] * 50
    result = dirichlet_convolution(ones, ones, count=50)
    expected = tuple(sum(n % d == 0 for d in range(1, n + 1)) for n in range(1, 51))
    assert result == expected

@pytest.mark.parametrize("prime", [2, 3, 5, 7])
def test_weighted_generalized_bell_digit_sum_theorem(prime):
    from komorebi_math.generating import audit_weighted_generalized_bell_digit_sum

    power = 1 + 2 * (prime - 1)
    weights = tuple(t**3 - 2 * t + 5 for t in range(64))
    audit = audit_weighted_generalized_bell_digit_sum(
        prime=prime,
        power=power,
        digit_weights=weights,
        count=160,
    )
    assert audit.failures == 0


@pytest.mark.parametrize("prime,power", [(2, 1), (3, 2), (5, 3), (7, 4)])
def test_factorial_power_digit_algebra(prime, power):
    from komorebi_math.generating import audit_factorial_power_digit_algebra

    left = tuple((digit * digit + 2 * digit + 1) % prime for digit in range(prime))
    right = tuple(1 if digit == 0 else (3 * digit + 2) % prime for digit in range(prime))
    audit = audit_factorial_power_digit_algebra(
        prime=prime,
        power=power,
        left_digits=left,
        right_digits=right,
        max_digits=3,
    )
    assert audit.weight_failures == 0
    assert audit.closure_failures == 0


def test_algebraic_custom_function_builder():
    from komorebi_math.generating import algebraic_function

    catalan = algebraic_function(
        "Catalan implicit branch",
        {(0, 0): 1, (0, 1): -1, (1, 2): 1},
        1,
    )
    assert tuple(int(value) for value in catalan.series(35).coefficients) == catalan_numbers(36)


def test_weighted_custom_function_builder():
    from komorebi_math.generating import weighted_factorial_power_bell_function

    weights = tuple(0 if n == 0 else n * n + 1 for n in range(30))
    function = weighted_factorial_power_bell_function(3, weights)
    values = function.series(18).coefficients
    assert len(values) == 19
    assert all(value.denominator == 1 for value in values)
