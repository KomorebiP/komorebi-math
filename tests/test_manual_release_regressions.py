"""Public edge cases corrected while qualifying the release manual."""

from fractions import Fraction
from math import comb, isinf
import subprocess
import sys

import pytest

from komorebi_math.arithmetic import LinearRecurrence
from komorebi_math.arithmetic.local_modes import simultaneous_unit_difference
from komorebi_math.coordinates.formal import FormalCoordinateChange
from komorebi_math.coordinates.operators import bezout_defect_identity, polynomial_extended_gcd
from komorebi_math.coordinates.packet import digit_reversal_order
from komorebi_math.exceptions import ValidationError
from komorebi_math.generating.closed_forms import (
    guess_algebraic_gf,
    guess_hypergeometric_ratio,
    guess_linear_recurrence,
    guess_p_recursive,
    guess_quasipolynomial,
)
from komorebi_math.generating.digit_algebra import factorial_power_digit_kernel
from komorebi_math.generating.functions import algebraic_function
from komorebi_math.generating.normalization import (
    digit_factorized_normalized_kernel,
    local_normalized_kernel,
)
from komorebi_math.generating.q_kummer import q_factorial
from komorebi_math.sequences import Sequence, compile_sequence, detect_linear_recurrence


@pytest.mark.parametrize("radix", [0, 1, True, 2.0])
def test_digit_reversal_requires_integer_radix_at_least_two(radix) -> None:
    with pytest.raises(ValidationError, match="radix"):
        digit_reversal_order(1, radix)


def test_radix_one_digit_reversal_fails_without_looping() -> None:
    program = (
        "from komorebi_math.coordinates.packet import digit_reversal_order\n"
        "from komorebi_math.exceptions import ValidationError\n"
        "try:\n"
        "    digit_reversal_order(2, 1)\n"
        "except ValidationError:\n"
        "    print('rejected')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=5, check=True
    )
    assert completed.stdout.strip() == "rejected"
    assert digit_reversal_order(9, 3) == (0, 3, 6, 1, 4, 7, 2, 5, 8)


@pytest.mark.parametrize("base", [1, -1])
@pytest.mark.parametrize("order", [0, 1, 3])
def test_zero_unit_difference_retains_infinite_valuation(base, order) -> None:
    result = simultaneous_unit_difference(base, {2: 2, 3: 1}, index=2, order=order)
    result.certificate.require()
    expected = 0 if order == 0 else float("inf")
    assert result.value == {2: expected, 3: expected}
    if order:
        assert all(isinf(value) for value in result.value.values())
        assert result.certificate.diagnostics["difference_value"] == 0


def test_nonzero_unit_difference_keeps_exact_integer_valuations() -> None:
    result = simultaneous_unit_difference(5, {2: 2, 3: 1}, index=1, order=2)
    result.certificate.require()
    # Common jump 2 gives 5 * (25 - 1)^2 = 5 * 2^6 * 3^2.
    assert result.value == {2: 6, 3: 2}
    assert all(isinstance(value, int) for value in result.value.values())


@pytest.mark.parametrize("base", [-2, 0, 1])
def test_empty_q_factorial_validates_base(base) -> None:
    with pytest.raises(ValueError, match="base"):
        q_factorial(0, base)
    assert q_factorial(0, 2) == 1
    assert q_factorial(3, 2) == 21


def test_detected_integral_recurrence_supports_fast_modular_execution() -> None:
    values = (0, 1, 1, 2, 3, 5, 8, 13)
    detected = detect_linear_recurrence(values)
    assert detected is not None
    reference = LinearRecurrence((1, 1), (0, 1))
    compiled = compile_sequence(Sequence(reference.term), sample_count=24)
    assert compiled.strategy == "detected linear recurrence"
    assert compiled.certificate.exactness.value == "conditional"
    for modulus in (2, 7, 25):
        for index in (0, 1, 20, 100, 1000):
            expected = reference.term(index) % modulus
            assert detected.term_mod(index, modulus) == expected
            assert compiled.modulo(index, modulus) == expected
        assert detected.period_mod(modulus).period == reference.period_mod(modulus).period


@pytest.mark.parametrize("coefficient, initial", [(Fraction(1, 2), 1), (1, Fraction(1, 2)), (1.0, 1)])
def test_nonintegral_or_float_recurrence_data_remain_rejected(coefficient, initial) -> None:
    recurrence = LinearRecurrence((coefficient,), (initial,))
    with pytest.raises(ValidationError, match="integral"):
        recurrence.term_mod(5, 7)
    with pytest.raises(ValidationError, match="integral"):
        recurrence.period_mod(7)


def test_formal_pushforward_preserves_constant_and_degree_zero_projection() -> None:
    coordinate = FormalCoordinateChange("quadratic", (0, 2, 1))
    constant = (Fraction(7, 3),)
    assert coordinate.pushforward_coefficients(constant) == constant
    assert coordinate.pushforward_coefficients((Fraction(7, 3), 2, 5), degree=0) == constant
    assert coordinate.pushforward_coefficients(constant, degree=2) == constant + (Fraction(0),) * 2
    for degree in (-1, True, 1.5, 3):
        with pytest.raises(ValidationError):
            coordinate.pushforward_coefficients(constant, degree=degree)


def test_monic_polynomial_gcd_rejects_two_zero_inputs_cleanly() -> None:
    with pytest.raises(ValidationError, match="nonzero"):
        polynomial_extended_gcd((0,), (0,))
    with pytest.raises(ValidationError, match="nonzero"):
        bezout_defect_identity((1, 2, 3), (), (0, 0))
    gcd, left, right = polynomial_extended_gcd((0,), (2, 4))
    assert gcd == (Fraction(1, 2), 1)
    assert left == (0,)
    assert right == (Fraction(1, 4),)


@pytest.mark.parametrize("degree", [Fraction(3, 2), 1.5, True, "1"])
def test_algebraic_function_rejects_coerced_sparse_degrees(degree) -> None:
    with pytest.raises(ValueError, match="degree"):
        algebraic_function("invalid degree", {(0, 1): 1, (degree, 0): -1}, 0)
    with pytest.raises(ValueError, match="degree"):
        algebraic_function("invalid zero term", {(0, 1): 1, (degree, 0): 0}, 0)
    valid = algebraic_function("linear branch", {(0, 1): 1, (1, 0): -1}, 0)
    assert valid.series(3).coefficients == (0, 1, 0, 0)


@pytest.mark.parametrize("prime", [4, 9, 1, True, 3.0])
def test_lucas_digit_helpers_require_a_prime(prime) -> None:
    with pytest.raises(ValueError, match="prime"):
        factorial_power_digit_kernel(1, prime)
    with pytest.raises(ValueError, match="prime"):
        local_normalized_kernel(0, 0, prime, 1, (1,))
    with pytest.raises(ValueError, match="prime"):
        digit_factorized_normalized_kernel(0, 0, prime, 1, (1,))


def test_valid_lucas_digit_lifts_match_independent_binomial_coefficients() -> None:
    for prime in (2, 3, 5):
        for power in (1, 2):
            kernel = factorial_power_digit_kernel(power, prime)
            for index in range(prime**2):
                for subindex in range(index + 1):
                    expected = pow(comb(index, subindex), power, prime)
                    assert kernel.global_weight(index, subindex) == expected
                    assert digit_factorized_normalized_kernel(index, subindex, prime, power, (1,) * prime) == expected


@pytest.mark.parametrize("guesser", [
    guess_linear_recurrence,
    guess_hypergeometric_ratio,
    guess_p_recursive,
    guess_algebraic_gf,
    guess_quasipolynomial,
])
@pytest.mark.parametrize("holdout", [-1, True, 1.5])
def test_recognition_holdout_requires_nonnegative_integer(guesser, holdout) -> None:
    with pytest.raises(ValueError, match="holdout"):
        guesser((0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89), holdout=holdout)
