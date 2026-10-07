from __future__ import annotations

from math import comb, gcd
from random import Random

import pytest

from komorebi_math.generating import (
    CarryLinearValuation,
    PrimePowerKernelCompiler,
    PrimePowerUnitCompiler,
    binomial_valuation,
    borrow_count,
    certify_cartier_carry_equivalence,
    classify_carry_linear_valuation,
    compile_rational_mask_automaton,
    cyclotomic_coboundary,
    digit_factorized_normalized_kernel,
    digit_unit,
    gaussian_binomial,
    q_binomial_support,
    q_binomial_valuation_decomposition,
    q_binomial_valuation_direct,
    rational_mask_coefficients_mod_prime,
    verify_cyclotomic_coboundary,
)


def brute_unit_factorial(index: int, prime: int, modulus: int) -> int:
    value = 1
    for factor in range(1, index + 1):
        unit = factor
        while unit % prime == 0:
            unit //= prime
        value = value * unit % modulus
    return value


@pytest.mark.parametrize("prime,exponent", [(2, 1), (2, 5), (3, 4), (5, 3), (7, 2)])
def test_unit_factorial_matches_direct(prime, exponent):
    compiler = PrimePowerUnitCompiler(prime, exponent)
    for index in range(240):
        assert compiler.unit_factorial(index) == brute_unit_factorial(
            index, prime, compiler.modulus
        )


@pytest.mark.parametrize("prime,exponent", [(2, 6), (3, 5), (5, 4), (7, 3)])
def test_prime_power_binomial_compiler(prime, exponent):
    compiler = PrimePowerUnitCompiler(prime, exponent)
    for index in range(150):
        for subindex in range(index + 1):
            data = compiler.binomial_data(index, subindex)
            assert data.residue == comb(index, subindex) % compiler.modulus
            assert data.valuation == binomial_valuation(index, subindex, prime)
            assert data.valuation == borrow_count(index, subindex, prime)


def test_large_index_prime_power_binomial():
    compiler = PrimePowerUnitCompiler(3, 6)
    for index, subindex in [(1000, 417), (4096, 1025), (10000, 3777)]:
        assert compiler.binomial(index, subindex) == comb(index, subindex) % compiler.modulus


@pytest.mark.parametrize(
    "prime,numerator,denominator",
    [
        (2, (1, 1, 1), (1,)),
        (2, (1,), (1, 1)),
        (3, (1, 1), (1, -1)),
        (3, (1, 2, 1), (1, 1, 1)),
        (5, (1, -2, 3), (1, 1)),
        (7, (1, 3, 0, 2), (1, -1, 1)),
    ],
)
def test_cartier_compiler_matches_recurrence(prime, numerator, denominator):
    automaton = compile_rational_mask_automaton(numerator, denominator, prime)
    automaton.certificate(4000).require()
    assert automaton.number_states <= automaton.theoretical_state_bound


def test_stern_cartier_automaton_is_three_state():
    automaton = compile_rational_mask_automaton((1, 1, 1), (1,), 2)
    assert automaton.number_states == 3
    assert automaton.sequence(15) == (1, 1, 0) * 5


def test_cartier_minimization_preserves_sequence():
    raw = compile_rational_mask_automaton((1, 1, 0, 1), (1, 1), 3, minimize=False)
    minimal = compile_rational_mask_automaton((1, 1, 0, 1), (1, 1), 3, minimize=True)
    assert minimal.number_states <= raw.number_states
    assert minimal.sequence(5000) == raw.sequence(5000)


def test_random_rational_masks():
    random = Random(2026081703)
    for prime in (2, 3, 5):
        for _case in range(20):
            numerator = [1] + [random.randrange(prime) for _ in range(random.randrange(1, 4))]
            denominator = [1] + [random.randrange(prime) for _ in range(random.randrange(1, 3))]
            automaton = compile_rational_mask_automaton(
                numerator, denominator, prime, max_states=50_000
            )
            expected = rational_mask_coefficients_mod_prime(
                numerator, denominator, prime, 700
            )
            assert automaton.sequence(700) == expected


@pytest.mark.parametrize("prime", [2, 3, 5, 7])
def test_cartier_carry_equivalence(prime):
    random = Random(prime * 1009)
    for degree in range(1, 9):
        mask = [1] + [random.randrange(prime) for _ in range(degree)]
        certify_cartier_carry_equivalence(mask, prime).require()


@pytest.mark.parametrize("prime,slope,weight", [(2, -1, 3), (3, 2, 4), (5, 0, 7)])
def test_carry_linear_classification(prime, slope, weight):
    profile = CarryLinearValuation(prime, slope, weight)
    certificate = classify_carry_linear_valuation(profile.value, prime, 80)
    certificate.require()
    assert certificate.linear_slope == slope
    assert certificate.carry_weight == weight


def test_perturbed_valuation_is_rejected_by_finite_audit():
    profile = CarryLinearValuation(3, 2, 5)

    def perturbed(index: int) -> int:
        return profile.value(index) + (1 if index == 17 else 0)

    certificate = classify_carry_linear_valuation(perturbed, 3, 40)
    assert not certificate.valid


@pytest.mark.parametrize("prime,power,weights", [(3, 2, (1, 2, 1)), (5, 3, (1, 2, 4, 3, 1)), (7, 4, (1, 2, 3, 4, 5, 6, 1))])
def test_digit_twisted_normalized_kernel(prime, power, weights):
    for index in range(180):
        for subindex in range(index + 1):
            direct = pow(comb(index, subindex), power, prime)
            numerator_unit = digit_unit(index, prime, weights)
            denominator_unit = (
                digit_unit(subindex, prime, weights)
                * digit_unit(index - subindex, prime, weights)
            ) % prime
            direct = direct * numerator_unit * pow(denominator_unit, -1, prime) % prime
            factored = digit_factorized_normalized_kernel(
                index, subindex, prime, power, weights
            )
            assert factored == direct


@pytest.mark.parametrize("prime,exponent,power", [(2, 6, 3), (3, 5, 2), (5, 4, 4)])
def test_custom_prime_power_kernel(prime, exponent, power):
    modulus = prime**exponent

    def unit(index: int) -> int:
        return 1 + prime * (index + index * index)

    compiler = PrimePowerKernelCompiler(prime, exponent, power, unit)
    for index in range(100):
        for subindex in range(index + 1):
            ratio = unit(index) % modulus
            ratio *= pow(unit(subindex) * unit(index - subindex) % modulus, -1, modulus)
            expected = pow(comb(index, subindex), power, modulus) * ratio % modulus
            assert compiler.kernel(index, subindex).residue == expected


@pytest.mark.parametrize("prime", [2, 3, 5, 7, 11])
def test_mixed_radix_q_kummer(prime):
    for base in range(2, 11):
        for index in range(35):
            for subindex in range(index + 1):
                decomposition = q_binomial_valuation_decomposition(
                    index, subindex, base, prime
                )
                assert decomposition.valuation == q_binomial_valuation_direct(
                    index, subindex, base, prime
                )


def test_q_support_matches_direct_prime_power_nonzero():
    for prime, base, exponent, power in [(2, 3, 6, 2), (2, 4, 5, 3), (3, 2, 4, 2), (5, 2, 3, 1), (7, 3, 4, 3)]:
        modulus = prime**exponent
        for index in range(35):
            expected = tuple(
                subindex
                for subindex in range(index + 1)
                if pow(gaussian_binomial(index, subindex, base), power, modulus) != 0
            )
            assert q_binomial_support(index, base, prime, exponent, power) == expected


@pytest.mark.parametrize(
    "prime,mask,expected",
    [
        (2, {2: 1}, {1: -1}),
        (3, {3: 1}, {1: -1}),
        (3, {3: 1, 9: -1}, {3: 1}),
        (5, {10: 2, 50: -3, 250: 1}, {10: 2, 50: -1}),
    ],
)
def test_cyclotomic_collapse(prime, mask, expected):
    collapse = cyclotomic_coboundary(prime, mask)
    collapse.require()
    assert dict(collapse.closed_form_exponents) == {
        key: value for key, value in expected.items() if value
    }
    assert verify_cyclotomic_coboundary(collapse)


def test_cyclotomic_p_free_obstruction():
    collapse = cyclotomic_coboundary(3, {1: 1, 3: -1, 10: 2})
    assert not collapse.collapses
    assert collapse.obstruction_orbits == (1, 10)


def test_cartier_linear_representation_matches_automaton():
    from komorebi_math.generating import compile_cartier_linear_representation

    for prime, numerator, denominator in [
        (2, (1, 1, 1), (1,)),
        (3, (1, 2, 1), (1, 1)),
        (5, (1, 0, 3, 2), (1, -1, 1)),
    ]:
        linear = compile_cartier_linear_representation(numerator, denominator, prime)
        automaton = compile_rational_mask_automaton(numerator, denominator, prime)
        assert linear.sequence(3000) == automaton.sequence(3000)


def test_finite_mask_carry_is_linear_representation():
    from komorebi_math.generating import compile_cartier_linear_representation, finite_mask_carry_matrices

    for prime, mask in [(2, (1, 1, 1)), (3, (1, 2, 1, 2)), (5, (1, 4, 2, 3, 1, 1))]:
        linear = compile_cartier_linear_representation(mask, (1,), prime)
        assert linear.transition_matrices == finite_mask_carry_matrices(mask, prime)


def test_cyclotomic_exponent_verifier_randomized():
    from random import Random
    from komorebi_math.generating import verify_cyclotomic_exponent_relation

    random = Random(303)
    for prime in (2, 3, 5, 7):
        for _case in range(40):
            mask = {}
            for unit in range(1, 6):
                if unit % prime == 0:
                    continue
                for level in range(1, 5):
                    exponent = random.randrange(-3, 4)
                    if exponent:
                        mask[unit * prime**level] = exponent
            collapse = cyclotomic_coboundary(prime, mask)
            assert verify_cyclotomic_exponent_relation(collapse)
