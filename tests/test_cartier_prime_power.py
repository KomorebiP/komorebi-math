"""Independent arithmetic and arbitrary-state controls for Cartier Chapter 23."""

from dataclasses import FrozenInstanceError
from math import comb
from random import Random

import pytest

from komorebi_math.certificates import Exactness
from komorebi_math.exceptions import DimensionError, ResourceLimitError, ValidationError
from komorebi_math.generating.budget import GeneratingBudget
from komorebi_math.generating.cartier import compile_cartier_linear_representation
from komorebi_math.generating.cartier_prime_power import (
    PrimePowerCartierRepresentation,
    compile_prime_power_cartier_representation,
)


def direct_mahler(a, b, p, modulus, count):
    """Scalar coefficient solve; no Cartier states or Frobenius expansion."""
    result = [0] * count
    if count:
        result[0] = 1
    for n in range(1, count):
        residual = 0
        for j in range(n + 1):
            if j < len(b) and j:
                residual -= b[j] * result[n - j]
            if j < len(a) and (n - j) % p == 0:
                residual += a[j] * result[(n - j) // p]
        result[n] = residual * pow(b[0], -1, modulus) % modulus
    return tuple(result)


def multiply(left, right, modulus, count):
    out = [0] * count
    for degree in range(count):
        out[degree] = (
            sum(
                left[j] * right[degree - j]
                for j in range(degree + 1)
                if j < len(left) and degree - j < len(right)
            )
            % modulus
        )
    return tuple(out)


def inverse_series(polynomial, modulus, count):
    """Triangular inverse, independent of the compiler's nilpotent expansion."""
    inverse = [pow(polynomial[0], -1, modulus)]
    for degree in range(1, count):
        inverse.append(
            -inverse[0]
            * sum(polynomial[j] * inverse[degree - j] for j in range(1, min(degree + 1, len(polynomial))))
            % modulus
        )
    return tuple(inverse)


def decode_state(rep, state, count):
    """Expand sum(P_l/B^l)*F directly as truncated power series."""
    f = direct_mahler(rep.numerator, rep.denominator, rep.prime, rep.modulus, count)
    reciprocal = inverse_series(rep.denominator, rep.modulus, count)
    power = (1,) + (0,) * (count - 1)
    out = [0] * count
    for polynomial in state:
        term = multiply(multiply(polynomial, power, rep.modulus, count), f, rep.modulus, count)
        out = [(x + y) % rep.modulus for x, y in zip(out, term, strict=True)]
        power = multiply(power, reciprocal, rep.modulus, count)
    return tuple(out)


@pytest.mark.parametrize("prime,exponent", [(2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3), (5, 2)])
def test_arbitrary_state_section_identity(prime, exponent):
    rep = compile_prime_power_cartier_representation((prime + 1, 2, -1), (prime + 1, -1, 1), prime, exponent)
    rng = Random(1037 + prime + exponent)
    count = 9
    for _ in range(4):
        state = tuple(
            tuple(rng.randrange(-rep.modulus, rep.modulus + 1) for _ in range(rep.degree_bound + 1))
            for _ in range(rep.layer_bound + 1)
        )
        expanded = decode_state(rep, state, prime * count)
        assert rep.output(state) == expanded[0]
        for digit in range(prime):
            target = rep.transition(state, digit)
            observed = decode_state(rep, target, count)
            assert observed == expanded[digit::prime]
            assert len(target) == prime * exponent + 1
            assert all(len(poly) == rep.degree_bound + 1 for poly in target)
        assert rep.output(rep.transition(state, 0)) == rep.output(state)


@pytest.mark.parametrize("prime,exponent", [(2, 1), (2, 2), (2, 4), (3, 2), (3, 3), (5, 2)])
def test_rational_masks_against_exact_scalar_recurrence(prime, exponent):
    modulus = prime**exponent
    masks = [
        ((1,), (1, -1)),
        ((1, 1), (1,)),
        ((1, -2, 1), (1, 1, 2)),
        ((1 + modulus, 2, -1), (1, -1, 1)),
        ((prime + 1 + modulus, 2, -1), (prime + 1, -1)),
    ]
    for a, b in masks:
        rep = compile_prime_power_cartier_representation(a, b, prime, exponent)
        assert rep.sequence(160) == direct_mahler(a, b, prime, modulus, 160)
        certificate = rep.certificate(160)
        assert certificate.valid and certificate.exactness is Exactness.EXACT
        assert certificate.diagnostics["compared_terms"] == 160


def test_binary_partition_worked_example_and_missing_frobenius_control():
    rep = compile_prime_power_cartier_representation((1,), (1, -1), 2, 3)
    assert rep.sequence(16) == (1, 1, 2, 2, 4, 4, 6, 6, 2, 2, 6, 6, 4, 4, 2, 2)
    prime_only = compile_cartier_linear_representation((1,), (1, -1), 2)
    assert prime_only.evaluate(2) == 0
    assert rep.evaluate(2) == 2  # Merely lifting mod-p output is false modulo 8.
    assert any(target > 1 and any(poly) for branches in rep.branches for target, poly in branches)


@pytest.mark.parametrize("prime,exponent,width", [(2, 4, 3), (3, 3, 2), (5, 2, 4)])
def test_large_index_rational_collapse_has_closed_form(prime, exponent, width):
    # Product of (1+z+...+z^(p-1))^width telescopes to (1-z)^(-width).
    polynomial = [1]
    for _ in range(width):
        expanded = [0] * (len(polynomial) + prime - 1)
        for j, value in enumerate(polynomial):
            for digit in range(prime):
                expanded[j + digit] += value
        polynomial = expanded
    rep = compile_prime_power_cartier_representation(polynomial, (1,), prime, exponent)
    index = 10**80 + 12345
    assert rep.evaluate(index) == comb(index + width - 1, width - 1) % rep.modulus


def test_finite_polynomial_solution_with_nontrivial_denominator():
    # F=1-z, A=1-z, B=1-z^p. Tests denominator layers far beyond mod p.
    rep = compile_prime_power_cartier_representation((1, -1), (1, 0, 0, -1), 3, 3)
    assert rep.sequence(60) == (1, 26) + (0,) * 58
    assert rep.evaluate(10**45) == 0


def test_pruning_and_precision_descent_preserve_values():
    for prime, exponent in [(2, 4), (3, 3)]:
        args = ((1, 2, 1), (1, -1), prime, exponent)
        pruned = compile_prime_power_cartier_representation(*args)
        full = compile_prime_power_cartier_representation(*args, prune=False)
        lower = compile_prime_power_cartier_representation(*args[:-1], exponent - 1)
        assert pruned.sequence(100) == full.sequence(100)
        assert tuple(value % lower.modulus for value in pruned.sequence(100)) == lower.sequence(100)
        assert sum(map(len, pruned.branches)) < sum(map(len, full.branches))


def test_canonical_lift_defect_requires_the_extra_padic_digit():
    rep = compile_prime_power_cartier_representation((1,), (1, -1), 2, 3)
    # Canonical B=1+7z; (B^2-B(z^2))/2 = 7z+21z^2 modulo 8.
    assert rep.denominator == (1, 7)
    assert rep.frobenius_defect == (0, 7, 5)
    # Reducing the numerator modulo 8 before dividing would give 3z+z^2.
    assert rep.frobenius_defect != (0, 3, 1)


def test_digit_orientation_high_zero_padding_and_input_ownership():
    a, b = [1], [1, -1]
    rep = PrimePowerCartierRepresentation(a, b, 3, 2)
    a[0] = 7
    b[1] = 4
    state = rep.initial_state
    # 14=(112)_3 is read as 2,1,1, then any number of high-order zeros.
    for digit in (2, 1, 1):
        state = rep.transition(state, digit)
    expected = direct_mahler((1,), (1, -1), 3, 9, 15)[14]
    assert rep.output(state) == rep.evaluate(14) == expected
    for _ in range(4):
        state = rep.transition(state, 0)
        assert rep.output(state) == expected
    assert rep.evaluate(0) == 1
    assert rep.sequence(0) == ()
    assert rep.theoretical_state_bound == rep.modulus**rep.coordinate_slots
    with pytest.raises(FrozenInstanceError):
        rep.prime = 5


@pytest.mark.parametrize(
    "a,b,p,r",
    [
        ((1,), (0, 1), 2, 2),
        ((1,), (2, 1), 2, 2),
        ((1,), (3, 1), 2, 2),
        ((1,), (1,), 4, 2),
        ((1.0,), (1,), 2, 2),
        ((True,), (1,), 2, 2),
        (("1",), (1,), 2, 2),
        ((1,), (1,), True, 2),
        ((1,), (1,), 2, 0),
        ((1,), (1,), 2, 1.0),
    ],
)
def test_domain_rejects_invalid_or_inexact_input(a, b, p, r):
    with pytest.raises(ValidationError):
        compile_prime_power_cartier_representation(a, b, p, r)


def test_state_digit_and_certificate_validation():
    rep = compile_prime_power_cartier_representation((1,), (1, -1), 2, 2)
    with pytest.raises(DimensionError):
        rep.transition(rep.initial_state[:-1], 0)
    with pytest.raises(DimensionError):
        rep.output(tuple(poly[:-1] for poly in rep.initial_state))
    invalid = list(rep.initial_state)
    invalid[0] = (0.0, *invalid[0][1:])
    with pytest.raises(ValidationError):
        rep.transition(invalid, 0)
    for digit in [-1, 2, True, 0.0]:
        with pytest.raises(ValidationError):
            rep.transition(rep.initial_state, digit)
    for index in [-1, True, 2.0]:
        with pytest.raises(ValidationError):
            rep.evaluate(index)
    with pytest.raises(ValidationError):
        rep.certificate(0)


def test_resource_guards_precede_polynomial_allocation(monkeypatch):
    import komorebi_math.generating.cartier_prime_power as module

    def unexpected(*args, **kwargs):
        raise AssertionError("polynomial allocation occurred before the resource guard")

    monkeypatch.setattr(module, "polynomial_power_mod", unexpected)
    with pytest.raises(ResourceLimitError):
        compile_prime_power_cartier_representation(
            (1,), (1, -1), 2, 2, budget=GeneratingBudget(maximum_matrix_dimension=2)
        )
    with pytest.raises(ResourceLimitError):
        compile_prime_power_cartier_representation((1,), (1,), 2, 10**20)
    with pytest.raises(ResourceLimitError):
        compile_prime_power_cartier_representation((1,), (1,), 10**100, 2)


def test_sequence_and_word_budgets():
    rep = compile_prime_power_cartier_representation(
        (1,), (1,), 2, 2, budget=GeneratingBudget(maximum_recursion_depth=3, maximum_local_combinations=20)
    )
    assert rep.evaluate(7) == 0
    with pytest.raises(ResourceLimitError):
        rep.evaluate(8)
    with pytest.raises(ResourceLimitError):
        rep.sequence(21)


def test_state_bit_budget_precedes_reduction_and_polynomial_work(monkeypatch):
    import komorebi_math.generating.cartier_prime_power as module

    rep = compile_prime_power_cartier_representation(
        (1,), (1,), 2, 2, budget=GeneratingBudget(maximum_coefficient_bits=3)
    )
    admissible = ((7,), *rep.initial_state[1:])
    assert rep.output(admissible) == 3
    assert rep.output(rep.transition(admissible, 0)) == 3

    def unexpected(*args, **kwargs):
        raise AssertionError("polynomial work ran before state bit validation")

    monkeypatch.setattr(module, "polynomial_multiply_mod", unexpected)
    for value in (1 << 1000, -(1 << 1000)):
        state = ((value,), *rep.initial_state[1:])
        # Both values reduce to zero mod 4. Validation must inspect the raw
        # coefficient, rather than accepting its tiny modular representative.
        with pytest.raises(ResourceLimitError, match="coefficient bit length"):
            rep.output(state)
        with pytest.raises(ResourceLimitError, match="coefficient bit length"):
            rep.transition(state, 0)


def test_huge_word_allowance_is_not_exponentiated():
    class DepthSentinel(int):
        def __rpow__(self, base):
            raise AssertionError("the configured depth was exponentiated")

    rep = compile_prime_power_cartier_representation(
        (1,),
        (1,),
        2,
        2,
        budget=GeneratingBudget(maximum_recursion_depth=DepthSentinel(10**20)),
    )
    # The sentinel makes the original unbounded exponentiation fail safely.
    assert rep.evaluate(0) == 1
    assert rep.evaluate(1) == 0
    assert rep.evaluate(1 << 999) == 0


def test_word_limit_is_checked_before_any_transition(monkeypatch):
    rep = compile_prime_power_cartier_representation(
        (1,), (1,), 2, 2, budget=GeneratingBudget(maximum_recursion_depth=3)
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("transition ran before the entire word-length check")

    monkeypatch.setattr(PrimePowerCartierRepresentation, "transition", unexpected)
    with pytest.raises(ResourceLimitError, match="digit-word length"):
        rep.evaluate(1 << 1000)


def test_modular_power_skips_unused_square_outside_working_degree(monkeypatch):
    import komorebi_math.generating.cartier as module

    original = module.polynomial_multiply_mod
    degrees = []

    def guarded(left, right, modulus):
        degree = len(left) + len(right) - 2
        assert degree <= 2, "an unused polynomial square exceeded the declared working degree"
        degrees.append(degree)
        return original(left, right, modulus)

    monkeypatch.setattr(module, "polynomial_multiply_mod", guarded)
    assert module.polynomial_power_mod((1, 1), 2, 8) == (1, 2, 1)
    rep = compile_prime_power_cartier_representation(
        (1,), (1, 1), 2, 1, budget=GeneratingBudget(maximum_degree=2)
    )
    assert rep.sequence(8) == direct_mahler((1,), (1, 1), 2, 2, 8)
    assert degrees and max(degrees) == 2
