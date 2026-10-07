from __future__ import annotations

from fractions import Fraction
from math import comb

import pytest

from komorebi_math.exceptions import ResourceLimitError, ValidationError
from komorebi_math.export import dumps_portable, loads_portable
from komorebi_math.generating import (
    BellParityAutomaton,
    BellResidueAutomaton,
    FiniteNormalizedSeries,
    GeneratingBudget,
    IntegerDomain,
    PrimeFieldDomain,
    RationalDomain,
    ResidueRingDomain,
    TruncatedPadicDomain,
    basis_from_descriptor,
    basis_from_values,
    bell_numbers,
    certify_pointed_subindex_identity,
    compile_cartier_linear_representation,
    exponential_basis,
    factorial_power_basis,
    generalized_bell_digit_reduction,
    generalized_bell_numbers,
    ordinary_basis,
    weighted_bell_digit_reduction,
)


def test_builtin_basis_descriptors_round_trip():
    for basis in (
        ordinary_basis(),
        exponential_basis(),
        factorial_power_basis(4),
        basis_from_values("finite", (1, 2, 6, 24, 120)),
    ):
        descriptor = basis.require_descriptor()
        restored = basis_from_descriptor(descriptor)
        assert restored.require_descriptor() == descriptor
        assert tuple(restored.gamma(n) for n in range(5)) == tuple(
            basis.gamma(n) for n in range(5)
        )
        portable = loads_portable(dumps_portable(basis))
        assert portable.require_descriptor() == descriptor


def test_callback_basis_is_live_but_not_directly_portable():
    from komorebi_math.generating import WeightedMonomialBasis

    basis = WeightedMonomialBasis("live", lambda n: n + 1)
    assert not basis.portable
    with pytest.raises(ValueError, match="cannot be persisted"):
        dumps_portable(basis)


def test_formal_series_snapshots_callback_basis_finitely():
    from komorebi_math.generating import FormalSeries, WeightedMonomialBasis

    basis = WeightedMonomialBasis("live", lambda n: n + 1)
    source = FormalSeries(basis, (1, 2, 3, 4))
    restored = loads_portable(dumps_portable(source))
    assert restored.coefficients == source.coefficients
    assert restored.ordinary_coefficients() == source.ordinary_coefficients()
    with pytest.raises(IndexError):
        restored.basis.gamma(4)


def test_exact_domain_arithmetic_and_descriptors():
    integers = IntegerDomain()
    rationals = RationalDomain()
    residues = ResidueRingDomain(12)
    field = PrimeFieldDomain(7)
    padic = TruncatedPadicDomain(3, 4)

    assert integers.multiply(6, -7) == -42
    with pytest.raises(ValidationError):
        integers.inverse(2)
    assert rationals.inverse(Fraction(3, 5)) == Fraction(5, 3)
    assert residues.multiply(7, 5) == 11
    assert field.inverse(3) == 5
    assert padic.modulus == 81
    assert padic.valuation(27) == 3
    assert padic.valuation(0) == padic.precision


def test_finite_series_product_matches_ordinary_product_over_rationals():
    domain = RationalDomain()
    basis = factorial_power_basis(3)
    left = FiniteNormalizedSeries(basis, domain, (1, 2, -3, 5))
    right = FiniteNormalizedSeries(basis, domain, (2, -1, 4))
    product = left.multiply(right)
    a = left.ordinary_coefficients()
    b = right.ordinary_coefficients()
    expected = tuple(
        sum(
            (a[k] if k < len(a) else 0) * (b[n - k] if n - k < len(b) else 0)
            for k in range(n + 1)
        )
        for n in range(product.degree + 1)
    )
    assert product.ordinary_coefficients() == expected


def test_finite_series_transfer_exp_log_and_composition():
    domain = RationalDomain()
    source = FiniteNormalizedSeries(ordinary_basis(), domain, (0, 1, 2, -3, 5))
    transferred, certificate = source.convert(exponential_basis())
    certificate.require()
    assert transferred.ordinary_coefficients() == source.ordinary_coefficients()
    assert source.exp(degree=4).log(degree=4).coefficients == source.coefficients

    outer = FiniteNormalizedSeries(ordinary_basis(), domain, (1, 2, 3))
    inner = FiniteNormalizedSeries(ordinary_basis(), domain, (0, 1, 1, 1))
    composed = outer.compose(inner, degree=3)
    assert composed.coefficients == (Fraction(1), Fraction(2), Fraction(5), Fraction(8))


def test_finite_series_residue_ring_and_field_paths():
    field = PrimeFieldDomain(7)
    basis = exponential_basis()
    left = FiniteNormalizedSeries(basis, field, (1, 2, 3, 4))
    right = FiniteNormalizedSeries(basis, field, (2, 1, 5))
    product = left.multiply(right)
    for n, value in enumerate(product.coefficients):
        expected = sum(
            comb(n, k) * left.coefficient(k) * right.coefficient(n - k)
            for k in range(n + 1)
        ) % 7
        assert value == expected

    ring = ResidueRingDomain(8)
    nonunit = FiniteNormalizedSeries(ordinary_basis(), ring, (2, 1))
    with pytest.raises(ValidationError, match="invertible"):
        nonunit.reciprocal(degree=3)


def test_finite_series_portable_round_trip_preserves_domain_and_budget():
    budget = GeneratingBudget(maximum_degree=30, maximum_coefficient_bits=1000)
    source = FiniteNormalizedSeries(
        factorial_power_basis(2),
        TruncatedPadicDomain(5, 3),
        (1, 2, 7, 11),
        budget=budget,
    )
    restored = loads_portable(dumps_portable(source))
    assert restored == source
    assert restored.domain.descriptor() == source.domain.descriptor()
    assert restored.budget == budget


def test_resource_contract_raises_typed_exception():
    budget = GeneratingBudget(maximum_degree=3, maximum_coefficient_bits=8)
    with pytest.raises(ResourceLimitError):
        FiniteNormalizedSeries(ordinary_basis(), RationalDomain(), range(5), budget=budget)
    with pytest.raises(ResourceLimitError):
        FiniteNormalizedSeries(ordinary_basis(), RationalDomain(), (2**20,), budget=budget)


def test_pointed_subindex_and_bell_digit_reductions():
    for prime in (2, 3, 5, 7):
        for index in range(1, 40):
            certify_pointed_subindex_identity(index, prime).certificate().require()
        power = 1 + 2 * (prime - 1)
        for index in range(35):
            generalized_bell_digit_reduction(index, prime, power).certificate().require()


def test_weighted_bell_digit_reduction():
    prime = 5
    power = 9
    weights = tuple(n * n + 3 for n in range(80))
    for index in range(30):
        weighted_bell_digit_reduction(index, prime, power, weights).certificate().require()


def test_bell_parity_automaton_matches_generalized_family():
    automaton = BellParityAutomaton()
    values = generalized_bell_numbers(5, 400)
    assert tuple(value % 2 for value in values) == tuple(
        automaton.evaluate(index) for index in range(400)
    )


def test_bell_residue_automaton_matches_direct_values():
    for prime in (2, 3, 5):
        power = 1 + 2 * (prime - 1)
        automaton = BellResidueAutomaton.compile(prime, power)
        automaton.certificate(120).require()
        direct_count = min(200, automaton.period_bound)
        ordinary = bell_numbers(direct_count)
        assert automaton.outputs[:direct_count] == tuple(value % prime for value in ordinary)
        assert all(
            automaton.outputs[n + prime]
            == (automaton.outputs[n + 1] + automaton.outputs[n]) % prime
            for n in range(automaton.period_bound - prime)
        )


def test_cartier_compiler_honors_matrix_budget():
    budget = GeneratingBudget(maximum_matrix_dimension=1)
    with pytest.raises(ResourceLimitError):
        compile_cartier_linear_representation((1, 1), (1, -1), 3, budget=budget)


def test_same_divided_power_basis_needs_no_factorial_inversion():
    field = PrimeFieldDomain(5)
    basis = exponential_basis()
    left = FiniteNormalizedSeries(basis, field, tuple(range(12)))
    right = FiniteNormalizedSeries(basis, field, tuple(range(11, -1, -1)))
    converted, certificate = right.convert(exponential_basis())
    certificate.require()
    assert converted.coefficients == right.coefficients
    product = left.multiply(right)
    for n, value in enumerate(product.coefficients):
        expected = sum(
            comb(n, k) * left.coefficient(k) * right.coefficient(n - k)
            for k in range(n + 1)
        ) % 5
        assert value == expected
