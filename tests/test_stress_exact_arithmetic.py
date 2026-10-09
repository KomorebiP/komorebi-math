"""Bounded, deterministic exact controls for release stress qualification.

These comparisons test finite ranges. They are not proofs of unbounded claims.
"""
from fractions import Fraction
from itertools import permutations
from math import comb, factorial
import random

import numpy as np
import pytest

from komorebi_math.arithmetic.hurwitz import (
    HurwitzSeries, compressed_hurwitz_inverse, hurwitz_inverse, hurwitz_product,
    subgroup_selector,
)
from komorebi_math.arithmetic.recurrence import (
    LinearRecurrence, matrix_power_exact, matrix_power_mod,
)
from komorebi_math.coordinates.packet import (
    digit_reversal_order, residue_class_order, tensor_axis_permutation_order,
)
from komorebi_math.coordinates.recurrence import (
    analyze_recurrence_coordinates, variation_of_constants_prefix,
)
from komorebi_math.coordinates.exact import as_matrix, matrix_power
from komorebi_math.exceptions import DimensionError, ResourceLimitError, ValidationError
from komorebi_math.generating import (
    FiniteNormalizedSeries, FormalSeries, GeneratingBudget, RationalDomain,
    exponential_basis, factorial_power_basis, fibonacci_factorial_basis,
    ordinary_basis, q_factorial_basis, IntegerDomain, ResidueRingDomain,
    WeightedMonomialBasis, basis_from_values, pochhammer_basis,
)
from komorebi_math.sequences.base import Sequence
from komorebi_math.sequences.oeis import OEISRecord, compare_oeis_prefix
from komorebi_math.sequences.transforms import (
    binomial_transform_values, inverse_binomial_transform_values, partial_sums,
)

SEED = 20261009
MODULI = (2, 7, 10**6 + 3, 10**12 + 39, 2**63 + 159, 10**100 + 7)


def _direct_matrix_power(matrix, power, modulus):
    """Repeated multiplication with Python integers, independent of NumPy."""
    n = len(matrix)
    result = [[int(i == j) for j in range(n)] for i in range(n)]
    for _ in range(power):
        result = [[sum(result[i][k] * int(matrix[k][j]) for k in range(n)) % modulus
                   for j in range(n)] for i in range(n)]
    return result


@pytest.mark.parametrize("modulus", MODULI)
def test_arbitrary_precision_modular_matrix_controls(modulus):
    rng = random.Random(SEED)
    for size in (1, 2, 3):
        for _ in range(8):
            matrix = [[rng.randrange(-10**40, 10**40) for _ in range(size)]
                      for _ in range(size)]
            # A small first entry must not force later entries into int64.
            matrix[0][0] = rng.choice((0, 1))
            for power in (0, 1, 2, 3, 6, 11):
                actual = matrix_power_mod(np.array(matrix, dtype=object), power, modulus)
                assert actual.tolist() == _direct_matrix_power(matrix, power, modulus)


@pytest.mark.parametrize("modulus", MODULI)
def test_recurrence_modular_terms_packets_and_periods(modulus):
    rng = random.Random(SEED)
    for order in (1, 2, 3, 5):
        for variant in range(4):
            coefficients = tuple(rng.randrange(-10**18, 10**18) for _ in range(order))
            initial = tuple(rng.randrange(-10**30, 10**30) for _ in range(order))
            if variant % 2:
                coefficients = tuple(Fraction(v) for v in coefficients)
                initial = tuple(Fraction(v) for v in initial)
            recurrence = LinearRecurrence(coefficients, initial)
            direct = list(initial)
            for n in range(40 - order):
                direct.append(sum(coefficients[k] * direct[n+k] for k in range(order)))
            for n in (0, order - 1, order, 6, 17, 39):
                assert recurrence.term_mod(n, modulus) == int(direct[n]) % modulus
            for step in (1, 3, 7):
                for address in range(step):
                    packet = recurrence.residue_packet(step, address)
                    for k in range(4):
                        assert packet.term_mod(k, modulus) == int(direct[step*k+address]) % modulus
    periodic = LinearRecurrence((Fraction(-1), Fraction(1)), (Fraction(0), Fraction(1)))
    period = periodic.period_mod(modulus, max_steps=10)
    assert (period.preperiod, period.period) == (0, 3 if modulus == 2 else 6)
    assert period.certificate.valid
    assert periodic.term_mod(6, modulus) == 0


def test_numpy_integer_inputs_keep_exact_arithmetic():
    recurrence = LinearRecurrence(np.array([2]), np.array([2**62]))
    assert recurrence.first(8) == tuple(2**(62+n) for n in range(8))
    assert recurrence.certificate(8).certificate.valid
    actual = matrix_power_exact(np.array([[2**32]], dtype=np.int64), 4)
    assert actual[0, 0] == 2**128
    values = [np.int64(2**62)] * 8
    transformed = binomial_transform_values(values)
    assert transformed == tuple(2**(62+n) for n in range(8))
    assert inverse_binomial_transform_values(transformed) == tuple(2**62 for _ in range(8))
    sequence = Sequence(lambda n: np.int64(2**62))
    assert partial_sums(sequence).first(8) == tuple((n+1)*2**62 for n in range(8))
    assert hurwitz_product(values[:3], values[:3], degree=4) == tuple(
        sum(comb(n,k)*2**124 for k in range(n+1) if k<3 and n-k<3)
        for n in range(5))


def test_4096_bit_recurrence_data_and_long_modular_index():
    coefficients = (Fraction(2**4096 + 3), Fraction(-(2**2048 + 1)))
    initial = (Fraction(-(2**4096 + 19)), Fraction(2**3072 + 7))
    recurrence = LinearRecurrence(coefficients, initial)
    modulus = 2**1024 + 643
    direct = list(initial)
    for n in range(10):
        direct.append(sum(coefficients[k] * direct[n+k] for k in range(2)))
    for n, expected in enumerate(direct):
        assert recurrence.term(n) == expected
        assert recurrence.term_mod(n, modulus) == int(expected) % modulus
    # A period-six sequence is an independent oracle at an enormous index.
    periodic = LinearRecurrence((-1, 1), (0, 1))
    huge_index = 10**1000 + 3
    expected = (0, 1, 1, 0, -1, -1)[huge_index % 6] % modulus
    assert periodic.term_mod(huge_index, modulus) == expected


@pytest.mark.parametrize("wrap", (lambda value: np.int64(value),
    lambda value: Fraction(np.int64(value)),
    lambda value: Fraction(np.int64(value), np.int64(1))))
def test_supported_numpy_rational_intakes_are_native_exact(wrap):
    value = wrap(2**32)
    integer = IntegerDomain()
    rational = RationalDomain()
    residue = ResidueRingDomain(2**80 + 7)
    assert integer.multiply(value,value) == 2**64
    assert rational.multiply(value,value) == Fraction(2**64)
    assert residue.multiply(value,value) == 2**64
    assert type(integer.coerce(value)) is int
    assert type(rational.coerce(value).numerator) is int
    matrix = as_matrix(((value,),))
    assert matrix_power(matrix,2) == ((Fraction(2**64),),)
    formal = FormalSeries(ordinary_basis(),(value,value))
    assert formal.multiply(formal,degree=2).coefficients == (2**64,2**65,2**64)
    assert formal.scale(value).coefficients == (2**64,2**64)
    custom = WeightedMonomialBasis('numpy-stress',lambda n: value if n else 1)
    table = basis_from_values('numpy-table',(1,value,value))
    for basis in (custom,table):
        assert basis.structure_constant(2,1) == Fraction(1,2**32)
    assert pochhammer_basis(value).gamma(2) == Fraction(2**32*(2**32+1))
    recurrence = LinearRecurrence((value,),(value,))
    assert recurrence.term(1) == 2**64
    assert binomial_transform_values((value,)*3) == (2**32,2**33,2**34)
    assert partial_sums(Sequence(lambda n:value)).term(2) == 3*2**32


@pytest.mark.parametrize("modulus", (0, 1, -2, True, False, 2.5, 3.0, "7"))
def test_hurwitz_rejects_invalid_modulus_before_iteration(modulus):
    def untouched():
        raise AssertionError("invalid modulus consumed coefficient iterable")
        yield 1
    for operation in (
        lambda: HurwitzSeries(untouched(), modulus),
        lambda: hurwitz_product(untouched(), untouched(), modulus=modulus),
        lambda: hurwitz_inverse(untouched(), 0, modulus=modulus),
        lambda: HurwitzSeries.exponential_mode(2, 3, modulus=modulus),
    ):
        with pytest.raises(ValidationError):
            operation()


@pytest.mark.parametrize("modulus", (2, 7, 12, 10**12 + 39, 10**100 + 7))
def test_hurwitz_dense_independent_convolution_and_inverse(modulus):
    rng = random.Random(SEED)
    for degree in range(13):
        a = (1, *(rng.randrange(-10**30,10**30) for _ in range(degree)))
        b = tuple(rng.randrange(-10**30,10**30) for _ in range(degree+1))
        expected = tuple(sum(comb(n,k)*a[k]*b[n-k] for k in range(n+1)) % modulus
                         for n in range(degree+1))
        assert hurwitz_product(a,b,degree=degree,modulus=modulus) == expected
        inverse = hurwitz_inverse(a,degree,modulus=modulus)
        assert hurwitz_product(a,inverse,degree=degree,modulus=modulus) == (1,)+(0,)*degree
    # Exact EGF conversion is a separate, independent product control.
    a = tuple(Fraction(rng.randrange(-50,51),rng.randrange(1,9)) for _ in range(9))
    b = tuple(Fraction(rng.randrange(-50,51),rng.randrange(1,9)) for _ in range(9))
    actual = hurwitz_product(a,b,degree=16)
    expected = tuple(factorial(n)*sum(a[k]/factorial(k)*b[n-k]/factorial(n-k)
                       for k in range(n+1) if k<len(a) and n-k<len(b)) for n in range(17))
    assert actual == expected


@pytest.mark.parametrize("prime", (2, 3, 5, 7, 11, 13, 17, 19))
def test_compressed_hurwitz_dense_alignment(prime):
    for compression in range(1,prime):
        if (prime-1) % compression:
            continue
        count = 12
        selector = subgroup_selector(prime, compression, compression*count)
        dense = selector.inverse().coefficients
        compressed = compressed_hurwitz_inverse(prime,compression,count)
        assert tuple(dense[compression*n] for n in range(count+1)) == compressed
        assert all(value == 0 for n,value in enumerate(dense) if n % compression)


@pytest.mark.parametrize("factory", (ordinary_basis, exponential_basis,
    lambda: factorial_power_basis(2), fibonacci_factorial_basis,
    lambda:q_factorial_basis(2)))
def test_normalized_series_controls(factory):
    rng = random.Random(SEED)
    basis = factory()
    ordinary = ordinary_basis()
    domain = RationalDomain()
    for degree in range(0,13):
        a = tuple(Fraction(rng.randrange(-10**8,10**8),rng.randrange(1,13)) for _ in range(degree+1))
        b = tuple(Fraction(rng.randrange(-10**8,10**8),rng.randrange(1,13)) for _ in range(degree+1))
        left = FormalSeries.from_ordinary_coefficients(basis,a)
        right = FormalSeries.from_ordinary_coefficients(basis,b)
        expected = tuple(sum(a[k]*b[n-k] for k in range(n+1)) for n in range(degree+1))
        assert left.multiply(right,degree=degree).ordinary_coefficients() == expected
        finite = FiniteNormalizedSeries.from_ordinary_coefficients(basis,domain,a)
        other = FiniteNormalizedSeries.from_ordinary_coefficients(basis,domain,b)
        assert finite.multiply(other,degree=degree).ordinary_coefficients() == expected
        converted,certificate = finite.convert(ordinary)
        assert certificate.valid and converted.ordinary_coefficients() == a
        inverse_source = FormalSeries.from_ordinary_coefficients(basis,(Fraction(1),)+a[1:])
        assert inverse_source.multiply(inverse_source.reciprocal(degree=degree),degree=degree).ordinary_coefficients() == (1,)+(0,)*degree
        zero = FormalSeries.from_ordinary_coefficients(basis,(Fraction(0),)+a[1:])
        assert zero.exp(degree=degree).log(degree=degree).ordinary_coefficients() == zero.ordinary_coefficients()
        assert left.ordinary_integral().ordinary_derivative().ordinary_coefficients() == a
        for modulus in (1,2,5):
            for address in range(modulus):
                assert left.packet_identity(modulus,address,target_basis=ordinary).valid


def test_resource_budget_stops_iterators_before_excess_coercion():
    consumed = []
    def coefficients():
        for n in range(100):
            consumed.append(n)
            yield object() if n == 3 else n
        raise AssertionError("budget failed to stop iteration")
    budget = GeneratingBudget(maximum_degree=2)
    with pytest.raises(ResourceLimitError):
        FiniteNormalizedSeries(ordinary_basis(), RationalDomain(), coefficients(),budget=budget)
    assert consumed == [0,1,2,3]
    def oversized():
        yield 2**200
        raise AssertionError("oversized coefficient failed to stop iteration")
    with pytest.raises(ResourceLimitError):
        FiniteNormalizedSeries(ordinary_basis(),RationalDomain(),oversized(),
            budget=GeneratingBudget(maximum_coefficient_bits=100))


def test_recurrence_coordinates_exact_roundtrips():
    rng = random.Random(SEED)
    for order in range(1,6):
        for length in (order,order+1,order+8,order+20):
            coefficients = tuple(Fraction(rng.randrange(-100,101),rng.randrange(1,8)) for _ in range(order))
            values = tuple(Fraction(rng.randrange(-10**60,10**60),rng.randrange(1,13)) for _ in range(length))
            record = analyze_recurrence_coordinates(values,coefficients)
            assert record.reconstruct() == values
            assert variation_of_constants_prefix(record) == values
            assert record.certificate().valid


@pytest.mark.parametrize("axes", ((False,True),(0.0,1.0),(0,"1"),(0,None),(0,0),(0,2)))
def test_tensor_axes_reject_noninteger_and_malformed_permutations(axes):
    with pytest.raises(ValidationError):
        tensor_axis_permutation_order((2,3), axes)


def test_packet_orders_independent_controls():
    for shape in ((1,),(2,3),(2,3,4),(1,2,3,2)):
        values = np.arange(np.prod(shape)).reshape(shape)
        for axes in permutations(range(len(shape))):
            assert tensor_axis_permutation_order(shape,axes) == tuple(values.transpose(axes).ravel())
    for radix in range(2,8):
        for width in range(4):
            n = radix**width
            order = digit_reversal_order(n,radix)
            assert tuple(sorted(order)) == tuple(range(n))
            assert tuple(order[k] for k in order) == tuple(range(n))
    for length in range(1,40):
        for modulus in range(1,50):
            expected = tuple(sorted(range(length),key=lambda n:(n % modulus,n)))
            assert residue_class_order(length,modulus) == expected
    assert residue_class_order(3,10**30) == (0,1,2)


def test_oeis_offset_exhaustion_and_mismatch_controls():
    for offset in (-100,-1,0,1,100):
        record = OEISRecord('A000001',tuple(range(12)),offset)
        for start in range(0,15):
            for count in range(1,17):
                query = tuple(range(start,start+count))
                match = compare_oeis_prefix(query,record,query_offset=offset+start,minimum_terms=1)
                assert match.compared_terms == min(count,max(12-start,0))
                assert match.passed == (start<12 and start+count<=12)
                if match.compared_terms:
                    corrupted = (*query[:-1],query[-1]+1)
                    altered = compare_oeis_prefix(corrupted,record,query_offset=offset+start,minimum_terms=1)
                    assert altered.passed is False
    unknown = compare_oeis_prefix((0,1,2),OEISRecord('A000001',(0,1,2)),minimum_terms=1)
    assert unknown.passed and not unknown.index_alignment_verified


def test_tensor_axes_preserve_numpy_integral_indices():
    axes = np.array([1, 0], dtype=np.int64)
    expected = tuple(np.arange(6).reshape(2, 3).transpose(axes).ravel())
    assert tensor_axis_permutation_order((2, 3), axes) == expected
    assert expected == (0, 3, 1, 4, 2, 5)
