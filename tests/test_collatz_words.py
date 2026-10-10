"""Independent finite controls for the Collatz paper's word/cylinder laws."""

from itertools import product, repeat

import pytest

from komorebi_math.arithmetic.collatz import (
    CollatzWord,
    FirstDescentResult,
    first_odd_descent,
    odd_collatz_step,
)
from komorebi_math.exceptions import ResourceLimitError, ValidationError


def direct_step(n):
    n = 3 * n + 1
    count = 0
    while n % 2 == 0:
        n //= 2
        count += 1
    return n, count


def test_all_small_words_against_direct_integer_orbits():
    checks = 0
    for length in range(4):
        for exponents in product(range(1, 5), repeat=length):
            word = CollatzWord(exponents)
            for seed in range(1, 512, 2):
                value = seed
                actual = []
                for _ in range(length):
                    value, valuation = direct_step(value)
                    actual.append(valuation)
                realizing = tuple(actual) == exponents
                assert word.realizes(seed) == realizing
                if realizing:
                    assert word.apply(seed) == value
                    assert word.orbit(seed)[-1] == value
                    assert word.descends(seed) == (value < seed)
                checks += 1
    assert checks == 21_760


def test_terminal_bit_order_and_original_source_comparison():
    forward, reverse = CollatzWord((1, 3)), CollatzWord((3, 1))
    assert (forward.depth, forward.offset, forward.residue, forward.modulus) == (4, 5, 19, 32)
    assert (reverse.offset, reverse.residue) == (11, 29)
    assert forward.affine_value(3) == 2
    assert not forward.realizes(3)
    with pytest.raises(ValidationError, match="realize"):
        forward.apply(3)
    for parameter in [0, 1, 2, 123, 2**80]:
        assert forward.orbit(32 * parameter + 19) == (
            32 * parameter + 19,
            48 * parameter + 29,
            18 * parameter + 11,
        )
        assert reverse.orbit(32 * parameter + 29) == (
            32 * parameter + 29,
            12 * parameter + 11,
            18 * parameter + 17,
        )


def test_contraction_alone_does_not_certify_endpoint_descent():
    word = CollatzWord((4, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2, 3))
    assert word.coefficient_gap == 5_077_565
    assert word.offset == 1_106_233_681
    assert word.descent_threshold == 217
    assert word.slope < 1 and word.apply(165) == 167 and not word.descends(165)
    first = first_odd_descent(165)
    assert first.status == "descent" and first.states == (165, 31)


def test_composition_keeps_offsets_and_realization():
    left, right = CollatzWord((1,)), CollatzWord((3, 1))
    combined = left.concatenate(right)
    assert combined.offset == right.multiplier * left.offset + left.denominator * right.offset
    for parameter in [0, 1, 10, 2**60]:
        seed = combined.residue + parameter * combined.modulus
        assert combined.apply(seed) == right.apply(left.apply(seed))
    with pytest.raises(ValidationError):
        left.concatenate((3, 1))


def test_uniform_descent_start_and_fixed_point_strictness():
    word = CollatzWord((2, 2))
    assert word.descent_threshold == 1
    assert word.uniform_descent_start == 33
    assert not word.descends(1)
    assert word.descends(33)
    assert CollatzWord((1,)).uniform_descent_start is None
    for length in range(1, 4):
        for values in product(range(1, 4), repeat=length):
            packet = CollatzWord(values)
            start = packet.uniform_descent_start
            if start is not None:
                assert packet.realizes(start) and packet.descends(start)
                previous = start - packet.modulus
                if previous > 0:
                    assert not packet.descends(previous)


def test_progression_through_151_and_clocks():
    word = CollatzWord((1, 1, 10))
    assert word.residue == 151 and word.modulus == 8192
    for parameter in [0, 1, 100, 2**80]:
        seed = 151 + 8192 * parameter
        assert word.orbit(seed) == (
            seed,
            227 + 12288 * parameter,
            341 + 18432 * parameter,
            1 + 54 * parameter,
        )
        assert first_odd_descent(seed, max_steps=3).word == word
    prefix = CollatzWord((1, 2))
    assert prefix.orbit(27) == (27, 41, 31)
    assert (prefix.length, prefix.depth, prefix.depth + prefix.length) == (2, 3, 5)


def test_bounded_search_and_record_validation():
    assert first_odd_descent(1).status == "terminal"
    result = first_odd_descent(27, max_steps=1)
    assert result.status == "inconclusive" and not result.certificate.valid
    with pytest.raises(ValidationError):
        result.certificate.require()
    assert first_odd_descent(27, max_steps=100).states[-1] < 27
    with pytest.raises(ValidationError):
        FirstDescentResult(27, (27, 41), CollatzWord((1,)), "descent")
    with pytest.raises(ValidationError):
        FirstDescentResult(165, (165, 31), CollatzWord((4,)), "inconclusive")


def test_empty_word_and_input_ownership():
    empty = CollatzWord((), max_steps=0, max_depth=0)
    assert empty.residue == 1 and empty.modulus == 2 and empty.apply(27) == 27
    values = [1, 3]
    word = CollatzWord(values)
    values[:] = [3, 1]
    assert word.valuations == (1, 3)
    assert word.realization_certificate(19).valid


@pytest.mark.parametrize("value", [True, False, 1.0, 1.5, "1", None, -1, 0])
def test_invalid_valuations_are_rejected(value):
    with pytest.raises(ValidationError):
        CollatzWord((value,))


@pytest.mark.parametrize("value", [True, 1.0, "1", None, -1, 0, 2])
def test_invalid_seeds_are_rejected(value):
    with pytest.raises(ValidationError):
        odd_collatz_step(value)


def test_resources_and_infinite_iterable_are_bounded():
    with pytest.raises(ResourceLimitError):
        CollatzWord(repeat(1), max_steps=3)
    with pytest.raises(ResourceLimitError):
        CollatzWord((100_001,))
    with pytest.raises(ResourceLimitError):
        first_odd_descent(27, max_bits=4)
    with pytest.raises(ResourceLimitError):
        first_odd_descent(27, max_depth=0)
    with pytest.raises(ValidationError):
        first_odd_descent(27, max_steps=True)
