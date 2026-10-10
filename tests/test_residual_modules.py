"""Independent finite module closure, word and nonfree quotient controls."""

from dataclasses import FrozenInstanceError
from itertools import product
from random import Random

import pytest

from komorebi_math.exceptions import DimensionError, ResourceLimitError, ValidationError
from komorebi_math.generating.residual_modules import (
    ResidualModuleBudget,
    RingLinearRealization,
    canonical_residual_module,
    finite_word_comparison,
)


def action(matrix, state, modulus):
    return tuple(sum(matrix[i][j] * state[j] for j in range(len(state))) % modulus for i in range(len(state)))


def output_after(rep, state, word):
    for digit in word:
        state = action(rep.matrices[digit], state, rep.modulus)
    return sum(rep.output_row[i] * state[i] for i in range(rep.dimension)) % rep.modulus


def short_words(alphabet_size, horizon):
    return tuple(word for length in range(horizon) for word in product(range(alphabet_size), repeat=length))


def independent_module_closure(rep):
    """Closure under all pairwise sums and transitions, rather than orbit span."""
    states = {rep.zero, rep.seed}
    while True:
        old = set(states)
        for left in old:
            for right in old:
                states.add(tuple((left[i] + right[i]) % rep.modulus for i in range(rep.dimension)))
            for matrix in rep.matrices:
                states.add(action(matrix, left, rep.modulus))
        if old == states:
            return states


def source_nonfree_example():
    # Paper 24.4 uses row states.  Transpose its matrix for this column API.
    return RingLinearRealization(2, 2, (((1, 0), (2, 1)),), (1, 0), (2, 0))


def test_paper_nonfree_quotient_and_explicit_module_laws():
    rep = source_nonfree_example()
    quotient = canonical_residual_module(rep)
    assert quotient.ambient_cardinality == 16
    assert quotient.orbit_states == ((1, 0), (1, 2))
    assert len(quotient.reachable_states) == 8
    assert quotient.carrier_cardinality == 2
    assert len(quotient.unobservable_states) == 4
    assert quotient.additive_p_exponents() == (1,)
    assert quotient.scale(2, quotient.seed) == 0
    assert quotient.output(quotient.seed) == 2
    assert quotient.add(quotient.seed, quotient.seed) == 0
    assert quotient.verify_laws().valid
    assert len(quotient.verify_laws().checks) == 12
    comparison = finite_word_comparison(rep, quotient)
    assert comparison.equivalent
    assert comparison.length_horizon == 5  # 16=2**4 and 2=2**1.
    assert comparison.visited_pairs == 2
    # Minimizing the output automaton would return just one state instead.
    assert len({rep.evaluate((0,) * length) for length in range(12)}) == 1
    for state in quotient.reachable_states:
        cls = quotient.project(state)
        for digit in rep.alphabet:
            assert quotient.project(rep.transition(state, digit)) == quotient.transition(cls, digit)
        for scalar in range(rep.modulus):
            scaled = tuple(scalar * x % rep.modulus for x in state)
            assert quotient.project(scaled) == quotient.scale(scalar, cls)
        for other in quotient.reachable_states:
            summed = tuple((a + b) % rep.modulus for a, b in zip(state, other, strict=True))
            assert quotient.project(summed) == quotient.add(cls, quotient.project(other))


def test_free_and_mixed_nonfree_additive_types():
    free = canonical_residual_module(RingLinearRealization(2, 2, (((1,),),), (1,), (1,)))
    assert free.carrier_cardinality == 4
    assert free.additive_p_exponents() == (2,)
    mixed = canonical_residual_module(RingLinearRealization(2, 2, (((1, 0), (0, 0)),), (1, 1), (1, 2)))
    assert len(mixed.reachable_states) == 16
    assert mixed.carrier_cardinality == 8
    assert mixed.additive_p_exponents() == (1, 2)
    assert mixed.verify_laws().valid


def test_paper_constant_behavior_linear_carrier_and_sharp_word_horizon():
    constant = canonical_residual_module(RingLinearRealization(5, 1, (((1,),),), (1,), (1,)))
    assert len(constant.orbit_states) == 1
    assert constant.carrier_cardinality == 5  # Paper Example 24.3.
    assert constant.additive_p_exponents() == (1,)
    assert constant.verify_laws().valid
    shift = ((0, 0, 0), (1, 0, 0), (0, 1, 0))
    delayed = RingLinearRealization(2, 1, (shift,), (1, 0, 0), (0, 0, 1))
    zero = RingLinearRealization(2, 1, ((),), (), ())
    result = finite_word_comparison(delayed, zero)
    assert result.length_horizon == 3
    assert result.witness == (0, 0)
    assert len(result.witness) == result.length_horizon - 1
    assert all(delayed.evaluate(word) == zero.evaluate(word) for word in short_words(1, 2))


def test_noncommuting_word_orientation_and_shortest_witness():
    lower = ((0, 0), (1, 0))
    upper = ((0, 1), (0, 0))
    left = RingLinearRealization(2, 1, (upper, lower), (1, 0), (1, 0))
    right = RingLinearRealization(2, 1, (lower, upper), (1, 0), (1, 0))
    assert left.evaluate((0, 1)) == 0
    assert left.evaluate((1, 0)) == 1
    assert all(left.evaluate(word) == right.evaluate(word) for word in short_words(2, 2))
    result = finite_word_comparison(left, right)
    assert not result.equivalent
    assert result.witness == (0, 1)
    assert result.left_output == 0 and result.right_output == 1
    assert len(result.witness) < result.length_horizon == 4
    assert not finite_word_comparison(
        canonical_residual_module(left), canonical_residual_module(right)
    ).equivalent


def test_integer_digit_subset_does_not_certify_all_words():
    # Canonical positive binary strings read least-significant digit first have
    # last digit 1.  These machines agree on every such string and on zero, but
    # differ on the noncanonical high-order-zero word (1,0).
    reset_zero = ((0, 0, 0),) * 3
    move_first = ((0, 0, 0), (1, 0, 0), (0, 0, 0))
    move_second = ((0, 0, 0), (0, 0, 0), (0, 1, 0))
    left = RingLinearRealization(2, 1, (move_second, move_first), (1, 0, 0), (0, 0, 1))
    right = RingLinearRealization(2, 1, (reset_zero, move_first), (1, 0, 0), (0, 0, 1))
    assert left.evaluate(()) == right.evaluate(()) == 0
    assert left.evaluate((0,)) == right.evaluate((0,)) == 0
    comparison = finite_word_comparison(left, right)
    assert comparison.witness == (1, 0)
    assert comparison.left_output == 1 and comparison.right_output == 0
    assert all(
        left.evaluate(word) == right.evaluate(word) for word in short_words(2, 6) if word and word[-1] == 1
    )


@pytest.mark.parametrize("prime,exponent,dimension", [(2, 1, 3), (2, 2, 2), (3, 1, 2)])
def test_random_small_models_against_independent_complete_short_words(prime, exponent, dimension):
    rng = Random(624 + prime * 10 + exponent)
    modulus = prime**exponent
    for _ in range(12):
        matrices = tuple(
            tuple(tuple(rng.randrange(modulus) for _ in range(dimension)) for _ in range(dimension))
            for _ in range(2)
        )
        seed = tuple(rng.randrange(modulus) for _ in range(dimension))
        output = tuple(rng.randrange(modulus) for _ in range(dimension))
        rep = RingLinearRealization(prime, exponent, matrices, seed, output)
        quotient = canonical_residual_module(rep)
        reachable = independent_module_closure(rep)
        assert set(quotient.reachable_states) == reachable
        words = short_words(2, exponent * dimension)
        signatures = {state: tuple(output_after(rep, state, word) for word in words) for state in reachable}
        zero_signature = signatures[rep.zero]
        assert set(quotient.unobservable_states) == {
            state for state in reachable if signatures[state] == zero_signature
        }
        assert quotient.carrier_cardinality == len(set(signatures.values()))
        for state in reachable:
            for other in reachable:
                assert (quotient.project(state) == quotient.project(other)) == (
                    signatures[state] == signatures[other]
                )
        assert quotient.verify_laws().valid
        assert finite_word_comparison(rep, quotient).equivalent
        for word in short_words(2, 7):
            assert quotient.evaluate(word) == output_after(rep, seed, word)


def test_every_two_dimensional_binary_one_digit_model():
    states = tuple(product(range(2), repeat=2))
    checked = 0
    for coefficients in product(range(2), repeat=4):
        matrix = (coefficients[:2], coefficients[2:])
        for seed in states:
            for output in states:
                rep = RingLinearRealization(2, 1, (matrix,), seed, output)
                quotient = canonical_residual_module(rep)
                expected_reachable = independent_module_closure(rep)
                assert set(quotient.reachable_states) == expected_reachable
                words = short_words(1, 2)
                expected_classes = {
                    tuple(output_after(rep, state, word) for word in words) for state in expected_reachable
                }
                assert quotient.carrier_cardinality == len(expected_classes)
                assert quotient.verify_laws().valid
                assert finite_word_comparison(rep, quotient).equivalent
                checked += 1
    assert checked == 256


@pytest.mark.parametrize("prime,exponent", [(2, 1), (2, 2), (3, 1)])
def test_comparator_against_exhaustive_theorem_horizon(prime, exponent):
    rng = Random(179 + prime + exponent)
    modulus = prime**exponent
    for _ in range(20):
        models = [
            RingLinearRealization(
                prime,
                exponent,
                tuple(((rng.randrange(modulus),),) for _ in range(2)),
                (rng.randrange(modulus),),
                (rng.randrange(modulus),),
            )
            for _ in range(2)
        ]
        comparison = finite_word_comparison(*models)
        words = short_words(2, 2 * exponent)
        differing = [word for word in words if models[0].evaluate(word) != models[1].evaluate(word)]
        assert comparison.equivalent == (not differing)
        if differing:
            assert comparison.witness == differing[0]
            assert comparison.left_output == models[0].evaluate(differing[0])
            assert comparison.right_output == models[1].evaluate(differing[0])


def test_empty_alphabet_zero_seed_and_zero_dimension():
    no_letters = RingLinearRealization(2, 2, (), (1,), (2,))
    quotient = canonical_residual_module(no_letters)
    assert quotient.carrier_cardinality == 2
    assert quotient.alphabet == ()
    assert quotient.evaluate(()) == 2
    assert quotient.verify_laws().valid
    zero_seed = canonical_residual_module(RingLinearRealization(3, 2, (((1,),),), (0,), (1,)))
    assert zero_seed.carrier_cardinality == 1
    assert zero_seed.additive_p_exponents() == ()
    trivial = RingLinearRealization(2, 1, ((), ()), (), ())
    assert trivial.carrier_cardinality == 1
    assert finite_word_comparison(trivial, canonical_residual_module(trivial)).length_horizon == 0
    assert finite_word_comparison(trivial, canonical_residual_module(trivial)).equivalent


def test_resource_preflight_and_immutability():
    with pytest.raises(ResourceLimitError, match="exponent"):
        RingLinearRealization(2, 10**100, (), (), ())
    with pytest.raises(ResourceLimitError, match="scalar count"):
        RingLinearRealization(2, 64, (), (), ())
    with pytest.raises(ResourceLimitError, match="ambient cardinality"):
        RingLinearRealization(2, 2, (), (0,) * 32, (0,) * 32)
    with pytest.raises(ResourceLimitError, match="prime"):
        RingLinearRealization(10**80 + 7, 1, (), (), ())
    rep = source_nonfree_example()
    with pytest.raises(ResourceLimitError, match="construction work"):
        canonical_residual_module(rep, budget=ResidualModuleBudget(max_work=1))
    with pytest.raises(ResourceLimitError, match="operation entries"):
        canonical_residual_module(rep, budget=ResidualModuleBudget(max_operation_entries=1))
    quotient = canonical_residual_module(rep)
    with pytest.raises(ResourceLimitError, match="module-law work"):
        quotient.verify_laws(budget=ResidualModuleBudget(max_work=1))
    with pytest.raises(ResourceLimitError, match="product cardinality"):
        finite_word_comparison(rep, rep, budget=ResidualModuleBudget(max_carrier_size=16))
    with pytest.raises(FrozenInstanceError):
        quotient.seed = 0
    with pytest.raises(TypeError):
        quotient._projection[rep.zero] = 1


def test_reject_incompatible_rings_malformed_inputs_and_outside_states():
    with pytest.raises(ValidationError):
        RingLinearRealization(4, 1, (), (1,), (1,))
    with pytest.raises(ValidationError):
        RingLinearRealization(2, 0, (), (1,), (1,))
    with pytest.raises(ValidationError):
        RingLinearRealization(2, 1, (), (True,), (1,))
    with pytest.raises(DimensionError):
        RingLinearRealization(2, 1, (((1,),),), (1, 0), (1, 1))
    with pytest.raises(ValidationError):
        finite_word_comparison(
            source_nonfree_example(), RingLinearRealization(2, 1, (((1, 0), (0, 1)),), (1, 0), (1, 0))
        )
    quotient = canonical_residual_module(source_nonfree_example())
    with pytest.raises(ValidationError, match="reachable"):
        quotient.project((0, 1))
    with pytest.raises(ValidationError):
        quotient.output(2)
    with pytest.raises(ValidationError):
        quotient.transition(quotient.seed, 1)
    with pytest.raises(ValidationError):
        quotient.scale(True, quotient.seed)
