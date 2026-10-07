from __future__ import annotations

from fractions import Fraction
from itertools import product
import math

import numpy as np
import pytest

from komorebi_math.core.carrier_change import (
    SignedMassPacket,
    TropicalResidualPacket,
    boolean_support_nonnegative,
    finite_factorization_audit,
)
from komorebi_math.exceptions import ValidationError


def direct_free_energy(scores: list[float], temperature: float) -> float:
    if not scores:
        return -math.inf
    m = max(scores)
    return m + temperature * math.log(sum(math.exp((x - m) / temperature) for x in scores))


def test_nonnegative_boolean_support_is_semiring_compatible() -> None:
    values = (0.0, 0.25, 2.0, 7.0)
    for left, right in product(values, repeat=2):
        assert boolean_support_nonnegative(left + right) == (
            boolean_support_nonnegative(left) or boolean_support_nonnegative(right)
        )
        assert boolean_support_nonnegative(left * right) == (
            boolean_support_nonnegative(left) and boolean_support_nonnegative(right)
        )

    with pytest.raises(ValidationError):
        boolean_support_nonnegative(-1.0)


def test_signed_cancellation_is_a_no_go_witness_for_zerosumfree_projection() -> None:
    left = SignedMassPacket.from_scalar(1)
    right = SignedMassPacket.from_scalar(-1)
    merged = left.merge(right)

    assert left.value == 1
    assert right.value == -1
    assert merged.value == 0
    assert merged.total_variation == 2
    assert merged.cancellation_mass == 1
    assert merged.cancellation_defect == 2

    # Any support-like target using OR cannot reproduce signed addition here:
    # both inputs are present, yet their exact sum vanishes.
    assert (left.value != 0) or (right.value != 0)
    assert merged.value == 0


def test_signed_mass_packet_observation_preserves_addition_and_multiplication_exactly() -> None:
    a = SignedMassPacket.from_values([Fraction(3, 2), Fraction(-1, 3), Fraction(5, 7)])
    b = SignedMassPacket.from_values([Fraction(-2, 5), Fraction(4, 9)])

    assert a.merge(b).value == a.value + b.value
    assert a.multiply(b).value == a.value * b.value
    assert a.cancellation_defect == 2 * min(a.positive, a.negative)


def test_haar_cancellation_is_recorded_by_signed_mass_packet() -> None:
    odd = SignedMassPacket.from_values([Fraction(1, 2), Fraction(-1, 2)])
    even = SignedMassPacket.from_values([Fraction(1, 2), Fraction(1, 2)])

    assert odd.value == 0
    assert odd.total_variation == 1
    assert odd.cancellation_defect == 1
    assert even.value == 1
    assert even.total_variation == 1
    assert even.cancellation_defect == 0


def test_tropical_residual_packet_recovers_log_sum_exp_exactly() -> None:
    scores = [2.0, 1.0, -3.0, 0.25]
    tau = 0.7
    packet = TropicalResidualPacket.from_scores(scores, temperature=tau)

    assert packet.tropical_value == max(scores)
    assert packet.free_energy == pytest.approx(direct_free_energy(scores, tau), abs=1e-14)
    assert 0.0 <= packet.correction <= packet.correction_upper_bound + 1e-14


def test_tropical_residual_merge_is_exact_and_projects_to_max() -> None:
    tau = 0.8
    left_scores = [1.3, -0.2, 0.1]
    right_scores = [2.1, -1.7]
    left = TropicalResidualPacket.from_scores(left_scores, temperature=tau)
    right = TropicalResidualPacket.from_scores(right_scores, temperature=tau)
    merged = left.merge(right)

    assert merged.tropical_value == max(left.tropical_value, right.tropical_value)
    assert merged.free_energy == pytest.approx(
        direct_free_energy(left_scores + right_scores, tau),
        abs=1e-14,
    )
    assert merged.term_count == len(left_scores) + len(right_scores)


def test_tropical_residual_multiplication_is_exact_and_projects_to_plus() -> None:
    tau = 0.6
    left_scores = [0.4, 1.7]
    right_scores = [-0.3, 0.2, 1.1]
    left = TropicalResidualPacket.from_scores(left_scores, temperature=tau)
    right = TropicalResidualPacket.from_scores(right_scores, temperature=tau)
    product_packet = left.multiply(right)
    pair_scores = [a + b for a in left_scores for b in right_scores]

    assert product_packet.tropical_value == pytest.approx(left.tropical_value + right.tropical_value)
    assert product_packet.free_energy == pytest.approx(direct_free_energy(pair_scores, tau), abs=1e-14)
    assert product_packet.term_count == len(pair_scores)
    assert product_packet.correction == pytest.approx(left.correction + right.correction, abs=1e-14)


def test_same_tropical_value_can_hide_different_source_semantics() -> None:
    tau = 1.0
    singleton = TropicalResidualPacket.from_scores([0.0], temperature=tau)
    duplicate = TropicalResidualPacket.from_scores([0.0, 0.0], temperature=tau)

    assert singleton.tropical_value == duplicate.tropical_value == 0.0
    assert singleton.correction == 0.0
    assert duplicate.correction == pytest.approx(math.log(2.0))
    assert singleton.free_energy != duplicate.free_energy


def test_dequantization_error_is_uniformly_bounded_and_vanishes_with_temperature() -> None:
    scores = [3.0, 3.0, 1.0, -4.0]
    previous = math.inf
    for tau in (1.0, 0.5, 0.2, 0.05, 0.01):
        packet = TropicalResidualPacket.from_scores(scores, temperature=tau)
        assert packet.correction <= tau * math.log(len(scores)) + 1e-14
        assert packet.correction <= previous + 1e-14
        previous = packet.correction
    assert previous < 0.02


def test_tropical_zero_and_temperature_contracts() -> None:
    zero = TropicalResidualPacket.from_scores([], temperature=0.5)
    nonzero = TropicalResidualPacket.from_scores([1.0, 0.0], temperature=0.5)

    assert zero.is_zero
    assert zero.merge(nonzero) == nonzero
    assert zero.multiply(nonzero).is_zero

    with pytest.raises(ValidationError):
        nonzero.merge(TropicalResidualPacket.from_scores([1.0], temperature=1.0))
    with pytest.raises(ValidationError):
        TropicalResidualPacket.from_scores([math.inf])


def test_factorization_audit_detects_semantic_collision() -> None:
    # Same max-plus value, different finite-temperature sum-product semantics.
    tau = 1.0
    raw = ([0.0], [0.0, 0.0])
    source = [TropicalResidualPacket.from_scores(x, temperature=tau).tropical_value for x in raw]
    target = [TropicalResidualPacket.from_scores(x, temperature=tau).free_energy for x in raw]
    audit = finite_factorization_audit(source, target)

    assert not audit.exact_on_sample
    assert audit.source_class_count == 1
    assert audit.joint_class_count == 2
    assert audit.witness is not None


def test_factorization_audit_confirms_nonnegative_support_factors_through_sum() -> None:
    raw_decompositions = (
        (),
        (0.0,),
        (1.0,),
        (0.0, 2.0),
        (0.5, 0.5),
        (3.0, 0.0, 0.0),
    )
    sums = [sum(values) for values in raw_decompositions]
    supports = [boolean_support_nonnegative(value) for value in sums]
    audit = finite_factorization_audit(sums, supports)

    assert audit.exact_on_sample
    assert audit.joint_class_count == audit.source_class_count


def test_sum_product_and_max_plus_do_not_factor_through_each_other() -> None:
    tau = 1.0
    # Same max, different finite-temperature sum.
    a = TropicalResidualPacket.from_scores([0.0], temperature=tau)
    b = TropicalResidualPacket.from_scores([0.0, 0.0], temperature=tau)
    assert a.tropical_value == b.tropical_value
    assert a.free_energy != b.free_energy

    # Same finite-temperature sum, different max: e^0 + e^0 = e^(log 2).
    c = TropicalResidualPacket.from_scores([0.0, 0.0], temperature=tau)
    d = TropicalResidualPacket.from_scores([math.log(2.0)], temperature=tau)
    assert c.free_energy == pytest.approx(d.free_energy)
    assert c.tropical_value != d.tropical_value


def test_equal_branches_saturate_tropical_correction_bound() -> None:
    tau = 0.3
    for count in (1, 2, 7, 100):
        packet = TropicalResidualPacket.from_scores([4.0] * count, temperature=tau)
        assert packet.correction == pytest.approx(tau * math.log(count), abs=1e-14)
        assert packet.effective_multiplicity == pytest.approx(float(count), rel=1e-13)


def test_no_uniform_finite_temperature_error_when_fan_in_is_unbounded() -> None:
    tau = 0.5
    small = TropicalResidualPacket.from_scores([0.0] * 2, temperature=tau)
    large = TropicalResidualPacket.from_scores([0.0] * 10000, temperature=tau)
    assert large.tropical_value == small.tropical_value == 0.0
    assert large.correction > small.correction
    assert large.correction == pytest.approx(tau * math.log(10000), rel=1e-13)


def test_boolean_to_additive_multiplicity_cannot_preserve_idempotence() -> None:
    # Boolean OR identifies one and two identical active histories, whereas
    # ordinary additive semantics distinguishes their multiplicities.
    boolean_one = True
    boolean_two = True or True
    additive_one = 1
    additive_two = 1 + 1
    assert boolean_one == boolean_two
    assert additive_one != additive_two


def test_boolean_support_requires_unbounded_residual_for_unbounded_multiplicity_sample() -> None:
    # On the finite sample 1..N, every count has the same Boolean source value,
    # so the joint quotient needs N distinct classes.  As N grows there can be
    # no fixed finite residual alphabet recovering exact multiplicity.
    for bound in (2, 5, 20):
        source = [True] * bound
        target = list(range(1, bound + 1))
        audit = finite_factorization_audit(source, target)
        assert not audit.exact_on_sample
        assert audit.source_class_count == 1
        assert audit.joint_class_count == bound


def test_scalar_tropical_residual_packet_does_not_determine_argmax_multiplicity() -> None:
    tau = 1.0
    # Both have max 0, three terms, and partition sum 2.5.
    two_maxima = [0.0, 0.0, math.log(0.5)]
    one_maximum = [0.0, math.log(0.75), math.log(0.75)]
    first = TropicalResidualPacket.from_scores(two_maxima, temperature=tau)
    second = TropicalResidualPacket.from_scores(one_maximum, temperature=tau)

    assert first.tropical_value == second.tropical_value == 0.0
    assert first.term_count == second.term_count == 3
    assert first.free_energy == pytest.approx(second.free_energy, abs=1e-14)
    assert first.correction == pytest.approx(second.correction, abs=1e-14)
    assert sum(x == max(two_maxima) for x in two_maxima) == 2
    assert sum(x == max(one_maximum) for x in one_maximum) == 1
