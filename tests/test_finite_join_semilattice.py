"""Independent finite-partition controls for generated join congruences."""
from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from komorebi_math.certificates import Exactness
from komorebi_math.core.semilattice import FiniteJoinSemilattice, SemilatticeQuotient
from komorebi_math.exceptions import DimensionError, ValidationError


def partitions(size: int):
    """Enumerate restricted-growth strings, independently of union-find closure."""
    def visit(prefix):
        if len(prefix) == size:
            yield tuple(prefix)
            return
        for label in range(max(prefix, default=-1) + 2):
            yield from visit(prefix + [label])
    yield from visit([])


def brute_least_partition(table, relations):
    size = len(table)
    congruences = [labels for labels in partitions(size)
                   if all(labels[left] == labels[right] for left, right in relations)
                   and all(labels[table[left][context]] == labels[table[right][context]]
                           for left in range(size) for right in range(size) for context in range(size)
                           if labels[left] == labels[right])]
    return tuple(min(right for right in range(size)
                     if all(labels[left] == labels[right] for labels in congruences))
                 for left in range(size))


@pytest.mark.parametrize("table", [
    tuple(tuple(max(left, right) for right in range(4)) for left in range(4)),
    tuple(tuple(left | right for right in range(4)) for left in range(4)),
])
def test_every_pair_and_two_relation_closure_matches_independent_all_partitions(table) -> None:
    semilattice = FiniteJoinSemilattice(table)
    pairs = tuple(product(range(4), repeat=2))
    for first in pairs:
        for second in pairs:
            relations = (first, second)
            result = semilattice.generated_congruence(relations)
            assert result.partition == brute_least_partition(table, relations)
            result.certificate().require()
            assert result.certificate().exactness is Exactness.EXACT
            assert result.partition == semilattice.generated_congruence(tuple(reversed(relations))).partition


def test_boolean_context_closure_and_quotient_homomorphism() -> None:
    semilattice = FiniteJoinSemilattice(tuple(tuple(left | right for right in range(8)) for left in range(8)))
    result = semilattice.generated_congruence(((0, 1),))
    assert result.classes == ((0, 1), (2, 3), (4, 5), (6, 7))
    projection, quotient = result.projection, result.quotient
    for left, right in product(range(8), repeat=2):
        assert projection[semilattice.join(left, right)] == quotient.join(projection[left], projection[right])
    assert len(result.derivations) == 4
    assert "finite least-congruence derivation" in result.certificate().assumptions_checked


def test_zero_relations_are_identity_and_full_relations_are_terminal() -> None:
    semilattice = FiniteJoinSemilattice(((0, 1), (1, 1)))
    assert semilattice.generated_congruence(()).partition == (0, 1)
    result = semilattice.generated_congruence(((0, 1),))
    assert result.quotient.join_table == ((0,),)
    assert result.projection == (0, 0)


def test_singleton_policy_and_mutable_input_snapshot() -> None:
    assert FiniteJoinSemilattice(((0,),)).generated_congruence(()).partition == (0,)
    table = [[0, 1], [1, 1]]
    semilattice = FiniteJoinSemilattice(table)
    table[0][1] = 0
    assert semilattice.join_table == ((0, 1), (1, 1))
    partition = [0, 0]
    result = SemilatticeQuotient(semilattice, partition)
    partition[1] = 1
    assert result.partition == (0, 0)


@pytest.mark.parametrize("table", [(), ((0, 1),), ((0, 1), (1,)), (1, 2)])
def test_join_table_requires_nonempty_square_data(table) -> None:
    with pytest.raises(DimensionError):
        FiniteJoinSemilattice(table)


@pytest.mark.parametrize("table", [((True,),), ((0.,),), ((np.bool_(False),),), ((1,),), ((-1,),)])
def test_join_table_has_strict_integer_closed_entries(table) -> None:
    with pytest.raises(ValidationError):
        FiniteJoinSemilattice(table)


@pytest.mark.parametrize("table,match", [
    (((1, 1), (1, 1)), "idempotent"),
    (((0, 0), (1, 1)), "commutative"),
    (((0, 2, 1), (2, 1, 0), (1, 0, 2)), "associative"),
])
def test_all_join_axioms_are_checked(table, match) -> None:
    with pytest.raises(ValidationError, match=match):
        FiniteJoinSemilattice(table)


@pytest.mark.parametrize("relations,error", [
    (((0,),), DimensionError), (((0, 1, 1),), DimensionError),
    ((0,), DimensionError), (((0, True),), ValidationError),
    (((0, 1.),), ValidationError), (((0, 2),), ValidationError),
])
def test_congruence_relations_require_valid_strict_integer_pairs(relations, error) -> None:
    with pytest.raises(error):
        FiniteJoinSemilattice(((0, 1), (1, 1))).generated_congruence(relations)


def test_supplied_noncongruence_partition_is_rejected() -> None:
    semilattice = FiniteJoinSemilattice(tuple(tuple(left | right for right in range(4)) for left in range(4)))
    with pytest.raises(ValidationError):
        semilattice.quotient((0, 0, 2, 3))
    result = semilattice.quotient((17, 17, 23, 23))
    assert result.partition == (0, 0, 2, 2)
    assert "finite least-congruence derivation" not in result.certificate().assumptions_checked


def test_forged_leastness_derivation_is_invalid() -> None:
    source = FiniteJoinSemilattice(((0, 1), (1, 1)))
    forged = SemilatticeQuotient(source, (0, 0), (), ((0, 1, -1, 0),))
    assert not forged.certificate().valid
    with pytest.raises(ValidationError):
        _ = forged.quotient


def test_numpy_integer_labels_are_exactly_normalized() -> None:
    source = FiniteJoinSemilattice(np.array([[0, 1], [1, 1]], dtype=np.int64))
    assert source.generated_congruence(((np.int64(0), np.int64(1)),)).partition == (0, 0)
