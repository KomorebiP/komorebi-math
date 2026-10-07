from __future__ import annotations

import pytest

from komorebi_math.algorithms import (
    SupportGraph,
    inverse_support_if_acyclic,
    longest_path_vertices,
    schur_fill_support,
    support_polynomial,
    support_power,
    support_product,
    topological_levels,
    transitive_closure,
)
from komorebi_math.exceptions import ValidationError


def test_support_product_and_power() -> None:
    support = {(1, 0), (2, 1)}
    assert support_product(support, support) == frozenset({(2, 0)})
    assert support_power(support, 2) == frozenset({(2, 0)})


def test_transitive_closure_and_inverse_support() -> None:
    support = {(1, 0), (2, 1), (3, 2)}
    closure = transitive_closure(4, support)
    assert (3, 0) in closure
    assert inverse_support_if_acyclic(4, support) == closure


def test_polynomial_support() -> None:
    support = {(1, 0), (2, 1)}
    result = support_polynomial(support, [1, 0, 2], vertex_count=3)
    assert {(0, 0), (1, 1), (2, 2), (2, 0)} <= result


def test_schur_fill_support() -> None:
    retained = [0, 3]
    hidden = [1, 2]
    support = {
        (1, 0),
        (2, 1),
        (3, 2),
        (0, 0),
        (3, 3),
    }
    fill = schur_fill_support(retained, hidden, support)
    assert (3, 0) in fill
    assert (0, 0) in fill
    assert (3, 3) in fill


def test_topological_levels_and_longest_path() -> None:
    support = {(1, 0), (2, 0), (3, 1), (3, 2)}
    assert topological_levels(4, support) == ((0,), (1, 2), (3,))
    assert longest_path_vertices(4, support) == 3


def test_support_graph_reachability_and_cycle() -> None:
    graph = SupportGraph(4, {(1, 0), (2, 1), (3, 2)})
    assert graph.reachable([0]) == frozenset({0, 1, 2, 3})
    assert graph.is_acyclic
    cyclic = SupportGraph(2, {(1, 0), (0, 1)})
    assert not cyclic.is_acyclic
    with pytest.raises(ValidationError):
        topological_levels(2, cyclic.support)

def test_support_strict_integer_validation() -> None:
    with pytest.raises(ValidationError):
        SupportGraph(3.0, {(1, 0)})
    with pytest.raises(ValidationError):
        SupportGraph(3, {(1.0, 0)})
    with pytest.raises(ValidationError):
        support_power({(1, 0)}, 2.5)
    with pytest.raises(ValidationError):
        schur_fill_support([0.0], [1], {(1, 0)})

