from __future__ import annotations

import pytest

from komorebi_math import SupportPreorder, TransferGraph
from komorebi_math.exceptions import ValidationError


def test_reachability_and_paths() -> None:
    graph = TransferGraph(
        [
            [False, True, True, False],
            [False, False, False, True],
            [False, False, False, True],
            [False, False, False, False],
        ]
    )
    assert graph.reachable([0]) == frozenset({0, 1, 2, 3})
    assert set(graph.paths(0, 3, length=2)) == {(0, 1, 3), (0, 2, 3)}


def test_strong_components_and_sources() -> None:
    graph = TransferGraph(
        [
            [False, True, False, False],
            [True, False, True, False],
            [False, False, False, True],
            [False, False, True, False],
        ]
    )
    assert graph.strongly_connected_components() == ((0, 1), (2, 3))
    assert graph.source_components() == ((0, 1),)


def test_condensation_is_acyclic() -> None:
    graph = TransferGraph([[False, True, False], [True, False, True], [False, False, False]])
    condensed, components = graph.condensation()
    assert components == ((0, 1), (2,))
    assert condensed.is_acyclic()
    assert condensed.longest_path_vertices() == 2


def test_topological_order_rejects_cycle() -> None:
    graph = TransferGraph([[False, True], [True, False]])
    with pytest.raises(ValidationError):
        graph.topological_order()


def test_support_preorder_classes_and_nilpotency() -> None:
    preorder = SupportPreorder(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ]
    )
    assert preorder.equivalence_classes() == ((0, 1), (2,), (3,))
    assert preorder.strict_ideal_nilpotency_bound() == 3


def test_nontransitive_relation_rejected() -> None:
    with pytest.raises(ValidationError):
        SupportPreorder([[1, 1, 0], [0, 1, 1], [0, 0, 1]])


def test_source_component_certificate_is_conditional() -> None:
    graph = TransferGraph([[False, True], [False, False]])
    result = graph.source_component_certificate()
    assert result.value == 1
    assert result.certificate.assumptions_unchecked
