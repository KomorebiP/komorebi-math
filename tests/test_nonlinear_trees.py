from __future__ import annotations

import numpy as np
import pytest

from komorebi_math import Exactness, PacketSystem
from komorebi_math.exceptions import ValidationError
from komorebi_math.nonlinear.derivatives import SympyDerivativeProvider
from komorebi_math.nonlinear.jets import PacketJetFamily, packet_jet_support
from komorebi_math.nonlinear.trees import (
    OrderedRootedTree,
    PolynomialFixedPointProblem,
    catalan_number,
    majorant_convergence,
    ordered_rooted_trees,
    ordered_tree_count,
    ordered_trees_with_leaves,
    packet_tree_reconstruction,
    quadratic_catalan_majorant,
)


def test_ordered_tree_counts_and_catalan_families() -> None:
    assert tuple(ordered_tree_count(height, (2,)) for height in range(4)) == (1, 2, 5, 26)
    assert len(ordered_rooted_trees(3, (2,))) == 26
    for leaves in range(1, 8):
        trees = ordered_trees_with_leaves(leaves, (2,))
        assert len(trees) == catalan_number(leaves - 1)
        assert all(tree.leaf_count == leaves for tree in trees)
        assert all(set(tree.arity_profile()) <= {2} for tree in trees)
    with pytest.raises(ValidationError, match="above max_trees"):
        ordered_rooted_trees(4, (2,), max_trees=100)


def test_tree_metadata_and_prefix() -> None:
    leaf = OrderedRootedTree.leaf()
    tree = OrderedRootedTree((leaf, OrderedRootedTree((leaf, leaf))))
    assert tree.height == 2
    assert tree.leaf_count == 3
    assert tree.internal_count == 2
    assert tree.vertex_count == 5
    assert tree.prefix() == "B2(s,B2(s,s))"
    assert tree.arity_profile() == {2: 2}


def test_finite_tree_sum_equals_picard_iterate() -> None:
    problem = PolynomialFixedPointProblem(
        np.array([0.1]),
        {2: lambda values: np.array([values[0][0] * values[1][0]])},
    )
    for depth in range(5):
        result = problem.tree_expansion(depth, max_trees=100_000)
        result.certificate.require()
        assert np.allclose(result.value, result.picard_value)
        expected_count = 0 if depth == 0 else ordered_tree_count(depth - 1, (2,))
        assert len(result.terms) == expected_count


def test_polynomial_tree_with_mixed_arities() -> None:
    problem = PolynomialFixedPointProblem(
        np.array([0.05]),
        {
            2: lambda values: 0.5 * values[0] * values[1],
            3: lambda values: 0.25 * values[0] * values[1] * values[2],
        },
    )
    result = problem.tree_expansion(3)
    result.certificate.require()
    assert np.allclose(result.value, problem.picard(3))
    assert set(result.terms[-1].tree.arity_profile()) <= {2, 3}


def test_majorant_and_catalan_convergence_certificate() -> None:
    result = majorant_convergence(0.1, {2: 0.5, 3: 0.1}, 0.2)
    result.certificate.require()
    assert result.image_bound <= result.radius
    assert result.contraction < 1
    assert result.scalar_iterates(5)[-1] <= result.radius
    assert result.picard_tail_bound(8) < result.picard_tail_bound(2)

    catalan = quadratic_catalan_majorant(0.1, 0.5)
    catalan.certificate.require()
    assert abs(catalan.radius - (1 - np.sqrt(0.8))) < 1e-14
    failed = quadratic_catalan_majorant(1.0, 1.0)
    assert not failed.certificate.valid


def test_exact_packet_tree_reconstruction_and_pruning() -> None:
    sp = pytest.importorskip("sympy")
    x, y = sp.symbols("x y")
    packets = PacketSystem(
        [sp.diag(1, 0), sp.diag(0, 1)],
        names=("x", "y"),
        backend="sympy",
    )
    problem = PolynomialFixedPointProblem(
        sp.Matrix([1, 2]),
        {2: lambda values: sp.Matrix([values[0][0] * values[1][0], values[0][1] * values[1][1]])},
        exactness=Exactness.EXACT,
    )
    tree = OrderedRootedTree((OrderedRootedTree.leaf(), OrderedRootedTree((OrderedRootedTree.leaf(), OrderedRootedTree.leaf()))))
    result = packet_tree_reconstruction(problem, tree, packets)
    result.certificate.require()
    assert result.value == problem.tree_value(tree)

    provider = SympyDerivativeProvider((x**2 / 2, y**2 / 2), (x, y))
    support = packet_jet_support(PacketJetFamily(provider, sp.Matrix([0, 0]), packets, packets, 2))
    pruned = packet_tree_reconstruction(problem, tree, packets, zero_support={2: support})
    pruned.certificate.require()
    assert pruned.value == result.value
    assert pruned.pruned_blocks > 0
    assert pruned.evaluated_blocks < result.evaluated_blocks
