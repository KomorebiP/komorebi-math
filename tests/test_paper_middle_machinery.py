"""Independent finite checks for machinery from Papers V--VIII."""

from dataclasses import replace
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from komorebi_math.algorithms.compiler import (
    TreeNode,
    TreePlan,
    compile_tree_budget,
    compile_tree_lagrangian,
)
from komorebi_math.algorithms.compiler_audit import (
    audit_tree_budget,
    audit_tree_lagrangian,
    enumerate_tree_prunings,
)
from komorebi_math.certificates import Exactness
from komorebi_math.coordinates.classical import hermite_probabilists_polynomial
from komorebi_math.coordinates.polynomials import polynomial_derivative
from komorebi_math.dynamics.hermite import (
    gaussian_moment,
    gaussian_polynomial_inner_product,
    gaussian_polynomial_variance,
    gaussian_transfer_polynomial,
    ornstein_uhlenbeck_generator,
)
from komorebi_math.exceptions import ValidationError


def _unsupported_frontier_tree():
    # Paper VIII's introductory three-pruning example has an unsupported
    # budget optimum: (cost,error) = (1,6), (3,4), (5,1).
    branch = TreeNode("a", 2, 4, 0, (TreeNode("a1", 3, 1), TreeNode("a2", 1, 0)))
    return TreeNode("r", 1, 6, 0, (branch, TreeNode("b", 1, 0)))


def test_paper_viii_unsupported_budget_optimum():
    root = _unsupported_frontier_tree()
    space = enumerate_tree_prunings(root)
    assert tuple((item.cost, item.error) for item in space) == ((1, 6), (3, 4), (5, 1))
    plan = compile_tree_budget(root, 4)
    assert plan.leaves == ("a", "b")
    audit = audit_tree_budget(root, plan, 4)
    assert audit.certificate.valid
    assert audit.certificate.exactness is Exactness.EXACT
    assert audit.prunings_examined == 3 and audit.feasible_prunings == 2
    # No nonnegative multiplier selects the middle point: below 4/5 the root
    # wins, above it the full split wins.  This finite sweep is only a control.
    for penalty in (0, 0.5, 4 / 5, 1, 2, 4, 100):
        result = compile_tree_lagrangian(root, penalty)
        assert result.leaves != ("a", "b")
        assert audit_tree_lagrangian(root, result, penalty).certificate.valid


def test_paper_viii_quantization_has_no_integer_boundary_tolerance():
    error = np.nextafter(1.0, np.inf)
    root = TreeNode("root", 0, error)
    assert enumerate_tree_prunings(root)[0].error_units(Fraction(1)) == 2
    assert compile_tree_budget(root, 1) is None
    assert audit_tree_budget(root, None, 1).certificate.valid
    forged = TreePlan(0, error, ("root",), (("root", "keep"),), quantized_error=1)
    assert not audit_tree_budget(root, forged, 1).certificate.valid
    assert audit_tree_budget(root, compile_tree_budget(root, 2), 2).certificate.valid


def test_paper_viii_leafwise_quantization_differs_from_rounding_total():
    root = TreeNode("r", 100, 100, 0, (TreeNode("a", 1, 0.4), TreeNode("b", 1, 0.4)))
    assert compile_tree_budget(root, 1) is None
    assert audit_tree_budget(root, None, 1).certificate.valid
    plan = compile_tree_budget(root, 2)
    assert plan.quantized_error == 2 and plan.error == 0.8
    assert audit_tree_budget(root, plan, 2).certificate.valid
    assert not audit_tree_budget(root, replace(plan, quantized_error=1), 1).certificate.valid


def test_paper_viii_exact_dyadic_comparison_survives_float_objective_tie():
    # Float arithmetic rounds 2**53+1 down to 2**53.  Exact dyadic totals
    # distinguish root cost 2**53 from split cost 2**53+1.
    root = TreeNode("r", 2**53, 0, 1, (TreeNode("a", 2**53, 0),))
    plan = compile_tree_lagrangian(root, 0)
    assert plan.leaves == ("r",)
    audit = audit_tree_lagrangian(root, plan, 0)
    assert audit.certificate.valid
    forged = TreePlan(float(2**53), 0, ("a",), (("r", "split"), ("a", "keep")), objective=float(2**53))
    assert not audit_tree_lagrangian(root, forged, 0).certificate.valid


def test_paper_viii_forged_totals_decisions_and_certificate_are_rejected():
    root = _unsupported_frontier_tree()
    plan = compile_tree_lagrangian(root, 1)
    for forged in (
        replace(plan, cost=plan.cost + 1),
        replace(plan, error=plan.error + 1),
        replace(plan, objective=plan.objective + 1),
        replace(plan, leaves=("nonexistent",)),
        replace(plan, leaves=np.array(plan.leaves)),
        replace(plan, decisions=np.array(plan.decisions)),
        replace(plan, decisions=tuple(list(decision) for decision in plan.decisions)),
        replace(plan, decisions=(("root", "split"),)),
    ):
        assert not audit_tree_lagrangian(root, forged, 1).certificate.valid
    nonoptimal = compile_tree_lagrangian(root, 0)
    assert nonoptimal.certificate.valid
    assert not audit_tree_lagrangian(root, nonoptimal, 1).certificate.valid
    assert not audit_tree_budget(root, None, 4).certificate.valid


def test_paper_viii_accepts_alternative_optimal_pruning():
    root = TreeNode("r", 2, 0, 0, (TreeNode("a", 2, 0),))
    alternative = TreePlan(2, 0, ("a",), (("r", "split"), ("a", "keep")), objective=2, quantized_error=0)
    assert audit_tree_lagrangian(root, alternative, 1).certificate.valid
    assert audit_tree_budget(root, alternative, 0).certificate.valid
    for malformed in (
        replace(alternative, leaves=np.array(alternative.leaves)),
        replace(alternative, decisions=np.array(alternative.decisions)),
    ):
        assert not audit_tree_lagrangian(root, malformed, 1).certificate.valid
        assert not audit_tree_budget(root, malformed, 0).certificate.valid
    assert not audit_tree_budget(root, replace(alternative, quantized_error=False), 0).certificate.valid
    assert not audit_tree_budget(root, replace(alternative, quantized_error=0.0), 0).certificate.valid
    zero = TreeNode("zero", 0, 0)
    plan = compile_tree_lagrangian(zero, 0)
    assert not audit_tree_lagrangian(zero, replace(plan, cost=False), 0).certificate.valid
    assert not audit_tree_lagrangian(zero, replace(plan, objective=np.inf), 0).certificate.valid


def test_paper_viii_bounded_oracle_rejects_partial_enumeration():
    root = _unsupported_frontier_tree()
    with pytest.raises(ValidationError, match="max_prunings"):
        enumerate_tree_prunings(root, max_prunings=2)
    with pytest.raises(ValidationError, match="max_nodes"):
        enumerate_tree_prunings(root, max_nodes=2)
    for bad in (True, 0, 1.2):
        with pytest.raises(ValidationError):
            enumerate_tree_prunings(root, max_prunings=bad)
    for bad in (0, -1, True, np.bool_(True), np.inf, np.nan, "1.0"):
        with pytest.raises(ValidationError):
            audit_tree_budget(root, None, 1, error_unit=bad)
    for bad in (True, np.bool_(True), "1.0"):
        with pytest.raises(ValidationError):
            audit_tree_lagrangian(root, None, bad)


def test_paper_viii_generated_trees_match_complete_exact_oracle():
    rng = np.random.default_rng(20261010)
    for index in range(60):

        def subtree(depth, name):
            values = tuple(float(rng.integers(0, 12)) / 8 for _ in range(3))
            children = () if depth == 0 else tuple(subtree(depth - 1, name + str(j)) for j in range(2))
            return TreeNode(name, *values, children)

        root = subtree(2, f"r{index}")
        assert len(enumerate_tree_prunings(root)) == 5
        for penalty in (0, 0.25, 1, 4):
            plan = compile_tree_lagrangian(root, penalty)
            assert audit_tree_lagrangian(root, plan, penalty).certificate.valid
        for budget, unit in product((0, 1, 3, 6, 12), (0.125, 0.25, 1)):
            plan = compile_tree_budget(root, budget, error_unit=unit)
            assert audit_tree_budget(root, plan, budget, error_unit=unit).certificate.valid


def test_paper_vi_gaussian_moments_match_independent_product():
    for order in range(80):
        expected = 0 if order % 2 else 1
        if order % 2 == 0:
            for odd in range(1, order, 2):
                expected *= odd
        assert gaussian_moment(order) == expected
    assert gaussian_transfer_polynomial((0, 0, 1), Fraction(1, 2)) == (Fraction(3, 4), 0, Fraction(1, 4))
    assert gaussian_transfer_polynomial((0, 0, 0, 0, 1), 0) == (3,)


def test_paper_vi_hermite_orthogonality_generator_and_transfer():
    from math import factorial

    for n in range(19):
        polynomial = hermite_probabilists_polynomial(n)
        for m in range(19):
            other = hermite_probabilists_polynomial(m)
            expected = factorial(n) if n == m else 0
            assert gaussian_polynomial_inner_product(polynomial, other) == expected
        assert ornstein_uhlenbeck_generator(polynomial) == (
            tuple(-n * coefficient for coefficient in polynomial) if n else (0,)
        )
        for rho in (Fraction(0), Fraction(1, 5), Fraction(1, 2), Fraction(1)):
            transferred = gaussian_transfer_polynomial(polynomial, rho)
            expected = tuple(rho**n * coefficient for coefficient in polynomial)
            while len(expected) > 1 and expected[-1] == 0:
                expected = expected[:-1]
            assert transferred == expected


def test_paper_vi_exact_gaussian_semigroup_dirichlet_and_spectral_gap():
    rng = np.random.default_rng(61010)
    first, second = Fraction(2, 3), Fraction(3, 7)
    for _ in range(40):
        polynomial = tuple(Fraction(int(rng.integers(-5, 6)), 3) for _ in range(9))
        left = gaussian_transfer_polynomial(gaussian_transfer_polynomial(polynomial, first), second)
        assert left == gaussian_transfer_polynomial(polynomial, first * second)
        mean = gaussian_polynomial_inner_product(polynomial, (1,))
        assert gaussian_polynomial_inner_product(left, (1,)) == mean
        derivative = polynomial_derivative(polynomial)
        dirichlet = gaussian_polynomial_inner_product(derivative, derivative)
        generator = ornstein_uhlenbeck_generator(polynomial)
        assert -gaussian_polynomial_inner_product(generator, polynomial) == dirichlet
        variance = gaussian_polynomial_variance(polynomial)
        assert variance <= dirichlet
        transferred_variance = gaussian_polynomial_variance(gaussian_transfer_polynomial(polynomial, first))
        assert transferred_variance <= first**2 * variance


def test_paper_vi_gaussian_rejects_inexact_inputs_and_degree_truncation():
    for bad in (True, 0.5, complex(1), "1/2"):
        with pytest.raises(ValidationError):
            gaussian_transfer_polynomial((1, 2), bad)
        with pytest.raises(ValidationError):
            gaussian_transfer_polynomial((1, bad), Fraction(1, 2))
    for bad in (Fraction(-1, 10), Fraction(11, 10)):
        with pytest.raises(ValidationError):
            gaussian_transfer_polynomial((1, 2), bad)
    with pytest.raises(ValidationError, match="max_degree"):
        gaussian_transfer_polynomial((1, 2, 3), Fraction(1, 2), max_degree=1)
    with pytest.raises(ValidationError, match="max_order"):
        gaussian_moment(4, max_order=3)

    def excessive_coefficients():
        yield from (1, 2, 3)
        raise AssertionError("degree guard consumed beyond the first excess coefficient")

    with pytest.raises(ValidationError, match="max_degree"):
        gaussian_transfer_polynomial(excessive_coefficients(), Fraction(1, 2), max_degree=1)
    assert ornstein_uhlenbeck_generator(()) == (0,)
    assert gaussian_polynomial_variance(()) == 0
