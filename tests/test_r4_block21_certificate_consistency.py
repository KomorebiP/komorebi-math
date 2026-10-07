from __future__ import annotations

from fractions import Fraction
from math import nextafter

import numpy as np
import pytest

from komorebi_math.algorithms.compiler import TreeNode, compile_tree_budget, compile_tree_lagrangian
from komorebi_math.algorithms.precision import DTypeChoice, assign_discrete_precisions
from komorebi_math.certificates import Exactness
from komorebi_math.exceptions import ValidationError
from komorebi_math.langlands.finite_image.groups import FiniteGroup
from komorebi_math.signals.wavelets import best_basis, haar_packet_analyze


def _lower_half() -> float:
    return nextafter(0.5, 0.0)


def _exact_split_cost() -> Fraction:
    return Fraction.from_float(0.5) + Fraction.from_float(_lower_half())


def test_best_basis_exact_certificate_uses_exact_dyadic_comparison() -> None:
    lower = _lower_half()
    assert _exact_split_cost() < 1
    assert 0.5 + lower == 1.0
    tree = haar_packet_analyze(np.array([1.0, 0.0]), 1)
    result = best_basis(tree, {"": 1.0, "0": 0.5, "1": lower})
    result.certificate.require()
    assert result.certificate.exactness is Exactness.EXACT
    assert result.leaves == ("0", "1")
    assert result.certificate.diagnostics["exact_cost_numerator"] == _exact_split_cost().numerator
    assert result.certificate.diagnostics["exact_cost_denominator"] == _exact_split_cost().denominator


def test_tree_compilers_do_not_turn_rounded_ties_into_false_exact_optima() -> None:
    lower = _lower_half()
    root = TreeNode(
        "a",
        1.0,
        0.0,
        0.0,
        (TreeNode("b", 0.5, 0.0), TreeNode("c", lower, 0.0)),
    )
    lagrangian = compile_tree_lagrangian(root, 0.0)
    lagrangian.certificate.require()
    assert lagrangian.certificate.exactness is Exactness.EXACT
    assert lagrangian.leaves == ("b", "c")
    assert lagrangian.certificate.diagnostics["exact_cost"] == str(_exact_split_cost())

    budget = compile_tree_budget(root, 0, error_unit=1.0)
    assert budget is not None
    budget.certificate.require()
    assert budget.certificate.exactness is Exactness.EXACT
    assert budget.leaves == ("b", "c")
    assert budget.certificate.diagnostics["exact_cost"] == str(_exact_split_cost())


def test_budget_quantization_is_exact_for_the_supplied_binary_floats() -> None:
    unit = 0.1
    root = TreeNode(
        "root",
        2.0,
        unit,
        0.0,
        (TreeNode("left", 0.5, unit), TreeNode("right", 0.5, 0.0)),
    )
    one_unit = compile_tree_budget(root, 1, error_unit=unit)
    assert one_unit is not None
    one_unit.certificate.require()
    assert one_unit.quantized_error == 1


def test_discrete_precision_assignment_uses_exact_dyadic_cost_comparison() -> None:
    lower = _lower_half()
    options = [
        [DTypeChoice("a", 1.0, 0), DTypeChoice("b", 0.5, 1)],
        [DTypeChoice("a", 0.0, 1), DTypeChoice("b", lower, 0)],
    ]
    result = assign_discrete_precisions(options, 1)
    assert result is not None
    result.certificate.require()
    assert result.certificate.exactness is Exactness.EXACT
    assert result.names == ("b", "b")
    assert result.certificate.diagnostics["exact_total_cost"] == str(_exact_split_cost())


def test_finite_group_certificate_revalidates_trusted_construction_path() -> None:
    invalid = FiniteGroup([0, 1], lambda left, right: 0, 0, validate=False, name="not-a-group")
    certificate = invalid.certificate()
    assert certificate.exactness is Exactness.EXACT
    assert not certificate.valid
    assert certificate.diagnostics["identity_failures"] > 0
    assert certificate.diagnostics["inverse_failures"] > 0
    with pytest.raises(ValidationError):
        certificate.require()


def test_finite_group_certificate_still_accepts_a_valid_group() -> None:
    group = FiniteGroup(range(5), lambda left, right: (left + right) % 5, 0, validate=False, name="C5")
    certificate = group.certificate()
    certificate.require()
    assert certificate.exactness is Exactness.EXACT
    assert certificate.diagnostics["associativity_triples"] == 125
