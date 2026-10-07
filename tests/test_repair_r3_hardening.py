"""Regression tests for the first post-manual implementation repair.

Negative cases are paired with finite positive controls. These are not proofs,
security certification, or a claim that arbitrary callbacks can be sandboxed.
"""
from __future__ import annotations

import copy
import json
from fractions import Fraction
from unittest.mock import Mock

import numpy as np
import pytest

from komorebi_math.certificates import Certificate, Exactness
from komorebi_math.core.packets import PacketSystem
from komorebi_math.coordinates.budget import CoordinateBudget
from komorebi_math.exceptions import ResourceLimitError, ValidationError
from komorebi_math.export.persistence import (
    to_portable_data, from_portable_data, dumps_portable, loads_portable,
)


@pytest.mark.parametrize('flag', ['false', 'true', '', 0, 1, 1.0, [], {}, None])
def test_certificate_rejects_non_boolean_flags(flag):
    with pytest.raises(ValidationError, match='boolean'):
        Certificate(flag, Exactness.EXACT)


@pytest.mark.parametrize('flag', [True, False, np.bool_(True), np.bool_(False)])
def test_certificate_real_boolean_controls(flag):
    cert = Certificate(flag, Exactness.EXACT)
    assert type(cert.valid) is bool
    assert cert.valid is bool(flag)
    if bool(flag):
        assert cert.require() is cert
    else:
        with pytest.raises(ValidationError):
            cert.require()


@pytest.mark.parametrize('flag', ['false', 1, np.bool_(True)])
def test_require_fails_closed_if_field_invariant_is_bypassed(flag):
    cert = Certificate(True, Exactness.EXACT)
    object.__setattr__(cert, 'valid', flag)
    with pytest.raises(ValidationError):
        cert.require()


def test_conditional_certificate_is_still_conditional():
    cert = Certificate(True, Exactness.CONDITIONAL, assumptions_unchecked=('external hypothesis',))
    assert cert.require() is cert
    assert cert.exactness is Exactness.CONDITIONAL
    assert cert.assumptions_unchecked


@pytest.mark.parametrize('flag', ['false', 0, 1, None, [], {}])
def test_portable_certificate_does_not_coerce_validity(flag):
    cert = Certificate(True, Exactness.EXACT)
    tree = to_portable_data(cert)
    fields = from_portable_data(tree['data'])
    fields['recorded_valid'] = flag
    tree['data'] = to_portable_data(fields)
    with pytest.raises(ValidationError, match='boolean'):
        from_portable_data(tree)


@pytest.mark.parametrize('backend', ['numpy', 'sympy'])
def test_hilbert_rechecks_packet_axioms(backend):
    invalid = PacketSystem([2 * np.eye(2)], backend=backend, validate=False)
    assert not invalid.certificate().valid
    assert not invalid.hilbert_certificate().valid
    with pytest.raises(ValidationError):
        invalid.hilbert_certificate().require()
    good = PacketSystem([np.diag([1, 0]), np.diag([0, 1])], backend=backend)
    assert good.hilbert_certificate().valid
    assert 'completeness' in good.hilbert_certificate().residuals


def test_hilbert_non_self_adjoint_valid_algebraic_packet_stays_negative():
    p = np.array([[1., 1.], [0., 0.]])
    system = PacketSystem([p, np.eye(2) - p])
    assert system.certificate().valid
    assert not system.hilbert_certificate().valid


@pytest.mark.parametrize('value', [True, 1.0, '1'])
def test_portable_schema_version_is_an_integer_not_equality_compatible(value):
    data = json.loads(dumps_portable(Fraction(1, 2)))
    data['schema_version'] = value
    with pytest.raises(ValidationError):
        loads_portable(json.dumps(data))


@pytest.mark.parametrize('value', [True, 1.0, '1'])
def test_portable_codec_version_is_an_integer(value):
    data = to_portable_data(Fraction(1, 2))
    data['__version__'] = value
    with pytest.raises(ValidationError):
        from_portable_data(data)


@pytest.mark.parametrize('field', ['numerator', 'denominator'])
@pytest.mark.parametrize('value', [True, 1.0, 1.5, '2', None])
def test_fraction_does_not_truncate_or_coerce(field, value):
    data = to_portable_data(Fraction(2, 3))
    data['data'][field] = value
    with pytest.raises(ValidationError):
        from_portable_data(data)


@pytest.mark.parametrize('shape', [[True], [1.0], [1.5], ['1'], [-1], [None]])
def test_array_dimensions_are_strict_nonnegative_integers(shape):
    data = to_portable_data(np.array([1]))
    data['data']['shape'] = shape
    with pytest.raises(ValidationError):
        from_portable_data(data)


def test_array_length_is_checked_before_nested_decoding():
    data = to_portable_data(np.array([1]))
    data['data']['shape'] = [2]
    data['data']['data'] = [{'__komorebi_type__': 'nonexistent.codec', '__version__': 1, 'data': None}]
    with pytest.raises(ValidationError, match='length'):
        from_portable_data(data)


def test_array_unrepresentable_zero_shape_is_rejected_without_allocation():
    data = to_portable_data(np.array([], dtype=np.float64))
    data['data']['shape'] = [0, int(np.iinfo(np.intp).max) + 1]
    with pytest.raises(ValidationError):
        from_portable_data(data)


def test_structured_array_writer_refuses_unsupported_roundtrip():
    array = np.array([(1, 2.)], dtype=[('a', 'i4'), ('b', 'f8')])
    with pytest.raises(ValidationError, match='structured|dtype'):
        to_portable_data(array)


@pytest.mark.parametrize('value', [np.array([1, 2]), np.array(2), np.empty((0, 3)),
    np.array([1+2j]), np.array([Fraction(1, 3)], dtype=object),
    np.array(['a', 'bc']), Fraction(-11, 7), {'key': (1, 2)}])
def test_supported_portable_roundtrip_controls(value):
    for result in (from_portable_data(to_portable_data(value)), loads_portable(dumps_portable(value))):
        if isinstance(value, np.ndarray):
            assert result.dtype == value.dtype and result.shape == value.shape
            assert np.array_equal(result, value)
        else:
            assert result == value


@pytest.mark.parametrize('version', [None, '', 123, False, {}])
def test_producer_version_requires_nonempty_string(version):
    data = json.loads(dumps_portable(1)); data['producer_version'] = version
    with pytest.raises(ValidationError):
        loads_portable(json.dumps(data))


def test_old_producer_version_is_not_falsely_rejected():
    data = json.loads(dumps_portable(1)); data['producer_version'] = '0.15.0a2'
    assert loads_portable(json.dumps(data)) == 1


def test_json_depth_is_rejected_before_recursive_parser():
    with pytest.raises(ValidationError, match='depth|nesting'):
        loads_portable('[' * 2000 + '0' + ']' * 2000, max_depth=8)


def test_json_depth_scanner_ignores_quoted_delimiters():
    text = '[{' * 1000 + '"\\' + ']}' * 1000
    assert loads_portable(dumps_portable(text), max_depth=4) == text


@pytest.mark.parametrize('row,column', [(True, 1), (1, False), (1.5, 2), (2, 1.0), ('1', 2)])
def test_coordinate_budget_rejects_non_integer_dimensions(row, column):
    with pytest.raises(ValidationError):
        CoordinateBudget().require_matrix(row, column, operation='test')


def test_coordinate_budget_zero_and_exact_boundary_controls():
    budget = CoordinateBudget(maximum_matrix_dimension=3, maximum_exact_entries=6)
    budget.require_matrix(0, 3, operation='test')
    budget.require_matrix(2, 3, operation='test')
    with pytest.raises(ResourceLimitError):
        budget.require_matrix(3, 3, operation='test')


def test_finite_series_stops_iterator_at_first_excess_before_coercion():
    from komorebi_math.generating.basis import ordinary_basis
    from komorebi_math.generating.domains import RationalDomain
    from komorebi_math.generating.budget import GeneratingBudget
    from komorebi_math.generating.finite_series import FiniteNormalizedSeries
    seen = []
    def values():
        for i in range(10):
            seen.append(i)
            yield i if i < 4 else object()
    with pytest.raises(ResourceLimitError):
        FiniteNormalizedSeries(ordinary_basis(), RationalDomain(), values(),
                               budget=GeneratingBudget(maximum_degree=3))
    assert seen == [0, 1, 2, 3, 4]


def test_finite_series_stops_at_coefficient_bit_limit():
    from komorebi_math.generating.basis import ordinary_basis
    from komorebi_math.generating.domains import RationalDomain
    from komorebi_math.generating.budget import GeneratingBudget
    from komorebi_math.generating.finite_series import FiniteNormalizedSeries
    seen = []
    def values():
        for i in [1, 2**30, 3, 4]:
            seen.append(i); yield i
    with pytest.raises(ResourceLimitError):
        FiniteNormalizedSeries(ordinary_basis(), RationalDomain(), values(),
                               budget=GeneratingBudget(maximum_coefficient_bits=8))
    assert seen == [1, 2**30]


def test_packet_plan_rejects_before_enumerating(monkeypatch):
    from komorebi_math.nonlinear.derivatives import PolynomialDerivativeProvider
    from komorebi_math.nonlinear.jets import PacketJetFamily
    from komorebi_math.nonlinear.execution import compile_packet_jet_plan
    p = PacketSystem([np.diag([1., 0.]), np.diag([0., 1.])])
    provider = PolynomialDerivativeProvider({2: np.zeros((2, 2, 2))}, source_dimension=2, target_dimension=2)
    family = PacketJetFamily(provider, np.zeros(2), p, p, 2)
    original = family.addresses
    spy = Mock(wraps=original); monkeypatch.setattr(family, 'addresses', spy)
    with pytest.raises(ValidationError, match='max_addresses'):
        compile_packet_jet_plan(family, max_addresses=7)
    spy.assert_not_called()
    assert compile_packet_jet_plan(family, max_addresses=8).full_address_count == 8


def test_expression_node_limit_before_visiting_rest_of_graph():
    from komorebi_math.expression.model import Expression, ExpressionBudget
    seen = []
    class Node(Expression):
        def __init__(self, child=None): self.child = child
        @property
        def children(self):
            seen.append(self)
            return () if self.child is None else (self.child,)
    root = Node(Node(Node(Node())))
    with pytest.raises(ResourceLimitError, match='max_nodes'):
        ExpressionBudget(max_nodes=1).preflight(root)
    assert len(seen) == 1


def test_expression_depth_limit_before_visiting_rest_of_graph():
    from komorebi_math.expression.model import Expression, ExpressionBudget
    seen = []
    class Node(Expression):
        def __init__(self, child=None): self.child = child
        @property
        def children(self): seen.append(self); return () if self.child is None else (self.child,)
    root = Node(Node(Node(Node())))
    with pytest.raises(ResourceLimitError, match='max_depth'):
        ExpressionBudget(max_depth=2).preflight(root)
    assert len(seen) <= 2


def test_leaf_tree_limit_rejects_before_generation_and_cache(monkeypatch):
    import komorebi_math.nonlinear.trees as trees
    original = trees._trees_with_leaves_cached
    spy = Mock(wraps=original)
    monkeypatch.setattr(trees, '_trees_with_leaves_cached', spy)
    with pytest.raises(ValidationError, match='max_trees'):
        trees.ordered_trees_with_leaves(6, (2,), max_trees=1)
    spy.assert_not_called()


@pytest.mark.parametrize('arities', [(2,), (3,), (2, 3), (3, 5), (9,)])
@pytest.mark.parametrize('leaves', range(1, 8))
def test_leaf_tree_counting_matches_existing_enumerator(arities, leaves):
    import komorebi_math.nonlinear.trees as trees
    expected = trees._trees_with_leaves_cached(leaves, arities)
    assert trees.ordered_trees_with_leaves(leaves, arities, max_trees=max(1, len(expected))) == expected
    if len(expected) > 1:
        with pytest.raises(ValidationError, match='max_trees'):
            trees.ordered_trees_with_leaves(leaves, arities, max_trees=len(expected) - 1)
