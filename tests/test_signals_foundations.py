"""Orthogonal identities and rejection controls for finite LSM foundations."""

from fractions import Fraction as F
from itertools import product

import numpy as np
import pytest

from komorebi_math.certificates import Exactness
from komorebi_math.exceptions import DimensionError, ResourceLimitError, ValidationError
from komorebi_math.signals.foundations import (
    compression_closure,
    delay_closure,
    identify_data_span,
    minimal_predictive_completion,
    projected_memory,
)

sp = pytest.importorskip("sympy")
L = sp.Matrix([[F(1, 2), 1, 0], [F(1, 4), F(1, 3), 0], [0, 0, F(5, 6)]])
P = sp.diag(1, 0, 0)


def test_return_path_is_a_defect_even_with_valid_computation_certificate():
    result = compression_closure(L, P)
    assert not result.closed
    assert result.closure_defect == sp.Matrix([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    assert result.composition_defect == sp.diag(F(1, 4), 0, 0)
    assert result.certificate.valid and result.certificate.exactness is Exactness.EXACT
    result.certificate.diagnostics["closed"] = True
    with pytest.raises(ValidationError, match="not closed"):
        result.require_closed()
    assert compression_closure(L, sp.eye(3)).require_closed().closed


def test_compositional_defect_has_outer_then_inner_order():
    outer = [[0, 2], [0, 0]]
    inner = [[0, 0], [3, 0]]
    result = compression_closure(outer, sp.diag(1, 0), inner_operator=inner)
    assert result.composition_defect == sp.diag(6, 0)
    assert result.certificate.valid


def test_minimal_completion_ignores_an_independent_hidden_component():
    result = minimal_predictive_completion([[1, 0, 0]], {"L": L})
    assert result.require_complete().basis == sp.Matrix([[1, 0, 0], [0, 1, 0]])
    assert result.dimension_chain == (1, 2)
    assert result.original_decoder == sp.Matrix([[1, 0]])
    assert result.reduced_operators["L"] == L[:2, :2]
    assert result.rounds <= result.round_limit
    # Independent annihilator characterization: ker R is L invariant.
    for vector in result.basis.nullspace():
        assert result.basis * L * vector == sp.zeros(2, 1)


def test_multiple_actions_need_their_common_observable_completion():
    a = sp.Matrix([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    b = sp.Matrix([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    result = minimal_predictive_completion([[1, 0, 0]], {"a": a, "b": b})
    assert result.require_complete().basis == sp.eye(3)
    assert result.dimension_chain == (1, 2, 3)
    partial = minimal_predictive_completion([[1, 0, 0]], {"a": a, "b": b}, max_rounds=1)
    assert not partial.complete and not partial.certificate.valid
    assert partial.dimension_chain == (1, 2)
    partial.certificate.diagnostics["complete"] = True
    with pytest.raises(ValidationError):
        partial.require_complete()


def test_zero_summary_is_the_zero_closed_row_space():
    result = minimal_predictive_completion([[0, 0]], {"identity": sp.eye(2)})
    assert result.require_complete().basis.shape == (0, 2)
    assert result.reduced_operators["identity"].shape == (0, 0)
    assert result.original_decoder.shape == (1, 0)


@pytest.mark.parametrize("entries", list(product((-1, 0, 1), repeat=4)))
def test_all_81_small_integer_transitions_against_independent_krylov_rank(entries):
    matrix = sp.Matrix(2, 2, entries)
    summary = sp.Matrix([[1, 0]])
    result = minimal_predictive_completion(summary, {"L": matrix})
    expected_rank = summary.col_join(summary * matrix).rank()
    assert result.require_complete().basis.rows == expected_rank
    assert result.basis * matrix == result.reduced_operators["L"] * result.basis
    assert result.certificate.exactness is Exactness.EXACT


@pytest.mark.parametrize(
    "projection",
    [
        sp.diag(1, 0),
        sp.diag(0, 1),
        sp.eye(2),
        sp.zeros(2),
        sp.Matrix([[F(1, 2), F(1, 2)], [F(1, 2), F(1, 2)]]),
    ],
)
def test_projection_memory_against_direct_full_state_iteration(projection):
    matrix = sp.Matrix([[F(1, 2), 2], [F(-1, 4), F(1, 3)]])
    initial = sp.Matrix([2, -3])
    result = projected_memory(matrix, projection, horizon=8)
    history = []
    state = initial
    q0 = (sp.eye(2) - projection) * initial
    for _time in range(9):
        history.append(projection * state)
        assert result.next_state(history, q0) == projection * matrix * state
        state = matrix * state
    assert result.certificate.valid


def test_memory_example_and_subspace_horizon_guards():
    memory = projected_memory(L, P, horizon=2)
    assert memory.kernels[0] == sp.diag(F(1, 4), 0, 0)
    assert memory.kernels[1] == sp.diag(F(1, 12), 0, 0)
    assert memory.next_state([[1, 0, 0], [F(1, 2), 0, 0], [F(1, 2), 0, 0]], [0, 0, 0]) == sp.Matrix(
        [F(11, 24), 0, 0]
    )
    with pytest.raises(ResourceLimitError):
        memory.next_state([[0, 0, 0]] * 4, [0, 0, 0])
    with pytest.raises(ValidationError, match="retained subspace"):
        memory.next_state([[1, 1, 0]], [0, 0, 0])
    with pytest.raises(ValidationError, match="unresolved subspace"):
        memory.next_state([[1, 0, 0]], [1, 0, 0])


def test_delay_closure_uses_known_operator_not_finite_sample_plateau():
    short = delay_closure(L, [[1, 0, 0]], delay=1)
    assert not short.closed and short.rank == 1 and short.extended_rank == 2
    assert not short.certificate.valid
    with pytest.raises(ValidationError):
        short.require_transition()
    exact = delay_closure(L, [[1, 0, 0]], delay=2)
    assert exact.require_transition() == sp.Matrix([[0, 1], [F(1, 12), F(5, 6)]])
    assert exact.rank == exact.extended_rank == 2
    assert exact.transition * exact.observation_map == exact.observation_map * L
    # Redundant outputs give a nonunique ambient extension, unique on im O_h.
    redundant = delay_closure(L, [[1, 0, 0], [2, 0, 0]], delay=3)
    assert redundant.closed and redundant.transition.shape == (6, 6)
    assert redundant.transition * redundant.observation_map == redundant.successor_observation_map


def test_zero_observation_closes_without_recovering_hidden_state():
    result = delay_closure(L, [[0, 0, 0]], delay=1)
    assert result.require_transition() == sp.zeros(1)
    assert result.rank == 0


def test_data_identification_recovers_worked_delay_operator():
    result = identify_data_span(
        [[1, F(1, 2)], [F(1, 2), F(1, 2)]], [[F(1, 2), F(1, 2)], [F(1, 2), F(11, 24)]]
    )
    assert result.require_closed_span() == sp.Matrix([[0, 1], [F(1, 12), F(5, 6)]])
    assert result.ambient_span_complete
    assert result.global_coverage == "not_established"
    assert "sampled states span the true realizable state space" in result.certificate.assumptions_unchecked


def test_fit_consistency_span_closure_and_coverage_are_separate():
    outside = identify_data_span([[1], [0]], [[0], [1]])
    assert outside.fit_consistent and not outside.span_closed
    assert not outside.ambient_span_complete and outside.global_coverage == "not_established"
    outside.require_fit()
    with pytest.raises(ValidationError, match="leave"):
        outside.require_closed_span()
    impossible = identify_data_span([[1, 1]], [[1, 2]])
    assert not impossible.fit_consistent and not impossible.certificate.valid
    assert impossible.dependency_obstruction != sp.zeros(1)
    with pytest.raises(ValidationError):
        impossible.require_fit()
    zero = identify_data_span([[0, 0], [0, 0]], [[0, 0], [0, 0]])
    assert zero.require_closed_span() == sp.zeros(2)
    assert zero.data_rank == 0 and zero.global_coverage == "not_established"


def test_retained_input_and_output_evidence_is_owned_and_immutable():
    inputs = [[1, 0], [0, 1]]
    successors = sp.Matrix([[1, 2], [0, 1]])
    result = identify_data_span(inputs, successors)
    inputs[0][0] = 99
    successors[0, 0] = 99
    assert result.input_states == sp.eye(2)
    assert result.successor_states[0, 0] == 1
    with pytest.raises(TypeError):
        result.fitted_operator[0, 0] = 99
    closure = minimal_predictive_completion([[1, 0]], {"L": [[1, 2], [0, 1]]})
    with pytest.raises(TypeError):
        closure.reduced_operators["L"] = sp.zeros(2)


@pytest.mark.parametrize(
    "bad", [0.0, float("nan"), float("inf"), True, "1/2", sp.Symbol("x"), sp.sqrt(2), 1j]
)
def test_inexact_symbolic_or_parsed_inputs_are_not_promoted_to_rational_proofs(bad):
    with pytest.raises(ValidationError):
        compression_closure([[bad]], [[1]])


def test_numpy_integer_inputs_are_exact_and_tiny_nonzero_is_not_float_zero():
    assert compression_closure(np.eye(2, dtype=int), np.eye(2, dtype=int)).closed
    tiny = F(1, 10**400)
    result = compression_closure([[0, tiny], [0, 0]], [[1, 0], [0, 0]])
    assert not result.closed and result.closure_defect[0, 1] != 0
    with pytest.raises(ValidationError):
        result.require_closed()


@pytest.mark.parametrize(
    "call",
    [
        lambda: compression_closure([[1, 2]], [[1]]),
        lambda: compression_closure([[1], 2], [[1]]),
        lambda: compression_closure(sp.eye(2), [[1, 1], [0, 0]]),
        lambda: minimal_predictive_completion([[1, 0]], {}),
        lambda: minimal_predictive_completion([[1, 0]], [1]),
        lambda: minimal_predictive_completion([[1]], {str(i): [[1]] for i in range(1_025)}),
        lambda: minimal_predictive_completion([[1, 0]], {"L": [[1]]}),
        lambda: minimal_predictive_completion([[1, 0]], {"L": sp.eye(2)}, max_rounds=True),
        lambda: projected_memory([[1]], [[1]], horizon=True),
        lambda: projected_memory([[1]], [[1]], horizon=10_001),
        lambda: projected_memory(sp.eye(20), sp.eye(20), horizon=2_000),
        lambda: delay_closure([[1]], [[1]], delay=0),
        lambda: delay_closure([[1]], [[1, 0]], delay=1),
        lambda: delay_closure([[1]], [[1]], delay=257),
        lambda: identify_data_span([[1, 0]], [[1]]),
        lambda: identify_data_span([[1]] * 257, [[1]] * 257),
    ],
)
def test_dimension_parameter_and_resource_rejections(call):
    with pytest.raises((ValidationError, DimensionError, ResourceLimitError)):
        call()


@pytest.mark.parametrize("raw", [[[0] * 257] * 257, [0] * 65_537])
def test_oversized_raw_inputs_are_rejected_before_sympy_allocation(monkeypatch, raw):
    def allocation_sentinel(*args, **kwargs):
        raise AssertionError("oversized input reached SymPy allocation")

    monkeypatch.setattr(sp, "Matrix", allocation_sentinel)
    with pytest.raises(ResourceLimitError, match="entry exact matrix limit"):
        compression_closure(raw, [[1]])


@pytest.mark.parametrize("shape", [(257, 257), (65_537,)])
def test_oversized_numpy_inputs_are_rejected_before_list_copy(monkeypatch, shape):
    class NoListCopy(np.ndarray):
        def tolist(self):
            raise AssertionError("oversized array reached .tolist()")

    raw = np.zeros(shape, dtype=int).view(NoListCopy)
    with pytest.raises(ResourceLimitError, match="entry exact matrix limit"):
        compression_closure(raw, [[1]])


@pytest.mark.parametrize("raw", [[], [[]], [[1, 2], [3]], [[1], 2]])
def test_invalid_shapes_are_rejected_before_sympy_allocation(monkeypatch, raw):
    def allocation_sentinel(*args, **kwargs):
        raise AssertionError("malformed input reached SymPy allocation")

    monkeypatch.setattr(sp, "Matrix", allocation_sentinel)
    with pytest.raises(DimensionError):
        compression_closure(raw, [[1]])
