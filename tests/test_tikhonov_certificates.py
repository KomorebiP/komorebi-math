"""Independent finite controls for Sufficient Resolution's reconstruction chain."""

from dataclasses import replace
from fractions import Fraction as F
from itertools import product

import numpy as np
import pytest

from komorebi_math.certificates import Exactness
from komorebi_math.exceptions import DimensionError, ResourceLimitError, ValidationError
from komorebi_math.signals.tikhonov import (
    TikhonovLimits,
    _affine_root_le_sqrt,
    _sqrt_le_affine_root,
    bracket_discrepancy,
    candidate_noise_acceptance,
    candidate_norm_domination,
    candidate_relative_feasibility,
    certify_squared_discrepancy,
    normal_residual_correction,
    protected_discrepancy_window,
    select_discrepancy,
    tikhonov_system,
)

sp = pytest.importorskip("sympy")


def _sq(values):
    return sum((F(value) ** 2 for value in values), F(0))


def _apply(matrix, vector):
    return [sum((F(a) * F(b) for a, b in zip(row, vector, strict=True)), F(0)) for row in matrix]


@pytest.mark.parametrize("c,parameter,data", [(2, F(4), 3), (-3, F(1, 7), 2), (0, 2, -3)])
def test_scalar_formula_and_sharp_noise_bound(c, parameter, data):
    system = tikhonov_system([[c]], parameter)
    solution = system.solve([data])
    expected = F(c * data, c * c + parameter)
    assert solution.source == sp.Matrix([expected])
    assert solution.source_squared == expected ** 2
    assert solution.fitting_squared == (c * expected - data) ** 2
    assert solution.certificate.valid and solution.certificate.exactness is Exactness.EXACT
    assert 4 * parameter * solution.source_squared <= data ** 2
    if c and parameter == c * c:
        assert 4 * parameter * solution.source_squared == data ** 2


def test_source_paper_two_coordinate_example():
    system = tikhonov_system([[1, 1], [0, 0], [0, 0]], 1)
    solution = system.solve([3, 4, 5])
    assert solution.source == sp.Matrix([1, 1])
    assert solution.fitting_squared == 42
    assert solution.objective == 44
    unregularized = system.objective_increment([3, 4, 5], [F(3, 2), F(3, 2)])
    assert unregularized.candidate_objective == F(91, 2)
    assert unregularized.certificate.valid
    assert system.kernel_projector == sp.Matrix([[F(1, 2), F(-1, 2)], [F(-1, 2), F(1, 2)]])
    assert system.range_projector == sp.diag(1, 0, 0)


@pytest.mark.parametrize("entries", list(product((-1, 0, 1), repeat=4)))
def test_all_81_small_matrices_against_two_by_two_fraction_inverse_and_primal_objective(entries):
    a, b, c, d = entries
    operator = [[a, b], [c, d]]
    parameter, data = F(2, 3), [F(1, 2), F(-2, 3)]
    system = tikhonov_system(operator, parameter)
    solution = system.solve(data)
    # Hand-coded 2x2 Cramer's rule is independent of SymPy elimination.
    n00, n01, n11 = a*a + c*c + parameter, a*b + c*d, b*b + d*d + parameter
    rhs0, rhs1 = a*data[0] + c*data[1], b*data[0] + d*data[1]
    det = n00*n11 - n01*n01
    expected = [(n11*rhs0 - n01*rhs1)/det, (n00*rhs1 - n01*rhs0)/det]
    assert solution.source == sp.Matrix(expected)
    fitted = _apply(operator, expected)
    expected_objective = _sq([fitted[i] - data[i] for i in range(2)]) + parameter*_sq(expected)
    assert solution.objective == expected_objective
    candidate = [F(-3, 5), F(7, 4)]
    result = system.objective_increment(data, candidate)
    difference = [candidate[i] - expected[i] for i in range(2)]
    candidate_fitted = _apply(operator, candidate)
    objective = _sq([candidate_fitted[i] - data[i] for i in range(2)]) + parameter*_sq(candidate)
    assert objective - expected_objective == _sq(_apply(operator, difference)) + parameter*_sq(difference)
    assert result.certificate.valid and solution.certificate.valid


@pytest.mark.parametrize("diagonal", [(1, 2, 0), (-2, 0, 3), (0, 0, 0), (F(1, 2), F(-2, 3), F(4, 5))])
def test_diagonal_coordinate_filter_and_distinct_correction_scales(diagonal):
    parameter, data, candidate = F(1, 9), [1, -2, 3], [F(4, 7), F(-3, 5), F(1, 11)]
    system = tikhonov_system(sp.diag(*diagonal), parameter)
    expected = [F(c)*v/(F(c)**2 + parameter) for c, v in zip(diagonal, data, strict=True)]
    correction = normal_residual_correction(system, data, candidate)
    normal = [(F(c)**2 + parameter)*x - F(c)*v for c, x, v in zip(diagonal, candidate, data, strict=True)]
    difference = [x-y for x, y in zip(candidate, expected, strict=True)]
    assert correction.normal_squared == _sq(normal)
    assert correction.source_correction_squared == _sq(difference)
    assert correction.image_correction_squared == _sq([F(c)*q for c, q in zip(diagonal, difference, strict=True)])
    assert correction.source_bound_squared == _sq(normal)/parameter**2
    assert correction.image_bound_squared == _sq(normal)/(4*parameter)
    assert correction.certificate.valid


def test_bias_noise_decomposition_keeps_kernel_and_bias_noise_cross_term():
    system = tikhonov_system([[1, 1]], 1)
    result = system.bias_noise([3, 1], [2])
    assert result.identifiable_source == sp.Matrix([2, 2])
    assert result.kernel_source == sp.Matrix([1, -1])
    assert result.error == sp.Matrix([-1, 1])
    assert result.identifiable_error_squared == 0
    assert result.kernel_squared == result.total_error_squared == 2
    assert result.certificate.valid
    # Bias and propagated noise need not be orthogonal: here they cancel
    # exactly in the identifiable sector, so adding their squared norms fails.
    assert _sq(result.bias) + _sq(result.propagated_noise) != result.total_error_squared


def test_resolvent_difference_changes_operator_rank_and_parameter():
    left = tikhonov_system([[1, 2], [2, 1]], F(1, 3))
    right = tikhonov_system([[1, 1], [0, 0]], F(3, 5))
    result = left.resolvent_difference(right)
    assert result.difference == result.factorization
    assert result.difference != -result.factorization
    assert result.certificate.valid


@pytest.mark.parametrize("operator", [[[0]], [[2]], [[1, 1], [0, 0]], [[1, 2], [3, 1]], [[1, 0, 2], [0, -3, 1]]])
def test_whole_map_bounds_against_exact_gram_principal_minors(operator):
    system = tikhonov_system(operator, F(1, 3))
    certificate = system.whole_map_bounds()
    assert certificate.valid
    # Independent orthogonal invariant: every principal minor is nonnegative.
    # This test oracle differs from the producer's Schur pivot recursion.
    from itertools import combinations
    margins = [sp.eye(system.operator.cols)/system.parameter**2 - system.inverse.T*system.inverse,
               sp.eye(system.operator.cols)/(4*system.parameter) - system.regularized*system.regularized.T]
    for matrix in margins:
        for size in range(1, matrix.rows+1):
            for indices in combinations(range(matrix.rows), size):
                assert matrix.extract(indices, indices).det() >= 0


def test_operator_sensitivity_checks_true_whole_map_enclosures_and_rank_change():
    left = tikhonov_system([[1, 1], [0, 0]], F(1, 3))
    right = tikhonov_system([[1, 1], [0, F(1, 10)]], F(1, 3))
    certificate = left.operator_sensitivity(right, operator_bound=2, other_operator_bound=2, difference_bound=F(1, 10))
    assert certificate.valid
    assert certificate.diagnostics["regularized_difference_norm_upper"] == F(15, 2)
    # Even a generous final allowance cannot repair a false whole-map premise.
    assert not left.operator_sensitivity(right, operator_bound=0, other_operator_bound=10, difference_bound=10).valid
    assert not left.operator_sensitivity(right, operator_bound=2, other_operator_bound=2, difference_bound=0).valid
    with pytest.raises(ValidationError, match="fixed-parameter"):
        left.operator_sensitivity(tikhonov_system([[1, 1], [0, 0]], 1),
                                  operator_bound=2, other_operator_bound=2, difference_bound=0)


def test_source_condition_requires_actual_witness_even_if_bias_bound_happens_to_hold():
    system = tikhonov_system([[1, 1], [0, 2]], F(1, 7))
    witness = sp.Matrix([F(2, 3), -1])
    source = system.operator.T*system.operator*witness
    certificate = system.source_condition_bias(source, witness)
    assert certificate.valid
    assert certificate.diagnostics["actual_bias_squared"] <= F(1, 49)*_sq(witness)
    assert not system.source_condition_bias(source, [100, 100]).valid


def test_conditioned_bias_noise_checks_restricted_coercivity_and_kernel_split():
    system = tikhonov_system([[1, 1]], F(1, 3))
    certificate = system.conditioned_bias_noise([3, 1], [F(1, 4)], 2)
    assert certificate.valid
    assert certificate.diagnostics["irreducible_kernel_squared"] == 2
    assert not system.conditioned_bias_noise([3, 1], [F(1, 4)], 3).valid
    # For the zero operator the identifiable sector is zero, so its restricted
    # coercivity is vacuous; kernel content remains the entire source error.
    zero = tikhonov_system([[0, 0]], 1).conditioned_bias_noise([2, 3], [4], 1)
    assert zero.valid and zero.diagnostics["irreducible_kernel_squared"] == 13


def test_scalar_discrepancy_exact_root_formula():
    bracket = bracket_discrepancy([[1]], [3], F(1))
    # SR-48: rho/(v-rho) = 1/(3-1) = 1/2.
    assert bracket.exact_parameter == F(1, 2)
    assert bracket.lower == bracket.upper == F(1, 2)
    assert bracket.lower_residual_squared == bracket.upper_residual_squared == 1
    assert bracket.converged and bracket.certificate.valid


def test_rank_deficient_discrepancy_root_is_a_rational_bracket_not_a_fake_exact_root():
    bracket = bracket_discrepancy([[1], [0]], [1, 1], F(3, 2), width=F(1, 1 << 18))
    # The positive real root is 1+sqrt(2), which is irrational.
    assert bracket.exact_parameter is None
    assert bracket.attainable_floor_squared == 1
    assert bracket.lower < 1 + sp.sqrt(2) < bracket.upper
    assert bracket.upper - bracket.lower <= F(1, 1 << 18)
    assert 1 + (bracket.lower/(1+bracket.lower))**2 == bracket.lower_residual_squared
    assert 1 + (bracket.upper/(1+bracket.upper))**2 == bracket.upper_residual_squared
    assert bracket.lower_residual_squared < F(3, 2) < bracket.upper_residual_squared
    assert bracket.certificate.valid


@pytest.mark.parametrize("target", [F(0), F(1), F(2), F(3)])
def test_discrepancy_strict_floor_and_data_endpoint_controls(target):
    with pytest.raises(ValidationError):
        bracket_discrepancy([[1], [0]], [1, 1], target)


def test_orthogonal_data_has_no_interior_root_and_budget_exhaustion_is_explicit():
    with pytest.raises(ValidationError, match="strict attainable"):
        bracket_discrepancy([[1], [0]], [0, 2], 2)
    with pytest.raises(ResourceLimitError, match="max_evaluations"):
        bracket_discrepancy([[1], [0]], [1, 1], F(3, 2), max_evaluations=2)


def test_total_small_data_and_interior_band_selection():
    zero = select_discrepancy([[0]], [1], F(1, 2), 3)
    assert zero.branch == "small_data" and zero.parameter is None
    assert zero.source == sp.Matrix([0]) and zero.fitting_squared == 1
    assert zero.certificate.valid
    assert "eta is a genuine true-data noise allowance" in zero.certificate.assumptions_unchecked
    selected = select_discrepancy([[1, 0], [0, 2]], [1, 1], F(1, 10), 2)
    assert selected.branch == "positive_parameter" and selected.parameter > 0
    assert F(1, 100) <= selected.fitting_squared <= F(1, 25)
    assert tikhonov_system([[1, 0], [0, 2]], selected.parameter).solve([1, 1]).source == selected.source
    assert selected.certificate.valid and selected.bracket.certificate.valid
    with pytest.raises(ValidationError, match="finite feasibility"):
        select_discrepancy([[1], [0]], [1, 1], F(1, 10), 2)


def _protected(**overrides):
    fields = dict(eta=F(1, 10), tau=3, rho=F(1, 80), xi=F(1, 80),
                  kappa=F(1, 20), measured=F(19, 100), omega=0)
    fields.update(overrides)
    return certify_squared_discrepancy(tikhonov_system([[1]], F(1, 4)), [1], [F(81, 100)], **fields)


def test_squared_protected_certificate_recomputes_every_actual_quantity():
    result = _protected()
    assert result.actual_normal_squared == F(1, 6400)
    assert result.actual_fitting_squared == F(361, 10000)
    assert result.actual_source_correction_squared == F(1, 10000)
    assert result.exact_source_fitting_squared == F(1, 25)
    assert result.certificate.valid
    assert all(check for _, check in result.checks)


def test_source_protected_scalar_window_and_empty_window_control():
    window = protected_discrepancy_window(F(1, 10), 2, F(1, 100), F(1, 100)).require_nonempty()
    assert (window.lower, window.upper, window.midpoint) == (F(3, 25), F(9, 50), F(3, 20))
    assert window.exists and window.certificate.valid
    empty = protected_discrepancy_window(F(1, 10), 2, F(1, 20), F(1, 20))
    assert not empty.exists and empty.certificate.valid
    empty.certificate.diagnostics["exists"] = True
    with pytest.raises(ValidationError, match="empty"):
        empty.require_nonempty()


@pytest.mark.parametrize("forgery", [dict(rho=0), dict(rho=F(1, 100)), dict(xi=0), dict(kappa=0),
                                   dict(measured=0), dict(omega=F(1, 5)), dict(eta=F(1, 4)), dict(tau=1)])
def test_forged_or_unprotected_certificate_is_rejected(forgery):
    result = _protected(**forgery)
    assert not result.certificate.valid
    assert any(not check for _, check in result.checks)
    with pytest.raises(ValidationError):
        result.certificate.require()


@pytest.mark.parametrize("field", ["rho", "xi", "kappa", "measured", "omega", "eta", "tau"])
def test_negative_certificate_fields_cannot_be_hidden_by_squaring(field):
    with pytest.raises(ValidationError, match="nonnegative"):
        _protected(**{field: F(-1, 100)})


def test_candidate_noise_identity_and_zero_model_source_controls():
    identity = tikhonov_system([[1]], F(1, 4))
    x = identity.solve([1]).source
    accepted = candidate_noise_acceptance(identity, [1], x, eta=F(1, 10), delta=0, tau=2)
    assert accepted.accepted and accepted.branch == "positive_parameter"
    assert accepted.fitting_squared == F(1, 25) and accepted.certificate.valid
    assert candidate_norm_domination(identity, [1], [1], eta=F(1, 10), delta=0).certificate.valid
    assert candidate_relative_feasibility([1], F(1, 10), 0, 2).valid
    zero_model = tikhonov_system([[0]], F(1, 4))
    rejected = candidate_noise_acceptance(zero_model, [1], [0], eta=F(1, 10), delta=1, tau=2)
    assert not rejected.accepted and not rejected.certificate.valid
    assert not candidate_relative_feasibility([1], F(1, 10), 1, 2).valid


def test_candidate_fit_inside_band_is_insufficient_without_actual_normal_equation():
    system = tikhonov_system([[1]], F(1, 4))
    result = candidate_noise_acceptance(system, [1], [F(17, 20)], eta=F(1, 10), delta=0, tau=2)
    assert F(1, 100) <= result.fitting_squared <= F(1, 25)
    assert result.normal_squared > 0
    assert not result.accepted


def test_nonsquare_candidate_norm_and_signed_squaring_controls():
    system = tikhonov_system(sp.eye(2), 1)
    candidate = system.solve([1, 1]).source
    result = candidate_noise_acceptance(system, [1, 1], candidate, eta=0, delta=1, tau=1)
    assert result.source_squared == result.fitting_squared == F(1, 2)
    assert result.accepted
    assert not candidate_noise_acceptance(system, [1, 1], candidate, eta=F(1, 100), delta=1, tau=2).accepted
    assert not _affine_root_le_sqrt(F(1), F(1), F(1), F(0))
    assert _sqrt_le_affine_root(F(0), F(1), F(1), F(1))


@pytest.mark.parametrize("a,b,c,d", list(product((F(0), F(1, 2), F(2)), repeat=4)))
def test_radical_comparisons_against_exact_sympy_ordering(a, b, c, d):
    # This oracle uses algebraic-number ordering; producer eliminates radicals
    # through signed rational squares and never asks SymPy to compare roots.
    left, right = sp.sqrt(sp.Rational(a.numerator, a.denominator)), b + c*sp.sqrt(sp.Rational(d.numerator, d.denominator))
    assert _sqrt_le_affine_root(a, b, c, d) == bool(left <= right)
    assert _affine_root_le_sqrt(b, c, d, a) == bool(right <= left)


@pytest.mark.parametrize("bad", [True, 1.0, "1/2", sp.sqrt(2), sp.Symbol("x")])
def test_inexact_or_symbolic_inputs_rejected(bad):
    with pytest.raises(ValidationError, match="exact rational"):
        tikhonov_system([[bad]], 1)
    with pytest.raises(ValidationError, match="exact rational"):
        tikhonov_system([[1]], bad)


@pytest.mark.parametrize("bad", [0, -1, F(-1, 2)])
def test_nonpositive_parameters_rejected(bad):
    with pytest.raises(ValidationError, match="strictly positive"):
        tikhonov_system([[1]], bad)


def test_shape_preflight_and_rational_bit_budget_precede_expensive_work():
    class ConversionSentinel:
        shape = (33, 33)

        def tolist(self):
            raise AssertionError("large conversion happened before preflight")

    with pytest.raises(ResourceLimitError, match="before conversion"):
        tikhonov_system(ConversionSentinel(), 1)
    with pytest.raises(ResourceLimitError, match="entry-bit"):
        tikhonov_system([[1 << 129]], 1, limits=TikhonovLimits(max_entry_bits=128))
    with pytest.raises(ResourceLimitError, match="forecast"):
        tikhonov_system([[1, 2], [3, 4]], 1, limits=TikhonovLimits(max_work_bits=32))
    with pytest.raises(DimensionError):
        tikhonov_system([[1, 0]], 1).solve([1, 2])
    with pytest.raises(DimensionError):
        tikhonov_system([[1, 0]], 1).objective_increment([1], [1])


def test_owned_immutable_matrices_survive_mutated_caller_inputs():
    operator = [[1, 1]]
    system = tikhonov_system(operator, 1)
    operator[0][0] = 42
    data = [3]
    solved = system.solve(data)
    data[0] = 99
    assert solved.source == sp.Matrix([1, 1])
    with pytest.raises(TypeError):
        system.normal[0, 0] = 99


def test_unrelated_true_certificate_cannot_validate_forged_model_fields():
    system = tikhonov_system([[1, 1]], 1)
    with pytest.raises(ValidationError, match="defining actual matrix"):
        replace(system, normal=sp.ImmutableMatrix(sp.eye(2)))
    with pytest.raises(ValidationError, match="defining actual matrix"):
        replace(system, kernel_projector=sp.ImmutableMatrix(sp.zeros(2)))
    with pytest.raises(ValidationError, match="immutable"):
        replace(system, inverse=sp.Matrix(system.inverse))


@pytest.mark.parametrize("parameter", [np.int64(1 << 32), F(np.int64(1 << 32))])
def test_fixed_width_scalar_and_fraction_inputs_are_owned_as_python_integers(parameter):
    data = np.array([1 << 40], dtype=np.int64)
    system = tikhonov_system(np.array([[1]], dtype=np.int64), parameter)
    solution = system.solve(data)
    assert type(system.parameter.numerator) is int
    assert type(system.parameter.denominator) is int
    assert system.parameter**2 == 1 << 64
    assert solution.source == sp.Matrix([F(1 << 40, (1 << 32)+1)])
    assert solution.certificate.valid
    assert system.whole_map_bounds().valid
    correction = normal_residual_correction(system, data, np.array([1], dtype=np.int64))
    assert correction.source_bound_squared == F(((1 << 32)+1-(1 << 40))**2, 1 << 64)
    assert correction.certificate.valid


@pytest.mark.parametrize("limits", [False, 1, {}, "default"])
def test_invalid_limit_records_rejected_consistently(limits):
    with pytest.raises(ValidationError, match="TikhonovLimits"):
        tikhonov_system([[1]], 1, limits=limits)
    with pytest.raises(ValidationError, match="TikhonovLimits"):
        bracket_discrepancy([[1]], [1], F(1, 2), limits=limits)
