import numpy as np
import pytest

from komorebi_math import Exactness
from komorebi_math.exceptions import ValidationError
from komorebi_math.nonlinear.correction import RightInverseCorrectionProblem, LyapunovSchmidtReduction
from komorebi_math.nonlinear.derivatives import CallableDerivativeProvider, SympyDerivativeProvider
from komorebi_math.nonlinear.elimination import NonlinearSchurElimination
from komorebi_math.nonlinear.trees import PolynomialFixedPointProblem


def test_right_inverse_does_not_relax_invariant_ball_by_tolerance():
    p = RightInverseCorrectionProblem([[1.0]], [[1.0]], lambda x: np.array([0.0]),
                                      [1.0 + 5e-11], radius=1.0,
                                      lipschitz_bound=0.0, tolerance=1e-10)
    assert not p.hypothesis_certificate().valid
    with pytest.raises(ValidationError):
        p.solve(max_iterations=2)


def test_right_inverse_tiny_nonconvergence_does_not_underflow_to_success():
    p = RightInverseCorrectionProblem([[1.0]], [[1.0]], lambda x: 0.5*np.asarray(x),
                                      [1e-200], radius=1.0,
                                      lipschitz_bound=0.5, tolerance=1e-210)
    r = p.solve(max_iterations=1)
    assert not r.certificate.valid
    assert r.fixed_point_residual > 0
    assert r.equation_residual > 0


def test_right_inverse_rejects_nonfinite_linear_data():
    with pytest.raises(ValidationError):
        RightInverseCorrectionProblem([[np.nan]], [[1.0]], lambda x: np.array([0.0]),
                                      [0.0], radius=1.0, lipschitz_bound=0.0)


def test_right_inverse_rejects_scaled_tolerance_overflow():
    with pytest.raises(ValidationError):
        RightInverseCorrectionProblem([[1.0]], [[1.0]], lambda x: np.array([0.0]),
                                      [0.0], radius=1.0, lipschitz_bound=0.0,
                                      tolerance=1e308)


def _ls_provider_constant_tiny():
    return CallableDerivativeProvider(
        lambda x: np.array([1e-200]),
        {1: lambda x, d: np.array([0.0])},
        source_dimension=1, target_dimension=1, exactness=Exactness.NUMERICAL)


def test_lyapunov_base_residual_cannot_underflow_to_zero():
    with pytest.raises(ValidationError):
        LyapunovSchmidtReduction(_ls_provider_constant_tiny(), [0.0], tolerance=1e-210)


def test_lyapunov_rejects_nonfinite_jacobian():
    provider = CallableDerivativeProvider(
        lambda x: np.array([0.0]),
        {1: lambda x, d: np.array([np.nan])},
        source_dimension=1, target_dimension=1, exactness=Exactness.NUMERICAL)
    with pytest.raises(ValidationError):
        LyapunovSchmidtReduction(provider, [0.0], tolerance=1e-12)


def test_lyapunov_rejects_scaled_tolerance_overflow():
    provider = CallableDerivativeProvider(
        lambda x: np.array([0.0]),
        {1: lambda x, d: np.array([d[0][0]])},
        source_dimension=1, target_dimension=1, exactness=Exactness.NUMERICAL)
    with pytest.raises(ValidationError):
        LyapunovSchmidtReduction(provider, [0.0], tolerance=1e308)


def _schur_tiny_delayed_provider():
    retained = CallableDerivativeProvider(
        lambda p: np.array([0.0]),
        {1: lambda p, d: np.array([0.0])},
        source_dimension=2, target_dimension=1, exactness=Exactness.NUMERICAL)
    hidden = CallableDerivativeProvider(
        lambda p: np.array([p[1] - 1e-200*p[0]]),
        {1: lambda p, d: np.array([d[0][1] - 1e-200*d[0][0]])},
        source_dimension=2, target_dimension=1, exactness=Exactness.NUMERICAL)
    return retained, hidden


def test_schur_hidden_solve_tiny_residual_does_not_underflow():
    retained, hidden = _schur_tiny_delayed_provider()
    problem = NonlinearSchurElimination(retained, hidden, 1, 1, [0.0], [0.0], tolerance=1e-210)
    solve = problem.solve_hidden([1.0], max_iterations=3)
    solve.certificate.require()
    assert solve.residual <= 1e-210
    assert solve.hidden[0] != 0
    assert abs(float(np.real(solve.hidden[0])) - 1e-200) <= 1e-215


def test_schur_base_residual_cannot_underflow_to_zero():
    retained = CallableDerivativeProvider(
        lambda p: np.array([0.0]), {1: lambda p,d: np.array([0.0])},
        source_dimension=2, target_dimension=1)
    hidden = CallableDerivativeProvider(
        lambda p: np.array([p[1] + 1e-200]),
        {1: lambda p,d: np.array([d[0][1]])},
        source_dimension=2, target_dimension=1)
    with pytest.raises(ValidationError):
        NonlinearSchurElimination(retained, hidden, 1, 1, [0.0], [0.0], tolerance=1e-210)


def test_schur_rejects_scaled_tolerance_overflow():
    retained, hidden = _schur_tiny_delayed_provider()
    with pytest.raises(ValidationError):
        NonlinearSchurElimination(retained, hidden, 1, 1, [0.0], [0.0], tolerance=1e308)


def test_fixed_point_seed_norm_does_not_underflow_for_majorant_guard():
    problem = PolynomialFixedPointProblem(
        np.array([1e-200]),
        {2: lambda args: np.array([0.0])},
        exactness=Exactness.NUMERICAL,
        tolerance=1e-210,
    )
    result = problem.convergence_certificate({2: 0.0}, radius=1e-201)
    assert not result.certificate.valid
    assert result.seed_norm > 0


def test_fixed_point_declared_seed_norm_must_dominate_stored_seed():
    problem = PolynomialFixedPointProblem(
        np.array([1.0]), {2: lambda args: np.array([0.0])},
        exactness=Exactness.NUMERICAL, tolerance=1e-12)
    with pytest.raises(ValidationError):
        problem.convergence_certificate({2: 0.0}, radius=0.5, seed_norm=0.0)


def test_lyapunov_certificate_keeps_exact_root_as_unchecked_for_numerical_base():
    provider = CallableDerivativeProvider(
        lambda x: np.array([5e-13]),
        {1: lambda x, d: np.array([d[0][0]])},
        source_dimension=1, target_dimension=1, exactness=Exactness.NUMERICAL)
    reduction = LyapunovSchmidtReduction(provider, [0.0], tolerance=1e-12)
    assert reduction.decomposition.certificate.valid
    assert "the base point is an exact zero of F" in reduction.decomposition.certificate.assumptions_unchecked
    assert reduction.decomposition.certificate.residuals["base_equation"] > 0


def test_schur_numerical_certificate_keeps_exact_base_zero_external():
    retained, hidden = _schur_tiny_delayed_provider()
    problem = NonlinearSchurElimination(retained, hidden, 1, 1, [0.0], [0.0], tolerance=1e-12)
    cert = problem.first_derivatives().certificate
    assert cert.valid
    assert "the numerical base point is an exact hidden-equation zero" in cert.assumptions_unchecked


def test_schur_exact_tiny_nonzero_base_is_rejected_without_float_underflow():
    sp = pytest.importorskip("sympy")
    u, v = sp.symbols("u v")
    eps = sp.Rational(1, 10**400)
    retained = SympyDerivativeProvider((u,), (u, v))
    hidden = SympyDerivativeProvider((v + eps,), (u, v))
    with pytest.raises(ValidationError):
        NonlinearSchurElimination(retained, hidden, 1, 1, sp.Matrix([0]), sp.Matrix([0]))


def test_right_inverse_result_does_not_claim_finite_residual_is_proof_of_convergence():
    p = RightInverseCorrectionProblem([[1.0]], [[1.0]], lambda x: 0.2*np.asarray(x)*np.asarray(x),
                                      [0.1], radius=0.2, lipschitz_bound=0.08, tolerance=1e-12)
    result = p.solve()
    assert result.certificate.valid
    assert "fixed-point convergence in the declared ball" not in result.certificate.assumptions_checked
    assert any("finite Picard residual" in item for item in result.certificate.assumptions_checked)
