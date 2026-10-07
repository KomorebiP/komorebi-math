from __future__ import annotations

import numpy as np

from komorebi_math.dynamics import (
    DirichletGreen1D,
    FiniteFeynmanKac,
    PhaseTypeSurvival,
    fractional_defect,
    fractional_naturality_certificate,
    fractional_power,
    fractional_semigroup,
)


def test_fractional_spectral_calculus() -> None:
    operator = np.diag([1.0, 4.0, 9.0])
    assert np.allclose(fractional_power(operator, 0.5), np.diag([1.0, 2.0, 3.0]))
    semigroup = fractional_semigroup(operator, 0.5, 0.7)
    assert np.allclose(semigroup, np.diag(np.exp(-0.7 * np.array([1.0, 2.0, 3.0]))))


def test_fractional_naturality_and_defect_bound() -> None:
    fine = np.diag([2.0, 5.0, 8.0])
    coarse = np.diag([2.0, 5.0])
    coarsen = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    fractional_naturality_certificate(coarsen, fine, coarse, 0.4).require()

    perturbed = np.diag([2.2, 4.7])
    result = fractional_defect(coarsen, fine, perturbed, 0.4)
    result.certificate.require()
    assert np.linalg.norm(result.defect, 2) <= result.bound + 1e-12


def test_dirichlet_green_identity_and_modes() -> None:
    green = DirichletGreen1D(12)
    green.certificate().require()
    source = np.linspace(-1.0, 1.0, 12)
    assert np.allclose(green.solve(source), np.linalg.solve(green.operator, source))
    eigenvalues = green.eigenvalues()
    for mode in range(1, 6):
        error = (mode * np.pi) ** 2 - eigenvalues[mode - 1]
        assert 0 <= error <= green.eigenvalue_error_bound(mode) + 1e-12


def test_continuous_and_discrete_feynman_kac() -> None:
    generator = np.array([[-0.4, 0.3], [0.4, -0.3]])
    potential = np.array([0.2, 0.5])
    continuous = FiniteFeynmanKac(generator, potential)
    assert np.allclose(continuous.operator(0.3) @ continuous.operator(0.7), continuous.operator(1.0))
    observable = np.array([1.0, 2.0])
    assert continuous.apply(observable, 0.0).shape == (2,)

    transition = np.array([[0.7, 0.2], [0.3, 0.8]])
    discrete = FiniteFeynmanKac(transition, potential, kind="transition")
    assert np.allclose(discrete.operator(2), discrete.one_step_operator @ discrete.one_step_operator)


def test_exact_coarse_feynman_kac() -> None:
    generator = np.array(
        [
            [-0.4, 0.2, 0.2, 0.0],
            [0.3, -0.3, 0.0, 0.2],
            [0.1, 0.0, -0.5, 0.4],
            [0.0, 0.1, 0.3, -0.6],
        ]
    )
    potential = np.array([0.2, 0.2, 0.7, 0.7])
    result = FiniteFeynmanKac(generator, potential).coarse_model([0, 0, 1, 1])
    model = result.require()
    assert model is not None
    assert model.states == 2


def test_phase_type_survival_identities() -> None:
    subgenerator = np.array([[-0.7, 0.2], [0.3, -0.6]])
    model = PhaseTypeSurvival(subgenerator, [1.0, 0.0])
    model.certificate().require()
    assert np.isclose(model.survival(0.0), 1.0)
    assert np.isclose(model.laplace_transform(0.0), 1.0)
    assert model.moment(1) > 0
    model.identities_certificate(0.5).require()
    restarted = model.residual_model(0.5)
    assert np.isclose(restarted.survival(0.0), 1.0)


def test_discrete_coarse_feynman_kac_intertwining() -> None:
    coarse_transition = np.array([[0.8, 0.3], [0.2, 0.7]])
    transition = np.kron(coarse_transition, np.eye(2))
    potential = np.array([0.1, 0.1, 0.4, 0.4])
    fine = FiniteFeynmanKac(transition, potential, kind="transition")
    result = fine.coarse_model([0, 0, 1, 1])
    coarse = result.require()
    assert coarse is not None
    lift = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    assert np.allclose(fine.one_step_operator @ lift, lift @ coarse.one_step_operator)


def test_phase_type_competing_risk_packets() -> None:
    subgenerator = np.array([[-0.7, 0.2], [0.3, -0.6]])
    cause_rates = np.array([[0.25, 0.15], [0.15, 0.25]])
    model = PhaseTypeSurvival(subgenerator, [0.4, 0.6], cause_rates=cause_rates)
    model.certificate().require()
    for time in (0.0, 0.2, 1.0):
        assert np.isclose(np.sum(model.cause_density(time)), model.density(time))
