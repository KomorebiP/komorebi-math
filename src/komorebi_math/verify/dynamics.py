"""Installed package verification for semigroups, probability, and memory."""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..dynamics import (
    BayesianUpdate,
    ConditionalExpectationTower,
    DirichletGreen1D,
    FiniteFeynmanKac,
    FeshbachReduction,
    HiddenMarkovFilter,
    MarkovLumping,
    MatrixSemigroup,
    PhaseTypeSurvival,
    matrix_exponential,
    fractional_power,
)
from .core import VerificationCheck, VerificationReport


def _capture(name: str, function: Callable[[], None]) -> VerificationCheck:
    try:
        function()
        return VerificationCheck(name, True)
    except Exception as exc:
        return VerificationCheck(name, False, f"{type(exc).__name__}: {exc}")


def run_dynamics_verification() -> VerificationReport:
    """Run lightweight checks for the installed dynamics implementation."""

    checks: list[VerificationCheck] = []

    def semigroup_check() -> None:
        generator = np.array([[-1.0, 0.0], [0.0, -2.0]])
        semigroup = MatrixSemigroup(generator)
        semigroup.semigroup_certificate(0.3, 0.7).require()
        assert np.allclose(semigroup.at(0.3) @ semigroup.at(0.7), semigroup.at(1.0))

    checks.append(_capture("matrix semigroup law", semigroup_check))

    def markov_check() -> None:
        generator = np.array(
            [
                [-0.4, 0.2, 0.2, 0.0],
                [0.3, -0.3, 0.0, 0.2],
                [0.1, 0.0, -0.5, 0.4],
                [0.0, 0.1, 0.3, -0.6],
            ]
        )
        lumping = MarkovLumping(generator, [0, 0, 1, 1])
        lumping.certificate().require()
        coarse = lumping.coarse_generator()
        assert np.allclose(lumping.aggregation @ generator, coarse @ lumping.aggregation)

    checks.append(_capture("exact Markov lumping", markov_check))

    def bayes_check() -> None:
        update = BayesianUpdate([0.2, 0.3, 0.5], [1.0, 2.0, 4.0])
        posterior = update.posterior
        assert np.isclose(np.sum(posterior), 1.0)
        assert np.all(posterior >= 0.0)

    checks.append(_capture("Bayesian normalization", bayes_check))

    def fractional_check() -> None:
        operator = np.diag([1.0, 4.0, 9.0])
        assert np.allclose(fractional_power(operator, 0.5), np.diag([1.0, 2.0, 3.0]))

    checks.append(_capture("fractional spectral calculus", fractional_check))

    def green_check() -> None:
        green = DirichletGreen1D(10)
        green.certificate().require()
        source = np.linspace(-1.0, 1.0, 10)
        assert np.allclose(green.solve(source), np.linalg.solve(green.operator, source))

    checks.append(_capture("Dirichlet Green reconstruction", green_check))

    def feshbach_check() -> None:
        matrix = np.array([[2.0, 0.4], [0.4, 5.0]])
        reduction = FeshbachReduction(matrix, retained=[0])
        reduction.resolvent_certificate(1.0j).require()
        kernel = reduction.memory_kernel()
        assert kernel.retained_dimension == 1

    checks.append(_capture("Feshbach resolvent and memory", feshbach_check))

    def survival_check() -> None:
        model = PhaseTypeSurvival(np.array([[-0.7, 0.2], [0.3, -0.6]]), [1.0, 0.0])
        model.certificate().require()
        assert np.isclose(model.survival(0.0), 1.0)
        assert model.moment(1) > 0.0

    checks.append(_capture("phase type survival identities", survival_check))


    def taylor_exponential_check() -> None:
        matrix = np.array([[0.25, 0.5], [0.0, 0.25]])
        expected = np.exp(0.25) * np.array([[1.0, 0.5], [0.0, 1.0]])
        assert np.allclose(matrix_exponential(matrix, method="taylor"), expected)

    checks.append(_capture("NumPy matrix exponential fallback", taylor_exponential_check))

    def martingale_check() -> None:
        tower = ConditionalExpectationTower(
            np.ones(4),
            ([0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 2, 3]),
        )
        tower.certificate().require()
        tower.energy(np.array([1.0, 2.0, 4.0, 8.0])).certificate.require()

    checks.append(_capture("conditional expectation packet tower", martingale_check))

    def feynman_kac_check() -> None:
        transition = np.kron(np.array([[0.8, 0.3], [0.2, 0.7]]), np.eye(2))
        model = FiniteFeynmanKac(transition, [0.1, 0.1, 0.4, 0.4], kind="transition")
        coarse = model.coarse_model([0, 0, 1, 1]).require()
        assert coarse is not None and coarse.states == 2

    checks.append(_capture("finite Feynman Kac coarse transfer", feynman_kac_check))

    def hidden_markov_check() -> None:
        transition = np.kron(np.array([[0.8, 0.3], [0.2, 0.7]]), np.eye(2))
        model = HiddenMarkovFilter(transition)
        result = model.coarse_step(
            [0.1, 0.2, 0.3, 0.4],
            [2.0, 2.0, 0.5, 0.5],
            [0, 0, 1, 1],
        )
        result.certificate.require()

    checks.append(_capture("hidden Markov coarse filter", hidden_markov_check))

    return VerificationReport(tuple(checks))


__all__ = ["run_dynamics_verification"]
