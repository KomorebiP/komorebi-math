from __future__ import annotations

import numpy as np

from komorebi_math.dynamics import (
    BayesianFlow,
    BayesianPartition,
    BayesianUpdate,
    ConditionalExpectationTower,
    HiddenMarkovFilter,
    coarsest_sufficient_partition,
    relative_entropy_decomposition,
)


def test_bayesian_update_and_two_stage_coarsening() -> None:
    prior = np.array([0.1, 0.2, 0.3, 0.4])
    likelihood = np.array([2.0, 2.0, 0.5, 0.5])
    update = BayesianUpdate(prior, likelihood)
    assert np.isclose(update.posterior.sum(), 1.0)
    partition = BayesianPartition([0, 0, 1, 1])
    partition.coarse_posterior(prior, likelihood).certificate.require()
    assert partition.is_fixed_sufficient([likelihood])
    assert np.allclose(partition.fixed_coarse_likelihood(likelihood), [2.0, 0.5])


def test_coarsest_sufficient_partition() -> None:
    first = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
    second = np.array([0.5, 0.5, 0.2, 0.2, 0.7])
    partition = coarsest_sufficient_partition([first, second])
    assert partition.blocks == ((0, 1), (2, 3), (4,))


def test_relative_entropy_decomposition() -> None:
    p = np.array([0.1, 0.2, 0.3, 0.4])
    q = np.array([0.2, 0.1, 0.4, 0.3])
    result = relative_entropy_decomposition(p, q, [0, 0, 1, 1])
    result.certificate.require()


def test_bayesian_flow_replicator_derivative() -> None:
    flow = BayesianFlow([0.2, 0.3, 0.5], [1.0, -0.2, 0.4])
    time = 0.7
    h = 1e-6
    numerical = (flow.at(time + h) - flow.at(time - h)) / (2 * h)
    assert np.allclose(numerical, flow.derivative(time), atol=1e-7)
    assert flow.score_variance(time) >= 0


def test_conditional_expectation_tower() -> None:
    tower = ConditionalExpectationTower(
        [0.1, 0.2, 0.3, 0.4],
        [
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 1, 2, 3],
        ],
    )
    tower.certificate().require()
    tower.energy([1.0, 2.0, -1.0, 0.5]).certificate.require()


def test_hidden_markov_exact_coarse_filter() -> None:
    transition = np.array(
        [
            [0.5, 0.2, 0.0, 0.0],
            [0.3, 0.6, 0.0, 0.0],
            [0.2, 0.2, 0.7, 0.4],
            [0.0, 0.0, 0.3, 0.6],
        ]
    )
    filter_model = HiddenMarkovFilter(transition)
    result = filter_model.coarse_step(
        [0.4, 0.1, 0.2, 0.3],
        [2.0, 2.0, 0.5, 0.5],
        [0, 0, 1, 1],
    )
    result.certificate.require()
