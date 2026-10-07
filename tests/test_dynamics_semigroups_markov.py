from __future__ import annotations

import numpy as np

from komorebi_math import PacketSystem
from komorebi_math.dynamics import (
    ContinuousTimeMarkovChain,
    FiniteSemigroupTower,
    MarkovLumping,
    MatrixSemigroup,
    PacketSemigroup,
    SemigroupRefinement,
    matrix_exponential,
)


def test_matrix_exponential_and_semigroup_law() -> None:
    generator = np.array([[-1.0, 0.5], [0.0, -2.0]])
    semigroup = MatrixSemigroup(generator)
    semigroup.semigroup_certificate(0.3, 0.7).require()
    assert np.allclose(matrix_exponential(np.zeros((2, 2))), np.eye(2))


def test_packet_semigroup_convolution() -> None:
    generator = np.array([[-1.0, 0.0], [2.0, -3.0]])
    packets = PacketSystem([np.diag([1.0, 0.0]), np.diag([0.0, 1.0])])
    semigroup = PacketSemigroup.from_generator(generator, packets)
    semigroup.convolution_certificate(0.2, 0.4).require()
    assert semigroup.minimum_transfer_order(0, 1) == 1


def test_semigroup_refinement_duhamel_and_resolvent() -> None:
    fine = np.diag([-1.0, -2.0, -3.0])
    coarse = np.diag([-1.2, -2.4])
    coarsen = np.array([[1.0, 0.2, 0.0], [0.0, 0.3, 1.0]])
    refinement = SemigroupRefinement(coarsen, fine, coarse)
    result = refinement.duhamel(0.7, quadrature_order=40)
    result.certificate.require()
    refinement.resolvent_defect(2.0 + 0.4j).certificate.require()
    assert np.linalg.norm(refinement.direct_defect(0.4), 2) <= refinement.norm_bound(0.4) + 1e-12


def test_finite_semigroup_tower() -> None:
    generators = [np.array([[-1.0]]), np.diag([-1.0, -2.0])]
    coarsenings = [np.array([[1.0, 0.0]])]
    tower = FiniteSemigroupTower(generators, coarsenings)
    tower.certificate().require()
    assert np.allclose(tower.total_defect(0, 1, 0.5), 0.0)


def test_markov_lumping_and_stationarity() -> None:
    generator = np.array(
        [
            [-0.4, 0.2, 0.0, 0.0],
            [0.3, -0.3, 0.0, 0.0],
            [0.1, 0.1, -0.6, 0.4],
            [0.0, 0.0, 0.6, -0.4],
        ]
    )
    chain = ContinuousTimeMarkovChain(generator)
    lumping = MarkovLumping(chain, [0, 0, 1, 1])
    lumping.certificate().require()
    coarse = lumping.coarse_chain()
    probability = np.array([0.4, 0.1, 0.2, 0.3])
    assert np.allclose(lumping.aggregate(chain.evolve(probability, 0.8)), coarse.evolve(lumping.aggregate(probability), 0.8))
    stationary = chain.stationary_distribution()
    stationary.certificate.require()


def test_nonexact_markov_lumping_has_obstruction() -> None:
    generator = np.array([[-1.0, 0.0, 0.0], [0.4, -0.3, 0.1], [0.6, 0.3, -0.1]])
    lumping = MarkovLumping(generator, [0, 0, 1])
    assert not lumping.is_exact()
    assert np.linalg.norm(lumping.obstruction) > 0
