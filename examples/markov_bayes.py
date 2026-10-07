"""Exact Markov lumping and Bayesian sufficiency on the same partition."""

from __future__ import annotations

import numpy as np

from komorebi_math.dynamics import BayesianPartition, MarkovLumping

generator = np.array(
    [
        [-0.4, 0.2, 0.0, 0.0],
        [0.3, -0.3, 0.0, 0.0],
        [0.1, 0.1, -0.6, 0.4],
        [0.0, 0.0, 0.6, -0.4],
    ]
)
partition = [0, 0, 1, 1]
lumping = MarkovLumping(generator, partition)
lumping.certificate().require()

prior = np.array([0.1, 0.2, 0.3, 0.4])
likelihood = np.array([2.0, 2.0, 0.5, 0.5])
bayesian = BayesianPartition(partition)
coarse_posterior = bayesian.coarse_posterior(prior, likelihood)
coarse_posterior.certificate.require()

print("coarse generator:")
print(lumping.coarse_operator)
print("coarse posterior:", coarse_posterior.value)
print("fixed sufficient:", bayesian.is_fixed_sufficient([likelihood]))
