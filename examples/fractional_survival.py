"""Fractional spectral transfer and phase type survival."""

from __future__ import annotations

import numpy as np

from komorebi_math.dynamics import PhaseTypeSurvival, fractional_power, fractional_semigroup

operator = np.diag([1.0, 4.0, 9.0])
root = fractional_power(operator, 0.5)
transfer = fractional_semigroup(operator, 0.5, 0.7)

survival = PhaseTypeSurvival([[-0.7, 0.2], [0.3, -0.6]], [1.0, 0.0])
survival.certificate().require()
survival.identities_certificate(0.5).require()

print("square root spectrum:", np.diag(root))
print("fractional transfer diagonal:", np.diag(transfer))
print("survival at t=1:", survival.survival(1.0))
print("mean lifetime:", survival.moment(1))
