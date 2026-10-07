"""Feshbach reduction, self energy, and hidden state memory."""

from __future__ import annotations

import numpy as np

from komorebi_math.dynamics import FeshbachReduction

operator = np.array(
    [
        [-0.4, 0.2, 0.3, 0.0],
        [0.0, -0.6, 0.0, -0.2],
        [0.4, 0.0, -1.0, 0.1],
        [0.0, 0.5, 0.0, -1.4],
    ]
)
reduction = FeshbachReduction(operator, retained=[0, 1])
reduction.resolvent_certificate(2.0 + 0.3j).require()

kernel = reduction.memory_kernel()
kernel.certificate(2.0 + 0.2j, horizon=15.0, points=3000).require()

print("self energy at z=2+0.3i:")
print(reduction.self_energy(2.0 + 0.3j))
print("memory kernel at zero:")
print(kernel.at(0.0))
print("exact local closure:", reduction.local_closure_certificate().valid)
