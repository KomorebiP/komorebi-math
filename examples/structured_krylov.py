"""Solve a positive definite structured system by conjugate gradient."""

from __future__ import annotations

import numpy as np

from komorebi_math.algorithms import PacketLayout, block_jacobi_certificate, block_jacobi_preconditioner, conjugate_gradient

matrix = np.array(
    [
        [4.0, 1.0, 0.2, 0.0],
        [1.0, 3.0, 0.0, 0.1],
        [0.2, 0.0, 5.0, 1.0],
        [0.0, 0.1, 1.0, 4.0],
    ]
)
right_hand_side = np.array([1.0, 2.0, 3.0, 4.0])
layout = PacketLayout((2, 2))
preconditioner_data = block_jacobi_certificate(matrix, layout)
preconditioner_data.certificate.require()

result = conjugate_gradient(
    lambda value: matrix @ value,
    right_hand_side,
    precondition=block_jacobi_preconditioner(preconditioner_data.diagonal),
    relative_tolerance=1e-12,
)
result.certificate.require()

print("iterations", result.iterations)
print("residual", np.linalg.norm(matrix @ result.solution - right_hand_side))
