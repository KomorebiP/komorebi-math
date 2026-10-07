from __future__ import annotations

import numpy as np

from komorebi_math.quantum import (
    BoundedHamiltonianTower,
    EuclideanTransfer,
    periodic_laplacian,
    periodic_lattice_eigenvalues,
    spectral_dimension,
)

transfer = EuclideanTransfer(np.diag([0.0, 1.0, 2.0]), 0.2)
transfer.reflection_certificate(
    (np.array([1, 0, 0]), np.array([0, 1, 1])),
    (0, 2),
).certificate.require()

laplacian = periodic_laplacian((4, 5))
assert np.allclose(np.linalg.eigvalsh(laplacian), periodic_lattice_eigenvalues((4, 5)))
assert spectral_dimension(laplacian, 0.2) >= 0

tower = BoundedHamiltonianTower(
    (
        np.diag([1.0, -1.0]),
        np.array([[1.0, 0.01], [0.01, -1.0]]),
        np.array([[1.0, 0.015], [0.015, -1.0]]),
    )
)
tower.report(0.4).certificate.require()
print("Euclidean and spectral example passed")
