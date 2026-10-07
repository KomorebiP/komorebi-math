from __future__ import annotations

import numpy as np

from komorebi_math.dynamics import matrix_exponential
from komorebi_math.nonlinear import (
    FilteredVectorSpace,
    FiniteFilteredCochainComplex,
    KatoTransport,
    MovingPacketSystem,
)


def main() -> None:
    source = FilteredVectorSpace(
        1,
        {
            0: np.eye(1, dtype=int),
            1: np.zeros((1, 0), dtype=int),
            2: np.zeros((1, 0), dtype=int),
        },
        prime=5,
    )
    target = FilteredVectorSpace.from_weights([1], prime=5)
    complex_ = FiniteFilteredCochainComplex(
        {0: 1, 1: 1},
        {0: [[1]]},
        {0: source, 1: target},
        prime=5,
    )
    spectral = complex_.spectral_sequence()
    spectral.certificate.require()
    for page in spectral.pages:
        print("page", page.page, "dimension", page.total_dimension, "rank", page.total_differential_rank)

    generator = 0.4 * np.array([[0, -1], [1, 0]], dtype=complex)
    base = np.diag([1, 0]).astype(complex)

    def packets(time: float):
        rotation = matrix_exponential(time * generator)
        first = rotation @ base @ rotation.conj().T
        return first, np.eye(2) - first

    def derivatives(time: float):
        return tuple(generator @ value - value @ generator for value in packets(time))

    system = MovingPacketSystem(packets, derivatives)
    system.derivative_certificate(0.3, step=1e-5).certificate.require()
    transport = KatoTransport(system).refinement_certificate(
        0.0,
        1.0,
        coarse_steps=32,
        tolerance=1e-6,
    )
    transport.certificate.require()
    print("transport coarse-fine difference:", transport.difference)


if __name__ == "__main__":
    main()
