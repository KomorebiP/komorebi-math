from __future__ import annotations

import numpy as np

from komorebi_math import Exactness, PacketSystem
from komorebi_math.nonlinear import (
    PacketJetFamily,
    PolynomialDerivativeProvider,
    faa_di_bruno,
    packet_jet_support,
)


def main() -> None:
    outer = PolynomialDerivativeProvider(
        {2: np.array([[[2]]], dtype=object)},
        exactness=Exactness.EXACT,
    )
    inner = PolynomialDerivativeProvider(
        {
            1: np.array([[1]], dtype=object),
            2: np.array([[[2]]], dtype=object),
        },
        exactness=Exactness.EXACT,
    )
    composition = faa_di_bruno(
        outer,
        inner,
        np.array([2], dtype=object),
        (np.array([3], dtype=object), np.array([4], dtype=object)),
    )
    composition.certificate.require()
    print("second directional derivative:", composition.value[0])
    print("set-partition terms:", composition.raw_term_count)

    tensor = np.zeros((2, 2, 2), dtype=float)
    tensor[0, 0, 0] = 2.0
    tensor[1, 1, 1] = 2.0
    provider = PolynomialDerivativeProvider({2: tensor}, source_dimension=2, target_dimension=2)
    packets = PacketSystem(
        (np.diag([1.0, 0.0]), np.diag([0.0, 1.0])),
        names=("x", "y"),
    )
    family = PacketJetFamily(provider, np.array([1.0, 2.0]), packets, packets, order=2)
    directions = (np.array([2.0, 3.0]), np.array([-1.0, 4.0]))
    family.reconstruction_certificate(directions).require()
    support = packet_jet_support(family)
    support.certificate.require()
    print("nonzero packet jets:", support.nonzero)


if __name__ == "__main__":
    main()
