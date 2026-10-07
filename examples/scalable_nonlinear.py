from __future__ import annotations

from fractions import Fraction

import numpy as np

from komorebi_math import Exactness, PacketSystem
from komorebi_math.export import dumps_portable, loads_portable
from komorebi_math.nonlinear import (
    PacketJetFamily,
    PolynomialDerivativeProvider,
    SparsePolynomialDerivativeProvider,
    compile_packet_jet_plan,
    cupy_runtime_status,
    execute_packet_jet_plan,
    packet_jet_support,
    snapshot_packet_jet,
)


def main() -> None:
    dense_tensors = {
        0: np.array([Fraction(1, 3), Fraction(-2, 5)], dtype=object),
        2: np.array(
            [
                [[Fraction(2), Fraction(0)], [Fraction(0), Fraction(0)]],
                [[Fraction(0), Fraction(0)], [Fraction(0), Fraction(-3)]],
            ],
            dtype=object,
        ),
    }
    dense = PolynomialDerivativeProvider(dense_tensors, exactness=Exactness.EXACT)
    sparse = SparsePolynomialDerivativeProvider.from_dense_provider(dense)
    print("sparse entries:", sparse.stored_entry_count)
    print("dense entries:", sparse.dense_entry_count)

    packets = PacketSystem(
        (
            np.diag([Fraction(1), Fraction(0)]),
            np.diag([Fraction(0), Fraction(1)]),
        ),
        names=("x", "y"),
    )
    family = PacketJetFamily(
        sparse,
        np.array([Fraction(0), Fraction(0)], dtype=object),
        packets,
        packets,
        order=2,
    )
    support = packet_jet_support(family)
    support.certificate.require()
    plan = compile_packet_jet_plan(family, support)
    plan.certificate.require()

    directions = (
        np.array([Fraction(2), Fraction(3)], dtype=object),
        np.array([Fraction(-1), Fraction(4)], dtype=object),
    )
    result = execute_packet_jet_plan(plan, family, directions)
    result.certificate.require()
    print("active packet blocks:", plan.active_address_count)
    print("pruned packet blocks:", plan.pruned_address_count)
    print("provider derivative calls:", result.derivative_calls)
    print("reconstructed derivative:", tuple(result.reconstructed))

    snapshot = snapshot_packet_jet(family.jet("x", ("x", "x")))
    encoded = dumps_portable(snapshot)
    restored = loads_portable(encoded)
    restored.comparison_certificate(family.jet("x", ("x", "x")), directions).require()
    print("snapshot coefficients:", restored.nnz)
    print("portable bytes:", len(encoded.encode("utf-8")))

    status = cupy_runtime_status()
    print("CuPy installed:", status.installed)
    print("CuPy runtime available:", status.runtime_available)
    print("CUDA devices:", status.device_count)


if __name__ == "__main__":
    main()
