from __future__ import annotations

import numpy as np

from komorebi_math import PacketSystem
from komorebi_math.nonlinear import (
    PacketJetFamily,
    PolynomialDerivativeProvider,
    SparsePolynomialDerivativeProvider,
    compile_packet_jet_plan,
    execute_packet_jet_plan,
    packet_jet_support,
)


def main() -> None:
    tensor = np.zeros((3, 3, 3), dtype=float)
    tensor[0, 0, 0] = 2.0
    tensor[1, 0, 2] = tensor[1, 2, 0] = -1.5
    tensor[2, 2, 2] = 4.0

    dense = PolynomialDerivativeProvider(
        {2: tensor},
        source_dimension=3,
        target_dimension=3,
    )
    sparse = SparsePolynomialDerivativeProvider.from_dense_provider(dense)
    sparse.storage_certificate().require()

    packets = PacketSystem(
        tuple(np.diag([1.0 if index == packet else 0.0 for index in range(3)]) for packet in range(3)),
        names=("x", "y", "z"),
    )
    family = PacketJetFamily(sparse, np.zeros(3), packets, packets, order=2)
    support = packet_jet_support(family)
    support.certificate.require()
    plan = compile_packet_jet_plan(family, support)
    plan.certificate.require()

    directions = (
        np.array([2.0, -1.0, 0.5]),
        np.array([-0.25, 3.0, 4.0]),
    )
    result = execute_packet_jet_plan(plan, family, directions)
    result.certificate.require()

    print("stored derivative entries:", sparse.stored_entry_count)
    print("dense derivative entries:", sparse.dense_entry_count)
    print("active packet blocks:", plan.active_address_count)
    print("pruned packet blocks:", result.pruned_blocks)
    print("planned derivative calls:", result.derivative_calls)
    print("naive derivative calls:", plan.naive_derivative_call_count)
    print("reconstruction:", np.asarray(result.reconstructed).tolist())


if __name__ == "__main__":
    main()
