from __future__ import annotations

import numpy as np

from komorebi_math import PacketOperator, PacketSystem, SeedSystem, UnifiedSystem

shift = np.array(
    [
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ],
    dtype=float,
)

seeds = SeedSystem({"shift": shift}, [np.array([1, 0, 0, 0], dtype=float)])

packets = PacketSystem(
    [np.diag([1, 0, 1, 0]), np.diag([0, 1, 0, 1])],
    names=["even", "odd"],
)

blocked = PacketOperator.from_operator(shift, packets)
system = UnifiedSystem(seed_system=seeds, packet_system=packets, operators={"shift": shift})

print(system.summary())
print("hull dimension:", seeds.hull_dimension())
print("block norms:", blocked.block_norms())
