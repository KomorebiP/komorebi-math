"""Build and execute a structured packet intermediate representation."""

from __future__ import annotations

import numpy as np

from komorebi_math.algorithms import PacketIR, PacketLayout, packet_cost_estimate

layout = PacketLayout((2, 1), ("bulk", "edge"))
matrix = np.array(
    [
        [2.0, 1.0, 0.0],
        [0.0, 3.0, 0.0],
        [4.0, 0.0, 5.0],
    ]
)
operator = PacketIR.from_dense(matrix, layout)
right_hand_side = np.array([1.0, 2.0, 3.0])
result = operator.apply(right_hand_side)
operator.certificate().require()

cost = packet_cost_estimate(layout, layout, operator.support, rhs=1)

print("support", sorted(operator.support))
print("result", result)
print("stored entries", operator.stored_entries)
print("modeled flops", cost.flops)
