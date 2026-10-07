from __future__ import annotations

import numpy as np

from komorebi_math.quantum import CircuitPacketExpansion, cyclic_symmetry_projectors

symmetry = np.diag([1, -1]).astype(complex)
packets = cyclic_symmetry_projectors(symmetry, 2)
hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

circuit = CircuitPacketExpansion(
    (hadamard, hadamard, hadamard),
    (packets, packets, packets, packets),
)
circuit.reconstruction_certificate().certificate.require()
circuit.block_unitarity_certificate().require()
report = circuit.interference(np.array([1, 0], dtype=complex), 0, 0)
report.certificate.require()
print("quantum circuit example passed", report.target_probability)
