from __future__ import annotations

import numpy as np

from komorebi_math.dynamics import matrix_exponential
from komorebi_math.quantum import KatoTransport, MovingPacketSystem, QuantumFeshbachReduction

generator = 0.3 * np.array([[0, -1], [1, 0]], dtype=complex)
p = np.diag([1, 0]).astype(complex)


def packets(time: float) -> tuple[np.ndarray, np.ndarray]:
    rotation = matrix_exponential(time * generator)
    first = rotation @ p @ rotation.conj().T
    return first, np.eye(2) - first


def derivatives(time: float) -> tuple[np.ndarray, np.ndarray]:
    return tuple(generator @ value - value @ generator for value in packets(time))


system = MovingPacketSystem(packets, derivatives)
transport = KatoTransport(system).evolve(0.0, 1.0, steps=160)
transport.certificate.require()

hamiltonian = np.array([[0.2, 0.3], [0.3, 1.4]], dtype=complex)
reduction = QuantumFeshbachReduction(hamiltonian, (0,))
z = 2.1 + 0.4j
reduction.resolvent_certificate(z).require()
kernel = reduction.memory_kernel()
assert np.allclose(kernel.laplace(-1j * z), 1j * reduction.self_energy(z), atol=1e-8)
print("Kato and Feshbach example passed")
