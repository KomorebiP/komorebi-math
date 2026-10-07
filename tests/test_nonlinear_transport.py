from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.dynamics import matrix_exponential
from komorebi_math.exceptions import ValidationError
from komorebi_math.nonlinear.transport import (
    KatoTransport,
    MovingPacketSystem,
    direct_packet_rotation,
)
from komorebi_math.quantum import KatoTransport as QuantumKatoTransport


def rotating_packets(speed: float = 0.4) -> MovingPacketSystem:
    generator = speed * np.array([[0, -1], [1, 0]], dtype=complex)
    base = np.diag([1, 0]).astype(complex)

    def packets(time: float):
        rotation = matrix_exponential(time * generator)
        first = rotation @ base @ rotation.conj().T
        return first, np.eye(2) - first

    def derivatives(time: float):
        return tuple(generator @ value - value @ generator for value in packets(time))

    return MovingPacketSystem(packets, derivatives)


def test_transport_is_shared_with_quantum_namespace() -> None:
    assert QuantumKatoTransport is KatoTransport


def test_declared_derivatives_and_generator_are_certified() -> None:
    system = rotating_packets()
    derivative = system.derivative_certificate(0.3, step=1e-5)
    assert derivative.certificate.valid
    assert derivative.maximum_derivative_residual < 1e-8
    system.generator_certificate(0.3).require()


def test_direct_rotation_midpoint_transport_and_refinement_agree() -> None:
    system = rotating_packets()
    rotation = direct_packet_rotation(system.packets(0.0), system.packets(0.2))
    rotation.certificate.require()
    transport = KatoTransport(system)
    midpoint = transport.evolve(0.0, 1.0, steps=128)
    midpoint.certificate.require()
    product = transport.direct_rotation_product(np.linspace(0, 1, 129))
    product.certificate.require()
    assert np.linalg.norm(midpoint.value - product.value, ord=2) < 1e-5
    refinement = transport.refinement_certificate(0.0, 1.0, coarse_steps=32, tolerance=1e-6)
    refinement.certificate.require()
    assert refinement.fine_steps == 64


def test_transport_rejects_invalid_derivative_and_partition_arguments() -> None:
    system = rotating_packets()
    with pytest.raises(ValidationError, match="step"):
        system.derivative_certificate(0.0, step=0.0)
    with pytest.raises(ValidationError, match="strictly increasing"):
        KatoTransport(system).direct_rotation_product((0.0, 0.0))
    with pytest.raises(ValidationError, match="refinement_factor"):
        KatoTransport(system).refinement_certificate(0.0, 1.0, refinement_factor=1)
