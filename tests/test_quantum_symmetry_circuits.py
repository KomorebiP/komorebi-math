from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.core import PacketSystem
from komorebi_math.exceptions import ValidationError
from komorebi_math.quantum import (
    CircuitPacketExpansion,
    QuantumSeedSystem,
    charge_certificate,
    charge_selection_certificate,
    computational_basis_generation,
    cyclic_symmetry_projectors,
    finite_group_projector,
    local_orbit_dimension,
    observable_seed_rank,
    schmidt_rank,
    total_charge_packets,
)


def test_quantum_seed_system_and_computational_addresses() -> None:
    shift = np.roll(np.eye(4), 1, axis=0)
    system = QuantumSeedSystem([shift], [np.array([1, 0, 0, 0])])
    assert system.orbit_dimension() == 4
    system.certificate().require()
    seed, addresses, certificate = computational_basis_generation(3)
    certificate.require()
    assert len(addresses) == 8
    assert all(np.count_nonzero(address @ seed) == 1 for address in addresses)


def test_schmidt_rank_and_local_orbit() -> None:
    product = np.array([1, 0, 0, 0], dtype=complex)
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    assert schmidt_rank(product, 2, 2) == 1
    assert schmidt_rank(bell, 2, 2) == 2
    assert local_orbit_dimension(product, 2, 2) == 2
    assert local_orbit_dimension(bell, 2, 2) == 4
    assert observable_seed_rank([2, 3], [5, 4]) == 3


def test_cyclic_projectors_and_charge_selection() -> None:
    z = np.diag([1, -1]).astype(complex)
    packets = cyclic_symmetry_projectors(z, 2)
    packets.certificate().require()
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    charge_certificate(z, x, 1, 2).require()
    charge_selection_certificate(packets, x, 1).require()


def test_total_charge_tensor_refinement() -> None:
    z = np.diag([1, -1]).astype(complex)
    packets = cyclic_symmetry_projectors(z, 2)
    total = total_charge_packets(packets, packets)
    total.certificate().require()
    assert total.packet_dimensions() == {"0": 2, "1": 2}


def test_finite_group_projector_for_z2() -> None:
    z = np.diag([1, -1]).astype(complex)
    representation = {0: np.eye(2), 1: z}
    trivial = finite_group_projector(representation, {0: 1, 1: 1}, 1)
    sign = finite_group_projector(representation, {0: 1, 1: -1}, 1)
    assert np.allclose(trivial, np.diag([1, 0]))
    assert np.allclose(sign, np.diag([0, 1]))


def test_circuit_path_expansion_and_interference() -> None:
    z = np.diag([1, -1]).astype(complex)
    packets = cyclic_symmetry_projectors(z, 2)
    h = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    circuit = CircuitPacketExpansion([h, h, h], [packets, packets, packets, packets])
    circuit.reconstruction_certificate().certificate.require()
    circuit.block_unitarity_certificate().require()
    report = circuit.interference(np.array([1, 0], dtype=complex), 0, 0)
    report.certificate.require()
    assert abs(report.target_probability - report.reconstructed_probability) < 1e-12


def test_circuit_path_safety_limit() -> None:
    packets = PacketSystem([np.diag([1, 0]), np.diag([0, 1])])
    gate = np.eye(2)
    with pytest.raises(ValidationError):
        CircuitPacketExpansion([gate] * 20, [packets] * 21, max_paths=100)


def test_charge_certificate_rejects_nonpositive_order() -> None:
    identity = np.eye(2, dtype=complex)
    with pytest.raises(ValidationError):
        charge_certificate(identity, identity, 0, 0)


def test_schmidt_rank_rejects_noninteger_or_nonpositive_dimensions() -> None:
    with pytest.raises(ValidationError):
        schmidt_rank(np.ones(4), 2.5, 2)
    with pytest.raises(ValidationError):
        schmidt_rank(np.ones(0), 0, 2)
