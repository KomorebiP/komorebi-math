"""Installed verification for finite quantum packet calculus."""

from __future__ import annotations

import numpy as np

from ..quantum import (
    BoundedHamiltonianTower,
    CircuitPacketExpansion,
    EuclideanTransfer,
    LindbladGenerator,
    MovingPacketSystem,
    QuantumCode,
    QuantumFeshbachReduction,
    amplitude_damping,
    computational_basis_generation,
    constraint_closure,
    cyclic_symmetry_projectors,
    direct_packet_rotation,
    fixed_packet_leakage,
    local_orbit_dimension,
    periodic_laplacian,
    periodic_lattice_eigenvalues,
    spectral_dimension,
)
from .core import VerificationCheck, VerificationReport, _capture


def run_quantum_verification() -> VerificationReport:
    """Run the installed finite quantum verification suite."""

    checks: list[VerificationCheck] = []

    def seed_check() -> None:
        _, addresses, certificate = computational_basis_generation(3)
        certificate.require()
        assert len(addresses) == 8
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        assert local_orbit_dimension(bell, 2, 2) == 4

    checks.append(_capture("quantum seeds and Schmidt orbit", seed_check))

    def symmetry_check() -> None:
        symmetry = np.diag([1, -1]).astype(complex)
        packets = cyclic_symmetry_projectors(symmetry, 2)
        packets.certificate().require()
        assert packets.packet_dimensions() == {"0": 1, "1": 1}

    checks.append(_capture("cyclic symmetry packets", symmetry_check))

    def circuit_check() -> None:
        symmetry = np.diag([1, -1]).astype(complex)
        packets = cyclic_symmetry_projectors(symmetry, 2)
        hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        circuit = CircuitPacketExpansion((hadamard, hadamard), (packets, packets, packets))
        circuit.reconstruction_certificate().certificate.require()
        circuit.block_unitarity_certificate().require()
        report = circuit.interference(np.array([1, 0], dtype=complex), 0, 0)
        report.certificate.require()

    checks.append(_capture("circuit packet paths", circuit_check))

    def leakage_transport_check() -> None:
        x = np.array([[0, 1], [1, 0]], dtype=complex)
        p = np.diag([1, 0]).astype(complex)
        fixed_packet_leakage(x, p, 0.2).certificate.require()
        generator = 0.4 * np.array([[0, -1], [1, 0]], dtype=complex)

        def packets(time: float):
            from ..dynamics import matrix_exponential

            rotation = matrix_exponential(time * generator)
            return (rotation @ p @ rotation.conj().T, rotation @ (np.eye(2) - p) @ rotation.conj().T)

        def derivatives(time: float):
            values = packets(time)
            return tuple(generator @ value - value @ generator for value in values)

        system = MovingPacketSystem(packets, derivatives)
        system.generator_certificate(0.3).require()
        step = direct_packet_rotation(system.packets(0.0), system.packets(0.2))
        step.certificate.require()

    checks.append(_capture("fixed leakage and moving transport", leakage_transport_check))

    def channel_lindblad_check() -> None:
        channel = amplitude_damping(0.2)
        channel.certificate().require()
        state = np.diag([0, 1]).astype(complex)
        assert abs(np.trace(channel.apply_density(state)) - 1) < 1e-12
        lowering = np.array([[0, 1], [0, 0]], dtype=complex)
        generator = LindbladGenerator(np.diag([0, 1]), (lowering,))
        generator.trace_preservation_certificate().require()
        evolved = generator.evolve(state, 0.1)
        assert abs(np.trace(evolved) - 1) < 1e-10

    checks.append(_capture("channels and Lindblad evolution", channel_lindblad_check))

    def error_correction_check() -> None:
        identity = np.eye(2)
        x = np.array([[0, 1], [1, 0]], dtype=complex)
        zero = np.zeros(8)
        zero[0] = 1
        one = np.zeros(8)
        one[-1] = 1
        projector = np.outer(zero, zero) + np.outer(one, one)
        errors = (
            np.eye(8),
            np.kron(np.kron(x, identity), identity),
            np.kron(np.kron(identity, x), identity),
            np.kron(np.kron(identity, identity), x),
        )
        code = QuantumCode(projector, errors)
        code.knill_laflamme_report().certificate.require()
        code.recovered_channel_certificate().require()

    checks.append(_capture("finite quantum error correction", error_correction_check))

    def effective_check() -> None:
        hamiltonian = np.array([[0.2, 0.1], [0.1, 1.5]], dtype=complex)
        reduction = QuantumFeshbachReduction(hamiltonian, (0,))
        reduction.resolvent_certificate(2.0 + 0.3j).require()
        reduction.upper_half_plane_sign_certificate(2.0 + 0.3j).require()

    checks.append(_capture("quantum Feshbach reduction", effective_check))

    def euclidean_spectral_check() -> None:
        transfer = EuclideanTransfer(np.diag([0.0, 1.0]), 0.2)
        transfer.reflection_certificate((np.array([1, 0]), np.array([0, 1])), (0, 1)).certificate.require()
        laplacian = periodic_laplacian((4,))
        expected = periodic_lattice_eigenvalues((4,))
        assert np.allclose(np.linalg.eigvalsh(laplacian), expected)
        assert spectral_dimension(laplacian, 0.1) >= 0

    checks.append(_capture("Euclidean transfer and spectral diagnostics", euclidean_spectral_check))

    def constraints_tower_check() -> None:
        x = np.array([[0, 1], [1, 0]], dtype=complex)
        y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        z = np.diag([1, -1]).astype(complex)
        constraint_closure((x, y, z)).certificate.require()
        tower = BoundedHamiltonianTower((z, z + 0.01 * x, z + 0.015 * x))
        tower.report(0.3).certificate.require()

    checks.append(_capture("constraint closure and bounded towers", constraints_tower_check))

    return VerificationReport(tuple(checks))


__all__ = ["run_quantum_verification"]
