from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.dynamics import matrix_exponential
from komorebi_math.exceptions import ValidationError
from komorebi_math.quantum import (
    BoundedHamiltonianTower,
    EuclideanTransfer,
    KatoTransport,
    MovingPacketSystem,
    QuantumFeshbachReduction,
    constraint_closure,
    direct_packet_rotation,
    fixed_packet_leakage,
    heat_trace,
    low_mode_dispersion_bound,
    moving_geometric_coupling,
    periodic_laplacian,
    periodic_lattice_eigenvalues,
    spectral_dimension,
    transported_constraint_defect,
)


def rotating_packets(speed: float = 0.4) -> MovingPacketSystem:
    generator = speed * np.array([[0, -1], [1, 0]], dtype=complex)
    p = np.diag([1, 0]).astype(complex)

    def packets(time: float):
        rotation = matrix_exponential(time * generator)
        return (rotation @ p @ rotation.conj().T, rotation @ (np.eye(2) - p) @ rotation.conj().T)

    def derivatives(time: float):
        return tuple(generator @ value - value @ generator for value in packets(time))

    return MovingPacketSystem(packets, derivatives)


def test_fixed_and_moving_packet_transport() -> None:
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    p = np.diag([1, 0]).astype(complex)
    fixed_packet_leakage(x, p, 0.7).certificate.require()
    system = rotating_packets()
    system.generator_certificate(0.3).require()
    rotation = direct_packet_rotation(system.packets(0.0), system.packets(0.2))
    rotation.certificate.require()
    transport = KatoTransport(system).evolve(0.0, 1.0, steps=200)
    transport.certificate.require()
    direct = KatoTransport(system).direct_rotation_product(np.linspace(0, 1, 101))
    direct.certificate.require()
    assert np.linalg.norm(transport.value - direct.value, ord=2) < 1e-4


def test_geometric_coupling_vanishes_for_kato_hamiltonian() -> None:
    system = rotating_packets()
    time = 0.4
    hamiltonian = 1j * system.canonical_generator(time)
    coupling = moving_geometric_coupling(hamiltonian, system, time, 0)
    assert np.linalg.norm(coupling) < 1e-12


def test_quantum_feshbach_and_memory() -> None:
    hamiltonian = np.array([[0.2, 0.3], [0.3, 1.4]], dtype=complex)
    reduction = QuantumFeshbachReduction(hamiltonian, (0,))
    reduction.resolvent_certificate(2.1 + 0.4j).require()
    reduction.upper_half_plane_sign_certificate(2.1 + 0.4j).require()
    kernel = reduction.memory_kernel()
    z = 2.1 + 0.4j
    assert np.allclose(kernel.laplace(-1j * z), 1j * reduction.self_energy(z), atol=1e-8)
    assert not reduction.bare_closure_certificate().valid


def test_euclidean_reflection_and_recovery() -> None:
    transfer = EuclideanTransfer(np.diag([0.0, 1.0, 2.0]), 0.2)
    vectors = (np.array([1, 0, 0]), np.array([0, 1, 1]))
    result = transfer.reflection_certificate(vectors, (0, 2))
    result.certificate.require()
    recovered = EuclideanTransfer.from_transfer(transfer.operator(), 0.2)
    assert np.allclose(recovered.operator(), transfer.operator(), atol=1e-10)


def test_periodic_lattice_spectrum_and_dimension() -> None:
    for shape in ((4,), (3, 4)):
        laplacian = periodic_laplacian(shape)
        expected = periodic_lattice_eigenvalues(shape)
        assert np.allclose(np.linalg.eigvalsh(laplacian), expected, atol=1e-9)
        assert heat_trace(laplacian, 0.2) > 0
        assert spectral_dimension(laplacian, 0.2) >= 0
    discrete, continuum, bound = low_mode_dispersion_bound((32,), (1,))
    assert 0 <= continuum - discrete <= bound + 1e-10


def test_constraint_closure_and_transport() -> None:
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.diag([1, -1]).astype(complex)
    report = constraint_closure((x, y, z))
    report.certificate.require()
    defects = transported_constraint_defect((x, z), (x, z), np.eye(2))
    assert max(np.linalg.norm(value) for value in defects) == 0


def test_nonclosed_constraint_span() -> None:
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.diag([1, -1]).astype(complex)
    assert not constraint_closure((x, z)).certificate.valid


def test_bounded_hamiltonian_tower() -> None:
    z = np.diag([1, -1]).astype(complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    tower = BoundedHamiltonianTower((z, z + 0.02 * x, z + 0.03 * x))
    report = tower.report(0.4)
    report.certificate.require()
    assert report.total_variation == pytest.approx(sum(report.adjacent_differences))


def test_invalid_euclidean_transfer() -> None:
    with pytest.raises(ValidationError):
        EuclideanTransfer(np.diag([0.0, 1.0]), 0.0)


def test_quantum_time_and_lattice_validation_boundaries() -> None:
    z = np.diag([1, -1]).astype(complex)
    p = np.diag([1, 0]).astype(complex)
    with pytest.raises(ValidationError):
        fixed_packet_leakage(z, p, float("nan"))
    with pytest.raises(ValidationError):
        KatoTransport(rotating_packets()).evolve(float("nan"), 1.0)
    with pytest.raises(ValidationError):
        KatoTransport(rotating_packets()).direct_rotation_product((0.0, float("nan")))
    with pytest.raises(ValidationError):
        periodic_laplacian((4.5,))
    with pytest.raises(ValidationError):
        periodic_lattice_eigenvalues((True,))
    with pytest.raises(ValidationError):
        low_mode_dispersion_bound((8,), (5,))
    with pytest.raises(ValidationError):
        low_mode_dispersion_bound((8,), (1,), spacing=float("nan"))
    with pytest.raises(ValidationError):
        EuclideanTransfer(np.diag([0.0, 1.0]), 0.2, ground_energy=float("nan"))
    with pytest.raises(ValidationError):
        BoundedHamiltonianTower((z,)).report(float("nan"))
