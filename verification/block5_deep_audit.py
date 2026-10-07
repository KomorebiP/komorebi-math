"""Deterministic independent checks for the Block 5 analytic and quantum release."""

from __future__ import annotations

import ast
import json
from collections import defaultdict
from fractions import Fraction
from math import comb, cos, factorial, pi
from pathlib import Path
from typing import Any

import numpy as np

from komorebi_math.analytic import (
    CyclicDenominator,
    CyclicSectionFamily,
    appell_coefficients,
    appell_polynomial,
    congruential_inverse,
    cyclic_addition,
    cyclic_packets,
    cyclic_product_packets,
    cyclic_project_coefficients,
    cyclic_project_value,
    cyclic_reconstruct,
    denominator,
    denominator_root_sum,
    logarithmic_coefficients,
    positive_zero_string,
    reciprocal_coefficients,
    reciprocal_semigroup_certificate,
    signed_sections,
    signed_shift_matrix,
    theta_polynomial,
    transfer_sections,
    transfer_shift_matrix,
    transfer_tangent,
)
from komorebi_math.core import PacketSystem
from komorebi_math.dynamics import matrix_exponential
from komorebi_math.quantum import (
    BoundedHamiltonianTower,
    CircuitPacketExpansion,
    EuclideanTransfer,
    KatoTransport,
    LindbladGenerator,
    MovingPacketSystem,
    QuantumChannel,
    QuantumCode,
    QuantumFeshbachReduction,
    QuantumSeedSystem,
    amplitude_damping,
    charge_certificate,
    charge_selection_certificate,
    computational_basis_generation,
    constraint_closure,
    cyclic_symmetry_projectors,
    dephase,
    direct_packet_rotation,
    fixed_packet_leakage,
    heat_trace,
    local_orbit_dimension,
    low_mode_dispersion_bound,
    observable_seed_rank,
    operator_packets,
    periodic_laplacian,
    periodic_lattice_eigenvalues,
    reconstruct_operator_packets,
    schmidt_rank,
    spectral_dimension,
    total_charge_packets,
    transported_constraint_defect,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "komorebi_math"


class Audit:
    """Count deterministic checks and retain a bounded failure report."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = defaultdict(int)
        self.failures: list[dict[str, Any]] = []

    def check(self, group: str, condition: bool, detail: str = "") -> None:
        self.counts[group] += 1
        if not bool(condition):
            self.failures.append({"group": group, "detail": detail or "condition failed"})

    def close(self) -> dict[str, Any]:
        total = sum(self.counts.values())
        return {
            "passed": not self.failures,
            "total_checks": total,
            "passed_checks": total - len(self.failures),
            "failed_checks": len(self.failures),
            "groups": dict(sorted(self.counts.items())),
            "failures": self.failures[:100],
        }


def random_unitary(rng: np.random.Generator, dimension: int) -> np.ndarray:
    """Return a deterministic Haar style random unitary from QR factorization."""

    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(matrix)
    phases = np.diag(r)
    phases = np.where(np.abs(phases) > 0, phases / np.abs(phases), 1.0)
    return q @ np.diag(np.conj(phases))


def rotating_packet_system(speed: float) -> MovingPacketSystem:
    """Return a two packet rotation with known canonical generator."""

    generator = speed * np.array([[0, -1], [1, 0]], dtype=complex)
    first = np.diag([1, 0]).astype(complex)

    def packets(time: float) -> tuple[np.ndarray, np.ndarray]:
        rotation = matrix_exponential(time * generator)
        p = rotation @ first @ rotation.conj().T
        return p, np.eye(2) - p

    def derivatives(time: float) -> tuple[np.ndarray, np.ndarray]:
        return tuple(generator @ value - value @ generator for value in packets(time))

    return MovingPacketSystem(packets, derivatives)


def bit_flip_code() -> QuantumCode:
    """Return the three qubit bit flip code with the standard error set."""

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
    return QuantumCode(projector, errors)


def run() -> dict[str, Any]:
    """Execute the complete deterministic Block 5 audit."""

    audit = Audit()
    rng = np.random.default_rng(20260815)

    # Cyclic coefficient projections and products.
    for _ in range(600):
        length = int(rng.integers(1, 90))
        order = int(rng.integers(1, 10))
        coefficients = rng.integers(-30, 31, size=length)
        packets = cyclic_packets(coefficients, order)
        audit.check("cyclic_coordinates", np.array_equal(cyclic_reconstruct(packets), coefficients))
        residue = int(rng.integers(0, order))
        audit.check(
            "cyclic_coordinates",
            np.array_equal(cyclic_project_coefficients(coefficients, order, residue), packets[residue]),
        )
        z = complex(rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4))
        direct = sum(value * z**index for index, value in enumerate(coefficients) if index % order == residue)
        function = lambda value, c=tuple(coefficients): sum(item * value**index for index, item in enumerate(c))
        audit.check("cyclic_coordinates", abs(cyclic_project_value(function, z, order, residue) - direct) < 2e-9)

    for _ in range(500):
        order = int(rng.integers(1, 9))
        left = rng.normal(size=int(rng.integers(1, 20)))
        right = rng.normal(size=int(rng.integers(1, 20)))
        product_packets = cyclic_product_packets(cyclic_packets(left, order), cyclic_packets(right, order))
        audit.check(
            "cyclic_coordinates",
            np.allclose(sum(product_packets), np.convolve(left, right), atol=2e-11, rtol=2e-11),
        )

    # Cyclic differential systems and addition laws.
    for _ in range(450):
        order = int(rng.integers(1, 9))
        z = complex(rng.uniform(-0.8, 0.8), rng.uniform(-0.4, 0.4))
        w = complex(rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.3))
        transfer_left = transfer_sections(order, z)
        transfer_right = transfer_sections(order, w)
        signed_left = signed_sections(order, z)
        signed_right = signed_sections(order, w)
        audit.check(
            "cyclic_sections",
            np.allclose(cyclic_addition(transfer_left, transfer_right, signed=False), transfer_sections(order, z + w), atol=5e-12),
        )
        audit.check(
            "cyclic_sections",
            np.allclose(cyclic_addition(signed_left, signed_right, signed=True), signed_sections(order, z + w), atol=5e-12),
        )
        transfer_shift = transfer_shift_matrix(order)
        signed_shift = signed_shift_matrix(order)
        audit.check("cyclic_sections", np.allclose(np.linalg.matrix_power(transfer_shift, order), np.eye(order)))
        audit.check("cyclic_sections", np.allclose(np.linalg.matrix_power(signed_shift, order), -np.eye(order)))
        audit.check("cyclic_sections", abs(denominator(order, z) - denominator_root_sum(order, z)) < 5e-11)
        family = CyclicSectionFamily(order, "signed")
        audit.check("cyclic_sections", np.allclose(family.fundamental_matrix(z)[:, 0], family.vector(z), atol=5e-11))

    # Exact reciprocal, congruential, Appell, and theta coefficient identities.
    for order in range(2, 8):
        for degree in range(0, 10):
            values = congruential_inverse(order, degree % order, degree)
            for n in range(degree + 1):
                total = sum(comb(order * n + degree % order, order * q) * values[q] for q in range(n + 1))
                audit.check("reciprocal_families", total == (1 if n == 0 else 0))

    for _ in range(450):
        order = int(rng.integers(2, 8))
        degree = int(rng.integers(0, 9))
        left = Fraction(int(rng.integers(-4, 5)), int(rng.integers(1, 6)))
        right = Fraction(int(rng.integers(-4, 5)), int(rng.integers(1, 6)))
        audit.check("reciprocal_families", reciprocal_semigroup_certificate(order, left, right, degree).certificate.valid)
        x = float(rng.uniform(-0.5, 0.5))
        polynomial_degree = int(rng.integers(1, 12))
        residue = int(rng.integers(0, order))
        parameter = int(rng.integers(-2, 5))
        coefficients = appell_coefficients(order, residue, parameter, polynomial_degree)
        direct = sum(value * x**power for power, value in enumerate(coefficients))
        audit.check(
            "reciprocal_families",
            abs(complex(direct - appell_polynomial(order, residue, parameter, polynomial_degree, x))) < 5e-11,
        )
        h = 1e-6
        derivative = (
            appell_polynomial(order, residue, parameter, polynomial_degree, x + h)
            - appell_polynomial(order, residue, parameter, polynomial_degree, x - h)
        ) / (2 * h)
        expected = polynomial_degree * appell_polynomial(order, residue, parameter, polynomial_degree - 1, x)
        derivative_error = abs(complex(derivative - expected))
        audit.check(
            "reciprocal_families",
            derivative_error <= 2e-8 * max(1.0, abs(complex(expected))),
        )
        if degree > 0:
            audit.check("reciprocal_families", theta_polynomial(order, degree, 0) == 0)
            if order % 2 == 0:
                audit.check("reciprocal_families", theta_polynomial(order, degree, -1) == 0)

    for order in range(2, 7):
        kappa = logarithmic_coefficients(order, 8)
        for value in (0.03, 0.07, 0.11):
            approximation = sum(order * kappa[index + 1] * value ** (order * index + order - 1) for index in range(8))
            audit.check("reciprocal_families", abs(complex(approximation) - transfer_tangent(order, value)) < 5e-8)

    # Numerical zero strings and exact low order models.
    for order in range(2, 9):
        result = positive_zero_string(order, 24)
        audit.check("zero_diagnostics", result.certificate.valid, f"order={order}")
        audit.check("zero_diagnostics", max(result.residuals) <= result.certificate.tolerance)
        audit.check("zero_diagnostics", all(result.zeros[i] < result.zeros[i + 1] for i in range(len(result.zeros) - 1)))
        # Direct root sum evaluation is stable only on the first finite representatives.
        # The exponentially scaled Newton residual in the certificate handles later roots.
        for root in result.zeros[:6]:
            audit.check("zero_diagnostics", abs(denominator(order, root)) / max(1.0, np.exp(np.cos(np.pi / order) * root)) < 1e-9)
        for orbit in result.rotational_orbits()[:6]:
            audit.check("zero_diagnostics", len(orbit) == order)
            audit.check("zero_diagnostics", max(abs(abs(value) - abs(orbit[0])) for value in orbit) < 1e-9)
    exact_two = positive_zero_string(2, 20)
    for index, root in enumerate(exact_two.zeros):
        audit.check("zero_diagnostics", abs(root - pi * (index + 0.5)) < 1e-11)
    model = CyclicDenominator(4).positive_zeros(20)
    for index, root in enumerate(model.zeros):
        audit.check("zero_diagnostics", abs(root - np.sqrt(2) * pi * (index + 0.5)) < 1e-10)

    # Quantum seeds, symmetry projectors, and circuit paths.
    for qubits in range(1, 7):
        seed, addresses, certificate = computational_basis_generation(qubits)
        audit.check("quantum_seeds_symmetry", certificate.valid)
        audit.check("quantum_seeds_symmetry", len(addresses) == 2**qubits)
        audit.check("quantum_seeds_symmetry", np.linalg.norm(seed) == 1)
    for _ in range(400):
        left = int(rng.integers(1, 5))
        right = int(rng.integers(1, 5))
        state = rng.normal(size=left * right) + 1j * rng.normal(size=left * right)
        rank = schmidt_rank(state, left, right)
        audit.check("quantum_seeds_symmetry", local_orbit_dimension(state, left, right) == left * rank)
        blocks = tuple(int(rng.integers(1, 5)) for _ in range(int(rng.integers(1, 5))))
        multiplicities = tuple(int(rng.integers(0, 9)) for _ in blocks)
        expected = max((int(np.ceil(m / d)) for d, m in zip(blocks, multiplicities, strict=True)), default=0)
        audit.check("quantum_seeds_symmetry", observable_seed_rank(blocks, multiplicities) == expected)

    for order in range(2, 8):
        roots = np.exp(2j * np.pi * np.arange(order) / order)
        symmetry = np.diag(roots)
        packets = cyclic_symmetry_projectors(symmetry, order)
        audit.check("quantum_seeds_symmetry", packets.certificate().valid)
        total = total_charge_packets(packets, packets)
        audit.check("quantum_seeds_symmetry", total.certificate().valid)
        audit.check("quantum_seeds_symmetry", sum(total.packet_dimensions().values()) == order * order)
        shift = np.roll(np.eye(order), 1, axis=0).astype(complex)
        audit.check("quantum_seeds_symmetry", charge_certificate(symmetry, shift, 1, order).valid)
        audit.check("quantum_seeds_symmetry", charge_selection_certificate(packets, shift, 1).valid)

    for _ in range(300):
        dimension = int(rng.integers(2, 6))
        roots = np.exp(2j * np.pi * np.arange(dimension) / dimension)
        packets = cyclic_symmetry_projectors(np.diag(roots), dimension)
        gates = tuple(random_unitary(rng, dimension) for _ in range(int(rng.integers(1, 4))))
        circuit = CircuitPacketExpansion(gates, (packets,) * (len(gates) + 1))
        audit.check("quantum_circuits", circuit.reconstruction_certificate().certificate.valid)
        audit.check("quantum_circuits", circuit.block_unitarity_certificate().valid)
        state = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
        state = state / np.linalg.norm(state)
        source = int(rng.integers(0, dimension))
        target = int(rng.integers(0, dimension))
        report = circuit.interference(state, source, target)
        audit.check("quantum_circuits", report.certificate.valid)

    # Leakage and moving packet geometry.
    for _ in range(300):
        hamiltonian = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        hamiltonian = (hamiltonian + hamiltonian.conj().T) / 2
        unitary = random_unitary(rng, 3)
        # Build one projector with a consistent randomly selected rank.
        rank = int(rng.integers(1, 3))
        projector = unitary[:, :rank] @ unitary[:, :rank].conj().T
        result = fixed_packet_leakage(hamiltonian, projector, float(rng.uniform(-1, 1)))
        audit.check("quantum_transport", result.certificate.valid)
        audit.check("quantum_transport", result.observed <= result.bound + result.certificate.tolerance)
    for _ in range(120):
        speed = float(rng.uniform(-0.8, 0.8))
        system = rotating_packet_system(speed)
        audit.check("quantum_transport", system.generator_certificate(float(rng.uniform(0, 1))).valid)
        start = float(rng.uniform(-0.3, 0.3))
        stop = start + float(rng.uniform(0.05, 0.6))
        direct = direct_packet_rotation(system.packets(start), system.packets(stop))
        audit.check("quantum_transport", direct.certificate.valid)
        kato = KatoTransport(system).evolve(start, stop, steps=160)
        audit.check("quantum_transport", kato.certificate.valid)
        product_result = KatoTransport(system).direct_rotation_product(np.linspace(start, stop, 81))
        audit.check("quantum_transport", product_result.certificate.valid)
        audit.check("quantum_transport", np.linalg.norm(kato.value - product_result.value, ord=2) < 2e-4)

    # Channels, Choi matrices, Lindblad evolution, and operator packet reconstruction.
    for gamma in np.linspace(0, 1, 201):
        channel = amplitude_damping(float(gamma))
        audit.check("quantum_channels", channel.certificate().valid)
        choi = channel.choi_matrix()
        audit.check("quantum_channels", np.min(np.linalg.eigvalsh((choi + choi.conj().T) / 2)) > -1e-10)
        state = np.diag([0, 1]).astype(complex)
        output = channel.apply_density(state)
        audit.check("quantum_channels", np.allclose(np.diag(output), [gamma, 1 - gamma]))
        z = np.diag([1, -1]).astype(complex)
        audit.check("quantum_channels", channel.covariance_certificate(z).valid)
    for _ in range(300):
        dimension = int(rng.integers(2, 6))
        unitary = random_unitary(rng, dimension)
        channel = QuantumChannel((unitary,))
        audit.check("quantum_channels", channel.certificate().valid)
        matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
        audit.check("quantum_channels", np.allclose(channel.apply(matrix), unitary @ matrix @ unitary.conj().T))
        projectors = tuple(np.diag([1 if index == j else 0 for index in range(dimension)]) for j in range(dimension))
        packets = PacketSystem(projectors)
        blocks = operator_packets(matrix, packets)
        audit.check("quantum_channels", np.allclose(reconstruct_operator_packets(blocks), matrix))
        audit.check("quantum_channels", np.allclose(dephase(matrix, packets), np.diag(np.diag(matrix))))
    for _ in range(200):
        lowering = np.array([[0, 1], [0, 0]], dtype=complex)
        raising = lowering.conj().T
        generator = LindbladGenerator(
            np.diag(rng.uniform(0, 2, size=2)),
            (float(rng.uniform(0, 1)) * lowering, float(rng.uniform(0, 1)) * raising),
        )
        audit.check("quantum_channels", generator.trace_preservation_certificate().valid)
        audit.check("quantum_channels", generator.coherence_preservation_certificate(np.diag([1, -1]), 2).valid)
        state = np.diag([float(rng.uniform(0, 1)), 0.0]).astype(complex)
        state[1, 1] = 1 - state[0, 0]
        evolved = generator.evolve(state, float(rng.uniform(0, 0.4)))
        audit.check("quantum_channels", abs(np.trace(evolved) - 1) < 2e-9)
        audit.check("quantum_channels", np.min(np.linalg.eigvalsh((evolved + evolved.conj().T) / 2)) > -2e-8)

    code = bit_flip_code()
    for _ in range(100):
        audit.check("quantum_error_correction", code.knill_laflamme_report().certificate.valid)
        audit.check("quantum_error_correction", code.recovery_channel().certificate().valid)
        audit.check("quantum_error_correction", code.recovered_channel_certificate().valid)
        logical = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        logical = logical @ logical.conj().T
        logical = logical / np.trace(logical)
        encoded = code.encode_density(logical)
        audit.check("quantum_error_correction", abs(np.trace(encoded) - 1) < 1e-12)

    # Effective dynamics, Euclidean transfer, and spectral diagnostics.
    for _ in range(400):
        dimension = int(rng.integers(2, 7))
        matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
        hamiltonian = (matrix + matrix.conj().T) / 2
        retained_count = int(rng.integers(1, dimension))
        reduction = QuantumFeshbachReduction(hamiltonian, tuple(range(retained_count)))
        z = complex(float(np.max(np.linalg.eigvalsh(hamiltonian)) + rng.uniform(0.5, 3)), float(rng.uniform(0.1, 1.5)))
        audit.check("quantum_effective", reduction.resolvent_certificate(z).valid)
        audit.check("quantum_effective", reduction.upper_half_plane_sign_certificate(z).valid)
        kernel = reduction.memory_kernel()
        audit.check("quantum_effective", np.allclose(kernel.laplace(-1j * z), 1j * reduction.self_energy(z), atol=2e-8, rtol=2e-8))
    for _ in range(250):
        dimension = int(rng.integers(2, 8))
        matrix = rng.normal(size=(dimension, dimension))
        hamiltonian = matrix @ matrix.T
        spacing = float(rng.uniform(0.02, 0.5))
        transfer = EuclideanTransfer(hamiltonian, spacing)
        vectors = tuple(rng.normal(size=dimension) for _ in range(3))
        times = tuple(int(value) for value in rng.integers(0, 4, size=3))
        audit.check("quantum_effective", transfer.reflection_certificate(vectors, times).certificate.valid)
        recovered = EuclideanTransfer.from_transfer(transfer.operator(), spacing)
        audit.check("quantum_effective", np.allclose(recovered.operator(), transfer.operator(), atol=2e-9))
    for _ in range(500):
        shape = tuple(int(rng.integers(2, 8)) for _ in range(int(rng.integers(1, 4))))
        spacing = float(rng.uniform(0.3, 1.7))
        laplacian = periodic_laplacian(shape, spacing=spacing)
        eigenvalues = periodic_lattice_eigenvalues(shape, spacing=spacing)
        audit.check("quantum_spectral", np.allclose(np.linalg.eigvalsh(laplacian), eigenvalues, atol=2e-8))
        time = float(rng.uniform(0.02, 1.0))
        audit.check("quantum_spectral", heat_trace(laplacian, time) > 0)
        audit.check("quantum_spectral", spectral_dimension(laplacian, time) >= -1e-12)
        mode = tuple(int(rng.integers(0, max(1, size // 2 + 1))) for size in shape)
        discrete, continuum, bound = low_mode_dispersion_bound(shape, mode, spacing=spacing)
        audit.check("quantum_spectral", discrete <= continuum + 1e-9)
        audit.check("quantum_spectral", continuum - discrete <= bound + 1e-8)

    # Constraint closure and bounded towers.
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.diag([1, -1]).astype(complex)
    for _ in range(300):
        unitary = random_unitary(rng, 2)
        constraints = tuple(unitary @ value @ unitary.conj().T for value in (x, y, z))
        report = constraint_closure(constraints)
        audit.check("quantum_constraints_towers", report.certificate.valid)
        defects = transported_constraint_defect(constraints, constraints, np.eye(2))
        audit.check("quantum_constraints_towers", max(np.linalg.norm(value) for value in defects) < 1e-12)
        perturbations = tuple(
            constraints[2] + factor * constraints[0]
            for factor in (0.0, float(rng.uniform(0, 0.02)), float(rng.uniform(0.02, 0.04)))
        )
        tower = BoundedHamiltonianTower(perturbations)
        tower_report = tower.report(float(rng.uniform(-1, 1)))
        audit.check("quantum_constraints_towers", tower_report.certificate.valid)
        audit.check("quantum_constraints_towers", np.isclose(tower_report.total_variation, sum(tower_report.adjacent_differences)))

    # Source and public API integrity.
    python_files = tuple(sorted(SRC.rglob("*.py")))
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            audit.check("source_integrity", True)
        except SyntaxError as exc:
            audit.check("source_integrity", False, f"{path}: {exc}")
    for package_name in ("analytic", "quantum"):
        module = __import__(f"komorebi_math.{package_name}", fromlist=["*"])
        audit.check("source_integrity", len(module.__all__) == len(set(module.__all__)), f"duplicate {package_name} exports")
        for name in module.__all__:
            audit.check("source_integrity", hasattr(module, name), f"missing {package_name}.{name}")

    return audit.close()


if __name__ == "__main__":
    report = run()
    output = Path(__file__).with_name("BLOCK5_DEEP_AUDIT.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)
