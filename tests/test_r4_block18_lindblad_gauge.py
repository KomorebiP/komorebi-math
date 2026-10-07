from __future__ import annotations

import numpy as np

from komorebi_math.quantum import LindbladGenerator, charge_certificate


def _gauge_transform(
    generator: LindbladGenerator,
    mixing: np.ndarray,
    shifts: np.ndarray,
) -> LindbladGenerator:
    jumps = generator.jumps
    mixed = tuple(
        sum((mixing[a, j] * jumps[j] for j in range(len(jumps))), np.zeros_like(jumps[0]))
        for a in range(mixing.shape[0])
    )
    identity = np.eye(generator.dimension, dtype=complex)
    shifted = tuple(operator + shifts[a] * identity for a, operator in enumerate(mixed))
    correction = np.zeros_like(generator.hamiltonian)
    for scalar, operator in zip(shifts, mixed, strict=True):
        correction += (np.conjugate(scalar) * operator - scalar * operator.conj().T) / (2j)
    return LindbladGenerator(
        generator.hamiltonian + correction,
        shifted,
        tolerance=generator.tolerance,
    )


def test_lindblad_unitary_mix_and_identity_shift_preserve_generator_semantics() -> None:
    lowering = np.array([[0, 1], [0, 0]], dtype=complex)
    raising = lowering.conj().T
    z = np.diag([1, -1]).astype(complex)
    generator = LindbladGenerator(np.diag([0.2, 1.1]), (0.7 * lowering, 0.4 * raising))

    mixing = np.array([[1, 1j], [1j, 1]], dtype=complex) / np.sqrt(2)
    shifts = np.array([0.37 + 0.2j, -0.15 + 0.31j])
    transformed = _gauge_transform(generator, mixing, shifts)

    assert np.allclose(transformed.superoperator(), generator.superoperator(), atol=2e-14, rtol=2e-14)
    assert transformed.trace_preservation_certificate().valid
    assert generator.coherence_preservation_certificate(z, 2).valid
    assert transformed.coherence_preservation_certificate(z, 2).valid


def test_cyclic_certificate_rejects_wrong_declared_order() -> None:
    generator = LindbladGenerator(np.zeros((2, 2), dtype=complex), ())
    order_six_unitary = np.diag([1, np.exp(1j * np.pi / 3)]).astype(complex)

    wrong = generator.coherence_preservation_certificate(order_six_unitary, 2)
    assert not wrong.valid
    assert wrong.residuals["coherence_commutator"] == 0.0
    assert wrong.residuals["cyclic_closure"] > 1.0
    assert "U^order = I cyclic closure" in wrong.assumptions_unchecked

    right = generator.coherence_preservation_certificate(order_six_unitary, 6)
    assert right.valid
    assert right.residuals["cyclic_closure"] < 1e-12


def test_nonfaithful_cyclic_representation_is_allowed() -> None:
    generator = LindbladGenerator(np.zeros((2, 2), dtype=complex), ())
    identity = np.eye(2, dtype=complex)
    certificate = generator.coherence_preservation_certificate(identity, 7)
    assert certificate.valid
    assert certificate.diagnostics["order"] == 7


def test_charge_certificate_rejects_noncyclic_declared_order() -> None:
    order_six_unitary = np.diag([1, np.exp(1j * np.pi / 3)]).astype(complex)
    zero_operator = np.zeros((2, 2), dtype=complex)

    wrong = charge_certificate(order_six_unitary, zero_operator, 0, 2)
    assert not wrong.valid
    assert wrong.residuals["charge_relation"] == 0.0
    assert wrong.residuals["cyclic_closure"] > 1.0

    right = charge_certificate(order_six_unitary, zero_operator, 0, 6)
    assert right.valid
