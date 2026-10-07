from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.exceptions import ValidationError
from komorebi_math.quantum import (
    LindbladGenerator,
    QuantumChannel,
    QuantumCode,
    amplitude_damping,
    cyclic_symmetry_projectors,
    dephase,
    operator_packets,
    reconstruct_operator_packets,
)


def test_operator_packets_reconstruct_and_dephase() -> None:
    packets = cyclic_symmetry_projectors(np.diag([1, -1]), 2)
    operator = np.array([[1, 2j], [-3j, 4]], dtype=complex)
    blocks = operator_packets(operator, packets)
    assert np.allclose(reconstruct_operator_packets(blocks), operator)
    assert np.allclose(dephase(operator, packets), np.diag(np.diag(operator)))
    assert abs(sum(np.linalg.norm(block, "fro") ** 2 for row in blocks for block in row) - np.linalg.norm(operator, "fro") ** 2) < 1e-12


def test_amplitude_damping_channel_and_choi() -> None:
    for gamma in np.linspace(0, 1, 11):
        channel = amplitude_damping(float(gamma))
        channel.certificate().require()
        choi = channel.choi_matrix()
        assert np.min(np.linalg.eigvalsh((choi + choi.conj().T) / 2)) > -1e-10
        excited = np.diag([0, 1]).astype(complex)
        output = channel.apply_density(excited)
        assert np.allclose(np.diag(output), [gamma, 1 - gamma])


def test_channel_composition_tensor_and_covariance() -> None:
    first = amplitude_damping(0.2)
    second = amplitude_damping(0.3)
    state = np.array([[0.4, 0.1], [0.1, 0.6]], dtype=complex)
    assert np.allclose(second.compose(first).apply(state), second.apply(first.apply(state)))
    tensor = first.tensor(QuantumChannel.identity(2))
    tensor.certificate().require()
    z = np.diag([1, -1]).astype(complex)
    first.covariance_certificate(z).require()


def test_lindblad_trace_and_coherence_preservation() -> None:
    z = np.diag([1, -1]).astype(complex)
    lowering = np.array([[0, 1], [0, 0]], dtype=complex)
    raising = lowering.conj().T
    generator = LindbladGenerator(np.diag([0.2, 1.1]), (0.7 * lowering, 0.4 * raising))
    generator.trace_preservation_certificate().require()
    generator.coherence_preservation_certificate(z, 2).require()
    state = np.diag([0, 1]).astype(complex)
    evolved = generator.evolve(state, 0.2)
    assert abs(np.trace(evolved) - 1) < 1e-10
    assert np.min(np.linalg.eigvalsh(evolved)) > -1e-9


def bit_flip_code() -> QuantumCode:
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


def test_bit_flip_error_correction() -> None:
    code = bit_flip_code()
    code.knill_laflamme_report().certificate.require()
    recovery = code.recovery_channel()
    recovery.certificate().require()
    code.recovered_channel_certificate().require()
    logical = np.array([[0.4, 0.2], [0.2, 0.6]], dtype=complex)
    encoded = code.encode_density(logical)
    assert encoded.shape == (8, 8)
    assert abs(np.trace(encoded) - 1) < 1e-12


def test_noncorrectable_code_fails() -> None:
    projector = np.diag([1, 1, 0]).astype(complex)
    errors = (np.eye(3), np.diag([1, 2, 0]))
    code = QuantumCode(projector, errors)
    assert not code.knill_laflamme_report().certificate.valid
    with pytest.raises(ValidationError):
        code.recovery_channel()


def test_equal_weight_recovery_requires_a_trace_preserving_error_family() -> None:
    projector = np.eye(2, dtype=complex)
    code = QuantumCode(projector, (np.eye(2), 2 * np.eye(2)))
    code.knill_laflamme_report().certificate.require()
    certificate = code.recovered_channel_certificate()
    assert not certificate.valid
    assert "noise_trace_preservation" in certificate.residuals
