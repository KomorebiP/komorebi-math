from __future__ import annotations

import numpy as np

from komorebi_math.quantum import QuantumChannel, QuantumCode, amplitude_damping


def _hadamard4() -> np.ndarray:
    return np.array(
        [
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, -1, -1, 1],
        ],
        dtype=complex,
    ) / 2


def _mix(operators: tuple[np.ndarray, ...], unitary: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(
        sum(unitary[row, column] * operators[column] for column in range(len(operators)))
        for row in range(unitary.shape[0])
    )


def _bit_flip_code_errors() -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    identity = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    zero = np.zeros(8, dtype=complex)
    zero[0] = 1
    one = np.zeros(8, dtype=complex)
    one[-1] = 1
    projector = np.outer(zero, zero.conj()) + np.outer(one, one.conj())
    errors = (
        np.eye(8, dtype=complex),
        np.kron(np.kron(x, identity), identity),
        np.kron(np.kron(identity, x), identity),
        np.kron(np.kron(identity, identity), x),
    )
    return projector, errors


def test_quantum_channel_semantics_are_invariant_under_kraus_unitary_mixing() -> None:
    channel = amplitude_damping(0.37)
    hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    mixed = QuantumChannel(_mix(channel.kraus, hadamard))
    state = np.array([[0.31, 0.12 - 0.07j], [0.12 + 0.07j, 0.69]], dtype=complex)
    assert np.allclose(channel.apply(state), mixed.apply(state), atol=1e-14, rtol=0.0)
    assert np.allclose(channel.superoperator(), mixed.superoperator(), atol=1e-14, rtol=0.0)
    assert np.allclose(channel.choi_matrix(), mixed.choi_matrix(), atol=1e-14, rtol=0.0)
    assert channel.certificate().valid
    assert mixed.certificate().valid


def test_knill_laflamme_certificate_is_error_span_invariant_under_label_mixing() -> None:
    # Same one-dimensional error space, represented once sparsely and once after
    # a unitary Hadamard change of Kraus/error labels.  The old entrywise-max
    # certificate flipped False -> True at the default tolerance because the
    # same residual was spread across sixteen label blocks.
    c = 2e-5
    error = np.diag([c, 0.0]).astype(complex)
    zero = np.zeros((2, 2), dtype=complex)
    sparse = (error, zero, zero, zero)
    mixed = _mix(sparse, _hadamard4())

    sparse_report = QuantumCode(np.eye(2), sparse).knill_laflamme_report()
    mixed_report = QuantumCode(np.eye(2), mixed).knill_laflamme_report()

    assert not sparse_report.certificate.valid
    assert not mixed_report.certificate.valid
    assert np.isclose(sparse_report.maximum_residual, 2e-10, rtol=1e-12, atol=0.0)
    assert np.isclose(mixed_report.maximum_residual, 5e-11, rtol=1e-12, atol=0.0)
    assert np.isclose(
        sparse_report.certificate.residuals["compression_obstruction"],
        mixed_report.certificate.residuals["compression_obstruction"],
        rtol=1e-12,
        atol=1e-14,
    )
    assert sparse_report.certificate.diagnostics["error_span_rank"] == 1
    assert mixed_report.certificate.diagnostics["error_span_rank"] == 1


def test_knill_laflamme_certificate_is_invariant_under_uniform_error_rescaling() -> None:
    # Correctability is a property of the represented error span. Uniformly
    # shrinking every generator must not turn an uncorrectable span into a
    # numerically correctable one.
    primitive = np.diag([1.0, 0.0]).astype(complex)
    zero = np.zeros((2, 2), dtype=complex)
    large = QuantumCode(np.eye(2), (primitive, zero)).knill_laflamme_report()
    tiny = QuantumCode(np.eye(2), (1e-12 * primitive, zero)).knill_laflamme_report()
    assert not large.certificate.valid
    assert not tiny.certificate.valid
    assert np.isclose(
        large.certificate.residuals["compression_obstruction"],
        tiny.certificate.residuals["compression_obstruction"],
        rtol=1e-12,
        atol=1e-14,
    )


def test_bit_flip_recovery_is_unchanged_by_unitary_error_label_gauge() -> None:
    projector, errors = _bit_flip_code_errors()
    base = QuantumCode(projector, errors)
    mixed = QuantumCode(projector, _mix(errors, _hadamard4()))

    base.knill_laflamme_report().certificate.require()
    mixed.knill_laflamme_report().certificate.require()
    base.recovered_channel_certificate().require()
    mixed.recovered_channel_certificate().require()

    assert np.allclose(
        base.recovery_channel().superoperator(),
        mixed.recovery_channel().superoperator(),
        atol=1e-12,
        rtol=0.0,
    )


def test_recovery_channel_depends_on_error_span_not_generator_scaling() -> None:
    projector, errors = _bit_flip_code_errors()
    baseline = QuantumCode(projector, errors).recovery_channel().superoperator()
    for scale in (1e-12, 1e-8, 1e-4, 1e4):
        scaled = QuantumCode(projector, tuple(scale * error for error in errors))
        scaled.knill_laflamme_report().certificate.require()
        assert np.allclose(
            scaled.recovery_channel().superoperator(),
            baseline,
            atol=1e-12,
            rtol=0.0,
        )
