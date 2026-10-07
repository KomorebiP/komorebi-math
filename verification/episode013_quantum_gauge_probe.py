from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from komorebi_math.quantum import QuantumChannel, QuantumCode

SEED = 20260923


def unitary(rng: np.random.Generator, n: int) -> np.ndarray:
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(z)
    diagonal = np.diag(r)
    phases = np.where(np.abs(diagonal) > 0, diagonal / np.abs(diagonal), 1.0)
    return q @ np.diag(np.conj(phases))


def mix(operators: tuple[np.ndarray, ...], matrix: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(
        sum(matrix[row, column] * operators[column] for column in range(len(operators)))
        for row in range(matrix.shape[0])
    )


def random_channel(rng: np.random.Generator, env: int = 3) -> QuantumChannel:
    z = rng.normal(size=(2 * env, 2)) + 1j * rng.normal(size=(2 * env, 2))
    q, _ = np.linalg.qr(z)
    kraus = tuple(q[2 * index : 2 * (index + 1), :] for index in range(env))
    return QuantumChannel(kraus)


def random_projector(rng: np.random.Generator, dimension: int = 4, rank: int = 2) -> np.ndarray:
    q = unitary(rng, dimension)
    basis = q[:, :rank]
    return basis @ basis.conj().T


def moderate_invertible(rng: np.random.Generator, n: int) -> np.ndarray:
    left = unitary(rng, n)
    right = unitary(rng, n)
    scales = np.exp(rng.uniform(-2.0, 2.0, size=n))
    return left @ np.diag(scales) @ right


def bit_flip_data() -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
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


def run() -> dict[str, object]:
    rng = np.random.default_rng(SEED)

    channel_trials = 400
    channel_failures = 0
    channel_max_superoperator_error = 0.0
    channel_max_choi_error = 0.0
    for _ in range(channel_trials):
        channel = random_channel(rng)
        transformed = QuantumChannel(mix(channel.kraus, unitary(rng, len(channel.kraus))))
        super_error = float(np.linalg.norm(channel.superoperator() - transformed.superoperator(), ord=2))
        choi_error = float(np.linalg.norm(channel.choi_matrix() - transformed.choi_matrix(), ord=2))
        channel_max_superoperator_error = max(channel_max_superoperator_error, super_error)
        channel_max_choi_error = max(channel_max_choi_error, choi_error)
        if not channel.certificate().valid or not transformed.certificate().valid or super_error > 1e-11 or choi_error > 1e-11:
            channel_failures += 1

    span_trials = 400
    span_failures = 0
    span_max_abs_error = 0.0
    span_max_rel_error = 0.0
    declared_max_ratio = 1.0
    for _ in range(span_trials):
        projector = random_projector(rng)
        errors = tuple(
            rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
            for _ in range(3)
        )
        transformed = mix(errors, moderate_invertible(rng, 3))
        first = QuantumCode(projector, errors).knill_laflamme_report()
        second = QuantumCode(projector, transformed).knill_laflamme_report()
        a = first.certificate.residuals["compression_obstruction"]
        b = second.certificate.residuals["compression_obstruction"]
        absolute = abs(a - b)
        relative = absolute / max(1.0, abs(a), abs(b))
        span_max_abs_error = max(span_max_abs_error, absolute)
        span_max_rel_error = max(span_max_rel_error, relative)
        positive = [value for value in (first.maximum_residual, second.maximum_residual) if value > 0]
        if len(positive) == 2:
            declared_max_ratio = max(declared_max_ratio, max(positive) / min(positive))
        if relative > 5e-11:
            span_failures += 1

    recovery_trials = 80
    recovery_failures = 0
    recovery_max_superoperator_error = 0.0
    projector, errors = bit_flip_data()
    baseline = QuantumCode(projector, errors)
    baseline_recovery = baseline.recovery_channel().superoperator()
    for _ in range(recovery_trials):
        transformed_code = QuantumCode(projector, mix(errors, unitary(rng, 4)))
        report = transformed_code.knill_laflamme_report()
        recovered = transformed_code.recovered_channel_certificate()
        difference = float(np.linalg.norm(transformed_code.recovery_channel().superoperator() - baseline_recovery, ord=2))
        recovery_max_superoperator_error = max(recovery_max_superoperator_error, difference)
        if not report.certificate.valid or not recovered.valid or difference > 1e-10:
            recovery_failures += 1

    recovery_scaling_failures = 0
    recovery_scaling_max_superoperator_error = 0.0
    for scale in (1e-12, 1e-8, 1e-4, 1e4):
        scaled = QuantumCode(projector, tuple(scale * error for error in errors))
        difference = float(np.linalg.norm(scaled.recovery_channel().superoperator() - baseline_recovery, ord=2))
        recovery_scaling_max_superoperator_error = max(recovery_scaling_max_superoperator_error, difference)
        if not scaled.knill_laflamme_report().certificate.valid or difference > 1e-10:
            recovery_scaling_failures += 1

    recovery_general_linear_trials = 80
    recovery_general_linear_failures = 0
    recovery_general_linear_max_superoperator_error = 0.0
    recovery_general_linear_max_condition = 0.0
    for _ in range(recovery_general_linear_trials):
        transform = moderate_invertible(rng, 4)
        recovery_general_linear_max_condition = max(
            recovery_general_linear_max_condition, float(np.linalg.cond(transform))
        )
        transformed_code = QuantumCode(projector, mix(errors, transform))
        difference = float(
            np.linalg.norm(
                transformed_code.recovery_channel().superoperator() - baseline_recovery,
                ord=2,
            )
        )
        recovery_general_linear_max_superoperator_error = max(
            recovery_general_linear_max_superoperator_error, difference
        )
        if not transformed_code.knill_laflamme_report().certificate.valid or difference > 1e-10:
            recovery_general_linear_failures += 1

    c = 2e-5
    error = np.diag([c, 0.0]).astype(complex)
    zeros = tuple(np.zeros((2, 2), dtype=complex) for _ in range(3))
    sparse = (error, *zeros)
    hadamard = np.array(
        [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
        dtype=complex,
    ) / 2
    spread = mix(sparse, hadamard)
    sparse_report = QuantumCode(np.eye(2), sparse).knill_laflamme_report()
    spread_report = QuantumCode(np.eye(2), spread).knill_laflamme_report()

    scale_large = QuantumCode(np.eye(2), (np.diag([1.0, 0.0]), np.zeros((2, 2)))).knill_laflamme_report()
    scale_tiny = QuantumCode(np.eye(2), (1e-12 * np.diag([1.0, 0.0]), np.zeros((2, 2)))).knill_laflamme_report()

    return {
        "schema": "komorebi-math.episode013-quantum-gauge-probe",
        "schema_version": 1,
        "seed": SEED,
        "channel_trials": channel_trials,
        "channel_failures": channel_failures,
        "channel_max_superoperator_error": channel_max_superoperator_error,
        "channel_max_choi_error": channel_max_choi_error,
        "span_trials": span_trials,
        "span_failures": span_failures,
        "span_max_abs_error": span_max_abs_error,
        "span_max_rel_error": span_max_rel_error,
        "declared_basis_max_residual_ratio_observed": declared_max_ratio,
        "recovery_trials": recovery_trials,
        "recovery_failures": recovery_failures,
        "recovery_max_superoperator_error": recovery_max_superoperator_error,
        "recovery_scaling_failures": recovery_scaling_failures,
        "recovery_scaling_max_superoperator_error": recovery_scaling_max_superoperator_error,
        "recovery_general_linear_trials": recovery_general_linear_trials,
        "recovery_general_linear_failures": recovery_general_linear_failures,
        "recovery_general_linear_max_superoperator_error": recovery_general_linear_max_superoperator_error,
        "recovery_general_linear_max_condition": recovery_general_linear_max_condition,
        "default_tolerance_counterexample": {
            "sparse_valid": sparse_report.certificate.valid,
            "spread_valid": spread_report.certificate.valid,
            "sparse_declared_max_residual": sparse_report.maximum_residual,
            "spread_declared_max_residual": spread_report.maximum_residual,
            "sparse_invariant_obstruction": sparse_report.certificate.residuals["compression_obstruction"],
            "spread_invariant_obstruction": spread_report.certificate.residuals["compression_obstruction"],
        },
        "uniform_scaling_probe": {
            "large_valid": scale_large.certificate.valid,
            "tiny_valid": scale_tiny.certificate.valid,
            "large_invariant_obstruction": scale_large.certificate.residuals["compression_obstruction"],
            "tiny_invariant_obstruction": scale_tiny.certificate.residuals["compression_obstruction"],
            "large_declared_max_residual": scale_large.maximum_residual,
            "tiny_declared_max_residual": scale_tiny.maximum_residual,
        },
        "pass": channel_failures == 0 and span_failures == 0 and recovery_failures == 0 and recovery_scaling_failures == 0 and recovery_general_linear_failures == 0 and not sparse_report.certificate.valid and not spread_report.certificate.valid and not scale_large.certificate.valid and not scale_tiny.certificate.valid,
        "non_claims": [
            "Randomized trials do not prove universal gauge invariance; the block-operator similarity argument supplies the proof for resolved spans.",
            "Numerical SVD cannot stably resolve arbitrarily ill-conditioned error generators; the singular spectrum remains part of the certificate diagnostics.",
            "The random general-linear recovery audit is bounded to moderate generator condition numbers; arbitrarily ill-conditioned error bases remain a finite-precision boundary even though exact span semantics is unchanged.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
