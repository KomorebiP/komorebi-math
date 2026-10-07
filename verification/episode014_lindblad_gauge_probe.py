from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from komorebi_math.quantum import LindbladGenerator, charge_certificate

SEED = 20260923


def random_unitary(rng: np.random.Generator, n: int) -> np.ndarray:
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(z)
    phases = np.diag(r)
    phases = np.where(np.abs(phases) > 0, phases / np.abs(phases), 1.0)
    return q * phases.conj()


def gauge_transform(
    generator: LindbladGenerator,
    mixing: np.ndarray,
    shifts: np.ndarray,
) -> LindbladGenerator:
    jumps = generator.jumps
    if not jumps:
        return LindbladGenerator(generator.hamiltonian, (), tolerance=generator.tolerance)
    mixed = tuple(
        sum((mixing[a, j] * jumps[j] for j in range(len(jumps))), np.zeros_like(jumps[0]))
        for a in range(mixing.shape[0])
    )
    identity = np.eye(generator.dimension, dtype=complex)
    shifted = tuple(operator + shifts[a] * identity for a, operator in enumerate(mixed))
    correction = np.zeros_like(generator.hamiltonian)
    for scalar, operator in zip(shifts, mixed, strict=True):
        correction += (np.conjugate(scalar) * operator - scalar * operator.conj().T) / (2j)
    return LindbladGenerator(generator.hamiltonian + correction, shifted, tolerance=generator.tolerance)


def main() -> int:
    rng = np.random.default_rng(SEED)
    gauge_trials = 800
    super_failures = 0
    apply_failures = 0
    trace_failures = 0
    max_super_rel = 0.0
    max_apply_rel = 0.0
    max_hamiltonian_change = 0.0

    for _ in range(gauge_trials):
        d = int(rng.integers(2, 5))
        k = int(rng.integers(1, 5))
        a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        h = (a + a.conj().T) / 2
        jumps = tuple(
            0.3 * (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))
            for _ in range(k)
        )
        base = LindbladGenerator(h, jumps)
        u = random_unitary(rng, k)
        shifts = 0.5 * (rng.normal(size=k) + 1j * rng.normal(size=k))
        moved = gauge_transform(base, u, shifts)

        s0 = base.superoperator()
        s1 = moved.superoperator()
        denom = max(1.0, float(np.linalg.norm(s0, ord=2)))
        rel = float(np.linalg.norm(s0 - s1, ord=2) / denom)
        max_super_rel = max(max_super_rel, rel)
        if rel > 2e-12:
            super_failures += 1

        x = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        y0 = base.apply(x)
        y1 = moved.apply(x)
        denom_y = max(1.0, float(np.linalg.norm(y0, ord=2)))
        rel_y = float(np.linalg.norm(y0 - y1, ord=2) / denom_y)
        max_apply_rel = max(max_apply_rel, rel_y)
        if rel_y > 2e-12:
            apply_failures += 1
        if not moved.trace_preservation_certificate().valid:
            trace_failures += 1
        max_hamiltonian_change = max(
            max_hamiltonian_change,
            float(np.linalg.norm(moved.hamiltonian - base.hamiltonian, ord=2)),
        )

    cyclic_trials = 400
    correct_order_failures = 0
    wrong_order_false_passes = 0
    charge_wrong_order_false_passes = 0
    genuine_wrong_order_trials = 0
    max_correct_closure = 0.0
    min_wrong_closure = float("inf")
    zero = LindbladGenerator(np.zeros((4, 4), dtype=complex), ())
    for _ in range(cyclic_trials):
        n = int(rng.integers(2, 10))
        exponents = rng.integers(0, n, size=4)
        u = np.diag(np.exp(2j * np.pi * exponents / n))
        correct = zero.coherence_preservation_certificate(u, n)
        max_correct_closure = max(max_correct_closure, correct.residuals["cyclic_closure"])
        if not correct.valid:
            correct_order_failures += 1

        # Pick a declared order that does not divide the sampled eigenvalue orders when possible.
        wrong_n = n + 1
        wrong = zero.coherence_preservation_certificate(u, wrong_n)
        closure = float(np.linalg.norm(np.linalg.matrix_power(u, wrong_n) - np.eye(4), ord=2))
        min_wrong_closure = min(min_wrong_closure, closure)
        # Some random U may accidentally also satisfy U^(n+1)=I (e.g. identity); count only genuine violations.
        if closure > zero.tolerance * max(1.0, float(wrong_n)):
            genuine_wrong_order_trials += 1
            if wrong.valid:
                wrong_order_false_passes += 1
            charge = charge_certificate(u, np.zeros((4, 4), dtype=complex), 0, wrong_n)
            if charge.valid:
                charge_wrong_order_false_passes += 1

    # Nontrivial covariance example: same Lindbladian, wildly different decomposition.
    lowering = np.array([[0, 1], [0, 0]], dtype=complex)
    raising = lowering.conj().T
    z = np.diag([1, -1]).astype(complex)
    base = LindbladGenerator(np.diag([0.2, 1.1]), (0.7 * lowering, 0.4 * raising))
    mixing = np.array([[1, 1j], [1j, 1]], dtype=complex) / np.sqrt(2)
    shifts = np.array([3.7 + 2.0j, -1.5 + 3.1j])
    moved = gauge_transform(base, mixing, shifts)
    semantic_example = {
        "superoperator_residual": float(np.linalg.norm(base.superoperator() - moved.superoperator(), ord=2)),
        "hamiltonian_change": float(np.linalg.norm(base.hamiltonian - moved.hamiltonian, ord=2)),
        "base_coherence_valid": bool(base.coherence_preservation_certificate(z, 2).valid),
        "moved_coherence_valid": bool(moved.coherence_preservation_certificate(z, 2).valid),
        "base_jump_traces": [[float(np.trace(j).real), float(np.trace(j).imag)] for j in base.jumps],
        "moved_jump_traces": [[float(np.trace(j).real), float(np.trace(j).imag)] for j in moved.jumps],
    }

    payload = {
        "schema": "komorebi-math.episode014-lindblad-gauge-probe",
        "schema_version": 1,
        "seed": SEED,
        "gauge_trials": gauge_trials,
        "gauge_failures": {
            "superoperator": super_failures,
            "apply": apply_failures,
            "trace_certificate": trace_failures,
        },
        "max_superoperator_relative_error": max_super_rel,
        "max_apply_relative_error": max_apply_rel,
        "max_hamiltonian_representation_change": max_hamiltonian_change,
        "cyclic_trials": cyclic_trials,
        "cyclic_failures": {
            "correct_order": correct_order_failures,
            "wrong_order_false_pass": wrong_order_false_passes,
            "charge_wrong_order_false_pass": charge_wrong_order_false_passes,
        },
        "genuine_wrong_order_trials": genuine_wrong_order_trials,
        "max_correct_order_closure_residual": max_correct_closure,
        "min_sampled_wrong_order_closure_residual": min_wrong_closure,
        "semantic_example": semantic_example,
        "non_claims": [
            "Randomized gauge tests do not prove the gauge theorem.",
            "The finite cyclic tests do not exhaust all unitary representations.",
            "Hamiltonian and jump-operator coordinates remain decomposition-dependent even when generator semantics agree.",
        ],
    }
    out = Path("verification/EPISODE014_LINDBLAD_GAUGE_PROBE.json")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not any(payload["gauge_failures"].values()) and not any(payload["cyclic_failures"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
