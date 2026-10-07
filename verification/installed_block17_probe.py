from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


def _mix(operators, matrix):
    return tuple(
        sum(matrix[row, column] * operators[column] for column in range(len(operators)))
        for row in range(matrix.shape[0])
    )


def main(output: str | None = None) -> int:
    import komorebi_math
    from komorebi_math.quantum import QuantumCode

    source = Path(komorebi_math.__file__).resolve()
    if "site-packages" not in str(source) and "dist-packages" not in str(source):
        raise RuntimeError(f"probe must import installed artifact, got {source}")

    c = 2e-5
    error = np.diag([c, 0.0]).astype(complex)
    zero = np.zeros((2, 2), dtype=complex)
    sparse = (error, zero, zero, zero)
    hadamard = np.array(
        [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
        dtype=complex,
    ) / 2
    spread = _mix(sparse, hadamard)
    sparse_report = QuantumCode(np.eye(2), sparse).knill_laflamme_report()
    spread_report = QuantumCode(np.eye(2), spread).knill_laflamme_report()

    identity = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    ket0 = np.zeros(8, dtype=complex)
    ket0[0] = 1
    ket1 = np.zeros(8, dtype=complex)
    ket1[-1] = 1
    projector = np.outer(ket0, ket0.conj()) + np.outer(ket1, ket1.conj())
    errors = (
        np.eye(8, dtype=complex),
        np.kron(np.kron(x, identity), identity),
        np.kron(np.kron(identity, x), identity),
        np.kron(np.kron(identity, identity), x),
    )
    baseline = QuantumCode(projector, errors).recovery_channel().superoperator()
    scaled = QuantumCode(projector, tuple(1e-12 * error for error in errors))
    scaled_recovery_error = float(np.linalg.norm(scaled.recovery_channel().superoperator() - baseline, ord=2))

    result = {
        "schema": "komorebi-math.episode013-installed-block17-probe",
        "schema_version": 1,
        "import_path": str(source),
        "sparse_valid": sparse_report.certificate.valid,
        "spread_valid": spread_report.certificate.valid,
        "sparse_declared_max": sparse_report.maximum_residual,
        "spread_declared_max": spread_report.maximum_residual,
        "sparse_invariant": sparse_report.certificate.residuals["compression_obstruction"],
        "spread_invariant": spread_report.certificate.residuals["compression_obstruction"],
        "scaled_recovery_superoperator_error": scaled_recovery_error,
    }
    result["passed"] = (
        not result["sparse_valid"]
        and not result["spread_valid"]
        and abs(result["sparse_invariant"] - result["spread_invariant"]) < 1e-12
        and scaled_recovery_error < 1e-10
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output is not None:
        Path(output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
