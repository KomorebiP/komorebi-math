from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


def main(output: str | None = None) -> int:
    import komorebi_math
    from komorebi_math.quantum import LindbladGenerator, charge_certificate

    source = Path(komorebi_math.__file__).resolve()
    if "site-packages" not in str(source) and "dist-packages" not in str(source):
        raise RuntimeError(f"probe must import installed artifact, got {source}")

    u = np.diag([1, np.exp(1j * np.pi / 3)]).astype(complex)
    generator = LindbladGenerator(np.zeros((2, 2), dtype=complex), ())
    wrong_lindblad = generator.coherence_preservation_certificate(u, 2)
    right_lindblad = generator.coherence_preservation_certificate(u, 6)
    wrong_charge = charge_certificate(u, np.zeros((2, 2), dtype=complex), 0, 2)
    right_charge = charge_certificate(u, np.zeros((2, 2), dtype=complex), 0, 6)

    lowering = np.array([[0, 1], [0, 0]], dtype=complex)
    raising = lowering.conj().T
    z = np.diag([1, -1]).astype(complex)
    base = LindbladGenerator(np.diag([0.2, 1.1]), (0.7 * lowering, 0.4 * raising))
    mixing = np.array([[1, 1j], [1j, 1]], dtype=complex) / np.sqrt(2)
    mixed = tuple(
        sum((mixing[a, j] * base.jumps[j] for j in range(2)), np.zeros_like(base.jumps[0]))
        for a in range(2)
    )
    shifts = np.array([0.37 + 0.2j, -0.15 + 0.31j])
    identity = np.eye(2, dtype=complex)
    shifted = tuple(mixed[a] + shifts[a] * identity for a in range(2))
    correction = np.zeros_like(base.hamiltonian)
    for scalar, operator in zip(shifts, mixed, strict=True):
        correction += (np.conjugate(scalar) * operator - scalar * operator.conj().T) / (2j)
    moved = LindbladGenerator(base.hamiltonian + correction, shifted)
    gauge_residual = float(np.linalg.norm(base.superoperator() - moved.superoperator(), ord=2))

    result = {
        "schema": "komorebi-math.episode014-installed-block18-probe",
        "schema_version": 1,
        "import_path": str(source),
        "wrong_lindblad_valid": bool(wrong_lindblad.valid),
        "right_lindblad_valid": bool(right_lindblad.valid),
        "wrong_charge_valid": bool(wrong_charge.valid),
        "right_charge_valid": bool(right_charge.valid),
        "wrong_closure_residual": float(wrong_lindblad.residuals["cyclic_closure"]),
        "gauge_superoperator_residual": gauge_residual,
        "gauge_coherence_valid": bool(moved.coherence_preservation_certificate(z, 2).valid),
    }
    result["passed"] = (
        not result["wrong_lindblad_valid"]
        and result["right_lindblad_valid"]
        and not result["wrong_charge_valid"]
        and result["right_charge_valid"]
        and result["wrong_closure_residual"] > 1.0
        and result["gauge_superoperator_residual"] < 1e-12
        and result["gauge_coherence_valid"]
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output is not None:
        Path(output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
