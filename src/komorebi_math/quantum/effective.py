"""Hermitian Feshbach reduction and hidden quantum memory."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..certificates import Certificate, Exactness
from ..dynamics import FeshbachReduction, MemoryKernel
from ..exceptions import ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._validation import require_hermitian


@documented(
    ProvenanceRecord(
        identifier="PSC-VII-T47",
        title="Quantum Feshbach reduction",
        status=ClaimStatus.PROVED,
        source="Paper VII",
        assumptions=("finite Hermitian Hamiltonian", "declared retained sector"),
        verification_suite="quantum.effective",
    )
)
class QuantumFeshbachReduction:
    """Hermitian specialization of the finite Schur Feshbach reduction."""

    def __init__(self, hamiltonian: Any, retained: Iterable[int], *, tolerance: float = 1e-10) -> None:
        value = require_hermitian(hamiltonian, tolerance=tolerance, name="Hamiltonian")
        self.reduction = FeshbachReduction(value, retained)
        self.hamiltonian = value
        self.tolerance = float(tolerance)

    def self_energy(self, spectral_parameter: complex) -> np.ndarray:
        """Return the retained self energy."""

        return self.reduction.self_energy(spectral_parameter)

    def effective_hamiltonian(self, spectral_parameter: complex) -> np.ndarray:
        """Return the energy dependent effective Hamiltonian."""

        return self.reduction.effective_operator(spectral_parameter)

    def projected_resolvent(self, spectral_parameter: complex) -> np.ndarray:
        """Return the exact projected resolvent through the Feshbach map."""

        return self.reduction.effective_resolvent(spectral_parameter)

    def resolvent_certificate(self, spectral_parameter: complex) -> Certificate:
        """Check the projected resolvent identity."""

        return self.reduction.resolvent_certificate(spectral_parameter, tolerance=self.tolerance).certificate

    def upper_half_plane_sign_certificate(self, spectral_parameter: complex) -> Certificate:
        """Check that the self energy has nonpositive imaginary part in the upper half plane."""

        z = complex(spectral_parameter)
        if z.imag <= 0:
            raise ValidationError("spectral parameter must lie in the open upper half plane")
        sigma = self.self_energy(z)
        imaginary = (sigma - sigma.conj().T) / (2j)
        maximum = float(np.max(np.linalg.eigvalsh((imaginary + imaginary.conj().T) / 2)))
        return Certificate(
            valid=maximum <= self.tolerance,
            exactness=Exactness.NUMERICAL,
            residuals={"positive_imaginary_part": max(0.0, maximum)},
            tolerance=self.tolerance,
            assumptions_checked=("Hermitian hidden Hamiltonian", "upper half plane spectral parameter"),
            theorem_id="PSC-VII-P49",
            message="the self energy has nonpositive imaginary part",
        )

    def memory_kernel(self) -> MemoryKernel:
        """Return the exact hidden packet memory kernel for Schrödinger evolution."""

        return MemoryKernel(
            self.reduction.a_pq,
            -1j * self.reduction.hidden_block,
            self.reduction.a_qp,
        )

    def bare_closure_certificate(self) -> Certificate:
        """Check whether the retained sector is dynamically closed."""

        coupling = float(np.linalg.norm(self.reduction.a_pq, ord=2))
        return Certificate(
            valid=coupling <= self.tolerance,
            exactness=Exactness.NUMERICAL,
            residuals={"retained_hidden_coupling": coupling},
            tolerance=self.tolerance,
            assumptions_checked=("finite Hermitian block decomposition",),
            theorem_id="PSC-VII-T53",
            message="retained dynamics is local exactly when hidden coupling vanishes",
        )


__all__ = ["QuantumFeshbachReduction"]
