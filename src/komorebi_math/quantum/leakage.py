"""Fixed and moving packet leakage bounds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..dynamics._linalg import matrix_exponential
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._validation import require_hermitian, require_projector
from .kato import MovingPacketSystem


@dataclass(frozen=True, slots=True)
class LeakageResult:
    """Observed leakage, theorem bound, and validation certificate."""

    observed: float
    bound: float
    coupling_norm: float
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="PSC-VII-T18",
        title="Fixed packet leakage bound",
        status=ClaimStatus.PROVED,
        source="Paper VII",
        assumptions=("finite Hermitian Hamiltonian", "orthogonal packet projector"),
        verification_suite="quantum.leakage",
    )
)
def fixed_packet_leakage(
    hamiltonian: Any,
    projector: Any,
    time: float,
    *,
    tolerance: float = 1e-10,
) -> LeakageResult:
    """Evaluate and certify ``||(I-P)e^{-itH}P|| <= |t| ||(I-P)HP||``."""

    h = require_hermitian(hamiltonian, tolerance=tolerance, name="Hamiltonian")
    p = require_projector(projector, tolerance=tolerance)
    if h.shape != p.shape:
        raise DimensionError("Hamiltonian and projector dimensions do not agree")
    t = float(time)
    if not np.isfinite(t):
        raise ValidationError("time must be finite")
    q = np.eye(h.shape[0], dtype=complex) - p
    coupling = float(np.linalg.norm(q @ h @ p, ord=2))
    observed = float(np.linalg.norm(q @ matrix_exponential(-1j * t * h) @ p, ord=2))
    bound = abs(t) * coupling
    residual = max(0.0, observed - bound)
    return LeakageResult(
        observed,
        bound,
        coupling,
        Certificate(
            valid=residual <= tolerance * max(1.0, observed, bound),
            exactness=Exactness.NUMERICAL,
            residuals={"bound_violation": residual},
            tolerance=tolerance * max(1.0, observed, bound),
            assumptions_checked=("Hermitian Hamiltonian", "orthogonal packet projector"),
            theorem_id="PSC-VII-T18",
            message="fixed packet leakage obeys the commutator bound",
        ),
    )


def moving_geometric_coupling(
    hamiltonian: Any,
    system: MovingPacketSystem,
    time: float,
    packet: int,
) -> np.ndarray:
    """Return ``Q(H-iK)P`` for one moving packet."""

    h = require_hermitian(hamiltonian, tolerance=system.tolerance, name="Hamiltonian")
    if h.shape != (system.dimension, system.dimension):
        raise DimensionError("Hamiltonian and moving packet dimensions do not agree")
    packets = system.packets(time)
    index = int(packet)
    if not 0 <= index < system.count:
        raise ValidationError("packet index lies outside the moving packet family")
    p = packets[index]
    q = np.eye(system.dimension, dtype=complex) - p
    return q @ (h - 1j * system.canonical_generator(time)) @ p


def moving_packet_leakage_bound(
    hamiltonian: Callable[[float], Any],
    system: MovingPacketSystem,
    start: float,
    stop: float,
    packet: int,
    *,
    samples: int = 256,
) -> ComputationResult[float]:
    """Integrate the norm of the moving geometric coupling."""

    count = int(samples)
    if isinstance(samples, bool) or count != samples or count < 2:
        raise ValidationError("samples must be an integer at least two")
    start_time = float(start)
    stop_time = float(stop)
    if not np.isfinite(start_time) or not np.isfinite(stop_time) or stop_time < start_time:
        raise ValidationError("start and stop must be finite with stop at least start")
    times = np.linspace(start_time, stop_time, count)
    norms = np.array(
        [
            np.linalg.norm(moving_geometric_coupling(hamiltonian(float(time)), system, float(time), packet), ord=2)
            for time in times
        ],
        dtype=float,
    )
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        # NumPy 1.26 is supported. This is the same composite trapezoid
        # formula, without relying on the API introduced in NumPy 2.0.
        value = float(np.sum(np.diff(times) * (norms[1:] + norms[:-1]) / 2.0))
    else:
        value = float(trapezoid(norms, times))
    return ComputationResult(
        value,
        Certificate(
            valid=np.isfinite(value) and value >= 0,
            exactness=Exactness.NUMERICAL,
            assumptions_checked=("sampled moving geometric coupling",),
            assumptions_unchecked=("quadrature error between samples",),
            diagnostics={"samples": count},
            theorem_id="PSC-VII-T22",
            message="moving packet leakage bound was evaluated by finite quadrature",
        ),
    )


__all__ = ["LeakageResult", "fixed_packet_leakage", "moving_geometric_coupling", "moving_packet_leakage_bound"]
