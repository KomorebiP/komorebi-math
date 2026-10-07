"""Optional CuPy execution for persisted packet-jet snapshots.

The executor consumes :class:`PacketJetSnapshot`, not arbitrary Python
callables.  This keeps the GPU boundary finite and auditable: sparse coefficient
indices and values are transferred once, then single or batched multilinear
evaluations use device gathers, products, and a matrix multiplication.

No hardware speed claim is made by this module.  Device validation must be
reported separately by :func:`validate_cupy_snapshot` on an actual CUDA system.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from typing import Any

import numpy as np

from ..certificates import Certificate, Exactness
from ..exceptions import BackendUnavailableError, DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .snapshots import PacketJetSnapshot


@dataclass(frozen=True, slots=True)
class CuPyRuntimeStatus:
    """Observed CuPy import and CUDA device state."""

    installed: bool
    runtime_available: bool
    device_count: int
    runtime_version: int | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CuPyValidationResult:
    """Matched CPU and CuPy packet-jet validation data."""

    probe_count: int
    maximum_residual: float
    runtime: CuPyRuntimeStatus
    certificate: Certificate


def cupy_runtime_status() -> CuPyRuntimeStatus:
    """Inspect CuPy and CUDA availability without raising on absent hardware."""

    if find_spec("cupy") is None:
        return CuPyRuntimeStatus(False, False, 0, None, "CuPy is not installed")
    try:
        cp = import_module("cupy")
        count = int(cp.cuda.runtime.getDeviceCount())
        version = int(cp.cuda.runtime.runtimeGetVersion())
        return CuPyRuntimeStatus(True, count > 0, count, version, None if count else "no CUDA device")
    except Exception as exc:  # pragma: no cover - depends on external CUDA state
        return CuPyRuntimeStatus(True, False, 0, None, f"{type(exc).__name__}: {exc}")


@documented(
    ProvenanceRecord(
        identifier="PSC-III-N59",
        title="Sparse CuPy packet-jet execution",
        status=ClaimStatus.FINITE_CERTIFICATE,
        source="Block 10 runtime",
        assumptions=("captured finite packet jet", "working CuPy CUDA runtime"),
        limitations=("no performance claim without matched hardware benchmarking",),
        verification_suite="nonlinear.gpu",
    )
)
class CuPySnapshotExecutor:
    """Device evaluator for one sparse :class:`PacketJetSnapshot`."""

    def __init__(self, snapshot: PacketJetSnapshot, *, dtype: Any | None = None) -> None:
        if not isinstance(snapshot, PacketJetSnapshot):
            raise ValidationError("CuPy packet-jet execution requires a PacketJetSnapshot")
        status = cupy_runtime_status()
        if not status.runtime_available:
            detail = status.error or "CUDA runtime is unavailable"
            raise BackendUnavailableError(f"CuPy packet-jet execution is unavailable: {detail}")
        cp = import_module("cupy")
        self._cp = cp
        self.snapshot = snapshot
        self.runtime = status
        indices = np.asarray(
            [coefficient.indices for coefficient in snapshot.coefficients],
            dtype=np.int64,
        )
        if indices.size == 0:
            indices = np.zeros((0, snapshot.order), dtype=np.int64)
        values = np.asarray(
            [coefficient.value for coefficient in snapshot.coefficients],
            dtype=dtype or complex,
        )
        if values.size == 0:
            values = np.zeros((0, snapshot.target_dimension), dtype=dtype or complex)
        self.indices = cp.asarray(indices)
        self.values = cp.asarray(values)
        self.dtype = self.values.dtype

    def _batch(self, directions: Any) -> Any:
        cp = self._cp
        array = cp.asarray(directions, dtype=self.dtype)
        if array.ndim == 2:
            array = array[cp.newaxis, ...]
        expected = (self.snapshot.order, self.snapshot.source_dimension)
        if array.ndim != 3 or tuple(int(value) for value in array.shape[1:]) != expected:
            raise DimensionError(
                "CuPy packet-jet directions must have shape "
                f"(batch, {expected[0]}, {expected[1]})"
            )
        return array

    def evaluate_batch(self, directions: Any) -> np.ndarray:
        """Evaluate a batch shaped ``(batch, order, source_dimension)``."""

        cp = self._cp
        batch = self._batch(directions)
        if self.indices.shape[0] == 0:
            result = cp.zeros((batch.shape[0], self.snapshot.target_dimension), dtype=self.dtype)
            return cp.asnumpy(result)
        factors = cp.ones((batch.shape[0], self.indices.shape[0]), dtype=self.dtype)
        for slot in range(self.snapshot.order):
            factors *= batch[:, slot, self.indices[:, slot]]
        return cp.asnumpy(factors @ self.values)

    def evaluate(self, directions: Sequence[Any]) -> np.ndarray:
        """Evaluate one ordered direction tuple and return a NumPy vector."""

        array = np.asarray(tuple(directions))
        result = self.evaluate_batch(array)
        return np.asarray(result[0])

    __call__ = evaluate


def validate_cupy_snapshot(
    snapshot: PacketJetSnapshot,
    probes: Sequence[Sequence[Any]],
    *,
    tolerance: float = 1e-9,
) -> CuPyValidationResult:
    """Compare matched CPU and CuPy evaluations on explicit probes."""

    tol = float(tolerance)
    if tol < 0 or not np.isfinite(tol):
        raise ValidationError("CuPy validation tolerance must be finite and nonnegative")
    probe_tuple = tuple(tuple(probe) for probe in probes)
    if not probe_tuple:
        raise ValidationError("CuPy validation requires at least one probe")
    executor = CuPySnapshotExecutor(snapshot)
    maximum = 0.0
    for probe in probe_tuple:
        cpu = np.asarray(snapshot.evaluate(probe), dtype=complex)
        gpu = np.asarray(executor.evaluate(probe), dtype=complex)
        maximum = max(maximum, float(np.linalg.norm(cpu - gpu)))
    valid = maximum <= tol
    return CuPyValidationResult(
        len(probe_tuple),
        maximum,
        executor.runtime,
        Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"maximum_cpu_gpu_residual": maximum},
            tolerance=tol,
            assumptions_checked=(
                "matched snapshot coefficients",
                "matched probe directions",
                "actual CuPy CUDA execution",
            ),
            assumptions_unchecked=(
                "performance comparison and timing-scope equivalence",
                "behavior outside the supplied finite probes",
            ),
            diagnostics={
                "probe_count": len(probe_tuple),
                "device_count": executor.runtime.device_count,
                "runtime_version": executor.runtime.runtime_version,
            },
            theorem_id="PSC-III-N59",
            message=(
                "CuPy packet-jet evaluations match the CPU snapshot"
                if valid
                else "CuPy packet-jet validation failed"
            ),
        ),
    )


# Descriptive alias retained for callers that prefer the full object name.
CuPyPacketJetSnapshotExecutor = CuPySnapshotExecutor


__all__ = [
    "CuPyPacketJetSnapshotExecutor",
    "CuPySnapshotExecutor",
    "CuPyRuntimeStatus",
    "CuPyValidationResult",
    "cupy_runtime_status",
    "validate_cupy_snapshot",
]
