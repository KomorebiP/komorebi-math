"""Crash-isolated CUDA qualification for optional PyTorch and CuPy paths."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from ..certificates import Certificate, Exactness
from ..exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class GpuBackendResult:
    """One optional GPU backend result returned by an isolated worker."""

    backend: str
    discovered: bool
    available: bool
    passed: bool
    version: str | None
    device: str | None
    detail: str
    returncode: int | None
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "discovered": self.discovered,
            "available": self.available,
            "passed": self.passed,
            "version": self.version,
            "device": self.device,
            "detail": self.detail,
            "returncode": self.returncode,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class GpuQualificationReport:
    """Aggregate CUDA correctness report."""

    backends: tuple[GpuBackendResult, ...]
    certificate: Certificate

    @property
    def passed(self) -> bool:
        return self.certificate.valid

    def to_dict(self) -> dict[str, Any]:
        return {
            "backends": [backend.to_dict() for backend in self.backends],
            "certificate": self.certificate.to_dict(),
        }


def _worker_result(backend: str, *, timeout: float) -> GpuBackendResult:
    command = (sys.executable, "-m", "komorebi_math.release._gpu_worker", "--backend", backend)
    environment = os.environ.copy()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout),
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        return GpuBackendResult(
            backend,
            True,
            False,
            False,
            None,
            None,
            f"isolated GPU worker exceeded {exc.timeout} seconds",
            None,
            {},
        )
    output = completed.stdout.strip().splitlines()
    payload: dict[str, Any] | None = None
    for line in reversed(output):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if payload is None:
        detail = "\n".join(part for part in (completed.stdout.strip(), completed.stderr.strip()) if part)
        return GpuBackendResult(
            backend,
            True,
            False,
            False,
            None,
            None,
            detail[-4000:] or "GPU worker returned no JSON payload",
            completed.returncode,
            {},
        )
    return GpuBackendResult(
        backend=str(payload.get("backend", backend)),
        discovered=bool(payload.get("discovered", True)),
        available=bool(payload.get("available", False)),
        passed=bool(payload.get("passed", False)) and completed.returncode == 0,
        version=None if payload.get("version") is None else str(payload.get("version")),
        device=None if payload.get("device") is None else str(payload.get("device")),
        detail=str(payload.get("detail", "")),
        returncode=completed.returncode,
        diagnostics=dict(payload.get("diagnostics", {})),
    )


def qualify_gpu(
    *,
    require_torch: bool = True,
    require_cupy: bool = True,
    timeout: float = 180.0,
) -> GpuQualificationReport:
    """Run CUDA smoke and equivalence checks in isolated child interpreters."""

    try:
        normalized_timeout = float(timeout)
    except (TypeError, ValueError) as exc:
        raise ValidationError("GPU qualification timeout must be positive") from exc
    if normalized_timeout <= 0:
        raise ValidationError("GPU qualification timeout must be positive")

    results = (
        _worker_result("torch", timeout=normalized_timeout),
        _worker_result("cupy", timeout=normalized_timeout),
    )
    required = {"torch": bool(require_torch), "cupy": bool(require_cupy)}
    failures = tuple(
        result.backend
        for result in results
        if required[result.backend] and not result.passed
    )
    passed_required = tuple(
        result.backend
        for result in results
        if required[result.backend] and result.passed
    )
    certificate = Certificate(
        valid=not failures,
        exactness=Exactness.NUMERICAL,
        residuals={"failed_required_backends": float(len(failures))},
        tolerance=1e-10,
        assumptions_checked=tuple(f"isolated {name} CUDA execution" for name in passed_required),
        assumptions_unchecked=tuple(
            f"{result.backend} was optional for this invocation"
            for result in results
            if not required[result.backend]
        ),
        diagnostics={
            "required": required,
            "failures": failures,
            "devices": {result.backend: result.device for result in results},
        },
        theorem_id="KM-GPU-016",
        message=(
            "GPU qualification passed for every required backend"
            if not failures
            else f"GPU qualification failed for required backends {failures}"
        ),
    )
    return GpuQualificationReport(results, certificate)


__all__ = ["GpuBackendResult", "GpuQualificationReport", "qualify_gpu"]
