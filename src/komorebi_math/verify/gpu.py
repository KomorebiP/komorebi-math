"""Crash-isolated hardware-backed CUDA verification."""

from __future__ import annotations

from ..qualification import gpu_diagnostics
from .core import VerificationCheck, VerificationReport


def run_gpu_verification() -> VerificationReport:
    """Run PyTorch, CuPy, and Komorebi CUDA checks in a child interpreter."""

    payload = gpu_diagnostics()
    detail = str(payload.get("detail", ""))
    available = bool(payload.get("available"))
    jvp = payload.get("torch_jvp")
    residual = payload.get("packet_executor_residual")
    checks = (
        VerificationCheck(
            "isolated CUDA worker completed",
            available,
            detail if not available else f"GPU: {payload.get('gpu', 'unknown')}",
        ),
        VerificationCheck(
            "PyTorch CUDA directional JVP",
            available and jvp == [8.0, 12.0],
            f"observed {jvp!r}",
        ),
        VerificationCheck(
            "CuPy CUDA device availability",
            available and int(payload.get("device_count", 0)) > 0,
            f"device_count={payload.get('device_count')!r}",
        ),
        VerificationCheck(
            "Komorebi CuPy PacketIR equivalence",
            available and residual is not None and float(residual) <= 1e-12,
            f"max_abs_residual={residual!r}",
        ),
    )
    return VerificationReport(checks)


__all__ = ["run_gpu_verification"]
