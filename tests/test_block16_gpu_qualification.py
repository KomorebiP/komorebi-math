"""Crash-isolated GPU qualification behavior without local CUDA."""

from __future__ import annotations

import json
import subprocess

import pytest

from komorebi_math.exceptions import ValidationError
from komorebi_math.release import gpu as gpu_module
from komorebi_math.release.gpu import GpuBackendResult, qualify_gpu


def _result(name: str, passed: bool) -> GpuBackendResult:
    return GpuBackendResult(
        backend=name,
        discovered=True,
        available=passed,
        passed=passed,
        version="test",
        device="test GPU" if passed else None,
        detail="ok" if passed else "unavailable",
        returncode=0 if passed else 3,
        diagnostics={},
    )


def test_gpu_qualification_requires_both_backends_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_module, "_worker_result", lambda backend, timeout: _result(backend, backend == "torch"))
    report = qualify_gpu()
    assert not report.passed
    assert report.certificate.diagnostics["failures"] == ("cupy",)


def test_gpu_qualification_can_make_one_backend_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_module, "_worker_result", lambda backend, timeout: _result(backend, backend == "torch"))
    assert qualify_gpu(require_cupy=False).passed


def test_gpu_qualification_rejects_nonpositive_timeout() -> None:
    with pytest.raises(ValidationError, match="timeout must be positive"):
        qualify_gpu(timeout=0)


def test_gpu_worker_parser_contains_native_process_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "backend": "torch",
        "discovered": True,
        "available": False,
        "passed": False,
        "detail": "native import aborted",
        "diagnostics": {},
    }
    monkeypatch.setattr(
        gpu_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=3221226505,
            stdout=json.dumps(payload) + "\n",
            stderr="Fatal Python error: Aborted",
        ),
    )
    result = gpu_module._worker_result("torch", timeout=5)
    assert not result.passed
    assert result.returncode == 3221226505
    assert result.detail == "native import aborted"
