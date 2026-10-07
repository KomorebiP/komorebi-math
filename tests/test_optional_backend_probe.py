from __future__ import annotations

import subprocess

import pytest

from komorebi_math.exceptions import BackendUnavailableError
from komorebi_math.nonlinear import autodiff as autodiff_module
from komorebi_math.nonlinear.autodiff import TorchDerivativeProvider


def _clear_probe_cache() -> None:
    autodiff_module._probe_ad_framework.cache_clear()


def test_isolated_probe_accepts_required_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_probe_cache()
    monkeypatch.setattr(autodiff_module, "find_spec", lambda name: object())
    monkeypatch.setattr(
        autodiff_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout='{"available": true, "version": "test", "detail": "capability present"}\n',
            stderr="",
        ),
    )
    diagnostics = autodiff_module.ad_framework_diagnostics(refresh=True)
    assert diagnostics["jax"]["available"]
    assert diagnostics["torch"]["available"]
    assert diagnostics["torch"]["version"] == "test"
    _clear_probe_cache()


def test_isolated_probe_converts_native_abort_to_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_probe_cache()
    monkeypatch.setattr(autodiff_module, "find_spec", lambda name: object())
    monkeypatch.setattr(
        autodiff_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=3221226505,
            stdout="",
            stderr="Fatal Python error: Aborted",
        ),
    )
    diagnostics = autodiff_module.ad_framework_diagnostics(refresh=True)
    assert diagnostics["torch"]["discovered"]
    assert not diagnostics["torch"]["available"]
    assert diagnostics["torch"]["returncode"] == 3221226505
    assert "Aborted" in diagnostics["torch"]["detail"]
    assert autodiff_module.available_ad_frameworks()["torch"] is False
    _clear_probe_cache()


def test_isolated_probe_converts_timeout_to_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_probe_cache()
    monkeypatch.setattr(autodiff_module, "find_spec", lambda name: object())

    def timeout(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="python", timeout=60.0)

    monkeypatch.setattr(autodiff_module.subprocess, "run", timeout)
    diagnostics = autodiff_module.ad_framework_diagnostics(refresh=True)
    assert not diagnostics["jax"]["available"]
    assert "exceeded" in diagnostics["jax"]["detail"]
    _clear_probe_cache()


def test_provider_guard_does_not_attempt_in_process_import_when_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = autodiff_module._ADFrameworkProbe(
        framework="torch",
        discovered=True,
        available=False,
        returncode=1,
        detail="isolated import failed",
    )
    monkeypatch.setattr(TorchDerivativeProvider, "available", staticmethod(lambda: False))
    monkeypatch.setattr(autodiff_module, "_probe_ad_framework", lambda framework: status)

    def forbidden_import(name: str) -> object:
        raise AssertionError(f"unexpected in-process import of {name}")

    monkeypatch.setattr(autodiff_module, "import_module", forbidden_import)
    provider = TorchDerivativeProvider(lambda x: x, 1, 1)
    with pytest.raises(BackendUnavailableError, match="unavailable"):
        provider.value([1.0])
