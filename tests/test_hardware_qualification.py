"""Fail-closed hardware handoff controls, runnable without CUDA hardware."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType
import zipfile

import pytest

from verification import hardware_qualification as gate


def _hardware_result() -> dict:
    return {
        "qualification": {
            "passed": True, "certificate": {"valid": True},
            "backends": [
                {"backend": name, "discovered": True, "available": True, "passed": True, "returncode": 0}
                for name in ("torch", "cupy")
            ],
        },
        "verification": {"passed": True, "checks": [{"name": "device equivalence", "passed": True}]},
    }


def _wheel(tmp_path: Path) -> Path:
    wheel = tmp_path / "candidate.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("komorebi_math/__init__.py", b"VERSION = 'fixture'\n")
    return wheel


def test_source_import_is_rejected_even_when_bytes_match(tmp_path: Path) -> None:
    source = tmp_path / "checkout/komorebi_math"
    source.mkdir(parents=True)
    (source / "__init__.py").write_bytes(b"VERSION = 'fixture'\n")
    with pytest.raises(RuntimeError, match="source or foreign package import"):
        gate._installed_inventory(source, tmp_path / "venv", _wheel(tmp_path))


def test_installed_package_bytes_must_match_wheel(tmp_path: Path) -> None:
    prefix = tmp_path / "venv"
    package = prefix / "site-packages/komorebi_math"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"VERSION = 'fixture'\n")
    wheel = _wheel(tmp_path)
    assert gate._installed_inventory(package, prefix, wheel)["passed"]
    (package / "__init__.py").write_text("VERSION = 'other build'\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="installed package does not match"):
        gate._installed_inventory(package, prefix, wheel)


def test_loaded_module_outside_environment_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(sys.modules):
        if name == "komorebi_math" or name.startswith("komorebi_math."):
            monkeypatch.delitem(sys.modules, name)
    root = ModuleType("komorebi_math")
    root.__file__ = str(tmp_path / "venv/site-packages/komorebi_math/__init__.py")
    monkeypatch.setitem(sys.modules, root.__name__, root)
    module = ModuleType("komorebi_math.hardware_test_leak")
    module.__file__ = str(tmp_path / "checkout/komorebi_math/leak.py")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(RuntimeError, match="hardware_test_leak"):
        gate._check_module_paths(tmp_path / "venv")


@pytest.mark.parametrize("case", ["missing_backend", "unavailable", "native_abort", "failed_verifier", "empty_verifier"])
def test_nested_gpu_failures_cannot_be_masked_by_top_level_pass(case: str) -> None:
    result = _hardware_result()
    if case == "missing_backend":
        result["qualification"]["backends"].pop()
    elif case == "unavailable":
        result["qualification"]["backends"][1]["available"] = False
    elif case == "native_abort":
        result["qualification"]["backends"][0]["returncode"] = 3221226505
    elif case == "failed_verifier":
        result["verification"]["checks"][0]["passed"] = False
    else:
        result["verification"]["checks"] = []
    with pytest.raises(RuntimeError):
        gate._require_hardware_success(result)


def test_failure_to_verify_candidate_stops_before_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def rejected(*args):
        raise ValueError("candidate wheel checksum mismatch")
    monkeypatch.setattr(gate, "_candidate_identity", rejected)
    report = gate.qualify(tmp_path, tmp_path / "candidate.json")
    assert not report["passed"]
    assert not report["commands"]
    assert "checksum mismatch" in report["error"]


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_timeouts_fail_before_launch(tmp_path: Path, timeout: float) -> None:
    report = gate.qualify(tmp_path, tmp_path / "candidate.json", timeout=timeout)
    assert not report["passed"]
    assert not report["commands"]
    assert "finite and positive" in report["error"]


def test_timeout_retains_partial_command_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1, output=b"worker started", stderr=b"driver stalled")
    monkeypatch.setattr(gate.subprocess, "run", timeout)
    report = {"commands": []}
    with pytest.raises(RuntimeError, match="exceeded"):
        gate._run(["test-worker"], tmp_path, {}, report, timeout=1)
    row = report["commands"][0]
    assert row["timed_out"] and row["returncode"] is None
    assert row["stdout"] == "worker started" and row["stderr"] == "driver stalled"


def test_native_abort_retains_exit_and_error_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 3221226505, "started", "native import aborted"))
    report = {"commands": []}
    with pytest.raises(RuntimeError, match="3221226505"):
        gate._run(["test-worker"], tmp_path, {}, report)
    assert report["commands"][0]["returncode"] == 3221226505
    assert report["commands"][0]["stderr"] == "native import aborted"


@pytest.mark.parametrize("worker_case", ["unavailable", "native_abort", "identity_changed", "success"])
def test_coordinator_preserves_failure_and_rechecks_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                           worker_case: str) -> None:
    dist = tmp_path / "release/dist"
    dist.mkdir(parents=True)
    _wheel(dist)
    identity = {"revision": "1" * 40, "manifest_sha256": "a" * 64, "release_binding": {}}
    identity_calls = []

    def candidate(*args):
        identity_calls.append(True)
        value = deepcopy(identity)
        if worker_case == "identity_changed" and len(identity_calls) > 1:
            value["manifest_sha256"] = "b" * 64
        return value

    def command(args, cwd, environment, report, **kwargs):
        for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONOPTIMIZE"):
            assert key not in environment
        if "venv" in args:
            (Path(args[-1]) / "site-packages").mkdir(parents=True)
        if "-c" in args:
            output = str(cwd / "installed/site-packages") + "\n"
        elif "--installed-worker" in args:
            assert "-I" in args
            if worker_case == "native_abort":
                return subprocess.CompletedProcess(args, 3221226505, "", "native abort")
            result = {"passed": True, "gpu": _hardware_result()}
            if worker_case == "unavailable":
                result["passed"] = False
                result["error"] = "CuPy CUDA unavailable"
                result["gpu"]["qualification"]["backends"][1]["available"] = False
            return subprocess.CompletedProcess(args, 0 if result["passed"] else 1, json.dumps(result), "")
        else:
            output = ""
        return subprocess.CompletedProcess(args, 0, output, "")

    monkeypatch.setattr(gate, "_candidate_identity", candidate)
    monkeypatch.setattr(gate, "_run", command)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "checkout/src"))
    monkeypatch.setenv("PYTHONOPTIMIZE", "1")
    report = gate.qualify(tmp_path, tmp_path / "candidate.json")
    assert len(identity_calls) == 2
    assert report["passed"] is (worker_case == "success")
    if worker_case == "unavailable":
        assert report["installed"]["error"] == "CuPy CUDA unavailable"
        assert report["candidate_unchanged"]
    elif worker_case == "native_abort":
        assert "no JSON" in report["error"]
    elif worker_case == "identity_changed":
        assert "changed" in report["candidate_revalidation_error"]


def test_failure_report_replaces_stale_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "hardware.json"
    output.write_text('{"passed": true}', encoding="utf-8")
    monkeypatch.setattr(gate, "qualify", lambda *args, **kwargs: {"passed": False, "error": "missing CUDA"})
    assert gate.main(["--root", str(tmp_path), "--output", str(output)]) == 1
    assert json.loads(output.read_text(encoding="utf-8")) == {"passed": False, "error": "missing CUDA"}
