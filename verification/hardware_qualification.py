"""Qualify a candidate's exact installed wheel on the selected CUDA host, offline.

The temporary environment inherits the selected interpreter's dependency sites.
It never installs into that interpreter or substitutes source-tree imports for
the wheel. Both PyTorch and CuPy CUDA execution are required for a passing run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import site
import subprocess
import sys
import tempfile
import time
from typing import Any
import zipfile


def _candidate_identity(root: Path, candidate: Path) -> dict[str, Any]:
    # Import only in the coordinator. The isolated worker needs no checkout
    # imports, including these release-tool helpers.
    if __package__:
        from .candidate_evidence import candidate_identity
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from candidate_evidence import candidate_identity
        finally:
            sys.path.pop(0)
    return candidate_identity(root, candidate)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _installed_inventory(package: Path, prefix: Path, wheel: Path) -> dict[str, Any]:
    package = package.resolve()
    prefix = prefix.resolve()
    if not package.is_relative_to(prefix):
        raise RuntimeError(f"source or foreign package import: {package}; expected inside {prefix}")
    with zipfile.ZipFile(wheel) as archive:
        expected = {
            info.filename: _sha256(archive.read(info))
            for info in archive.infolist()
            if info.filename.startswith("komorebi_math/") and not info.is_dir()
        }
    if not expected:
        raise RuntimeError("wheel contains no Komorebi package files")
    observed = {}
    for path in sorted(package.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"installed package contains a symlink: {path}")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo"):
            observed["komorebi_math/" + path.relative_to(package).as_posix()] = _sha256(path.read_bytes())
    if observed != expected:
        different = sorted(name for name in observed.keys() & expected.keys() if observed[name] != expected[name])
        raise RuntimeError(
            "installed package does not match candidate wheel: "
            f"missing={sorted(expected.keys() - observed.keys())}, "
            f"unexpected={sorted(observed.keys() - expected.keys())}, changed={different}"
        )
    return {
        "passed": True,
        "package_directory": str(package),
        "package_files": len(observed),
        "package_sha256": _sha256(json.dumps(observed, sort_keys=True, separators=(",", ":")).encode()),
    }


def _check_module_paths(prefix: Path) -> dict[str, Any]:
    prefix = prefix.resolve()
    checked = {}
    for name, module in tuple(sys.modules.items()):
        if name == "komorebi_math" or name.startswith("komorebi_math."):
            filename = getattr(module, "__file__", None)
            if filename is None or not Path(filename).resolve().is_relative_to(prefix):
                raise RuntimeError(f"source or foreign module import: {name}: {filename}")
            checked[name] = str(Path(filename).resolve())
    if not checked:
        raise RuntimeError("no installed Komorebi modules were observed")
    return {"passed": True, "modules_checked": len(checked), "module_paths": checked}


def _run_gpu_checks() -> dict[str, Any]:
    from komorebi_math.release.gpu import qualify_gpu
    from komorebi_math.verify.gpu import run_gpu_verification

    qualification = qualify_gpu(require_torch=True, require_cupy=True)
    verification = run_gpu_verification()
    return {
        "qualification": {"passed": qualification.passed, **qualification.to_dict()},
        "verification": {
            "passed": verification.passed,
            "checks": [
                {"name": check.name, "passed": check.passed, "detail": check.detail}
                for check in verification.checks
            ],
        },
    }


def _require_hardware_success(result: dict[str, Any]) -> None:
    qualification = result["qualification"]
    backends = qualification["backends"]
    names = [item["backend"] for item in backends]
    if sorted(names) != ["cupy", "torch"] or any(
        item.get("discovered") is not True or item.get("available") is not True
        or item.get("passed") is not True or item.get("returncode") != 0
        for item in backends
    ):
        raise RuntimeError("both PyTorch and CuPy CUDA workers must execute and pass")
    if qualification.get("passed") is not True or qualification["certificate"].get("valid") is not True:
        raise RuntimeError("CUDA qualification failed")
    verification = result["verification"]
    checks = verification["checks"]
    if verification.get("passed") is not True or not checks or any(item.get("passed") is not True for item in checks):
        raise RuntimeError("CUDA verification failed")


def installed_worker(wheel: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"passed": False, "python": sys.executable, "python_prefix": sys.prefix}
    try:
        if sys.flags.optimize:
            raise RuntimeError("hardware qualification requires enabled Python assertions")
        import komorebi_math

        package = Path(komorebi_math.__file__).resolve().parent
        report["package_version"] = komorebi_math.__version__
        report["installed_before"] = _installed_inventory(package, Path(sys.prefix), wheel)
        report["imports_before"] = _check_module_paths(Path(sys.prefix))
        dependency_names = ("numpy", "torch", "cupy", "cupy-cuda11x", "cupy-cuda12x", "cupy-cuda13x")
        dependencies = {}
        for name in dependency_names:
            try:
                dependencies[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                dependencies[name] = None
        report["dependencies"] = dependencies
        report["gpu"] = _run_gpu_checks()
        report["imports_after"] = _check_module_paths(Path(sys.prefix))
        report["installed_after"] = _installed_inventory(package, Path(sys.prefix), wheel)
        _require_hardware_success(report["gpu"])
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def _run(command: list[str], cwd: Path, environment: dict[str, str], report: dict[str, Any],
         *, timeout: float = 180.0, allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
    row: dict[str, Any] = {"command": command, "timeout_seconds": timeout, "cwd": str(cwd)}
    start = time.monotonic()
    try:
        result = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, text=True, timeout=timeout)
        row.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr, timed_out=False)
    except subprocess.TimeoutExpired as exc:
        def text(value: str | bytes | None) -> str:
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        row.update(returncode=None, stdout=text(exc.stdout), stderr=text(exc.stderr), timed_out=True)
        raise RuntimeError(f"hardware command exceeded {timeout} seconds") from exc
    except OSError as exc:
        row.update(returncode=None, error=f"{type(exc).__name__}: {exc}", timed_out=False)
        raise
    finally:
        row["seconds"] = round(time.monotonic() - start, 3)
        report["commands"].append(row)
    if result.returncode and not allow_failure:
        raise RuntimeError(f"hardware command failed with exit code {result.returncode}: {command}")
    return result


def qualify(root: Path, candidate: Path, *, timeout: float = 660.0) -> dict[str, Any]:
    root = root.resolve()
    candidate = candidate.resolve()
    report: dict[str, Any] = {
        "schema": "komorebi-math.hardware-qualification", "schema_version": 1,
        "passed": False, "started_at": datetime.now(timezone.utc).isoformat(),
        "host": {"platform": platform.platform(), "system": platform.system(),
                 "machine": platform.machine(), "python": platform.python_version(), "executable": sys.executable},
        "dependency_isolated": False, "dependencies_inherited": True,
        "network_access_requested": False, "required_backends": ["torch", "cupy"], "commands": [],
    }
    try:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if sys.flags.optimize:
            raise RuntimeError("hardware qualification requires enabled Python assertions")
        report["candidate"] = _candidate_identity(root, candidate)
        wheels = sorted((root / "release/dist").glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("candidate must have exactly one wheel")
        wheel = wheels[0]
        report["wheel"] = {"name": wheel.name, "sha256": _sha256(wheel.read_bytes()), "bytes": wheel.stat().st_size}
        environment = os.environ.copy()
        for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONOPTIMIZE", "PYTEST_ADDOPTS"):
            environment.pop(key, None)
        environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        parent_sites = sorted({str(Path(path).resolve()) for path in site.getsitepackages() if Path(path).is_dir()})
        report["inherited_dependency_sites"] = parent_sites
        with tempfile.TemporaryDirectory(prefix="komorebi-hardware-") as temporary:
            work = Path(temporary)
            target = work / "installed"
            # A clean venv plus explicit selected-interpreter site paths avoids
            # silently inheriting additional packages from its base interpreter.
            _run([sys.executable, "-I", "-B", "-m", "venv", str(target)], work, environment, report)
            python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            location = _run([str(python), "-I", "-B", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
                            work, environment, report).stdout.strip()
            purelib = Path(location).resolve()
            if not purelib.is_relative_to(target.resolve()):
                raise RuntimeError("temporary environment package directory escaped its prefix")
            (purelib / "komorebi_hardware_parent_dependencies.pth").write_text("\n".join(parent_sites) + "\n", encoding="utf-8")
            _run([str(python), "-I", "-B", "-m", "pip", "--isolated", "install", "--no-index", "--no-deps",
                  "--force-reinstall", "--disable-pip-version-check", str(wheel)], work, environment, report)
            # The producer file itself is covered by the candidate source binding.
            worker = root / "verification/hardware_qualification.py"
            result = _run([str(python), "-I", "-B", str(worker), "--installed-worker", str(wheel)],
                          work, environment, report, timeout=timeout, allow_failure=True)
            lines = result.stdout.strip().splitlines()
            if not lines:
                raise RuntimeError("installed hardware worker returned no JSON result")
            report["installed"] = json.loads(lines[-1])
            if result.returncode != 0 or report["installed"].get("passed") is not True:
                raise RuntimeError("installed hardware checks failed; inspect installed and command diagnostics")
            _require_hardware_success(report["installed"]["gpu"])
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if "candidate" in report:
            try:
                report["candidate_after"] = _candidate_identity(root, candidate)
                report["candidate_unchanged"] = report["candidate_after"] == report["candidate"]
                if not report["candidate_unchanged"]:
                    raise RuntimeError("candidate identity changed during hardware qualification")
            except Exception as exc:
                report["passed"] = False
                report["candidate_revalidation_error"] = f"{type(exc).__name__}: {exc}"
                report.setdefault("error", report["candidate_revalidation_error"])
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=660.0)
    parser.add_argument("--installed-worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.installed_worker is not None:
        report = installed_worker(args.installed_worker)
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 0 if report["passed"] else 1
    candidate = args.candidate or args.root / "release/CANDIDATE.json"
    output = args.output or args.root / "release/CURRENT_HARDWARE_QUALIFICATION.json"
    report = qualify(args.root, candidate, timeout=args.timeout)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(output.resolve()), "error": report.get("error")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
