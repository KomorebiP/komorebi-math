from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import komorebi_math
from komorebi_math.nonlinear import cupy_runtime_status


def _git(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *command],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _module_status(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"installed": False, "version": None}
    try:
        module = __import__(name)
        version = getattr(module, "__version__", None)
    except Exception as exc:
        return {
            "installed": True,
            "version": None,
            "import_error": f"{type(exc).__name__}: {exc}",
        }
    return {"installed": True, "version": None if version is None else str(version)}


def build_report() -> dict[str, Any]:
    cupy = cupy_runtime_status()
    commands = {
        name: shutil.which(name)
        for name in ("git", "pwsh", "powershell", "ruff", "mypy", "mkdocs", "latexmk")
    }
    return {
        "schema": "komorebi-math.block11-environment-audit",
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": komorebi_math.__version__,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "modules": {
            name: _module_status(name)
            for name in (
                "numpy",
                "sympy",
                "jax",
                "torch",
                "cupy",
                "pytest",
                "ruff",
                "mypy",
                "mkdocs",
            )
        },
        "commands": commands,
        "cupy_runtime": {
            "installed": cupy.installed,
            "runtime_available": cupy.runtime_available,
            "device_count": cupy.device_count,
            "runtime_version": cupy.runtime_version,
            "error": cupy.error,
        },
        "git": {
            "commit": _git(["rev-parse", "HEAD"]),
            "branch": _git(["branch", "--show-current"]),
            "status_porcelain": _git(["status", "--porcelain"]),
        },
        "execution_scope": {
            "cross_platform_matrix_executed_here": False,
            "windows_executed_here": platform.system() == "Windows",
            "cuda_device_validation_executed_here": cupy.runtime_available,
            "hosted_ci_executed_here": False,
            "note": (
                "This report records only the current process. Workflow definitions are not "
                "evidence that hosted jobs or other operating systems executed."
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record the actual Block 11 release environment")
    parser.add_argument(
        "--output",
        default="verification/BLOCK11_ENVIRONMENT_AUDIT.json",
    )
    args = parser.parse_args(argv)
    report = build_report()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
