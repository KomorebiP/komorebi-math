"""Test support for crash-isolated optional backend checks."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from komorebi_math.nonlinear.autodiff import ad_framework_diagnostics


def run_isolated_ad_backend_smoke(framework: str, case: str) -> None:
    status = ad_framework_diagnostics()[framework]
    if not status["available"]:
        pytest.skip(
            f"{framework} backend unavailable in an isolated import probe: {status['detail']}"
        )

    root = Path(__file__).resolve().parents[1]
    worker = root / "verification" / "optional_ad_backend_smoke.py"
    environment = os.environ.copy()
    source = str(root / "src")
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = source if not existing else source + os.pathsep + existing
    try:
        completed = subprocess.run(
            (
                sys.executable,
                str(worker),
                "--framework",
                framework,
                "--case",
                case,
            ),
            cwd=root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=180.0,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"isolated {framework} {case} smoke check timed out after {exc.timeout} seconds")
    if completed.returncode != 0:
        detail = "\n".join(part for part in (completed.stdout.strip(), completed.stderr.strip()) if part)
        pytest.fail(
            f"isolated {framework} {case} smoke check exited with code "
            f"{completed.returncode}\n{detail[-4000:]}"
        )
