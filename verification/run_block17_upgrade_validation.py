"""Install predecessor wheels and verify their records under the 1.0 beta."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any

TARGET_VERSION = "1.0.0b1"
PREDECESSORS = ("0.15.0a2", "0.16.0a2")


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _venv_python(environment: Path) -> Path:
    windows = environment / "Scripts" / "python.exe"
    return windows if windows.is_file() else environment / "bin" / "python"


def _run(command: list[str], *, cwd: Path, env: dict[str, str], timeout: float = 240.0) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "passed": completed.returncode == 0,
    }


def validate(project: Path, *, python: str = sys.executable) -> dict[str, Any]:
    project = project.resolve()
    current_wheels = sorted((project / "release" / "dist").glob(f"komorebi_math-{TARGET_VERSION}-*.whl"))
    if len(current_wheels) != 1:
        raise RuntimeError(f"expected one {TARGET_VERSION} wheel, found {len(current_wheels)}")
    current = current_wheels[0]
    predecessor_directory = project / "release" / "predecessors"
    records: list[dict[str, Any]] = []

    worker = r'''
import json
from pathlib import Path
import komorebi_math
expected = __import__('os').environ['KM_EXPECTED_VERSION']
assert komorebi_math.__version__ == expected, (komorebi_math.__version__, expected)
if expected == '1.0.0b1':
    from komorebi_math.release import check_upgrade_contract
    from komorebi_math.export.persistence import load_portable
    report = check_upgrade_contract()
    report.certificate.require()
    data = Path(komorebi_math.__file__).resolve().parent / 'data'
    loaded = []
    for fixture in report.contract.fixtures:
        source = data / fixture.filename
        value = load_portable(source)
        loaded.append({
            'name': fixture.name,
            'type': f'{type(value).__module__}.{type(value).__qualname__}',
        })
    assert len(loaded) == len(report.contract.fixtures) == 7
    print(json.dumps({'version': expected, 'fixtures': loaded, 'upgrade_checks': len(report.checks)}, sort_keys=True))
else:
    print(json.dumps({'version': expected}, sort_keys=True))
'''

    for predecessor in PREDECESSORS:
        candidates = sorted(predecessor_directory.glob(f"komorebi_math-{predecessor}-*.whl"))
        if len(candidates) != 1:
            raise RuntimeError(f"expected one predecessor wheel for {predecessor}, found {len(candidates)}")
        predecessor_wheel = candidates[0]
        with tempfile.TemporaryDirectory(prefix=f"komorebi-upgrade-{predecessor}-") as temporary:
            environment_path = Path(temporary) / "venv"
            create = _run(
                [python, "-m", "venv", "--system-site-packages", str(environment_path)],
                cwd=project,
                env=dict(os.environ),
            )
            environment_python = _venv_python(environment_path)
            inherited_env = dict(os.environ)
            inherited_env.pop("PYTHONPATH", None)
            predecessor_install = _run(
                [str(environment_python), "-m", "pip", "install", "--no-deps", "--force-reinstall", str(predecessor_wheel)],
                cwd=project,
                env=inherited_env,
            )
            predecessor_env = dict(inherited_env, KM_EXPECTED_VERSION=predecessor)
            predecessor_import = _run(
                [str(environment_python), "-c", worker],
                cwd=Path(temporary),
                env=predecessor_env,
            )
            target_install = _run(
                [str(environment_python), "-m", "pip", "install", "--no-deps", "--force-reinstall", str(current)],
                cwd=project,
                env=inherited_env,
            )
            target_env = dict(inherited_env, KM_EXPECTED_VERSION=TARGET_VERSION)
            target_import = _run(
                [str(environment_python), "-c", worker],
                cwd=Path(temporary),
                env=target_env,
            )
            steps = {
                "create_environment": create,
                "install_predecessor": predecessor_install,
                "import_predecessor": predecessor_import,
                "install_target": target_install,
                "import_target_and_records": target_import,
            }
            records.append(
                {
                    "predecessor": predecessor,
                    "predecessor_wheel": {
                        "name": predecessor_wheel.name,
                        "size": predecessor_wheel.stat().st_size,
                        "sha256": _hash(predecessor_wheel),
                    },
                    "target_wheel": {
                        "name": current.name,
                        "size": current.stat().st_size,
                        "sha256": _hash(current),
                    },
                    "passed": all(step["passed"] for step in steps.values()),
                    "steps": steps,
                }
            )

    payload: dict[str, Any] = {
        "schema": "komorebi-math.block17-upgrade-validation",
        "schema_version": 1,
        "target_version": TARGET_VERSION,
        "passed": all(record["passed"] for record in records),
        "records": records,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["sha256"] = sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    payload = validate(Path(args.project), python=args.python)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8", newline="\n")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
