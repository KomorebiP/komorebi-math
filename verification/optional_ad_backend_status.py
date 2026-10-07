"""Report optional AD backend health without importing native runtimes in-process."""

from __future__ import annotations

import json
import platform
import sys

from komorebi_math.nonlinear.autodiff import ad_framework_diagnostics


def main() -> int:
    diagnostics = ad_framework_diagnostics(refresh=True)
    print(f"python {platform.python_version()} ({sys.executable})")
    for framework in ("jax", "torch"):
        status = diagnostics[framework]
        state = "available" if status["available"] else "unavailable"
        version = f" {status['version']}" if status["version"] else ""
        code = "" if status["returncode"] is None else f" returncode={status['returncode']}"
        print(f"optional backend {framework}: {state}{version}{code}")
        if not status["available"]:
            print(f"  {status['detail']}")
    print(json.dumps(diagnostics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
