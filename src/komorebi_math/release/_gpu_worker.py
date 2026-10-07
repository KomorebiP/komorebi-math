"""Private isolated worker for :mod:`komorebi_math.release.gpu`."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from typing import Any


def _emit(payload: dict[str, Any], returncode: int) -> int:
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return returncode


def _torch() -> tuple[dict[str, Any], int]:
    if importlib.util.find_spec("torch") is None:
        return ({"backend": "torch", "discovered": False, "available": False, "passed": False, "detail": "torch is not installed", "diagnostics": {}}, 2)
    try:
        import torch

        if not torch.cuda.is_available():
            return ({"backend": "torch", "discovered": True, "available": False, "passed": False, "version": torch.__version__, "detail": "torch.cuda.is_available() is false", "diagnostics": {"compiled_cuda": torch.version.cuda}}, 3)
        device = torch.cuda.get_device_name(0)
        x = torch.tensor(2.0, device="cuda")
        value, tangent = torch.func.jvp(lambda z: z**3, (x,), (torch.ones_like(x),))
        matrix = torch.arange(16, dtype=torch.float64, device="cuda").reshape(4, 4)
        vector = torch.arange(4, dtype=torch.float64, device="cuda")
        product = matrix @ vector
        torch.cuda.synchronize()
        observed = (float(value.item()), float(tangent.item()))
        expected_product = (14.0, 38.0, 62.0, 86.0)
        passed = observed == (8.0, 12.0) and tuple(float(item) for item in product.cpu().tolist()) == expected_product
        payload = {
            "backend": "torch",
            "discovered": True,
            "available": True,
            "passed": passed,
            "version": torch.__version__,
            "device": device,
            "detail": "CUDA JVP and matrix product passed" if passed else "CUDA numerical result mismatch",
            "diagnostics": {
                "compiled_cuda": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
                "capability": list(torch.cuda.get_device_capability(0)),
                "jvp": list(observed),
                "matrix_product": [float(item) for item in product.cpu().tolist()],
            },
        }
        return payload, 0 if passed else 4
    except Exception as exc:
        return ({"backend": "torch", "discovered": True, "available": False, "passed": False, "detail": f"{type(exc).__name__}: {exc}", "diagnostics": {}}, 5)


def _cupy() -> tuple[dict[str, Any], int]:
    if importlib.util.find_spec("cupy") is None:
        return ({"backend": "cupy", "discovered": False, "available": False, "passed": False, "detail": "cupy is not installed", "diagnostics": {}}, 2)
    try:
        import cupy as cp
        import numpy as np

        from komorebi_math.algorithms import PacketIR, PacketLayout, packet_executor

        count = int(cp.cuda.runtime.getDeviceCount())
        if count < 1:
            return ({"backend": "cupy", "discovered": True, "available": False, "passed": False, "version": cp.__version__, "detail": "CuPy reported no CUDA devices", "diagnostics": {"device_count": count}}, 3)
        properties = cp.cuda.runtime.getDeviceProperties(0)
        raw_name = properties.get("name", b"CUDA device 0")
        device = raw_name.decode(errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
        layout = PacketLayout((2, 2))
        ir = PacketIR.square(layout, {(0, 0): np.eye(2), (1, 1): 2 * np.eye(2)})
        executor = packet_executor(ir, "cupy")
        vector = cp.arange(4, dtype=cp.float64)
        result = executor.apply(vector, return_host=False, input_on_device=True)
        executor.synchronize()
        observed = cp.asnumpy(result.value)
        expected = np.asarray([0.0, 1.0, 4.0, 6.0])
        residual = float(np.linalg.norm(observed - expected))
        passed = result.backend == "cupy" and residual <= 1e-12
        payload = {
            "backend": "cupy",
            "discovered": True,
            "available": True,
            "passed": passed,
            "version": cp.__version__,
            "device": device,
            "detail": "Komorebi CuPy packet executor passed" if passed else "CuPy packet executor mismatch",
            "diagnostics": {
                "device_count": count,
                "backend": result.backend,
                "observed": observed.tolist(),
                "residual": residual,
                "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
                "driver_version": int(cp.cuda.runtime.driverGetVersion()),
            },
        }
        return payload, 0 if passed else 4
    except Exception as exc:
        return ({"backend": "cupy", "discovered": True, "available": False, "passed": False, "detail": f"{type(exc).__name__}: {exc}", "diagnostics": {}}, 5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("torch", "cupy"), required=True)
    args = parser.parse_args(argv)
    payload, returncode = _torch() if args.backend == "torch" else _cupy()
    return _emit(payload, returncode)


if __name__ == "__main__":
    raise SystemExit(main())
