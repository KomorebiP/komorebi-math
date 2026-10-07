from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import komorebi_math
from komorebi_math import PacketSystem
from komorebi_math.nonlinear import (
    PacketJetFamily,
    PolynomialDerivativeProvider,
    cupy_runtime_status,
    snapshot_packet_jet,
    validate_cupy_snapshot,
)


def _snapshot_and_probes(seed: int, probe_count: int):
    rng = random.Random(seed)
    dimension = 4
    order = 3
    tensor = np.zeros((dimension,) + (dimension,) * order, dtype=float)
    for output in range(dimension):
        for i in range(dimension):
            for j in range(dimension):
                for k in range(dimension):
                    tensor[output, i, j, k] = rng.uniform(-1.0, 1.0)
    provider = PolynomialDerivativeProvider(
        {order: tensor},
        source_dimension=dimension,
        target_dimension=dimension,
        name="block11 CUDA validation tensor",
    )
    packets = PacketSystem((np.eye(dimension),), names=("all",))
    family = PacketJetFamily(provider, np.zeros(dimension), packets, packets, order)
    snapshot = snapshot_packet_jet(family.jet("all", ("all",) * order))
    probes = tuple(
        tuple(
            np.array([rng.uniform(-2.0, 2.0) for _ in range(dimension)])
            for _ in range(order)
        )
        for _ in range(probe_count)
    )
    return snapshot, probes


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run matched CPU/CuPy packet-jet validation")
    parser.add_argument("--output", default="verification/BLOCK11_CUDA_DEVICE_VALIDATION.json")
    parser.add_argument("--seed", type=int, default=11011)
    parser.add_argument("--probes", type=int, default=64)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument(
        "--allow-unavailable",
        action="store_true",
        help="record an honest not-executed result instead of failing without CUDA",
    )
    args = parser.parse_args(argv)
    destination = Path(args.output)
    status = cupy_runtime_status()
    base: dict[str, Any] = {
        "schema": "komorebi-math.block11-cuda-device-validation",
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": komorebi_math.__version__,
        "seed": args.seed,
        "requested_probes": args.probes,
        "tolerance": args.tolerance,
        "runtime": {
            "installed": status.installed,
            "runtime_available": status.runtime_available,
            "device_count": status.device_count,
            "runtime_version": status.runtime_version,
            "error": status.error,
        },
    }
    if not status.runtime_available:
        base.update(
            {
                "status": "not_executed",
                "valid": None,
                "reason": status.error or "CuPy CUDA runtime is unavailable",
                "claim": "No device-backed validation was executed in this environment.",
            }
        )
        _write(destination, base)
        print(json.dumps(base, indent=2, sort_keys=True))
        return 0 if args.allow_unavailable else 2

    snapshot, probes = _snapshot_and_probes(args.seed, args.probes)
    result = validate_cupy_snapshot(snapshot, probes, tolerance=args.tolerance)
    base.update(
        {
            "status": "executed",
            "valid": result.certificate.valid,
            "probe_count": result.probe_count,
            "maximum_residual": result.maximum_residual,
            "certificate": result.certificate.to_dict(),
            "claim": "Matched finite CPU/CuPy evaluations were executed on an actual CUDA runtime.",
        }
    )
    _write(destination, base)
    print(json.dumps(base, indent=2, sort_keys=True))
    return 0 if result.certificate.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
