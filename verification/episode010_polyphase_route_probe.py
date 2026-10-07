"""Deterministic finite audit for R4 Block 14 additive polyphase route packets."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from komorebi_math.signals.filterbanks import HaarFilterBank
from komorebi_math.signals.route_packets import PolyphaseRoutePacket


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trim(values: np.ndarray) -> tuple[int, ...]:
    result = [int(value) for value in np.asarray(values, dtype=object)]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return tuple(result)


def raw_route_aggregate(filters: tuple[tuple[int, ...], ...], order: int) -> tuple[dict[tuple[int, int, int], int], int]:
    """Enumerate tap histories and aggregate by (target, source, coarse lag)."""

    aggregate: dict[tuple[int, int, int], int] = {}
    raw_count = 0
    tap_ranges = [range(len(kernel)) for kernel in filters]
    for source in range(order):
        for taps in itertools.product(*tap_ranges):
            raw_count += 1
            total_lag = sum(taps)
            target = (source + total_lag) % order
            coarse_lag = (source + total_lag - target) // order
            weight = math.prod(filters[index][tap] for index, tap in enumerate(taps))
            key = (target, source, coarse_lag)
            aggregate[key] = aggregate.get(key, 0) + weight
    return aggregate, raw_count


def compare_aggregate(packet: PolyphaseRoutePacket, aggregate: dict[tuple[int, int, int], int]) -> bool:
    for target in range(packet.order):
        for source in range(packet.order):
            maximum = max(
                (q for r, s, q in aggregate if r == target and s == source),
                default=0,
            )
            length = max(maximum + 1, packet.kernels[target][source].size)
            for coarse_lag in range(length):
                if packet.coefficient(target, source, coarse_lag) != aggregate.get(
                    (target, source, coarse_lag), 0
                ):
                    return False
    return True


def run(seed: int, trials: int) -> dict[str, object]:
    started = time.monotonic()
    rng = np.random.default_rng(seed)
    failures = 0
    compression_ratios: list[float] = []

    for _ in range(trials):
        order = int(rng.integers(2, 6))
        depth = int(rng.integers(2, 6))
        filters: list[tuple[int, ...]] = []
        for _stage in range(depth):
            length = int(rng.integers(2, 7))
            coefficients = rng.integers(-3, 4, size=length)
            if not np.any(coefficients):
                coefficients[0] = 1
            filters.append(tuple(int(value) for value in coefficients))

        packet = PolyphaseRoutePacket.from_filter(np.asarray(filters[0], dtype=object), order)
        combined = np.asarray(filters[0], dtype=object)
        for kernel in filters[1:]:
            packet = packet.compose(PolyphaseRoutePacket.from_filter(np.asarray(kernel, dtype=object), order))
            combined = np.convolve(combined, np.asarray(kernel, dtype=object))

        direct = PolyphaseRoutePacket.from_filter(combined, order)
        if not np.array_equal(packet.reconstruct_filter(), combined):
            failures += 1
            break
        for target in range(order):
            for source in range(order):
                if trim(packet.kernels[target][source]) != trim(direct.kernels[target][source]):
                    failures += 1
                    break

        aggregate, raw_count = raw_route_aggregate(tuple(filters), order)
        if not compare_aggregate(packet, aggregate):
            failures += 1
            break
        nonzero = sum(value != 0 for value in aggregate.values())
        compression_ratios.append(raw_count / max(1, nonzero))

    deep_filters = tuple([(1, 2, -1, 3, 1)] * 6)
    deep_order = 4
    deep_packet = PolyphaseRoutePacket.from_filter(np.asarray(deep_filters[0], dtype=object), deep_order)
    for kernel in deep_filters[1:]:
        deep_packet = deep_packet.compose(
            PolyphaseRoutePacket.from_filter(np.asarray(kernel, dtype=object), deep_order)
        )
    deep_aggregate, deep_raw_count = raw_route_aggregate(deep_filters, deep_order)
    deep_nonzero = sum(value != 0 for value in deep_aggregate.values())
    deep_exact = compare_aggregate(deep_packet, deep_aggregate)

    haar = HaarFilterBank(normalized=True)
    signal = np.array([1.0, 0.0])
    low, high = haar.analyze(signal)
    scale = math.sqrt(2.0)
    even = (float(low[0] / scale), float(high[0] / scale))
    odd = (float(low[0] / scale), float(-high[0] / scale))

    return {
        "schema": "komorebi-math.episode010-polyphase-route-probe",
        "schema_version": 1,
        "seed": seed,
        "random_trials": trials,
        "composition_or_aggregation_failures": failures,
        "compression_ratio": {
            "minimum": float(min(compression_ratios)),
            "median": float(np.median(compression_ratios)),
            "maximum": float(max(compression_ratios)),
        },
        "deep_example": {
            "order": deep_order,
            "depth": len(deep_filters),
            "filter_length": len(deep_filters[0]),
            "raw_source_route_histories": deep_raw_count,
            "nonzero_aggregated_polyphase_coefficients": deep_nonzero,
            "compression_ratio": deep_raw_count / deep_nonzero,
            "exact_match": deep_exact,
            "combined_kernel_length": deep_packet.kernel_length,
        },
        "haar_carrier_counterexample": {
            "input": signal.tolist(),
            "even_contributions": even,
            "even_additive_result": sum(even),
            "even_max_result": max(even),
            "odd_contributions": odd,
            "odd_additive_result": sum(odd),
            "odd_max_result": max(odd),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "runtime_seconds": round(time.monotonic() - started, 6),
        "passed": failures == 0 and deep_exact,
        "non_claims": [
            "Finite random trials do not prove the polyphase composition theorem.",
            "Compression ratios are case-specific and are not universal asymptotic rates.",
            "The probe does not establish novelty of the observer-plus-carrier formulation.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--output", type=Path, default=ROOT / "verification/EPISODE010_POLYPHASE_ROUTE_PROBE.json")
    args = parser.parse_args()
    result = run(args.seed, args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
