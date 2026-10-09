"""Measure bounded, independently checked CPU examples; no universal size limit."""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
import sys
import time
import tracemalloc

import mpmath as mp
import numpy as np

from komorebi_math.core.semilattice import FiniteJoinSemilattice
from komorebi_math.quadratic_forms import certify_float64_factor_spectrum, certified_positive_logsf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error("resource qualification requires enabled Python assertions")
    records = []

    def measure(name, size, operation):
        tracemalloc.start()
        wall = time.perf_counter()
        cpu = time.process_time()
        result = operation()
        cpu_elapsed = time.process_time() - cpu
        wall_elapsed = time.perf_counter() - wall
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        records.append({"operation": name, "size": size, "wall_seconds": wall_elapsed,
                        "cpu_seconds": cpu_elapsed, "python_traced_peak_bytes": peak, **result})

    for size in (4, 8, 16, 32, 64):
        def quotient_control(size=size):
            source = FiniteJoinSemilattice(tuple(tuple(i | j for j in range(size)) for i in range(size)))
            result = source.generated_congruence(((0, 1),))
            result.certificate().require()
            assert result.classes == tuple((i, i + 1) for i in range(0, size, 2))
            return {"passed": True, "quotient_size": len(result.classes)}
        measure("finite join validation and quotient", size, quotient_control)
    for size in (4, 8, 16):
        def spectrum_control(size=size):
            values = np.array([2.0 ** ((i % 5) - 2) for i in range(size)])
            result = certify_float64_factor_spectrum(np.diag(values), candidate_digits=25, interval_dps=40)
            # The spectrum bridge reports candidates in descending order.
            reference = sorted((Decimal(str(v)) ** 2 for v in values), reverse=True)
            assert result.rank == size
            assert all(Decimal(lo) <= value <= Decimal(hi)
                       for lo, hi, value in zip(result.lower, result.upper, reference, strict=True))
            return {"passed": True, "candidate_digits": result.candidate_digits, "interval_dps": result.interval_dps}
        measure("directed stored-factor spectrum", size, spectrum_control)
    for rank in (4, 8, 16):
        def tail_control(rank=rank):
            result = certified_positive_logsf(str(3 * rank), ["1"] * rank, relative_accuracy="1e-3",
                                             initial_K=64, max_K=512, initial_L="10", max_L=14,
                                             initial_interval_dps=40, max_interval_dps=60, working_dps=100)
            # An exhausted cap is an expected bounded-controller outcome.
            # Independently compare the returned interval to the finite Erlang sum.
            ctx = mp.mp.clone()
            ctx.dps = 80
            x = ctx.mpf(3 * rank) / 2
            reference = (-x + ctx.log(sum(x**k / ctx.factorial(k) for k in range(rank // 2)))) / ctx.log(10)
            lower, upper = (ctx.mpf(s.strip()) for s in result.log10_interval[1:-1].split(","))
            assert lower <= reference <= upper
            if result.certified:
                assert Decimal(result.achieved_relative_ratio_minus_one) <= Decimal("1e-3")
            else:
                assert result.failure_reason
            return {"passed": True, "certified": result.certified,
                    "failure_reason": result.failure_reason, "K": result.K,
                    "refinements": result.refinements}
        measure("bounded directed positive tail", rank, tail_control)
    report = {"status": "passed", "records": records,
              "limits": {"external_wall_timeout": "Enforced by stress_release_candidate.py; not by this standalone probe",
                         "memory_measurement": "Python allocations traced by tracemalloc; native library allocations are not a total RSS bound",
                         "interpretation": "Timings describe this machine and these finite inputs, not a universal practical size limit"}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "measured_cases": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
