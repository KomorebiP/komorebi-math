"""Matched benchmark contracts and lightweight timing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from statistics import median
from time import perf_counter
from typing import Any, Callable, Mapping

from ..exceptions import ValidationError
from ._validation import strict_integer


class TimingScope(StrEnum):
    """Declared boundary of one timing measurement."""

    KERNEL = "kernel"
    DEVICE_RESIDENT = "device_resident"
    END_TO_END = "end_to_end"


@dataclass(frozen=True, slots=True)
class BenchmarkContract:
    """Metadata that must agree before timing results are compared."""

    case_signature: tuple[tuple[str, Any], ...]
    output_contract: str
    timing_scope: TimingScope
    backend: str

    @classmethod
    def create(
        cls,
        case_signature: Mapping[str, Any],
        output_contract: str,
        timing_scope: TimingScope | str,
        backend: str,
    ) -> "BenchmarkContract":
        if not output_contract or not backend:
            raise ValidationError("output_contract and backend must be nonempty")
        normalized_signature = tuple(
            sorted(((str(key), value) for key, value in case_signature.items()), key=lambda item: item[0])
        )
        if len({key for key, _ in normalized_signature}) != len(normalized_signature):
            raise ValidationError("case signature keys must remain unique after string conversion")
        return cls(
            normalized_signature,
            str(output_contract),
            TimingScope(timing_scope),
            str(backend),
        )

    def matches(self, other: "BenchmarkContract", *, allow_backend_difference: bool = True) -> bool:
        return (
            self.case_signature == other.case_signature
            and self.output_contract == other.output_contract
            and self.timing_scope is other.timing_scope
            and (allow_backend_difference or self.backend == other.backend)
        )


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Repeated timing samples and the value returned by the last run."""

    contract: BenchmarkContract
    samples_seconds: tuple[float, ...]
    median_seconds: float
    minimum_seconds: float
    maximum_seconds: float
    value: Any


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    """A matched baseline and candidate with their median speed ratio."""

    baseline: BenchmarkResult
    candidate: BenchmarkResult
    speedup: float


def benchmark_callable(
    function: Callable[[], Any],
    contract: BenchmarkContract,
    *,
    warmups: int = 2,
    repeats: int = 7,
    synchronize: Callable[[], None] | None = None,
) -> BenchmarkResult:
    """Measure a callable under a declared benchmark contract."""

    warmup_count = strict_integer(warmups, "warmups", minimum=0)
    repeat_count = strict_integer(repeats, "repeats", minimum=1)
    value: Any = None
    for _ in range(warmup_count):
        value = function()
        if synchronize is not None:
            synchronize()
    samples: list[float] = []
    for _ in range(repeat_count):
        if synchronize is not None:
            synchronize()
        start = perf_counter()
        value = function()
        if synchronize is not None:
            synchronize()
        samples.append(perf_counter() - start)
    if any(not isfinite(sample) or sample < 0 for sample in samples):
        raise ValidationError("benchmark produced an invalid timing sample")
    return BenchmarkResult(
        contract,
        tuple(samples),
        float(median(samples)),
        float(min(samples)),
        float(max(samples)),
        value,
    )


def compare_benchmarks(
    baseline: BenchmarkResult,
    candidate: BenchmarkResult,
    *,
    allow_backend_difference: bool = True,
) -> BenchmarkComparison:
    """Compare two benchmark results only when their contracts match."""

    if not baseline.contract.matches(candidate.contract, allow_backend_difference=allow_backend_difference):
        raise ValidationError("benchmark contracts do not match")
    if candidate.median_seconds <= 0:
        speedup = float("inf")
    else:
        speedup = baseline.median_seconds / candidate.median_seconds
    return BenchmarkComparison(baseline, candidate, float(speedup))


__all__ = [
    "BenchmarkComparison",
    "BenchmarkContract",
    "BenchmarkResult",
    "TimingScope",
    "benchmark_callable",
    "compare_benchmarks",
]
