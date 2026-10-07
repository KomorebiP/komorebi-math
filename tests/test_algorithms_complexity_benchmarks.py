from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.algorithms import (
    BenchmarkContract,
    PacketLayout,
    TimingScope,
    arithmetic_intensity,
    benchmark_callable,
    compare_benchmarks,
    dense_apply_flops,
    packet_apply_flops,
    packet_cost_estimate,
    roofline_time_lower_bound,
)
from komorebi_math.exceptions import ValidationError


def test_complexity_counts() -> None:
    source = PacketLayout((2, 3))
    target = PacketLayout((4, 1))
    support = {(0, 0), (1, 1)}
    assert dense_apply_flops(5, 5, 2) == 100
    expected = 2 * 2 * (4 * 2 + 1 * 3)
    assert packet_apply_flops(source, target, support, 2) == expected
    estimate = packet_cost_estimate(source, target, support, rhs=2, itemsize=8)
    assert estimate.flops == expected
    assert estimate.streaming_bytes > 0
    assert np.isclose(estimate.arithmetic_intensity, arithmetic_intensity(estimate.flops, estimate.streaming_bytes))


def test_complexity_normalizes_support_and_handles_zero_operator() -> None:
    layout = PacketLayout((2, 1))
    duplicated = [(0, 0), (0, 0)]
    assert packet_apply_flops(layout, layout, duplicated) == packet_apply_flops(
        layout, layout, {(0, 0)}
    )
    zero = packet_cost_estimate(layout, layout, ())
    assert zero.flops == 0
    assert zero.storage_entries == 0
    assert zero.streaming_bytes == 0
    assert zero.arithmetic_intensity == 0.0
    with pytest.raises(ValidationError):
        packet_apply_flops(layout, layout, {(2, 0)})


def test_roofline_bound() -> None:
    assert roofline_time_lower_bound(100.0, 1000.0, 100.0, 200.0) == 5.0


def test_matched_benchmark_contracts() -> None:
    contract_cpu = BenchmarkContract.create(
        {"n": 16, "rhs": 2},
        "dense array",
        TimingScope.END_TO_END,
        "numpy",
    )
    contract_other = BenchmarkContract.create(
        {"n": 16, "rhs": 2},
        "dense array",
        TimingScope.END_TO_END,
        "other",
    )
    baseline = benchmark_callable(lambda: sum(range(100)), contract_cpu, warmups=0, repeats=2)
    candidate = benchmark_callable(lambda: sum(range(10)), contract_other, warmups=0, repeats=2)
    comparison = compare_benchmarks(baseline, candidate)
    assert comparison.speedup > 0


def test_benchmark_contract_rejects_ambiguous_string_keys() -> None:
    with pytest.raises(ValidationError):
        BenchmarkContract.create({1: "a", "1": "b"}, "vector", "kernel", "numpy")
    with pytest.raises(ValidationError):
        benchmark_callable(lambda: 1, BenchmarkContract.create({}, "x", "kernel", "numpy"), repeats=1.5)


def test_benchmark_mismatch_rejected() -> None:
    first = BenchmarkContract.create({"n": 4}, "vector", "kernel", "numpy")
    second = BenchmarkContract.create({"n": 5}, "vector", "kernel", "numpy")
    baseline = benchmark_callable(lambda: 1, first, warmups=0, repeats=1)
    candidate = benchmark_callable(lambda: 1, second, warmups=0, repeats=1)
    with pytest.raises(ValidationError):
        compare_benchmarks(baseline, candidate)

def test_complexity_strict_integer_validation() -> None:
    layout = PacketLayout((1, 1))
    with pytest.raises(ValidationError):
        dense_apply_flops(2.5, 2, 1)
    with pytest.raises(ValidationError):
        packet_apply_flops(layout, layout, {(0, 0)}, rhs=1.5)
    with pytest.raises(ValidationError):
        packet_cost_estimate(layout, layout, {(0, 0)}, itemsize=8.0)

