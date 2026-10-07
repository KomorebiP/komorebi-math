from __future__ import annotations

import pytest

from komorebi_math.algorithms import (
    Task,
    amortization_threshold,
    batched_launch_count,
    break_even_analysis,
    schedule_unlimited,
    shape_classes,
    verify_schedule,
)
from komorebi_math.exceptions import ValidationError


def test_unlimited_schedule_and_critical_path() -> None:
    tasks = [
        Task("a", 2.0, shape_key=(4, 4)),
        Task("b", 3.0, shape_key=(4, 4)),
        Task("c", 5.0, ("a",), shape_key=(8, 8)),
        Task("d", 1.0, ("a", "b"), shape_key=(4, 4)),
        Task("e", 2.0, ("c", "d"), shape_key=(8, 8)),
    ]
    schedule = schedule_unlimited(tasks)
    assert verify_schedule(tasks, schedule)
    assert schedule.makespan == 9.0
    assert schedule.critical_path == ("a", "c", "e")
    assert schedule.levels == (("a", "b"), ("c", "d"), ("e",))


def test_schedule_rejects_cycle() -> None:
    with pytest.raises(ValidationError):
        schedule_unlimited([Task("a", dependencies=("b",)), Task("b", dependencies=("a",))])


def test_task_rejects_nonstring_dependency() -> None:
    with pytest.raises(ValidationError):
        Task("a", dependencies=(1,))  # type: ignore[arg-type]


def test_task_and_amortization_reject_nonfinite_times() -> None:
    with pytest.raises(ValidationError):
        Task("a", duration=float("nan"))
    with pytest.raises(ValidationError):
        amortization_threshold(float("inf"), 2.0, 1.0)


def test_shape_classes_and_batching() -> None:
    tasks = [Task("a", shape_key=(4, 4)), Task("b", shape_key=(4, 4)), Task("c", shape_key=(8, 8))]
    groups = shape_classes(tasks)
    assert len(groups[(4, 4)]) == 2
    assert batched_launch_count(tasks) == 2
    assert batched_launch_count(tasks, batching_available=False) == 3


def test_amortization() -> None:
    assert amortization_threshold(10.0, 4.0, 2.0) == 6
    assert amortization_threshold(1.0, 2.0, 3.0) is None
    result = break_even_analysis(10.0, 4.0, 2.0)
    assert result.reuse_count == 6
    assert result.asymptotic_gain == 2.0

def test_task_rejects_bad_name_and_shape_key() -> None:
    with pytest.raises(ValidationError):
        Task("   ")
    with pytest.raises(ValidationError):
        Task("a", shape_key=[1, 2])

