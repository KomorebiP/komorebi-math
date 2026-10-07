"""Finite task graph scheduling and amortization models."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Hashable, Iterable, Sequence

from ..exceptions import ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._validation import finite_float


@dataclass(frozen=True, slots=True)
class Task:
    """One task in a finite dependency graph."""

    name: str
    duration: float = 1.0
    dependencies: tuple[str, ...] = ()
    shape_key: Hashable | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValidationError("a task must have a nonempty string name")
        duration = finite_float(self.duration, "task duration", minimum=0.0)
        object.__setattr__(self, "duration", duration)
        if any(not isinstance(dependency, str) or not dependency.strip() for dependency in self.dependencies):
            raise ValidationError("task dependencies must be nonempty strings")
        if self.name in self.dependencies:
            raise ValidationError("a task cannot depend on itself")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValidationError("task dependencies must be unique")
        try:
            hash(self.shape_key)
        except TypeError as exc:
            raise ValidationError("shape_key must be hashable") from exc


@dataclass(frozen=True, slots=True)
class ScheduledTask:
    """Earliest start and finish data for one scheduled task."""

    name: str
    start: float
    finish: float
    level: int


@dataclass(frozen=True, slots=True)
class TaskSchedule:
    """Unlimited concurrency schedule and one critical path."""

    tasks: tuple[ScheduledTask, ...]
    makespan: float
    critical_path: tuple[str, ...]
    levels: tuple[tuple[str, ...], ...]

    def by_name(self) -> dict[str, ScheduledTask]:
        return {task.name: task for task in self.tasks}


@documented(
    ProvenanceRecord(
        identifier="PSC-VIII-T38",
        title="Unlimited concurrency task graph schedule",
        status=ClaimStatus.PROVED,
        source="Paper VIII",
        assumptions=("finite acyclic task graph", "declared task durations", "unlimited concurrency"),
        verification_suite="algorithms.scheduling",
    )
)
def schedule_unlimited(tasks: Sequence[Task]) -> TaskSchedule:
    """Return the earliest start schedule under unlimited concurrency."""

    if not tasks:
        raise ValidationError("at least one task is required")
    by_name = {task.name: task for task in tasks}
    if len(by_name) != len(tasks):
        raise ValidationError("task names must be unique")
    for task in tasks:
        missing = set(task.dependencies) - set(by_name)
        if missing:
            raise ValidationError(f"task {task.name} has unknown dependencies: {sorted(missing)}")

    indegree = {name: len(task.dependencies) for name, task in by_name.items()}
    successors = {name: [] for name in by_name}
    for task in tasks:
        for dependency in task.dependencies:
            successors[dependency].append(task.name)
    current = sorted(name for name, degree in indegree.items() if degree == 0)
    levels: list[tuple[str, ...]] = []
    order: list[str] = []
    while current:
        levels.append(tuple(current))
        next_level: list[str] = []
        for name in current:
            order.append(name)
            for successor in sorted(successors[name]):
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    next_level.append(successor)
        current = sorted(next_level)
    if len(order) != len(tasks):
        raise ValidationError("task graph contains a directed cycle")

    finish: dict[str, float] = {}
    predecessor: dict[str, str | None] = {}
    level_index = {
        name: index
        for index, level in enumerate(levels)
        for name in level
    }
    scheduled: list[ScheduledTask] = []
    for name in order:
        task = by_name[name]
        if task.dependencies:
            critical_predecessor = max(
                task.dependencies,
                key=lambda dependency: (finish[dependency], dependency),
            )
            start = finish[critical_predecessor]
            predecessor[name] = critical_predecessor
        else:
            start = 0.0
            predecessor[name] = None
        task_finish = start + float(task.duration)
        finish[name] = task_finish
        scheduled.append(ScheduledTask(name, start, task_finish, level_index[name]))
    terminal = max(order, key=lambda name: (finish[name], name))
    path: list[str] = []
    current_name: str | None = terminal
    while current_name is not None:
        path.append(current_name)
        current_name = predecessor[current_name]
    path.reverse()
    return TaskSchedule(
        tuple(scheduled),
        max(finish.values()),
        tuple(path),
        tuple(levels),
    )


def shape_classes(tasks: Iterable[Task]) -> dict[Hashable | None, tuple[str, ...]]:
    """Group task names by a backend supplied shape key."""

    groups: dict[Hashable | None, list[str]] = {}
    for task in tasks:
        groups.setdefault(task.shape_key, []).append(task.name)
    return {
        key: tuple(sorted(names))
        for key, names in sorted(groups.items(), key=lambda item: repr(item[0]))
    }


def batched_launch_count(tasks: Sequence[Task], *, batching_available: bool = True) -> int:
    """Return the ideal launch count with or without shape batching."""

    if not tasks:
        return 0
    if not batching_available:
        return len(tasks)
    return len(shape_classes(tasks))


@dataclass(frozen=True, slots=True)
class BreakEvenResult:
    """Setup, per use timings, and the strict profitable reuse count."""

    setup_time: float
    baseline_time: float
    candidate_time: float
    reuse_count: int | None
    asymptotic_gain: float


def amortization_threshold(
    setup_time: float,
    baseline_time: float,
    candidate_time: float,
) -> int | None:
    """Return the smallest positive reuse count that repays setup cost."""

    setup = finite_float(setup_time, "setup_time", minimum=0.0)
    baseline = finite_float(baseline_time, "baseline_time", minimum=0.0)
    candidate = finite_float(candidate_time, "candidate_time", minimum=0.0)
    gain = baseline - candidate
    if gain <= 0:
        return None
    return max(1, floor(setup / gain) + 1)


def break_even_analysis(
    setup_time: float,
    baseline_time: float,
    candidate_time: float,
) -> BreakEvenResult:
    """Summarize the amortized comparison of two execution paths."""

    setup = finite_float(setup_time, "setup_time", minimum=0.0)
    baseline = finite_float(baseline_time, "baseline_time", minimum=0.0)
    candidate = finite_float(candidate_time, "candidate_time", minimum=0.0)
    threshold = amortization_threshold(setup, baseline, candidate)
    gain = baseline / candidate if candidate > 0 else float("inf")
    return BreakEvenResult(
        setup,
        baseline,
        candidate,
        threshold,
        gain,
    )


def verify_schedule(tasks: Sequence[Task], schedule: TaskSchedule, *, tolerance: float = 1e-12) -> bool:
    """Check durations, dependencies, and makespan for a schedule."""

    tolerance_value = finite_float(tolerance, "tolerance", minimum=0.0)
    by_task = {task.name: task for task in tasks}
    by_schedule = schedule.by_name()
    if set(by_task) != set(by_schedule):
        return False
    for name, task in by_task.items():
        scheduled = by_schedule[name]
        if abs((scheduled.finish - scheduled.start) - task.duration) > tolerance_value:
            return False
        if any(by_schedule[dependency].finish > scheduled.start + tolerance_value for dependency in task.dependencies):
            return False
    return abs(max(item.finish for item in schedule.tasks) - schedule.makespan) <= tolerance_value


__all__ = [
    "BreakEvenResult",
    "ScheduledTask",
    "Task",
    "TaskSchedule",
    "amortization_threshold",
    "batched_launch_count",
    "break_even_analysis",
    "schedule_unlimited",
    "shape_classes",
    "verify_schedule",
]
