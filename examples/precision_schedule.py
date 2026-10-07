"""Choose packet precisions and schedule the resulting task graph."""

from __future__ import annotations

from komorebi_math.algorithms import (
    DTypeChoice,
    Task,
    assign_discrete_precisions,
    schedule_unlimited,
)

assignment = assign_discrete_precisions(
    [
        [DTypeChoice("low32", 1.0, 3, "float32"), DTypeChoice("low64", 3.0, 0, "float64")],
        [DTypeChoice("high32", 1.0, 2, "float32"), DTypeChoice("high64", 3.0, 0, "float64")],
    ],
    budget=3,
)
if assignment is None:
    raise RuntimeError("the declared precision budget is infeasible")
assignment.certificate.require()

schedule = schedule_unlimited(
    [
        Task("analyze", 1.0, shape_key=(2, 2)),
        Task("low", 2.0, ("analyze",), shape_key=(2, 2)),
        Task("high", 3.0, ("analyze",), shape_key=(2, 2)),
        Task("synthesize", 1.0, ("low", "high"), shape_key=(2, 2)),
    ]
)

print("precision choices", assignment.names)
print("total cost", assignment.total_cost)
print("schedule makespan", schedule.makespan)
print("critical path", schedule.critical_path)
