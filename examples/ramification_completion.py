"""Assemble a finite ramification packet and a five gate completion report."""

from __future__ import annotations

from komorebi_math.langlands.completion import (
    AutomorphicCompletionReport,
    CompletionGate,
    GateStatus,
)
from komorebi_math.langlands.local import AlternatingForm, QuadraticRefinement
from komorebi_math.langlands.ramification import (
    ProjectiveMultiplicityPacket,
    RamificationCliffordTower,
    RamificationStage,
)


def main() -> None:
    """Run the ramification and conditional completion example."""

    form = AlternatingForm(((0, 1), (1, 0)), prime=2)
    refinement = QuadraticRefinement(form, (0, 0))
    packet = ProjectiveMultiplicityPacket(
        prime=2,
        quotient_rank=2,
        form=form,
        quadratic_refinement=refinement,
    )
    tower = RamificationCliffordTower(
        (
            RamificationStage(0, 8, 4, 2, 2, packet, "wild"),
            RamificationStage(1, 4, 2, 2, 1, None, "deeper"),
        )
    )
    tower_report = tower.report()

    completion = AutomorphicCompletionReport(
        {
            CompletionGate.RESIDUAL: GateStatus(True, "residual carrier supplied"),
            CompletionGate.PATCHING: GateStatus(True, "full patched support supplied"),
            CompletionGate.COMPARISON: GateStatus(True, "support preserving comparison supplied"),
            CompletionGate.FIBER: GateStatus(True, "nonzero characteristic zero fiber supplied"),
            CompletionGate.REALIZATION: GateStatus(False, "classicality remains external"),
        }
    )
    result = completion.result()

    print(
        {
            "ramification_breaks": tower_report.breaks,
            "ramification_dimensions": tower_report.dimensions,
            "arf_invariant": packet.phase_diagnostic()["arf_invariant"],
            "completion_ready": bool(result.value),
            "unmet_gates": tuple(gate.value for gate in completion.unmet_gates),
        }
    )


if __name__ == "__main__":
    main()
