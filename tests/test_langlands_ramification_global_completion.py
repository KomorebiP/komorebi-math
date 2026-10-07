import numpy as np

from komorebi_math.langlands.completion import (
    AutomorphicCompletionReport,
    CompatiblePropagationReport,
    CompletionGate,
    GateStatus,
)
from komorebi_math.langlands.global_ import (
    FiniteHeckePackets,
    PatchingSupportReport,
    SupportComparison,
    dyadic_sign_diagnostic,
)
from komorebi_math.langlands.local import AlternatingForm, QuadraticRefinement
from komorebi_math.langlands.ramification import (
    ProjectiveMultiplicityPacket,
    RamificationCliffordTower,
    RamificationStage,
)


def test_ramification_tower_and_dyadic_phase() -> None:
    form = AlternatingForm([[0, 1], [1, 0]], 2)
    refinement = QuadraticRefinement(form, (0, 0))
    packet = ProjectiveMultiplicityPacket(2, 2, form, quadratic_refinement=refinement)
    tower = RamificationCliffordTower(
        (
            RamificationStage(0, 8, 4, 2, 2, packet, "wild"),
            RamificationStage(1, 4, 2, 2, 1, None, "deeper"),
        )
    )
    report = tower.report()
    assert report.certificate.exactness.value == "conditional"
    assert packet.heisenberg_dimension == 2
    assert packet.phase_diagnostic()["arf_invariant"] == 0


def test_finite_hecke_packets_and_occupancy() -> None:
    p0 = np.diag([1, 1, 0])
    p1 = np.diag([0, 0, 1])
    hecke = FiniteHeckePackets({"m0": p0, "m1": p1}, commuting_operators=(np.diag([2, 2, 3]),))
    hecke.certificate().require()
    assert tuple(packet.dimension for packet in hecke.packets()) == (2, 1)
    assert hecke.occupied("m0", np.array([[1], [0], [0]], dtype=float))
    assert not hecke.occupied("m1", np.array([[1], [0], [0]], dtype=float))


def test_support_comparison_and_patching_report() -> None:
    comparison = SupportComparison.construct(("C1", "C2"), ("C1", "C2", "C3"), ("C1", "C2"))
    comparison.certificate.require()
    patching = PatchingSupportReport(True, True, True, True, False, derived=True)
    certificate = patching.certificate()
    assert not certificate.valid
    assert certificate.assumptions_unchecked == ("characteristic zero fiber is nonzero",)


def test_dyadic_sign_diagnostic() -> None:
    diagnostic = dyadic_sign_diagnostic()
    diagnostic.certificate.require()
    assert not diagnostic.rational_projectors_integral
    assert diagnostic.residual_algebra_local


def test_five_gate_completion_reports() -> None:
    complete = AutomorphicCompletionReport(
        {
            gate: GateStatus(True, f"{gate.value} supplied")
            for gate in CompletionGate
        }
    )
    result = complete.result()
    assert result.value and result.certificate.valid

    incomplete = AutomorphicCompletionReport(
        {
            CompletionGate.RESIDUAL: GateStatus(True, "occupied"),
            CompletionGate.PATCHING: GateStatus(True, "full support"),
            CompletionGate.COMPARISON: GateStatus(True, "R equals T"),
            CompletionGate.FIBER: GateStatus(True, "nonzero fiber"),
            CompletionGate.REALIZATION: GateStatus(False, "classicality missing"),
        }
    )
    assert incomplete.unmet_gates == (CompletionGate.REALIZATION,)
    assert not incomplete.result().certificate.valid


def test_compatible_one_member_propagation_boundary() -> None:
    report = CompatiblePropagationReport(True, True, True, False)
    result = report.result()
    assert not result.certificate.valid
    assert result.certificate.assumptions_unchecked == ("local global compatibility",)
