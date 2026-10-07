"""Installed verification for finite local, finite image, and completion models."""

from __future__ import annotations

import numpy as np

from ..langlands.completion import AutomorphicCompletionReport, CompletionGate, GateStatus
from ..langlands.finite_image import (
    CliffordStep,
    CliffordTower,
    InducedRepresentation,
    LinearCharacter,
    MonomialAction,
    cyclic_group,
    mackey_tensor_packets,
)
from ..langlands.global_ import FiniteHeckePackets, PatchingSupportReport, dyadic_sign_diagnostic
from ..langlands.local import (
    AlternatingForm,
    FiniteField,
    FrobeniusOrbit,
    HeisenbergModel,
    ProjectiveIntertwinerEnvelope,
    QuadraticRefinement,
    TameWeilPacket,
    gauss_sum,
)
from ..langlands.ramification import ProjectiveMultiplicityPacket, RamificationCliffordTower, RamificationStage
from .core import VerificationCheck, VerificationReport, _capture


def run_langlands_local_verification() -> VerificationReport:
    """Run the installed finite local packet verification suite."""

    checks: list[VerificationCheck] = []

    def field_gauss_check() -> None:
        field = FiniteField(3, 2, (1, 0, 1))
        for value in range(1, field.order):
            assert field.multiply(value, field.inverse(value)) == 1
        gauss_sum(field, 1).certificate.require()

    checks.append(_capture("finite fields and Gauss phases", field_gauss_check))

    def tame_check() -> None:
        orbit = FrobeniusOrbit(2, 3, 1)
        orbit.certificate().require()
        TameWeilPacket(orbit).relation_certificate().require()

    checks.append(_capture("tame Frobenius packet", tame_check))

    def heisenberg_check() -> None:
        model = HeisenbergModel(3, 1)
        model.group_law_certificate(model.element((1,), (2,), 1), model.element((2,), (1,), 2)).require()
        model.trace_support_certificate().require()
        model.selector_certificate().require()

    checks.append(_capture("finite Heisenberg packet", heisenberg_check))

    def symplectic_check() -> None:
        form = AlternatingForm([[0, 1], [1, 0]], 2)
        refinement = QuadraticRefinement(form, (1, 1))
        refinement.polar_certificate().require()
        assert refinement.arf_invariant() == 1

    checks.append(_capture("dyadic quadratic phase", symplectic_check))

    def envelope_check() -> None:
        model = HeisenbergModel(3, 1)
        envelope = ProjectiveIntertwinerEnvelope(
            prime=3,
            labels=((0, 0), (1, 0), (0, 1)),
            operators=(np.eye(3), model.translation_matrix((1,)), model.modulation_matrix((1,))),
        ).recover()
        envelope.certificate.require()
        assert envelope.form.rank == 2

    checks.append(_capture("projective commutator envelope", envelope_check))
    return VerificationReport(tuple(checks))


def run_langlands_finite_image_verification() -> VerificationReport:
    """Run the installed finite image and Clifford verification suite."""

    checks: list[VerificationCheck] = []

    def induced_check() -> None:
        group = cyclic_group(4)
        subgroup = group.subgroup((0, 2))
        character = LinearCharacter(subgroup, {0: 1, 2: -1})
        representation = InducedRepresentation(group, character)
        representation.certificate().require()
        action = MonomialAction(representation)
        packets = action.symmetric_power_packets(3)
        action.packet_certificate(packets, expected_dimension=4).require()

    checks.append(_capture("induced and symmetric orbit packets", induced_check))

    def mackey_check() -> None:
        group = cyclic_group(4)
        subgroup = group.subgroup((0, 2))
        character = LinearCharacter(subgroup, {0: 1, 2: -1})
        mackey_tensor_packets(group, character, character).certificate.require()

    checks.append(_capture("Mackey double coset packets", mackey_check))

    def clifford_check() -> None:
        tower = CliffordTower((CliffordStep(24, 12, 3, 3, "E"), CliffordStep(12, 4, 3, 1, "I")))
        assert tower.word == "EI"
        assert tower.depth == 1
        assert not tower.automorphic_plan().certificate.valid

    checks.append(_capture("Clifford word and depth", clifford_check))
    return VerificationReport(tuple(checks))


def run_langlands_completion_verification() -> VerificationReport:
    """Run global packet and conditional completion diagnostics."""

    checks: list[VerificationCheck] = []

    def hecke_check() -> None:
        packets = FiniteHeckePackets({"a": np.diag([1, 0]), "b": np.diag([0, 1])})
        packets.certificate().require()
        assert all(packet.dimension == 1 for packet in packets.packets())

    checks.append(_capture("finite Hecke packets", hecke_check))

    def patching_check() -> None:
        report = PatchingSupportReport(True, True, True, True, False)
        assert not report.certificate().valid
        dyadic_sign_diagnostic().certificate.require()

    checks.append(_capture("patching and dyadic diagnostics", patching_check))

    def ramification_check() -> None:
        form = AlternatingForm([[0, 1], [1, 0]], 2)
        packet = ProjectiveMultiplicityPacket(2, 2, form, quadratic_refinement=QuadraticRefinement(form))
        tower = RamificationCliffordTower((RamificationStage(0, 4, 2, 2, 1, packet),))
        assert tower.report().certificate.valid

    checks.append(_capture("ramification Clifford packet", ramification_check))

    def completion_check() -> None:
        report = AutomorphicCompletionReport({gate: GateStatus(True, gate.value) for gate in CompletionGate})
        assert report.result().certificate.valid

    checks.append(_capture("five gate completion report", completion_check))
    return VerificationReport(tuple(checks))


def run_langlands_verification() -> VerificationReport:
    """Run every installed finite Langlands verification suite."""

    reports = (
        run_langlands_local_verification(),
        run_langlands_finite_image_verification(),
        run_langlands_completion_verification(),
    )
    return VerificationReport(tuple(check for report in reports for check in report.checks))


__all__ = [
    "run_langlands_completion_verification",
    "run_langlands_finite_image_verification",
    "run_langlands_local_verification",
    "run_langlands_verification",
]
