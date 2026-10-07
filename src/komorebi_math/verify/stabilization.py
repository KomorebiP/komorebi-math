"""Installed verification for Block 15 polyphase stabilization and 1.0 contracts."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from ..arithmetic import polyphase_analyze as arithmetic_polyphase_analyze
from ..compatibility import check_stable_api, create_stable_api_manifest, load_stable_api_manifest
from ..core import PacketSystem
from ..export.persistence import dumps_portable, loads_portable
from ..persistence_contract import check_portable_codec_contract
from ..refinement import (
    LatticePolyphaseRefinement,
    PacketRefinement,
    PolyphaseRefinement,
    PolyphaseRefinementTower,
)
from ..signals import lattice_polyphase_analyze, polyphase_analyze
from ..stable_contract import check_stable_contract
from ..stable_semantics import run_stable_semantics_fixture
from .core import VerificationCheck, VerificationReport, _capture


def _singleton_packet_system(size: int) -> PacketSystem:
    return PacketSystem(
        [
            np.diag([1.0 if row == column else 0.0 for row in range(size)])
            for column in range(size)
        ]
    )


def run_polyphase_stabilization_verification() -> VerificationReport:
    """Verify the canonical finite polyphase layer introduced in Block 15."""

    checks: list[VerificationCheck] = []

    def signal_refinement() -> None:
        values = np.arange(41, dtype=np.int64)
        coarse = polyphase_analyze(values, 4)
        plan = PolyphaseRefinement(4, 12)
        fine = plan.refine(coarse)
        direct = polyphase_analyze(values, 12)
        assert plan.children(1) == (1, 5, 9)
        assert all(
            np.array_equal(left, right)
            for left, right in zip(fine.channels, direct.channels, strict=True)
        )
        assert np.array_equal(plan.coarsen(fine).reconstruct(), values)
        plan.certificate(coarse).require()

    checks.append(_capture("canonical signal polyphase refinement", signal_refinement))

    def arithmetic_refinement() -> None:
        values = tuple(Fraction(index + 1, index + 3) for index in range(19))
        coarse = arithmetic_polyphase_analyze(values, 3)
        plan = PolyphaseRefinement(3, 12)
        fine = plan.refine(coarse)
        assert fine == arithmetic_polyphase_analyze(values, 12)
        assert plan.coarsen(fine) == coarse
        plan.certificate(coarse).require()

    checks.append(_capture("exact arithmetic polyphase refinement", arithmetic_refinement))

    def lattice_refinement() -> None:
        values = np.arange(5 * 7, dtype=np.int64).reshape(5, 7)
        coarse = lattice_polyphase_analyze(values, (1, 2))
        plan = LatticePolyphaseRefinement((1, 2), (5, 6))
        fine = plan.refine(coarse)
        direct = lattice_polyphase_analyze(values, (5, 6))
        assert set(fine.channels) == set(direct.channels)
        assert all(np.array_equal(fine.channels[key], direct.channels[key]) for key in direct.channels)
        assert np.array_equal(plan.coarsen(fine).reconstruct(), values)
        plan.certificate(coarse).require()

    checks.append(_capture("rectangular lattice polyphase refinement", lattice_refinement))

    def tower_coherence() -> None:
        values = np.arange(73, dtype=np.int64)
        coarse = polyphase_analyze(values, 2)
        tower = PolyphaseRefinementTower((2, 6, 12, 36))
        sequential = tower.refine(coarse)
        direct = PolyphaseRefinement(2, 36).refine(coarse)
        assert all(
            np.array_equal(left, right)
            for left, right in zip(sequential.channels, direct.channels, strict=True)
        )
        tower.coherence_certificate(coarse).require()

    checks.append(_capture("multilevel refinement tower coherence", tower_coherence))

    def general_packet_refinement() -> None:
        coarse = PacketSystem(
            [
                np.diag([1.0, 1.0, 0.0, 0.0]),
                np.diag([0.0, 0.0, 1.0, 1.0]),
            ]
        )
        fine = _singleton_packet_system(4)
        plan = PacketRefinement(coarse, fine, (0, 0, 1, 1))
        values = np.asarray([1.0, 2.0, 3.0, 4.0])
        coordinates = coarse.analyze(values)
        assert np.allclose(plan.coarsen(plan.refine(coordinates)).reconstruct(), values)
        plan.certificate().require()

    checks.append(_capture("general packet refinement coherence", general_packet_refinement))

    def persistence_roundtrip() -> None:
        value = PolyphaseRefinementTower((2, 6, 18))
        assert loads_portable(dumps_portable(value)) == value

    checks.append(_capture("polyphase portable record roundtrip", persistence_roundtrip))

    return VerificationReport(tuple(checks))


def run_stable_contract_verification() -> VerificationReport:
    """Verify structural, semantic, and persistence compatibility for 1.0."""

    checks: list[VerificationCheck] = []

    def structural_api() -> None:
        report = check_stable_api()
        report.certificate.require()
        assert not report.added
        assert not report.removed
        assert not report.kind_changed
        assert not report.signature_changed

    checks.append(_capture("strict stable API manifest", structural_api))

    def deterministic_manifest() -> None:
        reference = load_stable_api_manifest()
        current = create_stable_api_manifest()
        assert reference.to_dict() == current.to_dict()
        assert current.digest == create_stable_api_manifest().digest

    checks.append(_capture("deterministic stable API snapshot", deterministic_manifest))

    def semantic_contract() -> None:
        run_stable_semantics_fixture().certificate.require()

    checks.append(_capture("closed stable semantic fixture", semantic_contract))

    def persistence_contract() -> None:
        check_portable_codec_contract().certificate.require()

    checks.append(_capture("portable codec migration contract", persistence_contract))

    def aggregate_contract() -> None:
        check_stable_contract().certificate.require()

    checks.append(_capture("aggregate 1.0-candidate contract", aggregate_contract))

    def versioned_aliases() -> None:
        from .. import stable
        from .. import v1
        from ..api import v1 as api_v1

        assert stable.__all__ == v1.__all__ == api_v1.__all__
        for name in stable.__all__:
            assert getattr(v1, name) is getattr(stable, name)
            assert getattr(api_v1, name) is getattr(stable, name)

    checks.append(_capture("versioned stable namespace identity", versioned_aliases))

    return VerificationReport(tuple(checks))


__all__ = [
    "run_polyphase_stabilization_verification",
    "run_stable_contract_verification",
]
