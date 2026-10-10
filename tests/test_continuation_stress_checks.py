"""Independent installed oracles must detect altered mathematical outputs."""

from dataclasses import replace

import pytest

from komorebi_math.audit import builtin_checks, continuation_checks, paper_checks, run_stress
from komorebi_math.audit.special_function_checks import special_function_checks
from verification.release_artifact_gate import INSTALLED_MATH_CHECKS


def test_new_checks_replay_exactly_and_gate_matches_the_complete_registry():
    pytest.importorskip("sympy")
    checks = continuation_checks(3)
    report = run_stress(checks, seed=20261010)
    assert report.passed, [(row["identifier"], row["detail"]) for row in report.to_dict()["checks"]]
    assert report.counts["passed"] == 5
    assert all(row["exactness"] == "exact" for row in report.to_dict()["checks"])
    assert report.replay_digest == run_stress(checks, seed=20261010).replay_digest
    complete = (
        *builtin_checks(samples=1),
        *paper_checks(1),
        *special_function_checks(1),
        *continuation_checks(1),
    )
    assert {check.identifier for check in complete} == INSTALLED_MATH_CHECKS


@pytest.mark.parametrize("samples", [True, 0, -1, 129, 1.0])
def test_invalid_sample_bounds(samples):
    with pytest.raises(ValueError):
        continuation_checks(samples)


def _selected(identifier):
    return tuple(check for check in continuation_checks(1) if check.identifier == identifier)


def test_channel_oracle_detects_forged_optimum(monkeypatch):
    from komorebi_math.decision.avoc import FiniteChannel

    original = FiniteChannel.optimal_policy

    def forged(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        return replace(result, value=result.value + 1)

    monkeypatch.setattr(FiniteChannel, "optimal_policy", forged)
    report = run_stress(_selected("paper.avoc.finite-decisions"))
    assert report.counts["failed"] == 1
    assert "complete channel policy enumeration" in report.to_dict()["checks"][0]["detail"]


def test_boolean_oracle_rejects_forged_bank(monkeypatch):
    import komorebi_math.algorithms.boolean_synthesis as boolean

    original = boolean.build_boolean_closure_bank

    def forged(*args, **kwargs):
        bank = original(*args, **kwargs)
        first = replace(bank.entries[0], node_count=bank.entries[0].node_count + 1)
        return replace(bank, entries=(first, *bank.entries[1:]))

    monkeypatch.setattr(boolean, "build_boolean_closure_bank", forged)
    report = run_stress(_selected("paper.avoc.boolean-minimum"))
    assert report.counts["failed"] == 1


def test_nonfree_ring_oracle_detects_invented_free_rank(monkeypatch):
    from komorebi_math.generating.residual_modules import CanonicalResidualModule

    monkeypatch.setattr(CanonicalResidualModule, "additive_p_exponents", lambda self: (2,))
    report = run_stress(_selected("paper.generating.residual-modules"))
    assert report.counts["failed"] == 1
    assert "nonfree quotient retained" in report.to_dict()["checks"][0]["detail"]


def test_shell_oracle_detects_changed_signed_total(monkeypatch):
    from komorebi_math.arithmetic.lehmer_shells import FiveCoreShell

    monkeypatch.setattr(
        FiveCoreShell, "weighted_sum", property(lambda self: sum(p.weight for p in self.packets) + 1)
    )
    report = run_stress(_selected("paper.arithmetic.lehmer-shells"))
    assert report.counts["failed"] == 1
    assert "q-product and content transforms" in report.to_dict()["checks"][0]["detail"]


def test_missing_tikhonov_dependency_cannot_qualify(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "sympy", None)
    report = run_stress(_selected("paper.signals.tikhonov-certificates"))
    assert report.counts["skipped"] == 1 and not report.passed
