"""Independent witnesses, dependency omissions and mutant detection."""

import json
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction

import pytest

from komorebi_math.audit import paper_checks, run_stress


def test_paper_controls_replay_finite_obstructions_and_exact_witnesses():
    pytest.importorskip("sympy")
    report = run_stress(paper_checks(3), seed=20261010)
    assert report.passed, [(r["identifier"], r["detail"]) for r in report.to_dict()["checks"]]
    assert report.counts["passed"] == 10
    rows = {r["identifier"]: r for r in report.to_dict()["checks"]}
    assert all(r["exactness"] == "exact" for r in rows.values())
    assert rows["paper.core.mapping-cones"]["evidence"]["noncommuting_square_rejected"]
    assert rows["paper.compiler.exact-audits"]["evidence"]["unsupported_budget_optimum"]
    assert rows["paper.compiler.exact-audits"]["evidence"]["strict_integer_boundary"]
    assert rows["paper.arithmetic.lifting-stencils"]["evidence"]["singular_later_obstruction"]
    assert rows["paper.arithmetic.lifting-stencils"]["evidence"]["incompatible_shared_digit"]
    assert rows["paper.langlands.tame-tensor"]["evidence"]["reducible_interaction_witness"]
    assert rows["paper.collatz.words"]["evidence"]["word_seed_comparisons"] == 10880
    assert rows["paper.collatz.words"]["evidence"]["zero_budget_inconclusive"]
    assert rows["paper.signals.predictive-foundations"]["evidence"]["fit_does_not_establish_global_coverage"]
    assert rows["paper.signals.predictive-foundations"]["evidence"]["metadata_cannot_promote_closure"]
    assert rows["paper.bayesian.gaussian-verifier"]["evidence"]["coarse_dyadic_score_rejected"]
    assert rows["paper.bayesian.gaussian-verifier"]["evidence"]["incomplete_packets_rejected"]
    assert rows["paper.generating.prime-power-cartier"]["evidence"]["extra_padic_digit_retained"]
    assert rows["paper.generating.prime-power-cartier"]["evidence"]["huge_index_matches"]
    assert report.replay_digest == run_stress(paper_checks(3), seed=20261010).replay_digest


def test_missing_exact_extra_is_visible_and_cannot_qualify(monkeypatch):
    monkeypatch.setitem(sys.modules, "sympy", None)
    checks = tuple(check for check in paper_checks(1) if "required_extra" in check.tested_range)
    report = run_stress(checks)
    assert not report.passed
    assert report.counts["skipped"] == 2
    assert all("SymPy exact extra" in r["detail"] for r in report.to_dict()["checks"])


def test_hermite_suite_detects_wrong_transfer_eigenvalues(monkeypatch):
    import komorebi_math.dynamics.hermite as hermite

    monkeypatch.setattr(
        hermite, "gaussian_transfer_polynomial", lambda polynomial, correlation, **kwargs: tuple(polynomial)
    )
    selected = tuple(c for c in paper_checks(1) if c.identifier == "paper.gaussian.hermite-transfer")
    report = run_stress(selected)
    assert not report.passed and report.counts["failed"] == 1
    assert "transfer eigenvalue" in report.to_dict()["checks"][0]["detail"]


def test_tame_suite_detects_forged_local_polynomial(monkeypatch):
    import komorebi_math.langlands.local.tame_tensor as tame

    original = tame.tame_tensor_character_packets

    def forged(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(result, local_polynomial_coefficients=(Fraction(7),))

    monkeypatch.setattr(tame, "tame_tensor_character_packets", forged)
    selected = tuple(c for c in paper_checks(1) if c.identifier == "paper.langlands.tame-tensor")
    report = run_stress(selected)
    assert not report.passed and report.counts["failed"] == 1
    assert "full rational tensor matrix" in report.to_dict()["checks"][0]["detail"]


def test_gaussian_suite_detects_forged_exact_posterior(monkeypatch):
    import komorebi_math.dynamics.gaussian_verifier as gaussian

    original = gaussian.exact_gaussian_posterior

    def forged(*args, **kwargs):
        result = original(*args, **kwargs)
        covariance = tuple(
            tuple(value + int(i == j) for j, value in enumerate(row))
            for i, row in enumerate(result.covariance)
        )
        return replace(result, covariance=covariance)

    monkeypatch.setattr(gaussian, "exact_gaussian_posterior", forged)
    selected = tuple(c for c in paper_checks(1) if c.identifier == "paper.bayesian.gaussian-verifier")
    report = run_stress(selected)
    assert not report.passed and report.counts["failed"] == 1
    assert "Schur complement" in report.to_dict()["checks"][0]["detail"]


def test_cartier_suite_detects_wrong_digit_sections(monkeypatch):
    import komorebi_math.generating.cartier_prime_power as cartier

    monkeypatch.setattr(
        cartier.PrimePowerCartierRepresentation,
        "transition",
        lambda self, state, digit: tuple(tuple(row) for row in state),
    )
    selected = tuple(c for c in paper_checks(1) if c.identifier == "paper.generating.prime-power-cartier")
    report = run_stress(selected)
    assert not report.passed and report.counts["failed"] == 1
    assert "Cartier residues" in report.to_dict()["checks"][0]["detail"]


@pytest.mark.parametrize("samples", [True, 0, 129, 1.0, -1])
def test_paper_factory_rejects_invalid_or_empty_bounds(samples):
    with pytest.raises(ValueError):
        paper_checks(samples)


def test_isolated_installed_cli_appends_paper_and_special_function_controls(tmp_path):
    pytest.importorskip("sympy")
    target = tmp_path / "papers.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "komorebi_math.audit",
            "--suite",
            "core",
            "--papers",
            "--samples",
            "1",
            "--timeout",
            "30",
            "--output",
            str(target),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(target.read_text())
    assert report == json.loads(result.stdout)
    assert report["counts"]["passed"] == 20
    assert report["counts"]["skipped"] == report["counts"]["error"] == 0
    assert any(row["suite"] == "paper_special_functions" for row in report["checks"])
