from __future__ import annotations

import json
import os
import subprocess
import sys

from komorebi_math import FiniteNormalizedSeries, feature
from komorebi_math.cli.main import main
from komorebi_math.compatibility import (
    DEFAULT_PUBLIC_NAMESPACES,
    check_beta_api,
    check_public_api,
    create_public_api_manifest,
)
from komorebi_math.generating import BellResidueAutomaton, GeneratingBudget


def test_block13_public_manifest_covers_generating_namespace() -> None:
    manifest = create_public_api_manifest()
    assert "komorebi_math.generating" in DEFAULT_PUBLIC_NAMESPACES
    assert manifest.namespaces == DEFAULT_PUBLIC_NAMESPACES
    assert len(manifest.symbols) >= 900
    assert any(
        symbol.namespace == "komorebi_math.generating"
        and symbol.name == "FiniteNormalizedSeries"
        for symbol in manifest.symbols
    )
    assert FiniteNormalizedSeries.__module__.startswith("komorebi_math.generating")
    check_public_api(strict_signatures=True).certificate.require()
    check_beta_api().certificate.require()


def test_generating_feature_is_implemented() -> None:
    specification = feature("generating")
    assert specification.status.value == "implemented"
    assert "Cartier" in specification.summary


def test_generating_cli_mask_cartier_kernel_and_bell(capsys) -> None:
    assert main(["generating", "mask", "--mask", "1,1,1", "--radix", "2", "--count", "8"]) == 0
    mask = json.loads(capsys.readouterr().out)
    assert mask["coefficients"] == ["1", "1", "2", "1", "3", "2", "3", "1"]
    assert mask["verified"] is True

    assert main([
        "generating", "cartier", "--numerator", "1,1", "--denominator", "1,-1",
        "--prime", "3", "--count", "12",
    ]) == 0
    cartier = json.loads(capsys.readouterr().out)
    assert cartier["prime"] == 3
    assert len(cartier["coefficients"]) == 12

    assert main([
        "generating", "kernel", "--index", "20", "--subindex", "7",
        "--prime", "3", "--exponent", "4", "--power", "2",
    ]) == 0
    kernel = json.loads(capsys.readouterr().out)
    assert kernel["residue"] == pow(__import__("math").comb(20, 7), 2, 3**4)

    assert main([
        "generating", "bell", "--index", "23", "--prime", "3", "--power", "5",
    ]) == 0
    bell = json.loads(capsys.readouterr().out)
    assert bell["valid"] is True


def test_generating_and_block13_cli_verification() -> None:
    env = os.environ.copy()
    for suite in ('generating', 'block13'):
        completed = subprocess.run(
            [sys.executable, "-m", "komorebi_math", "verify", suite],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr or completed.stdout
        assert "verification passed" in completed.stdout


def test_bell_touchard_compiler_scales_with_explicit_budget() -> None:
    budget = GeneratingBudget(maximum_reachable_states=140_000)
    automaton = BellResidueAutomaton.compile(7, 13, budget=budget)
    assert automaton.period_bound == 137_257
    assert all(
        automaton.outputs[n + 7] == (automaton.outputs[n + 1] + automaton.outputs[n]) % 7
        for n in range(automaton.period_bound - 7)
    )
