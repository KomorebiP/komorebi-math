"""Installed verification for lazy exact sequence execution."""

from __future__ import annotations

from ..sequences import (
    apery_recurrence,
    apery_sequence,
    bell_sequence,
    binomial_transform_values,
    catalan_sequence,
    compile_sequence,
    fibonacci_sequence,
    inverse_binomial_transform_values,
    motzkin_recurrence,
    partition_sequence,
    thue_morse_automaton,
    thue_morse_sequence,
)
from .core import VerificationCheck, VerificationReport, _capture


def run_sequences_verification() -> VerificationReport:
    """Run the installed sequence execution verification suite."""

    checks: list[VerificationCheck] = []

    def catalog_check() -> None:
        assert bell_sequence().first(8) == (1, 1, 2, 5, 15, 52, 203, 877)
        assert catalan_sequence().first(8) == (1, 1, 2, 5, 14, 42, 132, 429)
        assert partition_sequence().first(12) == (1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56)
        assert apery_sequence().first(6) == (1, 5, 73, 1445, 33001, 819005)

    checks.append(_capture("classical exact sequence catalog", catalog_check))

    def transform_check() -> None:
        values = tuple(range(-6, 15))
        assert inverse_binomial_transform_values(binomial_transform_values(values)) == values

    checks.append(_capture("binomial transform inversion", transform_check))

    def automatic_check() -> None:
        model = thue_morse_automaton()
        model.certificate(1024).certificate.require()
        sequence = thue_morse_sequence()
        for n in range(512):
            assert sequence.term(2 * n) == sequence.term(n)
            assert sequence.term(2 * n + 1) == 1 - sequence.term(n)
        compiled = compile_sequence(sequence)
        compiled.certificate.require()
        assert compiled.strategy == "declared finite automaton"

    checks.append(_capture("Thue Morse finite automaton", automatic_check))

    def holonomic_check() -> None:
        motzkin = motzkin_recurrence()
        apery = apery_recurrence()
        motzkin.certificate(32).certificate.require()
        apery.certificate(20).certificate.require()
        assert compile_sequence(apery_sequence()).strategy == "declared holonomic recurrence"

    checks.append(_capture("exact holonomic sequence execution", holonomic_check))

    def compiler_check() -> None:
        sequence = fibonacci_sequence()
        compiled = compile_sequence(sequence)
        compiled.certificate.require()
        assert compiled.first(30) == sequence.first(30)
        numerator, denominator = compiled.generating_function()
        assert numerator == (0, 1)
        assert denominator == (1, -1, -1)

    checks.append(_capture("declared recurrence compilation", compiler_check))

    return VerificationReport(tuple(checks))


__all__ = ["run_sequences_verification"]
