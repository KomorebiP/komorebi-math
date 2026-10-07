"""Installed-package verification for normalized generating functions."""

from __future__ import annotations

from fractions import Fraction
from math import comb
from typing import Callable

import numpy as np

from ..export import dumps_portable, loads_portable
from ..generating import (
    BellParityAutomaton,
    FiniteNormalizedSeries,
    FormalSeries,
    PrimeFieldDomain,
    PrimePowerUnitCompiler,
    RationalDomain,
    analytic_mahler_product,
    certify_mahler_functional_equation,
    certify_rational_mask_algebraic_relation,
    compile_cartier_linear_representation,
    compile_finite_mask_carry_plan,
    euler_product_collapse,
    exponential_basis,
    factorial_power_basis,
    factorial_power_support_count,
    fir_taps_to_series,
    generalized_bell_numbers,
    kummer_layer_distribution,
    mahler_mask_inverse,
    mahler_mask_product,
    q_binomial_valuation_decomposition,
    q_binomial_valuation_direct,
    rational_mask_coefficients_mod_prime,
    series_to_fir_adapter,
    solve_polynomial_coboundary,
)
from .core import VerificationCheck, VerificationReport


def _capture(name: str, function: Callable[[], None]) -> VerificationCheck:
    try:
        function()
        return VerificationCheck(name, True)
    except Exception as exc:
        return VerificationCheck(name, False, f"{type(exc).__name__}: {exc}")


def run_generating_verification() -> VerificationReport:
    """Run lightweight exact checks for the installed generating subsystem."""

    checks: list[VerificationCheck] = []

    def normalized_product() -> None:
        basis = factorial_power_basis(3)
        left = FormalSeries(basis, (1, 2, -3, 5))
        right = FormalSeries(basis, (2, -1, 4))
        product = left.multiply(right)
        a = left.ordinary_coefficients()
        b = right.ordinary_coefficients()
        expected = tuple(
            sum(
                (a[k] if k < len(a) else 0)
                * (b[n - k] if n - k < len(b) else 0)
                for k in range(n + 1)
            )
            for n in range(product.degree + 1)
        )
        assert product.ordinary_coefficients() == expected

    checks.append(_capture("normalized product kernel", normalized_product))

    def domain_series() -> None:
        domain = PrimeFieldDomain(7)
        basis = exponential_basis()
        left = FiniteNormalizedSeries(basis, domain, (1, 2, 3))
        right = FiniteNormalizedSeries(basis, domain, (2, 4, 1))
        product = left.multiply(right)
        for n, value in enumerate(product.coefficients):
            expected = sum(
                comb(n, k) * left.coefficient(k) * right.coefficient(n - k)
                for k in range(n + 1)
            ) % 7
            assert value == expected

    checks.append(_capture("explicit coefficient domain", domain_series))

    def mask_inverse() -> None:
        mask = (1, 2, -1, 3)
        series = mahler_mask_product(mask, 3, 70)
        recovered = mahler_mask_inverse(series, 3, 70)
        certify_mahler_functional_equation(series, recovered, 3, 70).require()
        assert recovered[: len(mask)] == tuple(Fraction(value) for value in mask)

    checks.append(_capture("Mahler mask inversion", mask_inverse))

    def analytic_mask() -> None:
        shallow = analytic_mahler_product((1, Fraction(1, 2), -1), 0.27 + 0.04j, 3, 4)
        deep = analytic_mahler_product((1, Fraction(1, 2), -1), 0.27 + 0.04j, 3, 9)
        shallow.certificate().require()
        assert abs(deep.value - shallow.value) <= shallow.absolute_error_bound

    checks.append(_capture("analytic mask realization", analytic_mask))

    def rational_algebraic() -> None:
        certify_rational_mask_algebraic_relation(
            (1, 1, 2), (1, -1, 1), 5, 80
        ).require()

    checks.append(_capture("rational mask algebraic relation", rational_algebraic))

    def coboundary_and_euler() -> None:
        result = solve_polynomial_coboundary((1, 2), (1, 0, 2), 2, 4)
        assert result.require() == (Fraction(1), Fraction(2))
        collapse = euler_product_collapse(3, {1: -1, 3: 1})
        collapse.require()
        assert dict(collapse.closed_form_exponents) == {1: -1}

    checks.append(_capture("finite coboundary compilers", coboundary_and_euler))

    def carry_plan() -> None:
        plan = compile_finite_mask_carry_plan((1, 1, 1), 2)
        plan.certificate(96).require()

    checks.append(_capture("finite mask carry compilation", carry_plan))

    def cartier_plan() -> None:
        representation = compile_cartier_linear_representation((1, 1), (1, -1), 3)
        expected = rational_mask_coefficients_mod_prime((1, 1), (1, -1), 3, 81)
        assert representation.sequence(81) == expected

    checks.append(_capture("rational Cartier representation", cartier_plan))

    def kummer_layers() -> None:
        distribution = kummer_layer_distribution(137, 5)
        distribution.certificate().require()
        expected = sum(
            pow(comb(137, k), 3, 5**4) != 0
            for k in range(138)
        )
        assert factorial_power_support_count(137, 5, 3, 4) == expected

    checks.append(_capture("Kummer support layers", kummer_layers))

    def prime_power_residue() -> None:
        compiler = PrimePowerUnitCompiler(3, 5)
        for n, k in ((50, 17), (127, 41), (400, 199)):
            assert compiler.binomial(n, k) == comb(n, k) % (3**5)

    checks.append(_capture("prime-power binomial residues", prime_power_residue))

    def q_kummer() -> None:
        for n, k in ((20, 7), (33, 13), (48, 22)):
            predicted = q_binomial_valuation_decomposition(n, k, 2, 5)
            assert predicted.valuation == q_binomial_valuation_direct(n, k, 2, 5)

    checks.append(_capture("Gaussian q-Kummer valuation", q_kummer))

    def bell_automaton() -> None:
        automaton = BellParityAutomaton()
        values = generalized_bell_numbers(4, 160)
        assert tuple(value % 2 for value in values) == tuple(
            automaton.evaluate(index) for index in range(160)
        )

    checks.append(_capture("generalized Bell parity automaton", bell_automaton))

    def persistence() -> None:
        source = FiniteNormalizedSeries(
            factorial_power_basis(2),
            RationalDomain(),
            (1, 2, Fraction(7, 3), -5),
        )
        restored = loads_portable(dumps_portable(source))
        assert restored == source

    checks.append(_capture("portable generating records", persistence))

    def signal_bridge() -> None:
        source = FormalSeries(
            exponential_basis(),
            (1, Fraction(1, 2), Fraction(-1, 3), 2),
        )
        adapter = series_to_fir_adapter(source)
        adapter.certificate().require()
        restored = fir_taps_to_series(
            adapter.taps,
            basis=source.basis,
            max_denominator=10_000,
        )
        assert restored.ordinary_coefficients() == source.ordinary_coefficients()

    checks.append(_capture("formal FIR bridge", signal_bridge))

    return VerificationReport(tuple(checks))


__all__ = ["run_generating_verification"]
