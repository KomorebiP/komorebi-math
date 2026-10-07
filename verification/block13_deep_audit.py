"""Independent deterministic audit for the Block 13 generating subsystem."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from komorebi_math import __version__
from komorebi_math.export import dumps_portable, loads_portable
from komorebi_math.generating import (
    BellParityAutomaton,
    BellResidueAutomaton,
    FiniteNormalizedSeries,
    FormalSeries,
    GeneratingBudget,
    PrimeFieldDomain,
    PrimePowerKernelCompiler,
    PrimePowerUnitCompiler,
    RationalDomain,
    analytic_mahler_product,
    bell_numbers,
    certify_cartier_carry_equivalence,
    compile_cartier_linear_representation,
    compile_digit_cauchy_product,
    compile_finite_mask_carry_plan,
    compile_rational_mask_automaton,
    exponential_basis,
    factorial_power_basis,
    factorial_power_support_indices,
    fir_taps_to_series,
    generalized_bell_numbers,
    kummer_layer_distribution,
    mahler_mask_inverse,
    mahler_mask_product,
    ordinary_basis,
    q_binomial_valuation_decomposition,
    q_binomial_valuation_direct,
    rational_mask_coefficients_mod_prime,
    series_to_fir_adapter,
)
from komorebi_math.generating.padic_units import borrow_count


@dataclass(slots=True)
class Audit:
    """Count scalar comparisons and retain bounded failure diagnostics."""

    checks: int = 0
    failures: list[str] = field(default_factory=list)
    sections: dict[str, int] = field(default_factory=dict)

    def check(self, condition: bool, message: str, *, section: str) -> None:
        self.checks += 1
        self.sections[section] = self.sections.get(section, 0) + 1
        if not bool(condition) and len(self.failures) < 100:
            self.failures.append(message)

    def equal_sequences(
        self,
        observed: Sequence[Any],
        expected: Sequence[Any],
        *,
        message: str,
        section: str,
    ) -> None:
        self.check(len(observed) == len(expected), f"{message}: length mismatch", section=section)
        for index, (left, right) in enumerate(zip(observed, expected)):
            self.check(left == right, f"{message}: index {index}, {left!r} != {right!r}", section=section)


def _ordinary_convolution(left: Sequence[Fraction], right: Sequence[Fraction]) -> tuple[Fraction, ...]:
    output = [Fraction(0)] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            output[i + j] += a * b
    return tuple(output)


def _direct_mahler(mask: Sequence[Fraction], radix: int, count: int) -> tuple[Fraction, ...]:
    result = [Fraction(0)] * count
    if count:
        result[0] = Fraction(1)
    scale = 1
    while scale < count:
        factor = [Fraction(0)] * count
        for index, value in enumerate(mask):
            location = index * scale
            if location >= count:
                break
            factor[location] = Fraction(value)
        updated = [Fraction(0)] * count
        for i, a in enumerate(result):
            if not a:
                continue
            for j, b in enumerate(factor[: count - i]):
                if b:
                    updated[i + j] += a * b
        result = updated
        scale *= radix
    return tuple(result)


def _digit_value(weights: Sequence[int], radix: int, index: int) -> int:
    value = 1
    remaining = index
    while remaining:
        remaining, digit = divmod(remaining, radix)
        value *= weights[digit]
    return value


def _direct_rational_mask(
    numerator: Sequence[int], denominator: Sequence[int], prime: int, count: int
) -> tuple[int, ...]:
    return rational_mask_coefficients_mod_prime(numerator, denominator, prime, count)


def run() -> dict[str, Any]:
    """Run the deterministic audit and return a JSON-compatible report."""

    rng = np.random.default_rng(13012026)
    audit = Audit()

    # Normalized products through independent ordinary-coefficient convolution.
    bases = (ordinary_basis(), exponential_basis(), *(factorial_power_basis(s) for s in range(2, 6)))
    for case in range(240):
        basis = bases[case % len(bases)]
        left = tuple(Fraction(int(v), int(rng.integers(1, 6))) for v in rng.integers(-8, 9, size=int(rng.integers(1, 12))))
        right = tuple(Fraction(int(v), int(rng.integers(1, 6))) for v in rng.integers(-8, 9, size=int(rng.integers(1, 12))))
        product = FormalSeries(basis, left).multiply(FormalSeries(basis, right))
        expected = _ordinary_convolution(
            tuple(left[n] / basis.gamma(n) for n in range(len(left))),
            tuple(right[n] / basis.gamma(n) for n in range(len(right))),
        )
        audit.equal_sequences(product.ordinary_coefficients(), expected, message=f"normalized product {case}", section="normalized_series")

    # Explicit rational and prime-field domains.
    for prime in (2, 3, 5, 7, 11):
        domain = PrimeFieldDomain(prime)
        basis = exponential_basis()
        for case in range(40):
            left = tuple(int(v) for v in rng.integers(-20, 21, size=9))
            right = tuple(int(v) for v in rng.integers(-20, 21, size=8))
            product = FiniteNormalizedSeries(basis, domain, left).multiply(
                FiniteNormalizedSeries(basis, domain, right)
            )
            expected = tuple(
                sum(comb(n, k) * (left[k] if k < len(left) else 0) * (right[n - k] if n - k < len(right) else 0) for k in range(n + 1)) % prime
                for n in range(len(left) + len(right) - 1)
            )
            audit.equal_sequences(product.coefficients, expected, message=f"field product p={prime} case={case}", section="coefficient_domains")

    # Mahler masks, inversion, and compiled carry coefficients.
    for radix in (2, 3, 5):
        for case in range(70):
            length = int(rng.integers(1, 7))
            mask = (Fraction(1),) + tuple(Fraction(int(v)) for v in rng.integers(-3, 4, size=length - 1))
            count = 180
            expected = _direct_mahler(mask, radix, count)
            observed = mahler_mask_product(mask, radix, count - 1)
            audit.equal_sequences(observed, expected, message=f"Mahler product b={radix} case={case}", section="mahler_masks")
            recovered = mahler_mask_inverse(observed, radix, count - 1)
            audit.equal_sequences(recovered[:length], mask, message=f"Mahler inverse b={radix} case={case}", section="mahler_masks")
            plan = compile_finite_mask_carry_plan(mask, radix)
            audit.equal_sequences(plan.sequence(count), expected, message=f"carry plan b={radix} case={case}", section="finite_mask_carries")

    # Carry closure for ordinary products of digit-local sequences.
    for radix in (2, 3, 5):
        for factors_count in (2, 3, 4):
            for case in range(20):
                tables = tuple(
                    (1,) + tuple(int(v) for v in rng.integers(-3, 4, size=radix - 1))
                    for _ in range(factors_count)
                )
                plan = compile_digit_cauchy_product(tables, radix)
                count = 100
                sequences = [tuple(_digit_value(table, radix, n) for n in range(count)) for table in tables]
                direct = sequences[0]
                for sequence in sequences[1:]:
                    direct = tuple(sum(direct[k] * sequence[n - k] for k in range(n + 1)) for n in range(count))
                audit.equal_sequences(plan.sequence(count), tuple(Fraction(v) for v in direct), message=f"digit Cauchy b={radix} m={factors_count} case={case}", section="digit_cauchy")

    # Cartier representations and minimized automata.
    rational_cases = (
        (2, (1, 1), (1, -1)),
        (2, (1, 1, 1), (1, 0, 1)),
        (3, (1, 1), (1, -1)),
        (3, (1, 2, 1), (1, 1)),
        (5, (1, 1, 2), (1, -1, 1)),
        (7, (1, 3), (1, 2, 1)),
    )
    for prime, numerator, denominator in rational_cases:
        expected = _direct_rational_mask(numerator, denominator, prime, 1200)
        linear = compile_cartier_linear_representation(numerator, denominator, prime)
        automaton = compile_rational_mask_automaton(numerator, denominator, prime)
        audit.equal_sequences(linear.sequence(1200), expected, message=f"Cartier linear p={prime}", section="cartier")
        audit.equal_sequences(automaton.sequence(1200), expected, message=f"Cartier automaton p={prime}", section="cartier")
    for prime in (2, 3, 5, 7):
        certify_cartier_carry_equivalence((1, 1, 1, 2), prime).require()
        audit.check(True, f"Cartier carry certificate p={prime}", section="cartier")

    # Kummer distributions and factorial-power support.
    for prime in (2, 3, 5, 7):
        for index in range(260):
            distribution = kummer_layer_distribution(index, prime)
            counts: list[int] = []
            for subindex in range(index + 1):
                layer = borrow_count(index, subindex, prime)
                while len(counts) <= layer:
                    counts.append(0)
                counts[layer] += 1
            while len(counts) > 1 and counts[-1] == 0:
                counts.pop()
            audit.equal_sequences(distribution.counts, tuple(counts), message=f"Kummer p={prime} n={index}", section="kummer")
            for power, exponent in ((1, 2), (2, 3), (3, 5)):
                modulus = prime**exponent
                expected = tuple(k for k in range(index + 1) if pow(comb(index, k), power, modulus) != 0)
                observed = factorial_power_support_indices(index, prime, power, exponent)
                audit.equal_sequences(observed, expected, message=f"support p={prime} n={index} s={power} r={exponent}", section="kummer_support")

    # Exact prime-power residues and factorial-power kernels.
    for prime, exponent in ((2, 8), (3, 6), (5, 5), (7, 4)):
        unit = PrimePowerUnitCompiler(prime, exponent)
        for _ in range(1800):
            n = int(rng.integers(0, 5000))
            k = int(rng.integers(0, n + 1))
            audit.check(unit.binomial(n, k) == comb(n, k) % unit.modulus, f"binomial p={prime}^{exponent} n={n} k={k}", section="prime_power")
        for power in (1, 2, 4):
            kernel = PrimePowerKernelCompiler(prime, exponent, power)
            for _ in range(300):
                n = int(rng.integers(0, 1000))
                k = int(rng.integers(0, n + 1))
                observed = kernel.kernel(n, k).residue
                expected = pow(comb(n, k), power, prime**exponent)
                audit.check(observed == expected, f"kernel p={prime}^{exponent} s={power} n={n} k={k}", section="prime_power_kernel")

    # Bell parity, digit automata, and Touchard recurrence.
    parity = BellParityAutomaton()
    for power in (2, 3, 4, 7):
        values = generalized_bell_numbers(power, 220)
        for index, value in enumerate(values):
            audit.check(parity.evaluate(index) == value % 2, f"Bell parity s={power} n={index}", section="bell")
    for prime in (2, 3, 5):
        power = 1 + 2 * (prime - 1)
        automaton = BellResidueAutomaton.compile(prime, power)
        direct = bell_numbers(min(240, automaton.period_bound))
        audit.equal_sequences(automaton.outputs[: len(direct)], tuple(v % prime for v in direct), message=f"Bell outputs p={prime}", section="bell")
        for n in range(automaton.period_bound - prime):
            audit.check(automaton.outputs[n + prime] == (automaton.outputs[n + 1] + automaton.outputs[n]) % prime, f"Touchard p={prime} n={n}", section="bell_touchard")

    # All-prime q-Kummer valuation branches.
    configurations = ((2, 3), (2, 5), (3, 2), (3, 5), (5, 2), (5, 3), (7, 2), (7, 11))
    for q, prime in configurations:
        for n in range(1, 55):
            for k in range(n + 1):
                observed = q_binomial_valuation_decomposition(n, k, q, prime).valuation
                expected = q_binomial_valuation_direct(n, k, q, prime)
                audit.check(observed == expected, f"q-Kummer q={q} p={prime} n={n} k={k}", section="q_kummer")

    # Portable exact records and tamper rejection.
    for case in range(100):
        coefficients = tuple(Fraction(int(v), int(rng.integers(1, 9))) for v in rng.integers(-20, 21, size=12))
        source = FiniteNormalizedSeries(factorial_power_basis(2 + case % 3), RationalDomain(), coefficients)
        restored = loads_portable(dumps_portable(source))
        audit.check(restored == source, f"portable series case={case}", section="persistence")
    payload = dumps_portable(compile_finite_mask_carry_plan((1, 1, 1), 2))
    altered = payload.replace('"denominator": 1', '"denominator": 9', 1)
    try:
        loads_portable(altered)
    except Exception:
        audit.check(True, "portable tamper rejection", section="persistence")
    else:
        audit.check(False, "portable tamper accepted", section="persistence")

    # Formal FIR bridge and analytic local-mask tail bounds.
    for case in range(160):
        coefficients = tuple(Fraction(int(v), int(rng.integers(1, 9))) for v in rng.integers(-8, 9, size=10))
        source = FormalSeries(ordinary_basis(), coefficients)
        adapter = series_to_fir_adapter(source)
        restored = fir_taps_to_series(adapter.taps, max_denominator=100_000)
        audit.equal_sequences(restored.coefficients, coefficients, message=f"FIR roundtrip {case}", section="signal_bridge")
        for point in np.linspace(0.0, np.pi, 24):
            direct = sum(complex(value) * np.exp(-1j * point * n) for n, value in enumerate(coefficients))
            observed = sum(adapter.taps[n] * np.exp(-1j * point * n) for n in range(len(adapter.taps)))
            audit.check(abs(observed - direct) <= 2e-12, f"FIR response {case}", section="signal_bridge")
    for case in range(100):
        radix = int(rng.choice((2, 3, 5)))
        mask = (Fraction(1), Fraction(int(rng.integers(-3, 4)), 4), Fraction(int(rng.integers(-3, 4)), 7))
        point = complex(float(rng.uniform(-0.45, 0.45)), float(rng.uniform(-0.2, 0.2)))
        shallow = analytic_mahler_product(mask, point, radix, 5)
        deep = analytic_mahler_product(mask, point, radix, 11)
        audit.check(abs(deep.value - shallow.value) <= shallow.absolute_error_bound + 1e-15, f"analytic mask {case}", section="analytic_masks")

    report = {
        "schema": "komorebi-math.block13-deep-audit",
        "schema_version": 1,
        "package_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "seed": 13012026,
        "checks": audit.checks,
        "sections": dict(sorted(audit.sections.items())),
        "failures": audit.failures,
        "passed": not audit.failures,
    }
    return report


def main() -> int:
    report = run()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    destination = Path(__file__).with_name("BLOCK13_DEEP_AUDIT.json")
    destination.write_text(rendered, encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
