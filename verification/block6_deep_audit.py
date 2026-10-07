"""Deterministic independent checks for the Block 6 arithmetic and sequence release."""

from __future__ import annotations

import ast
import json
from collections import defaultdict
from fractions import Fraction
from math import comb, factorial, gcd
from pathlib import Path
from typing import Any

import numpy as np

from komorebi_math.arithmetic import (
    EulerLocalCoordinates,
    HurwitzSeries,
    LinearRecurrence,
    MahlerSeries,
    StandardWindow,
    TruncatedPadic,
    affine_kernel_composition_certificate,
    affine_mahler_kernel,
    binomial_valuation,
    binomial_valuation_from_digits,
    carmichael_exponent_prime_power,
    compressed_hurwitz_inverse,
    digit_sum,
    dyadic_kernel_valuation,
    finite_difference_value,
    hurwitz_inverse,
    hurwitz_product,
    local_unit_jump,
    mahler_polyphase,
    matrix_mahler_coefficients,
    evaluate_matrix_mahler,
    padic_valuation,
    polyphase_analyze,
    polyphase_synthesize,
    prime_power_kernel_certificate,
    signed_euler_number,
    simultaneous_unit_difference,
    subgroup_selector,
)
from komorebi_math.sequences import (
    Sequence,
    apery_recurrence,
    apery_sequence,
    bell_sequence,
    binomial_transform_values,
    catalan_sequence,
    compile_sequence,
    detect_linear_recurrence,
    fibonacci_sequence,
    inverse_binomial_transform_values,
    motzkin_recurrence,
    partition_sequence,
    thue_morse_automaton,
    thue_morse_sequence,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "komorebi_math"


class Audit:
    """Count deterministic checks and retain a bounded failure report."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = defaultdict(int)
        self.failures: list[dict[str, Any]] = []

    def check(self, group: str, condition: bool, detail: str = "") -> None:
        self.counts[group] += 1
        if not bool(condition):
            self.failures.append({"group": group, "detail": detail or "condition failed"})

    def close(self) -> dict[str, Any]:
        total = sum(self.counts.values())
        return {
            "passed": not self.failures,
            "total_checks": total,
            "passed_checks": total - len(self.failures),
            "failed_checks": len(self.failures),
            "groups": dict(sorted(self.counts.items())),
            "failures": self.failures[:100],
        }


def independent_euler_numbers(maximum: int) -> tuple[int, ...]:
    """Compute signed Euler numbers by independently inverting cosh."""

    values = [0] * (maximum + 1)
    values[0] = 1
    for n in range(2, maximum + 1, 2):
        values[n] = -sum(comb(n, k) * values[k] for k in range(0, n, 2))
    return tuple(values)


def direct_hurwitz(left: tuple[Fraction, ...], right: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    """Return the full finite Hurwitz product by the definition."""

    degree = len(left) + len(right) - 2
    return tuple(
        sum(
            Fraction(comb(n, r)) * left[r] * right[n - r]
            for r in range(n + 1)
            if r < len(left) and n - r < len(right)
        )
        for n in range(degree + 1)
    )


def brute_recurrence_terms(coefficients: tuple[int, ...], initial: tuple[int, ...], count: int) -> tuple[int, ...]:
    """Generate recurrence terms without using the production recurrence class."""

    values = list(initial)
    order = len(coefficients)
    while len(values) < count:
        start = len(values) - order
        values.append(sum(coefficients[j] * values[start + j] for j in range(order)))
    return tuple(values[:count])


def run() -> dict[str, Any]:
    """Execute the complete deterministic Block 6 audit."""

    audit = Audit()
    rng = np.random.default_rng(20260815)
    primes = (2, 3, 5, 7, 11, 13, 17, 19)

    # Exact valuations and local domains.
    for prime in primes:
        for n in range(0, 80):
            audit.check("valuations", digit_sum(n, prime) >= 0)
            for k in range(n + 1):
                left = binomial_valuation(n, k, prime)
                right = binomial_valuation_from_digits(n, k, prime)
                audit.check("valuations", left == right, f"p={prime} n={n} k={k}")
        for depth in range(1, 7):
            exponent = carmichael_exponent_prime_power(prime, depth)
            modulus = prime**depth
            for unit in range(1, min(modulus, 100)):
                if gcd(unit, prime) == 1:
                    audit.check("valuations", pow(unit, exponent, modulus) == 1)
            local = TruncatedPadic(prime, depth)
            audit.check("valuations", local.modulus == modulus)
            audit.check("valuations", local.refine().coarsen(depth) == local)

    # Hurwitz products, inverses, modes, and finite field selectors.
    for _ in range(350):
        left = tuple(Fraction(int(x), int(y)) for x, y in zip(rng.integers(-5, 6, 5), rng.integers(1, 7, 5)))
        right = tuple(Fraction(int(x), int(y)) for x, y in zip(rng.integers(-5, 6, 4), rng.integers(1, 7, 4)))
        audit.check("hurwitz", hurwitz_product(left, right) == direct_hurwitz(left, right))
        source = (Fraction(1),) + left[1:]
        inverse = tuple(Fraction(value) for value in hurwitz_inverse(source, 12))
        product = hurwitz_product(source, inverse, degree=12)
        audit.check("hurwitz", product == (Fraction(1),) + (Fraction(0),) * 12)
    for prime in primes:
        for order in range(1, prime):
            if (prime - 1) % order:
                continue
            selector = subgroup_selector(prime, order, 4 * prime)
            audit.check(
                "hurwitz",
                selector.coefficients == tuple(1 if n % order == 0 else 0 for n in range(4 * prime + 1)),
            )
            inverse = selector.inverse(4 * prime)
            audit.check(
                "hurwitz",
                selector.multiply(inverse, degree=4 * prime).coefficients == (1,) + (0,) * (4 * prime),
            )
            mode_a = HurwitzSeries.exponential_mode(2, 20, modulus=prime)
            mode_b = HurwitzSeries.exponential_mode(3, 20, modulus=prime)
            audit.check(
                "hurwitz",
                mode_a.multiply(mode_b, degree=20).coefficients
                == HurwitzSeries.exponential_mode(5, 20, modulus=prime).coefficients,
            )

    # Standard windows and address independence.
    for prime in primes[1:]:
        for compression in range(1, prime):
            if (prime - 1) % compression:
                continue
            window = StandardWindow(prime, compression)
            period = (prime - 1) // compression
            degree = 2 * period + 2
            reduced = window.reduced_inverse(degree)
            audit.check("standard_windows", reduced == compressed_hurwitz_inverse(prime, compression, degree))
            for n in range(1, degree - period + 1):
                audit.check("standard_windows", reduced[n + period] == reduced[n])
            addresses = sorted({0, min(1, prime - 1), prime - 1})
            for address in addresses:
                audit.check("standard_windows", window.residues(address, degree) == reduced)
            generating = window.rational_generating_function()
            audit.check("standard_windows", generating.first(degree + 1) == reduced)
            audit.check("standard_windows", period % window.minimum_period() == 0)

    # Unified odd and dyadic Euler charts.
    independent = independent_euler_numbers(44)
    for prime in (2, 3, 5, 7, 11, 13):
        coordinates = EulerLocalCoordinates(prime)
        for n in range(0, 45, 2):
            audit.check("euler", coordinates.reconstruct(n) == independent[n], f"p={prime} n={n}")
            audit.check("euler", coordinates.inverse_reconstruct_source(n) == coordinates.source(n))
            for r in range(n // 2 + 1):
                audit.check("euler", coordinates.row_weight(n, r) == coordinates.formula_row_weight(n, r))
            for depth in range(0, n // 2 + 1):
                result = coordinates.truncate(n, depth)
                audit.check("euler", result.certificate.valid)
                if result.modulus is not None:
                    audit.check("euler", padic_valuation(result.error, prime) >= padic_valuation(result.modulus, prime))
    for depth in range(0, 10):
        n = 2 * depth + 2
        result = EulerLocalCoordinates(2).truncate(n, depth)
        audit.check("euler", result.modulus == 2 ** (4 * depth + 3))
        audit.check("euler", padic_valuation(result.error, 2) == 4 * depth + 3)
        audit.check("euler", dyadic_kernel_valuation(depth + 1) == 4 * (depth + 1) - 1)

    # Mahler analysis, affine transfer, local unit modes, and matrix modes.
    for _ in range(220):
        degree = int(rng.integers(0, 8))
        polynomial = tuple(int(value) for value in rng.integers(-8, 9, degree + 1))

        def evaluate(x: int) -> int:
            return sum(polynomial[j] * x**j for j in range(len(polynomial)))

        values = tuple(evaluate(n) for n in range(degree + 1))
        series = MahlerSeries.from_values(values)
        for n in range(0, 16):
            audit.check("mahler", series.evaluate(n) == evaluate(n))
        multiplier = int(rng.integers(1, 6))
        shift = int(rng.integers(0, 8))
        transformed = series.affine_subsequence(multiplier, shift)
        for n in range(0, 12):
            audit.check("mahler", transformed.evaluate(n) == evaluate(multiplier * n + shift))
        packet = mahler_polyphase(series, multiplier, shift % multiplier)
        for n in range(0, 12):
            audit.check("mahler", packet.evaluate(n) == evaluate(multiplier * n + shift % multiplier))
    for multiplier in range(1, 7):
        for shift in range(0, 7):
            for q in range(0, 9):
                for source_degree in range(0, 14):
                    left = sum(
                        affine_mahler_kernel(multiplier, shift, source_degree, target_degree) * comb(q, target_degree)
                        for target_degree in range(source_degree + 1)
                    )
                    right = comb(multiplier * q + shift, source_degree) if source_degree <= multiplier * q + shift else 0
                    audit.check("mahler", left == right)
    for first in range(1, 5):
        for second in range(1, 5):
            affine_kernel_composition_certificate(first, 2, second, 1, 10).require()
            audit.check("mahler", True)
    for prime in (2, 3, 5, 7):
        for exponent in (1, 2, 3):
            for shift in range(min(prime**exponent, 5)):
                audit.check("mahler", prime_power_kernel_certificate(prime, exponent, shift, 14).valid)
    fibonacci_matrix = np.array([[1, 1], [1, 0]], dtype=object)
    coefficients = matrix_mahler_coefficients(fibonacci_matrix, 12, 5, 14)
    for x in range(15):
        audit.check(
            "mahler",
            np.array_equal(evaluate_matrix_mahler(coefficients, x), np.linalg.matrix_power(fibonacci_matrix, 12 * x + 5)),
        )
    for base in (5, 7, 11, 13):
        result = simultaneous_unit_difference(base, {2: 4, 3: 3}, index=5, order=4)
        audit.check("mahler", result.certificate.valid)
        audit.check("mahler", result.value[2] >= 16 and result.value[3] >= 12)
        jump = local_unit_jump(3, 4)
        value = finite_difference_value(lambda n, u=base: u**n, 7, jump, 3)
        audit.check("mahler", padic_valuation(value, 3) >= 12)

    # Arithmetic polyphase.
    for length in range(0, 80):
        values = tuple(range(length))
        for modulus in range(1, 10):
            packets = polyphase_analyze(values, modulus)
            audit.check("polyphase", packets.reconstruct() == values)
            audit.check("polyphase", polyphase_synthesize(packets.channels) == values)

    # Constant coefficient recurrences and residue packets.
    for _ in range(180):
        order = int(rng.integers(1, 5))
        coefficients = tuple(int(value) for value in rng.integers(-3, 4, order))
        initial = tuple(int(value) for value in rng.integers(-5, 6, order))
        recurrence = LinearRecurrence(coefficients, initial)
        expected = brute_recurrence_terms(coefficients, initial, 48)
        audit.check("recurrences", recurrence.first(48) == expected)
        for _ in range(4):
            index = int(rng.integers(0, 140))
            modulus = int(rng.integers(2, 100))
            expected_mod = brute_recurrence_terms(coefficients, initial, index + 1)[index] % modulus
            audit.check("recurrences", recurrence.term_mod(index, modulus) == expected_mod)
        step = int(rng.integers(1, 9))
        address = int(rng.integers(0, step))
        packet = recurrence.residue_packet(step, address)
        for k in range(12):
            audit.check("recurrences", packet.term(k) == recurrence.term(step * k + address))
            audit.check("recurrences", packet.term_mod(k, 37) == packet.term(k) % 37)
    fibonacci = fibonacci_sequence()
    detected = detect_linear_recurrence(fibonacci.first(30), max_order=8)
    audit.check("recurrences", detected is not None and detected.order == 2)
    if detected is not None:
        audit.check("recurrences", detected.first(100) == fibonacci.first(100))

    # Automatic and holonomic sequence models.
    automaton = thue_morse_automaton()
    audit.check("sequence_models", automaton.certificate(4096).certificate.valid)
    for exponent in range(0, 7):
        modulus = 2**exponent
        for address in range(modulus):
            packet = automaton.residue_model(exponent, address)
            for n in range(32):
                audit.check("sequence_models", packet.term(n) == automaton.term(modulus * n + address))
    for model in (motzkin_recurrence(), apery_recurrence()):
        audit.check("sequence_models", model.certificate(40).certificate.valid)
    for sequence, strategy in (
        (fibonacci_sequence(), "declared linear recurrence"),
        (thue_morse_sequence(), "declared finite automaton"),
        (apery_sequence(), "declared holonomic recurrence"),
    ):
        compiled = compile_sequence(sequence)
        audit.check("sequence_models", compiled.strategy == strategy)
        audit.check("sequence_models", compiled.certificate.valid)
        audit.check("sequence_models", compiled.first(40) == sequence.first(40))

    # Independent catalog and transform checks.
    catalan = catalan_sequence()
    for n in range(100):
        audit.check("sequence_catalog", catalan.term(n) == comb(2 * n, n) // (n + 1))
    bell = bell_sequence()
    for n in range(1, 30):
        audit.check("sequence_catalog", bell.term(n + 1) == sum(comb(n, k) * bell.term(k) for k in range(n + 1)))
    partitions = partition_sequence()
    audit.check(
        "sequence_catalog",
        partitions.first(20)
        == (1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77, 101, 135, 176, 231, 297, 385, 490),
    )
    for _ in range(250):
        values = tuple(int(value) for value in rng.integers(-20, 21, int(rng.integers(1, 30))))
        audit.check(
            "sequence_catalog",
            inverse_binomial_transform_values(binomial_transform_values(values)) == values,
        )
    squares = Sequence(lambda n: n * n, name="squares")
    audit.check("sequence_catalog", squares.residue(5, 3).first(40) == tuple((5 * n + 3) ** 2 for n in range(40)))

    # Source and public API integrity.
    python_files = tuple(sorted(SRC.rglob("*.py")))
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            audit.check("source_integrity", True)
        except SyntaxError as exc:
            audit.check("source_integrity", False, f"{path}: {exc}")
    for package_name in ("arithmetic", "sequences"):
        module = __import__(f"komorebi_math.{package_name}", fromlist=["*"])
        audit.check("source_integrity", len(module.__all__) == len(set(module.__all__)), f"duplicate {package_name} exports")
        for name in module.__all__:
            audit.check("source_integrity", hasattr(module, name), f"missing {package_name}.{name}")

    return audit.close()


if __name__ == "__main__":
    report = run()
    output = Path(__file__).with_name("BLOCK6_DEEP_AUDIT.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)
