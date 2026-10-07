"""Finite congruence discovery and exact verification tools."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb
from typing import Callable, Iterable, Sequence

from .basis import WeightedMonomialBasis, factorial_power_basis
from .sequences import (
    bell_numbers,
    generalized_bell_numbers,
    generalized_bell_polynomials,
    weighted_generalized_bell_polynomials,
)


def is_prime(value: int) -> bool:
    """Return whether ``value`` is a prime integer."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def p_adic_valuation_int(value: int, prime: int) -> int | float:
    """Return ``v_p(value)`` with ``v_p(0)=infinity``."""

    if not is_prime(prime):
        raise ValueError("prime must be prime")
    if value == 0:
        return float("inf")
    n = abs(int(value))
    valuation = 0
    while n % prime == 0:
        valuation += 1
        n //= prime
    return valuation


def p_adic_valuation_fraction(value: int | Fraction, prime: int) -> int | float:
    r"""Return the exact ``p``-adic valuation of a rational, with infinity at zero."""

    rational = Fraction(value)
    if rational == 0:
        return float("inf")
    numerator = p_adic_valuation_int(rational.numerator, prime)
    denominator = p_adic_valuation_int(rational.denominator, prime)
    return int(numerator) - int(denominator)


def rational_mod(value: int | Fraction, modulus: int) -> int:
    """Reduce a rational whose denominator is invertible modulo ``modulus``."""

    rational = Fraction(value)
    if modulus < 2:
        raise ValueError("modulus must be at least two")
    if _gcd(rational.denominator, modulus) != 1:
        raise ZeroDivisionError("rational denominator is not invertible modulo the modulus")
    return rational.numerator * pow(rational.denominator, -1, modulus) % modulus


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def base_digit_sum(index: int, base: int) -> int:
    """Return the sum of the nonnegative integer index digits in the declared base."""

    if index < 0 or base < 2:
        raise ValueError("invalid digit-sum input")
    total = 0
    n = index
    while n:
        n, digit = divmod(n, base)
        total += digit
    return total


def mahler_coefficients(values: Iterable[int | Fraction]) -> tuple[Fraction, ...]:
    """Return finite Newton/Mahler coefficients ``Delta^r f(0)``."""

    row = [Fraction(value) for value in values]
    if not row:
        return ()
    coefficients = []
    while row:
        coefficients.append(row[0])
        row = [row[index + 1] - row[index] for index in range(len(row) - 1)]
    return tuple(coefficients)


def step_differences(values: Sequence[int | Fraction], step: int, order: int) -> tuple[Fraction, ...]:
    """Return the finite ``order``th forward difference at a declared step."""

    if step < 1 or order < 0:
        raise ValueError("step must be positive and order nonnegative")
    row = [Fraction(value) for value in values]
    for _ in range(order):
        if len(row) <= step:
            return ()
        row = [row[index + step] - row[index] for index in range(len(row) - step)]
    return tuple(row)


@dataclass(frozen=True, slots=True)
class DifferenceValuationProfile:
    """Finite stepped-difference valuation profile."""

    prime: int
    step: int
    valuations: tuple[int | float, ...]
    sample_count: int


def difference_valuation_profile(
    values: Sequence[int | Fraction],
    *,
    prime: int,
    step: int,
    max_order: int,
) -> DifferenceValuationProfile:
    r"""Measure minimum ``p``-adic valuations of successive stepped differences."""

    valuations: list[int | float] = []
    for order in range(max_order + 1):
        differences = step_differences(values, step, order)
        if not differences:
            break
        valuations.append(min(p_adic_valuation_fraction(value, prime) for value in differences))
    return DifferenceValuationProfile(prime, step, tuple(valuations), len(values))


@dataclass(frozen=True, slots=True)
class ResidueLaneFinding:
    """One constant residue-lane finding on a finite prefix."""

    modulus: int
    lane_modulus: int
    residue: int
    observed_terms: int
    constant_residue: int | None
    all_zero: bool


def scan_residue_lanes(
    values: Sequence[int],
    *,
    modulus: int,
    lane_modulus: int,
    minimum_terms: int = 8,
) -> tuple[ResidueLaneFinding, ...]:
    """Find residue lanes that are constant modulo ``modulus`` on the prefix."""

    if modulus < 2 or lane_modulus < 1:
        raise ValueError("invalid modulus")
    findings = []
    for residue in range(lane_modulus):
        lane = [values[n] % modulus for n in range(residue, len(values), lane_modulus)]
        if len(lane) < minimum_terms:
            continue
        if len(set(lane)) == 1:
            findings.append(
                ResidueLaneFinding(
                    modulus,
                    lane_modulus,
                    residue,
                    len(lane),
                    lane[0],
                    lane[0] == 0,
                )
            )
    return tuple(findings)


def minimal_prefix_period(values: Sequence[int], modulus: int, *, preperiod_limit: int = 0) -> tuple[int, int] | None:
    """Return a finite-prefix preperiod/period certificate when visible."""

    reduced = tuple(value % modulus for value in values)
    for preperiod in range(preperiod_limit + 1):
        tail = reduced[preperiod:]
        for period in range(1, len(tail) // 3 + 1):
            if all(tail[n] == tail[n % period] for n in range(len(tail))):
                return preperiod, period
    return None


def berlekamp_massey(values: Sequence[int], prime: int) -> tuple[int, tuple[int, ...]]:
    """Return the minimal recurrence over a prime field for the supplied prefix."""

    if not is_prime(prime):
        raise ValueError("Berlekamp-Massey requires a prime modulus")
    sequence = [value % prime for value in values]
    connection = [1]
    previous = [1]
    length = 0
    shift = 1
    discrepancy_scale = 1
    for n in range(len(sequence)):
        discrepancy = sequence[n]
        for i in range(1, length + 1):
            discrepancy = (discrepancy + connection[i] * sequence[n - i]) % prime
        if discrepancy == 0:
            shift += 1
            continue
        old = connection[:]
        factor = discrepancy * pow(discrepancy_scale, -1, prime) % prime
        if len(connection) < len(previous) + shift:
            connection.extend(0 for _ in range(len(previous) + shift - len(connection)))
        for j, coefficient in enumerate(previous):
            connection[j + shift] = (connection[j + shift] - factor * coefficient) % prime
        if 2 * length <= n:
            length = n + 1 - length
            previous = old
            discrepancy_scale = discrepancy
            shift = 1
        else:
            shift += 1
    recurrence = tuple((-connection[i]) % prime for i in range(1, length + 1))
    return length, recurrence


def verify_modular_recurrence(values: Sequence[int], prime: int, coefficients: Sequence[int]) -> bool:
    """Verify a declared linear recurrence modulo a prime on a prefix."""

    order = len(coefficients)
    return all(
        (values[n] - sum(coefficients[j - 1] * values[n - j] for j in range(1, order + 1))) % prime == 0
        for n in range(order, len(values))
    )


def _solve_modular_system(matrix: list[list[int]], target: list[int], prime: int) -> tuple[int, ...] | None:
    if len(matrix) != len(target) or not matrix:
        return None
    columns = len(matrix[0])
    augmented = [[value % prime for value in row] + [rhs % prime] for row, rhs in zip(matrix, target, strict=True)]
    pivot_row = 0
    pivot_columns: list[int] = []
    for column in range(columns):
        found = next((row for row in range(pivot_row, len(augmented)) if augmented[row][column] % prime), None)
        if found is None:
            continue
        augmented[pivot_row], augmented[found] = augmented[found], augmented[pivot_row]
        inverse = pow(augmented[pivot_row][column], -1, prime)
        augmented[pivot_row] = [(value * inverse) % prime for value in augmented[pivot_row]]
        for row in range(len(augmented)):
            if row == pivot_row:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    (augmented[row][index] - factor * augmented[pivot_row][index]) % prime
                    for index in range(columns + 1)
                ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(augmented):
            break
    for row in augmented:
        if all(value == 0 for value in row[:columns]) and row[-1] != 0:
            return None
    solution = [0] * columns
    for row, column in enumerate(pivot_columns):
        solution[column] = augmented[row][-1]
    return tuple(solution)


@dataclass(frozen=True, slots=True)
class DecimationRelation:
    """Finite affine decimation relation modulo a prime."""

    prime: int
    residue: int
    order: int
    coefficients: tuple[int, ...]
    tested_relations: int
    exact_on_prefix: bool


def search_decimation_relation(
    values: Sequence[int],
    *,
    prime: int,
    residue: int,
    max_order: int = 8,
    holdout: int = 12,
) -> DecimationRelation | None:
    r"""Search ``a[p*n+r]=sum_j c_j a[n+j] (mod p)``."""

    if not is_prime(prime) or not 0 <= residue < prime:
        raise ValueError("invalid prime or residue")
    relation_count = (len(values) - 1 - residue) // prime + 1
    for order in range(1, max_order + 1):
        usable = min(relation_count, len(values) - order + 1)
        fit = usable - holdout
        if fit < order:
            continue
        matrix = [[values[n + j] % prime for j in range(order)] for n in range(fit)]
        target = [values[prime * n + residue] % prime for n in range(fit)]
        solution = _solve_modular_system(matrix, target, prime)
        if solution is None:
            continue
        valid = all(
            (values[prime * n + residue] - sum(solution[j] * values[n + j] for j in range(order))) % prime == 0
            for n in range(usable)
        )
        if valid:
            return DecimationRelation(prime, residue, order, solution, usable, True)
    return None


@dataclass(frozen=True, slots=True)
class LucasBasisAudit:
    """Finite audit record for normalized-basis Lucas factorization."""

    basis: str
    prime: int
    tested_cases: int
    failures: int
    lucas_compatible: bool


def audit_generalized_lucas(
    basis: WeightedMonomialBasis,
    *,
    prime: int,
    high_limit: int = 12,
) -> LucasBasisAudit:
    r"""Test ``C(pn+r,pk+i)=C(n,k)C(r,i) (mod p)`` when integral."""

    if not is_prime(prime):
        raise ValueError("prime must be prime")
    tested = 0
    failures = 0
    for n in range(high_limit + 1):
        for r in range(prime):
            total = prime * n + r
            for k in range(n + 1):
                for i in range(prime):
                    low = prime * k + i
                    if low > total:
                        continue
                    left = basis.structure_constant(total, low)
                    if left.denominator != 1:
                        failures += 1
                        tested += 1
                        continue
                    right_low = basis.structure_constant(r, i) if i <= r else Fraction(0)
                    right = basis.structure_constant(n, k) * right_low
                    if right.denominator != 1 or (left.numerator - right.numerator) % prime:
                        failures += 1
                    tested += 1
    return LucasBasisAudit(basis.name, prime, tested, failures, failures == 0)


@dataclass(frozen=True, slots=True)
class ConvolutionPacketAudit:
    """Finite audit record for a digitwise convolution identity."""

    power: int
    prime: int
    tested_cases: int
    failures: int


def audit_factorial_power_packet_convolution(
    left: Sequence[int],
    right: Sequence[int],
    *,
    power: int,
    prime: int,
) -> ConvolutionPacketAudit:
    """Verify the Lucas/polyphase convolution identity modulo a prime."""

    basis = factorial_power_basis(power)
    maximum = min(len(left), len(right)) - 1
    product = []
    for n in range(2 * maximum + 1):
        product.append(
            sum(
                int(basis.structure_constant(n, k)) * (left[k] if k < len(left) else 0) * (right[n - k] if n - k < len(right) else 0)
                for k in range(n + 1)
                if k < len(left) and n - k < len(right)
            )
        )
    tested = 0
    failures = 0
    for n in range((len(product) - 1) // prime + 1):
        for r in range(prime):
            index = prime * n + r
            if index >= len(product):
                continue
            rhs = 0
            for i in range(r + 1):
                low_weight = comb(r, i) ** power
                for k in range(n + 1):
                    left_index = prime * k + i
                    right_index = prime * (n - k) + (r - i)
                    if left_index < len(left) and right_index < len(right):
                        rhs += comb(n, k) ** power * low_weight * left[left_index] * right[right_index]
            if (product[index] - rhs) % prime:
                failures += 1
            tested += 1
    return ConvolutionPacketAudit(power, prime, tested, failures)


@dataclass(frozen=True, slots=True)
class QLucasAudit:
    """Finite audit record for Gaussian-binomial Lucas factorization."""

    q: int
    prime: int
    radix: int
    tested_cases: int
    failures: int


def multiplicative_order_mod(value: int, prime: int) -> int:
    """Return the multiplicative order of ``value`` modulo a prime."""

    if not is_prime(prime) or value % prime == 0:
        raise ValueError("value must be a unit modulo the prime")
    current = 1
    for order in range(1, prime):
        current = current * value % prime
        if current == 1:
            return order
    raise ArithmeticError("multiplicative order was not found")


def audit_q_lucas(*, q: int, prime: int, high_limit: int = 10) -> QLucasAudit:
    r"""Audit the order-radix Gaussian-binomial Lucas law.

    If ``d`` is the multiplicative order of ``q`` modulo ``p``, the tested
    identity is

    ``[ad+b choose rd+s]_q = C(a,r) [b choose s]_q (mod p)``.
    """

    from .basis import q_factorial_basis

    radix = multiplicative_order_mod(q, prime)
    basis = q_factorial_basis(q)
    tested = 0
    failures = 0
    for a in range(high_limit + 1):
        for b in range(radix):
            n = radix * a + b
            for r in range(a + 1):
                for s in range(radix):
                    k = radix * r + s
                    if k > n:
                        continue
                    left = basis.structure_constant(n, k)
                    low = basis.structure_constant(b, s) if s <= b else Fraction(0)
                    right = Fraction(comb(a, r)) * low
                    if left.denominator != 1 or right.denominator != 1 or (left.numerator - right.numerator) % prime:
                        failures += 1
                    tested += 1
    return QLucasAudit(q, prime, radix, tested, failures)


@dataclass(frozen=True, slots=True)
class DigitSumCongruenceAudit:
    """Finite audit record for a Bell digit-sum congruence."""

    prime: int
    power: int
    tested_indices: int
    tested_polynomial_coefficients: int
    number_failures: int
    polynomial_failures: int


def audit_generalized_bell_digit_sum(
    *,
    prime: int,
    power: int,
    count: int = 300,
) -> DigitSumCongruenceAudit:
    r"""Verify ``B_n^(s)(x)=B_(s_p(n))(x) (mod p)`` on a finite range."""

    if not is_prime(prime):
        raise ValueError("prime must be prime")
    if power < 2 or (power - 1) % (prime - 1):
        raise ValueError("audit requires power >= 2 and power-1 divisible by prime-1")
    custom_numbers = generalized_bell_numbers(power, count)
    custom_polynomials = generalized_bell_polynomials(power, count)
    maximum_digit_sum = max(base_digit_sum(n, prime) for n in range(count))
    standard_numbers = bell_numbers(maximum_digit_sum + 1)
    standard_polynomials = generalized_bell_polynomials(1, maximum_digit_sum + 1)
    number_failures = 0
    polynomial_failures = 0
    coefficient_tests = 0
    for n in range(count):
        target = base_digit_sum(n, prime)
        if (custom_numbers[n] - standard_numbers[target]) % prime:
            number_failures += 1
        custom_row = custom_polynomials[n]
        standard_row = standard_polynomials[target]
        degree = max(len(custom_row), len(standard_row))
        for k in range(degree):
            left = custom_row[k] if k < len(custom_row) else 0
            right = standard_row[k] if k < len(standard_row) else 0
            if (left - right) % prime:
                polynomial_failures += 1
            coefficient_tests += 1
    return DigitSumCongruenceAudit(
        prime,
        power,
        count,
        coefficient_tests,
        number_failures,
        polynomial_failures,
    )


@dataclass(frozen=True, slots=True)
class WeightedDigitSumCongruenceAudit:
    """Finite audit for the custom weighted digit-sum reduction theorem."""

    prime: int
    power: int
    tested_indices: int
    tested_polynomial_coefficients: int
    failures: int


def audit_weighted_generalized_bell_digit_sum(
    *,
    prime: int,
    power: int,
    digit_weights: Sequence[int],
    count: int = 300,
) -> WeightedDigitSumCongruenceAudit:
    r"""Audit a custom weighted factorial-power Bell congruence.

    Let ``c_m = d_(s_p(m))`` and define

    ``F(z,x)=exp(x*sum_(m>=1) c_m z^m/(m!)^power)``.

    When ``power >= 2`` and ``power-1`` is divisible by ``prime-1``, the
    normalized coefficient polynomial at index ``n`` is predicted to equal,
    modulo ``prime``, the ordinary weighted Bell polynomial at
    ``s_p(n)`` with block weights ``d_t``.
    """

    if not is_prime(prime):
        raise ValueError("prime must be prime")
    if power < 2 or (power - 1) % (prime - 1):
        raise ValueError("audit requires power >= 2 and power-1 divisible by prime-1")
    if count < 1:
        raise ValueError("count must be positive")
    maximum_digit_sum = max(base_digit_sum(n, prime) for n in range(count))
    table = tuple(int(value) for value in digit_weights)
    if len(table) <= maximum_digit_sum:
        raise ValueError("digit weight table is too short for the requested range")

    high_rows = weighted_generalized_bell_polynomials(
        power,
        count,
        lambda index: table[base_digit_sum(index, prime)],
    )
    low_rows = weighted_generalized_bell_polynomials(
        1,
        maximum_digit_sum + 1,
        table,
    )
    failures = 0
    coefficient_tests = 0
    for index, high_row in enumerate(high_rows):
        low_row = low_rows[base_digit_sum(index, prime)]
        degree = max(len(high_row), len(low_row))
        for coefficient_index in range(degree):
            high_value = high_row[coefficient_index] if coefficient_index < len(high_row) else 0
            low_value = low_row[coefficient_index] if coefficient_index < len(low_row) else 0
            if (high_value - low_value).numerator % prime:
                failures += 1
            coefficient_tests += 1
    return WeightedDigitSumCongruenceAudit(
        prime,
        power,
        count,
        coefficient_tests,
        failures,
    )


@dataclass(frozen=True, slots=True)
class RelationAudit:
    """Finite modular relation audit summary."""

    name: str
    modulus: int
    tested_cases: int
    failures: int


def audit_touchard(values: Sequence[int], prime: int) -> RelationAudit:
    """Verify ``B_(n+p)=B_(n+1)+B_n (mod p)`` on a prefix."""

    tested = max(0, len(values) - prime)
    failures = sum((values[n + prime] - values[n + 1] - values[n]) % prime != 0 for n in range(tested))
    return RelationAudit("Touchard", prime, tested, failures)


def audit_apery_supercongruence(values: Sequence[int], prime: int, exponent: int = 3) -> RelationAudit:
    """Verify ``A_(p*n)=A_n (mod p^exponent)`` on the available prefix."""

    modulus = prime**exponent
    tested = (len(values) - 1) // prime + 1
    failures = sum((values[prime * n] - values[n]) % modulus != 0 for n in range(tested))
    return RelationAudit(f"Apery p^{exponent}", modulus, tested, failures)


def audit_partition_lane(values: Sequence[int], prime: int, residue: int) -> RelationAudit:
    r"""Audit divisibility by ``prime`` on indices ``prime*n + residue``."""

    tested = (len(values) - 1 - residue) // prime + 1 if len(values) > residue else 0
    failures = sum(values[prime * n + residue] % prime != 0 for n in range(tested))
    return RelationAudit("partition lane", prime, tested, failures)


__all__ = [
    "ConvolutionPacketAudit",
    "DecimationRelation",
    "DifferenceValuationProfile",
    "DigitSumCongruenceAudit",
    "LucasBasisAudit",
    "QLucasAudit",
    "RelationAudit",
    "WeightedDigitSumCongruenceAudit",
    "ResidueLaneFinding",
    "audit_apery_supercongruence",
    "audit_factorial_power_packet_convolution",
    "audit_generalized_bell_digit_sum",
    "audit_generalized_lucas",
    "audit_partition_lane",
    "audit_q_lucas",
    "audit_touchard",
    "audit_weighted_generalized_bell_digit_sum",
    "base_digit_sum",
    "berlekamp_massey",
    "difference_valuation_profile",
    "is_prime",
    "mahler_coefficients",
    "multiplicative_order_mod",
    "minimal_prefix_period",
    "p_adic_valuation_fraction",
    "p_adic_valuation_int",
    "rational_mod",
    "scan_residue_lanes",
    "search_decimation_relation",
    "step_differences",
    "verify_modular_recurrence",
]
