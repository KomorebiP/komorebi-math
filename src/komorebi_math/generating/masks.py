"""Local Mahler masks, exact carry compilers, and prime-power support layers."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import ceil, exp, isfinite, isqrt, prod
from typing import Iterable, Mapping, Sequence

from ..certificates import Certificate, Exactness
from ..exceptions import ValidationError
from .budget import DEFAULT_GENERATING_BUDGET, GeneratingBudget
from .padic_units import PrimePowerKernelCompiler, borrow_count

Scalar = int | Fraction
Matrix = tuple[tuple[Fraction, ...], ...]


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor <= isqrt(value):
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def _fractions(values: Iterable[Scalar]) -> tuple[Fraction, ...]:
    return tuple(Fraction(value) for value in values)


def _trim(values: Sequence[Fraction]) -> tuple[Fraction, ...]:
    result = list(values) or [Fraction(0)]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return tuple(result)


def _validate_unit_series(values: Iterable[Scalar], *, name: str) -> tuple[Fraction, ...]:
    normalized = _trim(_fractions(values))
    if normalized[0] != 1:
        raise ValidationError(f"{name} must have constant coefficient one")
    return normalized


def _multiply_truncated(
    left: Sequence[Fraction], right: Sequence[Fraction], degree: int
) -> tuple[Fraction, ...]:
    output = [Fraction(0) for _ in range(degree + 1)]
    for i, left_value in enumerate(left):
        if i > degree or left_value == 0:
            continue
        for j, right_value in enumerate(right):
            if i + j > degree:
                break
            if right_value:
                output[i + j] += left_value * right_value
    return tuple(output)


def _reciprocal_truncated(values: Sequence[Fraction], degree: int) -> tuple[Fraction, ...]:
    if not values or values[0] == 0:
        raise ValidationError("formal reciprocal requires a nonzero constant coefficient")
    output = [Fraction(1, 1) / values[0]]
    for n in range(1, degree + 1):
        output.append(
            -sum(values[k] * output[n - k] for k in range(1, min(n, len(values) - 1) + 1))
            / values[0]
        )
    return tuple(output)


def _radix_digits(index: int, radix: int) -> tuple[int, ...]:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValidationError("index must be a nonnegative integer")
    if isinstance(radix, bool) or not isinstance(radix, int) or radix < 2:
        raise ValidationError("radix must be an integer at least two")
    if index == 0:
        return ()
    digits: list[int] = []
    remaining = index
    while remaining:
        remaining, digit = divmod(remaining, radix)
        digits.append(digit)
    return tuple(digits)


def mahler_mask_product(
    mask: Iterable[Scalar],
    radix: int,
    degree: int,
    *,
    budget: GeneratingBudget = DEFAULT_GENERATING_BUDGET,
) -> tuple[Fraction, ...]:
    r"""Return ``prod_(j>=0) R(z**radix**j)`` through ``z**degree``.

    Only finitely many mask levels affect a finite truncation.  Arithmetic is
    exact over :class:`fractions.Fraction`.
    """

    degree = budget.require_degree(degree)
    if isinstance(radix, bool) or not isinstance(radix, int) or radix < 2:
        raise ValidationError("radix must be an integer at least two")
    coefficients = _validate_unit_series(mask, name="Mahler mask")
    result = (Fraction(1),) + (Fraction(0),) * degree
    scale = 1
    while scale <= degree:
        factor = [Fraction(0) for _ in range(degree + 1)]
        for exponent, coefficient in enumerate(coefficients):
            target = exponent * scale
            if target > degree:
                break
            factor[target] = coefficient
        result = _multiply_truncated(result, factor, degree)
        scale *= radix
    return result


def mahler_mask_inverse(
    series: Iterable[Scalar],
    radix: int,
    degree: int | None = None,
    *,
    budget: GeneratingBudget = DEFAULT_GENERATING_BUDGET,
) -> tuple[Fraction, ...]:
    r"""Recover the unique local mask ``R(z)=F(z)/F(z**radix)``."""

    coefficients = _validate_unit_series(series, name="Mahler series")
    selected_degree = len(coefficients) - 1 if degree is None else degree
    selected_degree = budget.require_degree(selected_degree)
    padded = coefficients[: selected_degree + 1] + (Fraction(0),) * max(
        0, selected_degree + 1 - len(coefficients)
    )
    substituted = [Fraction(0) for _ in range(selected_degree + 1)]
    for index, value in enumerate(padded):
        target = radix * index
        if target > selected_degree:
            break
        substituted[target] = value
    return _multiply_truncated(
        padded,
        _reciprocal_truncated(substituted, selected_degree),
        selected_degree,
    )


def certify_mahler_functional_equation(
    series: Iterable[Scalar],
    mask: Iterable[Scalar],
    radix: int,
    degree: int | None = None,
    *,
    budget: GeneratingBudget = DEFAULT_GENERATING_BUDGET,
) -> Certificate:
    r"""Certify ``F(z)=R(z)F(z**radix)`` on a declared finite truncation."""

    f = _validate_unit_series(series, name="Mahler series")
    r = _validate_unit_series(mask, name="Mahler mask")
    selected_degree = len(f) - 1 if degree is None else budget.require_degree(degree)
    f = f[: selected_degree + 1] + (Fraction(0),) * max(0, selected_degree + 1 - len(f))
    substituted = [Fraction(0) for _ in range(selected_degree + 1)]
    for index, value in enumerate(f):
        target = radix * index
        if target > selected_degree:
            break
        substituted[target] = value
    reconstructed = _multiply_truncated(r, substituted, selected_degree)
    failures = tuple(
        (index, f[index], reconstructed[index])
        for index in range(selected_degree + 1)
        if f[index] != reconstructed[index]
    )
    return Certificate(
        valid=not failures,
        exactness=Exactness.EXACT,
        assumptions_checked=(
            "constant coefficients equal one",
            "finite formal truncation",
        ),
        diagnostics={"degree": selected_degree, "failures": failures[:16]},
        theorem_id="GF-MASK-INVERSE",
        message=(
            f"Mahler functional equation holds through degree {selected_degree}"
            if not failures
            else f"Mahler functional equation failed first at degree {failures[0][0]}"
        ),
    )


@dataclass(frozen=True, slots=True)
class FiniteMaskCarryPlan:
    """Exact carry-matrix representation of a finite local mask."""

    radix: int
    mask: tuple[Fraction, ...]
    carry_bound: int
    matrices: tuple[Matrix, ...]

    @property
    def dimension(self) -> int:
        """Return the carry-state dimension."""

        return self.carry_bound + 1

    def evaluate(self, index: int) -> Fraction:
        """Return one coefficient using least-significant-first radix digits."""

        digits = _radix_digits(index, self.radix)
        vector = [Fraction(0) for _ in range(self.dimension)]
        vector[0] = Fraction(1)
        for digit in digits:
            matrix = self.matrices[digit]
            vector = [
                sum(vector[source] * matrix[source][target] for source in range(self.dimension))
                for target in range(self.dimension)
            ]
        return vector[0]

    def sequence(
        self,
        count: int,
        *,
        budget: GeneratingBudget = DEFAULT_GENERATING_BUDGET,
    ) -> tuple[Fraction, ...]:
        """Return a finite coefficient prefix."""

        budget.require_terms(count, label="coefficient count")
        return tuple(self.evaluate(index) for index in range(count))

    def certificate(
        self,
        count: int,
        *,
        budget: GeneratingBudget = DEFAULT_GENERATING_BUDGET,
    ) -> Certificate:
        """Compare the matrix representation with the direct finite product."""

        budget.require_terms(count, label="certificate coefficient count")
        expected = mahler_mask_product(
            self.mask,
            self.radix,
            max(0, count - 1),
            budget=budget,
        )
        failures = tuple(
            (index, self.evaluate(index), expected[index])
            for index in range(count)
            if self.evaluate(index) != expected[index]
        )
        return Certificate(
            valid=not failures,
            exactness=Exactness.EXACT,
            assumptions_checked=("finite mask", "exact rational arithmetic"),
            diagnostics={
                "count": count,
                "carry_bound": self.carry_bound,
                "failures": failures[:16],
            },
            theorem_id="GF-FINITE-MASK-CARRY",
            message=(
                f"finite-mask carry plan matched {count} coefficients"
                if not failures
                else f"finite-mask carry plan failed first at coefficient {failures[0][0]}"
            ),
        )


def compile_finite_mask_carry_plan(
    mask: Iterable[Scalar],
    radix: int,
    *,
    budget: GeneratingBudget = DEFAULT_GENERATING_BUDGET,
) -> FiniteMaskCarryPlan:
    r"""Compile ``T_d(q,q')=r_(d+radix*q'-q)`` for a finite mask."""

    if isinstance(radix, bool) or not isinstance(radix, int) or radix < 2:
        raise ValidationError("radix must be an integer at least two")
    coefficients = _validate_unit_series(mask, name="finite mask")
    degree = len(coefficients) - 1
    carry_bound = ceil(degree / (radix - 1)) if degree else 0
    dimension = budget.require_matrix_dimension(carry_bound + 1)
    matrices: list[Matrix] = []
    for digit in range(radix):
        rows: list[tuple[Fraction, ...]] = []
        for source in range(dimension):
            row = []
            for target in range(dimension):
                mask_index = digit + radix * target - source
                row.append(
                    coefficients[mask_index]
                    if 0 <= mask_index < len(coefficients)
                    else Fraction(0)
                )
            rows.append(tuple(row))
        matrices.append(tuple(rows))
    return FiniteMaskCarryPlan(radix, coefficients, carry_bound, tuple(matrices))


@dataclass(frozen=True, slots=True)
class DigitCauchyProductPlan:
    """Finite carry representation of an ordinary product of digit-local series."""

    radix: int
    digit_weights: tuple[tuple[Fraction, ...], ...]
    matrices: tuple[Matrix, ...]

    @property
    def factor_count(self) -> int:
        return len(self.digit_weights)

    @property
    def dimension(self) -> int:
        return self.factor_count

    def evaluate(self, index: int) -> Fraction:
        """Evaluate one ordinary Cauchy-product coefficient."""

        digits = _radix_digits(index, self.radix)
        vector = [Fraction(0) for _ in range(self.dimension)]
        vector[0] = Fraction(1)
        for digit in digits:
            matrix = self.matrices[digit]
            vector = [
                sum(vector[source] * matrix[source][target] for source in range(self.dimension))
                for target in range(self.dimension)
            ]
        return vector[0]

    def sequence(
        self,
        count: int,
        *,
        budget: GeneratingBudget = DEFAULT_GENERATING_BUDGET,
    ) -> tuple[Fraction, ...]:
        budget.require_terms(count, label="coefficient count")
        return tuple(self.evaluate(index) for index in range(count))


def compile_digit_cauchy_product(
    digit_weights: Sequence[Sequence[Scalar]],
    radix: int,
    *,
    budget: GeneratingBudget = DEFAULT_GENERATING_BUDGET,
) -> DigitCauchyProductPlan:
    r"""Compile finite carries for a Cauchy product of digit-local sequences."""

    if isinstance(radix, bool) or not isinstance(radix, int) or radix < 2:
        raise ValidationError("radix must be an integer at least two")
    normalized = tuple(_fractions(values) for values in digit_weights)
    if not normalized:
        raise ValidationError("at least one digit-local factor is required")
    if any(len(values) != radix for values in normalized):
        raise ValidationError("each digit-weight table must contain one value per radix digit")
    if any(values[0] != 1 for values in normalized):
        raise ValidationError("each digit-local factor must assign weight one to digit zero")
    factor_count = len(normalized)
    budget.require_matrix_dimension(factor_count)
    combination_count = radix**factor_count
    budget.require_terms(combination_count, label="local digit combinations")
    local: dict[int, Fraction] = {}
    for digits in product(range(radix), repeat=factor_count):
        total = sum(digits)
        weight = prod(normalized[index][digit] for index, digit in enumerate(digits))
        local[total] = local.get(total, Fraction(0)) + weight
    matrices: list[Matrix] = []
    for output_digit in range(radix):
        rows: list[tuple[Fraction, ...]] = []
        for incoming in range(factor_count):
            rows.append(
                tuple(
                    local.get(output_digit + radix * outgoing - incoming, Fraction(0))
                    for outgoing in range(factor_count)
                )
            )
        matrices.append(tuple(rows))
    return DigitCauchyProductPlan(radix, normalized, tuple(matrices))


@dataclass(frozen=True, slots=True)
class KummerLayerDistribution:
    """Counts of subindices grouped by the number of base-prime borrows."""

    index: int
    prime: int
    counts: tuple[int, ...]

    @property
    def total(self) -> int:
        return sum(self.counts)

    def count(self, borrows: int) -> int:
        if borrows < 0:
            return 0
        return self.counts[borrows] if borrows < len(self.counts) else 0

    def support_count(self, maximum_borrows: int) -> int:
        if maximum_borrows < 0:
            return 0
        return sum(self.counts[: maximum_borrows + 1])

    def certificate(self) -> Certificate:
        valid = self.total == self.index + 1
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT,
            assumptions_checked=("prime radix", "finite borrow automaton"),
            diagnostics={"total": self.total, "expected": self.index + 1},
            theorem_id="GF-KUMMER-LAYERS",
            message=(
                "Kummer layer counts exhaust all subindices"
                if valid
                else "Kummer layer counts do not exhaust all subindices"
            ),
        )


def _poly_add_scaled(
    target: list[int], source: Sequence[int], multiplier: int, shift: int
) -> None:
    if multiplier == 0:
        return
    needed = len(source) + shift
    if len(target) < needed:
        target.extend([0] * (needed - len(target)))
    for degree, value in enumerate(source):
        target[degree + shift] += multiplier * value


def kummer_layer_distribution(index: int, prime: int) -> KummerLayerDistribution:
    r"""Return the coefficient polynomial counting ``v_p(binomial(index,k))``."""

    if not _is_prime(prime):
        raise ValidationError("prime must be prime")
    digits = _radix_digits(index, prime) or (0,)
    states: tuple[list[int], list[int]] = ([1], [0])
    for digit in digits:
        next_zero: list[int] = []
        next_one: list[int] = []
        _poly_add_scaled(next_zero, states[0], digit + 1, 0)
        _poly_add_scaled(next_one, states[0], prime - 1 - digit, 1)
        _poly_add_scaled(next_zero, states[1], digit, 0)
        _poly_add_scaled(next_one, states[1], prime - digit, 1)
        states = (next_zero or [0], next_one or [0])
    counts = list(states[0])
    while len(counts) > 1 and counts[-1] == 0:
        counts.pop()
    result = KummerLayerDistribution(index, prime, tuple(counts))
    result.certificate().require()
    return result


def zero_borrow_count(index: int, prime: int) -> int:
    """Return the exact number of no-borrow subindices."""

    if not _is_prime(prime):
        raise ValidationError("prime must be prime")
    return prod(digit + 1 for digit in (_radix_digits(index, prime) or (0,)))


def one_borrow_count(index: int, prime: int) -> int:
    """Return the exact number of subindices producing exactly one borrow."""

    if not _is_prime(prime):
        raise ValidationError("prime must be prime")
    digits = _radix_digits(index, prime) or (0,)
    total = 0
    for position in range(len(digits) - 1):
        factor = (prime - 1 - digits[position]) * digits[position + 1]
        for other, digit in enumerate(digits):
            if other not in (position, position + 1):
                factor *= digit + 1
        total += factor
    return total


def factorial_power_support_count(
    index: int,
    prime: int,
    factorial_power: int,
    modulus_exponent: int,
) -> int:
    r"""Count ``k`` for which ``binomial(index,k)**power`` survives modulo ``p**r``."""

    if factorial_power < 1 or modulus_exponent < 1:
        raise ValidationError("factorial_power and modulus_exponent must be positive")
    maximum_borrows = (modulus_exponent - 1) // factorial_power
    return kummer_layer_distribution(index, prime).support_count(maximum_borrows)


def factorial_power_support_indices(
    index: int,
    prime: int,
    factorial_power: int,
    modulus_exponent: int,
    *,
    budget: GeneratingBudget = DEFAULT_GENERATING_BUDGET,
) -> tuple[int, ...]:
    """Enumerate the surviving subindices under an explicit finite budget."""

    if factorial_power < 1 or modulus_exponent < 1:
        raise ValidationError("factorial_power and modulus_exponent must be positive")
    budget.require_terms(index + 1, label="support scan size")
    maximum_borrows = (modulus_exponent - 1) // factorial_power
    return tuple(
        subindex
        for subindex in range(index + 1)
        if borrow_count(index, subindex, prime) <= maximum_borrows
    )


@dataclass(frozen=True, slots=True)
class CarryLayerConvolution:
    """Prime-power convolution decomposed by exact borrow count."""

    index: int
    prime: int
    exponent: int
    factorial_power: int
    modulus: int
    layer_sums: tuple[int, ...]

    @property
    def residue(self) -> int:
        return sum(self.layer_sums) % self.modulus


def factorial_power_convolution_layers(
    left: Sequence[int],
    right: Sequence[int],
    index: int,
    prime: int,
    factorial_power: int,
    modulus_exponent: int,
    *,
    budget: GeneratingBudget = DEFAULT_GENERATING_BUDGET,
) -> CarryLayerConvolution:
    """Evaluate a factorial-power convolution and retain each carry layer."""

    if index < 0 or len(left) <= index or len(right) <= index:
        raise ValidationError("left and right coefficient tables must contain indices through index")
    if factorial_power < 1 or modulus_exponent < 1:
        raise ValidationError("factorial_power and modulus_exponent must be positive")
    budget.require_terms(index + 1, label="convolution scan size")
    modulus = prime**modulus_exponent
    budget.require_modulus(modulus)
    compiler = PrimePowerKernelCompiler(
        prime,
        modulus_exponent,
        factorial_power,
        max_modulus=budget.maximum_modulus,
    )
    maximum_borrows = (modulus_exponent - 1) // factorial_power
    layers = [0] * (maximum_borrows + 1)
    for subindex in range(index + 1):
        borrow_layer = borrow_count(index, subindex, prime)
        if borrow_layer > maximum_borrows:
            continue
        kernel = compiler.kernel(index, subindex).residue
        layers[borrow_layer] = (
            layers[borrow_layer]
            + kernel * int(left[subindex]) * int(right[index - subindex])
        ) % modulus
    return CarryLayerConvolution(
        index,
        prime,
        modulus_exponent,
        factorial_power,
        modulus,
        tuple(layers),
    )


def frobenius_defect_mask(
    mask: Iterable[Scalar],
    prime: int,
    degree: int,
    *,
    budget: GeneratingBudget = DEFAULT_GENERATING_BUDGET,
) -> tuple[Fraction, ...]:
    r"""Return ``R(z**p)/R(z)**p`` through a finite degree."""

    if not _is_prime(prime):
        raise ValidationError("prime must be prime")
    degree = budget.require_degree(degree)
    r = _validate_unit_series(mask, name="Frobenius mask")
    substituted = [Fraction(0) for _ in range(degree + 1)]
    for index, value in enumerate(r):
        target = prime * index
        if target > degree:
            break
        substituted[target] = value
    powered = (Fraction(1),) + (Fraction(0),) * degree
    base = r[: degree + 1]
    exponent = prime
    while exponent:
        if exponent & 1:
            powered = _multiply_truncated(powered, base, degree)
        base = _multiply_truncated(base, base, degree)
        exponent >>= 1
    return _multiply_truncated(
        substituted,
        _reciprocal_truncated(powered, degree),
        degree,
    )


def first_frobenius_lift(
    mask: Sequence[int], prime: int, degree: int
) -> tuple[int, ...]:
    r"""Return ``(R(z**p)/R(z)**p-1)/p`` modulo ``p`` when integral."""

    defect = frobenius_defect_mask(mask, prime, degree)
    output: list[int] = []
    for index, coefficient in enumerate(defect):
        adjusted = coefficient - (1 if index == 0 else 0)
        if adjusted.denominator != 1 or adjusted.numerator % prime:
            raise ValidationError("Frobenius defect is not divisible by the selected prime")
        output.append((adjusted.numerator // prime) % prime)
    return tuple(output)


def polynomial_mask_coboundary(
    candidate: Iterable[Scalar], radix: int, degree: int | None = None
) -> tuple[Fraction, ...]:
    r"""Return the mask ``Q(z)/Q(z**radix)`` of a declared polynomial unit."""

    q = _validate_unit_series(candidate, name="coboundary candidate")
    selected_degree = len(q) * radix - 1 if degree is None else degree
    return mahler_mask_inverse(q, radix, selected_degree)


@dataclass(frozen=True, slots=True)
class AnalyticMaskApproximation:
    """Finite analytic Mahler product with a conservative tail bound."""

    radix: int
    point: complex
    terms: int
    value: complex
    relative_tail_bound: float
    absolute_error_bound: float

    def certificate(self) -> Certificate:
        """Return the numerical convergence certificate."""

        valid = isfinite(self.relative_tail_bound) and isfinite(
            self.absolute_error_bound
        )
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"absolute_tail_bound": self.absolute_error_bound},
            tolerance=self.absolute_error_bound,
            assumptions_checked=(
                "polynomial mask has constant coefficient one",
                "evaluation point lies inside the unit disk",
            ),
            diagnostics={
                "radix": self.radix,
                "point": str(self.point),
                "terms": self.terms,
                "relative_tail_bound": self.relative_tail_bound,
            },
            theorem_id="GF-ANALYTIC-MASK-REALIZATION",
            message=(
                "analytic Mahler product has a finite tail bound"
                if valid
                else "the selected truncation is too shallow for this bound"
            ),
        )


def analytic_mahler_product(
    mask: Sequence[int | float | complex | Fraction],
    point: complex,
    radix: int,
    terms: int,
) -> AnalyticMaskApproximation:
    r"""Evaluate a finite analytic scale product with an explicit tail bound.

    For ``R(z)=1+sum_(k>=1) r_k z^k`` and ``|z|<1``, the implementation uses
    ``|R(w)-1| <= C|w|`` with ``C=sum_(k>=1)|r_k|``.  The resulting bound is
    conservative.  It is numerical rather than interval-rigorous because the
    product itself is evaluated with ordinary complex floating arithmetic.
    """

    if isinstance(radix, bool) or not isinstance(radix, int) or radix < 2:
        raise ValidationError("radix must be an integer at least two")
    if isinstance(terms, bool) or not isinstance(terms, int) or terms < 0:
        raise ValidationError("terms must be a nonnegative integer")
    coefficients = tuple(complex(value) for value in mask)
    if not coefficients or coefficients[0] != 1:
        raise ValidationError("analytic mask must have constant coefficient one")
    z = complex(point)
    rho = abs(z)
    if not rho < 1:
        raise ValidationError("analytic Mahler products require |point| < 1")
    value = 1 + 0j
    for level in range(terms):
        argument = z ** (radix**level)
        value *= sum(
            coefficient * argument**degree
            for degree, coefficient in enumerate(coefficients)
        )
    coefficient_bound = sum(abs(coefficient) for coefficient in coefficients[1:])
    first_tail = rho ** (radix**terms)
    if coefficient_bound == 0 or first_tail == 0:
        relative = 0.0
    else:
        maximum_delta = coefficient_bound * first_tail
        if maximum_delta >= 1:
            relative = float("inf")
        else:
            geometric_tail = first_tail / (1 - first_tail ** (radix - 1))
            logarithmic_tail = (
                coefficient_bound * geometric_tail / (1 - maximum_delta)
            )
            relative = exp(logarithmic_tail) - 1
    return AnalyticMaskApproximation(
        radix,
        z,
        terms,
        value,
        relative,
        abs(value) * relative,
    )


def _multiply_mod_truncated(
    left: Sequence[int],
    right: Sequence[int],
    modulus: int,
    degree: int,
) -> tuple[int, ...]:
    output = [0] * (degree + 1)
    for i, left_value in enumerate(left):
        if i > degree:
            break
        for j, right_value in enumerate(right):
            if i + j > degree:
                break
            output[i + j] = (
                output[i + j] + int(left_value) * int(right_value)
            ) % modulus
    return tuple(output)


@dataclass(frozen=True, slots=True)
class RationalMaskAlgebraicCertificate:
    """Finite exact certificate for ``A F^(p-1)=B`` modulo a prime."""

    prime: int
    degree: int
    failures: tuple[tuple[int, int], ...]

    @property
    def valid(self) -> bool:
        return not self.failures

    def require(self) -> None:
        if self.failures:
            raise ValidationError(
                f"rational-mask algebraic relation failed: {self.failures[0]}"
            )

    def certificate(self) -> Certificate:
        return Certificate(
            valid=self.valid,
            exactness=Exactness.EXACT,
            assumptions_checked=("prime characteristic", "unit rational mask"),
            diagnostics={
                "prime": self.prime,
                "degree": self.degree,
                "failures": self.failures,
            },
            theorem_id="GF-RATIONAL-MASK-ALGEBRAIC",
            message=(
                "rational-mask algebraic relation verified"
                if self.valid
                else "rational-mask algebraic relation failed"
            ),
        )


def certify_rational_mask_algebraic_relation(
    numerator: Sequence[int],
    denominator: Sequence[int],
    prime: int,
    degree: int,
    *,
    budget: GeneratingBudget = DEFAULT_GENERATING_BUDGET,
) -> RationalMaskAlgebraicCertificate:
    r"""Certify ``A(z)F(z)^(p-1)=B(z)`` modulo ``p`` through ``degree``."""

    if not _is_prime(prime):
        raise ValidationError("prime must be prime")
    selected_degree = budget.require_degree(degree)
    from .cartier import rational_mask_coefficients_mod_prime

    coefficients = rational_mask_coefficients_mod_prime(
        numerator,
        denominator,
        prime,
        selected_degree + 1,
    )
    result = (1,) + (0,) * selected_degree
    base = coefficients
    exponent = prime - 1
    while exponent:
        if exponent & 1:
            result = _multiply_mod_truncated(
                result, base, prime, selected_degree
            )
        exponent >>= 1
        if exponent:
            base = _multiply_mod_truncated(base, base, prime, selected_degree)
    left = _multiply_mod_truncated(
        numerator, result, prime, selected_degree
    )
    right = tuple(
        int(denominator[index]) % prime if index < len(denominator) else 0
        for index in range(selected_degree + 1)
    )
    failures = tuple(
        (index, (left[index] - right[index]) % prime)
        for index in range(selected_degree + 1)
        if left[index] != right[index]
    )
    certificate = RationalMaskAlgebraicCertificate(
        prime, selected_degree, failures
    )
    certificate.require()
    return certificate


def _solve_fractional_linear_system(
    rows: list[list[Fraction]],
    target: list[Fraction],
    unknowns: int,
) -> tuple[Fraction, ...] | None:
    augmented = [row[:] + [value] for row, value in zip(rows, target, strict=True)]
    pivot_columns: list[int] = []
    cursor = 0
    for column in range(unknowns):
        pivot = next(
            (row for row in range(cursor, len(augmented)) if augmented[row][column]),
            None,
        )
        if pivot is None:
            continue
        augmented[cursor], augmented[pivot] = augmented[pivot], augmented[cursor]
        scale = augmented[cursor][column]
        augmented[cursor] = [value / scale for value in augmented[cursor]]
        for row in range(len(augmented)):
            if row == cursor or augmented[row][column] == 0:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                augmented[row][entry] - factor * augmented[cursor][entry]
                for entry in range(unknowns + 1)
            ]
        pivot_columns.append(column)
        cursor += 1
        if cursor == len(augmented):
            break
    if any(
        all(value == 0 for value in row[:unknowns]) and row[unknowns] != 0
        for row in augmented
    ):
        return None
    solution = [Fraction(0) for _ in range(unknowns)]
    for row, column in enumerate(pivot_columns):
        solution[column] = augmented[row][unknowns]
    return tuple(solution)


@dataclass(frozen=True, slots=True)
class PolynomialCoboundaryResult:
    """Bounded exact solution of ``B Q=A Q(z^radix)`` with ``Q(0)=1``."""

    radix: int
    degree_bound: int
    coefficients: tuple[Fraction, ...] | None
    residual: tuple[Fraction, ...]

    @property
    def found(self) -> bool:
        return self.coefficients is not None

    @property
    def valid(self) -> bool:
        return self.found and not any(self.residual)

    def require(self) -> tuple[Fraction, ...]:
        if not self.valid or self.coefficients is None:
            raise ValidationError(
                "no verified polynomial coboundary was found within the degree bound"
            )
        return self.coefficients

    def certificate(self) -> Certificate:
        return Certificate(
            valid=self.valid,
            exactness=Exactness.EXACT,
            assumptions_checked=("finite degree bound", "unit rational mask"),
            diagnostics={
                "radix": self.radix,
                "degree_bound": self.degree_bound,
                "found": self.found,
                "residual": self.residual,
            },
            theorem_id="GF-POLYNOMIAL-COBOUNDARY",
            message=(
                "polynomial coboundary verified"
                if self.valid
                else "no polynomial coboundary found within the declared bound"
            ),
        )


def solve_polynomial_coboundary(
    numerator: Iterable[Scalar],
    denominator: Iterable[Scalar],
    radix: int,
    degree_bound: int,
    *,
    budget: GeneratingBudget = DEFAULT_GENERATING_BUDGET,
) -> PolynomialCoboundaryResult:
    r"""Solve ``B(z)Q(z)=A(z)Q(z**radix)`` over the rationals."""

    if isinstance(radix, bool) or not isinstance(radix, int) or radix < 2:
        raise ValidationError("radix must be an integer at least two")
    degree = budget.require_degree(degree_bound)
    a = _trim(_fractions(numerator))
    b = _trim(_fractions(denominator))
    if a[0] != b[0] or a[0] == 0:
        raise ValidationError(
            "numerator and denominator constants must agree and be nonzero"
        )
    maximum = max((len(b) - 1) + degree, (len(a) - 1) + radix * degree)
    budget.require_degree(maximum)
    rows: list[list[Fraction]] = []
    targets: list[Fraction] = []
    for output_degree in range(maximum + 1):
        constant = (
            b[output_degree] if output_degree < len(b) else Fraction(0)
        ) - (a[output_degree] if output_degree < len(a) else Fraction(0))
        row: list[Fraction] = []
        for q_degree in range(1, degree + 1):
            left_index = output_degree - q_degree
            right_index = output_degree - radix * q_degree
            left = b[left_index] if 0 <= left_index < len(b) else Fraction(0)
            right = a[right_index] if 0 <= right_index < len(a) else Fraction(0)
            row.append(left - right)
        rows.append(row)
        targets.append(-constant)
    if degree == 0:
        tail: tuple[Fraction, ...] | None = () if not any(targets) else None
    else:
        tail = _solve_fractional_linear_system(rows, targets, degree)
    if tail is None:
        return PolynomialCoboundaryResult(radix, degree, None, ())
    coefficients = _trim((Fraction(1),) + tail)
    left = _multiply_truncated(b, coefficients, maximum)
    substituted = [Fraction(0)] * (maximum + 1)
    for index, value in enumerate(coefficients):
        target = radix * index
        if target <= maximum:
            substituted[target] = value
    right = _multiply_truncated(a, substituted, maximum)
    residual = _trim(
        tuple(left[index] - right[index] for index in range(maximum + 1))
    )
    return PolynomialCoboundaryResult(radix, degree, coefficients, residual)


@dataclass(frozen=True, slots=True)
class EulerProductCollapse:
    """Orbitwise finite-collapse data for an Euler-product local mask."""

    radix: int
    mask_exponents: tuple[tuple[int, int], ...]
    closed_form_exponents: tuple[tuple[int, int], ...]
    obstruction_orbits: tuple[int, ...]

    @property
    def collapses(self) -> bool:
        return not self.obstruction_orbits

    def require(self) -> None:
        if self.obstruction_orbits:
            raise ValidationError(
                f"Euler mask has nonzero orbit sums: {self.obstruction_orbits}"
            )


def _radix_free_part(index: int, radix: int) -> tuple[int, int]:
    unit = index
    level = 0
    while unit % radix == 0:
        unit //= radix
        level += 1
    return unit, level


def euler_product_collapse(
    radix: int,
    exponents: Mapping[int, int],
) -> EulerProductCollapse:
    r"""Compute the finite Euler product represented by a local Euler mask."""

    if isinstance(radix, bool) or not isinstance(radix, int) or radix < 2:
        raise ValidationError("radix must be an integer at least two")
    normalized: dict[int, int] = {}
    for index, exponent in exponents.items():
        if isinstance(index, bool) or not isinstance(index, int) or index < 1:
            raise ValidationError("Euler factor indices must be positive integers")
        if isinstance(exponent, bool) or not isinstance(exponent, int):
            raise ValidationError("Euler factor exponents must be integers")
        if exponent:
            normalized[index] = normalized.get(index, 0) + exponent
    orbits: dict[int, dict[int, int]] = {}
    for index, exponent in normalized.items():
        unit, level = _radix_free_part(index, radix)
        orbits.setdefault(unit, {})[level] = exponent
    closed: dict[int, int] = {}
    obstructions: list[int] = []
    for unit, levels in sorted(orbits.items()):
        maximum = max(levels)
        running = 0
        for level in range(maximum + 1):
            running += levels.get(level, 0)
            if running:
                closed[unit * radix**level] = running
        if running:
            obstructions.append(unit)
    if obstructions:
        closed = {}
    return EulerProductCollapse(
        radix,
        tuple(sorted(normalized.items())),
        tuple(sorted(closed.items())),
        tuple(obstructions),
    )


NormalizedMaskPlan = FiniteMaskCarryPlan


__all__ = [
    "AnalyticMaskApproximation",
    "CarryLayerConvolution",
    "EulerProductCollapse",
    "PolynomialCoboundaryResult",
    "RationalMaskAlgebraicCertificate",
    "analytic_mahler_product",
    "certify_rational_mask_algebraic_relation",
    "euler_product_collapse",
    "DigitCauchyProductPlan",
    "FiniteMaskCarryPlan",
    "KummerLayerDistribution",
    "NormalizedMaskPlan",
    "certify_mahler_functional_equation",
    "compile_digit_cauchy_product",
    "compile_finite_mask_carry_plan",
    "factorial_power_convolution_layers",
    "factorial_power_support_count",
    "factorial_power_support_indices",
    "first_frobenius_lift",
    "frobenius_defect_mask",
    "kummer_layer_distribution",
    "mahler_mask_inverse",
    "mahler_mask_product",
    "one_borrow_count",
    "polynomial_mask_coboundary",
    "solve_polynomial_coboundary",
    "zero_borrow_count",
]
