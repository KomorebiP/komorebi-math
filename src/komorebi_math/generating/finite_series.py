"""Immutable finite normalized series over explicit exact coefficient domains."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Sequence

from ..certificates import Certificate, Exactness
from ..exceptions import ValidationError
from .basis import WeightedMonomialBasis
from .budget import DEFAULT_GENERATING_BUDGET, GeneratingBudget
from .domains import CoefficientDomain


def _bit_length(value: Any) -> int:
    if isinstance(value, Fraction):
        return max(abs(value.numerator).bit_length(), value.denominator.bit_length())
    if isinstance(value, int):
        return abs(value).bit_length()
    return 0


def _ordinary_multiply(
    domain: CoefficientDomain[Any],
    left: Sequence[Any],
    right: Sequence[Any],
    degree: int,
) -> tuple[Any, ...]:
    output = [domain.zero for _ in range(degree + 1)]
    for i, left_value in enumerate(left):
        if i > degree:
            break
        for j, right_value in enumerate(right):
            if i + j > degree:
                break
            output[i + j] = domain.add(
                output[i + j], domain.multiply(left_value, right_value)
            )
    return tuple(output)


def _ordinary_reciprocal(
    domain: CoefficientDomain[Any],
    values: Sequence[Any],
    degree: int,
) -> tuple[Any, ...]:
    if not values or not domain.is_unit(values[0]):
        raise ValidationError("formal reciprocal requires an invertible constant coefficient")
    output = [domain.zero for _ in range(degree + 1)]
    inverse = domain.inverse(values[0])
    output[0] = inverse
    for n in range(1, degree + 1):
        total = domain.zero
        for k in range(1, min(n, len(values) - 1) + 1):
            total = domain.add(total, domain.multiply(values[k], output[n - k]))
        output[n] = domain.multiply(domain.subtract(domain.zero, total), inverse)
    return tuple(output)


@dataclass(frozen=True, slots=True)
class SeriesTransferCertificate:
    """Exact finite certificate for a normalized basis transfer."""

    valid: bool
    degree: int
    source_basis: str
    target_basis: str
    domain: str
    failures: tuple[int, ...] = ()

    def require(self) -> None:
        if not self.valid:
            raise ValidationError(f"basis transfer failed at degrees {self.failures[:4]}")

    def certificate(self) -> Certificate:
        return Certificate(
            valid=self.valid,
            exactness=Exactness.EXACT,
            assumptions_checked=("finite truncation", "exact coefficient domain"),
            diagnostics={
                "degree": self.degree,
                "source_basis": self.source_basis,
                "target_basis": self.target_basis,
                "domain": self.domain,
                "failures": self.failures,
            },
            theorem_id="NGF.UNIVERSAL_PRODUCT_AND_TRANSFER",
            message="normalized basis transfer verified" if self.valid else "basis transfer failed",
        )


@dataclass(frozen=True, slots=True)
class FiniteNormalizedSeries:
    r"""A finite exact series ``sum a_n z^n/gamma_n`` over one domain."""

    basis: WeightedMonomialBasis
    domain: CoefficientDomain[Any]
    coefficients: tuple[Any, ...]
    budget: GeneratingBudget = DEFAULT_GENERATING_BUDGET

    def __init__(
        self,
        basis: WeightedMonomialBasis,
        domain: CoefficientDomain[Any],
        coefficients: Iterable[Any],
        *,
        budget: GeneratingBudget = DEFAULT_GENERATING_BUDGET,
    ) -> None:
        if not isinstance(basis, WeightedMonomialBasis):
            raise TypeError("basis must be a WeightedMonomialBasis")
        if not isinstance(domain, CoefficientDomain):
            raise TypeError("domain does not satisfy the coefficient-domain protocol")
        if not isinstance(budget, GeneratingBudget):
            raise TypeError("budget must be a GeneratingBudget")
        # Consume at most the allowed prefix plus one excess item. Check before
        # coercing that excess item, and stop on oversized coefficients immediately.
        collected: list[Any] = []
        for degree, raw in enumerate(coefficients):
            budget.require_degree(degree)
            value = domain.coerce(raw)
            budget.require_coefficient_bits(_bit_length(value))
            collected.append(value)
        values = tuple(collected) if collected else (domain.zero,)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "coefficients", values)
        object.__setattr__(self, "budget", budget)

    @property
    def degree(self) -> int:
        return len(self.coefficients) - 1

    def coefficient(self, index: int) -> Any:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValidationError("coefficient index must be nonnegative")
        return self.coefficients[index] if index <= self.degree else self.domain.zero

    def truncate(self, degree: int) -> "FiniteNormalizedSeries":
        bound = self.budget.require_degree(degree)
        values = self.coefficients[: bound + 1]
        values += (self.domain.zero,) * max(0, bound + 1 - len(values))
        return FiniteNormalizedSeries(
            self.basis, self.domain, values, budget=self.budget
        )

    def _require_compatible(self, other: "FiniteNormalizedSeries") -> None:
        if not isinstance(other, FiniteNormalizedSeries):
            raise TypeError("series operation requires another FiniteNormalizedSeries")
        if self.domain.descriptor() != other.domain.descriptor():
            raise ValidationError("coefficient domains do not match")

    def convert(
        self, target_basis: WeightedMonomialBasis
    ) -> tuple["FiniteNormalizedSeries", SeriesTransferCertificate]:
        """Transfer normalized coefficients without changing the formal series.

        Identical finite normalization tables require no division and therefore
        remain valid in residue characteristics where individual ``gamma_n``
        values are not units.  A genuinely different transfer is accepted only
        when both normalization ratios exist in the declared coefficient domain.
        """

        if not isinstance(target_basis, WeightedMonomialBasis):
            raise TypeError("target_basis must be a WeightedMonomialBasis")
        if self.basis.equivalent_through(target_basis, self.degree):
            target = FiniteNormalizedSeries(
                target_basis,
                self.domain,
                self.coefficients,
                budget=self.budget,
            )
            certificate = SeriesTransferCertificate(
                True,
                self.degree,
                self.basis.name,
                target_basis.name,
                self.domain.name,
            )
            return target, certificate

        values: list[Any] = []
        failures: list[int] = []
        for n, coefficient in enumerate(self.coefficients):
            forward = self.domain.coerce(
                target_basis.gamma(n) / self.basis.gamma(n)
            )
            reverse = self.domain.coerce(
                self.basis.gamma(n) / target_basis.gamma(n)
            )
            transferred = self.domain.multiply(coefficient, forward)
            values.append(transferred)
            reconstructed = self.domain.multiply(transferred, reverse)
            if not self.domain.equal(reconstructed, coefficient):
                failures.append(n)
        target = FiniteNormalizedSeries(
            target_basis, self.domain, values, budget=self.budget
        )
        certificate = SeriesTransferCertificate(
            not failures,
            self.degree,
            self.basis.name,
            target_basis.name,
            self.domain.name,
            tuple(failures),
        )
        return target, certificate

    def ordinary_coefficients(self) -> tuple[Any, ...]:
        return tuple(
            self.domain.multiply(
                coefficient, self.domain.coerce(Fraction(1, 1) / self.basis.gamma(n))
            )
            for n, coefficient in enumerate(self.coefficients)
        )

    @classmethod
    def from_ordinary_coefficients(
        cls,
        basis: WeightedMonomialBasis,
        domain: CoefficientDomain[Any],
        coefficients: Iterable[Any],
        *,
        budget: GeneratingBudget = DEFAULT_GENERATING_BUDGET,
    ) -> "FiniteNormalizedSeries":
        return cls(
            basis,
            domain,
            (
                domain.multiply(domain.coerce(value), domain.coerce(basis.gamma(n)))
                for n, value in enumerate(coefficients)
            ),
            budget=budget,
        )

    def add(
        self, other: "FiniteNormalizedSeries", *, degree: int | None = None
    ) -> "FiniteNormalizedSeries":
        self._require_compatible(other)
        converted, certificate = other.convert(self.basis)
        certificate.require()
        bound = max(self.degree, converted.degree) if degree is None else self.budget.require_degree(degree)
        left = self.truncate(bound)
        right = converted.truncate(bound)
        return FiniteNormalizedSeries(
            self.basis,
            self.domain,
            (
                self.domain.add(left.coefficients[n], right.coefficients[n])
                for n in range(bound + 1)
            ),
            budget=self.budget,
        )

    def subtract(
        self, other: "FiniteNormalizedSeries", *, degree: int | None = None
    ) -> "FiniteNormalizedSeries":
        self._require_compatible(other)
        converted, certificate = other.convert(self.basis)
        certificate.require()
        bound = max(self.degree, converted.degree) if degree is None else self.budget.require_degree(degree)
        left = self.truncate(bound)
        right = converted.truncate(bound)
        return FiniteNormalizedSeries(
            self.basis,
            self.domain,
            (
                self.domain.subtract(left.coefficients[n], right.coefficients[n])
                for n in range(bound + 1)
            ),
            budget=self.budget,
        )

    def scale(self, scalar: Any) -> "FiniteNormalizedSeries":
        value = self.domain.coerce(scalar)
        return FiniteNormalizedSeries(
            self.basis,
            self.domain,
            (self.domain.multiply(value, coefficient) for coefficient in self.coefficients),
            budget=self.budget,
        )

    def multiply(
        self, other: "FiniteNormalizedSeries", *, degree: int | None = None
    ) -> "FiniteNormalizedSeries":
        self._require_compatible(other)
        converted, certificate = other.convert(self.basis)
        certificate.require()
        bound = (
            self.degree + converted.degree
            if degree is None
            else self.budget.require_degree(degree)
        )
        left = self.truncate(bound)
        right = converted.truncate(bound)
        output = [self.domain.zero for _ in range(bound + 1)]
        for n in range(bound + 1):
            total = self.domain.zero
            for k in range(n + 1):
                factor = self.domain.coerce(self.basis.structure_constant(n, k))
                term = self.domain.multiply(
                    factor,
                    self.domain.multiply(
                        left.coefficients[k], right.coefficients[n - k]
                    ),
                )
                total = self.domain.add(total, term)
            output[n] = total
        return FiniteNormalizedSeries(
            self.basis, self.domain, output, budget=self.budget
        )

    def reciprocal(self, *, degree: int | None = None) -> "FiniteNormalizedSeries":
        bound = self.degree if degree is None else self.budget.require_degree(degree)
        ordinary = self.ordinary_coefficients()
        ordinary += (self.domain.zero,) * max(0, bound + 1 - len(ordinary))
        inverse = _ordinary_reciprocal(self.domain, ordinary, bound)
        return FiniteNormalizedSeries.from_ordinary_coefficients(
            self.basis, self.domain, inverse, budget=self.budget
        )

    def compose(
        self, inner: "FiniteNormalizedSeries", *, degree: int | None = None
    ) -> "FiniteNormalizedSeries":
        self._require_compatible(inner)
        bound = max(self.degree, inner.degree) if degree is None else self.budget.require_degree(degree)
        outer = self.ordinary_coefficients()
        inside = inner.ordinary_coefficients()
        inside += (self.domain.zero,) * max(0, bound + 1 - len(inside))
        if not self.domain.equal(inside[0], self.domain.zero):
            raise ValidationError("composition requires an inner zero constant coefficient")
        output = [self.domain.zero for _ in range(bound + 1)]
        power = [self.domain.zero for _ in range(bound + 1)]
        power[0] = self.domain.one
        for k in range(min(len(outer), bound + 1)):
            for n in range(bound + 1):
                output[n] = self.domain.add(
                    output[n], self.domain.multiply(outer[k], power[n])
                )
            power = list(_ordinary_multiply(self.domain, power, inside, bound))
        return FiniteNormalizedSeries.from_ordinary_coefficients(
            self.basis, self.domain, output, budget=self.budget
        )

    def exp(self, *, degree: int | None = None) -> "FiniteNormalizedSeries":
        bound = self.degree if degree is None else self.budget.require_degree(degree)
        source = self.ordinary_coefficients()
        source += (self.domain.zero,) * max(0, bound + 1 - len(source))
        if not self.domain.equal(source[0], self.domain.zero):
            raise ValidationError("formal exponential requires a zero constant coefficient")
        output = [self.domain.zero for _ in range(bound + 1)]
        output[0] = self.domain.one
        for n in range(1, bound + 1):
            total = self.domain.zero
            for k in range(1, n + 1):
                total = self.domain.add(
                    total,
                    self.domain.multiply(
                        self.domain.coerce(k),
                        self.domain.multiply(source[k], output[n - k]),
                    ),
                )
            output[n] = self.domain.multiply(total, self.domain.inverse(self.domain.coerce(n)))
        return FiniteNormalizedSeries.from_ordinary_coefficients(
            self.basis, self.domain, output, budget=self.budget
        )

    def log(self, *, degree: int | None = None) -> "FiniteNormalizedSeries":
        bound = self.degree if degree is None else self.budget.require_degree(degree)
        source = self.ordinary_coefficients()
        source += (self.domain.zero,) * max(0, bound + 1 - len(source))
        if not self.domain.equal(source[0], self.domain.one):
            raise ValidationError("formal logarithm requires constant coefficient one")
        derivative = tuple(
            self.domain.multiply(self.domain.coerce(n + 1), source[n + 1])
            for n in range(bound)
        )
        reciprocal = _ordinary_reciprocal(self.domain, source, bound)
        quotient = _ordinary_multiply(
            self.domain, derivative, reciprocal, max(bound - 1, 0)
        )
        output = [self.domain.zero]
        output.extend(
            self.domain.multiply(
                quotient[n - 1], self.domain.inverse(self.domain.coerce(n))
            )
            for n in range(1, bound + 1)
        )
        return FiniteNormalizedSeries.from_ordinary_coefficients(
            self.basis, self.domain, output, budget=self.budget
        )

    def project(self, modulus: int, residue: int) -> "FiniteNormalizedSeries":
        if (
            isinstance(modulus, bool)
            or not isinstance(modulus, int)
            or modulus < 1
            or isinstance(residue, bool)
            or not isinstance(residue, int)
            or not 0 <= residue < modulus
        ):
            raise ValidationError("invalid residue projector")
        return FiniteNormalizedSeries(
            self.basis,
            self.domain,
            (
                coefficient if n % modulus == residue else self.domain.zero
                for n, coefficient in enumerate(self.coefficients)
            ),
            budget=self.budget,
        )

    def custom_lower(self) -> "FiniteNormalizedSeries":
        """Apply the canonical normalized lowering operator."""

        return FiniteNormalizedSeries(
            self.basis,
            self.domain,
            self.coefficients[1:] if self.degree else (self.domain.zero,),
            budget=self.budget,
        )

    def certificate(self) -> Certificate:
        """Return an exact finite construction certificate."""

        return Certificate(
            valid=True,
            exactness=Exactness.EXACT,
            assumptions_checked=("finite degree", "explicit coefficient domain"),
            diagnostics={
                "degree": self.degree,
                "basis": self.basis.name,
                "domain": self.domain.name,
            },
            theorem_id="NGF.FINITE_NORMALIZED_SERIES",
            message="finite normalized series is valid",
        )

    def __add__(self, other: "FiniteNormalizedSeries") -> "FiniteNormalizedSeries":
        return self.add(other)

    def __sub__(self, other: "FiniteNormalizedSeries") -> "FiniteNormalizedSeries":
        return self.subtract(other)

    def __mul__(self, other: "FiniteNormalizedSeries" | Any) -> "FiniteNormalizedSeries":
        if isinstance(other, FiniteNormalizedSeries):
            return self.multiply(other)
        return self.scale(other)

    def __rmul__(self, other: Any) -> "FiniteNormalizedSeries":
        return self.scale(other)


__all__ = ["FiniteNormalizedSeries", "SeriesTransferCertificate"]
