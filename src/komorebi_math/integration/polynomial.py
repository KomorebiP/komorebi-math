"""Exact finite multivariate polynomials for measure integration.

The coordinate subsystem already contains efficient univariate polynomial
operations.  This module supplies the missing measure-facing object: an
immutable sparse polynomial with an explicit ambient dimension.  Coefficients
are rational and floating-point inputs are rejected, so ``EXACT`` certificates
never hide a binary floating-point conversion.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError

Rational = int | Fraction
Exponent = tuple[int, ...]
Term = tuple[Exponent, Fraction]


def _fraction(value: Rational, *, name: str = "coefficient") -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise ValidationError(f"{name} must be an integer or Fraction")
    return Fraction(value)


def _dimension(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DimensionError("polynomial dimension must be a positive integer")
    return value


def _exponent(value: Sequence[int], dimension: int) -> Exponent:
    data = tuple(value)
    if len(data) != dimension:
        raise DimensionError("monomial exponent dimension does not match the polynomial")
    if any(isinstance(entry, bool) or not isinstance(entry, int) or entry < 0 for entry in data):
        raise ValidationError("monomial exponents must be nonnegative integers")
    return data


def _point(value: Any, dimension: int) -> tuple[Any, ...]:
    if dimension == 1 and not isinstance(value, (tuple, list)):
        return (value,)
    try:
        data = tuple(value)
    except TypeError as exc:
        raise DimensionError("polynomial evaluation point is not vector-like") from exc
    if len(data) != dimension:
        raise DimensionError("polynomial evaluation point has the wrong dimension")
    return data


@dataclass(frozen=True, slots=True, init=False)
class ExactPolynomial:
    r"""A sparse polynomial ``sum_exponent c_exponent x^exponent`` over ``Q``.

    Terms are stored canonically in lexicographic exponent order.  Duplicate
    exponents are combined and zero coefficients are removed.
    """

    dimension: int
    terms: tuple[Term, ...]

    def __init__(
        self,
        dimension: int,
        terms: Mapping[Sequence[int], Rational] | Iterable[tuple[Sequence[int], Rational]],
    ) -> None:
        size = _dimension(dimension)
        source = terms.items() if isinstance(terms, Mapping) else terms
        combined: dict[Exponent, Fraction] = {}
        for raw_exponent, raw_coefficient in source:
            exponent = _exponent(raw_exponent, size)
            coefficient = _fraction(raw_coefficient)
            combined[exponent] = combined.get(exponent, Fraction(0)) + coefficient
        canonical = tuple(sorted((key, value) for key, value in combined.items() if value != 0))
        object.__setattr__(self, "dimension", size)
        object.__setattr__(self, "terms", canonical)

    @classmethod
    def zero(cls, dimension: int) -> "ExactPolynomial":
        return cls(dimension, ())

    @classmethod
    def constant(cls, value: Rational, dimension: int = 1) -> "ExactPolynomial":
        size = _dimension(dimension)
        coefficient = _fraction(value)
        return cls(size, () if coefficient == 0 else (((0,) * size, coefficient),))

    @classmethod
    def monomial(cls, coefficient: Rational, exponents: Sequence[int]) -> "ExactPolynomial":
        powers = tuple(exponents)
        return cls(len(powers), ((powers, coefficient),))

    @classmethod
    def univariate(cls, coefficients: Iterable[Rational]) -> "ExactPolynomial":
        values = tuple(_fraction(value) for value in coefficients)
        if not values:
            values = (Fraction(0),)
        return cls(1, (((degree,), coefficient) for degree, coefficient in enumerate(values)))

    @property
    def degree(self) -> int:
        return max((sum(exponent) for exponent, _ in self.terms), default=0)

    @property
    def degrees(self) -> tuple[int, ...]:
        return tuple(
            max((exponent[variable] for exponent, _ in self.terms), default=0)
            for variable in range(self.dimension)
        )

    @property
    def term_count(self) -> int:
        return len(self.terms)

    def coefficient(self, exponents: Sequence[int]) -> Fraction:
        target = _exponent(exponents, self.dimension)
        return dict(self.terms).get(target, Fraction(0))

    def evaluate(self, point: Any) -> Any:
        values = _point(point, self.dimension)
        total: Any = Fraction(0)
        for exponent, coefficient in self.terms:
            term: Any = coefficient
            for value, power in zip(values, exponent, strict=True):
                try:
                    term *= value**power
                except OverflowError as exc:
                    raise ValidationError(
                        "polynomial evaluation overflowed numerical range"
                    ) from exc
            try:
                total += term
            except OverflowError as exc:
                raise ValidationError(
                    "polynomial evaluation overflowed numerical range"
                ) from exc
        return total

    __call__ = evaluate

    def add(self, other: "ExactPolynomial") -> "ExactPolynomial":
        if not isinstance(other, ExactPolynomial):
            raise TypeError("polynomial addition requires an ExactPolynomial")
        if self.dimension != other.dimension:
            raise DimensionError("polynomial dimensions do not agree")
        return ExactPolynomial(self.dimension, self.terms + other.terms)

    def subtract(self, other: "ExactPolynomial") -> "ExactPolynomial":
        return self.add(other.scale(-1))

    def scale(self, scalar: Rational) -> "ExactPolynomial":
        factor = _fraction(scalar, name="scale")
        return ExactPolynomial(
            self.dimension,
            ((exponent, factor * coefficient) for exponent, coefficient in self.terms),
        )

    def multiply(
        self,
        other: "ExactPolynomial",
        *,
        max_terms: int = 100_000,
    ) -> "ExactPolynomial":
        if not isinstance(other, ExactPolynomial):
            raise TypeError("polynomial multiplication requires an ExactPolynomial")
        if self.dimension != other.dimension:
            raise DimensionError("polynomial dimensions do not agree")
        if isinstance(max_terms, bool) or not isinstance(max_terms, int) or max_terms < 1:
            raise ValidationError("max_terms must be a positive integer")
        combined: dict[Exponent, Fraction] = {}
        for left_exponent, left_coefficient in self.terms:
            for right_exponent, right_coefficient in other.terms:
                exponent = tuple(
                    left + right
                    for left, right in zip(left_exponent, right_exponent, strict=True)
                )
                combined[exponent] = (
                    combined.get(exponent, Fraction(0))
                    + left_coefficient * right_coefficient
                )
                if len(combined) > max_terms:
                    raise ValidationError("polynomial expansion exceeded max_terms")
        return ExactPolynomial(self.dimension, combined)

    def power(self, exponent: int, *, max_terms: int = 100_000) -> "ExactPolynomial":
        if isinstance(exponent, bool) or not isinstance(exponent, int) or exponent < 0:
            raise ValidationError("polynomial powers require a nonnegative integer")
        result = ExactPolynomial.constant(1, self.dimension)
        base = self
        power = exponent
        while power:
            if power & 1:
                result = result.multiply(base, max_terms=max_terms)
            power >>= 1
            if power:
                base = base.multiply(base, max_terms=max_terms)
        return result

    def derivative(self, variable: int = 0, order: int = 1) -> "ExactPolynomial":
        if isinstance(variable, bool) or not isinstance(variable, int) or not 0 <= variable < self.dimension:
            raise DimensionError("derivative variable is outside the polynomial dimension")
        if isinstance(order, bool) or not isinstance(order, int) or order < 0:
            raise ValidationError("derivative order must be a nonnegative integer")
        result = self
        for _ in range(order):
            terms: list[Term] = []
            for exponent, coefficient in result.terms:
                power = exponent[variable]
                if power == 0:
                    continue
                updated = list(exponent)
                updated[variable] -= 1
                terms.append((tuple(updated), coefficient * power))
            result = ExactPolynomial(self.dimension, terms)
        return result

    def antiderivative(self, variable: int = 0, constant: Rational = 0) -> "ExactPolynomial":
        if isinstance(variable, bool) or not isinstance(variable, int) or not 0 <= variable < self.dimension:
            raise DimensionError("integration variable is outside the polynomial dimension")
        terms: list[Term] = []
        for exponent, coefficient in self.terms:
            updated = list(exponent)
            updated[variable] += 1
            terms.append((tuple(updated), coefficient / updated[variable]))
        constant_value = _fraction(constant, name="integration constant")
        if constant_value:
            terms.append(((0,) * self.dimension, constant_value))
        return ExactPolynomial(self.dimension, terms)

    def integrate_out(
        self,
        variable: int,
        lower: Rational,
        upper: Rational,
        *,
        density: Rational = 1,
    ) -> Fraction | "ExactPolynomial":
        """Integrate one variable over a constant-density rational interval."""

        if isinstance(variable, bool) or not isinstance(variable, int) or not 0 <= variable < self.dimension:
            raise DimensionError("integration variable is outside the polynomial dimension")
        a = _fraction(lower, name="lower endpoint")
        b = _fraction(upper, name="upper endpoint")
        rho = _fraction(density, name="density")
        if b < a:
            raise ValidationError("integration interval must satisfy lower <= upper")
        reduced: dict[Exponent, Fraction] = {}
        for exponent, coefficient in self.terms:
            power = exponent[variable]
            moment = rho * (b ** (power + 1) - a ** (power + 1)) / (power + 1)
            target = exponent[:variable] + exponent[variable + 1 :]
            reduced[target] = reduced.get(target, Fraction(0)) + coefficient * moment
        if self.dimension == 1:
            return reduced.get((), Fraction(0))
        return ExactPolynomial(self.dimension - 1, reduced)

    def pullback_affine(
        self,
        matrix: Sequence[Sequence[Rational]],
        shift: Sequence[Rational],
        *,
        max_terms: int = 100_000,
    ) -> "ExactPolynomial":
        r"""Return ``self(M x + b)`` exactly.

        The matrix has shape ``(self.dimension, source_dimension)``.  A term
        budget makes multivariate expansion failure explicit rather than
        allowing accidental combinatorial blow-up.
        """

        rows = tuple(tuple(_fraction(value, name="affine matrix entry") for value in row) for row in matrix)
        offsets = tuple(_fraction(value, name="affine shift entry") for value in shift)
        if len(rows) != self.dimension or len(offsets) != self.dimension:
            raise DimensionError("affine target dimension does not match the polynomial")
        if not rows or not rows[0]:
            raise DimensionError("affine maps require positive source and target dimensions")
        source_dimension = len(rows[0])
        if any(len(row) != source_dimension for row in rows):
            raise DimensionError("affine matrix rows must have a common length")
        coordinates: list[ExactPolynomial] = []
        for row, offset in zip(rows, offsets, strict=True):
            terms: list[tuple[Exponent, Fraction]] = [((0,) * source_dimension, offset)]
            for variable, coefficient in enumerate(row):
                exponent = tuple(int(index == variable) for index in range(source_dimension))
                terms.append((exponent, coefficient))
            coordinates.append(ExactPolynomial(source_dimension, terms))
        result = ExactPolynomial.zero(source_dimension)
        for exponent, coefficient in self.terms:
            term = ExactPolynomial.constant(coefficient, source_dimension)
            for coordinate, power in zip(coordinates, exponent, strict=True):
                if power:
                    term = term.multiply(
                        coordinate.power(power, max_terms=max_terms),
                        max_terms=max_terms,
                    )
            result = result.add(term)
            if result.term_count > max_terms:
                raise ValidationError("affine pullback exceeded max_terms")
        return result

    def to_univariate_coefficients(self) -> tuple[Fraction, ...]:
        if self.dimension != 1:
            raise DimensionError("only one-dimensional polynomials have univariate coefficients")
        degree = self.degrees[0]
        coefficients = [Fraction(0) for _ in range(degree + 1)]
        for exponent, coefficient in self.terms:
            coefficients[exponent[0]] = coefficient
        return tuple(coefficients)

    def triangle_derivative_bound(
        self,
        lower: Rational,
        upper: Rational,
        *,
        order: int = 4,
    ) -> Fraction:
        r"""Return a rigorous triangle-inequality bound for ``|p^(order)|``.

        This intentionally uses a conservative coefficient bound rather than
        pretending to solve the global polynomial maximization problem.
        """

        if self.dimension != 1:
            raise DimensionError("triangle derivative bounds currently require dimension one")
        a = _fraction(lower, name="lower endpoint")
        b = _fraction(upper, name="upper endpoint")
        if b < a:
            raise ValidationError("bound interval must satisfy lower <= upper")
        derivative = self.derivative(0, order)
        radius = max(abs(a), abs(b))
        return sum(
            (abs(coefficient) * radius ** exponent[0] for exponent, coefficient in derivative.terms),
            Fraction(0),
        )

    def certificate(self) -> Certificate:
        return Certificate(
            valid=True,
            exactness=Exactness.EXACT,
            assumptions_checked=(
                "positive ambient dimension",
                "nonnegative integer exponents",
                "exact rational coefficients",
                "canonical sparse representation",
            ),
            diagnostics={
                "dimension": self.dimension,
                "degree": self.degree,
                "degrees": self.degrees,
                "term_count": self.term_count,
            },
            theorem_id="INT.POLY.1",
            message="exact sparse polynomial validated",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "terms": [
                {"exponents": list(exponent), "coefficient": coefficient}
                for exponent, coefficient in self.terms
            ],
        }


__all__ = ["ExactPolynomial", "Exponent", "Rational", "Term"]
