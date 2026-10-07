"""Finite measure objects and exact pushforward semantics."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import fsum, isclose, isfinite
from numbers import Real
from typing import Any, Protocol, runtime_checkable

from ..certificates import Certificate, ComputationResult, Exactness
from ..coordinates.exact import determinant, matrix_multiply
from ..exceptions import DimensionError, ValidationError
from .polynomial import ExactPolynomial, Rational

RealScalar = Fraction | float


_EXACTNESS_STRENGTH = {
    Exactness.EXACT: 0,
    Exactness.NUMERICAL: 1,
    Exactness.CONDITIONAL: 2,
    Exactness.HEURISTIC: 3,
    Exactness.PLANNED: 4,
}


def _aggregate_exactness(values: Iterable[Exactness]) -> Exactness:
    """Return the weakest certificate status without float-coercing values."""

    statuses = tuple(values)
    if not statuses:
        return Exactness.EXACT
    return max(statuses, key=_EXACTNESS_STRENGTH.__getitem__)


def _unique_assumptions(*groups: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for group in groups for item in group))


def _finite_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValidationError(f"{name} overflowed floating-point range") from exc
    if not isfinite(result):
        raise ValidationError(f"{name} must be finite")
    return result


def _finite_float_sum(values: Iterable[Any], *, name: str) -> float:
    try:
        result = fsum(_finite_float(value, name=name) for value in values)
    except OverflowError as exc:
        raise ValidationError(f"{name} overflowed floating-point range") from exc
    if not isfinite(result):
        raise ValidationError(f"{name} overflowed or became nonfinite")
    return result


def _finite_real_result(value: Any, *, name: str) -> Any:
    """Validate a scalar result while preserving arbitrary-size exact values."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValidationError(f"{name} must be a finite real scalar")
    if isinstance(value, (int, Fraction)):
        return value
    return _finite_float(value, name=name)


def _finite_product(values: Iterable[Any], *, name: str) -> Any:
    """Multiply scalars exactly when possible and reject numerical overflow."""

    scalars = tuple(_finite_real_result(value, name=name) for value in values)
    if any(value == 0 for value in scalars):
        return (
            Fraction(0)
            if all(isinstance(value, (int, Fraction)) for value in scalars)
            else 0.0
        )
    result: Any = Fraction(1)
    for scalar in scalars:
        try:
            result *= scalar
        except OverflowError as exc:
            raise ValidationError(f"{name} overflowed floating-point range") from exc
        result = _finite_real_result(result, name=name)
    return result


_CALLABLE_PUSHFORWARD_ASSUMPTIONS = (
    "callable is globally defined on base support",
    "callable returns finite real points of the declared target dimension",
    "callable is deterministic, pure, and side-effect free on base support",
)


def _real(value: Any, *, name: str) -> RealScalar:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValidationError(f"{name} must be a finite real scalar")
    if isinstance(value, (int, Fraction)):
        return Fraction(value)
    result = float(value)
    if not isfinite(result):
        raise ValidationError(f"{name} must be finite")
    return result


def _rational(value: Any, *, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise ValidationError(f"{name} must be an integer or Fraction")
    return Fraction(value)


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DimensionError(f"{name} must be a positive integer")
    return value


def _exponents(values: Sequence[int], dimension: int) -> tuple[int, ...]:
    data = tuple(values)
    if len(data) != dimension:
        raise DimensionError("moment exponent dimension does not match the measure")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in data):
        raise ValidationError("moment exponents must be nonnegative integers")
    return data


def _canonical_point(value: Any, dimension: int) -> tuple[RealScalar, ...]:
    if dimension == 1 and isinstance(value, Real) and not isinstance(value, bool):
        return (_real(value, name="point coordinate"),)
    try:
        data = tuple(value)
    except TypeError as exc:
        raise DimensionError("point is not vector-like") from exc
    if len(data) != dimension:
        raise DimensionError("point dimension does not match the measure")
    return tuple(_real(entry, name="point coordinate") for entry in data)


def _call_point(function: Callable[[Any], Any], point: tuple[RealScalar, ...]) -> Any:
    return function(point[0] if len(point) == 1 else point)


@runtime_checkable
class Measure(Protocol):
    """Protocol for finite measures usable by :func:`integrate`."""

    dimension: int
    name: str
    exactness: Exactness

    def mass(self) -> ComputationResult[Any]:
        """Return total measure mass."""

    def moment(self, exponents: Sequence[int]) -> ComputationResult[Any]:
        """Return one monomial moment."""

    def certificate(self) -> Certificate:
        """Return a structural measure certificate."""


@dataclass(frozen=True, slots=True, init=False)
class ExactAffineMap:
    r"""An exact affine map ``x -> matrix*x + shift`` over ``Q``."""

    matrix: tuple[tuple[Fraction, ...], ...]
    shift: tuple[Fraction, ...]
    name: str

    def __init__(
        self,
        matrix: Iterable[Iterable[Rational]],
        shift: Iterable[Rational],
        *,
        name: str = "exact affine map",
    ) -> None:
        rows: list[tuple[Fraction, ...]] = []
        for row in matrix:
            rows.append(
                tuple(_rational(value, name="affine matrix entry") for value in row)
            )
        data = tuple(rows)
        offsets = tuple(
            _rational(value, name="affine shift entry")
            for value in shift
        )
        if not data or not data[0]:
            raise DimensionError("affine maps require positive source and target dimensions")
        width = len(data[0])
        if any(len(row) != width for row in data):
            raise DimensionError("affine matrix rows must have a common length")
        if len(offsets) != len(data):
            raise DimensionError("affine shift dimension does not match the target")
        label = str(name).strip()
        if not label:
            raise ValidationError("affine map name must be nonempty")
        object.__setattr__(self, "matrix", data)
        object.__setattr__(self, "shift", offsets)
        object.__setattr__(self, "name", label)

    @classmethod
    def identity(cls, dimension: int, *, name: str = "identity affine map") -> "ExactAffineMap":
        size = _positive_integer(dimension, name="affine dimension")
        return cls(
            tuple(tuple(Fraction(int(row == column)) for column in range(size)) for row in range(size)),
            (Fraction(0),) * size,
            name=name,
        )

    @property
    def source_dimension(self) -> int:
        return len(self.matrix[0])

    @property
    def target_dimension(self) -> int:
        return len(self.matrix)

    @property
    def square(self) -> bool:
        return self.source_dimension == self.target_dimension

    def apply_point(self, point: Any) -> tuple[Any, ...]:
        if self.source_dimension == 1 and not isinstance(point, (tuple, list)):
            values = (point,)
        else:
            try:
                values = tuple(point)
            except TypeError as exc:
                raise DimensionError("affine input point is not vector-like") from exc
        if len(values) != self.source_dimension:
            raise DimensionError("affine input point has the wrong dimension")
        try:
            return tuple(
                offset
                + sum(
                    (
                        coefficient * value
                        for coefficient, value in zip(row, values, strict=True)
                    ),
                    Fraction(0),
                )
                for row, offset in zip(self.matrix, self.shift, strict=True)
            )
        except OverflowError as exc:
            raise ValidationError("affine point evaluation overflowed numerical range") from exc

    def __call__(self, point: Any) -> Any:
        result = self.apply_point(point)
        return result[0] if self.target_dimension == 1 else result

    def then(self, following: "ExactAffineMap") -> "ExactAffineMap":
        """Compose this map with ``following``."""

        if not isinstance(following, ExactAffineMap):
            raise TypeError("affine composition requires another ExactAffineMap")
        if self.target_dimension != following.source_dimension:
            raise DimensionError("affine map dimensions do not compose")
        matrix = matrix_multiply(following.matrix, self.matrix)
        shift = tuple(
            following.shift[row]
            + sum(
                (
                    following.matrix[row][index] * self.shift[index]
                    for index in range(self.target_dimension)
                ),
                Fraction(0),
            )
            for row in range(following.target_dimension)
        )
        return ExactAffineMap(matrix, shift, name=f"{following.name} after {self.name}")

    def pullback(self, polynomial: ExactPolynomial, *, max_terms: int = 100_000) -> ExactPolynomial:
        if not isinstance(polynomial, ExactPolynomial):
            raise TypeError("affine pullback requires an ExactPolynomial")
        if polynomial.dimension != self.target_dimension:
            raise DimensionError("polynomial dimension does not match the affine target")
        return polynomial.pullback_affine(self.matrix, self.shift, max_terms=max_terms)

    def determinant(self) -> Fraction:
        if not self.square:
            raise DimensionError("determinants require a square affine linear part")
        return determinant(self.matrix)

    def certificate(self) -> Certificate:
        return Certificate(
            valid=True,
            exactness=Exactness.EXACT,
            assumptions_checked=("exact rational matrix", "exact rational shift", "rectangular affine typing"),
            diagnostics={
                "source_dimension": self.source_dimension,
                "target_dimension": self.target_dimension,
                "square": self.square,
                "determinant": str(self.determinant()) if self.square else None,
            },
            theorem_id="INT.MAP.1",
            message="exact affine map validated",
        )


@dataclass(frozen=True, slots=True, init=False)
class AtomicMeasure:
    """A finite positive atomic measure with exact or floating-point data."""

    points: tuple[tuple[RealScalar, ...], ...]
    weights: tuple[RealScalar, ...]
    dimension: int
    name: str
    exactness: Exactness

    def __init__(
        self,
        points: Iterable[Any],
        weights: Iterable[Any],
        *,
        name: str = "atomic measure",
    ) -> None:
        raw_points = tuple(points)
        raw_weights = tuple(weights)
        if not raw_points:
            raise ValidationError("atomic measures require at least one point")
        if len(raw_points) != len(raw_weights):
            raise DimensionError("atomic points and weights must have equal length")
        first = raw_points[0]
        if isinstance(first, Real) and not isinstance(first, bool):
            dimension = 1
        else:
            try:
                dimension = len(tuple(first))
            except TypeError as exc:
                raise DimensionError("atomic point is not vector-like") from exc
            _positive_integer(dimension, name="atomic dimension")
        canonical_points = tuple(_canonical_point(point, dimension) for point in raw_points)
        canonical_weights = tuple(_real(weight, name="atomic weight") for weight in raw_weights)
        if any(weight < 0 for weight in canonical_weights):
            raise ValidationError("positive atomic measures require nonnegative weights")
        label = str(name).strip()
        if not label:
            raise ValidationError("measure name must be nonempty")
        exact = all(
            isinstance(value, Fraction)
            for point in canonical_points
            for value in point
        ) and all(isinstance(weight, Fraction) for weight in canonical_weights)
        object.__setattr__(self, "points", canonical_points)
        object.__setattr__(self, "weights", canonical_weights)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "name", label)
        object.__setattr__(self, "exactness", Exactness.EXACT if exact else Exactness.NUMERICAL)

    @property
    def atom_count(self) -> int:
        return len(self.points)

    def mass(self) -> ComputationResult[Any]:
        if all(isinstance(weight, Fraction) for weight in self.weights):
            value: Any = sum(self.weights, Fraction(0))
        else:
            value = _finite_float_sum(self.weights, name="atomic mass")
        return ComputationResult(
            value,
            Certificate(
                valid=True,
                exactness=self.exactness,
                assumptions_checked=("finite positive atomic weights",),
                diagnostics={"atom_count": self.atom_count},
                theorem_id="INT.MEASURE.ATOM.MASS",
                message="atomic mass evaluated",
            ),
        )

    def moment(self, exponents: Sequence[int]) -> ComputationResult[Any]:
        powers = _exponents(exponents, self.dimension)
        if all(power == 0 for power in powers):
            value: Any = self.mass().value
        elif self.exactness is Exactness.EXACT:
            value: Any = Fraction(0)
            for point, weight in zip(self.points, self.weights, strict=True):
                term: Any = weight
                for coordinate, power in zip(point, powers, strict=True):
                    term *= coordinate**power
                value += term
        else:
            terms: list[float] = []
            for point, weight in zip(self.points, self.weights, strict=True):
                numeric_weight = _finite_float(weight, name="atomic moment weight")
                if numeric_weight == 0.0:
                    terms.append(0.0)
                    continue
                term = numeric_weight
                for coordinate, power in zip(point, powers, strict=True):
                    numeric_coordinate = _finite_float(
                        coordinate,
                        name="atomic moment coordinate",
                    )
                    try:
                        powered = numeric_coordinate**power
                        term *= powered
                    except OverflowError as exc:
                        raise ValidationError(
                            "atomic moment overflowed floating-point range"
                        ) from exc
                    term = _finite_float(term, name="atomic moment")
                terms.append(term)
            value = _finite_float_sum(terms, name="atomic moment")
        return ComputationResult(
            value,
            Certificate(
                valid=True,
                exactness=self.exactness,
                assumptions_checked=("finite atomic sum",),
                diagnostics={"exponents": powers, "atom_count": self.atom_count},
                theorem_id="INT.MEASURE.ATOM.MOMENT",
                message="atomic moment evaluated",
            ),
        )

    def normalized(self) -> "AtomicMeasure":
        total = self.mass().require()
        if total == 0:
            raise ValidationError("zero-mass atomic measure cannot be normalized")
        normalized_weights: list[Any] = []
        for weight in self.weights:
            try:
                normalized = weight / total
            except (OverflowError, ZeroDivisionError) as exc:
                raise ValidationError("atomic normalization overflowed") from exc
            normalized = _finite_real_result(normalized, name="normalized atomic weight")
            if weight > 0 and normalized == 0:
                raise ValidationError(
                    "atomic normalization underflowed a positive weight to zero"
                )
            normalized_weights.append(normalized)
        result = AtomicMeasure(
            self.points,
            normalized_weights,
            name=f"normalized {self.name}",
        )
        if result.mass().require() == 0:
            raise ValidationError("atomic normalization produced zero mass")
        return result

    def coalesced(self) -> "AtomicMeasure":
        """Merge atoms with exactly equal canonical point tuples."""

        grouped: dict[tuple[RealScalar, ...], list[RealScalar]] = {}
        order: list[tuple[RealScalar, ...]] = []
        for point, weight in zip(self.points, self.weights, strict=True):
            if point not in grouped:
                grouped[point] = [weight]
                order.append(point)
            else:
                grouped[point].append(weight)
        combined = {
            point: sum(weights, Fraction(0))
            if all(isinstance(weight, Fraction) for weight in weights)
            else _finite_float_sum(weights, name="coalesced atomic weight")
            for point, weights in grouped.items()
        }
        points = tuple(point for point in order if combined[point] != 0)
        weights = tuple(combined[point] for point in points)
        if not points:
            points = (self.points[0],)
            weights = (Fraction(0),)
        return AtomicMeasure(points, weights, name=f"coalesced {self.name}")

    def certificate(self) -> Certificate:
        total = self.mass().value
        return Certificate(
            valid=True,
            exactness=self.exactness,
            assumptions_checked=("finite support", "finite coordinates", "nonnegative weights"),
            diagnostics={
                "dimension": self.dimension,
                "atom_count": self.atom_count,
                "mass": str(total),
            },
            theorem_id="INT.MEASURE.ATOM.1",
            message="atomic measure validated",
        )


@dataclass(frozen=True, slots=True, init=False)
class IntervalMeasure:
    """Constant-density Lebesgue measure on one bounded interval."""

    lower: RealScalar
    upper: RealScalar
    density: RealScalar
    dimension: int
    name: str
    exactness: Exactness

    def __init__(
        self,
        lower: Any,
        upper: Any,
        *,
        density: Any = 1,
        name: str = "interval measure",
    ) -> None:
        a = _real(lower, name="lower endpoint")
        b = _real(upper, name="upper endpoint")
        rho = _real(density, name="density")
        if b < a:
            raise ValidationError("interval measures require lower <= upper")
        if rho < 0:
            raise ValidationError("positive interval measures require density >= 0")
        label = str(name).strip()
        if not label:
            raise ValidationError("measure name must be nonempty")
        exact = all(isinstance(value, Fraction) for value in (a, b, rho))
        object.__setattr__(self, "lower", a)
        object.__setattr__(self, "upper", b)
        object.__setattr__(self, "density", rho)
        object.__setattr__(self, "dimension", 1)
        object.__setattr__(self, "name", label)
        object.__setattr__(self, "exactness", Exactness.EXACT if exact else Exactness.NUMERICAL)

    @property
    def structurally_zero(self) -> bool:
        """Whether the interval or its density makes this the zero measure."""

        return self.lower == self.upper or self.density == 0

    def mass(self) -> ComputationResult[Any]:
        if self.exactness is Exactness.EXACT:
            value: Any = self.density * (self.upper - self.lower)
        elif self.structurally_zero:
            value = 0.0
        else:
            lower = _finite_float(self.lower, name="interval lower endpoint")
            upper = _finite_float(self.upper, name="interval upper endpoint")
            density = _finite_float(self.density, name="interval density")
            width = _finite_float(upper - lower, name="interval width")
            try:
                value = density * width
            except OverflowError as exc:
                raise ValidationError("interval mass overflowed floating-point range") from exc
            value = _finite_float(value, name="interval mass")
            if value == 0.0:
                raise ValidationError(
                    "structurally positive interval mass underflowed to zero"
                )
        return ComputationResult(
            value,
            Certificate(
                valid=True,
                exactness=self.exactness,
                assumptions_checked=("bounded interval", "constant nonnegative density"),
                theorem_id="INT.MEASURE.INTERVAL.MASS",
                message="interval mass evaluated",
            ),
        )

    def moment(self, exponents: Sequence[int]) -> ComputationResult[Any]:
        (power,) = _exponents(exponents, 1)
        if power == 0:
            value: Any = self.mass().value
        elif self.exactness is Exactness.EXACT:
            value: Any = self.density * (
                self.upper ** (power + 1) - self.lower ** (power + 1)
            ) / (power + 1)
        elif self.structurally_zero:
            value = 0.0
        else:
            lower = _finite_float(self.lower, name="interval lower endpoint")
            upper = _finite_float(self.upper, name="interval upper endpoint")
            density = _finite_float(self.density, name="interval density")
            try:
                upper_power = upper ** (power + 1)
                lower_power = lower ** (power + 1)
                difference = upper_power - lower_power
                value = density * difference / (power + 1)
            except OverflowError as exc:
                raise ValidationError("interval moment overflowed floating-point range") from exc
            _finite_float(upper_power, name="interval moment endpoint power")
            _finite_float(lower_power, name="interval moment endpoint power")
            _finite_float(difference, name="interval moment power difference")
            value = _finite_float(value, name="interval moment")
        return ComputationResult(
            value,
            Certificate(
                valid=True,
                exactness=self.exactness,
                assumptions_checked=("closed-form monomial antiderivative",),
                diagnostics={"exponent": power},
                theorem_id="INT.MEASURE.INTERVAL.MOMENT",
                message="interval moment evaluated",
            ),
        )

    def normalized(self) -> "IntervalMeasure":
        if self.structurally_zero:
            raise ValidationError("zero-mass interval measure cannot be normalized")
        total = self.mass().require()
        if total == 0:
            raise ValidationError("zero-mass interval measure cannot be normalized")
        if self.exactness is Exactness.EXACT:
            normalized_density: Any = Fraction(1, 1) / (self.upper - self.lower)
        else:
            lower = _finite_float(self.lower, name="interval lower endpoint")
            upper = _finite_float(self.upper, name="interval upper endpoint")
            width = _finite_float(upper - lower, name="interval width")
            try:
                normalized_density = 1.0 / width
            except (OverflowError, ZeroDivisionError) as exc:
                raise ValidationError("interval normalization overflowed") from exc
            normalized_density = _finite_float(
                normalized_density,
                name="normalized interval density",
            )
            if normalized_density == 0.0:
                raise ValidationError(
                    "interval normalization underflowed positive density to zero"
                )
        return IntervalMeasure(
            self.lower,
            self.upper,
            density=normalized_density,
            name=f"normalized {self.name}",
        )

    def pushforward_affine(self, mapping: ExactAffineMap) -> "IntervalMeasure | AtomicMeasure":
        if not isinstance(mapping, ExactAffineMap):
            raise TypeError("interval affine pushforward requires an ExactAffineMap")
        if mapping.source_dimension != 1 or mapping.target_dimension != 1:
            raise DimensionError("interval affine pushforward requires a one-dimensional map")
        scale = mapping.matrix[0][0]
        shift = mapping.shift[0]
        if scale == 0:
            return AtomicMeasure((shift,), (self.mass().require(),), name=f"{mapping.name} pushforward")
        try:
            first = scale * self.lower + shift
            second = scale * self.upper + shift
            transformed_density = self.density / abs(scale)
        except (OverflowError, ZeroDivisionError) as exc:
            raise ValidationError("affine interval pushforward overflowed") from exc
        if self.exactness is Exactness.NUMERICAL:
            first = _finite_real_result(first, name="affine pushforward endpoint")
            second = _finite_real_result(second, name="affine pushforward endpoint")
            transformed_density = _finite_real_result(
                transformed_density,
                name="affine pushforward density",
            )
        if (
            self.exactness is Exactness.NUMERICAL
            and not self.structurally_zero
            and first == second
        ):
            raise ValidationError(
                "affine pushforward collapsed a positive numerical interval"
            )
        result = IntervalMeasure(
            min(first, second),
            max(first, second),
            density=transformed_density,
            name=f"{mapping.name} pushforward of {self.name}",
        )
        if self.exactness is Exactness.NUMERICAL and not self.structurally_zero:
            base_mass = _finite_float(
                self.mass().require(),
                name="source interval mass",
            )
            pushed_mass = _finite_float(
                result.mass().require(),
                name="affine-pushforward interval mass",
            )
            if not isclose(base_mass, pushed_mass, rel_tol=1e-12, abs_tol=0.0):
                raise ValidationError(
                    "affine pushforward lost numerical interval mass"
                )
        return result

    def certificate(self) -> Certificate:
        return Certificate(
            valid=True,
            exactness=self.exactness,
            assumptions_checked=("ordered finite endpoints", "constant nonnegative density"),
            diagnostics={
                "lower": str(self.lower),
                "upper": str(self.upper),
                "density": str(self.density),
                "mass": str(self.mass().value),
            },
            theorem_id="INT.MEASURE.INTERVAL.1",
            message="interval measure validated",
        )


@dataclass(frozen=True, slots=True, init=False)
class ProductMeasure:
    """Finite product of typed component measures."""

    factors: tuple[Measure, ...]
    dimension: int
    name: str
    exactness: Exactness

    def __init__(self, factors: Iterable[Measure], *, name: str = "product measure") -> None:
        values = tuple(factors)
        if not values:
            raise ValidationError("product measures require at least one factor")
        if any(not isinstance(value, Measure) for value in values):
            raise TypeError("every product factor must implement the Measure protocol")
        label = str(name).strip()
        if not label:
            raise ValidationError("measure name must be nonempty")
        object.__setattr__(self, "factors", values)
        object.__setattr__(self, "dimension", sum(value.dimension for value in values))
        object.__setattr__(self, "name", label)
        object.__setattr__(
            self,
            "exactness",
            _aggregate_exactness(value.exactness for value in values),
        )

    def mass(self) -> ComputationResult[Any]:
        results: list[ComputationResult[Any]] = []
        factor_certificates: list[Certificate] = []
        for factor in self.factors:
            result = factor.mass()
            results.append(result)
            factor_certificates.append(factor.certificate())
        value = _finite_product(
            (result.value for result in results),
            name="product-measure mass",
        )
        if value == 0 and all(result.value > 0 for result in results):
            raise ValidationError("product-measure mass underflowed to zero")
        certificates = tuple(factor_certificates) + tuple(
            result.certificate for result in results
        )
        valid = all(certificate.valid for certificate in certificates)
        exactness = _aggregate_exactness(
            (self.exactness, *(certificate.exactness for certificate in certificates))
        )
        return ComputationResult(
            value,
            Certificate(
                valid=valid,
                exactness=exactness,
                assumptions_checked=_unique_assumptions(
                    ("finite product mass",),
                    *(certificate.assumptions_checked for certificate in certificates),
                ),
                assumptions_unchecked=_unique_assumptions(
                    *(certificate.assumptions_unchecked for certificate in certificates)
                ),
                diagnostics={
                    "factor_count": len(self.factors),
                    "factor_validity": tuple(
                        certificate.valid for certificate in factor_certificates
                    ),
                    "factor_exactness": tuple(
                        certificate.exactness.value for certificate in factor_certificates
                    ),
                },
                theorem_id="INT.MEASURE.PRODUCT.MASS",
                message=(
                    "product mass evaluated"
                    if valid
                    else "a product-measure factor or mass result is invalid"
                ),
            ),
        )

    def moment(self, exponents: Sequence[int]) -> ComputationResult[Any]:
        powers = _exponents(exponents, self.dimension)
        cursor = 0
        results: list[ComputationResult[Any]] = []
        factor_certificates: list[Certificate] = []
        for factor in self.factors:
            piece = powers[cursor : cursor + factor.dimension]
            result = factor.moment(piece)
            results.append(result)
            factor_certificates.append(factor.certificate())
            cursor += factor.dimension
        value = _finite_product(
            (result.value for result in results),
            name="product-measure moment",
        )
        certificates = tuple(factor_certificates) + tuple(
            result.certificate for result in results
        )
        valid = all(certificate.valid for certificate in certificates)
        exactness = _aggregate_exactness(
            (self.exactness, *(certificate.exactness for certificate in certificates))
        )
        return ComputationResult(
            value,
            Certificate(
                valid=valid,
                exactness=exactness,
                assumptions_checked=_unique_assumptions(
                    ("factorization of product-measure monomial moments",),
                    *(certificate.assumptions_checked for certificate in certificates),
                ),
                assumptions_unchecked=_unique_assumptions(
                    *(certificate.assumptions_unchecked for certificate in certificates)
                ),
                diagnostics={
                    "exponents": powers,
                    "factor_count": len(self.factors),
                    "factor_validity": tuple(
                        certificate.valid for certificate in factor_certificates
                    ),
                    "factor_exactness": tuple(
                        certificate.exactness.value for certificate in factor_certificates
                    ),
                },
                theorem_id="INT.MEASURE.PRODUCT.MOMENT",
                message=(
                    "product moment evaluated"
                    if valid
                    else "a product-measure factor or moment result is invalid"
                ),
            ),
        )

    def certificate(self) -> Certificate:
        factor_certificates = tuple(factor.certificate() for factor in self.factors)
        mass_result = self.mass()
        mass_certificate = mass_result.certificate
        certificates = factor_certificates + (mass_certificate,)
        valid = all(certificate.valid for certificate in certificates)
        return Certificate(
            valid=valid,
            exactness=_aggregate_exactness(
                (self.exactness, *(certificate.exactness for certificate in certificates))
            ),
            assumptions_checked=_unique_assumptions(
                ("typed finite measure factors", "finite Cartesian product"),
                *(certificate.assumptions_checked for certificate in certificates),
            ),
            assumptions_unchecked=_unique_assumptions(
                *(certificate.assumptions_unchecked for certificate in certificates)
            ),
            diagnostics={
                "dimension": self.dimension,
                "factor_count": len(self.factors),
                "mass": str(mass_result.value),
                "factor_validity": tuple(
                    certificate.valid for certificate in factor_certificates
                ),
                "factor_exactness": tuple(
                    certificate.exactness.value for certificate in factor_certificates
                ),
            },
            theorem_id="INT.MEASURE.PRODUCT.1",
            message="product measure validated" if valid else "a product-measure factor is invalid",
        )


@dataclass(frozen=True, slots=True, init=False)
class PushforwardMeasure:
    r"""The pushforward ``mapping_# base``.

    For exact affine maps, polynomial pullback is exact.  For arbitrary
    callables, the object remains a definitionally correct pushforward, while
    global mapping regularity, purity, and dimension claims remain explicit
    unchecked assumptions.  An atomic materialization is only a one-time
    evaluated snapshot and does not discharge those callable assumptions.
    """

    base: Measure
    mapping: ExactAffineMap | Callable[[Any], Any]
    dimension: int
    name: str
    exactness: Exactness

    def __init__(
        self,
        base: Measure,
        mapping: ExactAffineMap | Callable[[Any], Any],
        *,
        target_dimension: int | None = None,
        name: str = "pushforward measure",
    ) -> None:
        if not isinstance(base, Measure):
            raise TypeError("pushforward base must implement the Measure protocol")
        if isinstance(mapping, ExactAffineMap):
            if mapping.source_dimension != base.dimension:
                raise DimensionError("affine source dimension does not match the base measure")
            dimension = mapping.target_dimension
            exactness = _aggregate_exactness((base.exactness, Exactness.EXACT))
        elif callable(mapping):
            if target_dimension is None:
                raise DimensionError("callable pushforwards require target_dimension")
            dimension = _positive_integer(target_dimension, name="pushforward target dimension")
            exactness = _aggregate_exactness((base.exactness, Exactness.CONDITIONAL))
        else:
            raise TypeError("pushforward mapping must be callable or ExactAffineMap")
        label = str(name).strip()
        if not label:
            raise ValidationError("measure name must be nonempty")
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "mapping", mapping)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "name", label)
        object.__setattr__(self, "exactness", exactness)

    def mass(self) -> ComputationResult[Any]:
        base_mass = self.base.mass()
        value = _finite_real_result(base_mass.value, name="pushforward mass")
        base_certificate = self.base.certificate()
        exact_map = isinstance(self.mapping, ExactAffineMap)
        map_certificate = self.mapping.certificate() if exact_map else None
        certificates = (base_mass.certificate, base_certificate) + (
            (map_certificate,) if map_certificate is not None else ()
        )
        valid = all(certificate.valid for certificate in certificates)
        return ComputationResult(
            value,
            Certificate(
                valid=valid,
                exactness=_aggregate_exactness(
                    (
                        self.exactness,
                        *(certificate.exactness for certificate in certificates),
                    )
                ),
                assumptions_checked=_unique_assumptions(
                    ("pushforward preserves total mass",),
                    *(certificate.assumptions_checked for certificate in certificates),
                ),
                assumptions_unchecked=_unique_assumptions(
                    *(certificate.assumptions_unchecked for certificate in certificates),
                    () if exact_map else _CALLABLE_PUSHFORWARD_ASSUMPTIONS,
                ),
                diagnostics={
                    "base_valid": base_certificate.valid,
                    "base_exactness": base_certificate.exactness.value,
                    "exact_affine": exact_map,
                },
                theorem_id="INT.MEASURE.PUSH.MASS",
                message=(
                    "pushforward mass inherited from base measure"
                    if valid
                    else "pushforward mass inherited an invalid base certificate"
                ),
            ),
        )

    def map_point(self, point: Any) -> tuple[RealScalar, ...]:
        """Apply the map and validate one finite, dimension-correct target point."""

        source = _canonical_point(point, self.base.dimension)
        try:
            if isinstance(self.mapping, ExactAffineMap):
                raw = self.mapping.apply_point(source)
            else:
                raw = _call_point(self.mapping, source)
        except OverflowError as exc:
            raise ValidationError("pushforward mapping overflowed") from exc
        return _canonical_point(raw, self.dimension)

    def materialize_atomic(self) -> AtomicMeasure:
        """Evaluate an atomic image snapshot without certifying callable purity."""

        if not isinstance(self.base, AtomicMeasure):
            raise ValidationError("only atomic pushforwards can be materialized exactly")
        points = [self.map_point(point) for point in self.base.points]
        return AtomicMeasure(
            points,
            self.base.weights,
            name=f"materialized {self.name}",
        ).coalesced()

    def moment(self, exponents: Sequence[int]) -> ComputationResult[Any]:
        powers = _exponents(exponents, self.dimension)
        if isinstance(self.mapping, ExactAffineMap):
            monomial = ExactPolynomial.monomial(1, powers)
            pulled = self.mapping.pullback(monomial)
            from .quadrature import integrate

            result: Any = integrate(pulled, self.base)
        elif isinstance(self.base, AtomicMeasure):
            result = self.materialize_atomic().moment(powers)
        else:
            raise ValidationError(
                "moments of a callable pushforward over a non-atomic base require explicit integration"
            )
        structure = self.certificate()
        certificates = (result.certificate, structure)
        valid = all(certificate.valid for certificate in certificates)
        return ComputationResult(
            result.value,
            Certificate(
                valid=valid,
                exactness=_aggregate_exactness(
                    certificate.exactness for certificate in certificates
                ),
                assumptions_checked=_unique_assumptions(
                    ("pushforward moment equals the pulled-back monomial integral",),
                    *(certificate.assumptions_checked for certificate in certificates),
                ),
                assumptions_unchecked=_unique_assumptions(
                    *(certificate.assumptions_unchecked for certificate in certificates)
                ),
                diagnostics={
                    "exponents": powers,
                    "exact_affine": isinstance(self.mapping, ExactAffineMap),
                },
                theorem_id="INT.MEASURE.PUSH.MOMENT",
                message=(
                    "pushforward moment evaluated"
                    if valid
                    else "pushforward moment inherited an invalid certificate"
                ),
            ),
        )

    def certificate(self) -> Certificate:
        exact_map = isinstance(self.mapping, ExactAffineMap)
        base_certificate = self.base.certificate()
        map_certificate = self.mapping.certificate() if exact_map else None
        certificates = (base_certificate,) + (
            (map_certificate,) if map_certificate is not None else ()
        )
        valid = all(certificate.valid for certificate in certificates)
        return Certificate(
            valid=valid,
            exactness=_aggregate_exactness(
                (
                    self.exactness,
                    *(certificate.exactness for certificate in certificates),
                )
            ),
            assumptions_checked=_unique_assumptions(
                (
                    "finite base measure",
                    "typed affine mapping"
                    if exact_map
                    else "pushforward defined by precomposition",
                ),
                *(certificate.assumptions_checked for certificate in certificates),
            ),
            assumptions_unchecked=_unique_assumptions(
                *(certificate.assumptions_unchecked for certificate in certificates),
                () if exact_map else _CALLABLE_PUSHFORWARD_ASSUMPTIONS,
            ),
            diagnostics={
                "source_dimension": self.base.dimension,
                "target_dimension": self.dimension,
                "exact_affine": exact_map,
                "base_valid": base_certificate.valid,
                "base_exactness": base_certificate.exactness.value,
            },
            theorem_id="INT.MEASURE.PUSH.1",
            message=(
                "pushforward measure validated"
                if valid
                else "pushforward measure inherited an invalid base or map certificate"
            ),
        )


def product_measure(*factors: Measure, name: str = "product measure") -> ProductMeasure:
    return ProductMeasure(factors, name=name)


def pushforward_measure(
    base: Measure,
    mapping: ExactAffineMap | Callable[[Any], Any],
    *,
    target_dimension: int | None = None,
    name: str = "pushforward measure",
) -> PushforwardMeasure:
    return PushforwardMeasure(
        base,
        mapping,
        target_dimension=target_dimension,
        name=name,
    )


__all__ = [
    "AtomicMeasure",
    "ExactAffineMap",
    "IntervalMeasure",
    "Measure",
    "ProductMeasure",
    "PushforwardMeasure",
    "RealScalar",
    "product_measure",
    "pushforward_measure",
]
