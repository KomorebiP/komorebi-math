"""Exact formal changes of variable and finite jet coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from .budget import DEFAULT_COORDINATE_BUDGET, CoordinateBudget
from .exact import Scalar, Vector, as_vector, matrix_from_columns
from .transform import CoordinateTransform


def _multiply_truncated(left: Sequence[Fraction], right: Sequence[Fraction], degree: int) -> list[Fraction]:
    output = [Fraction(0) for _ in range(degree + 1)]
    for i, left_value in enumerate(left):
        if i > degree or left_value == 0:
            continue
        for j, right_value in enumerate(right):
            if i + j > degree:
                break
            if right_value != 0:
                output[i + j] += left_value * right_value
    return output


def compose_ordinary_series(
    outer: Sequence[Scalar],
    inner: Sequence[Scalar],
    degree: int,
) -> Vector:
    r"""Return ``outer(inner(z))`` modulo ``z^(degree+1)`` exactly."""

    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
        raise ValidationError("composition degree must be a nonnegative integer")
    f = as_vector(outer)
    g = as_vector(inner)
    if not g:
        raise DimensionError("inner series must be nonempty")
    output = [Fraction(0) for _ in range(degree + 1)]
    power = [Fraction(0) for _ in range(degree + 1)]
    power[0] = 1
    for exponent in range(min(len(f), degree + 1)):
        coefficient = f[exponent]
        if coefficient != 0:
            for index in range(degree + 1):
                output[index] += coefficient * power[index]
        power = _multiply_truncated(power, g, degree)
    return tuple(output)


@dataclass(frozen=True, slots=True)
class FormalCoordinateChange:
    r"""A finite exact jet ``phi(z)=a_1 z+a_2 z^2+...``.

    The constant coefficient must vanish and the linear coefficient must be
    nonzero.  The object represents a formal coordinate change modulo the
    first omitted power.
    """

    name: str
    coefficients: Vector
    source: str = "source formal coordinate"
    target: str = "target formal coordinate"

    def __post_init__(self) -> None:
        values = as_vector(self.coefficients)
        if len(values) < 2:
            raise DimensionError("a formal coordinate jet needs constant and linear coefficients")
        if values[0] != 0:
            raise ValidationError("formal coordinate changes must fix the origin")
        if values[1] == 0:
            raise ValidationError("formal coordinate changes need a nonzero linear coefficient")
        if not str(self.name).strip():
            raise ValidationError("formal coordinate-change names must be nonempty")
        object.__setattr__(self, "coefficients", values)
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "target", str(self.target))

    @property
    def degree(self) -> int:
        """Return the retained jet degree."""

        return len(self.coefficients) - 1

    @property
    def linear_coefficient(self) -> Fraction:
        """Return the exact derivative at the origin."""

        return self.coefficients[1]

    def truncate(self, degree: int) -> "FormalCoordinateChange":
        """Return the same change truncated to a smaller positive degree."""

        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
            raise ValidationError("formal coordinate degree must be a positive integer")
        if degree > self.degree:
            raise DimensionError("cannot extend a formal coordinate jet by truncation")
        return FormalCoordinateChange(
            self.name,
            self.coefficients[: degree + 1],
            self.source,
            self.target,
        )

    def compose(
        self,
        inner: "FormalCoordinateChange",
        *,
        degree: int | None = None,
        name: str | None = None,
    ) -> "FormalCoordinateChange":
        r"""Return ``self(inner(z))`` as a formal coordinate change."""

        if not isinstance(inner, FormalCoordinateChange):
            raise ValidationError("inner must be a FormalCoordinateChange")
        if inner.target != self.source:
            raise ValidationError("formal coordinate names do not compose")
        bound = min(self.degree, inner.degree) if degree is None else degree
        if isinstance(bound, bool) or not isinstance(bound, int) or bound < 1:
            raise ValidationError("composition degree must be positive")
        if bound > self.degree or bound > inner.degree:
            raise DimensionError("composition degree exceeds an available jet")
        values = compose_ordinary_series(self.coefficients, inner.coefficients, bound)
        return FormalCoordinateChange(
            name or f"{self.name} after {inner.name}",
            values,
            inner.source,
            self.target,
        )

    def inverse(self, *, degree: int | None = None, name: str | None = None) -> "FormalCoordinateChange":
        """Return the unique compositional inverse jet."""

        bound = self.degree if degree is None else degree
        if isinstance(bound, bool) or not isinstance(bound, int) or bound < 1:
            raise ValidationError("inverse degree must be positive")
        if bound > self.degree:
            raise DimensionError("inverse degree exceeds the available jet")
        inverse = [Fraction(0) for _ in range(bound + 1)]
        inverse[1] = 1 / self.linear_coefficient
        target = [Fraction(0) for _ in range(bound + 1)]
        target[1] = 1
        for current_degree in range(2, bound + 1):
            trial = inverse.copy()
            trial[current_degree] = 0
            composed = compose_ordinary_series(
                self.coefficients[: bound + 1],
                trial,
                current_degree,
            )
            inverse[current_degree] = (target[current_degree] - composed[current_degree]) / self.linear_coefficient
        return FormalCoordinateChange(
            name or f"inverse {self.name}",
            tuple(inverse),
            self.target,
            self.source,
        )

    def pullback_coefficients(
        self,
        coefficients: Sequence[Scalar],
        *,
        degree: int | None = None,
    ) -> Vector:
        r"""Return coefficients of ``f(self(z))`` in ordinary powers."""

        values = as_vector(coefficients)
        if not values:
            return ()
        bound = min(len(values) - 1, self.degree) if degree is None else degree
        if isinstance(bound, bool) or not isinstance(bound, int) or bound < 0:
            raise ValidationError("pullback degree must be nonnegative")
        if bound > self.degree:
            raise DimensionError("pullback degree exceeds the coordinate jet")
        return compose_ordinary_series(values, self.coefficients, bound)

    def pushforward_coefficients(
        self,
        coefficients: Sequence[Scalar],
        *,
        degree: int | None = None,
    ) -> Vector:
        r"""Return coefficients of ``f(self^(-1)(z))`` in ordinary powers."""

        values = as_vector(coefficients)
        if not values:
            return ()
        bound = min(len(values) - 1, self.degree) if degree is None else degree
        if isinstance(bound, bool) or not isinstance(bound, int) or bound < 0:
            raise ValidationError("pushforward degree must be nonnegative")
        if bound > self.degree:
            raise DimensionError("pushforward degree exceeds the coordinate jet")
        if bound == 0:
            return values[:1]
        return self.inverse(degree=bound).pullback_coefficients(values, degree=bound)

    def substitution_transform(
        self,
        degree: int | None = None,
        *,
        budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
    ) -> CoordinateTransform:
        r"""Return the matrix of ``f(z) -> f(self(z))`` on truncated powers."""

        bound = self.degree if degree is None else degree
        if isinstance(bound, bool) or not isinstance(bound, int) or bound < 0:
            raise ValidationError("substitution degree must be nonnegative")
        if bound > self.degree:
            raise DimensionError("substitution degree exceeds the coordinate jet")
        budget.require_matrix(bound + 1, bound + 1, operation="formal substitution transform")
        columns: list[Vector] = []
        for exponent in range(bound + 1):
            monomial = tuple(Fraction(int(index == exponent)) for index in range(bound + 1))
            columns.append(compose_ordinary_series(monomial, self.coefficients, bound))
        return CoordinateTransform(
            f"ordinary coefficients in {self.target} through degree {bound}",
            f"ordinary coefficients pulled back to {self.source} through degree {bound}",
            matrix_from_columns(columns),
            "CC.J2",
            f"formal substitution by {self.name}",
        )

    def certificate(self) -> Certificate:
        """Return an exact inverse-jet certificate."""

        inverse = self.inverse()
        identity = tuple(Fraction(int(index == 1)) for index in range(self.degree + 1))
        left = compose_ordinary_series(self.coefficients, inverse.coefficients, self.degree)
        right = compose_ordinary_series(inverse.coefficients, self.coefficients, self.degree)
        valid = left == identity and right == identity
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT,
            assumptions_checked=("origin fixed", "nonzero linear coefficient", "finite jet truncation"),
            diagnostics={
                "degree": self.degree,
                "linear_coefficient": self.linear_coefficient,
                "left_composition": left,
                "right_composition": right,
            },
            theorem_id="CC.J1",
            message="formal coordinate jet has a two-sided inverse" if valid else "formal inverse check failed",
        )


def identity_formal_coordinate(
    degree: int,
    *,
    name: str = "identity formal coordinate",
    coordinate: str = "formal coordinate",
) -> FormalCoordinateChange:
    """Return the identity formal coordinate jet."""

    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValidationError("identity jet degree must be positive")
    coefficients = tuple(Fraction(int(index == 1)) for index in range(degree + 1))
    return FormalCoordinateChange(name, coefficients, coordinate, coordinate)


def formal_coordinate_from_iterable(
    coefficients: Iterable[Scalar],
    *,
    name: str = "custom formal coordinate",
    source: str = "source formal coordinate",
    target: str = "target formal coordinate",
) -> FormalCoordinateChange:
    """Construct a formal coordinate change from exact coefficients."""

    return FormalCoordinateChange(name, as_vector(coefficients), source, target)


__all__ = [
    "FormalCoordinateChange",
    "compose_ordinary_series",
    "formal_coordinate_from_iterable",
    "identity_formal_coordinate",
]
