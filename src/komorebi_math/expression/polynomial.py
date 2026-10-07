"""Replayable exact polynomial primitives for the expression calculus.

Every matrix in this module is derived from the snapshotted basis witnesses in
its source and target types.  Display names and caller supplied theorem labels
never determine mathematical behavior.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Any

from ..certificates import Certificate, Exactness
from ..coordinates.exact import (
    Matrix,
    Vector,
    inverse_matrix,
    matrix_from_columns,
    matrix_multiply,
    matrix_shape,
    matrix_vector_multiply,
    zero_matrix,
)
from ..exceptions import DimensionError, ValidationError
from .model import (
    RATIONAL_SPACE,
    Expression,
    LinearMapType,
    PolynomialCoordinateSpace,
    RationalSpace,
)


def _fraction(value: Any, *, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise ValidationError(f"{name} must be an integer or Fraction")
    return Fraction(value)


def _trim(polynomial: Sequence[Fraction]) -> Vector:
    values = list(polynomial)
    if not values:
        return (Fraction(0),)
    while len(values) > 1 and values[-1] == 0:
        values.pop()
    return tuple(values)


def _pad(polynomial: Sequence[Fraction], size: int) -> Vector:
    values = _trim(polynomial)
    if len(values) > size:
        raise DimensionError("polynomial image exceeds the declared target degree")
    return values + tuple(Fraction(0) for _ in range(size - len(values)))


def _evaluate(polynomial: Sequence[Fraction], point: Fraction) -> Fraction:
    result = Fraction(0)
    for coefficient in reversed(polynomial):
        result = result * point + coefficient
    return result


def _add(left: Sequence[Fraction], right: Sequence[Fraction]) -> Vector:
    size = max(len(left), len(right))
    return _trim(
        tuple(
            (left[index] if index < len(left) else Fraction(0))
            + (right[index] if index < len(right) else Fraction(0))
            for index in range(size)
        )
    )


def _subtract(left: Sequence[Fraction], right: Sequence[Fraction]) -> Vector:
    return _add(left, tuple(-value for value in right))


def _multiply(left: Sequence[Fraction], right: Sequence[Fraction]) -> Vector:
    output = [Fraction(0) for _ in range(len(left) + len(right) - 1)]
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            output[left_index + right_index] += left_value * right_value
    return _trim(output)


def _compose_affine(
    polynomial: Sequence[Fraction],
    scale: Fraction,
    shift: Fraction,
) -> Vector:
    affine = (shift, scale)
    result: Vector = (Fraction(0),)
    for coefficient in reversed(polynomial):
        result = _add(_multiply(result, affine), (coefficient,))
    return _trim(result)


def _derivative(polynomial: Sequence[Fraction]) -> Vector:
    if len(polynomial) <= 1:
        return (Fraction(0),)
    return _trim(tuple(Fraction(index) * polynomial[index] for index in range(1, len(polynomial))))


def _antiderivative(polynomial: Sequence[Fraction], basepoint: Fraction) -> Vector:
    result = [Fraction(0)] + [
        coefficient / Fraction(index + 1)
        for index, coefficient in enumerate(polynomial)
    ]
    result[0] = -_evaluate(result, basepoint)
    return _trim(result)


def _basis_matrix(space: PolynomialCoordinateSpace) -> Matrix:
    return matrix_from_columns(
        tuple(_pad(polynomial, space.dimension) for polynomial in space.basis)
    )


def _operator_matrix(
    source: PolynomialCoordinateSpace,
    target: PolynomialCoordinateSpace,
    rule: Callable[[Vector], Vector],
) -> Matrix:
    target_inverse = inverse_matrix(_basis_matrix(target))
    columns: list[Vector] = []
    for polynomial in source.basis:
        physical_image = _pad(rule(polynomial), target.dimension)
        columns.append(matrix_vector_multiply(target_inverse, physical_image))
    return matrix_from_columns(columns)


def _row(values: Iterable[Fraction]) -> Matrix:
    return (tuple(values),)


def _normalize_matrix(matrix: Iterable[Iterable[Any]]) -> Matrix:
    try:
        rows = tuple(matrix)
    except TypeError as exc:
        raise ValidationError("claimed primitive matrix must be a finite row iterable") from exc
    normalized: list[tuple[Fraction, ...]] = []
    for row in rows:
        try:
            entries = tuple(row)
        except TypeError as exc:
            raise ValidationError("claimed primitive matrix rows must be finite") from exc
        normalized.append(
            tuple(_fraction(value, name="claimed matrix entry") for value in entries)
        )
    if normalized and any(len(row) != len(normalized[0]) for row in normalized):
        raise DimensionError("claimed primitive matrix rows must have a common length")
    return tuple(normalized)


class PrimitiveKind(StrEnum):
    COORDINATE_CHANGE = "coordinate_change"
    DERIVATIVE = "derivative"
    ANTIDERIVATIVE = "antiderivative"
    EVALUATION = "evaluation"
    INTERVAL_INTEGRATION = "interval_integration"
    AFFINE_SUBSTITUTION = "affine_substitution"
    FINITE_DIFFERENCE = "finite_difference"
    BOUNDARY_FUNCTIONAL = "boundary_functional"
    CONSTANT_INJECTION = "constant_injection"


_THEOREM_IDS = {
    PrimitiveKind.COORDINATE_CHANGE: "EXPR.POLY.COORDINATE",
    PrimitiveKind.DERIVATIVE: "EXPR.POLY.DERIVATIVE",
    PrimitiveKind.ANTIDERIVATIVE: "EXPR.POLY.ANTIDERIVATIVE",
    PrimitiveKind.EVALUATION: "EXPR.POLY.EVALUATION",
    PrimitiveKind.INTERVAL_INTEGRATION: "EXPR.POLY.INTEGRATION",
    PrimitiveKind.AFFINE_SUBSTITUTION: "EXPR.POLY.AFFINE",
    PrimitiveKind.FINITE_DIFFERENCE: "EXPR.POLY.DIFFERENCE",
    PrimitiveKind.BOUNDARY_FUNCTIONAL: "EXPR.POLY.BOUNDARY",
    PrimitiveKind.CONSTANT_INJECTION: "EXPR.POLY.CONSTANT",
}


def _polynomial_source(map_type: LinearMapType) -> PolynomialCoordinateSpace:
    if not isinstance(map_type.source, PolynomialCoordinateSpace):
        raise ValidationError("primitive source must be a polynomial-coordinate space")
    return map_type.source


def _polynomial_target(map_type: LinearMapType) -> PolynomialCoordinateSpace:
    if not isinstance(map_type.target, PolynomialCoordinateSpace):
        raise ValidationError("primitive target must be a polynomial-coordinate space")
    return map_type.target


def _rational_source(map_type: LinearMapType) -> RationalSpace:
    if not isinstance(map_type.source, RationalSpace):
        raise ValidationError("primitive source must be the rational scalar space")
    return map_type.source


def _rational_target(map_type: LinearMapType) -> RationalSpace:
    if not isinstance(map_type.target, RationalSpace):
        raise ValidationError("primitive target must be the rational scalar space")
    return map_type.target


def _same_variable(
    source: PolynomialCoordinateSpace,
    target: PolynomialCoordinateSpace,
) -> None:
    if source.variable != target.variable:
        raise ValidationError("polynomial primitives require the same physical variable")


def _validate_primitive(
    kind: PrimitiveKind,
    map_type: LinearMapType,
    parameters: tuple[Fraction, ...],
) -> None:
    if kind is PrimitiveKind.COORDINATE_CHANGE:
        source = _polynomial_source(map_type)
        target = _polynomial_target(map_type)
        _same_variable(source, target)
        if source.degree != target.degree:
            raise DimensionError("coordinate changes require equal polynomial degrees")
        if parameters:
            raise ValidationError("coordinate changes do not take rational parameters")
        return

    if kind is PrimitiveKind.DERIVATIVE:
        source = _polynomial_source(map_type)
        target = _polynomial_target(map_type)
        expected = source if source.degree == 0 else source.truncate(source.degree - 1)
        if target != expected:
            raise ValidationError("derivative target is not the tight compatible truncation")
        if parameters:
            raise ValidationError("derivatives do not take rational parameters")
        return

    if kind is PrimitiveKind.ANTIDERIVATIVE:
        source = _polynomial_source(map_type)
        target = _polynomial_target(map_type)
        _same_variable(source, target)
        if target.degree != source.degree + 1:
            raise DimensionError("antiderivative target degree must be source degree plus one")
        if target.truncate(source.degree) != source:
            raise ValidationError("antiderivative target must extend the exact source basis witness")
        if len(parameters) != 1:
            raise ValidationError("antiderivatives require exactly one basepoint")
        return

    if kind is PrimitiveKind.EVALUATION:
        _polynomial_source(map_type)
        _rational_target(map_type)
        if len(parameters) != 1:
            raise ValidationError("evaluation requires exactly one point")
        return

    if kind in (PrimitiveKind.INTERVAL_INTEGRATION, PrimitiveKind.BOUNDARY_FUNCTIONAL):
        _polynomial_source(map_type)
        _rational_target(map_type)
        if len(parameters) != 3:
            raise ValidationError("interval functionals require lower, upper, and density")
        lower, upper, density = parameters
        if upper < lower:
            raise ValidationError("interval endpoints must satisfy lower <= upper")
        if density < 0:
            raise ValidationError("interval density must be nonnegative")
        return

    if kind is PrimitiveKind.AFFINE_SUBSTITUTION:
        source = _polynomial_source(map_type)
        target = _polynomial_target(map_type)
        if target != source:
            raise ValidationError("affine substitution is an endomorphism of one coordinate space")
        if len(parameters) != 2:
            raise ValidationError("affine substitution requires scale and shift")
        return

    if kind is PrimitiveKind.FINITE_DIFFERENCE:
        source = _polynomial_source(map_type)
        target = _polynomial_target(map_type)
        if target != source:
            raise ValidationError("finite difference is an endomorphism of one coordinate space")
        if len(parameters) != 1:
            raise ValidationError("finite difference requires exactly one step")
        if parameters[0] == 0:
            raise ValidationError("finite-difference step must be nonzero")
        return

    if kind is PrimitiveKind.CONSTANT_INJECTION:
        _rational_source(map_type)
        _polynomial_target(map_type)
        if parameters:
            raise ValidationError("constant injection does not take rational parameters")
        return

    raise ValidationError(f"unsupported polynomial primitive kind: {kind}")


@dataclass(frozen=True, slots=True)
class PolynomialPrimitive(Expression):
    """One replayable basis-aware polynomial linear map."""

    kind: PrimitiveKind
    map_type: LinearMapType
    parameters: tuple[Fraction, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PrimitiveKind):
            raise TypeError("kind must be a PrimitiveKind")
        if not isinstance(self.map_type, LinearMapType):
            raise TypeError("map_type must be a LinearMapType")
        try:
            raw_parameters = tuple(self.parameters)
        except TypeError as exc:
            raise ValidationError("primitive parameters must be finite") from exc
        parameters = tuple(
            _fraction(value, name="primitive parameter") for value in raw_parameters
        )
        _validate_primitive(self.kind, self.map_type, parameters)
        object.__setattr__(self, "parameters", parameters)

    @property
    def expression_type(self) -> LinearMapType:
        return self.map_type

    @property
    def theorem_id(self) -> str:
        return _THEOREM_IDS[self.kind]

    @property
    def matrix(self) -> Matrix:
        return replay(self)

    def certificate(self, *, claimed_matrix: Iterable[Iterable[Any]] | None = None) -> Certificate:
        return primitive_certificate(self, claimed_matrix=claimed_matrix)


def coordinate_change(
    source: PolynomialCoordinateSpace,
    target: PolynomialCoordinateSpace,
) -> PolynomialPrimitive:
    return PolynomialPrimitive(
        PrimitiveKind.COORDINATE_CHANGE,
        LinearMapType(source, target),
    )


def derivative(source: PolynomialCoordinateSpace) -> PolynomialPrimitive:
    """Return the tight derivative; degree zero uses its zero endomorphism."""

    if not isinstance(source, PolynomialCoordinateSpace):
        raise TypeError("derivative source must be a PolynomialCoordinateSpace")
    target = source if source.degree == 0 else source.truncate(source.degree - 1)
    return PolynomialPrimitive(
        PrimitiveKind.DERIVATIVE,
        LinearMapType(source, target),
    )


def antiderivative(
    source: PolynomialCoordinateSpace,
    target: PolynomialCoordinateSpace,
    basepoint: int | Fraction = 0,
) -> PolynomialPrimitive:
    return PolynomialPrimitive(
        PrimitiveKind.ANTIDERIVATIVE,
        LinearMapType(source, target),
        (_fraction(basepoint, name="antiderivative basepoint"),),
    )


def evaluation(
    source: PolynomialCoordinateSpace,
    point: int | Fraction,
) -> PolynomialPrimitive:
    return PolynomialPrimitive(
        PrimitiveKind.EVALUATION,
        LinearMapType(source, RATIONAL_SPACE),
        (_fraction(point, name="evaluation point"),),
    )


def interval_integration(
    source: PolynomialCoordinateSpace,
    lower: int | Fraction,
    upper: int | Fraction,
    density: int | Fraction = 1,
) -> PolynomialPrimitive:
    return PolynomialPrimitive(
        PrimitiveKind.INTERVAL_INTEGRATION,
        LinearMapType(source, RATIONAL_SPACE),
        (
            _fraction(lower, name="lower endpoint"),
            _fraction(upper, name="upper endpoint"),
            _fraction(density, name="interval density"),
        ),
    )


def affine_substitution(
    source: PolynomialCoordinateSpace,
    scale: int | Fraction = 1,
    shift: int | Fraction = 0,
) -> PolynomialPrimitive:
    return PolynomialPrimitive(
        PrimitiveKind.AFFINE_SUBSTITUTION,
        LinearMapType(source, source),
        (
            _fraction(scale, name="affine scale"),
            _fraction(shift, name="affine shift"),
        ),
    )


def finite_difference(
    source: PolynomialCoordinateSpace,
    step: int | Fraction = 1,
) -> PolynomialPrimitive:
    return PolynomialPrimitive(
        PrimitiveKind.FINITE_DIFFERENCE,
        LinearMapType(source, source),
        (_fraction(step, name="finite-difference step"),),
    )


def boundary_functional(
    source: PolynomialCoordinateSpace,
    lower: int | Fraction,
    upper: int | Fraction,
    density: int | Fraction = 1,
) -> PolynomialPrimitive:
    return PolynomialPrimitive(
        PrimitiveKind.BOUNDARY_FUNCTIONAL,
        LinearMapType(source, RATIONAL_SPACE),
        (
            _fraction(lower, name="lower endpoint"),
            _fraction(upper, name="upper endpoint"),
            _fraction(density, name="boundary density"),
        ),
    )


def constant_injection(target: PolynomialCoordinateSpace) -> PolynomialPrimitive:
    return PolynomialPrimitive(
        PrimitiveKind.CONSTANT_INJECTION,
        LinearMapType(RATIONAL_SPACE, target),
    )


def replay(primitive: PolynomialPrimitive) -> Matrix:
    """Recompute a primitive matrix solely from exact structural witnesses."""

    if not isinstance(primitive, PolynomialPrimitive):
        raise TypeError("replay requires a PolynomialPrimitive")
    kind = primitive.kind
    map_type = primitive.map_type

    if kind is PrimitiveKind.COORDINATE_CHANGE:
        source = _polynomial_source(map_type)
        target = _polynomial_target(map_type)
        return matrix_multiply(inverse_matrix(_basis_matrix(target)), _basis_matrix(source))

    if kind is PrimitiveKind.DERIVATIVE:
        source = _polynomial_source(map_type)
        target = _polynomial_target(map_type)
        if source.degree == 0:
            return zero_matrix(1, 1)
        return _operator_matrix(source, target, _derivative)

    if kind is PrimitiveKind.ANTIDERIVATIVE:
        source = _polynomial_source(map_type)
        target = _polynomial_target(map_type)
        basepoint = primitive.parameters[0]
        return _operator_matrix(
            source,
            target,
            lambda polynomial: _antiderivative(polynomial, basepoint),
        )

    if kind is PrimitiveKind.EVALUATION:
        source = _polynomial_source(map_type)
        point = primitive.parameters[0]
        return _row(_evaluate(polynomial, point) for polynomial in source.basis)

    if kind is PrimitiveKind.INTERVAL_INTEGRATION:
        source = _polynomial_source(map_type)
        lower, upper, density = primitive.parameters
        return _row(
            density
            * sum(
                (
                    coefficient
                    * (upper ** (power + 1) - lower ** (power + 1))
                    / Fraction(power + 1)
                    for power, coefficient in enumerate(polynomial)
                ),
                Fraction(0),
            )
            for polynomial in source.basis
        )

    if kind is PrimitiveKind.AFFINE_SUBSTITUTION:
        source = _polynomial_source(map_type)
        scale, shift = primitive.parameters
        return _operator_matrix(
            source,
            source,
            lambda polynomial: _compose_affine(polynomial, scale, shift),
        )

    if kind is PrimitiveKind.FINITE_DIFFERENCE:
        source = _polynomial_source(map_type)
        step = primitive.parameters[0]
        return _operator_matrix(
            source,
            source,
            lambda polynomial: _subtract(
                _compose_affine(polynomial, Fraction(1), step),
                polynomial,
            ),
        )

    if kind is PrimitiveKind.BOUNDARY_FUNCTIONAL:
        source = _polynomial_source(map_type)
        lower, upper, density = primitive.parameters
        return _row(
            density * (_evaluate(polynomial, upper) - _evaluate(polynomial, lower))
            for polynomial in source.basis
        )

    if kind is PrimitiveKind.CONSTANT_INJECTION:
        target = _polynomial_target(map_type)
        monomial_constant = (Fraction(1),) + tuple(
            Fraction(0) for _ in range(target.degree)
        )
        coordinates = matrix_vector_multiply(
            inverse_matrix(_basis_matrix(target)),
            monomial_constant,
        )
        return matrix_from_columns((coordinates,))

    raise ValidationError(f"unsupported polynomial primitive kind: {kind}")


replay_matrix = replay


def primitive_matrix_equal(
    primitive: PolynomialPrimitive,
    matrix: Iterable[Iterable[Any]],
) -> bool:
    """Return exact equality after validating the claimed rational matrix."""

    claimed = _normalize_matrix(matrix)
    expected_shape = (
        primitive.map_type.target.dimension,
        primitive.map_type.source.dimension,
    )
    if matrix_shape(claimed) != expected_shape:
        return False
    return claimed == replay(primitive)


def primitive_certificate(
    primitive: PolynomialPrimitive,
    *,
    claimed_matrix: Iterable[Iterable[Any]] | None = None,
) -> Certificate:
    """Certify structural replay and, when supplied, exact matrix equality."""

    if not isinstance(primitive, PolynomialPrimitive):
        raise TypeError("primitive_certificate requires a PolynomialPrimitive")
    matrix = replay(primitive)
    valid = True
    checked = [
        "snapshotted exact graded basis witnesses",
        "exact rational primitive parameters",
        "typed source and target dimensions",
        "matrix replay independent of display names",
    ]
    if claimed_matrix is not None:
        valid = primitive_matrix_equal(primitive, claimed_matrix)
        checked.append("claimed matrix compared entrywise with exact replay")
    rows, columns = matrix_shape(matrix)
    return Certificate(
        valid=valid,
        exactness=Exactness.EXACT,
        residuals={"matrix_replay": 0.0 if valid else 1.0},
        tolerance=0.0,
        assumptions_checked=tuple(checked),
        diagnostics={
            "kind": primitive.kind.value,
            "shape": (rows, columns),
            "parameter_count": len(primitive.parameters),
            "claimed_matrix_checked": claimed_matrix is not None,
        },
        theorem_id=primitive.theorem_id,
        message=(
            "polynomial primitive replayed exactly"
            if valid
            else "claimed matrix does not equal the replayed polynomial primitive"
        ),
    )


__all__ = [
    "PolynomialPrimitive",
    "PrimitiveKind",
    "affine_substitution",
    "antiderivative",
    "boundary_functional",
    "constant_injection",
    "coordinate_change",
    "derivative",
    "evaluation",
    "finite_difference",
    "interval_integration",
    "primitive_certificate",
    "primitive_matrix_equal",
    "replay",
    "replay_matrix",
]
