"""Typed immutable expressions for finite exact rational computations.

The model deliberately separates semantic spaces from display labels.  In
particular, a polynomial-coordinate space is identified by its complete exact
basis witness, not by a user supplied family name.  This makes type checking a
mathematical check rather than a string-matching convention.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, TypeAlias
from unicodedata import normalize

from ..coordinates.exact import Matrix, Vector
from ..exceptions import DimensionError, ResourceLimitError, ValidationError


def _nfc_text(value: Any, *, name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a string")
    result = normalize("NFC", value)
    if not allow_empty and not result.strip():
        raise ValidationError(f"{name} must be nonempty")
    return result if allow_empty else result.strip()


def _index(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise ValidationError(f"{name} must be a {qualifier} integer")
    return value


def _fraction(value: Any, *, name: str = "coefficient") -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise ValidationError(f"{name} must be an integer or Fraction")
    return Fraction(value)


def _vector(values: Iterable[Any], *, name: str) -> Vector:
    try:
        data = tuple(values)
    except TypeError as exc:
        raise ValidationError(f"{name} must be a finite iterable") from exc
    return tuple(_fraction(value, name=name) for value in data)


def _matrix(rows: Iterable[Iterable[Any]], *, name: str) -> Matrix:
    try:
        source = tuple(rows)
    except TypeError as exc:
        raise ValidationError(f"{name} must be a finite row iterable") from exc
    data: list[tuple[Fraction, ...]] = []
    for row in source:
        try:
            values = tuple(row)
        except TypeError as exc:
            raise ValidationError(f"{name} rows must be finite iterables") from exc
        data.append(tuple(_fraction(value, name=name) for value in values))
    if data and any(len(row) != len(data[0]) for row in data):
        raise DimensionError(f"{name} rows must have a common length")
    return tuple(data)


@dataclass(frozen=True, slots=True)
class RationalSpace:
    """The one-dimensional scalar space over the rational numbers.

    The class has no instance state, so every instance has the same semantic
    identity and hash.  ``RATIONAL_SPACE`` is supplied as a convenient shared
    instance, but callers do not need to preserve object identity.
    """

    @property
    def dimension(self) -> int:
        return 1

    @property
    def semantic_key(self) -> tuple[str]:
        return ("Q",)


RATIONAL_SPACE = RationalSpace()


@dataclass(frozen=True, slots=True)
class PolynomialCoordinateSpace:
    """Coordinates of univariate polynomials through one finite degree.

    ``basis`` contains one ascending-coefficient polynomial of each degree
    from zero through ``degree``.  The display ``label`` is intentionally
    excluded from equality.  The variable and complete rational witness are
    semantic data.
    """

    variable: str
    degree: int
    basis: tuple[tuple[Fraction, ...], ...]
    label: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        variable = _nfc_text(self.variable, name="polynomial variable")
        degree = _index(self.degree, name="polynomial degree")
        try:
            raw_basis = tuple(self.basis)
        except TypeError as exc:
            raise ValidationError("polynomial basis witness must be finite") from exc
        if len(raw_basis) != degree + 1:
            raise DimensionError("polynomial basis witness length must equal degree plus one")
        normalized_basis: list[tuple[Fraction, ...]] = []
        for basis_degree, polynomial in enumerate(raw_basis):
            coefficients = _vector(polynomial, name="polynomial basis coefficient")
            if len(coefficients) != basis_degree + 1:
                raise DimensionError(
                    "basis witness must contain one exact polynomial of every graded degree"
                )
            if coefficients[-1] == 0:
                raise ValidationError("every basis polynomial must have nonzero leading coefficient")
            normalized_basis.append(coefficients)
        label = _nfc_text(self.label, name="polynomial-space label", allow_empty=True)
        object.__setattr__(self, "variable", variable)
        object.__setattr__(self, "degree", degree)
        object.__setattr__(self, "basis", tuple(normalized_basis))
        object.__setattr__(self, "label", label)

    @classmethod
    def from_family(
        cls,
        family: Any,
        degree: int,
        *,
        variable: str = "x",
        label: str | None = None,
    ) -> PolynomialCoordinateSpace:
        """Snapshot a graded polynomial family through ``degree`` exactly."""

        bound = _index(degree, name="polynomial degree")
        try:
            witness = tuple(family.polynomial(index) for index in range(bound + 1))
        except AttributeError as exc:
            raise TypeError("family must provide polynomial(degree)") from exc
        family_name = getattr(family, "name", "polynomial coordinates")
        display = str(family_name) if label is None else label
        return cls(variable, bound, witness, display)

    @property
    def dimension(self) -> int:
        return self.degree + 1

    @property
    def semantic_key(
        self,
    ) -> tuple[str, str, int, tuple[tuple[Fraction, ...], ...]]:
        return ("polynomial", self.variable, self.degree, self.basis)

    def truncate(self, degree: int) -> PolynomialCoordinateSpace:
        """Return the compatible prefix space through a lower degree."""

        bound = _index(degree, name="truncation degree")
        if bound > self.degree:
            raise DimensionError("a polynomial space cannot be truncated to a larger degree")
        return PolynomialCoordinateSpace(
            self.variable,
            bound,
            self.basis[: bound + 1],
            self.label,
        )


CoordinateSpace: TypeAlias = RationalSpace | PolynomialCoordinateSpace


def _space(value: Any, *, name: str) -> CoordinateSpace:
    if not isinstance(value, (RationalSpace, PolynomialCoordinateSpace)):
        raise TypeError(f"{name} must be a supported exact coordinate space")
    return value


@dataclass(frozen=True, slots=True)
class LinearMapType:
    """A typed hom-space from ``source`` to ``target``."""

    source: CoordinateSpace
    target: CoordinateSpace

    def __post_init__(self) -> None:
        _space(self.source, name="map source")
        _space(self.target, name="map target")

    @property
    def semantic_key(self) -> tuple[str, Any, Any]:
        return ("linear_map", self.source.semantic_key, self.target.semantic_key)


ExpressionType: TypeAlias = CoordinateSpace | LinearMapType


def _expression_type(value: Any, *, name: str = "expression") -> ExpressionType:
    if not isinstance(value, Expression):
        raise TypeError(f"{name} must be an Expression")
    return value.expression_type


class Expression:
    """Base class for immutable typed expression nodes."""

    __slots__ = ()

    @property
    def expression_type(self) -> ExpressionType:
        raise NotImplementedError

    @property
    def children(self) -> tuple[Expression, ...]:
        return ()


@dataclass(frozen=True, slots=True)
class ExactValueLiteral(Expression):
    space: CoordinateSpace
    entries: Vector

    def __post_init__(self) -> None:
        space = _space(self.space, name="literal space")
        entries = _vector(self.entries, name="literal entry")
        if len(entries) != space.dimension:
            raise DimensionError("literal entry count does not match its space")
        object.__setattr__(self, "entries", entries)

    @property
    def expression_type(self) -> CoordinateSpace:
        return self.space


@dataclass(frozen=True, slots=True)
class ExactMapLiteral(Expression):
    map_type: LinearMapType
    matrix: Matrix
    label: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.map_type, LinearMapType):
            raise TypeError("map_type must be a LinearMapType")
        matrix = _matrix(self.matrix, name="map matrix entry")
        expected = (self.map_type.target.dimension, self.map_type.source.dimension)
        shape = (len(matrix), len(matrix[0]) if matrix else 0)
        if shape != expected:
            raise DimensionError(f"map matrix shape must be {expected}")
        label = _nfc_text(self.label, name="map label", allow_empty=True)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "label", label)

    @property
    def expression_type(self) -> LinearMapType:
        return self.map_type


@dataclass(frozen=True, slots=True, eq=False)
class Symbol(Expression):
    name: str
    declared_type: ExpressionType

    def __post_init__(self) -> None:
        name = _nfc_text(self.name, name="symbol name")
        if not isinstance(
            self.declared_type,
            (RationalSpace, PolynomialCoordinateSpace, LinearMapType),
        ):
            raise TypeError("declared_type must be an expression type")
        object.__setattr__(self, "name", name)

    @property
    def expression_type(self) -> ExpressionType:
        return self.declared_type


@dataclass(frozen=True, slots=True)
class Let(Expression):
    symbol: Symbol
    value: Expression
    body: Expression

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, Symbol):
            raise TypeError("let bindings require a Symbol")
        value_type = _expression_type(self.value, name="let value")
        _expression_type(self.body, name="let body")
        if value_type != self.symbol.declared_type:
            raise ValidationError("let value type does not match its declared symbol type")

    @property
    def expression_type(self) -> ExpressionType:
        return self.body.expression_type

    @property
    def children(self) -> tuple[Expression, ...]:
        return (self.symbol, self.value, self.body)


@dataclass(frozen=True, slots=True)
class IdentityMap(Expression):
    space: CoordinateSpace

    def __post_init__(self) -> None:
        _space(self.space, name="identity space")

    @property
    def expression_type(self) -> LinearMapType:
        return LinearMapType(self.space, self.space)


@dataclass(frozen=True, slots=True)
class ZeroMap(Expression):
    map_type: LinearMapType

    def __post_init__(self) -> None:
        if not isinstance(self.map_type, LinearMapType):
            raise TypeError("zero-map type must be a LinearMapType")

    @property
    def expression_type(self) -> LinearMapType:
        return self.map_type


@dataclass(frozen=True, slots=True)
class Compose(Expression):
    """Map composition with factors stored outermost to innermost."""

    factors: tuple[Expression, ...]

    def __post_init__(self) -> None:
        try:
            factors = tuple(self.factors)
        except TypeError as exc:
            raise ValidationError("composition factors must be finite") from exc
        if len(factors) < 2:
            raise ValidationError("a Compose node requires at least two factors")
        types = tuple(_expression_type(factor, name="composition factor") for factor in factors)
        if any(not isinstance(map_type, LinearMapType) for map_type in types):
            raise ValidationError("every composition factor must be a linear map")
        map_types = tuple(map_type for map_type in types if isinstance(map_type, LinearMapType))
        for outer, inner in zip(map_types, map_types[1:], strict=False):
            if inner.target != outer.source:
                raise ValidationError("adjacent composition endpoints do not agree")
        object.__setattr__(self, "factors", factors)

    @property
    def expression_type(self) -> LinearMapType:
        outer = self.factors[0].expression_type
        inner = self.factors[-1].expression_type
        assert isinstance(outer, LinearMapType) and isinstance(inner, LinearMapType)
        return LinearMapType(inner.source, outer.target)

    @property
    def children(self) -> tuple[Expression, ...]:
        return self.factors


@dataclass(frozen=True, slots=True)
class Sum(Expression):
    terms: tuple[Expression, ...]

    def __post_init__(self) -> None:
        try:
            terms = tuple(self.terms)
        except TypeError as exc:
            raise ValidationError("sum terms must be finite") from exc
        if len(terms) < 2:
            raise ValidationError("a Sum node requires at least two terms")
        first_type = _expression_type(terms[0], name="sum term")
        if any(_expression_type(term, name="sum term") != first_type for term in terms[1:]):
            raise ValidationError("all sum terms must have the same expression type")
        object.__setattr__(self, "terms", terms)

    @property
    def expression_type(self) -> ExpressionType:
        return self.terms[0].expression_type

    @property
    def children(self) -> tuple[Expression, ...]:
        return self.terms


@dataclass(frozen=True, slots=True)
class Scale(Expression):
    scalar: Fraction
    value: Expression

    def __post_init__(self) -> None:
        scalar = _fraction(self.scalar, name="scale")
        _expression_type(self.value, name="scaled value")
        object.__setattr__(self, "scalar", scalar)

    @property
    def expression_type(self) -> ExpressionType:
        return self.value.expression_type

    @property
    def children(self) -> tuple[Expression, ...]:
        return (self.value,)


@dataclass(frozen=True, slots=True)
class Apply(Expression):
    operator: Expression
    argument: Expression

    def __post_init__(self) -> None:
        operator_type = _expression_type(self.operator, name="applied operator")
        argument_type = _expression_type(self.argument, name="application argument")
        if not isinstance(operator_type, LinearMapType):
            raise ValidationError("the applied operator must have a linear-map type")
        if argument_type != operator_type.source:
            raise ValidationError("application argument does not belong to the operator source")

    @property
    def expression_type(self) -> CoordinateSpace:
        operator_type = self.operator.expression_type
        assert isinstance(operator_type, LinearMapType)
        return operator_type.target

    @property
    def children(self) -> tuple[Expression, ...]:
        return (self.operator, self.argument)


def exact_value(space: CoordinateSpace, entries: Iterable[Any]) -> ExactValueLiteral:
    return ExactValueLiteral(space, tuple(entries))


def exact_map(
    map_type: LinearMapType,
    matrix: Iterable[Iterable[Any]],
    *,
    label: str = "",
) -> ExactMapLiteral:
    return ExactMapLiteral(map_type, tuple(tuple(row) for row in matrix), label)


def symbol(name: str, declared_type: ExpressionType) -> Symbol:
    return Symbol(name, declared_type)


def let(binding: Symbol, value: Expression, body: Expression) -> Let:
    return Let(binding, value, body)


def identity(space: CoordinateSpace) -> IdentityMap:
    return IdentityMap(space)


def zero_map(map_type: LinearMapType) -> ZeroMap:
    return ZeroMap(map_type)


def compose(*maps: Expression) -> Expression:
    """Compose maps supplied in outermost-to-innermost order."""

    if not maps:
        raise ValidationError("compose requires at least one map")
    if len(maps) == 1:
        map_type = _expression_type(maps[0], name="composition factor")
        if not isinstance(map_type, LinearMapType):
            raise ValidationError("composition factors must be linear maps")
        return maps[0]
    return Compose(tuple(maps))


def then(*maps: Expression) -> Expression:
    """Compose maps supplied in the order in which they act."""

    if not maps:
        raise ValidationError("then requires at least one map")
    return compose(*reversed(maps))


def add(*terms: Expression) -> Expression:
    if not terms:
        raise ValidationError("add requires at least one term")
    if len(terms) == 1:
        _expression_type(terms[0], name="sum term")
        return terms[0]
    return Sum(tuple(terms))


def scale(scalar: int | Fraction, value: Expression) -> Scale:
    return Scale(_fraction(scalar, name="scale coefficient"), value)


def apply(operator: Expression, argument: Expression) -> Apply:
    return Apply(operator, argument)


@dataclass(frozen=True, slots=True)
class ExpressionStats:
    """Finite structural totals computed without recursive traversal."""

    nodes: int
    depth: int
    steps: int
    exact_entries: int
    maximum_degree: int
    maximum_coefficient_bits: int
    serialized_bytes: int


def _spaces(expression_type: ExpressionType) -> tuple[CoordinateSpace, ...]:
    if isinstance(expression_type, LinearMapType):
        return (expression_type.source, expression_type.target)
    return (expression_type,)


def _fraction_bits(value: Fraction) -> int:
    return max(abs(value.numerator).bit_length(), value.denominator.bit_length())


def _fraction_bytes(value: Fraction) -> int:
    return len(f"{value.numerator}/{value.denominator}".encode("ascii"))


def _node_coefficients(node: Expression) -> tuple[Fraction, ...]:
    if isinstance(node, ExactValueLiteral):
        return node.entries
    if isinstance(node, ExactMapLiteral):
        return tuple(value for row in node.matrix for value in row)
    if isinstance(node, Scale):
        return (node.scalar,)
    parameters = getattr(node, "parameters", ())
    if isinstance(parameters, tuple) and all(isinstance(value, Fraction) for value in parameters):
        return parameters
    return ()


def _node_text(node: Expression) -> tuple[str, ...]:
    values = [type(node).__name__]
    if isinstance(node, Symbol):
        values.append(node.name)
    if isinstance(node, ExactMapLiteral):
        values.append(node.label)
    kind = getattr(node, "kind", None)
    if kind is not None:
        values.append(str(kind))
    return tuple(values)


@dataclass(frozen=True, slots=True)
class ExpressionBudget:
    """Pre-allocation limits for a finite expression DAG.

    ``serialized_bytes`` is a deterministic structural accounting of node tags,
    normalized text, semantic space witnesses, and rational tokens.  It is not
    a pickle-size estimate.
    """

    max_nodes: int = 10_000
    max_depth: int = 1_000
    max_steps: int = 50_000
    max_degree: int = 512
    max_entries: int = 1_000_000
    max_coefficient_bits: int = 1_000_000
    max_serialized_bytes: int = 64_000_000
    max_symbol_length: int = 1_024

    def __post_init__(self) -> None:
        for name, value in (
            ("max_nodes", self.max_nodes),
            ("max_depth", self.max_depth),
            ("max_steps", self.max_steps),
            ("max_degree", self.max_degree),
            ("max_entries", self.max_entries),
            ("max_coefficient_bits", self.max_coefficient_bits),
            ("max_serialized_bytes", self.max_serialized_bytes),
            ("max_symbol_length", self.max_symbol_length),
        ):
            _index(value, name=name, minimum=1)

    def preflight(self, root: Expression) -> ExpressionStats:
        """Validate a DAG, reject cycles, and return bounded structural totals."""

        if not isinstance(root, Expression):
            raise TypeError("expression root must be an Expression")

        color: dict[int, int] = {}
        postorder: list[Expression] = []
        edge_count = 0
        stack: list[tuple[Expression, bool, int]] = [(root, False, 1)]
        while stack:
            node, exiting, path_depth = stack.pop()
            if not isinstance(node, Expression):
                raise TypeError("expression children must be Expression nodes")
            identifier = id(node)
            state = color.get(identifier, 0)
            if exiting:
                if state == 1:
                    color[identifier] = 2
                    postorder.append(node)
                continue
            if state == 1:
                raise ValidationError("expression graph contains a cycle")
            if path_depth > self.max_depth:
                raise ResourceLimitError("expression depth exceeds max_depth")
            if state == 2:
                continue
            if len(color) >= self.max_nodes:
                raise ResourceLimitError("expression node count exceeds max_nodes")
            color[identifier] = 1
            children = node.children
            if any(not isinstance(child, Expression) for child in children):
                raise TypeError("expression children must be Expression nodes")
            edge_count += len(children)
            if len(color) + edge_count > self.max_steps:
                raise ResourceLimitError("expression graph traversal exceeds max_steps")
            stack.append((node, True, path_depth))
            for child in reversed(children):
                if color.get(id(child), 0) == 1:
                    raise ValidationError("expression graph contains a cycle")
                stack.append((child, False, path_depth + 1))

        node_count = len(postorder)
        steps = node_count + edge_count
        if node_count > self.max_nodes:
            raise ResourceLimitError("expression node count exceeds max_nodes")
        if steps > self.max_steps:
            raise ResourceLimitError("expression graph traversal exceeds max_steps")

        depths: dict[int, int] = {}
        for node in postorder:
            child_depth = max((depths[id(child)] for child in node.children), default=0)
            depths[id(node)] = child_depth + 1
        depth = depths[id(root)]
        if depth > self.max_depth:
            raise ResourceLimitError("expression depth exceeds max_depth")

        unique_spaces: dict[Any, CoordinateSpace] = {}
        exact_entries = 0
        maximum_degree = 0
        maximum_bits = 0
        serialized_bytes = 0
        for node in postorder:
            for text in _node_text(node):
                serialized_bytes += len(text.encode("utf-8")) + 1
            coefficients = _node_coefficients(node)
            exact_entries += len(coefficients)
            for coefficient in coefficients:
                maximum_bits = max(maximum_bits, _fraction_bits(coefficient))
                serialized_bytes += _fraction_bytes(coefficient) + 1
            if isinstance(node, Symbol) and len(node.name) > self.max_symbol_length:
                raise ResourceLimitError("symbol name exceeds max_symbol_length")
            for space in _spaces(node.expression_type):
                unique_spaces.setdefault(space.semantic_key, space)

        for space in unique_spaces.values():
            serialized_bytes += len(repr(space.semantic_key).encode("utf-8"))
            if isinstance(space, PolynomialCoordinateSpace):
                if len(space.variable) > self.max_symbol_length:
                    raise ResourceLimitError("polynomial variable exceeds max_symbol_length")
                maximum_degree = max(maximum_degree, space.degree)
                basis_coefficients = tuple(value for polynomial in space.basis for value in polynomial)
                exact_entries += len(basis_coefficients)
                for coefficient in basis_coefficients:
                    maximum_bits = max(maximum_bits, _fraction_bits(coefficient))

        if maximum_degree > self.max_degree:
            raise ResourceLimitError("expression polynomial degree exceeds max_degree")
        if exact_entries > self.max_entries:
            raise ResourceLimitError("expression exact-entry count exceeds max_entries")
        if maximum_bits > self.max_coefficient_bits:
            raise ResourceLimitError("expression coefficient size exceeds max_coefficient_bits")
        if serialized_bytes > self.max_serialized_bytes:
            raise ResourceLimitError("expression structural size exceeds max_serialized_bytes")
        return ExpressionStats(
            nodes=node_count,
            depth=depth,
            steps=steps,
            exact_entries=exact_entries,
            maximum_degree=maximum_degree,
            maximum_coefficient_bits=maximum_bits,
            serialized_bytes=serialized_bytes,
        )

    require = preflight


__all__ = [
    "RATIONAL_SPACE",
    "Apply",
    "Compose",
    "CoordinateSpace",
    "ExactMapLiteral",
    "ExactValueLiteral",
    "Expression",
    "ExpressionBudget",
    "ExpressionStats",
    "ExpressionType",
    "IdentityMap",
    "Let",
    "LinearMapType",
    "PolynomialCoordinateSpace",
    "RationalSpace",
    "Scale",
    "Sum",
    "Symbol",
    "ZeroMap",
    "add",
    "apply",
    "compose",
    "exact_map",
    "exact_value",
    "identity",
    "let",
    "scale",
    "symbol",
    "then",
    "zero_map",
]
