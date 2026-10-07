"""Strict portable JSON for finite typed expressions.

The transport form retains display labels, while the semantic digest omits
them.  Bound symbols are hashed by lexical distance rather than by their
display names, so alpha-equivalent ``Let`` expressions have one digest.  No
class names, import paths, callbacks, or executable source are deserialized.
"""

from __future__ import annotations

import hmac
import json
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from math import gcd
from typing import Any
from unicodedata import normalize

from ..exceptions import ResourceLimitError, ValidationError
from .model import (
    RATIONAL_SPACE,
    Apply,
    Compose,
    ExactMapLiteral,
    ExactValueLiteral,
    Expression,
    ExpressionBudget,
    ExpressionType,
    IdentityMap,
    Let,
    LinearMapType,
    PolynomialCoordinateSpace,
    RationalSpace,
    Scale,
    Sum,
    Symbol,
    ZeroMap,
)
from .polynomial import PolynomialPrimitive, PrimitiveKind

EXPRESSION_SCHEMA = "komorebi_math.expression.v1"
EXPRESSION_SCHEMA_VERSION = 1
EXPRESSION_NODE_VERSION = 1
EXPRESSION_TYPE_VERSION = 1

_HEX_DIGITS = frozenset("0123456789abcdef")


def _budget(value: ExpressionBudget | None) -> ExpressionBudget:
    if value is None:
        return ExpressionBudget()
    if type(value) is not ExpressionBudget:
        raise TypeError("budget must be an ExpressionBudget")
    return value


def _preflight(expression: Expression, budget: ExpressionBudget) -> None:
    from .engine import check_expression

    try:
        check_expression(expression, budget=budget)
    except UnicodeEncodeError as exc:
        raise ValidationError("expression text must be valid UTF-8") from exc


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValidationError("expression payload is not canonical finite JSON") from exc


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"expression JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_float(_: str) -> Any:
    raise ValidationError("expression JSON rejects floating-point tokens")


def _reject_constant(value: str) -> Any:
    raise ValidationError(f"expression JSON rejects nonfinite token {value}")


def _strict_integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise ValidationError(f"{name} must be a {qualifier} integer")
    return value


def _strict_text(value: Any, *, name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValidationError(f"{name} must be valid UTF-8") from exc
    if normalize("NFC", value) != value:
        raise ValidationError(f"{name} must use Unicode NFC")
    if not allow_empty:
        if not value.strip():
            raise ValidationError(f"{name} must be nonempty")
        if value != value.strip():
            raise ValidationError(f"{name} must not contain surrounding whitespace")
    return value


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{name} must be an object")
    for key in value:
        _strict_text(key, name=f"{name} key")
    return value


def _array(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValidationError(f"{name} must be an array")
    return value


def _require_fields(value: Mapping[str, Any], fields: set[str], *, name: str) -> None:
    observed = set(value)
    if observed != fields:
        missing = tuple(sorted(fields - observed))
        extra = tuple(sorted(observed - fields))
        raise ValidationError(
            f"{name} fields do not match the schema; missing={missing}, extra={extra}"
        )


def _validate_json_tree(value: Any, *, depth: int, budget: ExpressionBudget) -> None:
    active: set[int] = set()
    stack: list[tuple[Any, int, bool]] = [(value, depth, False)]
    while stack:
        item, item_depth, exiting = stack.pop()
        if item_depth > budget.max_depth + 8:
            raise ResourceLimitError("expression JSON depth exceeds max_depth")
        if exiting:
            active.remove(id(item))
            continue
        if isinstance(item, str):
            _strict_text(item, name="JSON string", allow_empty=True)
            continue
        if isinstance(item, bool) or item is None or isinstance(item, float):
            raise ValidationError("expression JSON contains a nonexact primitive")
        if isinstance(item, int):
            continue
        if isinstance(item, (list, Mapping)):
            identifier = id(item)
            if identifier in active:
                raise ValidationError("expression JSON contains a circular container")
            active.add(identifier)
            stack.append((item, item_depth, True))
            if isinstance(item, list):
                stack.extend(
                    (child, item_depth + 1, False) for child in reversed(item)
                )
            else:
                children: list[tuple[Any, int, bool]] = []
                for key, child in item.items():
                    _strict_text(key, name="JSON object key")
                    children.append((child, item_depth + 1, False))
                stack.extend(reversed(children))
            continue
        raise ValidationError("expression JSON contains an unsupported value")


def _check_encoded_size(value: Any, budget: ExpressionBudget) -> str:
    text = _canonical_json(value)
    if len(text.encode("utf-8")) > budget.max_serialized_bytes:
        raise ResourceLimitError("expression JSON exceeds max_serialized_bytes")
    return text


def _fraction_to_dict(value: Fraction) -> dict[str, int]:
    if not isinstance(value, Fraction):
        raise TypeError("exact expression coefficients must be Fractions")
    return {"denominator": value.denominator, "numerator": value.numerator}


@dataclass(slots=True)
class _DecodeState:
    budget: ExpressionBudget
    nodes: int = 0
    edges: int = 0
    exact_entries: int = 0

    def node(self, *, depth: int) -> None:
        if depth > self.budget.max_depth:
            raise ResourceLimitError("expression depth exceeds max_depth")
        self.nodes += 1
        if self.nodes > self.budget.max_nodes:
            raise ResourceLimitError("expression node count exceeds max_nodes")
        if self.nodes + self.edges > self.budget.max_steps:
            raise ResourceLimitError("expression token construction exceeds max_steps")

    def edge(self, count: int) -> None:
        self.edges += count
        if self.nodes + self.edges > self.budget.max_steps:
            raise ResourceLimitError("expression token construction exceeds max_steps")

    def fraction(self, value: Any, *, name: str) -> Fraction:
        payload = _mapping(value, name=name)
        _require_fields(payload, {"numerator", "denominator"}, name=name)
        numerator = payload["numerator"]
        denominator = payload["denominator"]
        if isinstance(numerator, bool) or not isinstance(numerator, int):
            raise ValidationError(f"{name} numerator must be an integer")
        if isinstance(denominator, bool) or not isinstance(denominator, int):
            raise ValidationError(f"{name} denominator must be an integer")
        if denominator <= 0:
            raise ValidationError(f"{name} denominator must be positive")
        if gcd(abs(numerator), denominator) != 1:
            raise ValidationError(f"{name} must be a canonical reduced fraction")
        bits = max(abs(numerator).bit_length(), denominator.bit_length())
        if bits > self.budget.max_coefficient_bits:
            raise ResourceLimitError("expression coefficient size exceeds max_coefficient_bits")
        self.exact_entries += 1
        if self.exact_entries > self.budget.max_entries:
            raise ResourceLimitError("expression exact-entry count exceeds max_entries")
        return Fraction(numerator, denominator)


def _type_to_payload(value: ExpressionType, *, semantic: bool) -> dict[str, Any]:
    if type(value) is RationalSpace:
        return {
            "kind": "rational_space",
            "version": EXPRESSION_TYPE_VERSION,
        }
    if type(value) is PolynomialCoordinateSpace:
        payload: dict[str, Any] = {
            "basis": [
                [_fraction_to_dict(coefficient) for coefficient in polynomial]
                for polynomial in value.basis
            ],
            "degree": value.degree,
            "kind": "polynomial_coordinate_space",
            "variable": value.variable,
            "version": EXPRESSION_TYPE_VERSION,
        }
        if not semantic:
            payload["label"] = value.label
        return payload
    if type(value) is LinearMapType:
        return {
            "kind": "linear_map",
            "source": _type_to_payload(value.source, semantic=semantic),
            "target": _type_to_payload(value.target, semantic=semantic),
            "version": EXPRESSION_TYPE_VERSION,
        }
    raise TypeError("value must be a supported expression type")


def _type_from_payload(
    value: Any,
    *,
    state: _DecodeState,
    depth: int,
) -> ExpressionType:
    if depth > state.budget.max_depth + 4:
        raise ResourceLimitError("expression type depth exceeds max_depth")
    payload = _mapping(value, name="expression type")
    kind = _strict_text(payload.get("kind"), name="expression type kind")
    version = _strict_integer(
        payload.get("version"),
        name="expression type version",
        minimum=1,
    )
    if version != EXPRESSION_TYPE_VERSION:
        raise ValidationError(f"unknown expression type version: {version}")

    if kind == "rational_space":
        _require_fields(payload, {"kind", "version"}, name="rational-space type")
        return RATIONAL_SPACE

    if kind == "polynomial_coordinate_space":
        _require_fields(
            payload,
            {"basis", "degree", "kind", "label", "variable", "version"},
            name="polynomial-coordinate-space type",
        )
        variable = _strict_text(payload["variable"], name="polynomial variable")
        label = _strict_text(
            payload["label"],
            name="polynomial-space label",
            allow_empty=True,
        )
        if len(variable) > state.budget.max_symbol_length:
            raise ResourceLimitError("polynomial variable exceeds max_symbol_length")
        if len(label) > state.budget.max_symbol_length:
            raise ResourceLimitError("polynomial label exceeds max_symbol_length")
        degree = _strict_integer(payload["degree"], name="polynomial degree")
        if degree > state.budget.max_degree:
            raise ResourceLimitError("expression polynomial degree exceeds max_degree")
        raw_basis = _array(payload["basis"], name="polynomial basis")
        if len(raw_basis) != degree + 1:
            raise ValidationError("polynomial basis length must equal degree plus one")
        basis: list[tuple[Fraction, ...]] = []
        for basis_degree, raw_polynomial in enumerate(raw_basis):
            coefficients = _array(
                raw_polynomial,
                name="polynomial basis member",
            )
            if len(coefficients) != basis_degree + 1:
                raise ValidationError(
                    "polynomial basis witness must contain every complete graded member"
                )
            basis.append(
                tuple(
                    state.fraction(
                        coefficient,
                        name="polynomial basis coefficient",
                    )
                    for coefficient in coefficients
                )
            )
        return PolynomialCoordinateSpace(variable, degree, tuple(basis), label)

    if kind == "linear_map":
        _require_fields(
            payload,
            {"kind", "source", "target", "version"},
            name="linear-map type",
        )
        source = _type_from_payload(payload["source"], state=state, depth=depth + 1)
        target = _type_from_payload(payload["target"], state=state, depth=depth + 1)
        if not isinstance(source, (RationalSpace, PolynomialCoordinateSpace)):
            raise ValidationError("linear-map source must be a coordinate space")
        if not isinstance(target, (RationalSpace, PolynomialCoordinateSpace)):
            raise ValidationError("linear-map target must be a coordinate space")
        return LinearMapType(source, target)

    raise ValidationError(f"unknown expression type kind: {kind!r}")


def type_to_dict(
    expression_type: ExpressionType,
    *,
    budget: ExpressionBudget | None = None,
) -> dict[str, Any]:
    """Return the strict transport record for one expression type."""

    limit = _budget(budget)
    payload = _type_to_payload(expression_type, semantic=False)
    _validate_json_tree(payload, depth=0, budget=limit)
    _check_encoded_size(payload, limit)
    state = _DecodeState(limit)
    _type_from_payload(payload, state=state, depth=0)
    return payload


def type_from_dict(
    value: Mapping[str, Any],
    *,
    budget: ExpressionBudget | None = None,
) -> ExpressionType:
    """Reconstruct one exact expression type from a strict transport record."""

    limit = _budget(budget)
    _validate_json_tree(value, depth=0, budget=limit)
    _check_encoded_size(value, limit)
    state = _DecodeState(limit)
    return _type_from_payload(value, state=state, depth=0)


def _bound_index(symbol: Symbol, environment: tuple[Symbol, ...]) -> int | None:
    for index, binder in enumerate(reversed(environment)):
        if symbol is binder:
            if symbol.declared_type != binder.declared_type:
                raise ValidationError("bound symbol type does not match its binder")
            return index
    return None


def _symbol_declaration(symbol: Symbol) -> dict[str, Any]:
    if type(symbol) is not Symbol:
        raise TypeError("let binder must be a Symbol")
    return {
        "declared_type": _type_to_payload(symbol.declared_type, semantic=False),
        "kind": "symbol",
        "name": symbol.name,
        "version": EXPRESSION_NODE_VERSION,
    }


def _expression_to_payload(
    expression: Expression,
    *,
    environment: tuple[Symbol, ...],
    semantic: bool,
    depth: int,
    budget: ExpressionBudget,
    occurrences: list[int],
) -> dict[str, Any]:
    if depth > budget.max_depth:
        raise ResourceLimitError("expression serialization depth exceeds max_depth")
    occurrences[0] += 1
    if occurrences[0] > budget.max_steps:
        raise ResourceLimitError("expression serialization exceeds max_steps")

    def child(value: Expression, *, scope: tuple[Symbol, ...] = environment) -> dict[str, Any]:
        return _expression_to_payload(
            value,
            environment=scope,
            semantic=semantic,
            depth=depth + 1,
            budget=budget,
            occurrences=occurrences,
        )

    if type(expression) is Symbol:
        index = _bound_index(expression, environment)
        if index is not None:
            return {
                "index": index,
                "kind": "bound_symbol",
                "version": EXPRESSION_NODE_VERSION,
            }
        return {
            "declared_type": _type_to_payload(
                expression.declared_type,
                semantic=semantic,
            ),
            "kind": "symbol",
            "name": expression.name,
            "version": EXPRESSION_NODE_VERSION,
        }

    if type(expression) is ExactValueLiteral:
        return {
            "entries": [_fraction_to_dict(value) for value in expression.entries],
            "kind": "exact_value",
            "space": _type_to_payload(expression.space, semantic=semantic),
            "version": EXPRESSION_NODE_VERSION,
        }

    if type(expression) is ExactMapLiteral:
        payload: dict[str, Any] = {
            "kind": "exact_map",
            "map_type": _type_to_payload(expression.map_type, semantic=semantic),
            "matrix": [
                [_fraction_to_dict(value) for value in row]
                for row in expression.matrix
            ],
            "version": EXPRESSION_NODE_VERSION,
        }
        if not semantic:
            payload["label"] = expression.label
        return payload

    if type(expression) is Let:
        binder = (
            {
                "declared_type": _type_to_payload(
                    expression.symbol.declared_type,
                    semantic=True,
                ),
                "kind": "binder",
                "version": EXPRESSION_NODE_VERSION,
            }
            if semantic
            else _symbol_declaration(expression.symbol)
        )
        return {
            "binder": binder,
            "body": child(
                expression.body,
                scope=(*environment, expression.symbol),
            ),
            "kind": "let",
            "value": child(expression.value),
            "version": EXPRESSION_NODE_VERSION,
        }

    if type(expression) is IdentityMap:
        return {
            "kind": "identity_map",
            "space": _type_to_payload(expression.space, semantic=semantic),
            "version": EXPRESSION_NODE_VERSION,
        }

    if type(expression) is ZeroMap:
        return {
            "kind": "zero_map",
            "map_type": _type_to_payload(expression.map_type, semantic=semantic),
            "version": EXPRESSION_NODE_VERSION,
        }

    if type(expression) is Compose:
        return {
            "factors": [child(factor) for factor in expression.factors],
            "kind": "compose",
            "version": EXPRESSION_NODE_VERSION,
        }

    if type(expression) is Sum:
        return {
            "kind": "sum",
            "terms": [child(term) for term in expression.terms],
            "version": EXPRESSION_NODE_VERSION,
        }

    if type(expression) is Scale:
        return {
            "kind": "scale",
            "scalar": _fraction_to_dict(expression.scalar),
            "value": child(expression.value),
            "version": EXPRESSION_NODE_VERSION,
        }

    if type(expression) is Apply:
        return {
            "argument": child(expression.argument),
            "kind": "apply",
            "operator": child(expression.operator),
            "version": EXPRESSION_NODE_VERSION,
        }

    if type(expression) is PolynomialPrimitive:
        return {
            "kind": "polynomial_primitive",
            "map_type": _type_to_payload(expression.map_type, semantic=semantic),
            "parameters": [
                _fraction_to_dict(parameter) for parameter in expression.parameters
            ],
            "primitive_kind": expression.kind.value,
            "version": EXPRESSION_NODE_VERSION,
        }

    raise TypeError(
        f"unsupported expression node {type(expression).__module__}.{type(expression).__qualname__}"
    )


def _node_header(value: Any) -> tuple[Mapping[str, Any], str]:
    payload = _mapping(value, name="expression node")
    kind = _strict_text(payload.get("kind"), name="expression node kind")
    version = _strict_integer(
        payload.get("version"),
        name="expression node version",
        minimum=1,
    )
    if version != EXPRESSION_NODE_VERSION:
        raise ValidationError(f"unknown expression node version: {version}")
    return payload, kind


def _decode_symbol_declaration(
    value: Any,
    *,
    state: _DecodeState,
    depth: int,
) -> Symbol:
    state.node(depth=depth)
    payload, kind = _node_header(value)
    if kind != "symbol":
        raise ValidationError("let binder must use a symbol declaration")
    _require_fields(
        payload,
        {"declared_type", "kind", "name", "version"},
        name="symbol declaration",
    )
    name = _strict_text(payload["name"], name="binder label")
    if len(name) > state.budget.max_symbol_length:
        raise ResourceLimitError("binder label exceeds max_symbol_length")
    declared = _type_from_payload(
        payload["declared_type"],
        state=state,
        depth=depth + 1,
    )
    return Symbol(name, declared)


def _expression_from_payload(
    value: Any,
    *,
    state: _DecodeState,
    environment: tuple[Symbol, ...],
    depth: int,
) -> Expression:
    state.node(depth=depth)
    payload, kind = _node_header(value)

    if kind == "bound_symbol":
        _require_fields(
            payload,
            {"index", "kind", "version"},
            name="bound-symbol node",
        )
        index = _strict_integer(payload["index"], name="bound-symbol index")
        if index >= len(environment):
            raise ValidationError("expression contains a dangling bound-symbol index")
        return environment[-1 - index]

    if kind == "symbol":
        _require_fields(
            payload,
            {"declared_type", "kind", "name", "version"},
            name="symbol node",
        )
        name = _strict_text(payload["name"], name="symbol name")
        if len(name) > state.budget.max_symbol_length:
            raise ResourceLimitError("symbol name exceeds max_symbol_length")
        declared = _type_from_payload(
            payload["declared_type"],
            state=state,
            depth=depth + 1,
        )
        return Symbol(name, declared)

    if kind == "exact_value":
        _require_fields(
            payload,
            {"entries", "kind", "space", "version"},
            name="exact-value node",
        )
        space = _type_from_payload(payload["space"], state=state, depth=depth + 1)
        if not isinstance(space, (RationalSpace, PolynomialCoordinateSpace)):
            raise ValidationError("exact-value space must be a coordinate space")
        raw_entries = _array(payload["entries"], name="exact-value entries")
        if len(raw_entries) != space.dimension:
            raise ValidationError("exact-value entry count does not match its space")
        entries = tuple(
            state.fraction(item, name="exact-value entry") for item in raw_entries
        )
        return ExactValueLiteral(space, entries)

    if kind == "exact_map":
        _require_fields(
            payload,
            {"kind", "label", "map_type", "matrix", "version"},
            name="exact-map node",
        )
        map_type = _type_from_payload(
            payload["map_type"],
            state=state,
            depth=depth + 1,
        )
        if not isinstance(map_type, LinearMapType):
            raise ValidationError("exact-map type must be a LinearMapType")
        label = _strict_text(payload["label"], name="map label", allow_empty=True)
        if len(label) > state.budget.max_symbol_length:
            raise ResourceLimitError("map label exceeds max_symbol_length")
        raw_matrix = _array(payload["matrix"], name="exact-map matrix")
        if len(raw_matrix) != map_type.target.dimension:
            raise ValidationError("exact-map row count does not match its target")
        matrix: list[tuple[Fraction, ...]] = []
        for raw_row in raw_matrix:
            row = _array(raw_row, name="exact-map row")
            if len(row) != map_type.source.dimension:
                raise ValidationError("exact-map column count does not match its source")
            matrix.append(
                tuple(state.fraction(item, name="exact-map entry") for item in row)
            )
        return ExactMapLiteral(map_type, tuple(matrix), label)

    if kind == "let":
        _require_fields(
            payload,
            {"binder", "body", "kind", "value", "version"},
            name="let node",
        )
        state.edge(3)
        binder = _decode_symbol_declaration(
            payload["binder"],
            state=state,
            depth=depth + 1,
        )
        bound_value = _expression_from_payload(
            payload["value"],
            state=state,
            environment=environment,
            depth=depth + 1,
        )
        body = _expression_from_payload(
            payload["body"],
            state=state,
            environment=(*environment, binder),
            depth=depth + 1,
        )
        return Let(binder, bound_value, body)

    if kind == "identity_map":
        _require_fields(
            payload,
            {"kind", "space", "version"},
            name="identity-map node",
        )
        space = _type_from_payload(payload["space"], state=state, depth=depth + 1)
        if not isinstance(space, (RationalSpace, PolynomialCoordinateSpace)):
            raise ValidationError("identity space must be a coordinate space")
        return IdentityMap(space)

    if kind == "zero_map":
        _require_fields(
            payload,
            {"kind", "map_type", "version"},
            name="zero-map node",
        )
        map_type = _type_from_payload(
            payload["map_type"],
            state=state,
            depth=depth + 1,
        )
        if not isinstance(map_type, LinearMapType):
            raise ValidationError("zero-map type must be a LinearMapType")
        return ZeroMap(map_type)

    if kind == "compose":
        _require_fields(
            payload,
            {"factors", "kind", "version"},
            name="compose node",
        )
        raw_factors = _array(payload["factors"], name="composition factors")
        if len(raw_factors) < 2:
            raise ValidationError("a Compose node requires at least two factors")
        state.edge(len(raw_factors))
        factors = tuple(
            _expression_from_payload(
                item,
                state=state,
                environment=environment,
                depth=depth + 1,
            )
            for item in raw_factors
        )
        return Compose(factors)

    if kind == "sum":
        _require_fields(
            payload,
            {"kind", "terms", "version"},
            name="sum node",
        )
        raw_terms = _array(payload["terms"], name="sum terms")
        if len(raw_terms) < 2:
            raise ValidationError("a Sum node requires at least two terms")
        state.edge(len(raw_terms))
        terms = tuple(
            _expression_from_payload(
                item,
                state=state,
                environment=environment,
                depth=depth + 1,
            )
            for item in raw_terms
        )
        return Sum(terms)

    if kind == "scale":
        _require_fields(
            payload,
            {"kind", "scalar", "value", "version"},
            name="scale node",
        )
        state.edge(1)
        scalar = state.fraction(payload["scalar"], name="scale scalar")
        scaled = _expression_from_payload(
            payload["value"],
            state=state,
            environment=environment,
            depth=depth + 1,
        )
        return Scale(scalar, scaled)

    if kind == "apply":
        _require_fields(
            payload,
            {"argument", "kind", "operator", "version"},
            name="apply node",
        )
        state.edge(2)
        operator = _expression_from_payload(
            payload["operator"],
            state=state,
            environment=environment,
            depth=depth + 1,
        )
        argument = _expression_from_payload(
            payload["argument"],
            state=state,
            environment=environment,
            depth=depth + 1,
        )
        return Apply(operator, argument)

    if kind == "polynomial_primitive":
        _require_fields(
            payload,
            {"kind", "map_type", "parameters", "primitive_kind", "version"},
            name="polynomial-primitive node",
        )
        map_type = _type_from_payload(
            payload["map_type"],
            state=state,
            depth=depth + 1,
        )
        if not isinstance(map_type, LinearMapType):
            raise ValidationError("polynomial primitive type must be a LinearMapType")
        raw_parameters = _array(
            payload["parameters"],
            name="polynomial primitive parameters",
        )
        parameters = tuple(
            state.fraction(item, name="polynomial primitive parameter")
            for item in raw_parameters
        )
        primitive_name = _strict_text(
            payload["primitive_kind"],
            name="polynomial primitive kind",
        )
        try:
            primitive_kind = PrimitiveKind(primitive_name)
        except ValueError as exc:
            raise ValidationError(
                f"unknown polynomial primitive kind: {primitive_name!r}"
            ) from exc
        return PolynomialPrimitive(primitive_kind, map_type, parameters)

    raise ValidationError(f"unknown expression node kind: {kind!r}")


def _semantic_payload(expression: Expression, budget: ExpressionBudget) -> dict[str, Any]:
    try:
        return _expression_to_payload(
            expression,
            environment=(),
            semantic=True,
            depth=1,
            budget=budget,
            occurrences=[0],
        )
    except RecursionError as exc:
        raise ResourceLimitError("expression serialization exceeded recursion capacity") from exc


def expression_digest(
    expression: Expression,
    *,
    budget: ExpressionBudget | None = None,
) -> str:
    """Return the SHA-256 digest of canonical semantic JSON.

    Display labels and bound-variable names are absent from the hashed tree.
    Free symbol names and complete expression types remain semantic.
    """

    limit = _budget(budget)
    _preflight(expression, limit)
    semantic = _semantic_payload(expression, limit)
    text = _check_encoded_size(semantic, limit)
    return sha256(text.encode("utf-8")).hexdigest()


def expression_to_dict(
    expression: Expression,
    *,
    budget: ExpressionBudget | None = None,
) -> dict[str, Any]:
    """Return a versioned, digest-bearing expression transport envelope."""

    limit = _budget(budget)
    _preflight(expression, limit)
    try:
        payload = _expression_to_payload(
            expression,
            environment=(),
            semantic=False,
            depth=1,
            budget=limit,
            occurrences=[0],
        )
    except RecursionError as exc:
        raise ResourceLimitError("expression serialization exceeded recursion capacity") from exc
    envelope = {
        "expression": payload,
        "schema": EXPRESSION_SCHEMA,
        "schema_version": EXPRESSION_SCHEMA_VERSION,
        "semantic_sha256": expression_digest(expression, budget=limit),
    }
    _validate_json_tree(envelope, depth=0, budget=limit)
    _check_encoded_size(envelope, limit)
    # The in-memory language is a DAG while the portable form is a tree.  A
    # shared node therefore appears once to graph preflight but once per
    # occurrence to the decoder.  Replay the generated envelope through the
    # strict decoder so node, edge, entry, label, and type-witness budgets are
    # symmetric: this function never emits data that its paired decoder would
    # reject under the same budget.
    expression_from_dict(envelope, budget=limit)
    return envelope


def _validate_digest(value: Any) -> str:
    digest = _strict_text(value, name="semantic SHA-256")
    if len(digest) != 64 or any(character not in _HEX_DIGITS for character in digest):
        raise ValidationError("semantic_sha256 must be a lowercase SHA-256 digest")
    return digest


def expression_from_dict(
    value: Mapping[str, Any],
    *,
    budget: ExpressionBudget | None = None,
) -> Expression:
    """Validate and reconstruct one expression transport envelope."""

    limit = _budget(budget)
    _validate_json_tree(value, depth=0, budget=limit)
    _check_encoded_size(value, limit)
    envelope = _mapping(value, name="expression envelope")
    _require_fields(
        envelope,
        {"expression", "schema", "schema_version", "semantic_sha256"},
        name="expression envelope",
    )
    schema = _strict_text(envelope["schema"], name="expression schema")
    if schema != EXPRESSION_SCHEMA:
        raise ValidationError(f"unknown expression schema: {schema!r}")
    schema_version = _strict_integer(
        envelope["schema_version"],
        name="expression schema version",
        minimum=1,
    )
    if schema_version != EXPRESSION_SCHEMA_VERSION:
        raise ValidationError(f"unknown expression schema version: {schema_version}")
    expected_digest = _validate_digest(envelope["semantic_sha256"])
    state = _DecodeState(limit)
    try:
        expression = _expression_from_payload(
            envelope["expression"],
            state=state,
            environment=(),
            depth=1,
        )
    except RecursionError as exc:
        raise ResourceLimitError("expression decoding exceeded recursion capacity") from exc
    _preflight(expression, limit)
    observed_digest = expression_digest(expression, budget=limit)
    if not hmac.compare_digest(observed_digest, expected_digest):
        raise ValidationError("expression semantic SHA-256 mismatch")
    return expression


def expression_to_json(
    expression: Expression,
    *,
    budget: ExpressionBudget | None = None,
) -> str:
    """Serialize one expression as compact canonical UTF-8/NFC JSON text."""

    limit = _budget(budget)
    envelope = expression_to_dict(expression, budget=limit)
    return _check_encoded_size(envelope, limit)


def expression_from_json(
    value: str | bytes,
    *,
    budget: ExpressionBudget | None = None,
) -> Expression:
    """Parse canonical JSON with duplicate-key, float, and digest checks."""

    limit = _budget(budget)
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValidationError("expression JSON must be valid UTF-8") from exc
    elif isinstance(value, str):
        text = value
    else:
        raise TypeError("expression JSON must be str or bytes")
    try:
        encoded_size = len(text.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValidationError("expression JSON must be valid UTF-8") from exc
    if encoded_size > limit.max_serialized_bytes:
        raise ResourceLimitError("expression JSON exceeds max_serialized_bytes")
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except ValidationError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ValidationError("expression data is not valid canonical JSON") from exc
    _validate_json_tree(payload, depth=0, budget=limit)
    canonical = _check_encoded_size(payload, limit)
    if text != canonical:
        raise ValidationError("expression JSON is not in canonical form")
    return expression_from_dict(payload, budget=limit)


__all__ = [
    "EXPRESSION_NODE_VERSION",
    "EXPRESSION_SCHEMA",
    "EXPRESSION_SCHEMA_VERSION",
    "EXPRESSION_TYPE_VERSION",
    "expression_digest",
    "expression_from_dict",
    "expression_from_json",
    "expression_to_dict",
    "expression_to_json",
    "type_from_dict",
    "type_to_dict",
]
