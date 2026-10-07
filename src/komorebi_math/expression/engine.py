"""Checking, exact normalization, certified rewriting, and execution plans.

The engine is intentionally small in mathematical scope.  It evaluates the
finite typed language from :mod:`komorebi_math.expression.model` to exact
rational matrices or coordinate vectors.  This gives every closed, well-typed
term a unique normal form and makes equality in this language decidable by a
finite entrywise comparison.

No rule in this module is selected by a display label or a caller supplied
theorem name.  Polynomial rules inspect branded primitive kinds, exact source
and target spaces, and rational parameters, and every accepted rewrite is
replayed against the exact normal form before it is reported as certified.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, cast
from unicodedata import normalize as unicode_normalize

from ..certificates import Certificate
from ..coordinates.exact import (
    Matrix,
    Vector,
    identity_matrix,
    matrix_add,
    matrix_multiply,
    matrix_shape,
    matrix_vector_multiply,
    zero_matrix,
)
from ..exceptions import DimensionError, ResourceLimitError, ValidationError
from .certification import (
    ExpressionBinding,
    NormalForm,
    RewriteStep,
    compose_certificates,
    exact_internal_certificate,
)
from .model import (
    Apply,
    Compose,
    CoordinateSpace,
    ExactMapLiteral,
    ExactValueLiteral,
    Expression,
    ExpressionBudget,
    ExpressionStats,
    IdentityMap,
    Let,
    LinearMapType,
    PolynomialCoordinateSpace,
    RationalSpace,
    Scale,
    Sum,
    Symbol,
    ZeroMap,
    add,
    compose,
    exact_value,
    identity,
    scale,
    zero_map,
)
from .planning import MatrixChainPlan, matrix_chain_multiply, matrix_chain_plan
from .polynomial import (
    PolynomialPrimitive,
    PrimitiveKind,
    boundary_functional,
    constant_injection,
    coordinate_change,
    derivative,
    evaluation,
    interval_integration,
    primitive_certificate,
    replay,
)

ExactData = Matrix | Vector


def _validate_coordinate_space(space: Any, *, name: str) -> CoordinateSpace:
    """Replay one semantic space and reject subclasses or mutated invariants."""

    if type(space) is RationalSpace:
        return space
    if type(space) is not PolynomialCoordinateSpace:
        raise TypeError(f"{name} must be an exact built-in coordinate-space class")
    polynomial = space
    if any(
        type(coefficient) is not Fraction
        for member in polynomial.basis
        for coefficient in member
    ):
        raise ValidationError(f"{name} basis coefficients must be Fractions")
    replayed = PolynomialCoordinateSpace(
        polynomial.variable,
        polynomial.degree,
        polynomial.basis,
        polynomial.label,
    )
    if (
        replayed.variable != polynomial.variable
        or replayed.degree != polynomial.degree
        or replayed.basis != polynomial.basis
        or replayed.label != polynomial.label
    ):
        raise ValidationError(f"{name} is not in canonical constructor form")
    return polynomial


def _validate_expression_type(
    expression_type: Any,
    *,
    name: str,
) -> CoordinateSpace | LinearMapType:
    if type(expression_type) is LinearMapType:
        map_type = expression_type
        _validate_coordinate_space(map_type.source, name=f"{name} source")
        _validate_coordinate_space(map_type.target, name=f"{name} target")
        replayed = LinearMapType(map_type.source, map_type.target)
        if replayed != map_type:
            raise ValidationError(f"{name} is not in canonical constructor form")
        return map_type
    return _validate_coordinate_space(expression_type, name=name)


def _replay_node_invariants(node: Expression) -> None:
    """Reconstruct one node without mutating it and replay every constructor check."""

    node_class = type(node)
    if node_class is ExactValueLiteral:
        literal = cast(ExactValueLiteral, node)
        if any(type(value) is not Fraction for value in literal.entries):
            raise ValidationError("exact-value entries must be Fractions")
        replayed: Expression = ExactValueLiteral(literal.space, literal.entries)
    elif node_class is ExactMapLiteral:
        literal_map = cast(ExactMapLiteral, node)
        if any(type(value) is not Fraction for row in literal_map.matrix for value in row):
            raise ValidationError("exact-map entries must be Fractions")
        replayed = ExactMapLiteral(
            literal_map.map_type,
            literal_map.matrix,
            literal_map.label,
        )
        if replayed.label != literal_map.label:
            raise ValidationError("exact-map label is not in canonical constructor form")
    elif node_class is Symbol:
        symbolic = cast(Symbol, node)
        replayed_symbol = Symbol(symbolic.name, symbolic.declared_type)
        if replayed_symbol.name != symbolic.name:
            raise ValidationError("symbol name is not in canonical constructor form")
        replayed = replayed_symbol
    elif node_class is Let:
        binding = cast(Let, node)
        replayed = Let(binding.symbol, binding.value, binding.body)
    elif node_class is IdentityMap:
        identity_node = cast(IdentityMap, node)
        replayed = IdentityMap(identity_node.space)
    elif node_class is ZeroMap:
        zero_node = cast(ZeroMap, node)
        replayed = ZeroMap(zero_node.map_type)
    elif node_class is Compose:
        composition = cast(Compose, node)
        replayed = Compose(composition.factors)
    elif node_class is Sum:
        summation = cast(Sum, node)
        replayed = Sum(summation.terms)
    elif node_class is Scale:
        scaled = cast(Scale, node)
        if type(scaled.scalar) is not Fraction:
            raise ValidationError("scale coefficient must be a Fraction")
        replayed = Scale(scaled.scalar, scaled.value)
    elif node_class is Apply:
        application = cast(Apply, node)
        replayed = Apply(application.operator, application.argument)
    elif node_class is PolynomialPrimitive:
        primitive = cast(PolynomialPrimitive, node)
        if type(primitive.kind) is not PrimitiveKind:
            raise ValidationError("polynomial primitive kind is not canonical")
        if any(type(value) is not Fraction for value in primitive.parameters):
            raise ValidationError("polynomial primitive parameters must be Fractions")
        replayed = PolynomialPrimitive(
            primitive.kind,
            primitive.map_type,
            primitive.parameters,
        )
    else:
        raise TypeError(
            "expression nodes must use an exact built-in Block 20 node class; "
            f"received {node_class.__module__}.{node_class.__qualname__}"
        )
    observed_type = _validate_expression_type(
        node.expression_type,
        name=f"{node_class.__name__} expression type",
    )
    if replayed.expression_type != observed_type:
        raise ValidationError(f"{node_class.__name__} expression type failed constructor replay")


def _exact_node_children(node: Expression) -> tuple[Expression, ...]:
    node_class = type(node)
    if node_class in {
        ExactValueLiteral,
        ExactMapLiteral,
        Symbol,
        IdentityMap,
        ZeroMap,
        PolynomialPrimitive,
    }:
        return ()
    if node_class is Let:
        binding = cast(Let, node)
        return (binding.symbol, binding.value, binding.body)
    if node_class is Compose:
        return cast(Compose, node).factors
    if node_class is Sum:
        return cast(Sum, node).terms
    if node_class is Scale:
        return (cast(Scale, node).value,)
    if node_class is Apply:
        application = cast(Apply, node)
        return (application.operator, application.argument)
    raise TypeError(
        "expression nodes must use an exact built-in Block 20 node class; "
        f"received {node_class.__module__}.{node_class.__qualname__}"
    )


def _replay_graph_invariants(root: Expression, budget: ExpressionBudget) -> None:
    """Cycle-check an exact-class graph, then replay nodes in postorder."""

    color: dict[int, int] = {}
    postorder: list[Expression] = []
    edge_count = 0
    stack: list[tuple[Expression, bool, int]] = [(root, False, 1)]
    while stack:
        node, exiting, depth = stack.pop()
        if depth > budget.max_depth:
            raise ResourceLimitError("expression depth exceeds max_depth")
        children = _exact_node_children(node)
        identifier = id(node)
        state = color.get(identifier, 0)
        if exiting:
            if state == 1:
                color[identifier] = 2
                postorder.append(node)
            continue
        if state == 1:
            raise ValidationError("expression graph contains a cycle")
        if state == 2:
            continue
        color[identifier] = 1
        if len(color) > budget.max_nodes:
            raise ResourceLimitError("expression node count exceeds max_nodes")
        edge_count += len(children)
        if len(color) + edge_count > budget.max_steps:
            raise ResourceLimitError("expression invariant replay exceeds max_steps")
        stack.append((node, True, depth))
        for child in reversed(children):
            if not isinstance(child, Expression):
                raise TypeError("expression children must be Expression nodes")
            if color.get(id(child), 0) == 1:
                raise ValidationError("expression graph contains a cycle")
            stack.append((child, False, depth + 1))
    for node in postorder:
        _replay_node_invariants(node)


def _fraction(value: Any, *, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise ValidationError(f"{name} must be an integer or Fraction")
    return Fraction(value)


def _exact_vector(value: Any, *, size: int, name: str) -> Vector:
    if isinstance(value, (int, Fraction)) and not isinstance(value, bool):
        values = (value,)
    else:
        try:
            values = tuple(value)
        except TypeError as exc:
            raise ValidationError(f"{name} must be a finite exact vector") from exc
    result = tuple(_fraction(entry, name=f"{name} entry") for entry in values)
    if len(result) != size:
        raise DimensionError(f"{name} must have {size} entries")
    return result


def _exact_matrix(value: Any, *, rows: int, columns: int, name: str) -> Matrix:
    try:
        raw_rows = tuple(value)
    except TypeError as exc:
        raise ValidationError(f"{name} must be a finite exact matrix") from exc
    result: list[tuple[Fraction, ...]] = []
    for row in raw_rows:
        try:
            entries = tuple(row)
        except TypeError as exc:
            raise ValidationError(f"{name} rows must be finite") from exc
        result.append(tuple(_fraction(entry, name=f"{name} entry") for entry in entries))
    matrix = tuple(result)
    if matrix_shape(matrix) != (rows, columns):
        raise DimensionError(f"{name} must have shape {(rows, columns)}")
    return matrix


def _data_for_type(value: Any, expression_type: Any, *, name: str) -> ExactData:
    checked_type = _validate_expression_type(expression_type, name=f"{name} type")
    if isinstance(checked_type, LinearMapType):
        map_type = checked_type
        return _exact_matrix(
            value,
            rows=map_type.target.dimension,
            columns=map_type.source.dimension,
            name=name,
        )
    return _exact_vector(value, size=checked_type.dimension, name=name)


def exact_binding(
    value: Any,
    declared_type: CoordinateSpace | LinearMapType,
    *,
    theorem_id: str = "EXPR.BINDING.EXACT",
) -> ExpressionBinding:
    """Construct a strictly validated exact rational free-symbol binding."""

    data = _data_for_type(value, declared_type, name="binding value")
    return ExpressionBinding(
        data,
        declared_type,
        exact_internal_certificate(
            "exact rational expression binding",
            theorem_id=theorem_id,
            assumptions_checked=("binding shape agrees with its declared type",),
        ),
    )


def _fraction_token(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _space_payload(space: CoordinateSpace) -> Any:
    checked = _validate_coordinate_space(space, name="digest space")
    if type(checked) is RationalSpace:
        return ["Q"]
    polynomial = cast(PolynomialCoordinateSpace, checked)
    return [
        "polynomial",
        polynomial.variable,
        polynomial.degree,
        [[_fraction_token(value) for value in member] for member in polynomial.basis],
    ]


def _type_payload(expression_type: Any) -> Any:
    checked = _validate_expression_type(expression_type, name="digest expression type")
    if isinstance(checked, LinearMapType):
        map_type = checked
        return ["map", _space_payload(map_type.source), _space_payload(map_type.target)]
    return ["value", _space_payload(checked)]


def _data_payload(data: ExactData, expression_type: Any) -> Any:
    if isinstance(expression_type, LinearMapType):
        matrix = cast(Matrix, data)
        return [[_fraction_token(value) for value in row] for row in matrix]
    vector = cast(Vector, data)
    return [_fraction_token(value) for value in vector]


def normal_form_digest(data: ExactData, expression_type: Any) -> str:
    """Return the canonical SHA-256 digest of typed exact normal-form data."""

    checked = _data_for_type(data, expression_type, name="normal-form data")
    payload = {
        "data": _data_payload(checked, expression_type),
        "type": _type_payload(expression_type),
        "version": "komorebi_math.expression.normal_form.v1",
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CheckedExpression:
    expression_type: CoordinateSpace | LinearMapType
    stats: ExpressionStats
    free_symbols: tuple[Symbol, ...]
    certificate: Certificate


def _binding_items(
    bindings: Mapping[str | Symbol, ExpressionBinding] | None,
) -> tuple[tuple[str | Symbol, ExpressionBinding], ...]:
    if bindings is None:
        return ()
    if not isinstance(bindings, Mapping):
        raise TypeError("bindings must be a mapping")
    result: list[tuple[str | Symbol, ExpressionBinding]] = []
    seen_names: dict[str, ExpressionBinding] = {}
    for key, binding in bindings.items():
        if not isinstance(key, str) and type(key) is not Symbol:
            raise TypeError("binding keys must be strings or Symbols")
        if type(binding) is not ExpressionBinding:
            raise TypeError("binding values must be ExpressionBinding records")
        if isinstance(key, str):
            name = unicode_normalize("NFC", key)
            if not name.strip():
                raise ValidationError("binding names must be nonempty")
            prior = seen_names.get(name)
            if prior is not None and prior != binding:
                raise ValidationError(f"conflicting bindings for symbol {name!r}")
            seen_names[name] = binding
            result.append((name, binding))
        else:
            result.append((key, binding))
    return tuple(result)


def _lookup_binding(
    target: Symbol,
    items: tuple[tuple[str | Symbol, ExpressionBinding], ...],
) -> ExpressionBinding | None:
    symbol_matches = [binding for key, binding in items if type(key) is Symbol and key is target]
    name_matches = [binding for key, binding in items if isinstance(key, str) and key == target.name]
    matches = symbol_matches or name_matches
    if not matches:
        return None
    first = matches[0]
    if any(binding != first for binding in matches[1:]):
        raise ValidationError(f"ambiguous bindings for symbol {target.name!r}")
    if first.declared_type != target.declared_type:
        raise ValidationError(f"binding type does not match symbol {target.name!r}")
    _data_for_type(first.value, first.declared_type, name=f"binding {target.name!r}")
    return first


def check_expression(
    expression: Expression,
    bindings: Mapping[str | Symbol, ExpressionBinding] | None = None,
    *,
    budget: ExpressionBudget | None = None,
) -> CheckedExpression:
    """Replay structural typing and report declared free symbols.

    A missing free-symbol value does not make the expression ill typed; it
    merely prevents normalization until a compatible binding is supplied.
    """

    if not isinstance(expression, Expression):
        raise TypeError("expression must be an Expression")
    selected_budget = ExpressionBudget() if budget is None else budget
    if type(selected_budget) is not ExpressionBudget:
        raise TypeError("budget must be an ExpressionBudget")
    try:
        _replay_graph_invariants(expression, selected_budget)
    except RecursionError as exc:
        raise ResourceLimitError(
            "expression invariants exceed the constructor replay call-stack guard"
        ) from exc
    stats = selected_budget.preflight(expression)
    items = _binding_items(bindings)
    free: dict[tuple[str, Any], Symbol] = {}
    visited: set[tuple[int, tuple[int, ...]]] = set()
    work = 0

    def walk(node: Expression, local: tuple[Symbol, ...]) -> None:
        nonlocal work
        work += 1
        if work > selected_budget.max_steps:
            raise ResourceLimitError("expression type-check traversal exceeds max_steps")
        key = (id(node), tuple(id(binder) for binder in local))
        if key in visited:
            return
        visited.add(key)
        if type(node) is Symbol:
            symbolic = node
            free_key = (symbolic.name, symbolic.declared_type)
            if not any(symbolic is binder for binder in local):
                free.setdefault(free_key, symbolic)
                _lookup_binding(symbolic, items)
            return
        if type(node) is Let:
            binding = node
            walk(binding.value, local)
            walk(binding.body, (*local, binding.symbol))
            return
        for child in _exact_node_children(node):
            walk(child, local)

    try:
        walk(expression, ())
    except RecursionError as exc:
        raise ResourceLimitError("expression depth exceeds the evaluator call-stack guard") from exc
    free_symbols = tuple(sorted(free.values(), key=lambda item: (item.name, repr(item.declared_type))))
    missing = tuple(symbol.name for symbol in free_symbols if _lookup_binding(symbol, items) is None)
    certificate = exact_internal_certificate(
        "finite expression type checking",
        theorem_id="EXPR.TYPE.FINITE",
        assumptions_checked=(
            "every node constructor replayed its source and target checks",
            "semantic spaces include complete exact basis witnesses",
            "resource budget preflight completed",
        ),
        diagnostics={
            "nodes": stats.nodes,
            "depth": stats.depth,
            "free_symbols": tuple(symbol.name for symbol in free_symbols),
            "missing_binding_values": missing,
        },
    )
    return CheckedExpression(expression.expression_type, stats, free_symbols, certificate)


def _literal_certificate(operation: str) -> Certificate:
    return exact_internal_certificate(operation, theorem_id="EXPR.NORMAL_FORM.EXACT")


def _normal_form(
    data: ExactData,
    expression_type: Any,
    certificate: Certificate,
    *,
    budget: ExpressionBudget | None = None,
) -> NormalForm:
    checked = _data_for_type(data, expression_type, name="normal-form result")
    if budget is not None:
        _validate_data_budget(checked, budget, name="normal-form result")
    return NormalForm(
        expression_type,
        checked,
        normal_form_digest(checked, expression_type),
        certificate,
    )


def _add_data(left: ExactData, right: ExactData, expression_type: Any) -> ExactData:
    if isinstance(expression_type, LinearMapType):
        return matrix_add(cast(Matrix, left), cast(Matrix, right))
    return tuple(
        a + b
        for a, b in zip(cast(Vector, left), cast(Vector, right), strict=True)
    )


def _scale_data(scalar: Fraction, data: ExactData) -> ExactData:
    if data and isinstance(data[0], tuple):
        return tuple(tuple(scalar * value for value in row) for row in data)
    vector = cast(Vector, data)
    return tuple(scalar * value for value in vector)


def _validate_data_budget(
    data: ExactData,
    budget: ExpressionBudget,
    *,
    name: str,
) -> None:
    if data and isinstance(data[0], tuple):
        entries = tuple(value for row in data for value in row)
    else:
        entries = cast(Vector, data)
    if len(entries) > budget.max_entries:
        raise ResourceLimitError(f"{name} exceeds max_entries")
    maximum_bits = max(
        (
            max(abs(value.numerator).bit_length(), value.denominator.bit_length())
            for value in entries
        ),
        default=0,
    )
    if maximum_bits > budget.max_coefficient_bits:
        raise ResourceLimitError(f"{name} exceeds max_coefficient_bits")
    encoded_size = sum(
        len(str(value.numerator).encode("ascii"))
        + len(str(value.denominator).encode("ascii"))
        + 2
        for value in entries
    )
    if encoded_size > budget.max_serialized_bytes:
        raise ResourceLimitError(f"{name} exceeds max_serialized_bytes")


def _normalize_node(
    node: Expression,
    binding_items: tuple[tuple[str | Symbol, ExpressionBinding], ...],
    local: tuple[tuple[Symbol, NormalForm], ...],
    budget: ExpressionBudget,
    cache: dict[tuple[int, tuple[tuple[int, int], ...]], NormalForm],
    work: list[int],
) -> NormalForm:
    work[0] += 1
    if work[0] > budget.max_steps:
        raise ResourceLimitError("expression normalization exceeds max_steps")
    key = (id(node), tuple((id(binder), id(value)) for binder, value in local))
    cached = cache.get(key)
    if cached is not None:
        return cached
    result = _normalize_node_uncached(
        node,
        binding_items,
        local,
        budget,
        cache,
        work,
    )
    _validate_data_budget(result.data, budget, name="normal-form intermediate")
    cache[key] = result
    return result


def _normalize_node_uncached(
    node: Expression,
    binding_items: tuple[tuple[str | Symbol, ExpressionBinding], ...],
    local: tuple[tuple[Symbol, NormalForm], ...],
    budget: ExpressionBudget,
    cache: dict[tuple[int, tuple[tuple[int, int], ...]], NormalForm],
    work: list[int],
) -> NormalForm:
    if isinstance(node, ExactValueLiteral):
        return _normal_form(
            node.entries,
            node.expression_type,
            _literal_certificate("exact value literal"),
            budget=budget,
        )
    if isinstance(node, ExactMapLiteral):
        return _normal_form(
            node.matrix,
            node.expression_type,
            _literal_certificate("exact map literal"),
            budget=budget,
        )
    if isinstance(node, PolynomialPrimitive):
        return _normal_form(
            replay(node),
            node.expression_type,
            primitive_certificate(node),
            budget=budget,
        )
    if isinstance(node, IdentityMap):
        return _normal_form(
            identity_matrix(node.space.dimension),
            node.expression_type,
            _literal_certificate("typed identity matrix"),
            budget=budget,
        )
    if isinstance(node, ZeroMap):
        return _normal_form(
            zero_matrix(node.map_type.target.dimension, node.map_type.source.dimension),
            node.expression_type,
            _literal_certificate("typed zero matrix"),
            budget=budget,
        )
    if isinstance(node, Symbol):
        for binder, value in reversed(local):
            if node is binder:
                return value
        binding = _lookup_binding(node, binding_items)
        if binding is None:
            raise ValidationError(f"free symbol {node.name!r} has no value binding")
        data = _data_for_type(binding.value, binding.declared_type, name=f"binding {node.name!r}")
        return _normal_form(
            data,
            node.expression_type,
            binding.certificate,
            budget=budget,
        )
    if isinstance(node, Let):
        value = _normalize_node(node.value, binding_items, local, budget, cache, work)
        body = _normalize_node(
            node.body,
            binding_items,
            (*local, (node.symbol, value)),
            budget,
            cache,
            work,
        )
        certificate = compose_certificates(
            (value.certificate, body.certificate),
            operation="typed lexical let normalization",
            theorem_id="EXPR.NORMAL_FORM.LET",
        )
        return _normal_form(
            body.data,
            body.expression_type,
            certificate,
            budget=budget,
        )
    if isinstance(node, Compose):
        forms = tuple(
            _normalize_node(factor, binding_items, local, budget, cache, work)
            for factor in node.factors
        )
        matrix = forms[0].data
        for form in forms[1:]:
            matrix = matrix_multiply(matrix, form.data)
            _validate_data_budget(matrix, budget, name="composition intermediate")
        certificate = compose_certificates(
            (form.certificate for form in forms),
            operation="exact matrix composition",
            theorem_id="EXPR.NORMAL_FORM.COMPOSE",
        )
        return _normal_form(matrix, node.expression_type, certificate, budget=budget)
    if isinstance(node, Sum):
        forms = tuple(
            _normalize_node(term, binding_items, local, budget, cache, work)
            for term in node.terms
        )
        data = forms[0].data
        for form in forms[1:]:
            data = _add_data(data, form.data, node.expression_type)
            _validate_data_budget(data, budget, name="sum intermediate")
        certificate = compose_certificates(
            (form.certificate for form in forms),
            operation="exact rational addition",
            theorem_id="EXPR.NORMAL_FORM.SUM",
        )
        return _normal_form(data, node.expression_type, certificate, budget=budget)
    if isinstance(node, Scale):
        value = _normalize_node(node.value, binding_items, local, budget, cache, work)
        data = _scale_data(node.scalar, value.data)
        certificate = compose_certificates(
            (value.certificate,),
            operation="exact rational scaling",
            theorem_id="EXPR.NORMAL_FORM.SCALE",
        )
        return _normal_form(data, node.expression_type, certificate, budget=budget)
    if isinstance(node, Apply):
        operator = _normalize_node(node.operator, binding_items, local, budget, cache, work)
        argument = _normalize_node(node.argument, binding_items, local, budget, cache, work)
        data = matrix_vector_multiply(operator.data, argument.data)
        certificate = compose_certificates(
            (operator.certificate, argument.certificate),
            operation="exact matrix-vector application",
            theorem_id="EXPR.NORMAL_FORM.APPLY",
        )
        return _normal_form(data, node.expression_type, certificate, budget=budget)
    raise TypeError(f"unsupported expression node: {type(node).__name__}")


def normalize_expression(
    expression: Expression,
    bindings: Mapping[str | Symbol, ExpressionBinding] | None = None,
    *,
    budget: ExpressionBudget | None = None,
) -> NormalForm:
    """Reduce a closed finite typed term to its unique exact rational data."""

    selected_budget = ExpressionBudget() if budget is None else budget
    checked = check_expression(expression, bindings, budget=selected_budget)
    items = _binding_items(bindings)
    missing = [symbol.name for symbol in checked.free_symbols if _lookup_binding(symbol, items) is None]
    if missing:
        raise ValidationError(f"normalization requires bindings for free symbols: {', '.join(missing)}")
    try:
        result = _normalize_node(
            expression,
            items,
            (),
            selected_budget,
            {},
            [0],
        )
    except RecursionError as exc:
        raise ResourceLimitError("expression depth exceeds the evaluator call-stack guard") from exc
    certificate = compose_certificates(
        (checked.certificate, result.certificate),
        operation="unique exact finite normal form",
        theorem_id="EXPR.NORMAL_FORM.UNIQUE",
        diagnostics={"normal_form_digest": result.digest},
    )
    final = _normal_form(
        result.data,
        result.expression_type,
        certificate,
        budget=selected_budget,
    )
    _validate_data_budget(final.data, selected_budget, name="normal-form result")
    return final


@dataclass(frozen=True, slots=True)
class EquivalenceResult:
    equivalent: bool
    left: NormalForm | None
    right: NormalForm | None
    certificate: Certificate

    def require_equivalent(self) -> EquivalenceResult:
        self.certificate.require()
        if not self.equivalent:
            raise ValidationError("expressions have different exact normal forms")
        return self


def equivalent(
    left: Expression,
    right: Expression,
    *,
    left_bindings: Mapping[str | Symbol, ExpressionBinding] | None = None,
    right_bindings: Mapping[str | Symbol, ExpressionBinding] | None = None,
    budget: ExpressionBudget | None = None,
) -> EquivalenceResult:
    """Decide equality of two closed expressions by exact normal forms."""

    if not isinstance(left, Expression) or not isinstance(right, Expression):
        raise TypeError("equivalent requires two Expression values")
    left_checked = check_expression(left, left_bindings, budget=budget)
    right_checked = check_expression(right, right_bindings, budget=budget)
    if left_checked.expression_type != right_checked.expression_type:
        certificate = exact_internal_certificate(
            "typed inequality decision",
            theorem_id="EXPR.EQUALITY.DECIDABLE",
            diagnostics={"equivalent": False, "reason": "different semantic types"},
        )
        return EquivalenceResult(False, None, None, certificate)
    left_form = normalize_expression(left, left_bindings, budget=budget)
    right_form = normalize_expression(right, right_bindings, budget=budget)
    same = left_form.data == right_form.data
    certificate = compose_certificates(
        (left_form.certificate, right_form.certificate),
        operation="finite exact normal-form equality decision",
        theorem_id="EXPR.EQUALITY.DECIDABLE",
        diagnostics={
            "equivalent": same,
            "left_digest": left_form.digest,
            "right_digest": right_form.digest,
            "comparison": "entrywise exact rational equality",
        },
    )
    return EquivalenceResult(same, left_form, right_form, certificate)


def _zero_expression(expression_type: Any) -> Expression:
    if isinstance(expression_type, LinearMapType):
        return zero_map(expression_type)
    return exact_value(expression_type, (Fraction(0) for _ in range(expression_type.dimension)))


def _flatten_compose(node: Expression) -> tuple[Expression, ...]:
    if not isinstance(node, Compose):
        return (node,)
    result: list[Expression] = []
    stack = list(reversed(node.factors))
    while stack:
        current = stack.pop()
        if isinstance(current, Compose):
            stack.extend(reversed(current.factors))
        else:
            result.append(current)
    return tuple(result)


def _flatten_sum(node: Expression) -> tuple[Expression, ...]:
    if not isinstance(node, Sum):
        return (node,)
    result: list[Expression] = []
    stack = list(reversed(node.terms))
    while stack:
        current = stack.pop()
        if isinstance(current, Sum):
            stack.extend(reversed(current.terms))
        else:
            result.append(current)
    return tuple(result)


def _primitive_pair_rewrite(outer: Expression, inner: Expression) -> tuple[Expression, str, str] | None:
    if not isinstance(outer, PolynomialPrimitive) or not isinstance(inner, PolynomialPrimitive):
        return None

    if outer.kind is PrimitiveKind.COORDINATE_CHANGE and inner.kind is PrimitiveKind.COORDINATE_CHANGE:
        source = inner.map_type.source
        target = outer.map_type.target
        assert isinstance(source, PolynomialCoordinateSpace)
        assert isinstance(target, PolynomialCoordinateSpace)
        candidate: Expression = identity(source) if source == target else coordinate_change(source, target)
        return candidate, "coordinate cocycle", "EXPR.REWRITE.COORDINATE_COCYCLE"

    if outer.kind is PrimitiveKind.DERIVATIVE and inner.kind is PrimitiveKind.ANTIDERIVATIVE:
        if outer.map_type.source == inner.map_type.target and outer.map_type.target == inner.map_type.source:
            return identity(inner.map_type.source), "derivative-antiderivative cancellation", "EXPR.REWRITE.DJ"

    if outer.kind is PrimitiveKind.DERIVATIVE and inner.kind is PrimitiveKind.COORDINATE_CHANGE:
        # D_B C_(A->B) = C_(A'->B') D_A for the tight derivative
        # spaces A' and B'.  The exact replay below is the final authority;
        # this structural match merely proposes the candidate.
        if outer.map_type.source == inner.map_type.target:
            source = inner.map_type.source
            target = outer.map_type.target
            if isinstance(source, PolynomialCoordinateSpace) and isinstance(
                target, PolynomialCoordinateSpace
            ):
                source_derivative = derivative(source)
                derivative_target = source_derivative.map_type.target
                assert isinstance(derivative_target, PolynomialCoordinateSpace)
                candidate = compose(
                    coordinate_change(derivative_target, target),
                    source_derivative,
                )
                return (
                    candidate,
                    "derivative coordinate naturality",
                    "EXPR.REWRITE.DERIVATIVE_NATURALITY",
                )

    if outer.kind is PrimitiveKind.ANTIDERIVATIVE and inner.kind is PrimitiveKind.DERIVATIVE:
        if outer.map_type.source == inner.map_type.target and outer.map_type.target == inner.map_type.source:
            source = inner.map_type.source
            assert isinstance(source, PolynomialCoordinateSpace)
            basepoint = outer.parameters[0]
            defect = compose(constant_injection(source), evaluation(source, basepoint))
            return (
                add(identity(source), scale(Fraction(-1), defect)),
                "basepoint antiderivative derivative defect",
                "EXPR.REWRITE.JD_DEFECT",
            )

    if outer.kind is PrimitiveKind.INTERVAL_INTEGRATION and inner.kind is PrimitiveKind.DERIVATIVE:
        if outer.map_type.source == inner.map_type.target:
            source = inner.map_type.source
            assert isinstance(source, PolynomialCoordinateSpace)
            lower, upper, density = outer.parameters
            return (
                boundary_functional(source, lower, upper, density),
                "density-weighted fundamental theorem row",
                "EXPR.REWRITE.FTC",
            )

    if inner.kind is PrimitiveKind.COORDINATE_CHANGE and outer.kind in {
        PrimitiveKind.EVALUATION,
        PrimitiveKind.INTERVAL_INTEGRATION,
        PrimitiveKind.BOUNDARY_FUNCTIONAL,
    }:
        if outer.map_type.source == inner.map_type.target:
            source = inner.map_type.source
            assert isinstance(source, PolynomialCoordinateSpace)
            if outer.kind is PrimitiveKind.EVALUATION:
                candidate = evaluation(source, outer.parameters[0])
                label = "evaluation coordinate transport"
                theorem = "EXPR.REWRITE.EVALUATION_TRANSPORT"
            elif outer.kind is PrimitiveKind.INTERVAL_INTEGRATION:
                candidate = interval_integration(source, *outer.parameters)
                label = "integration coordinate transport"
                theorem = "EXPR.REWRITE.INTEGRATION_TRANSPORT"
            else:
                candidate = boundary_functional(source, *outer.parameters)
                label = "boundary coordinate transport"
                theorem = "EXPR.REWRITE.BOUNDARY_TRANSPORT"
            return candidate, label, theorem
    return None


def _local_rewrite(node: Expression) -> tuple[Expression, str, str] | None:
    if isinstance(node, Scale):
        if node.scalar == 0:
            return _zero_expression(node.expression_type), "zero scaling", "EXPR.REWRITE.SCALE_ZERO"
        if node.scalar == 1:
            return node.value, "unit scaling", "EXPR.REWRITE.SCALE_ONE"
        if isinstance(node.value, Scale):
            return scale(node.scalar * node.value.scalar, node.value.value), "combine scales", "EXPR.REWRITE.SCALE_ASSOC"

    if isinstance(node, Sum):
        terms = _flatten_sum(node)
        retained = tuple(
            term
            for term in terms
            if not isinstance(term, ZeroMap)
            and not (
                isinstance(term, ExactValueLiteral)
                and all(value == 0 for value in term.entries)
            )
        )
        if not retained:
            return _zero_expression(node.expression_type), "remove additive zeros", "EXPR.REWRITE.ADD_ZERO"
        if len(retained) == 1:
            return retained[0], "remove additive zero", "EXPR.REWRITE.ADD_ZERO"
        if retained != node.terms:
            return add(*retained), "flatten sum and remove zeros", "EXPR.REWRITE.ADD_ASSOC"

    if isinstance(node, Compose):
        factors = _flatten_compose(node)
        if any(isinstance(factor, ZeroMap) for factor in factors):
            return zero_map(node.expression_type), "zero-map absorption", "EXPR.REWRITE.COMPOSE_ZERO"
        retained = tuple(factor for factor in factors if not isinstance(factor, IdentityMap))
        if not retained:
            return identity(node.expression_type.source), "identity composition", "EXPR.REWRITE.COMPOSE_IDENTITY"
        if len(retained) == 1:
            return retained[0], "remove identity map", "EXPR.REWRITE.COMPOSE_IDENTITY"
        if retained != node.factors:
            return compose(*retained), "flatten composition and remove identities", "EXPR.REWRITE.COMPOSE_ASSOC"
        for index in range(len(factors) - 1):
            replacement = _primitive_pair_rewrite(factors[index], factors[index + 1])
            if replacement is not None:
                candidate, label, theorem = replacement
                combined = (*factors[:index], candidate, *factors[index + 2 :])
                return (combined[0] if len(combined) == 1 else compose(*combined), label, theorem)

    if isinstance(node, Apply):
        if isinstance(node.operator, IdentityMap):
            return node.argument, "identity application", "EXPR.REWRITE.APPLY_IDENTITY"
        if isinstance(node.operator, ZeroMap):
            return _zero_expression(node.expression_type), "zero-map application", "EXPR.REWRITE.APPLY_ZERO"
        if isinstance(node.operator, ExactMapLiteral) and isinstance(node.argument, ExactValueLiteral):
            value = matrix_vector_multiply(node.operator.matrix, node.argument.entries)
            return exact_value(node.expression_type, value), "literal application", "EXPR.REWRITE.APPLY_LITERAL"

    if isinstance(node, PolynomialPrimitive) and node.kind is PrimitiveKind.COORDINATE_CHANGE:
        if node.map_type.source == node.map_type.target:
            return identity(node.map_type.source), "reflexive coordinate change", "EXPR.REWRITE.COORDINATE_IDENTITY"
    return None


def _rebuild_with_simplified_children(
    node: Expression,
    bindings: Mapping[str | Symbol, ExpressionBinding] | None,
    budget: ExpressionBudget,
    steps: list[RewriteStep],
    cache: dict[int, tuple[Expression, Expression]],
    work: list[int],
) -> Expression:
    if isinstance(node, Let):
        # Keep lexical scopes opaque here.  Replaying an isolated body subterm
        # would incorrectly treat identity-bound occurrences as free symbols.
        # The whole Let is still normalized before and after simplification.
        return node
    if isinstance(node, Compose):
        return compose(
            *(
                _simplify_node(child, bindings, budget, steps, cache, work)
                for child in node.factors
            )
        )
    if isinstance(node, Sum):
        return add(
            *(
                _simplify_node(child, bindings, budget, steps, cache, work)
                for child in node.terms
            )
        )
    if isinstance(node, Scale):
        return scale(
            node.scalar,
            _simplify_node(node.value, bindings, budget, steps, cache, work),
        )
    if isinstance(node, Apply):
        return Apply(
            _simplify_node(node.operator, bindings, budget, steps, cache, work),
            _simplify_node(node.argument, bindings, budget, steps, cache, work),
        )
    return node


def _expression_digest(expression: Expression, budget: ExpressionBudget) -> str:
    from .portable import expression_digest

    return expression_digest(expression, budget=budget)


def _simplify_node(
    node: Expression,
    bindings: Mapping[str | Symbol, ExpressionBinding] | None,
    budget: ExpressionBudget,
    steps: list[RewriteStep],
    cache: dict[int, tuple[Expression, Expression]],
    work: list[int],
) -> Expression:
    work[0] += 1
    if work[0] > budget.max_steps:
        raise ResourceLimitError("expression simplification exceeds max_steps")
    cached = cache.get(id(node))
    if cached is not None and cached[0] is node:
        return cached[1]
    current = _rebuild_with_simplified_children(
        node,
        bindings,
        budget,
        steps,
        cache,
        work,
    )
    for _ in range(256):
        rewrite = _local_rewrite(current)
        if rewrite is None:
            cache[id(node)] = (node, current)
            return current
        candidate, rule, theorem_id = rewrite
        budget.preflight(candidate)
        before_form = normalize_expression(current, bindings, budget=budget)
        after_form = normalize_expression(candidate, bindings, budget=budget)
        if before_form.expression_type != after_form.expression_type or before_form.data != after_form.data:
            raise ValidationError(f"rewrite {rule!r} failed exact denotation replay")
        certificate = compose_certificates(
            (before_form.certificate, after_form.certificate),
            operation=f"certified rewrite: {rule}",
            theorem_id=theorem_id,
            assumptions_checked=("source and target types agree", "exact normal forms agree entrywise"),
            diagnostics={"rule": rule},
        )
        if len(steps) >= budget.max_steps:
            raise ResourceLimitError("expression rewrite count exceeds max_steps")
        steps.append(
            RewriteStep(
                rule,
                theorem_id,
                _expression_digest(current, budget),
                _expression_digest(candidate, budget),
                ("source and target types agree", "exact normal forms agree entrywise"),
                certificate,
            )
        )
        current = _rebuild_with_simplified_children(
            candidate,
            bindings,
            budget,
            steps,
            cache,
            work,
        )
    raise ResourceLimitError("local simplification exceeded its 256-step convergence guard")


@dataclass(frozen=True, slots=True)
class SimplificationResult:
    expression: Expression
    before: NormalForm
    after: NormalForm
    steps: tuple[RewriteStep, ...]
    certificate: Certificate


def simplify_expression(
    expression: Expression,
    bindings: Mapping[str | Symbol, ExpressionBinding] | None = None,
    *,
    budget: ExpressionBudget | None = None,
) -> SimplificationResult:
    """Apply conservative rewrites, replaying every step against exact data."""

    selected_budget = ExpressionBudget() if budget is None else budget
    before = normalize_expression(expression, bindings, budget=selected_budget)
    steps: list[RewriteStep] = []
    try:
        simplified = _simplify_node(
            expression,
            bindings,
            selected_budget,
            steps,
            {},
            [0],
        )
    except RecursionError as exc:
        raise ResourceLimitError("expression depth exceeds the simplifier call-stack guard") from exc
    after = normalize_expression(simplified, bindings, budget=selected_budget)
    if before.expression_type != after.expression_type or before.data != after.data:
        raise ValidationError("simplification changed the exact denotation")
    certificate = compose_certificates(
        (before.certificate, *(step.certificate for step in steps), after.certificate),
        operation="certified finite expression simplification",
        theorem_id="EXPR.REWRITE.DENOTATION",
        diagnostics={"rewrite_count": len(steps), "normal_form_digest": after.digest},
    )
    return SimplificationResult(simplified, before, after, tuple(steps), certificate)


@dataclass(frozen=True, slots=True)
class PlanInstruction:
    opcode: str
    inputs: tuple[int, ...]
    expression_type: CoordinateSpace | LinearMapType
    payload: Any
    scalar_operation_cost: int
    certificate: Certificate

    def __post_init__(self) -> None:
        if type(self.opcode) is not str:
            raise TypeError("plan opcode must be a built-in string")
        if type(self.inputs) is not tuple:
            raise TypeError("plan inputs must be a tuple")
        if self.opcode not in {"constant", "add", "scale", "matrix_chain", "apply"}:
            raise ValidationError(f"unsupported plan opcode: {self.opcode!r}")
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in self.inputs):
            raise ValidationError("plan inputs must be nonnegative integer indices")
        if (
            isinstance(self.scalar_operation_cost, bool)
            or not isinstance(self.scalar_operation_cost, int)
            or self.scalar_operation_cost < 0
        ):
            raise ValidationError("plan scalar-operation cost must be a nonnegative integer")
        if not isinstance(self.certificate, Certificate):
            raise TypeError("plan instruction certificate must be a Certificate")
        _validate_expression_type(
            self.expression_type,
            name="plan instruction expression type",
        )


def _plan_payload(instruction: PlanInstruction) -> Any:
    if instruction.opcode == "constant":
        payload = _data_payload(instruction.payload, instruction.expression_type)
    elif instruction.opcode == "scale":
        payload = _fraction_token(instruction.payload)
    elif instruction.opcode == "matrix_chain":
        plan = instruction.payload
        if type(plan) is not MatrixChainPlan:
            raise TypeError("matrix_chain payload must be a MatrixChainPlan")
        payload = {
            "dimensions": list(plan.dimensions),
            "cost": plan.scalar_operation_cost,
            "tree": [plan.tree.start, plan.tree.end, plan.tree.split],
        }
    else:
        payload = None
    return {
        "certificate": {
            "assumptions_checked": list(instruction.certificate.assumptions_checked),
            "assumptions_unchecked": list(instruction.certificate.assumptions_unchecked),
            "exactness": instruction.certificate.exactness.value,
            "residuals": dict(sorted(instruction.certificate.residuals.items())),
            "theorem_id": instruction.certificate.theorem_id,
            "tolerance": instruction.certificate.tolerance,
            "valid": instruction.certificate.valid,
        },
        "cost": instruction.scalar_operation_cost,
        "inputs": list(instruction.inputs),
        "opcode": instruction.opcode,
        "payload": payload,
        "type": _type_payload(instruction.expression_type),
    }


def _plan_digest(
    instructions: Sequence[PlanInstruction],
    output_index: int,
    expression_type: Any,
    expression_digest: str,
) -> str:
    payload = {
        "expression_digest": expression_digest,
        "instructions": [_plan_payload(instruction) for instruction in instructions],
        "output_index": output_index,
        "type": _type_payload(expression_type),
        "version": "komorebi_math.expression.execution_plan.v1",
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ExpressionPlan:
    expression_digest: str
    expression_type: CoordinateSpace | LinearMapType
    instructions: tuple[PlanInstruction, ...]
    output_index: int
    scalar_operation_cost: int
    plan_digest: str
    certificate: Certificate

    def __post_init__(self) -> None:
        _validate_expression_type(self.expression_type, name="plan expression type")
        if type(self.instructions) is not tuple:
            raise TypeError("plan instructions must be a tuple")
        if not isinstance(self.expression_digest, str) or len(self.expression_digest) != 64:
            raise ValidationError("expression_digest must be a SHA-256 hexadecimal digest")
        try:
            int(self.expression_digest, 16)
        except ValueError as exc:
            raise ValidationError("expression_digest must be hexadecimal") from exc
        if not self.instructions:
            raise ValidationError("an expression plan requires at least one instruction")
        if (
            isinstance(self.output_index, bool)
            or not isinstance(self.output_index, int)
            or not 0 <= self.output_index < len(self.instructions)
        ):
            raise ValidationError("plan output index is outside the instruction table")
        for index, instruction in enumerate(self.instructions):
            if type(instruction) is not PlanInstruction:
                raise TypeError("instructions must be PlanInstruction records")
            instruction.__post_init__()
            if any(source >= index for source in instruction.inputs):
                raise ValidationError("plan instructions must reference earlier outputs")
            input_types = tuple(self.instructions[source].expression_type for source in instruction.inputs)
            if instruction.opcode == "constant":
                if instruction.inputs or instruction.scalar_operation_cost != 0:
                    raise ValidationError("constant instructions require no inputs and zero cost")
                _data_for_type(
                    instruction.payload,
                    instruction.expression_type,
                    name="plan constant",
                )
            elif instruction.opcode == "add":
                if len(input_types) != 2 or any(
                    item != instruction.expression_type for item in input_types
                ):
                    raise ValidationError("add instruction types do not agree")
                expected = (
                    instruction.expression_type.source.dimension
                    * instruction.expression_type.target.dimension
                    if isinstance(instruction.expression_type, LinearMapType)
                    else instruction.expression_type.dimension
                )
                if instruction.payload is not None or instruction.scalar_operation_cost != expected:
                    raise ValidationError("add instruction payload or cost is inconsistent")
            elif instruction.opcode == "scale":
                if len(input_types) != 1 or input_types[0] != instruction.expression_type:
                    raise ValidationError("scale instruction type does not match its input")
                _fraction(instruction.payload, name="plan scale")
                expected = (
                    instruction.expression_type.source.dimension
                    * instruction.expression_type.target.dimension
                    if isinstance(instruction.expression_type, LinearMapType)
                    else instruction.expression_type.dimension
                )
                if instruction.scalar_operation_cost != expected:
                    raise ValidationError("scale instruction cost is inconsistent")
            elif instruction.opcode == "apply":
                if len(input_types) != 2 or not isinstance(input_types[0], LinearMapType):
                    raise ValidationError("apply instruction requires a map and a value")
                operator_type = input_types[0]
                if (
                    input_types[1] != operator_type.source
                    or instruction.expression_type != operator_type.target
                    or instruction.payload is not None
                    or instruction.scalar_operation_cost
                    != operator_type.target.dimension * operator_type.source.dimension
                ):
                    raise ValidationError("apply instruction type or cost is inconsistent")
            elif instruction.opcode == "matrix_chain":
                if len(input_types) < 2 or any(
                    not isinstance(item, LinearMapType) for item in input_types
                ):
                    raise ValidationError("matrix-chain instructions require at least two maps")
                map_types = tuple(item for item in input_types if isinstance(item, LinearMapType))
                if any(
                    inner.target != outer.source
                    for outer, inner in zip(map_types, map_types[1:], strict=False)
                ):
                    raise ValidationError("matrix-chain instruction endpoints do not compose")
                expected_type = LinearMapType(map_types[-1].source, map_types[0].target)
                expected_dimensions = (
                    map_types[0].target.dimension,
                    *(item.source.dimension for item in map_types),
                )
                if (
                    not isinstance(instruction.payload, MatrixChainPlan)
                    or instruction.payload.dimensions != expected_dimensions
                    or instruction.expression_type != expected_type
                    or instruction.scalar_operation_cost
                    != instruction.payload.scalar_operation_cost
                ):
                    raise ValidationError("matrix-chain instruction plan, type, or cost is inconsistent")
        if (
            isinstance(self.scalar_operation_cost, bool)
            or not isinstance(self.scalar_operation_cost, int)
            or self.scalar_operation_cost < 0
        ):
            raise ValidationError("plan cost must be a nonnegative integer")
        expected_cost = sum(instruction.scalar_operation_cost for instruction in self.instructions)
        if self.scalar_operation_cost != expected_cost:
            raise ValidationError("plan cost does not equal its instruction costs")
        expected_digest = _plan_digest(
            self.instructions,
            self.output_index,
            self.expression_type,
            self.expression_digest,
        )
        if self.plan_digest != expected_digest:
            raise ValidationError("plan digest does not match its instructions")
        if not isinstance(self.certificate, Certificate):
            raise TypeError("plan certificate must be a Certificate")


class _PlanBuilder:
    def __init__(
        self,
        binding_items: tuple[tuple[str | Symbol, ExpressionBinding], ...],
        budget: ExpressionBudget,
    ) -> None:
        self.instructions: list[PlanInstruction] = []
        self.bindings = binding_items
        self.budget = budget
        self.cache: dict[tuple[int, tuple[tuple[int, int], ...]], int] = {}
        self.work = 0

    def emit(
        self,
        opcode: str,
        inputs: tuple[int, ...],
        expression_type: Any,
        payload: Any,
        cost: int,
        certificate: Certificate,
    ) -> int:
        if len(self.instructions) >= self.budget.max_steps:
            raise ResourceLimitError("expression plan instruction count exceeds max_steps")
        if opcode == "constant":
            checked_payload = _data_for_type(
                payload,
                expression_type,
                name="plan constant payload",
            )
            _validate_data_budget(
                checked_payload,
                self.budget,
                name="plan constant payload",
            )
        index = len(self.instructions)
        self.instructions.append(PlanInstruction(opcode, inputs, expression_type, payload, cost, certificate))
        return index

    def compile(self, node: Expression, local: tuple[tuple[Symbol, int], ...] = ()) -> int:
        self.work += 1
        if self.work > self.budget.max_steps:
            raise ResourceLimitError("expression plan compilation exceeds max_steps")
        key = (id(node), tuple((id(symbol), index) for symbol, index in local))
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        result = self._compile_uncached(node, local)
        self.cache[key] = result
        return result

    def _compile_uncached(
        self,
        node: Expression,
        local: tuple[tuple[Symbol, int], ...] = (),
    ) -> int:
        if isinstance(node, ExactValueLiteral):
            return self.emit("constant", (), node.expression_type, node.entries, 0, _literal_certificate("plan value literal"))
        if isinstance(node, ExactMapLiteral):
            return self.emit("constant", (), node.expression_type, node.matrix, 0, _literal_certificate("plan map literal"))
        if isinstance(node, PolynomialPrimitive):
            return self.emit("constant", (), node.expression_type, replay(node), 0, primitive_certificate(node))
        if isinstance(node, IdentityMap):
            return self.emit("constant", (), node.expression_type, identity_matrix(node.space.dimension), 0, _literal_certificate("plan identity"))
        if isinstance(node, ZeroMap):
            zero_data = zero_matrix(node.map_type.target.dimension, node.map_type.source.dimension)
            return self.emit("constant", (), node.expression_type, zero_data, 0, _literal_certificate("plan zero map"))
        if isinstance(node, Symbol):
            for binder, index in reversed(local):
                if node is binder:
                    return index
            binding = _lookup_binding(node, self.bindings)
            if binding is None:
                raise ValidationError(f"free symbol {node.name!r} has no value binding")
            binding_data = _data_for_type(
                binding.value,
                binding.declared_type,
                name=f"binding {node.name!r}",
            )
            return self.emit(
                "constant",
                (),
                node.expression_type,
                binding_data,
                0,
                binding.certificate,
            )
        if isinstance(node, Let):
            value_index = self.compile(node.value, local)
            return self.compile(
                node.body,
                (*local, (node.symbol, value_index)),
            )
        if isinstance(node, Sum):
            current = self.compile(node.terms[0], local)
            size = (
                node.expression_type.source.dimension * node.expression_type.target.dimension
                if isinstance(node.expression_type, LinearMapType)
                else node.expression_type.dimension
            )
            for term in node.terms[1:]:
                other = self.compile(term, local)
                certificate = compose_certificates(
                    (self.instructions[current].certificate, self.instructions[other].certificate),
                    operation="planned exact addition",
                    theorem_id="EXPR.PLAN.ADD",
                )
                current = self.emit("add", (current, other), node.expression_type, None, size, certificate)
            return current
        if isinstance(node, Scale):
            source = self.compile(node.value, local)
            size = (
                node.expression_type.source.dimension * node.expression_type.target.dimension
                if isinstance(node.expression_type, LinearMapType)
                else node.expression_type.dimension
            )
            certificate = compose_certificates(
                (self.instructions[source].certificate,),
                operation="planned exact scale",
                theorem_id="EXPR.PLAN.SCALE",
            )
            return self.emit("scale", (source,), node.expression_type, node.scalar, size, certificate)
        if isinstance(node, Apply):
            operator = self.compile(node.operator, local)
            argument = self.compile(node.argument, local)
            operator_type = node.operator.expression_type
            assert isinstance(operator_type, LinearMapType)
            cost = operator_type.target.dimension * operator_type.source.dimension
            certificate = compose_certificates(
                (self.instructions[operator].certificate, self.instructions[argument].certificate),
                operation="planned exact application",
                theorem_id="EXPR.PLAN.APPLY",
            )
            return self.emit("apply", (operator, argument), node.expression_type, None, cost, certificate)
        if isinstance(node, Compose):
            factors = _flatten_compose(node)
            inputs = tuple(self.compile(factor, local) for factor in factors)
            unchecked_map_types = tuple(factor.expression_type for factor in factors)
            assert all(isinstance(item, LinearMapType) for item in unchecked_map_types)
            map_types = cast(tuple[LinearMapType, ...], unchecked_map_types)
            dimensions = (map_types[0].target.dimension, *(item.source.dimension for item in map_types))
            matrix_count = len(map_types)
            planning_work = matrix_count * (matrix_count - 1) * (matrix_count + 1) // 6
            if planning_work > self.budget.max_steps:
                raise ResourceLimitError("matrix-chain dynamic program exceeds max_steps")
            chain = matrix_chain_plan(dimensions)
            certificate = compose_certificates(
                (*(self.instructions[index].certificate for index in inputs), chain.certificate),
                operation="planned exact matrix chain",
                theorem_id="EXPR.PLAN.COMPOSE",
                diagnostics={"dimensions": dimensions, "optimal_cost": chain.scalar_operation_cost},
            )
            return self.emit(
                "matrix_chain",
                inputs,
                node.expression_type,
                chain,
                chain.scalar_operation_cost,
                certificate,
            )
        raise TypeError(f"unsupported expression node: {type(node).__name__}")


def execute_plan(
    plan: ExpressionPlan,
    *,
    budget: ExpressionBudget | None = None,
) -> NormalForm:
    """Replay a self-contained exact plan after validating its content digest."""

    if type(plan) is not ExpressionPlan:
        raise TypeError("plan must be an ExpressionPlan")
    selected_budget = ExpressionBudget() if budget is None else budget
    if type(selected_budget) is not ExpressionBudget:
        raise TypeError("budget must be an ExpressionBudget")
    plan.__post_init__()
    if len(plan.instructions) > selected_budget.max_steps:
        raise ResourceLimitError("plan instruction count exceeds max_steps")
    if (
        _plan_digest(
            plan.instructions,
            plan.output_index,
            plan.expression_type,
            plan.expression_digest,
        )
        != plan.plan_digest
    ):
        raise ValidationError("plan content digest failed replay")
    outputs: list[ExactData] = []
    certificates: list[Certificate] = []
    for index, instruction in enumerate(plan.instructions):
        if any(source >= index for source in instruction.inputs):
            raise ValidationError("plan dependency is not topologically ordered")
        if instruction.opcode == "constant":
            data = _data_for_type(instruction.payload, instruction.expression_type, name="plan constant")
        elif instruction.opcode == "add":
            if len(instruction.inputs) != 2:
                raise ValidationError("add instructions require two inputs")
            data = _add_data(outputs[instruction.inputs[0]], outputs[instruction.inputs[1]], instruction.expression_type)
        elif instruction.opcode == "scale":
            if len(instruction.inputs) != 1:
                raise ValidationError("scale instructions require one input")
            scalar = _fraction(instruction.payload, name="plan scale")
            data = _scale_data(scalar, outputs[instruction.inputs[0]])
        elif instruction.opcode == "matrix_chain":
            if not instruction.inputs:
                raise ValidationError("matrix-chain instructions require inputs")
            data = matrix_chain_multiply(
                tuple(cast(Matrix, outputs[source]) for source in instruction.inputs),
                instruction.payload,
            )
        elif instruction.opcode == "apply":
            if len(instruction.inputs) != 2:
                raise ValidationError("apply instructions require two inputs")
            data = matrix_vector_multiply(
                cast(Matrix, outputs[instruction.inputs[0]]),
                cast(Vector, outputs[instruction.inputs[1]]),
            )
        else:  # guarded by PlanInstruction, repeated for tamper resistance
            raise ValidationError(f"unsupported plan opcode: {instruction.opcode!r}")
        checked = _data_for_type(data, instruction.expression_type, name="plan instruction result")
        _validate_data_budget(
            checked,
            selected_budget,
            name="plan instruction result",
        )
        outputs.append(checked)
        certificates.append(instruction.certificate)
    result = outputs[plan.output_index]
    certificate = compose_certificates(
        (*certificates, plan.certificate),
        operation="exact expression plan execution",
        theorem_id="EXPR.PLAN.EXECUTE",
        diagnostics={"plan_digest": plan.plan_digest, "instruction_count": len(plan.instructions)},
    )
    final = _normal_form(
        result,
        plan.expression_type,
        certificate,
        budget=selected_budget,
    )
    _validate_data_budget(final.data, selected_budget, name="plan result")
    return final


def compile_expression(
    expression: Expression,
    bindings: Mapping[str | Symbol, ExpressionBinding] | None = None,
    *,
    budget: ExpressionBudget | None = None,
) -> ExpressionPlan:
    """Compile and independently replay a self-contained exact execution plan."""

    selected_budget = ExpressionBudget() if budget is None else budget
    checked = check_expression(expression, bindings, budget=selected_budget)
    direct = normalize_expression(expression, bindings, budget=selected_budget)
    builder = _PlanBuilder(_binding_items(bindings), selected_budget)
    try:
        output = builder.compile(expression)
    except RecursionError as exc:
        raise ResourceLimitError("expression depth exceeds the planner call-stack guard") from exc
    instructions = tuple(builder.instructions)
    source_digest = _expression_digest(expression, selected_budget)
    digest = _plan_digest(instructions, output, expression.expression_type, source_digest)
    certificate = compose_certificates(
        (
            checked.certificate,
            direct.certificate,
            *(instruction.certificate for instruction in instructions),
        ),
        operation="typed finite expression compilation",
        theorem_id="EXPR.PLAN.DENOTATION",
        diagnostics={
            "expression_digest": source_digest,
            "expected_normal_form_digest": direct.digest,
            "plan_digest": digest,
        },
    )
    plan = ExpressionPlan(
        source_digest,
        expression.expression_type,
        instructions,
        output,
        sum(instruction.scalar_operation_cost for instruction in instructions),
        digest,
        certificate,
    )
    executed = execute_plan(plan, budget=selected_budget)
    if executed.expression_type != direct.expression_type or executed.data != direct.data:
        raise ValidationError("compiled plan does not match independent direct normalization")
    return plan


@dataclass(frozen=True, slots=True)
class ExpressionResult:
    normal_form: NormalForm
    direct_normal_form: NormalForm
    plan: ExpressionPlan
    certificate: Certificate

    @property
    def value(self) -> ExactData:
        return self.normal_form.data

    @property
    def scalar(self) -> Fraction:
        if not isinstance(self.normal_form.expression_type, RationalSpace):
            raise DimensionError("only rational-space results have a scalar projection")
        return self.normal_form.data[0]

    def require(self) -> ExactData:
        self.certificate.require()
        return self.value


def evaluate_expression(
    expression: Expression,
    bindings: Mapping[str | Symbol, ExpressionBinding] | None = None,
    *,
    budget: ExpressionBudget | None = None,
) -> ExpressionResult:
    """Normalize directly, compile, execute, and compare both exact paths."""

    direct = normalize_expression(expression, bindings, budget=budget)
    plan = compile_expression(expression, bindings, budget=budget)
    selected_budget = ExpressionBudget() if budget is None else budget
    executed = execute_plan(plan, budget=selected_budget)
    same = direct.expression_type == executed.expression_type and direct.data == executed.data
    certificate = compose_certificates(
        (direct.certificate, plan.certificate, executed.certificate),
        operation="dual-path exact expression evaluation",
        theorem_id="EXPR.EVALUATE.DUAL_PATH",
        diagnostics={
            "direct_digest": direct.digest,
            "executed_digest": executed.digest,
            "entrywise_match": same,
        },
    )
    if not same:
        raise ValidationError("planned execution disagrees with direct exact normalization")
    return ExpressionResult(executed, direct, plan, certificate)


__all__ = [
    "CheckedExpression",
    "EquivalenceResult",
    "ExpressionPlan",
    "ExpressionResult",
    "PlanInstruction",
    "SimplificationResult",
    "check_expression",
    "compile_expression",
    "equivalent",
    "evaluate_expression",
    "exact_binding",
    "execute_plan",
    "normal_form_digest",
    "normalize_expression",
    "simplify_expression",
]
