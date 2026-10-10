"""Exact three-input Boolean synthesis from AVOC Sections 6.1--6.4.

The grammar is a *tree*: two literals, variables x1/x2/x3, NOT, AND, OR,
and XOR.  Both binary children are eagerly visited.  A signature's bit i
is its value at the i-th input in lexicographic order (000, ..., 111).
The tariff counts tree nodes, including repeated occurrences of a shared
Python object.  It does not describe short-circuit execution or circuit DAGs.

A global minimum is certified by checking all 256 witnesses and *every*
constructor inequality, independently of the synthesis search.  This is
the finite algebra certificate of AVOC Theorem 6.4, not a finite-depth claim
mislabelled as unrestricted optimality.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice, product

from ..certificates import Certificate, Exactness
from ..exceptions import ValidationError
from ._validation import strict_integer

_INPUTS = tuple(product((False, True), repeat=3))
_BINARY = ("and", "or", "xor")
_MASK = 255


@dataclass(frozen=True, slots=True)
class BooleanExpr:
    """An immutable, validated AST in the source's precise Boolean grammar."""

    operation: str
    children: tuple[BooleanExpr, ...] = ()
    atom: bool | int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.children, (tuple, list)):
            raise ValidationError("children must be a finite tuple or list")
        if len(self.children) > 2:
            raise ValidationError("the Boolean grammar permits at most two children")
        object.__setattr__(self, "children", tuple(self.children))
        if any(not isinstance(child, BooleanExpr) for child in self.children):
            raise ValidationError("children must be BooleanExpr instances")
        if self.operation == "literal":
            if self.children or type(self.atom) is not bool:
                raise ValidationError("a literal has a boolean atom and no children")
        elif self.operation == "variable":
            if self.children:
                raise ValidationError("a variable has no children")
            object.__setattr__(self, "atom", strict_integer(self.atom, "variable", minimum=1, maximum=3))
        elif self.operation in ("not", *_BINARY):
            arity = 1 if self.operation == "not" else 2
            if len(self.children) != arity or self.atom is not None:
                raise ValidationError("constructor arity/atom does not match the grammar")
        else:
            raise ValidationError("unknown Boolean constructor")

    @staticmethod
    def literal(value: bool) -> BooleanExpr:
        return BooleanExpr("literal", atom=value)

    @staticmethod
    def variable(index: int) -> BooleanExpr:
        """Return x1, x2, or x3 (indices are one based)."""
        return BooleanExpr("variable", atom=index)

    @staticmethod
    def negate(child: BooleanExpr) -> BooleanExpr:
        return BooleanExpr("not", (child,))

    @staticmethod
    def binary(operation: str, left: BooleanExpr, right: BooleanExpr) -> BooleanExpr:
        if operation not in _BINARY:
            raise ValidationError("binary operation must be and, or, or xor")
        return BooleanExpr(operation, (left, right))


@dataclass(frozen=True, slots=True)
class BooleanEvaluation:
    value: bool
    visits: int


def _limits(max_nodes: int, max_depth: int) -> tuple[int, int]:
    return (
        strict_integer(max_nodes, "max_nodes", minimum=1),
        strict_integer(max_depth, "max_depth", minimum=0, maximum=256),
    )


def boolean_node_count(expression: BooleanExpr, *, max_nodes: int = 10000, max_depth: int = 256) -> int:
    """Count occurrences, rejecting a tree beyond the explicit audit limits."""
    limit, depth_limit = _limits(max_nodes, max_depth)
    if not isinstance(expression, BooleanExpr):
        raise ValidationError("expression must be a BooleanExpr")
    pending = [(expression, 0)]
    count = 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if count > limit or depth > depth_limit:
            raise ValidationError("expression exceeds its node/depth resource limit")
        pending.extend((child, depth + 1) for child in node.children)
    return count


def evaluate_boolean(
    expression: BooleanExpr,
    inputs: Iterable[bool | int],
    *,
    max_nodes: int = 10000,
    max_depth: int = 256,
) -> BooleanEvaluation:
    """Evaluate both binary children, with an exact eager visit meter."""
    limit, depth_limit = _limits(max_nodes, max_depth)
    if not isinstance(expression, BooleanExpr):
        raise ValidationError("expression must be a BooleanExpr")
    try:
        values = tuple(islice(inputs, 4))
    except TypeError as exc:
        raise ValidationError("inputs must be three Boolean bits") from exc
    if len(values) != 3 or any(type(value) not in (bool, int) or value not in (0, 1) for value in values):
        raise ValidationError("inputs must be three Boolean bits")
    pending = [(expression, 0, False)]
    results: list[bool] = []
    visits = 0
    while pending:
        node, depth, finish = pending.pop()
        if not finish:
            visits += 1
            if visits > limit or depth > depth_limit:
                raise ValidationError("expression exceeds its node/depth resource limit")
            if node.operation == "literal":
                results.append(bool(node.atom))
            elif node.operation == "variable":
                results.append(bool(values[int(node.atom) - 1]))
            else:
                pending.append((node, depth, True))
                pending.extend((child, depth + 1, False) for child in reversed(node.children))
        elif node.operation == "not":
            results.append(not results.pop())
        else:
            right, left = results.pop(), results.pop()
            # Child evaluations have already finished before any logical operation.
            if node.operation == "and":
                results.append(left and right)
            elif node.operation == "or":
                results.append(left or right)
            else:
                results.append(left != right)
    return BooleanEvaluation(results[0], visits)


def boolean_signature(expression: BooleanExpr, *, max_nodes: int = 10000) -> int:
    """All eight ordinary outputs, in the fixed lexicographic input order."""
    return sum(
        int(evaluate_boolean(expression, inputs, max_nodes=max_nodes).value) << index
        for index, inputs in enumerate(_INPUTS)
    )


def boolean_equivalent(left: BooleanExpr, right: BooleanExpr, *, max_nodes: int = 10000) -> bool:
    """Sound and complete equivalence checking on the eight-input domain."""
    return boolean_signature(left, max_nodes=max_nodes) == boolean_signature(right, max_nodes=max_nodes)


def _atoms() -> tuple[BooleanExpr, ...]:
    return (
        BooleanExpr.literal(False),
        BooleanExpr.literal(True),
        *(BooleanExpr.variable(index) for index in range(1, 4)),
    )


def _layer_counts(depth: int) -> tuple[int, int, int]:
    # The full syntax layer has doubly exponential entry count; cap arithmetic
    # depth as well as actual materialization.  No candidate allocation occurs here.
    level = strict_integer(depth, "depth", minimum=0, maximum=8)
    count, total, bound = 5, 5, 1
    for _ in range(level):
        count = 2 * count + 3 * count * count
        total += count
        bound = 2 * bound + 1
    return count, total, bound


def boolean_grammar_layer(depth: int, *, max_entries: int = 100000) -> tuple[BooleanExpr, ...]:
    """The complete duplicate-preserving source grammar layer, for depth <= 8.

    Emission order is retained expressions, negations, then AND/OR/XOR ordered
    pairs.  The entry bound is checked before any layer is materialized.
    """
    limit = strict_integer(max_entries, "max_entries", minimum=1)
    count, _, _ = _layer_counts(depth)
    if count > limit:
        raise ValidationError("complete grammar layer exceeds max_entries")
    layer = _atoms()
    for _ in range(depth):
        layer = (
            *layer,
            *(BooleanExpr.negate(child) for child in layer),
            *(
                BooleanExpr.binary(operation, left, right)
                for operation in _BINARY
                for left in layer
                for right in layer
            ),
        )
    return layer


@dataclass(frozen=True, slots=True)
class BooleanSearchResult:
    expression: BooleanExpr
    reference_nodes: int
    selected_nodes: int
    depth: int
    examined: int
    layer_entries: int
    generation_charge: int
    comparison_tariff: int
    modeled_charge: int
    complete_layer: bool
    certificate: Certificate


def bounded_boolean_search(
    reference: BooleanExpr,
    depth: int,
    *,
    budget: int,
    trial_cap: int,
    max_entries: int = 100000,
    max_nodes: int = 10000,
) -> BooleanSearchResult:
    """Reserve generation, then examine the affordable capped syntax prefix.

    Generation charge is sum(M_i, i=0..depth).  Each comparison is charged
    8*(N(reference)+B_depth+1), regardless of when a mismatch is noticed.
    If generation does not fit, return the reference with zero modeled charge.
    """
    allowance = strict_integer(budget, "budget", minimum=0)
    trial_limit = strict_integer(trial_cap, "trial_cap", minimum=0)
    entry_limit = strict_integer(max_entries, "max_entries", minimum=1)
    count, generation, bound = _layer_counts(depth)
    reference_nodes = boolean_node_count(reference, max_nodes=max_nodes)
    tariff = 8 * (reference_nodes + bound + 1)
    best, best_nodes, examined = reference, reference_nodes, 0
    charged_generation = 0
    if generation <= allowance:
        layer = boolean_grammar_layer(depth, max_entries=entry_limit)
        charged_generation = generation
        trials = min(count, trial_limit, (allowance - generation) // tariff)
        target = boolean_signature(reference, max_nodes=max_nodes)
        for candidate in layer[:trials]:
            nodes = boolean_node_count(candidate, max_nodes=max_nodes)
            # Even inferior candidates are semantic comparisons in the source model.
            if boolean_signature(candidate, max_nodes=max_nodes) == target and nodes < best_nodes:
                best, best_nodes = candidate, nodes
            examined += 1
    charge = charged_generation + examined * tariff
    complete = examined == count
    certificate = Certificate(
        valid=boolean_equivalent(reference, best, max_nodes=max_nodes)
        and best_nodes <= reference_nodes
        and charge <= allowance,
        exactness=Exactness.EXACT,
        assumptions_checked=(
            "all eight input valuations",
            "eager tree-node tariff",
            "reserved generation budget",
        ),
        diagnostics={
            "depth": int(depth),
            "examined": examined,
            "layer_entries": count,
            "complete_layer": complete,
            "modeled_charge": charge,
            "optimality_scope": "complete height-bounded layer"
            if complete
            else "examined prefix with reference fallback",
        },
        theorem_id="AVOC.6.1-6.2",
        message="Equivalent incumbent; prefix optimum and declared budget bound.",
    )
    return BooleanSearchResult(
        best,
        reference_nodes,
        best_nodes,
        int(depth),
        examined,
        count,
        charged_generation,
        tariff,
        charge,
        complete,
        certificate,
    )


@dataclass(frozen=True, slots=True)
class BooleanSemanticEntry:
    signature: int
    expression: BooleanExpr
    node_count: int


@dataclass(frozen=True, slots=True)
class BooleanCompactionResult:
    entries: tuple[BooleanSemanticEntry, ...]
    emitted_entries: int
    total_node_mass: int
    index_probes: int
    depth: int | None = None


def compact_boolean_entries(
    expressions: Iterable[BooleanExpr], *, max_entries: int = 250000, max_nodes: int = 10000
) -> BooleanCompactionResult:
    """Keep the first minimum-node witness for each complete signature.

    index_probes counts one semantic-index lookup per submitted expression;
    this is a declared lookup count, not a hardware/dictionary timing claim.
    """
    limit = strict_integer(max_entries, "max_entries", minimum=1)
    retained: dict[int, BooleanSemanticEntry] = {}
    emitted = mass = 0
    for expression in expressions:
        if emitted >= limit:
            raise ValidationError("submitted compaction entries exceed max_entries")
        nodes = boolean_node_count(expression, max_nodes=max_nodes)
        signature = boolean_signature(expression, max_nodes=max_nodes)
        emitted += 1
        mass += nodes
        incumbent = retained.get(signature)
        if incumbent is None or nodes < incumbent.node_count:
            retained[signature] = BooleanSemanticEntry(signature, expression, nodes)
    return BooleanCompactionResult(tuple(retained[key] for key in sorted(retained)), emitted, mass, emitted)


def _binary_signature(operation: str, left: int, right: int) -> int:
    if operation == "and":
        return left & right
    if operation == "or":
        return left | right
    return left ^ right


def _expand_compacted(
    entries: tuple[BooleanSemanticEntry, ...], max_emitted: int, max_nodes: int
) -> BooleanCompactionResult:
    size, mass = len(entries), sum(entry.node_count for entry in entries)
    emitted = 2 * size + 3 * size * size
    if emitted > max_emitted:
        raise ValidationError("compacted constructor layer exceeds max_emitted")
    retained = {entry.signature: entry for entry in entries}
    for entry in entries:
        signature, cost = entry.signature ^ _MASK, entry.node_count + 1
        if signature not in retained or cost < retained[signature].node_count:
            if cost > max_nodes:
                raise ValidationError("compacted witness exceeds max_witness_nodes")
            retained[signature] = BooleanSemanticEntry(signature, BooleanExpr.negate(entry.expression), cost)
    for operation in _BINARY:
        for left in entries:
            for right in entries:
                signature = _binary_signature(operation, left.signature, right.signature)
                cost = left.node_count + right.node_count + 1
                if signature not in retained or cost < retained[signature].node_count:
                    if cost > max_nodes:
                        raise ValidationError("compacted witness exceeds max_witness_nodes")
                    retained[signature] = BooleanSemanticEntry(
                        signature, BooleanExpr.binary(operation, left.expression, right.expression), cost
                    )
    return BooleanCompactionResult(
        tuple(retained[key] for key in sorted(retained)),
        emitted,
        (2 + 6 * size) * mass + size + 3 * size * size,
        emitted,
    )


def compacted_boolean_layer(
    depth: int, *, max_emitted: int = 250000, max_witness_nodes: int = 10000
) -> BooleanCompactionResult:
    """Recursive minimum representatives with AVOC Proposition 6.3's scope.

    Returns cumulative construction counters over layers 0..depth.  Candidates
    are streamed by exact signature algebra; only improvements allocate ASTs.
    """
    level = strict_integer(depth, "depth", minimum=0, maximum=32)
    emitted_limit = strict_integer(max_emitted, "max_emitted", minimum=5)
    node_limit = strict_integer(max_witness_nodes, "max_witness_nodes", minimum=1)
    result = compact_boolean_entries(_atoms(), max_entries=emitted_limit, max_nodes=node_limit)
    emitted, mass, probes = result.emitted_entries, result.total_node_mass, result.index_probes
    for _ in range(level):
        result = _expand_compacted(result.entries, emitted_limit, node_limit)
        emitted += result.emitted_entries
        mass += result.total_node_mass
        probes += result.index_probes
    return BooleanCompactionResult(result.entries, emitted, mass, probes, level)


@dataclass(frozen=True, slots=True)
class BooleanClosureBank:
    """A *proposed* complete witness bank; acceptance requires fresh auditing."""

    entries: tuple[BooleanSemanticEntry, ...]
    construction_rounds: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.entries, (tuple, list)):
            raise ValidationError("bank entries must be a finite tuple or list")
        if len(self.entries) > 256:
            raise ValidationError("a Boolean closure bank permits at most 256 rows")
        object.__setattr__(self, "entries", tuple(self.entries))


def audit_boolean_closure_bank(bank: BooleanClosureBank, *, max_witness_nodes: int = 10000) -> Certificate:
    """Check all witnesses and all constructor inequalities without synthesis.

    The checker evaluates ASTs at all eight valuations, validates cached costs,
    and checks 5 atom, 256 unary, and 3*256^2 ordered binary inequalities.
    Malformed, incomplete, or forged banks are rejected.  Resource exhaustion
    also rejects acceptance; it never grants a weaker global certificate.
    """
    node_limit = strict_integer(max_witness_nodes, "max_witness_nodes", minimum=1)
    failure = ""
    unary_checks = binary_checks = atom_checks = 0
    costs: list[int] = [0] * 256
    try:
        if not isinstance(bank, BooleanClosureBank) or len(bank.entries) != 256:
            raise ValidationError("bank must contain exactly 256 rows")
        seen: set[int] = set()
        for entry in bank.entries:
            if not isinstance(entry, BooleanSemanticEntry):
                raise ValidationError("bank row must be a BooleanSemanticEntry")
            signature = strict_integer(entry.signature, "signature", minimum=0, maximum=255)
            cost = strict_integer(entry.node_count, "node_count", minimum=1, maximum=node_limit)
            if signature in seen:
                raise ValidationError("duplicate signature in bank")
            seen.add(signature)
            evaluations = tuple(
                evaluate_boolean(entry.expression, inputs, max_nodes=node_limit) for inputs in _INPUTS
            )
            actual_signature = sum(int(value.value) << index for index, value in enumerate(evaluations))
            if actual_signature != signature or any(value.visits != cost for value in evaluations):
                raise ValidationError("witness signature or eager node count does not match its row")
            costs[signature] = cost
        for atom in _atoms():
            atom_checks += 1
            if costs[boolean_signature(atom)] > 1:
                raise ValidationError("atom lower-certificate inequality failed")
        for signature in range(256):
            unary_checks += 1
            if costs[signature ^ 255] > costs[signature] + 1:
                raise ValidationError("negation lower-certificate inequality failed")
        for operation in _BINARY:
            for left in range(256):
                for right in range(256):
                    binary_checks += 1
                    if costs[_binary_signature(operation, left, right)] > costs[left] + costs[right] + 1:
                        raise ValidationError(f"{operation} lower-certificate inequality failed")
    except (ValidationError, TypeError, AttributeError) as exc:
        failure = str(exc)
    return Certificate(
        valid=not failure,
        exactness=Exactness.EXACT,
        assumptions_checked=(
            "three Boolean inputs in lexicographic order",
            "literal/variable/NOT/AND/OR/XOR tree grammar",
            "all witness outputs and eager visit counts",
            "all constructor lower-bound inequalities",
        )
        if not failure
        else (),
        diagnostics={
            "rows": len(bank.entries) if isinstance(bank, BooleanClosureBank) else 0,
            "atom_checks": atom_checks,
            "unary_checks": unary_checks,
            "binary_checks": binary_checks,
            "max_witness_nodes": node_limit,
            "max_attained_nodes": max(costs),
        },
        theorem_id="AVOC.6.4",
        message=failure or "Attained global node minima in the declared three-input tree grammar.",
    )


def build_boolean_closure_bank(
    *, max_rounds: int = 16, max_emitted: int = 250000, max_witness_nodes: int = 10000
) -> BooleanClosureBank:
    """Find a complete closed bank, then run the independent exhaustive audit.

    max_rounds bounds recursive semantic expansion.  max_emitted is a per-layer
    constructor-count bound, not a syntax-depth guarantee.  Failure to reach a
    closed bank raises instead of returning a purported global optimum.
    """
    rounds = strict_integer(max_rounds, "max_rounds", minimum=0, maximum=32)
    emitted_limit = strict_integer(max_emitted, "max_emitted", minimum=5)
    node_limit = strict_integer(max_witness_nodes, "max_witness_nodes", minimum=1)
    entries = compact_boolean_entries(_atoms(), max_entries=emitted_limit, max_nodes=node_limit).entries
    for round_index in range(rounds + 1):
        bank = BooleanClosureBank(entries, round_index)
        if len(entries) == 256 and audit_boolean_closure_bank(bank, max_witness_nodes=node_limit).valid:
            return bank
        if round_index < rounds:
            entries = _expand_compacted(entries, emitted_limit, node_limit).entries
    raise ValidationError("max_rounds exhausted without a complete constructor-closed bank")


def globally_minimal_boolean(
    bank: BooleanClosureBank, reference: BooleanExpr, *, max_witness_nodes: int = 10000
) -> BooleanExpr:
    """Freshly audit a bank and return its minimum witness for the reference."""
    audit_boolean_closure_bank(bank, max_witness_nodes=max_witness_nodes).require()
    signature = boolean_signature(reference, max_nodes=max_witness_nodes)
    return next(entry.expression for entry in bank.entries if entry.signature == signature)


__all__ = [
    "BooleanClosureBank",
    "BooleanCompactionResult",
    "BooleanEvaluation",
    "BooleanExpr",
    "BooleanSearchResult",
    "BooleanSemanticEntry",
    "audit_boolean_closure_bank",
    "boolean_equivalent",
    "boolean_grammar_layer",
    "boolean_node_count",
    "boolean_signature",
    "bounded_boolean_search",
    "build_boolean_closure_bank",
    "compact_boolean_entries",
    "compacted_boolean_layer",
    "evaluate_boolean",
    "globally_minimal_boolean",
]
