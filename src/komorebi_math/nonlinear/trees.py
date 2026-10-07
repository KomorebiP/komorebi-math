"""Ordered rooted-tree expansions for polynomial nonlinear fixed points.

The finite runtime mirrors the Paper III contraction theorem without hiding its
analytic hypotheses.  Tree enumeration is exact and guarded against accidental
combinatorial explosion.  Convergence is certified only when a supplied scalar
majorant maps a declared ball into itself and is contractive there.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import comb, isfinite, sqrt, nextafter, inf
from typing import Any

import numpy as np

from .._residuals import value_audit, finite_tolerance
from .._prerequisites import packet_certificate_dependencies
from ..certificates import Certificate, Exactness
from .._majorants import guard_values, tail_upper, upper_float, rational_record
from fractions import Fraction
from ..core.packets import PacketSystem
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .jets import PacketJetAddress, PacketJetSupport

MultilinearOperation = Callable[[tuple[Any, ...]], Any]


def _normalize_arities(arities: Sequence[int]) -> tuple[int, ...]:
    normalized: list[int] = []
    for raw in arities:
        value = int(raw)
        if isinstance(raw, bool) or value != raw or value < 2:
            raise ValidationError("tree arities must be integers at least two")
        normalized.append(value)
    result = tuple(sorted(set(normalized)))
    if not result:
        raise ValidationError("at least one internal tree arity is required")
    return result


def _vector_length(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        normalized = tuple(int(entry) for entry in shape)
        if len(normalized) == 1:
            return normalized[0]
        if len(normalized) == 2 and 1 in normalized:
            return max(normalized)
    array = np.asarray(value, dtype=object)
    if array.ndim == 1:
        return int(array.shape[0])
    if array.ndim == 2 and 1 in array.shape:
        return int(max(array.shape))
    raise DimensionError("tree values must be vector-like")


def _zero_like(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return np.zeros_like(np.asarray(value))
    try:
        return 0 * value
    except TypeError:
        return np.zeros_like(np.asarray(value))


def _add(left: Any, right: Any) -> Any:
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return np.asarray(left, dtype=object) + np.asarray(right, dtype=object)
    try:
        return left + right
    except TypeError:
        return np.asarray(left, dtype=object) + np.asarray(right, dtype=object)


def _subtract(left: Any, right: Any) -> Any:
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return np.asarray(left, dtype=object) - np.asarray(right, dtype=object)
    try:
        return left - right
    except TypeError:
        return np.asarray(left, dtype=object) - np.asarray(right, dtype=object)


def _norm(value: Any) -> float:
    try:
        array = np.asarray(value, dtype=complex).reshape(-1)
        if not np.all(np.isfinite(array)):
            raise ValidationError("norm input must be finite")
        magnitudes = np.hypot(array.real, array.imag)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            result = float(np.hypot.reduce(magnitudes, initial=0.0))
        if not isfinite(result):
            raise ValidationError("norm is not representable")
        return result
    except (TypeError, ValueError):
        entries = np.asarray(value, dtype=object).reshape(-1)
        squares = [abs(complex(entry)) ** 2 for entry in entries]
        result = float(sum(squares) ** 0.5)
        if not isfinite(result):
            raise ValidationError("norm is not representable")
        return result


@documented(
    ProvenanceRecord(
        identifier="PSC-III-D13",
        title="Finite ordered rooted tree",
        status=ClaimStatus.DEFINITION,
        source="Paper III",
        assumptions=("ordered children", "internal arity at least two"),
        verification_suite="nonlinear.trees",
    )
)
@dataclass(frozen=True, slots=True, order=True)
class OrderedRootedTree:
    """An unlabeled ordered rooted tree.

    A leaf has ``children=()``.  An internal vertex is labeled implicitly by
    its arity, so a polynomial fixed-point problem supplies one operation for
    every arity that can occur.
    """

    children: tuple["OrderedRootedTree", ...] = ()

    def __post_init__(self) -> None:
        children = tuple(self.children)
        if children and len(children) < 2:
            raise ValidationError("an internal nonlinear tree vertex must have arity at least two")
        if any(not isinstance(child, OrderedRootedTree) for child in children):
            raise ValidationError("tree children must be OrderedRootedTree instances")
        object.__setattr__(self, "children", children)

    @classmethod
    def leaf(cls) -> "OrderedRootedTree":
        return cls(())

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def arity(self) -> int:
        return len(self.children)

    @property
    def height(self) -> int:
        return 0 if self.is_leaf else 1 + max(child.height for child in self.children)

    @property
    def leaf_count(self) -> int:
        return 1 if self.is_leaf else sum(child.leaf_count for child in self.children)

    @property
    def internal_count(self) -> int:
        return 0 if self.is_leaf else 1 + sum(child.internal_count for child in self.children)

    @property
    def vertex_count(self) -> int:
        return self.leaf_count + self.internal_count

    def arity_profile(self) -> dict[int, int]:
        profile: dict[int, int] = {}
        if not self.is_leaf:
            profile[self.arity] = 1
            for child in self.children:
                for arity, count in child.arity_profile().items():
                    profile[arity] = profile.get(arity, 0) + count
        return profile

    def prefix(self) -> str:
        if self.is_leaf:
            return "s"
        return f"B{self.arity}(" + ",".join(child.prefix() for child in self.children) + ")"

    def to_dict(self) -> dict[str, Any]:
        return {
            "arity": self.arity,
            "height": self.height,
            "leaves": self.leaf_count,
            "children": tuple(child.to_dict() for child in self.children),
        }


def ordered_tree_count(max_height: int, arities: Sequence[int]) -> int:
    """Count trees whose height is at most ``max_height``.

    The exact recurrence is ``N_0=1`` and
    ``N_h=1+sum_r N_{h-1}**r``.
    """

    height = int(max_height)
    if isinstance(max_height, bool) or height != max_height or height < 0:
        raise ValidationError("maximum tree height must be a nonnegative integer")
    normalized = _normalize_arities(arities)
    count = 1
    for _ in range(height):
        count = 1 + sum(count**arity for arity in normalized)
    return count


@lru_cache(maxsize=None)
def _ordered_rooted_trees_cached(
    max_height: int,
    arities: tuple[int, ...],
) -> tuple[OrderedRootedTree, ...]:
    if max_height == 0:
        return (OrderedRootedTree.leaf(),)
    previous = _ordered_rooted_trees_cached(max_height - 1, arities)
    trees: list[OrderedRootedTree] = [OrderedRootedTree.leaf()]
    for arity in arities:
        trees.extend(OrderedRootedTree(tuple(children)) for children in product(previous, repeat=arity))
    return tuple(trees)


def ordered_rooted_trees(
    max_height: int,
    arities: Sequence[int],
    *,
    max_trees: int = 100_000,
) -> tuple[OrderedRootedTree, ...]:
    """Enumerate all ordered rooted trees up to a height bound."""

    limit = int(max_trees)
    if isinstance(max_trees, bool) or limit != max_trees or limit < 1:
        raise ValidationError("max_trees must be a positive integer")
    normalized = _normalize_arities(arities)
    count = ordered_tree_count(max_height, normalized)
    if count > limit:
        raise ValidationError(
            f"tree family has {count} members, above max_trees={limit}; lower the height or raise the explicit limit"
        )
    return _ordered_rooted_trees_cached(int(max_height), normalized)


@lru_cache(maxsize=None)
def _positive_compositions(total: int, parts: int) -> tuple[tuple[int, ...], ...]:
    if parts == 1:
        return ((total,),) if total >= 1 else ()
    output: list[tuple[int, ...]] = []
    for first in range(1, total - parts + 2):
        for tail in _positive_compositions(total - first, parts - 1):
            output.append((first,) + tail)
    return tuple(output)


@lru_cache(maxsize=None)
def _trees_with_leaves_cached(
    leaves: int,
    arities: tuple[int, ...],
) -> tuple[OrderedRootedTree, ...]:
    if leaves == 1:
        return (OrderedRootedTree.leaf(),)
    output: list[OrderedRootedTree] = []
    for arity in arities:
        if arity > leaves:
            continue
        for composition in _positive_compositions(leaves, arity):
            child_families = tuple(_trees_with_leaves_cached(size, arities) for size in composition)
            output.extend(
                OrderedRootedTree(tuple(children))
                for children in product(*child_families)
            )
    return tuple(output)


def _bounded_leaf_tree_count(leaves: int, arities: tuple[int, ...], limit: int) -> int:
    """Count without constructing/caching trees, saturating at limit + 1.

    For F(x) = x + sum(F(x)**a), compute coefficients by increasing
    degree. Every product coefficient uses strictly smaller degrees.
    Zero counts (for example even leaf counts of ternary trees) remain
    zero; a large count at another degree cannot cause a false refusal.
    The counting table itself is finite work, not a universal memory cap.
    """
    if leaves == 1:
        return 1
    relevant = tuple(a for a in arities if a <= leaves)
    if not relevant:
        return 0
    cap = limit + 1
    counts = [0] * (leaves + 1)
    counts[1] = 1
    powers = {1: counts}
    for arity in range(2, max(relevant) + 1):
        powers[arity] = [0] * (leaves + 1)
    for degree in range(2, leaves + 1):
        for arity in range(2, min(max(relevant), degree) + 1):
            total = 0
            for left in range(1, degree - arity + 2):
                total += counts[left] * powers[arity - 1][degree - left]
                if total >= cap:
                    total = cap
                    break
            powers[arity][degree] = total
        counts[degree] = min(cap, sum(powers[a][degree] for a in relevant))
    return counts[leaves]


def ordered_trees_with_leaves(
    leaves: int,
    arities: Sequence[int],
    *,
    max_trees: int = 100_000,
) -> tuple[OrderedRootedTree, ...]:
    """Enumerate trees with exactly the requested number of leaves."""

    count = int(leaves)
    if isinstance(leaves, bool) or count != leaves or count < 1:
        raise ValidationError("leaf count must be a positive integer")
    limit = int(max_trees)
    if isinstance(max_trees, bool) or limit != max_trees or limit < 1:
        raise ValidationError("max_trees must be a positive integer")
    normalized = _normalize_arities(arities)
    cardinality = _bounded_leaf_tree_count(count, normalized, limit)
    if cardinality > limit:
        raise ValidationError(
            f"tree family exceeds max_trees={limit}; lower the leaf count or raise the explicit limit"
        )
    if cardinality == 0:
        return ()
    return _trees_with_leaves_cached(count, normalized)


def catalan_number(index: int) -> int:
    """Return the exact Catalan number ``C_index``."""

    n = int(index)
    if isinstance(index, bool) or n != index or n < 0:
        raise ValidationError("Catalan index must be a nonnegative integer")
    return comb(2 * n, n) // (n + 1)


@dataclass(frozen=True, slots=True)
class MajorantConvergence:
    """Scalar majorant data for a polynomial fixed-point problem."""

    seed_norm: float
    operation_norms: dict[int, float]
    radius: float
    image_bound: float
    contraction: float
    certificate: Certificate

    def scalar_map(self, value: float) -> float:
        t = float(value)
        return self.seed_norm + sum(bound * t**arity for arity, bound in self.operation_norms.items())

    def scalar_iterates(self, iterations: int) -> tuple[float, ...]:
        count = int(iterations)
        if isinstance(iterations, bool) or count != iterations or count < 0:
            raise ValidationError("iteration count must be a nonnegative integer")
        values = [0.0]
        for _ in range(count):
            values.append(self.scalar_map(values[-1]))
        return tuple(values)

    def picard_tail_bound(self, iterate: int) -> float:
        """Upward-rounded tail bound after freshly checking the stored scalars.

        The mutable operation-norm mapping is rechecked. The certificate flag
        alone and the displayed floating contraction do not unlock this bound.
        """
        self.certificate.require()
        depth = int(iterate)
        if isinstance(iterate, bool) or depth != iterate or depth < 0:
            raise ValidationError("Picard depth must be a nonnegative integer")
        checked = majorant_convergence(self.seed_norm, self.operation_norms, self.radius,
                                      tolerance=self.certificate.tolerance or 0.0)
        checked.certificate.require()
        aq, _, _, q = guard_values(checked.seed_norm, checked.operation_norms, checked.radius)
        return tail_upper(aq, q, depth)


@documented(
    ProvenanceRecord(
        identifier="PSC-III-T14",
        title="Polynomial rooted-tree convergence certificate",
        status=ClaimStatus.PROVED,
        source="Paper III, polynomial tree expansion theorem",
        assumptions=("declared multilinear norm bounds", "closed Banach ball", "majorant contraction"),
        verification_suite="nonlinear.trees",
    )
)
def majorant_convergence(
    seed_norm: float,
    operation_norms: Mapping[int, float],
    radius: float,
    *,
    tolerance: float = 1e-12,
) -> MajorantConvergence:
    """Check scalar guards exactly for the normalized binary-float inputs.

    Tolerance is retained as diagnostic policy only. It never relaxes phi(R)<=R
    or q<1. Invalid guards give an invalid certificate; excessive exact-arithmetic
    work is rejected explicitly before powering.
    """
    a = float(seed_norm)
    R = float(radius)
    tol = float(tolerance)
    if not isfinite(a) or a < 0:
        raise ValidationError("seed norm must be finite and nonnegative")
    if not isfinite(R) or R <= 0:
        raise ValidationError("majorant radius must be finite and positive")
    if not isfinite(tol) or tol < 0:
        raise ValidationError("tolerance must be finite and nonnegative")
    bounds: dict[int, float] = {}
    for raw_arity, raw_bound in operation_norms.items():
        arity = int(raw_arity)
        bound = float(raw_bound)
        if isinstance(raw_arity, bool) or arity != raw_arity or arity < 2:
            raise ValidationError("operation norm labels must be integer arities at least two")
        if not isfinite(bound) or bound < 0:
            raise ValidationError("operation norm bounds must be finite and nonnegative")
        bounds[arity] = bound
    if not bounds:
        raise ValidationError("at least one operation norm is required")
    aq, rq, image_q, contraction_q = guard_values(a, bounds, R)
    self_map = image_q <= rq
    contractive = contraction_q < 1
    valid = self_map and contractive
    image, contraction = upper_float(image_q), upper_float(contraction_q)
    checked = ("nonnegative declared norm bounds",)
    if self_map:
        checked += ("majorant maps declared ball into itself",)
    if contractive:
        checked += ("majorant derivative is strictly below one",)
    certificate = Certificate(
        valid=valid,
        exactness=Exactness.NUMERICAL,
        residuals={"ball_image_excess": upper_float(max(Fraction(0), image_q-rq)),
                   "contraction_excess": upper_float(max(Fraction(0), contraction_q-1))},
        tolerance=tol,
        assumptions_checked=checked,
        assumptions_unchecked=("declared operation bounds dominate the actual multilinear norms",
                               "ambient space is complete"),
        diagnostics={
            "radius": R, "image_bound": image, "contraction": contraction,
            "operation_arities": tuple(sorted(bounds)),
            "guard_status": "PROVED_GUARDS" if valid else "FAILED_GUARDS",
            "guard_arithmetic": "exact rational values of normalized binary-float inputs",
            "self_map_proved": self_map, "strict_contraction_proved": contractive,
            "exact_seed": rational_record(aq), "exact_radius": rational_record(rq),
            "exact_image": rational_record(image_q), "exact_contraction": rational_record(contraction_q),
            "exact_image_excess_signed": rational_record(image_q-rq),
            "tolerance_relaxes_guards": False,
        },
        theorem_id="PSC-III-T14",
        message="scalar majorant guards hold; external analytic assumptions remain" if valid else "scalar majorant guards fail",
    )
    return MajorantConvergence(a, bounds, R, image, contraction, certificate)


def quadratic_catalan_majorant(
    seed_norm: float,
    bilinear_norm: float,
    *,
    tolerance: float = 1e-12,
) -> MajorantConvergence:
    """Find a verified float radius near the small quadratic root.

    The candidate radical is only a proposal. Exact guard checks accept it or
    nearby upper floats. Failure to find a representable radius is inconclusive,
    not a proof that the exact positive-discriminant problem has no solution.
    """
    a, b = float(seed_norm), float(bilinear_norm)
    if not isfinite(a) or a < 0 or not isfinite(b) or b < 0:
        raise ValidationError("quadratic majorant norms must be finite and nonnegative")
    if b == 0.0:
        return majorant_convergence(a, {2: b}, a if a else 1.0, tolerance=tolerance)
    if a == 0.0:
        radius = upper_float(min(Fraction(1), Fraction(1) / (4*Fraction.from_float(b))))
        return majorant_convergence(a, {2: b}, radius, tolerance=tolerance)
    discriminant = 1 - 4*Fraction.from_float(a)*Fraction.from_float(b)
    if discriminant <= 0:
        radius = upper_float(Fraction(1)/(2*Fraction.from_float(b)))
        result = majorant_convergence(a, {2: b}, radius, tolerance=tolerance)
        # At a double root q=1 is not a strict contraction.
        certificate = Certificate(
            valid=False, exactness=Exactness.NUMERICAL,
            residuals={"catalan_discriminant_defect": upper_float(-discriminant)},
            tolerance=result.certificate.tolerance,
            assumptions_checked=("nonnegative scalar norms",),
            assumptions_unchecked=result.certificate.assumptions_unchecked,
            diagnostics={"guard_status": "FAILED_GUARDS", "exact_discriminant": rational_record(discriminant),
                         "candidate_radius": radius},
            theorem_id="PSC-III-C15", message="strict Catalan condition 4ab<1 fails")
        return MajorantConvergence(a, {2: b}, radius, result.image_bound, result.contraction, certificate)
    # Rationalized root avoids cancellation in 1-sqrt(discriminant).
    radius = a / ((1.0 + sqrt(float(discriminant))) / 2.0)
    if not isfinite(radius) or radius <= 0:
        raise ValidationError("no finite positive floating candidate radius")
    for _ in range(16):
        result = majorant_convergence(a, {2: b}, radius, tolerance=tolerance)
        if result.certificate.valid:
            return result
        if not result.certificate.diagnostics["strict_contraction_proved"]:
            break
        candidate = nextafter(radius, inf)
        if not isfinite(candidate):
            break
        radius = candidate
    old = result.certificate
    certificate = Certificate(
        valid=False, exactness=old.exactness, residuals=old.residuals,
        tolerance=old.tolerance, assumptions_checked=old.assumptions_checked,
        assumptions_unchecked=old.assumptions_unchecked + ("representable invariant-ball radius not established",),
        diagnostics={**old.diagnostics, "guard_status": "UNRESOLVED_BOUNDARY",
                     "exact_discriminant": rational_record(discriminant), "radius_search_budget": 16},
        theorem_id=old.theorem_id, message="positive discriminant; float radius search did not establish all guards")
    return MajorantConvergence(a, {2: b}, result.radius, result.image_bound, result.contraction, certificate)


@dataclass(frozen=True, slots=True)
class TreeTerm:
    """One ordered rooted tree and its evaluated contribution."""


    tree: OrderedRootedTree
    value: Any


@dataclass(frozen=True, slots=True)
class TreeExpansionResult:
    """Finite rooted-tree expansion compared with the matching Picard iterate."""


    value: Any
    picard_value: Any
    terms: tuple[TreeTerm, ...]
    depth: int
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="PSC-III-T14A",
        title="Finite polynomial rooted-tree expansion",
        status=ClaimStatus.FINITE_CERTIFICATE,
        source="Paper III",
        assumptions=("finite-dimensional declared multilinear operations",),
        verification_suite="nonlinear.trees",
    )
)
class PolynomialFixedPointProblem:
    """A finite-dimensional equation ``u = s + sum_r B_r(u,...,u)``."""

    def __init__(
        self,
        seed: Any,
        operations: Mapping[int, MultilinearOperation],
        *,
        exactness: Exactness = Exactness.NUMERICAL,
        tolerance: float = 1e-10,
        name: str = "polynomial fixed-point problem",
    ) -> None:
        self.seed = seed
        self.dimension = _vector_length(seed)
        normalized: dict[int, MultilinearOperation] = {}
        for raw_arity, operation in operations.items():
            arity = int(raw_arity)
            if isinstance(raw_arity, bool) or arity != raw_arity or arity < 2:
                raise ValidationError("operation arities must be integers at least two")
            if not callable(operation):
                raise ValidationError("every polynomial operation must be callable")
            normalized[arity] = operation
        if not normalized:
            raise ValidationError("a polynomial fixed-point problem requires at least one operation")
        self.operations = normalized
        self.arities = tuple(sorted(normalized))
        self.exactness = exactness
        self.tolerance = float(tolerance)
        self.name = str(name)
        if not isfinite(self.tolerance) or self.tolerance < 0:
            raise ValidationError("tolerance must be finite and nonnegative")

    def _validate_value(self, value: Any) -> Any:
        if _vector_length(value) != self.dimension:
            raise DimensionError("nonlinear operation output dimension does not match the seed")
        return value

    def operation(self, arity: int, arguments: Sequence[Any]) -> Any:
        n = int(arity)
        argument_tuple = tuple(arguments)
        if n not in self.operations:
            raise ValidationError(f"no operation is declared for arity {n}")
        if len(argument_tuple) != n:
            raise ValidationError(f"arity {n} operation requires exactly {n} arguments")
        if any(_vector_length(argument) != self.dimension for argument in argument_tuple):
            raise DimensionError("nonlinear operation argument dimension does not match the seed")
        return self._validate_value(self.operations[n](argument_tuple))

    def map(self, value: Any) -> Any:
        self._validate_value(value)
        total = self.seed
        for arity in self.arities:
            total = _add(total, self.operation(arity, (value,) * arity))
        return self._validate_value(total)

    def residual(self, value: Any) -> Any:
        return _subtract(value, self.map(value))

    def picard(self, iterations: int, *, initial: Any | None = None) -> Any:
        count = int(iterations)
        if isinstance(iterations, bool) or count != iterations or count < 0:
            raise ValidationError("iteration count must be a nonnegative integer")
        value = _zero_like(self.seed) if initial is None else self._validate_value(initial)
        for _ in range(count):
            value = self.map(value)
        return value

    def tree_value(self, tree: OrderedRootedTree) -> Any:
        if tree.is_leaf:
            return self.seed
        values = tuple(self.tree_value(child) for child in tree.children)
        return self.operation(tree.arity, values)

    def tree_expansion(
        self,
        depth: int,
        *,
        max_trees: int = 100_000,
        retain_terms: bool = True,
    ) -> TreeExpansionResult:
        """Sum all trees of height less than ``depth``.

        This finite sum equals the Picard iterate starting from zero at the same
        depth.  It is a combinatorial identity and does not itself assert
        convergence as depth tends to infinity.
        """

        count = int(depth)
        if isinstance(depth, bool) or count != depth or count < 0:
            raise ValidationError("tree expansion depth must be a nonnegative integer")
        if count == 0:
            trees: tuple[OrderedRootedTree, ...] = ()
        else:
            trees = ordered_rooted_trees(count - 1, self.arities, max_trees=max_trees)
        total = _zero_like(self.seed)
        terms: list[TreeTerm] = []
        for tree in trees:
            value = self.tree_value(tree)
            total = _add(total, value)
            if retain_terms:
                terms.append(TreeTerm(tree, value))
        picard = self.picard(count)
        audit = value_audit(exact=self.exactness is Exactness.EXACT, tolerance=self.tolerance)
        residual = audit.check_value("tree_sum_minus_picard", _subtract(total, picard))
        valid = audit.valid
        return TreeExpansionResult(
            total,
            picard,
            tuple(terms),
            count,
            Certificate(
                valid=valid,
                exactness=self.exactness,
                residuals={"tree_sum_minus_picard": residual},
                tolerance=None if self.exactness is Exactness.EXACT else self.tolerance,
                assumptions_checked=("ordered tree enumeration", "recursive multilinear evaluation"),
                assumptions_unchecked=("convergence beyond the finite depth",
                                       "global correctness and multilinearity of declared operations") + audit.unresolved,
                diagnostics={"depth": count, "tree_count": len(trees), "arities": self.arities,
                             "residual_decisions": audit.decisions},
                theorem_id="PSC-III-T14A",
                message="finite tree sum equals the Picard iterate" if valid else "tree and Picard values disagree",
            ),
        )

    def convergence_certificate(
        self,
        operation_norms: Mapping[int, float],
        radius: float,
        *,
        seed_norm: float | None = None,
    ) -> MajorantConvergence:
        actual_seed_norm = _norm(self.seed)
        if seed_norm is None:
            declared_seed_norm = actual_seed_norm
            seed_status = "measured from the stored seed"
        else:
            declared_seed_norm = float(seed_norm)
            if not isfinite(declared_seed_norm) or declared_seed_norm < 0:
                raise ValidationError("declared seed norm must be finite and nonnegative")
            if declared_seed_norm < actual_seed_norm:
                raise ValidationError("declared seed norm must dominate the stored seed norm")
            seed_status = "declared bound checked against the stored seed"
        result = majorant_convergence(
            declared_seed_norm, operation_norms, radius, tolerance=self.tolerance)
        old = result.certificate
        certificate = Certificate(
            valid=old.valid, exactness=old.exactness, residuals=old.residuals,
            tolerance=old.tolerance,
            assumptions_checked=old.assumptions_checked + (seed_status,),
            assumptions_unchecked=old.assumptions_unchecked,
            diagnostics={**old.diagnostics, "stored_seed_norm": actual_seed_norm,
                         "declared_seed_norm": declared_seed_norm},
            theorem_id=old.theorem_id, message=old.message)
        return MajorantConvergence(result.seed_norm, result.operation_norms, result.radius,
                                   result.image_bound, result.contraction, certificate)


@dataclass(frozen=True, slots=True)
class PacketTreeResult:
    """Packet-labeled rooted-tree evaluation with pruning statistics."""


    value: Any
    packet_values: tuple[Any, ...]
    evaluated_blocks: int
    pruned_blocks: int
    certificate: Certificate


def _zero_addresses(
    support: Mapping[int, PacketJetSupport] | None,
    arity: int,
) -> frozenset[PacketJetAddress]:
    if support is None or arity not in support:
        return frozenset()
    return frozenset(support[arity].zero)


@documented(
    ProvenanceRecord(
        identifier="PSC-III-T16",
        title="Packet-labeled rooted-tree reconstruction",
        status=ClaimStatus.PROVED,
        source="Paper III",
        assumptions=("complete finite packet system", "declared multilinear vertex maps"),
        verification_suite="nonlinear.trees",
    )
)
def packet_tree_reconstruction(
    problem: PolynomialFixedPointProblem,
    tree: OrderedRootedTree,
    packets: PacketSystem,
    *,
    zero_support: Mapping[int, PacketJetSupport] | None = None,
    tolerance: float | None = None,
) -> PacketTreeResult:
    """Reconstruct one tree by summing packet labels at every vertex."""

    if packets.dimension != problem.dimension:
        raise DimensionError("packet dimension does not match the fixed-point problem")
    tol = finite_tolerance(packets.tolerance if tolerance is None else tolerance)
    evaluated = 0
    pruned = 0

    def recurse(node: OrderedRootedTree) -> tuple[Any, ...]:
        nonlocal evaluated, pruned
        if node.is_leaf:
            seed_vector = packets.backend.as_vector(problem.seed)
            return tuple(
                packets.backend.matmul(projector, seed_vector)
                for projector in packets.projectors
            )
        child_packets = tuple(recurse(child) for child in node.children)
        zero = _zero_addresses(zero_support, node.arity)
        outputs: list[Any] = []
        for output_index, output_name in enumerate(packets.names):
            total = packets.backend.zeros(packets.dimension, 1)
            if packets.backend.name != "sympy":
                total = np.asarray(total).reshape(-1)
            for input_indices in product(range(len(packets)), repeat=node.arity):
                address = PacketJetAddress(
                    output_name,
                    tuple(packets.names[index] for index in input_indices),
                )
                if address in zero:
                    pruned += 1
                    continue
                arguments = tuple(
                    child_packets[position][packet_index]
                    for position, packet_index in enumerate(input_indices)
                )
                raw = packets.backend.as_vector(problem.operation(node.arity, arguments))
                value = packets.backend.matmul(packets.projectors[output_index], raw)
                total = packets.backend.add(total, value)
                evaluated += 1
            outputs.append(total)
        return tuple(outputs)

    packet_values = recurse(tree)
    total = packets.synthesize(packet_values)
    direct = packets.backend.as_vector(problem.tree_value(tree))
    exactness = (
        Exactness.EXACT
        if problem.exactness is Exactness.EXACT and packets.backend.exact
        else Exactness.NUMERICAL
    )
    audit = value_audit(exact=exactness is Exactness.EXACT, tolerance=tol)
    residual = audit.check_value("packet_tree_reconstruction",
                                packets.backend.subtract(direct, packets.backend.as_vector(total)))
    parents_valid, parent_checked, parent_unchecked, parent_diagnostics = packet_certificate_dependencies(
        packets=packets
    )
    valid = parents_valid and audit.valid
    return PacketTreeResult(
        total,
        packet_values,
        evaluated,
        pruned,
        Certificate(
            valid=valid,
            exactness=exactness,
            residuals={"packet_tree_reconstruction": residual},
            tolerance=None if exactness is Exactness.EXACT else tol,
            assumptions_checked=parent_checked + (
                "complete packet identity at every input and output slot",
                "finite recursive tree evaluation",
                "declared zero blocks used only for pruning",
            ),
            assumptions_unchecked=parent_unchecked + ("global correctness of declared multilinear operations",) + audit.unresolved,
            diagnostics={
                **parent_diagnostics,
                "residual_decisions": audit.decisions,
                "tree_height": tree.height,
                "tree_leaves": tree.leaf_count,
                "evaluated_blocks": evaluated,
                "pruned_blocks": pruned,
            },
            theorem_id="PSC-III-T16",
            message="packet-labeled tree reconstructs the original value" if valid else "packet tree reconstruction failed",
        ),
    )


__all__ = [
    "MajorantConvergence",
    "MultilinearOperation",
    "OrderedRootedTree",
    "PacketTreeResult",
    "PolynomialFixedPointProblem",
    "TreeExpansionResult",
    "TreeTerm",
    "catalan_number",
    "majorant_convergence",
    "ordered_rooted_trees",
    "ordered_tree_count",
    "ordered_trees_with_leaves",
    "packet_tree_reconstruction",
    "quadratic_catalan_majorant",
]
