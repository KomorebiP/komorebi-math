"""Exact dense matrix-chain planning independent of expression evaluation.

The cost model counts scalar multiplications performed by classical dense
matrix multiplication.  It optimizes only over binary parenthesizations of the
supplied, order-preserving matrix chain.  It is not a hardware timing model and
does not include additions, sparsity, Strassen-like algorithms, or reuse across
different chains.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any
from unicodedata import normalize

from ..certificates import Certificate, Exactness
from ..coordinates.exact import Matrix, matrix_multiply, matrix_shape
from ..exceptions import DimensionError, ResourceLimitError, ValidationError

_THEOREM_ID = "EXPR.PLAN.MATRIX_CHAIN_DP"
_MAX_PLANNING_STEPS = 1_000_000


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{name} must be a nonnegative integer")
    return value


def _dimensions(values: Sequence[int]) -> tuple[int, ...]:
    try:
        dimensions = tuple(values)
    except TypeError as exc:
        raise ValidationError("matrix-chain dimensions must be a finite sequence") from exc
    if len(dimensions) < 2:
        raise ValidationError("a matrix chain requires at least two dimensions")
    return tuple(
        _positive_integer(value, name=f"matrix-chain dimension {index}")
        for index, value in enumerate(dimensions)
    )


@dataclass(frozen=True, slots=True)
class MatrixChainTree:
    """The root decision for a half-open matrix interval ``[start, end)``.

    Matrix ``i`` has shape ``dimensions[i]`` by ``dimensions[i + 1]``.
    A leaf therefore has ``end == start + 1`` and ``split is None``.
    Nonleaf splits satisfy ``start < split < end``.
    """

    start: int
    end: int
    split: int | None
    rows: int
    columns: int
    scalar_operation_cost: int

    def __post_init__(self) -> None:
        start = _nonnegative_integer(self.start, name="tree start")
        end = _positive_integer(self.end, name="tree end")
        if end <= start:
            raise ValidationError("matrix-chain tree intervals must be nonempty")
        _positive_integer(self.rows, name="tree row count")
        _positive_integer(self.columns, name="tree column count")
        cost = _nonnegative_integer(
            self.scalar_operation_cost,
            name="tree scalar-operation cost",
        )
        leaf = end == start + 1
        if leaf:
            if self.split is not None:
                raise ValidationError("a matrix-chain leaf cannot have a split")
            if cost != 0:
                raise ValidationError("a matrix-chain leaf must have zero multiplication cost")
        else:
            if (
                isinstance(self.split, bool)
                or not isinstance(self.split, int)
                or not start < self.split < end
            ):
                raise ValidationError("a nonleaf matrix-chain split must lie inside its interval")


def _matrix_chain_tables(
    dimensions: tuple[int, ...],
) -> tuple[
    MatrixChainTree,
    dict[tuple[int, int], int],
    dict[tuple[int, int], int],
]:
    """Return the optimal root, interval costs, and deterministic split table."""

    matrix_count = len(dimensions) - 1
    planning_steps = matrix_count * (matrix_count - 1) * (matrix_count + 1) // 6
    if planning_steps > _MAX_PLANNING_STEPS:
        raise ResourceLimitError(
            "matrix-chain dynamic program exceeds its fixed planning-step guard"
        )
    costs: dict[tuple[int, int], int] = {
        (index, index + 1): 0 for index in range(matrix_count)
    }
    splits: dict[tuple[int, int], int] = {}
    for length in range(2, matrix_count + 1):
        for start in range(matrix_count - length + 1):
            end = start + length
            best_cost: int | None = None
            best_split: int | None = None
            for split in range(start + 1, end):
                candidate = (
                    costs[(start, split)]
                    + costs[(split, end)]
                    + dimensions[start] * dimensions[split] * dimensions[end]
                )
                if best_cost is None or candidate < best_cost:
                    best_cost = candidate
                    best_split = split
            assert best_cost is not None and best_split is not None
            costs[(start, end)] = best_cost
            splits[(start, end)] = best_split

    root_interval = (0, matrix_count)
    root = MatrixChainTree(
        0,
        matrix_count,
        None if matrix_count == 1 else splits[root_interval],
        dimensions[0],
        dimensions[-1],
        costs[root_interval],
    )
    return root, costs, splits


def _left_associative_cost(dimensions: tuple[int, ...]) -> int:
    return sum(
        dimensions[0] * dimensions[index] * dimensions[index + 1]
        for index in range(1, len(dimensions) - 1)
    )


def _right_associative_cost(dimensions: tuple[int, ...]) -> int:
    matrix_count = len(dimensions) - 1
    return sum(
        dimensions[index] * dimensions[index + 1] * dimensions[-1]
        for index in range(matrix_count - 1)
    )


@dataclass(frozen=True, slots=True)
class MatrixChainPlan:
    """An exact optimal parenthesization record for one dimension chain."""

    dimensions: tuple[int, ...]
    tree: MatrixChainTree
    scalar_operation_cost: int
    left_associative_cost: int
    right_associative_cost: int
    certificate: Certificate

    def __post_init__(self) -> None:
        dimensions = _dimensions(self.dimensions)
        if type(self.tree) is not MatrixChainTree:
            raise TypeError("tree must be a MatrixChainTree")
        optimal_tree, _, _ = _matrix_chain_tables(dimensions)
        if self.tree != optimal_tree:
            raise ValidationError("matrix-chain tree does not match the exact dynamic program")
        optimal_cost = _nonnegative_integer(
            self.scalar_operation_cost,
            name="plan scalar-operation cost",
        )
        if optimal_cost != optimal_tree.scalar_operation_cost:
            raise ValidationError("plan cost does not match its optimal tree")
        left_cost = _nonnegative_integer(
            self.left_associative_cost,
            name="left-associative cost",
        )
        right_cost = _nonnegative_integer(
            self.right_associative_cost,
            name="right-associative cost",
        )
        if left_cost != _left_associative_cost(dimensions):
            raise ValidationError("left-associative cost does not match the dimension chain")
        if right_cost != _right_associative_cost(dimensions):
            raise ValidationError("right-associative cost does not match the dimension chain")
        if not isinstance(self.certificate, Certificate):
            raise TypeError("certificate must be a Certificate")
        if (
            not self.certificate.valid
            or self.certificate.exactness is not Exactness.EXACT
            or self.certificate.assumptions_unchecked
            or self.certificate.theorem_id != _THEOREM_ID
        ):
            raise ValidationError("matrix-chain plans require an unconditional exact DP certificate")
        object.__setattr__(self, "dimensions", dimensions)


def matrix_chain_plan(dimensions: Sequence[int]) -> MatrixChainPlan:
    """Find the exact minimum-cost binary parenthesization of a dense chain."""

    checked = _dimensions(dimensions)
    tree, _, _ = _matrix_chain_tables(checked)
    left_cost = _left_associative_cost(checked)
    right_cost = _right_associative_cost(checked)
    certificate = Certificate(
        valid=True,
        exactness=Exactness.EXACT,
        assumptions_checked=(
            "finite order-preserving matrix chain",
            "positive integer matrix dimensions",
            "all binary parenthesizations compared by dynamic programming",
            "classical dense scalar multiplication count",
        ),
        diagnostics={
            "matrix_count": len(checked) - 1,
            "dimensions": checked,
            "optimal_scalar_multiplications": tree.scalar_operation_cost,
            "left_associative_scalar_multiplications": left_cost,
            "right_associative_scalar_multiplications": right_cost,
            "tie_break": "smallest split index",
            "scope": "binary parenthesizations under the classical dense scalar multiplication count",
        },
        theorem_id=_THEOREM_ID,
        message=(
            "exact minimum over binary parenthesizations under the declared "
            "dense scalar multiplication count"
        ),
    )
    return MatrixChainPlan(
        checked,
        tree,
        tree.scalar_operation_cost,
        left_cost,
        right_cost,
        certificate,
    )


def _validated_tables(
    plan: MatrixChainPlan,
) -> tuple[tuple[int, ...], dict[tuple[int, int], int]]:
    if type(plan) is not MatrixChainPlan:
        raise TypeError("plan must be a MatrixChainPlan")
    dimensions = _dimensions(plan.dimensions)
    tree, _, splits = _matrix_chain_tables(dimensions)
    if (
        tree != plan.tree
        or plan.scalar_operation_cost != tree.scalar_operation_cost
        or plan.left_associative_cost != _left_associative_cost(dimensions)
        or plan.right_associative_cost != _right_associative_cost(dimensions)
        or not plan.certificate.valid
        or plan.certificate.exactness is not Exactness.EXACT
        or plan.certificate.assumptions_unchecked
        or plan.certificate.theorem_id != _THEOREM_ID
    ):
        raise ValidationError("matrix-chain plan failed exact replay validation")
    return dimensions, splits


def matrix_chain_parenthesization(
    plan: MatrixChainPlan,
    labels: Sequence[str] | None = None,
) -> str:
    """Render the selected binary tree without recursive traversal."""

    dimensions, splits = _validated_tables(plan)
    matrix_count = len(dimensions) - 1
    if labels is None:
        names = tuple(f"A{index}" for index in range(matrix_count))
    else:
        try:
            raw_names = tuple(labels)
        except TypeError as exc:
            raise ValidationError("matrix labels must be a finite sequence") from exc
        if len(raw_names) != matrix_count:
            raise DimensionError("matrix label count does not match the chain")
        names_list: list[str] = []
        for index, value in enumerate(raw_names):
            if not isinstance(value, str):
                raise ValidationError(f"matrix label {index} must be a string")
            name = normalize("NFC", value).strip()
            if not name:
                raise ValidationError(f"matrix label {index} must be nonempty")
            names_list.append(name)
        names = tuple(names_list)

    rendered: dict[tuple[int, int], str] = {}
    root = (0, matrix_count)
    stack: list[tuple[int, int, bool]] = [(root[0], root[1], False)]
    while stack:
        start, end, exiting = stack.pop()
        interval = (start, end)
        if end == start + 1:
            rendered[interval] = names[start]
            continue
        split = splits[interval]
        if exiting:
            rendered[interval] = (
                f"({rendered[(start, split)]} @ {rendered[(split, end)]})"
            )
            continue
        stack.append((start, end, True))
        stack.append((split, end, False))
        stack.append((start, split, False))
    return rendered[root]


def _exact_matrix(value: Iterable[Iterable[Any]], *, index: int) -> Matrix:
    try:
        rows = tuple(value)
    except TypeError as exc:
        raise ValidationError(f"matrix {index} must be a finite row iterable") from exc
    normalized: list[tuple[Fraction, ...]] = []
    for row in rows:
        try:
            entries = tuple(row)
        except TypeError as exc:
            raise ValidationError(f"matrix {index} rows must be finite") from exc
        exact_row: list[Fraction] = []
        for entry in entries:
            if isinstance(entry, bool) or not isinstance(entry, (int, Fraction)):
                raise ValidationError(
                    f"matrix {index} entries must be integers or Fractions"
                )
            exact_row.append(Fraction(entry))
        normalized.append(tuple(exact_row))
    if not normalized:
        raise DimensionError(f"matrix {index} must have at least one row")
    if not normalized[0]:
        raise DimensionError(f"matrix {index} must have at least one column")
    if any(len(row) != len(normalized[0]) for row in normalized):
        raise DimensionError(f"matrix {index} rows must have a common length")
    return tuple(normalized)


def matrix_chain_multiply(
    matrices: Sequence[Iterable[Iterable[Any]]],
    plan: MatrixChainPlan,
) -> Matrix:
    """Multiply an exact matrix chain according to a replayed optimal plan."""

    dimensions, splits = _validated_tables(plan)
    try:
        raw_matrices = tuple(matrices)
    except TypeError as exc:
        raise ValidationError("matrices must be a finite sequence") from exc
    matrix_count = len(dimensions) - 1
    if len(raw_matrices) != matrix_count:
        raise DimensionError("matrix count does not match the plan")
    exact_matrices: list[Matrix] = []
    for index, value in enumerate(raw_matrices):
        matrix = _exact_matrix(value, index=index)
        expected = (dimensions[index], dimensions[index + 1])
        if matrix_shape(matrix) != expected:
            raise DimensionError(
                f"matrix {index} shape must be {expected}, received {matrix_shape(matrix)}"
            )
        exact_matrices.append(matrix)

    products: dict[tuple[int, int], Matrix] = {}
    root = (0, matrix_count)
    stack: list[tuple[int, int, bool]] = [(root[0], root[1], False)]
    while stack:
        start, end, exiting = stack.pop()
        interval = (start, end)
        if end == start + 1:
            products[interval] = exact_matrices[start]
            continue
        split = splits[interval]
        if exiting:
            products[interval] = matrix_multiply(
                products[(start, split)],
                products[(split, end)],
            )
            continue
        stack.append((start, end, True))
        stack.append((split, end, False))
        stack.append((start, split, False))

    result = products[root]
    expected_result_shape = (dimensions[0], dimensions[-1])
    if matrix_shape(result) != expected_result_shape:
        raise ValidationError("matrix-chain execution produced an inconsistent result shape")
    return result


__all__ = [
    "MatrixChainPlan",
    "MatrixChainTree",
    "matrix_chain_multiply",
    "matrix_chain_parenthesization",
    "matrix_chain_plan",
]
