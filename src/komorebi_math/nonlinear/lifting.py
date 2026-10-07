"""Exact finite lifting over prime power rings."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Iterable

import numpy as np

from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .._number_theory import require_prime


def _as_int_matrix(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=object)
    if array.ndim != 2:
        raise DimensionError("matrix data must be two dimensional")
    converted = np.empty(array.shape, dtype=object)
    for index, entry in np.ndenumerate(array):
        integer = int(entry)
        if integer != entry:
            raise ValidationError("lifting data must be integral")
        converted[index] = integer
    return converted


def _as_int_vector(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=object)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1:
        raise DimensionError("vector data must be one dimensional")
    converted = np.empty(array.shape, dtype=object)
    for index, entry in np.ndenumerate(array):
        integer = int(entry)
        if integer != entry:
            raise ValidationError("lifting data must be integral")
        converted[index] = integer
    return converted


@dataclass(frozen=True, slots=True)
class ModularLinearSolution:
    """The affine solution space of a linear system over a prime field."""

    prime: int
    solvable: bool
    particular: tuple[int, ...] | None
    kernel_basis: tuple[tuple[int, ...], ...]
    pivot_columns: tuple[int, ...]
    obstruction_rows: tuple[int, ...]

    @property
    def nullity(self) -> int:
        return len(self.kernel_basis)

    @property
    def solution_count(self) -> int:
        return 0 if not self.solvable else self.prime**self.nullity

    def enumerate(self, *, limit: int = 100000) -> tuple[tuple[int, ...], ...]:
        if not self.solvable or self.particular is None:
            return ()
        if self.solution_count > limit:
            raise ValueError(f"solution count {self.solution_count} exceeds enumeration limit {limit}")
        solutions: list[tuple[int, ...]] = []
        for coefficients in product(range(self.prime), repeat=self.nullity):
            vector = list(self.particular)
            for coefficient, basis in zip(coefficients, self.kernel_basis, strict=True):
                vector = [
                    (value + coefficient * direction) % self.prime
                    for value, direction in zip(vector, basis, strict=True)
                ]
            solutions.append(tuple(vector))
        return tuple(solutions)


@documented(
    ProvenanceRecord(
        identifier="PSC-III-LIN-FP",
        title="Linear correction space over a prime field",
        status=ClaimStatus.CLASSICAL,
        source="Paper III",
        assumptions=("prime modulus",),
        verification_suite="nonlinear.lifting",
    )
)
def solve_mod_prime(matrix: Any, rhs: Any, prime: int) -> ModularLinearSolution:
    """Solve a finite linear system modulo a prime by exact row reduction."""

    prime = require_prime(prime)
    coefficients = _as_int_matrix(matrix) % prime
    target = _as_int_vector(rhs) % prime
    rows, columns = coefficients.shape
    if target.shape[0] != rows:
        raise DimensionError("right hand side dimension does not match the matrix")
    augmented = np.concatenate([coefficients, target.reshape(rows, 1)], axis=1).astype(object)
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(columns):
        candidate = next((row for row in range(pivot_row, rows) if int(augmented[row, column]) % prime), None)
        if candidate is None:
            continue
        if candidate != pivot_row:
            augmented[[pivot_row, candidate], :] = augmented[[candidate, pivot_row], :]
        inverse = pow(int(augmented[pivot_row, column]) % prime, -1, prime)
        augmented[pivot_row, :] = [(int(value) * inverse) % prime for value in augmented[pivot_row, :]]
        for row in range(rows):
            if row == pivot_row:
                continue
            factor = int(augmented[row, column]) % prime
            if factor:
                augmented[row, :] = [
                    (int(left) - factor * int(right)) % prime
                    for left, right in zip(augmented[row, :], augmented[pivot_row, :], strict=True)
                ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == rows:
            break

    obstruction_rows = tuple(
        row
        for row in range(rows)
        if all(int(augmented[row, column]) % prime == 0 for column in range(columns))
        and int(augmented[row, columns]) % prime != 0
    )
    if obstruction_rows:
        return ModularLinearSolution(prime, False, None, (), tuple(pivot_columns), obstruction_rows)

    particular = [0] * columns
    for row, column in enumerate(pivot_columns):
        particular[column] = int(augmented[row, columns]) % prime
    free_columns = [column for column in range(columns) if column not in pivot_columns]
    kernel_basis: list[tuple[int, ...]] = []
    for free_column in free_columns:
        vector = [0] * columns
        vector[free_column] = 1
        for row, pivot_column in enumerate(pivot_columns):
            vector[pivot_column] = (-int(augmented[row, free_column])) % prime
        kernel_basis.append(tuple(vector))
    return ModularLinearSolution(
        prime,
        True,
        tuple(particular),
        tuple(kernel_basis),
        tuple(pivot_columns),
        (),
    )


@dataclass(frozen=True, slots=True)
class LiftStepResult:
    """One depth obstruction and all finite correction branches."""

    prime: int
    depth: int
    point: tuple[int, ...]
    residual: tuple[int, ...]
    correction_space: ModularLinearSolution
    children: tuple[tuple[int, ...], ...]
    certificate: Certificate

    @property
    def lifts(self) -> bool:
        return self.correction_space.solvable


@documented(
    ProvenanceRecord(
        identifier="PSC-III-T28",
        title="One step lifting criterion",
        status=ClaimStatus.PROVED,
        source="Paper III",
        assumptions=("current point solves through the declared depth", "linearized correction law"),
        verification_suite="nonlinear.lifting",
    )
)
class LinearLiftProblem:
    """Lift a linear congruence one prime power at a time."""

    def __init__(self, matrix: Any, rhs: Any, point: Any, *, prime: int, depth: int) -> None:
        self.matrix = _as_int_matrix(matrix)
        self.rhs = _as_int_vector(rhs)
        self.point = _as_int_vector(point)
        if self.matrix.shape[0] != self.rhs.shape[0] or self.matrix.shape[1] != self.point.shape[0]:
            raise DimensionError("linear lifting dimensions do not agree")
        prime = require_prime(prime)
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.prime = prime
        self.depth = int(depth)

    def step(self, *, enumerate_limit: int = 100000) -> LiftStepResult:
        modulus = self.prime**self.depth
        error = self.matrix @ self.point - self.rhs
        if any(int(value) % modulus != 0 for value in error):
            raise ValidationError("current point is not a solution at the declared depth")
        residual = tuple((int(value) // modulus) % self.prime for value in error)
        correction_space = solve_mod_prime(self.matrix, tuple((-value) % self.prime for value in residual), self.prime)
        corrections = correction_space.enumerate(limit=enumerate_limit)
        next_modulus = modulus * self.prime
        children = tuple(
            tuple((int(value) + modulus * correction) % next_modulus for value, correction in zip(self.point, vector, strict=True))
            for vector in corrections
        )
        valid = all(
            all(int(value) % next_modulus == 0 for value in self.matrix @ np.asarray(child, dtype=object) - self.rhs)
            for child in children
        )
        certificate = Certificate(
            valid=bool(valid),
            exactness=Exactness.EXACT,
            residuals={"child_verification": 0.0 if valid else 1.0},
            assumptions_checked=("current congruence", "associated graded correction equation"),
            diagnostics={
                "lifts": correction_space.solvable,
                "child_count": len(children),
                "kernel_dimension": correction_space.nullity,
            },
            theorem_id="PSC-III-T28",
            message="one step lift space computed exactly" if valid else "one or more computed children failed verification",
        )
        return LiftStepResult(
            self.prime,
            self.depth,
            tuple(int(value) for value in self.point),
            residual,
            correction_space,
            children,
            certificate,
        )


class PolynomialLiftProblem:
    """Lift a polynomial system by its Jacobian over the residual field."""

    def __init__(
        self,
        function: Callable[[tuple[int, ...]], Iterable[int]],
        jacobian: Callable[[tuple[int, ...]], Any],
        point: Any,
        *,
        prime: int,
        depth: int,
    ) -> None:
        prime = require_prime(prime)
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.function = function
        self.jacobian = jacobian
        self.point = tuple(int(value) for value in _as_int_vector(point))
        self.prime = prime
        self.depth = int(depth)

    def step(
        self,
        *,
        enumerate_limit: int = 100000,
        linearization_limit: int = 100000,
        assume_valid_jacobian: bool = False,
    ) -> LiftStepResult:
        """Compute all one step lifts from the Jacobian correction equation.

        By default the method verifies the first order congruence for every
        correction vector over the residual field.  When that finite check is
        too large, callers may set ``assume_valid_jacobian=True``.  The result
        is then marked conditional rather than exact.
        """

        modulus = self.prime**self.depth
        values = tuple(int(value) for value in self.function(self.point))
        if any(value % modulus != 0 for value in values):
            raise ValidationError("current point is not a polynomial solution at the declared depth")
        residual = tuple((value // modulus) % self.prime for value in values)
        jacobian = _as_int_matrix(self.jacobian(self.point))
        if jacobian.shape != (len(values), len(self.point)):
            raise DimensionError("Jacobian shape does not match the polynomial system")

        correction_count = self.prime ** len(self.point)
        linearization_verified = correction_count <= linearization_limit
        if not linearization_verified and not assume_valid_jacobian:
            raise ValueError(
                f"Jacobian verification requires {correction_count} residual corrections, "
                f"above the limit {linearization_limit}. Increase the limit or explicitly trust the Jacobian."
            )
        if linearization_verified:
            base = np.asarray(values, dtype=object)
            for correction in product(range(self.prime), repeat=len(self.point)):
                correction_vector = np.asarray(correction, dtype=object)
                candidate = tuple(
                    int(value) + modulus * int(direction)
                    for value, direction in zip(self.point, correction, strict=True)
                )
                candidate_values = np.asarray(
                    tuple(int(value) for value in self.function(candidate)),
                    dtype=object,
                )
                difference = candidate_values - base
                if any(int(value) % modulus != 0 for value in difference):
                    raise ValidationError(
                        "the supplied function does not satisfy the required congruence preservation at this depth"
                    )
                observed = np.asarray(
                    [(int(value) // modulus) % self.prime for value in difference],
                    dtype=object,
                )
                expected = (jacobian @ correction_vector) % self.prime
                if not np.array_equal(observed % self.prime, expected % self.prime):
                    raise ValidationError(
                        "the supplied Jacobian does not give the first order correction law modulo the prime"
                    )

        correction_space = solve_mod_prime(
            jacobian,
            tuple((-value) % self.prime for value in residual),
            self.prime,
        )
        corrections = correction_space.enumerate(limit=enumerate_limit)
        next_modulus = modulus * self.prime
        children = tuple(
            tuple(
                (value + modulus * correction) % next_modulus
                for value, correction in zip(self.point, vector, strict=True)
            )
            for vector in corrections
        )
        children_valid = all(
            all(int(value) % next_modulus == 0 for value in self.function(child))
            for child in children
        )
        exactness = Exactness.EXACT if linearization_verified else Exactness.CONDITIONAL
        assumptions_unchecked = () if linearization_verified else (
            "the supplied Jacobian gives the complete first order correction law modulo the prime",
        )
        certificate = Certificate(
            valid=bool(children_valid),
            exactness=exactness,
            residuals={"child_verification": 0.0 if children_valid else 1.0},
            assumptions_checked=(
                "current polynomial congruence",
                "all computed children",
            ) + (("Jacobian linearization over every residual correction",) if linearization_verified else ()),
            assumptions_unchecked=assumptions_unchecked,
            diagnostics={
                "lifts": correction_space.solvable,
                "child_count": len(children),
                "kernel_dimension": correction_space.nullity,
                "linearization_verified": linearization_verified,
                "residual_correction_count": correction_count,
            },
            theorem_id="PSC-III-T28",
            message=(
                "polynomial lift space computed exactly"
                if children_valid and linearization_verified
                else "polynomial lift space computed conditionally on the supplied Jacobian"
                if children_valid
                else "one or more polynomial children failed verification"
            ),
        )
        return LiftStepResult(
            self.prime,
            self.depth,
            self.point,
            residual,
            correction_space,
            children,
            certificate,
        )



@dataclass(frozen=True, slots=True)
class LiftTreeNode:
    """One congruence class in a finite prime power lift tree."""

    value: tuple[int, ...]
    depth: int
    parent_index: int | None


@dataclass(frozen=True, slots=True)
class FiniteLiftTree:
    """A finite prefix of a prime power lift tree."""

    prime: int
    levels: tuple[tuple[LiftTreeNode, ...], ...]

    def __post_init__(self) -> None:
        require_prime(self.prime)
        for level in self.levels:
            for node in level:
                if node.depth <= 0:
                    raise ValueError("lift node depth must be positive")

    @property
    def max_depth(self) -> int:
        return self.levels[-1][0].depth if self.levels and self.levels[-1] else 0

    def terminal_branches(self) -> tuple[tuple[LiftTreeNode, ...], ...]:
        if not self.levels:
            return ()
        branches: list[tuple[LiftTreeNode, ...]] = []
        for terminal_index, terminal in enumerate(self.levels[-1]):
            path = [terminal]
            current_index = terminal_index
            for level_index in range(len(self.levels) - 1, 0, -1):
                parent_index = self.levels[level_index][current_index].parent_index
                if parent_index is None:
                    raise ArithmeticError("nonroot lift node is missing its parent index")
                current_index = parent_index
                path.append(self.levels[level_index - 1][current_index])
            branches.append(tuple(reversed(path)))
        return tuple(branches)


@documented(
    ProvenanceRecord(
        identifier="PSC-III-T30",
        title="Finite lift tree construction",
        status=ClaimStatus.PROVED,
        source="Paper III",
        assumptions=("finite residue field", "finite correction spaces"),
        verification_suite="nonlinear.lifting",
    )
)
def build_lift_tree(
    problem_factory: Callable[[tuple[int, ...], int], LinearLiftProblem | PolynomialLiftProblem],
    roots: Iterable[tuple[int, ...]],
    *,
    prime: int,
    start_depth: int,
    max_depth: int,
    max_nodes: int = 100000,
) -> FiniteLiftTree:
    """Build a finite lift tree from exact one step correction problems."""

    prime = require_prime(prime)
    if start_depth <= 0:
        raise ValueError("start depth must be positive")
    if max_depth < start_depth:
        raise ValueError("maximum depth must not precede the start depth")
    if max_nodes <= 0:
        raise ValueError("maximum node count must be positive")
    current_level = tuple(LiftTreeNode(tuple(root), start_depth, None) for root in roots)
    levels: list[tuple[LiftTreeNode, ...]] = [current_level]
    node_count = len(current_level)
    for depth in range(start_depth, max_depth):
        next_nodes: list[LiftTreeNode] = []
        for parent_index, node in enumerate(current_level):
            problem = problem_factory(node.value, depth)
            if problem.prime != prime:
                raise ValidationError("lift tree problem uses a different prime")
            result = problem.step(enumerate_limit=max_nodes)
            result.certificate.require()
            for child in result.children:
                next_nodes.append(LiftTreeNode(child, depth + 1, parent_index))
                node_count += 1
                if node_count > max_nodes:
                    raise ValueError("lift tree exceeds the declared node limit")
        current_level = tuple(next_nodes)
        levels.append(current_level)
        if not current_level:
            break
    return FiniteLiftTree(prime, tuple(levels))


__all__ = [
    "FiniteLiftTree",
    "LiftStepResult",
    "LiftTreeNode",
    "LinearLiftProblem",
    "ModularLinearSolution",
    "PolynomialLiftProblem",
    "build_lift_tree",
    "solve_mod_prime",
]
