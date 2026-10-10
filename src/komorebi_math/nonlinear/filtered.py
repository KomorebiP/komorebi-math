"""Finite filtered obstructions, lift completion, and spectral-sequence records.

The executable layer works over explicit prime fields, storing Python integers
in object arrays so every product is exact before modular reduction.  Infinite inverse-limit
completion remains conditional and is represented by :class:`ConditionalResult`
with all completeness and continuity assumptions visible.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._number_theory import require_prime
from ..certificates import Certificate, ConditionalResult, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .lifting import ModularLinearSolution, solve_mod_prime


def _mod_matrix(value: Any, prime: int, *, name: str = "matrix") -> np.ndarray:
    array = np.asarray(value, dtype=object)
    if array.ndim != 2:
        raise DimensionError(f"{name} must be two dimensional")
    output = np.empty(array.shape, dtype=object)
    for index, entry in np.ndenumerate(array):
        integer = int(entry)
        if integer != entry:
            raise ValidationError(f"{name} entries must be integral")
        output[index] = integer % prime
    return output


def _mod_vector(value: Any, prime: int, dimension: int, *, name: str = "vector") -> np.ndarray:
    array = np.asarray(value, dtype=object)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1 or array.shape[0] != dimension:
        raise DimensionError(f"{name} must have dimension {dimension}")
    output = np.empty(array.shape, dtype=object)
    for index, entry in np.ndenumerate(array):
        integer = int(entry)
        if integer != entry:
            raise ValidationError(f"{name} entries must be integral")
        output[index] = integer % prime
    return output


def _rref(matrix: np.ndarray, prime: int) -> tuple[np.ndarray, tuple[int, ...]]:
    # Native integer arrays overflow before the modular reduction. Keep the
    # row operations in Python integers even when the prime fits in int64.
    value = _mod_matrix(matrix, prime).copy()
    rows, columns = value.shape
    pivots: list[int] = []
    row = 0
    for column in range(columns):
        pivot = next((index for index in range(row, rows) if value[index, column] % prime), None)
        if pivot is None:
            continue
        if pivot != row:
            value[[row, pivot]] = value[[pivot, row]]
        inverse = pow(int(value[row, column]), -1, prime)
        value[row] = (value[row] * inverse) % prime
        for other in range(rows):
            if other == row:
                continue
            factor = int(value[other, column]) % prime
            if factor:
                value[other] = (value[other] - factor * value[row]) % prime
        pivots.append(column)
        row += 1
        if row == rows:
            break
    return value, tuple(pivots)


def _rank(matrix: np.ndarray, prime: int) -> int:
    return len(_rref(matrix, prime)[1])


def _column_basis(matrix: np.ndarray, prime: int) -> np.ndarray:
    value = np.asarray(matrix, dtype=object) % prime
    rows, columns = value.shape
    if columns == 0:
        return np.zeros((rows, 0), dtype=object)
    _, pivots = _rref(value, prime)
    return value[:, pivots] if pivots else np.zeros((rows, 0), dtype=object)


def _nullspace(matrix: np.ndarray, prime: int) -> np.ndarray:
    value = np.asarray(matrix, dtype=object) % prime
    rows, columns = value.shape
    reduced, pivots = _rref(value, prime)
    free = [column for column in range(columns) if column not in pivots]
    if not free:
        return np.zeros((columns, 0), dtype=object)
    basis = np.zeros((columns, len(free)), dtype=object)
    for basis_index, free_column in enumerate(free):
        basis[free_column, basis_index] = 1
        for row, pivot_column in enumerate(pivots):
            basis[pivot_column, basis_index] = (-int(reduced[row, free_column])) % prime
    return basis


def _span(*bases: np.ndarray, prime: int, ambient_dimension: int | None = None) -> np.ndarray:
    nonempty = [np.asarray(basis, dtype=object) for basis in bases if basis.shape[1] > 0]
    if nonempty:
        return _column_basis(np.column_stack(nonempty), prime)
    if ambient_dimension is None:
        ambient_dimension = bases[0].shape[0] if bases else 0
    return np.zeros((ambient_dimension, 0), dtype=object)


def _is_subspace(child: np.ndarray, parent: np.ndarray, prime: int) -> bool:
    return _rank(_span(parent, child, prime=prime, ambient_dimension=parent.shape[0]), prime) == _rank(parent, prime)


def _coordinates(basis: np.ndarray, vector: np.ndarray, prime: int) -> np.ndarray:
    solution = solve_mod_prime(basis, vector, prime)
    if not solution.solvable or solution.particular is None:
        raise ValidationError("vector does not lie in the declared subspace")
    return np.asarray(solution.particular, dtype=object) % prime


def _intersection(left: np.ndarray, right: np.ndarray, prime: int) -> np.ndarray:
    ambient = left.shape[0]
    if right.shape[0] != ambient:
        raise DimensionError("subspace ambient dimensions do not match")
    if left.shape[1] == 0 or right.shape[1] == 0:
        return np.zeros((ambient, 0), dtype=object)
    system = np.column_stack((left, (-right) % prime)) % prime
    null = _nullspace(system, prime)
    if null.shape[1] == 0:
        return np.zeros((ambient, 0), dtype=object)
    left_coefficients = null[: left.shape[1], :]
    return _column_basis((left @ left_coefficients) % prime, prime)


def _image(matrix: np.ndarray, subspace: np.ndarray, prime: int) -> np.ndarray:
    if subspace.shape[1] == 0:
        return np.zeros((matrix.shape[0], 0), dtype=object)
    return _column_basis((matrix @ subspace) % prime, prime)


def _preimage_within(
    matrix: np.ndarray,
    source_subspace: np.ndarray,
    target_subspace: np.ndarray,
    prime: int,
) -> np.ndarray:
    ambient_source = source_subspace.shape[0]
    if source_subspace.shape[1] == 0:
        return np.zeros((ambient_source, 0), dtype=object)
    system = np.column_stack(((matrix @ source_subspace) % prime, (-target_subspace) % prime))
    null = _nullspace(system, prime)
    if null.shape[1] == 0:
        return np.zeros((ambient_source, 0), dtype=object)
    coefficients = null[: source_subspace.shape[1], :]
    return _column_basis((source_subspace @ coefficients) % prime, prime)


def _quotient_complement(space: np.ndarray, subspace: np.ndarray, prime: int) -> np.ndarray:
    if not _is_subspace(subspace, space, prime):
        raise ValidationError("quotient denominator is not a subspace of the numerator")
    current = _column_basis(subspace, prime)
    additions: list[np.ndarray] = []
    current_rank = current.shape[1]
    for index in range(space.shape[1]):
        column = space[:, index : index + 1]
        candidate = _span(current, column, prime=prime, ambient_dimension=space.shape[0])
        if candidate.shape[1] > current_rank:
            additions.append(column)
            current = candidate
            current_rank += 1
    return _column_basis(np.column_stack(additions), prime) if additions else np.zeros((space.shape[0], 0), dtype=object)


@documented(
    ProvenanceRecord(
        identifier="PSC-III-D50",
        title="Finite decreasing filtration over a prime field",
        status=ClaimStatus.DEFINITION,
        source="Paper III runtime",
        assumptions=("explicit nested finite-dimensional subspaces",),
        verification_suite="nonlinear.filtered",
    )
)
class FilteredVectorSpace:
    """A finite decreasing filtration ``F^0 V >= ... >= F^N V = 0``."""

    def __init__(
        self,
        dimension: int,
        levels: Mapping[int, Any],
        *,
        prime: int,
    ) -> None:
        self.prime = require_prime(prime)
        self.dimension = int(dimension)
        if isinstance(dimension, bool) or self.dimension != dimension or self.dimension < 0:
            raise DimensionError("filtered vector-space dimension must be a nonnegative integer")
        if not levels:
            raise ValidationError("a filtration requires explicit levels")
        raw_depths = sorted(int(depth) for depth in levels)
        if raw_depths != list(range(raw_depths[-1] + 1)):
            raise ValidationError("filtration levels must be contiguous from depth zero")
        self.terminal_depth = raw_depths[-1]
        if self.terminal_depth < 1:
            raise ValidationError("a filtration requires at least F^0 and a terminal zero level")
        normalized: dict[int, np.ndarray] = {}
        for depth in raw_depths:
            basis = _mod_matrix(levels[depth], self.prime, name=f"F^{depth} basis")
            if basis.shape[0] != self.dimension:
                raise DimensionError("filtration basis has the wrong ambient dimension")
            normalized[depth] = _column_basis(basis, self.prime)
        if normalized[0].shape[1] != self.dimension:
            raise ValidationError("F^0 must be the full ambient space")
        if normalized[self.terminal_depth].shape[1] != 0:
            raise ValidationError("the terminal filtration level must be zero")
        for depth in range(self.terminal_depth):
            if not _is_subspace(normalized[depth + 1], normalized[depth], self.prime):
                raise ValidationError("filtration levels are not nested")
        self.levels = normalized

    @classmethod
    def from_weights(cls, weights: Sequence[int], *, prime: int) -> "FilteredVectorSpace":
        normalized: list[int] = []
        for raw in weights:
            weight = int(raw)
            if isinstance(raw, bool) or weight != raw or weight < 0:
                raise ValidationError("filtration weights must be nonnegative integers")
            normalized.append(weight)
        dimension = len(normalized)
        terminal = max(normalized, default=0) + 1
        levels: dict[int, np.ndarray] = {}
        for depth in range(terminal + 1):
            columns = [
                np.eye(dimension, dtype=object)[:, index]
                for index, weight in enumerate(normalized)
                if weight >= depth
            ]
            levels[depth] = np.column_stack(columns) if columns else np.zeros((dimension, 0), dtype=object)
        return cls(dimension, levels, prime=prime)

    def basis(self, depth: int) -> np.ndarray:
        p = int(depth)
        if p <= 0:
            return self.levels[0].copy()
        if p >= self.terminal_depth:
            return np.zeros((self.dimension, 0), dtype=object)
        return self.levels[p].copy()

    def dimension_at(self, depth: int) -> int:
        return self.basis(depth).shape[1]

    def contains(self, vector: Any, depth: int) -> bool:
        value = _mod_vector(vector, self.prime, self.dimension)
        return _is_subspace(value.reshape(-1, 1), self.basis(depth), self.prime)

    def graded_basis(self, depth: int) -> np.ndarray:
        p = int(depth)
        if p < 0 or p >= self.terminal_depth:
            return np.zeros((self.dimension, 0), dtype=object)
        return _quotient_complement(self.basis(p), self.basis(p + 1), self.prime)

    def graded_dimension(self, depth: int) -> int:
        return self.graded_basis(depth).shape[1]

    def graded_coordinates(self, vector: Any, depth: int) -> np.ndarray:
        p = int(depth)
        value = _mod_vector(vector, self.prime, self.dimension)
        if not self.contains(value, p):
            raise ValidationError("vector does not lie in the requested filtration level")
        quotient = self.graded_basis(p)
        deeper = self.basis(p + 1)
        combined = np.column_stack((quotient, deeper)) if deeper.shape[1] else quotient
        coordinates = _coordinates(combined, value, self.prime)
        return coordinates[: quotient.shape[1]]

    def lift_graded(self, coordinates: Any, depth: int) -> np.ndarray:
        basis = self.graded_basis(depth)
        values = _mod_vector(coordinates, self.prime, basis.shape[1], name="graded coordinates")
        return (basis @ values) % self.prime

    def certificate(self) -> Certificate:
        return Certificate(
            valid=True,
            exactness=Exactness.EXACT,
            residuals={},
            assumptions_checked=("full F^0", "nested levels", "terminal zero level"),
            diagnostics={
                "prime": self.prime,
                "ambient_dimension": self.dimension,
                "terminal_depth": self.terminal_depth,
                "level_dimensions": tuple(self.dimension_at(depth) for depth in range(self.terminal_depth + 1)),
            },
            theorem_id="PSC-III-D50",
            message="valid finite decreasing filtration",
        )


@dataclass(frozen=True, slots=True)
class DepthObstructionResult:
    """One finite filtration-depth obstruction and its correction space."""


    depth: int
    induced_correction_map: np.ndarray
    defect_coordinates: tuple[int, ...]
    correction_space: ModularLinearSolution
    correction_representatives: tuple[tuple[int, ...], ...]
    certificate: Certificate

    @property
    def vanishes(self) -> bool:
        return self.correction_space.solvable


@documented(
    ProvenanceRecord(
        identifier="PSC-III-T51",
        title="One-step filtered lifting criterion",
        status=ClaimStatus.PROVED,
        source="Paper III",
        assumptions=("filtration-preserving correction map at the requested depth",),
        verification_suite="nonlinear.filtered",
    )
)
class FilteredCorrectionProblem:
    """One associated-graded correction problem over a prime field."""

    def __init__(
        self,
        source: FilteredVectorSpace,
        target: FilteredVectorSpace,
        correction_map: Any,
        defect: Any,
        depth: int,
    ) -> None:
        if source.prime != target.prime:
            raise ValidationError("source and target filtrations must use one prime field")
        self.source = source
        self.target = target
        self.prime = source.prime
        self.correction_map = _mod_matrix(correction_map, self.prime, name="correction map")
        if self.correction_map.shape != (target.dimension, source.dimension):
            raise DimensionError("correction map has incompatible dimensions")
        self.defect = _mod_vector(defect, self.prime, target.dimension, name="defect")
        self.depth = int(depth)
        if isinstance(depth, bool) or self.depth != depth or not (0 <= self.depth < min(source.terminal_depth, target.terminal_depth)):
            raise ValidationError("depth is outside the associated graded filtration range")

    def _preservation_residual(self) -> bool:
        for depth in (self.depth, self.depth + 1):
            image = _image(self.correction_map, self.source.basis(depth), self.prime)
            if not _is_subspace(image, self.target.basis(depth), self.prime):
                return False
        return True

    def induced_map(self) -> np.ndarray:
        if not self._preservation_residual():
            raise ValidationError("correction map does not preserve the requested filtration levels")
        source_basis = self.source.graded_basis(self.depth)
        target_basis = self.target.graded_basis(self.depth)
        target_deeper = self.target.basis(self.depth + 1)
        combined = np.column_stack((target_basis, target_deeper)) if target_deeper.shape[1] else target_basis
        columns: list[np.ndarray] = []
        for index in range(source_basis.shape[1]):
            image = (self.correction_map @ source_basis[:, index]) % self.prime
            coordinates = _coordinates(combined, image, self.prime)
            columns.append(coordinates[: target_basis.shape[1]])
        return np.column_stack(columns) if columns else np.zeros((target_basis.shape[1], 0), dtype=object)

    def solve(self, *, enumerate_limit: int = 100_000) -> DepthObstructionResult:
        if not self.target.contains(self.defect, self.depth):
            raise ValidationError("defect does not lie in the declared filtration depth")
        induced = self.induced_map()
        defect_coordinates = self.target.graded_coordinates(self.defect, self.depth)
        correction_space = solve_mod_prime(induced, (-defect_coordinates) % self.prime, self.prime)
        classes = correction_space.enumerate(limit=enumerate_limit)
        representatives = tuple(
            tuple(int(value) for value in self.source.lift_graded(class_coordinates, self.depth))
            for class_coordinates in classes
        )
        valid = True
        for representative in representatives:
            corrected = (self.defect + self.correction_map @ np.asarray(representative, dtype=object)) % self.prime
            if not self.target.contains(corrected, self.depth + 1):
                valid = False
                break
        return DepthObstructionResult(
            self.depth,
            induced,
            tuple(int(value) for value in defect_coordinates),
            correction_space,
            representatives,
            Certificate(
                valid=valid,
                exactness=Exactness.EXACT,
                residuals={"child_verification": 0.0 if valid else 1.0},
                assumptions_checked=(
                    "filtration preservation at consecutive levels",
                    "associated graded correction equation",
                    "all enumerated corrections improve the defect by one level",
                ),
                diagnostics={
                    "depth": self.depth,
                    "vanishes": correction_space.solvable,
                    "kernel_dimension": correction_space.nullity,
                    "correction_class_count": correction_space.solution_count,
                },
                theorem_id="PSC-III-T51",
                message="depth obstruction and correction coset computed exactly",
            ),
        )


@dataclass(frozen=True, slots=True)
class FiniteFilteredBranchResult:
    """Compatibility result for a branch through a terminal finite filtration."""


    approximations: tuple[tuple[int, ...], ...]
    limit: tuple[int, ...]
    exact_solution: bool
    certificate: Certificate


def verify_finite_filtered_branch(
    source: FilteredVectorSpace,
    target: FilteredVectorSpace,
    map_function: Callable[[tuple[int, ...]], Sequence[int]],
    approximations: Sequence[Sequence[int]],
    *,
    start_depth: int = 0,
) -> FiniteFilteredBranchResult:
    """Verify a compatible branch through a finite terminal filtration."""

    if source.prime != target.prime:
        raise ValidationError("branch source and target must use one prime field")
    if not callable(map_function):
        raise ValidationError("filtered branch map must be callable")
    points = tuple(
        tuple(int(value) for value in _mod_vector(point, source.prime, source.dimension, name="approximation"))
        for point in approximations
    )
    if not points:
        raise ValidationError("a branch requires at least one approximation")
    first_depth = int(start_depth)
    if isinstance(start_depth, bool) or first_depth != start_depth or first_depth < 0:
        raise ValidationError("start_depth must be a nonnegative integer")
    final_depth = first_depth + len(points) - 1
    if final_depth > min(source.terminal_depth, target.terminal_depth):
        raise ValidationError("branch extends beyond the finite terminal filtration")
    compatibility_failures = 0
    defect_failures = 0
    for offset, point in enumerate(points):
        depth = first_depth + offset
        defect = _mod_vector(map_function(point), target.prime, target.dimension, name="branch defect")
        if not target.contains(defect, depth):
            defect_failures += 1
        if offset:
            previous = np.asarray(points[offset - 1], dtype=object)
            difference = (np.asarray(point, dtype=object) - previous) % source.prime
            if not source.contains(difference, depth - 1):
                compatibility_failures += 1
    limit = points[-1]
    final_defect = _mod_vector(map_function(limit), target.prime, target.dimension, name="final defect")
    exact_solution = not np.any(final_defect % target.prime)
    terminal_reached = final_depth >= target.terminal_depth
    valid = compatibility_failures == 0 and defect_failures == 0 and (not terminal_reached or exact_solution)
    return FiniteFilteredBranchResult(
        points,
        limit,
        exact_solution,
        Certificate(
            valid=valid,
            exactness=Exactness.EXACT,
            residuals={
                "compatibility_failures": float(compatibility_failures),
                "defect_depth_failures": float(defect_failures),
                "terminal_defect": 0.0 if exact_solution else 1.0,
            },
            assumptions_checked=("successive congruence compatibility", "defect depth at every finite level"),
            assumptions_unchecked=() if terminal_reached else ("continuation beyond the verified finite prefix",),
            diagnostics={
                "start_depth": first_depth,
                "final_depth": final_depth,
                "terminal_depth": target.terminal_depth,
                "terminal_reached": terminal_reached,
            },
            theorem_id="PSC-III-T52",
            message="finite filtered branch verified" if valid else "filtered branch verification failed",
        ),
    )


@dataclass(frozen=True, slots=True)
class ProjectiveLimitRecord:
    """Declared hypotheses and conclusion for a conditional projective-limit step."""


    verified_prefix_depth: int
    separated: bool
    complete: bool
    continuous: bool
    compatible_infinite_branch: bool
    conclusion: str


def conditional_projective_limit_completion(
    verified_prefix_depth: int,
    *,
    separated: bool,
    complete: bool,
    continuous: bool,
    compatible_infinite_branch: bool,
) -> ConditionalResult[ProjectiveLimitRecord]:
    """Record the conditional infinite filtered-lifting conclusion."""

    depth = int(verified_prefix_depth)
    if isinstance(verified_prefix_depth, bool) or depth != verified_prefix_depth or depth < 0:
        raise ValidationError("verified prefix depth must be a nonnegative integer")
    flags = {
        "separated": bool(separated),
        "complete": bool(complete),
        "continuous": bool(continuous),
        "compatible_infinite_branch": bool(compatible_infinite_branch),
    }
    valid = all(flags.values())
    record = ProjectiveLimitRecord(
        depth,
        flags["separated"],
        flags["complete"],
        flags["continuous"],
        flags["compatible_infinite_branch"],
        "the compatible branch determines an exact inverse-limit solution" if valid else "the inverse-limit conclusion is not licensed",
    )
    unchecked = tuple(name.replace("_", " ") for name, enabled in flags.items() if not enabled)
    if valid:
        unchecked = (
            "the declared infinite branch and topological hypotheses are external inputs, not established by the finite runtime",
        )
    return ConditionalResult(
        record,
        Certificate(
            valid=valid,
            exactness=Exactness.CONDITIONAL,
            residuals={},
            assumptions_checked=tuple(name.replace("_", " ") for name, enabled in flags.items() if enabled),
            assumptions_unchecked=unchecked,
            diagnostics={"verified_prefix_depth": depth, **flags},
            theorem_id="PSC-III-T53",
            message=record.conclusion,
        ),
    )


@dataclass(frozen=True, slots=True)
class SpectralSequenceEntry:
    """One bidegree entry on a finite filtered spectral-sequence page."""


    page: int
    filtration_degree: int
    complementary_degree: int
    total_degree: int
    dimension: int
    outgoing_rank: int
    incoming_rank: int


@dataclass(frozen=True, slots=True)
class SpectralSequencePage:
    """One exact finite page with entry and differential-rank data."""


    page: int
    entries: tuple[SpectralSequenceEntry, ...]
    total_dimension: int
    total_differential_rank: int
    certificate: Certificate

    def entry(self, filtration_degree: int, complementary_degree: int) -> SpectralSequenceEntry:
        for value in self.entries:
            if value.filtration_degree == filtration_degree and value.complementary_degree == complementary_degree:
                return value
        return SpectralSequenceEntry(
            self.page,
            int(filtration_degree),
            int(complementary_degree),
            int(filtration_degree) + int(complementary_degree),
            0,
            0,
            0,
        )


@dataclass(frozen=True, slots=True)
class FilteredSpectralSequenceResult:
    """Finite spectral-sequence pages and limiting dimension comparison."""


    pages: tuple[SpectralSequencePage, ...]
    total_cohomology_dimensions: dict[int, int]
    limiting_dimensions: dict[int, int]
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="PSC-III-S54",
        title="Spectral sequence of a finite filtered complex",
        status=ClaimStatus.CLASSICAL,
        source="Paper III",
        assumptions=("bounded finite cochain complex", "finite decreasing filtration by subcomplexes"),
        verification_suite="nonlinear.filtered",
    )
)
class FiniteFilteredCochainComplex:
    """An exact filtered cochain complex over a prime field."""

    def __init__(
        self,
        dimensions: Mapping[int, int],
        differentials: Mapping[int, Any],
        filtrations: Mapping[int, FilteredVectorSpace],
        *,
        prime: int,
    ) -> None:
        self.prime = require_prime(prime)
        self.dimensions = {int(degree): int(dimension) for degree, dimension in dimensions.items()}
        if not self.dimensions or any(dimension < 0 for dimension in self.dimensions.values()):
            raise ValidationError("filtered complex dimensions must be nonempty and nonnegative")
        if set(filtrations) != set(self.dimensions):
            raise ValidationError("a filtration is required in every cochain degree")
        self.filtrations = dict(filtrations)
        if any(space.prime != self.prime for space in self.filtrations.values()):
            raise ValidationError("all filtered spaces must use the complex prime")
        if any(space.dimension != self.dimensions[degree] for degree, space in self.filtrations.items()):
            raise DimensionError("filtered-space dimensions do not match the complex")
        terminal_depths = {space.terminal_depth for space in self.filtrations.values()}
        if len(terminal_depths) != 1:
            raise ValidationError("every degree must use the same terminal filtration depth")
        self.terminal_depth = terminal_depths.pop()
        self.max_filtration_degree = self.terminal_depth - 1
        self.differentials: dict[int, np.ndarray] = {}
        for degree, raw in differentials.items():
            d = int(degree)
            if d not in self.dimensions or d + 1 not in self.dimensions:
                raise ValidationError("differential degree is outside the complex")
            matrix = _mod_matrix(raw, self.prime, name=f"differential d{d}")
            expected = (self.dimensions[d + 1], self.dimensions[d])
            if matrix.shape != expected:
                raise DimensionError(f"differential d{d} has shape {matrix.shape}, expected {expected}")
            self.differentials[d] = matrix
        for degree in self.dimensions:
            if degree in self.differentials and degree + 1 in self.differentials:
                if np.any((self.differentials[degree + 1] @ self.differentials[degree]) % self.prime):
                    raise ValidationError("successive filtered differentials do not compose to zero")
        for degree, matrix in self.differentials.items():
            for depth in range(self.terminal_depth + 1):
                image = _image(matrix, self.filtrations[degree].basis(depth), self.prime)
                if not _is_subspace(image, self.filtrations[degree + 1].basis(depth), self.prime):
                    raise ValidationError("differential does not preserve the filtration")

    def differential(self, degree: int) -> np.ndarray:
        source = self.dimensions.get(int(degree), 0)
        target = self.dimensions.get(int(degree) + 1, 0)
        return self.differentials.get(int(degree), np.zeros((target, source), dtype=object)).copy()

    def _f(self, degree: int, filtration_degree: int) -> np.ndarray:
        dimension = self.dimensions.get(degree, 0)
        if degree not in self.filtrations:
            return np.zeros((dimension, 0), dtype=object)
        return self.filtrations[degree].basis(filtration_degree)

    def _z(self, page: int, filtration_degree: int, total_degree: int) -> np.ndarray:
        source = self._f(total_degree, filtration_degree)
        if page == 0:
            return source
        matrix = self.differential(total_degree)
        target = self._f(total_degree + 1, filtration_degree + page)
        return _preimage_within(matrix, source, target, self.prime)

    def _b(self, page: int, filtration_degree: int, total_degree: int) -> np.ndarray:
        source = self._f(total_degree - 1, filtration_degree - page)
        image = _image(self.differential(total_degree - 1), source, self.prime)
        return _intersection(self._f(total_degree, filtration_degree), image, self.prime)

    def _numerator_denominator(self, page: int, filtration_degree: int, total_degree: int) -> tuple[np.ndarray, np.ndarray]:
        if page == 0:
            numerator = self._f(total_degree, filtration_degree)
            denominator = self._f(total_degree, filtration_degree + 1)
            return numerator, denominator
        numerator = self._z(page, filtration_degree, total_degree)
        denominator = _span(
            self._z(page - 1, filtration_degree + 1, total_degree),
            self._b(page - 1, filtration_degree, total_degree),
            prime=self.prime,
            ambient_dimension=self.dimensions.get(total_degree, 0),
        )
        if not _is_subspace(denominator, numerator, self.prime):
            raise ArithmeticError("spectral-sequence denominator is not contained in its numerator")
        return numerator, denominator

    def _entry_dimension(self, page: int, filtration_degree: int, total_degree: int) -> int:
        numerator, denominator = self._numerator_denominator(page, filtration_degree, total_degree)
        return numerator.shape[1] - denominator.shape[1]

    def _differential_rank(self, page: int, filtration_degree: int, total_degree: int) -> int:
        source_numerator, _ = self._numerator_denominator(page, filtration_degree, total_degree)
        if source_numerator.shape[1] == 0:
            return 0
        target_p = filtration_degree + page
        target_n = total_degree + 1
        if target_p < 0 or target_p > self.max_filtration_degree or target_n not in self.dimensions:
            return 0
        image = _image(self.differential(total_degree), source_numerator, self.prime)
        _, target_denominator = self._numerator_denominator(page, target_p, target_n)
        rank = _span(
            target_denominator,
            image,
            prime=self.prime,
            ambient_dimension=self.dimensions[target_n],
        ).shape[1] - target_denominator.shape[1]
        return rank

    def page(self, page: int) -> SpectralSequencePage:
        r = int(page)
        if isinstance(page, bool) or r != page or r < 0:
            raise ValidationError("spectral-sequence page must be a nonnegative integer")
        entries: list[SpectralSequenceEntry] = []
        relation_failures = 0
        for total_degree in sorted(self.dimensions):
            for filtration_degree in range(self.max_filtration_degree + 1):
                dimension = self._entry_dimension(r, filtration_degree, total_degree)
                outgoing = self._differential_rank(r, filtration_degree, total_degree)
                incoming = (
                    self._differential_rank(r, filtration_degree - r, total_degree - 1)
                    if filtration_degree - r >= 0 and total_degree - 1 in self.dimensions
                    else 0
                )
                if dimension or outgoing or incoming:
                    entries.append(
                        SpectralSequenceEntry(
                            r,
                            filtration_degree,
                            total_degree - filtration_degree,
                            total_degree,
                            dimension,
                            outgoing,
                            incoming,
                        )
                    )
                if r <= self.max_filtration_degree:
                    next_dimension = self._entry_dimension(r + 1, filtration_degree, total_degree)
                    if next_dimension != dimension - outgoing - incoming:
                        relation_failures += 1
        total_dimension = sum(entry.dimension for entry in entries)
        total_rank = sum(entry.outgoing_rank for entry in entries)
        return SpectralSequencePage(
            r,
            tuple(entries),
            total_dimension,
            total_rank,
            Certificate(
                valid=relation_failures == 0,
                exactness=Exactness.EXACT,
                residuals={"page_cohomology_relation_failures": float(relation_failures)},
                assumptions_checked=("Z_r and B_r subquotients", "induced differential ranks"),
                diagnostics={
                    "page": r,
                    "entry_count": len(entries),
                    "total_dimension": total_dimension,
                    "total_differential_rank": total_rank,
                },
                theorem_id="PSC-III-S54",
                message="valid finite spectral-sequence page" if relation_failures == 0 else "spectral-sequence page relation failed",
            ),
        )

    def total_cohomology_dimensions(self) -> dict[int, int]:
        output: dict[int, int] = {}
        for degree, dimension in self.dimensions.items():
            outgoing = _rank(self.differential(degree), self.prime)
            incoming = _rank(self.differential(degree - 1), self.prime)
            value = dimension - outgoing - incoming
            if value < 0:
                raise ArithmeticError("boundary rank exceeds cycle dimension")
            output[degree] = value
        return output

    def spectral_sequence(self) -> FilteredSpectralSequenceResult:
        pages = tuple(self.page(page) for page in range(self.terminal_depth + 1))
        total = self.total_cohomology_dimensions()
        final_page = pages[-1]
        limiting = {
            degree: sum(entry.dimension for entry in final_page.entries if entry.total_degree == degree)
            for degree in self.dimensions
        }
        mismatches = sum(total[degree] != limiting[degree] for degree in self.dimensions)
        stable_failures = sum(entry.outgoing_rank for entry in final_page.entries)
        valid = mismatches == 0 and stable_failures == 0 and all(page.certificate.valid for page in pages)
        return FilteredSpectralSequenceResult(
            pages,
            total,
            limiting,
            Certificate(
                valid=valid,
                exactness=Exactness.EXACT,
                residuals={
                    "convergence_dimension_mismatches": float(mismatches),
                    "limiting_differential_rank": float(stable_failures),
                },
                assumptions_checked=(
                    "finite filtration by subcomplexes",
                    "page-to-page cohomology identities",
                    "limiting dimensions equal total cohomology dimensions",
                ),
                diagnostics={
                    "prime": self.prime,
                    "terminal_depth": self.terminal_depth,
                    "page_count": len(pages),
                },
                theorem_id="PSC-III-S54",
                message="finite filtered spectral sequence converges dimensionally" if valid else "filtered spectral-sequence validation failed",
            ),
        )


__all__ = [
    "DepthObstructionResult",
    "FilteredCorrectionProblem",
    "FilteredSpectralSequenceResult",
    "FilteredVectorSpace",
    "FiniteFilteredBranchResult",
    "FiniteFilteredCochainComplex",
    "ProjectiveLimitRecord",
    "SpectralSequenceEntry",
    "SpectralSequencePage",
    "conditional_projective_limit_completion",
    "verify_finite_filtered_branch",
]
