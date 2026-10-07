"""Finite dimensional seed systems and generator calculations."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import ceil
from typing import Any

from ..backends import LinearAlgebraBackend, get_backend
from ..certificates import Certificate, ComputationResult, Exactness
from ..config import get_settings
from ..exceptions import BackendUnavailableError, DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented


@dataclass(frozen=True, slots=True)
class FreePresentation:
    """Finite evaluation matrix and kernel for a supplied algebra basis."""

    evaluation_matrix: Any
    kernel_basis: tuple[Any, ...]
    rank: int
    domain_dimension: int
    codomain_dimension: int
    backend: str

    @property
    def nullity(self) -> int:
        return self.domain_dimension - self.rank


@dataclass(frozen=True, slots=True)
class StandardSeedSearchResult:
    count: int
    indices: tuple[int, ...]
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="PSC-I-T02",
        title="Finite seed hull generation",
        status=ClaimStatus.PROVED,
        source="Paper I",
        assumptions=("finite dimensional realized operator family",),
        limitations=("numerical rank depends on the declared tolerance",),
        verification_suite="core.seeds",
    )
)
class SeedSystem:
    """A finite family of operators acting on a finite seed family."""

    def __init__(
        self,
        operators: Mapping[str, Any] | Sequence[Any],
        seeds: Sequence[Any],
        *,
        backend: str | LinearAlgebraBackend | None = None,
        tolerance: float | None = None,
    ) -> None:
        if not seeds:
            raise ValidationError("a seed system requires at least one seed")
        self.backend = get_backend(backend)
        self.tolerance = get_settings().tolerance if tolerance is None else float(tolerance)

        if isinstance(operators, Mapping):
            operator_items = tuple((str(name), value) for name, value in operators.items())
        else:
            operator_items = tuple((f"A{index}", value) for index, value in enumerate(operators))
        self.operators: dict[str, Any] = {
            name: self.backend.as_matrix(value) for name, value in operator_items
        }

        self.seeds = tuple(self.backend.as_vector(seed) for seed in seeds)
        first_shape = self.backend.shape(self.seeds[0])
        self.dimension = first_shape[0]
        if any(self.backend.shape(seed) != first_shape for seed in self.seeds):
            raise DimensionError("all seeds must have one common dimension")
        for name, operator in self.operators.items():
            if self.backend.shape(operator) != (self.dimension, self.dimension):
                raise DimensionError(
                    f"operator {name!r} must have shape {(self.dimension, self.dimension)}"
                )

    @property
    def exact(self) -> bool:
        return self.backend.exact

    def _rank(self, vectors: list[Any]) -> int:
        if not vectors:
            return 0
        return self.backend.rank(self.backend.column_stack(vectors), tolerance=self.tolerance)

    def hull_vectors(self, *, max_rounds: int | None = None) -> tuple[Any, ...]:
        """Return a maximal independent orbit family."""
        basis: list[Any] = []
        frontier: deque[Any] = deque()

        def add(candidate: Any) -> bool:
            vector = self.backend.as_vector(candidate)
            if self.backend.is_zero(vector, tolerance=self.tolerance):
                return False
            old_rank = self._rank(basis)
            new_rank = self._rank([*basis, vector])
            if new_rank > old_rank:
                basis.append(vector)
                frontier.append(vector)
                return True
            return False

        for seed in self.seeds:
            add(seed)

        limit = (
            self.dimension * max(1, len(self.operators)) + 2
            if max_rounds is None
            else int(max_rounds)
        )
        rounds = 0
        while frontier and len(basis) < self.dimension and rounds < limit:
            layer_size = len(frontier)
            for _ in range(layer_size):
                vector = frontier.popleft()
                for operator in self.operators.values():
                    add(self.backend.matmul(operator, vector))
            rounds += 1
        return tuple(basis)

    def hull_basis(self, *, max_rounds: int | None = None) -> Any:
        vectors = list(self.hull_vectors(max_rounds=max_rounds))
        if not vectors:
            return self.backend.zeros(self.dimension, 0)
        return self.backend.column_stack(vectors)

    def hull_dimension(self, *, max_rounds: int | None = None) -> int:
        return len(self.hull_vectors(max_rounds=max_rounds))

    def is_generating(self) -> bool:
        return self.hull_dimension() == self.dimension

    def generation_certificate(self) -> Certificate:
        dimension = self.hull_dimension()
        valid = dimension == self.dimension
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT if self.backend.exact else Exactness.NUMERICAL,
            tolerance=None if self.backend.exact else self.tolerance,
            assumptions_checked=("operator dimensions", "seed dimensions", "finite orbit closure"),
            diagnostics={
                "ambient_dimension": self.dimension,
                "hull_dimension": dimension,
                "operator_count": len(self.operators),
                "seed_count": len(self.seeds),
                "backend": self.backend.name,
            },
            theorem_id="PSC-I-T02",
            message="the seeds generate the ambient space" if valid else "the seed hull is proper",
        )

    def orbit_layers(self, *, max_rounds: int | None = None) -> tuple[tuple[Any, ...], ...]:
        """Return independent vectors grouped by the round in which they appear."""
        basis: list[Any] = []
        current: list[Any] = []
        layers: list[tuple[Any, ...]] = []

        def independent(candidate: Any) -> bool:
            vector = self.backend.as_vector(candidate)
            if self.backend.is_zero(vector, tolerance=self.tolerance):
                return False
            if self._rank([*basis, vector]) > self._rank(basis):
                basis.append(vector)
                return True
            return False

        for seed in self.seeds:
            if independent(seed):
                current.append(seed)
        if current:
            layers.append(tuple(current))

        limit = (
            self.dimension * max(1, len(self.operators)) + 2
            if max_rounds is None
            else int(max_rounds)
        )
        rounds = 0
        while current and len(basis) < self.dimension and rounds < limit:
            next_layer: list[Any] = []
            for vector in current:
                for operator in self.operators.values():
                    candidate = self.backend.matmul(operator, vector)
                    if independent(candidate):
                        next_layer.append(candidate)
            if next_layer:
                layers.append(tuple(next_layer))
            current = next_layer
            rounds += 1
        return tuple(layers)

    def presentation(self, algebra_basis: Sequence[Any]) -> FreePresentation:
        """Build the evaluation map for a supplied finite algebra basis.

        The caller is responsible for supplying an algebra basis or spanning
        family. The method does not infer the full operator algebra.
        """
        matrices = [self.backend.as_matrix(value) for value in algebra_basis]
        if not matrices:
            raise ValidationError("an algebra basis requires at least one matrix")
        if any(self.backend.shape(matrix) != (self.dimension, self.dimension) for matrix in matrices):
            raise DimensionError("every algebra basis matrix must act on the seed space")
        columns = [
            self.backend.matmul(matrix, seed)
            for seed in self.seeds
            for matrix in matrices
        ]
        evaluation = self.backend.column_stack(columns)
        rank = self.backend.rank(evaluation, tolerance=self.tolerance)
        kernel = tuple(self.backend.nullspace(evaluation, tolerance=self.tolerance))
        return FreePresentation(
            evaluation_matrix=evaluation,
            kernel_basis=kernel,
            rank=rank,
            domain_dimension=len(columns),
            codomain_dimension=self.dimension,
            backend=self.backend.name,
        )

    def minimum_standard_basis_seeds(
        self,
        *,
        max_dimension: int = 12,
        force: bool = False,
    ) -> StandardSeedSearchResult:
        """Search standard basis subsets for a smallest generating family.

        This is an exponential diagnostic, not a general seed rank algorithm.
        """
        if self.dimension > max_dimension and not force:
            raise ValidationError(
                f"standard basis search is exponential and is limited to dimension {max_dimension}"
            )
        identity = self.backend.eye(self.dimension)
        standard = [identity[:, index] for index in range(self.dimension)]
        for count in range(1, self.dimension + 1):
            for indices in combinations(range(self.dimension), count):
                candidate = SeedSystem(
                    self.operators,
                    [standard[index] for index in indices],
                    backend=self.backend,
                    tolerance=self.tolerance,
                )
                if candidate.is_generating():
                    certificate = candidate.generation_certificate()
                    return StandardSeedSearchResult(count, tuple(indices), certificate)
        certificate = Certificate(
            valid=False,
            exactness=Exactness.EXACT if self.backend.exact else Exactness.NUMERICAL,
            tolerance=None if self.backend.exact else self.tolerance,
            diagnostics={"ambient_dimension": self.dimension},
            message="no standard basis subset generated the ambient space",
        )
        return StandardSeedSearchResult(self.dimension + 1, (), certificate)


@documented(
    ProvenanceRecord(
        identifier="PSC-I-T10",
        title="Semisimple seed rank formula",
        status=ClaimStatus.PROVED,
        source="Paper I",
        assumptions=("semisimple Artinian acting algebra",),
        verification_suite="core.seed_rank",
    )
)
def semisimple_seed_rank(block_dimensions: Sequence[int], multiplicities: Sequence[int]) -> int:
    """Return the generator number from semisimple block multiplicities."""

    dimensions = [int(value) for value in block_dimensions]
    counts = [int(value) for value in multiplicities]
    if not dimensions or len(dimensions) != len(counts):
        raise ValidationError("matching nonempty block dimensions and multiplicities are required")
    if any(value <= 0 for value in dimensions) or any(value < 0 for value in counts):
        raise ValidationError("block dimensions must be positive and multiplicities nonnegative")
    return max(ceil(count / dimension) for dimension, count in zip(dimensions, counts, strict=True))


def _sympy() -> Any:
    try:
        import sympy as sp
    except ImportError as exc:
        raise BackendUnavailableError(
            "single_operator_seed_rank requires the 'exact' optional dependency"
        ) from exc
    return sp


def _polynomial_at_matrix(poly: Any, matrix: Any, variable: Any) -> Any:
    sp = _sympy()
    coefficients = sp.Poly(poly, variable).all_coeffs()
    result = sp.zeros(matrix.rows)
    identity = sp.eye(matrix.rows)
    for coefficient in coefficients:
        result = result * matrix + coefficient * identity
    return result


@documented(
    ProvenanceRecord(
        identifier="PSC-I-T13",
        title="Single operator primary seed rank formula",
        status=ClaimStatus.PROVED,
        source="Paper I",
        assumptions=("exact matrix over the rational field",),
        limitations=("the current executable factorization path is implemented over rational entries",),
        verification_suite="core.seed_rank",
    )
)
def single_operator_seed_rank(operator: Any) -> int:
    """Return the exact generator number for a rational matrix as a Q[t] module.

    For every irreducible factor ``p`` of the characteristic polynomial, the
    number of primary cyclic blocks is ``nullity(p(T)) / degree(p)``. The seed
    rank is the largest such number.
    """
    sp = _sympy()
    matrix = sp.Matrix(operator)
    if matrix.rows != matrix.cols:
        raise DimensionError("operator must be square")
    if matrix.rows == 0:
        return 0
    if any(entry not in sp.QQ for entry in matrix):
        raise ValidationError("the executable exact formula currently requires rational entries")
    variable = sp.symbols("t")
    characteristic = sp.Poly(matrix.charpoly(variable).as_expr(), variable, domain=sp.QQ)
    factors = sp.factor_list(characteristic)[1]
    ranks: list[int] = []
    for factor, _ in factors:
        evaluated = _polynomial_at_matrix(factor.as_expr(), matrix, variable)
        nullity = matrix.rows - evaluated.rank()
        degree = factor.degree()
        if nullity % degree != 0:
            raise ArithmeticError("primary nullity is not divisible by the irreducible degree")
        ranks.append(nullity // degree)
    return max(ranks, default=0)


def cyclic_span(
    operators: Mapping[str, Any] | Sequence[Any],
    seeds: Sequence[Any],
    *,
    backend: str | LinearAlgebraBackend | None = None,
    tolerance: float | None = None,
) -> ComputationResult[Any]:
    """Compute the finite operator hull generated by the supplied seeds."""

    system = SeedSystem(operators, seeds, backend=backend, tolerance=tolerance)
    basis = system.hull_basis()
    return ComputationResult(basis, system.generation_certificate())


def shared_direct_sum_seed_bounds(seed_ranks: Sequence[int]) -> tuple[int, int]:
    """Return the general lower and upper bounds for a shared acting algebra."""
    ranks = tuple(int(rank) for rank in seed_ranks)
    if not ranks or any(rank < 0 for rank in ranks):
        raise ValidationError("nonnegative seed ranks are required")
    return max(ranks), sum(ranks)


def independent_direct_sum_seed_rank(seed_ranks: Sequence[int]) -> int:
    """Return the exact generator number for independent product actions."""
    ranks = tuple(int(rank) for rank in seed_ranks)
    if not ranks or any(rank < 0 for rank in ranks):
        raise ValidationError("nonnegative seed ranks are required")
    return sum(ranks)


def tensor_seed_rank_upper_bound(left_rank: int, right_rank: int) -> int:
    """Return the product upper bound for tensor product seed rank."""

    if left_rank < 0 or right_rank < 0:
        raise ValidationError("seed ranks must be nonnegative")
    return int(left_rank) * int(right_rank)
