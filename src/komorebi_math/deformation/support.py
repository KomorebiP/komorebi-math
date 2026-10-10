"""Finite presentation, Fitting, support, and occupancy calculations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Mapping

import numpy as np

from ..certificates import Certificate, Exactness
from ..exceptions import BackendUnavailableError, DimensionError, ValidationError
from .._number_theory import require_integer, require_prime
from .complexes import row_rank_mod


def _sympy() -> Any:
    try:
        import sympy as sp
    except ImportError as exc:
        raise BackendUnavailableError("presentation and Fitting operations require the exact SymPy extra") from exc
    return sp


@dataclass(frozen=True, slots=True)
class FittingGenerators:
    """A finite generating family of minors for one Fitting ideal."""

    order: int
    minor_size: int
    generators: tuple[Any, ...]

    @property
    def is_unit_ideal(self) -> bool:
        """Return whether a unit occurs among the supplied generators."""

        sp = _sympy()
        return any(sp.sympify(generator) in (sp.Integer(1), sp.Integer(-1)) for generator in self.generators)


@dataclass(frozen=True, slots=True)
class FiberSupportResult:
    """Fiber dimension and support status at one finite point."""

    point: Mapping[str, int | complex | float]
    generator_count: int
    relation_rank: int
    fiber_dimension: int
    occupied: bool
    certificate: Certificate


class PresentedModule:
    """A finite module presented by a matrix of relations.

    The matrix has one row for each module generator and one column for each
    relation.  The module is the cokernel of the corresponding map.
    """

    def __init__(self, presentation: Any, *, variables: Iterable[Any] = ()) -> None:
        sp = _sympy()
        self.presentation = sp.Matrix(presentation)
        self.variables = tuple(variables)
        self.generator_count = self.presentation.rows
        self.relation_count = self.presentation.cols

    def fitting_generators(self, order: int = 0) -> FittingGenerators:
        """Return the determinantal generators of the declared Fitting ideal."""

        index = require_integer(order, name="order")
        if index < 0:
            raise ValidationError("Fitting order must be nonnegative")
        size = self.generator_count - index
        if size <= 0:
            return FittingGenerators(index, 0, (1,))
        if size > min(self.generator_count, self.relation_count):
            return FittingGenerators(index, size, ())
        generators: list[Any] = []
        for rows in combinations(range(self.generator_count), size):
            for columns in combinations(range(self.relation_count), size):
                determinant = self.presentation.extract(rows, columns).det().expand()
                generators.append(determinant)
        unique: list[Any] = []
        for generator in generators:
            if generator not in unique:
                unique.append(generator)
        return FittingGenerators(index, size, tuple(unique))

    def evaluate(self, substitutions: Mapping[Any, Any]) -> Any:
        """Evaluate the presentation matrix at one point."""

        return self.presentation.subs(dict(substitutions))

    def fiber(
        self,
        substitutions: Mapping[Any, Any],
        *,
        prime: int | None = None,
    ) -> FiberSupportResult:
        """Return the module fiber dimension and support status at one point."""

        sp = _sympy()
        evaluated = self.evaluate(substitutions)
        if any(entry.free_symbols for entry in evaluated):
            raise ValidationError("fiber substitutions do not evaluate every symbolic entry")
        if prime is None:
            rank = int(evaluated.rank())
            exactness = Exactness.EXACT
            assumptions = ("exact characteristic zero fiber",)
        else:
            p = require_prime(prime)
            integer_matrix = np.empty(evaluated.shape, dtype=object)
            for index, entry in np.ndenumerate(np.asarray(evaluated.tolist(), dtype=object)):
                rational = sp.Rational(entry)
                denominator = int(rational.q) % p
                if denominator == 0:
                    raise ValidationError("fiber entry has denominator divisible by the residue characteristic")
                integer_matrix[index] = int(rational.p) * pow(denominator, -1, p)
            rank = row_rank_mod(integer_matrix, p)
            exactness = Exactness.EXACT
            assumptions = ("prime field fiber",)
        dimension = self.generator_count - rank
        occupied = dimension > 0
        fitting = self.fitting_generators(0)
        fitting_values = tuple(sp.sympify(generator).subs(dict(substitutions)) for generator in fitting.generators)
        if prime is None:
            fitting_vanishes = all(value == 0 for value in fitting_values)
        else:
            p = require_prime(prime)
            fitting_vanishes = True
            for value in fitting_values:
                rational = sp.Rational(value)
                denominator = int(rational.q) % p
                if denominator == 0:
                    raise ValidationError("Fitting value has a nonunit denominator at the point")
                residue = int(rational.p) * pow(denominator, -1, p)
                fitting_vanishes = fitting_vanishes and residue % p == 0
        valid = occupied == fitting_vanishes
        return FiberSupportResult(
            point={str(key): value for key, value in substitutions.items()},
            generator_count=self.generator_count,
            relation_rank=rank,
            fiber_dimension=dimension,
            occupied=occupied,
            certificate=Certificate(
                valid=valid,
                exactness=exactness,
                assumptions_checked=assumptions + ("zeroth Fitting minors",),
                diagnostics={
                    "rank": rank,
                    "fiber_dimension": dimension,
                    "fitting_generator_count": len(fitting.generators),
                    "fitting_vanishes": fitting_vanishes,
                },
                theorem_id="PSC-X-FITTING-SUPPORT",
                message="fiber occupancy agrees with the zeroth Fitting criterion" if valid else "Fitting support criterion failed",
            ),
        )

    def finite_support(
        self,
        points: Iterable[Mapping[Any, Any]],
        *,
        prime: int | None = None,
    ) -> tuple[FiberSupportResult, ...]:
        """Evaluate support on a declared finite set of points."""

        return tuple(self.fiber(point, prime=prime) for point in points)


@dataclass(frozen=True, slots=True)
class NestedOccupancyResult:
    """Intersection dimensions for nested finite-dimensional carriers."""

    dimensions: tuple[int, ...]
    intersection_dimension: int
    stabilized_at: int | None
    certificate: Certificate


def nested_subspace_occupancy(
    bases: Iterable[Any],
    *,
    tolerance: float = 1e-10,
) -> NestedOccupancyResult:
    """Compute the intersection of a finite sequence of column spaces.

    ``stabilized_at`` records the first unchanged intersection in the final
    constant run of the supplied finite sequence. It is ``None`` when the
    final step shrinks the intersection or no comparison was supplied. This
    finite observation does not certify stability at any unseen later level.
    """

    matrices = [np.asarray(matrix, dtype=complex) for matrix in bases]
    if not matrices:
        raise ValidationError("nested occupancy requires at least one subspace")
    if not np.isfinite(tolerance) or not 0 <= tolerance < 1:
        raise ValidationError("occupancy tolerance must be finite and lie in [0, 1)")
    if any(matrix.ndim != 2 for matrix in matrices):
        raise DimensionError("every subspace basis must be a matrix")
    ambient = matrices[0].shape[0]
    if any(matrix.shape[0] != ambient for matrix in matrices):
        raise DimensionError("every subspace basis must use one ambient dimension")
    if any(not np.all(np.isfinite(matrix)) for matrix in matrices):
        raise ValidationError("subspace bases must have finite entries")

    def orthonormal_columns(matrix: np.ndarray) -> np.ndarray:
        if matrix.shape[1] == 0:
            return np.zeros((ambient, 0), dtype=complex)
        # Unpivoted QR followed by the first ``rank`` columns can lose the
        # actual span when zero or dependent columns precede an independent
        # one. Left singular vectors select the full column space directly.
        left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
        return left[:, singular_values > tolerance]

    current = orthonormal_columns(matrices[0])
    dimensions = [current.shape[1]]
    for matrix in matrices[1:]:
        right = orthonormal_columns(matrix)
        if current.shape[1] == 0 or right.shape[1] == 0:
            current = np.zeros((ambient, 0), dtype=complex)
        else:
            u, singular_values, vh = np.linalg.svd(current.conj().T @ right)
            indices = np.flatnonzero(singular_values >= 1.0 - tolerance)
            current = current @ u[:, indices]
        dimensions.append(current.shape[1])
    stabilized_at = next(
        (
            index
            for index in range(1, len(dimensions))
            if dimensions[index - 1] == dimensions[index]
            and all(dimension == dimensions[index] for dimension in dimensions[index:])
        ),
        None,
    )
    nested = all(dimensions[index + 1] <= dimensions[index] for index in range(len(dimensions) - 1))
    return NestedOccupancyResult(
        dimensions=tuple(dimensions),
        intersection_dimension=current.shape[1],
        stabilized_at=stabilized_at,
        certificate=Certificate(
            valid=nested,
            exactness=Exactness.NUMERICAL,
            tolerance=tolerance,
            assumptions_checked=("finite-dimensional column spaces", "numerical principal-angle intersection"),
            diagnostics={
                "dimensions": tuple(dimensions),
                "stabilized_at": stabilized_at,
                "stability_scope": "supplied finite prefix only",
                "span_method": "singular value decomposition",
            },
            theorem_id="PSC-X-FINITE-OCCUPANCY",
            message="finite-dimensional occupancy intersection computed" if nested else "intersection dimensions are inconsistent",
        ),
    )


__all__ = [
    "FiberSupportResult",
    "FittingGenerators",
    "NestedOccupancyResult",
    "PresentedModule",
    "nested_subspace_occupancy",
]
