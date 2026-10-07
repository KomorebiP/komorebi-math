"""Finite companion forests, Frobenius spines, and trace separation."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from numbers import Rational
from typing import Any, Hashable, Iterable, Mapping, Sequence

from ..certificates import Certificate, Exactness
from ..exceptions import BackendUnavailableError, DimensionError, ValidationError
from .._number_theory import require_integer, require_prime
from .forests import FiniteLiftForest, LiftForestReport


def _sympy() -> Any:
    try:
        import sympy as sp
    except ImportError as exc:
        raise BackendUnavailableError("exact Frobenius separation requires the exact SymPy extra") from exc
    return sp


def _fraction(value: Any) -> Fraction:
    """Convert one rational scalar to an exact Fraction."""

    if isinstance(value, bool) or not isinstance(value, Rational):
        raise TypeError("value is not a rational scalar")
    return Fraction(value)


def _fraction_rank(rows: Sequence[Sequence[Any]]) -> int:
    """Return the exact row rank of a rational matrix."""

    matrix = [[_fraction(value) for value in row] for row in rows]
    if not matrix:
        return 0
    width = len(matrix[0])
    rank = 0
    for column in range(width):
        pivot = next((index for index in range(rank, len(matrix)) if matrix[index][column]), None)
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        scale = matrix[rank][column]
        matrix[rank] = [value / scale for value in matrix[rank]]
        for index in range(len(matrix)):
            if index == rank:
                continue
            factor = matrix[index][column]
            if factor:
                matrix[index] = [left - factor * right for left, right in zip(matrix[index], matrix[rank], strict=True)]
        rank += 1
        if rank == len(matrix):
            break
    return rank


def _fraction_solve(rows: Sequence[Sequence[Any]], rhs_values: Sequence[Any]) -> tuple[Any, ...]:
    """Solve a full column rank rational linear system exactly."""

    matrix = [[_fraction(value) for value in row] for row in rows]
    rhs = [_fraction(value) for value in rhs_values]
    if len(matrix) != len(rhs):
        raise DimensionError("observed trace count does not match the separator")
    if not matrix:
        return ()
    width = len(matrix[0])
    augmented = [row + [value] for row, value in zip(matrix, rhs, strict=True)]
    pivot_rows: dict[int, int] = {}
    rank = 0
    for column in range(width):
        pivot = next((index for index in range(rank, len(augmented)) if augmented[index][column]), None)
        if pivot is None:
            continue
        augmented[rank], augmented[pivot] = augmented[pivot], augmented[rank]
        scale = augmented[rank][column]
        augmented[rank] = [value / scale for value in augmented[rank]]
        for index in range(len(augmented)):
            if index == rank:
                continue
            factor = augmented[index][column]
            if factor:
                augmented[index] = [left - factor * right for left, right in zip(augmented[index], augmented[rank], strict=True)]
        pivot_rows[column] = rank
        rank += 1
    for row in augmented:
        if all(value == 0 for value in row[:width]) and row[width] != 0:
            raise ValidationError("observed traces are inconsistent with the separator")
    if len(pivot_rows) != width:
        raise ValidationError("trace system has unresolved free parameters")
    values = [augmented[pivot_rows[column]][width] for column in range(width)]
    return tuple(int(value) if value.denominator == 1 else value for value in values)


@dataclass(frozen=True, slots=True)
class FrobeniusSpine:
    """One finite algebraic spine of monic Frobenius polynomials."""

    polynomials: Mapping[Hashable, tuple[Any, ...]]
    local_labels: Mapping[Hashable, Any] | None = None

    def __post_init__(self) -> None:
        if not self.polynomials:
            raise ValidationError("a Frobenius spine requires at least one place")
        degrees: set[int] = set()
        for place, coefficients in self.polynomials.items():
            values = tuple(coefficients)
            if len(values) < 2:
                raise ValidationError(f"Frobenius polynomial at {place!r} must have positive degree")
            if values[0] != 1:
                raise ValidationError("Frobenius polynomials use descending monic coefficients")
            degrees.add(len(values) - 1)
        if len(degrees) != 1:
            raise ValidationError("every Frobenius polynomial in one spine must have the same degree")
        object.__setattr__(self, "polynomials", {place: tuple(values) for place, values in self.polynomials.items()})
        object.__setattr__(self, "local_labels", dict(self.local_labels or {}))

    @property
    def rank(self) -> int:
        """Return the common polynomial degree."""

        return len(next(iter(self.polynomials.values()))) - 1

    def agrees(self, other: "FrobeniusSpine", *, places: Iterable[Hashable] | None = None) -> bool:
        """Return whether two spines agree on the declared places."""

        selected = set(self.polynomials) & set(other.polynomials) if places is None else set(places)
        return all(self.polynomials.get(place) == other.polynomials.get(place) for place in selected)


class CompanionForest:
    """A finite local companion forest tied to one Frobenius spine."""

    def __init__(
        self,
        *,
        prime: int,
        spine: FrobeniusSpine,
        forest: FiniteLiftForest[Any],
    ) -> None:
        self.prime = require_prime(prime)
        self.spine = spine
        self.forest = forest

    def report(self) -> LiftForestReport[Any]:
        """Return the finite stable-core report."""

        return self.forest.report()

    @property
    def has_complete_finite_branch(self) -> bool:
        """Return whether the represented finite levels have a full branch."""

        return bool(self.forest.branches())


@dataclass(frozen=True, slots=True)
class FrobeniusSeparationResult:
    """Finite trace matrix and exact multiplicity recovery data."""

    places: tuple[Hashable, ...]
    trace_matrix: Any
    rank: int
    constituent_count: int
    separating: bool
    certificate: Certificate

    def recover(self, observed_traces: Sequence[Any]) -> tuple[Any, ...]:
        """Recover a unique multiplicity vector from observed traces."""

        if not self.separating:
            raise ValidationError("trace matrix does not separate the declared constituents")
        try:
            return _fraction_solve(self.trace_matrix, observed_traces)
        except TypeError:
            sp = _sympy()
            rhs = sp.Matrix(tuple(observed_traces))
            matrix = sp.Matrix(self.trace_matrix)
            if rhs.rows != matrix.rows:
                raise DimensionError("observed trace count does not match the separator")
            solution, parameters = matrix.gauss_jordan_solve(rhs)
            if parameters.rows:
                raise ValidationError("trace system has unresolved free parameters")
            return tuple(sp.simplify(value) for value in solution)


def finite_frobenius_separator(
    traces: Mapping[Hashable, Sequence[Any]],
    *,
    constituent_count: int | None = None,
) -> FrobeniusSeparationResult:
    """Select finitely many supplied places that separate trace columns."""

    if not traces:
        raise ValidationError("Frobenius separation requires trace data")
    rows = [(place, tuple(values)) for place, values in traces.items()]
    width = len(rows[0][1]) if constituent_count is None else require_integer(constituent_count, name="constituent_count")
    if width < 1:
        raise ValidationError("constituent count must be positive")
    if any(len(values) != width for _, values in rows):
        raise DimensionError("every Frobenius trace row must have the same constituent count")
    rational = True
    try:
        for _, values in rows:
            tuple(_fraction(value) for value in values)
    except TypeError:
        rational = False
    sp = None if rational else _sympy()
    selected_places: list[Hashable] = []
    selected_rows: list[tuple[Any, ...]] = []
    current_rank = 0
    for place, values in rows:
        candidate_rows = [*selected_rows, values]
        rank = _fraction_rank(candidate_rows) if rational else int(sp.Matrix(candidate_rows).rank())
        if rank > current_rank:
            selected_places.append(place)
            selected_rows.append(values)
            current_rank = rank
        if current_rank == width:
            break
    matrix: Any = tuple(selected_rows) if rational else sp.Matrix(selected_rows)
    separating = current_rank == width
    return FrobeniusSeparationResult(
        places=tuple(selected_places),
        trace_matrix=matrix,
        rank=current_rank,
        constituent_count=width,
        separating=separating,
        certificate=Certificate(
            valid=separating,
            exactness=Exactness.EXACT,
            assumptions_checked=("finite supplied Frobenius trace table", "exact column rank"),
            assumptions_unchecked=("Chebotarev density beyond the supplied table",),
            diagnostics={
                "selected_places": tuple(selected_places),
                "rank": current_rank,
                "constituents": width,
                "rank_deficiency": width - current_rank,
                "separating": separating,
            },
            theorem_id="PSC-X-FROBENIUS-SEPARATOR",
            message="finite trace separator constructed" if separating else "supplied trace rows do not separate every constituent",
        ),
    )


@dataclass(frozen=True, slots=True)
class CompatibleCarrier:
    """A declared finite compatible carrier and constituent multiplicities."""

    constituents: tuple[str, ...]
    capacities: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.constituents or len(self.constituents) != len(self.capacities):
            raise ValidationError("carrier constituents and capacities must be nonempty and aligned")
        capacities = tuple(require_integer(value, name="capacity") for value in self.capacities)
        if any(value < 0 for value in capacities):
            raise ValidationError("carrier capacities must be nonnegative")
        if len(set(self.constituents)) != len(self.constituents):
            raise ValidationError("carrier constituent names must be unique")
        object.__setattr__(self, "capacities", capacities)

    def select(self, multiplicities: Sequence[int]) -> dict[str, int]:
        """Return one valid compatible summand multiplicity dictionary."""

        values = tuple(require_integer(value, name="multiplicity") for value in multiplicities)
        if len(values) != len(self.capacities):
            raise DimensionError("multiplicity vector has the wrong length")
        if any(value < 0 or value > capacity for value, capacity in zip(values, self.capacities, strict=True)):
            raise ValidationError("multiplicity vector lies outside the declared carrier")
        return dict(zip(self.constituents, values, strict=True))


__all__ = [
    "CompanionForest",
    "CompatibleCarrier",
    "FrobeniusSeparationResult",
    "FrobeniusSpine",
    "finite_frobenius_separator",
]
