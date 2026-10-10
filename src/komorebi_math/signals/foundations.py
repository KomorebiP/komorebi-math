"""Exact finite realization and predictive-closure primitives.

These functions implement the rational matrix sector of Large Signal Model
Foundations 0.8.0, Propositions 7, 8, 10, 14 and 15. They check the supplied
finite operators or sample pairs. A fit does not establish coverage of an
unknown process. SymPy is loaded only when one of these functions is called.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
from types import MappingProxyType
from typing import Any

from ..certificates import Certificate, Exactness
from ..exceptions import BackendUnavailableError, DimensionError, ResourceLimitError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented


def _sympy() -> Any:
    try:
        import sympy
    except ImportError as exc:
        raise BackendUnavailableError(
            "finite signal foundations requires the 'exact' optional dependency"
        ) from exc
    return sympy


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValidationError(f"{name} must be an integer at least {minimum}")
    return int(value)


def _matrix(value: Any, name: str, *, max_entries: int = 65_536) -> Any:
    sp = _sympy()
    # Inspect ndarray/SymPy dimensions before their potentially large .tolist()
    # copies. A one-dimensional input is interpreted as a column, as below.
    shape = getattr(value, "shape", None)
    if shape is not None:
        if (
            not isinstance(shape, tuple)
            or len(shape) not in (1, 2)
            or any(not isinstance(size, Integral) or isinstance(size, bool) or size <= 0 for size in shape)
        ):
            raise DimensionError(f"{name} must be a matrix with positive dimensions")
        # Own accepted dimension scalars before multiplication: np.int64
        # dimensions can otherwise wrap and bypass the pre-copy entry cap.
        shaped_rows, shaped_cols = tuple(int(size) for size in shape) if len(shape) == 2 else (int(shape[0]), 1)
        if shaped_rows * shaped_cols > max_entries:
            raise ResourceLimitError(f"{name} exceeds the {max_entries}-entry exact matrix limit")
    raw = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(raw, (list, tuple)):
        raise DimensionError(f"{name} must be a rectangular rational matrix")
    if not raw:
        raise DimensionError(f"{name} must have positive dimensions")
    if raw and isinstance(raw[0], (list, tuple)):
        rows, cols = len(raw), len(raw[0])
        if any(not isinstance(row, (list, tuple)) or len(row) != cols for row in raw):
            raise DimensionError(f"{name} must be a rectangular rational matrix")
    else:
        rows, cols = len(raw), 1
        if any(isinstance(entry, (list, tuple)) for entry in raw):
            raise DimensionError(f"{name} must be a rectangular rational matrix")
    if cols == 0:
        raise DimensionError(f"{name} must have positive dimensions")
    if rows * cols > max_entries:
        raise ResourceLimitError(f"{name} exceeds the {max_entries}-entry exact matrix limit")
    entries = (entry for row in raw for entry in row) if isinstance(raw[0], (list, tuple)) else iter(raw)
    if any(
        isinstance(entry, bool) or not isinstance(entry, (Integral, Fraction, sp.Rational))
        for entry in entries
    ):
        raise ValidationError(
            f"{name} entries must be exact rational numbers, never floats, strings or symbols"
        )
    try:
        matrix = sp.Matrix(raw)
    except (TypeError, ValueError) as exc:
        raise DimensionError(f"{name} must be a rectangular rational matrix") from exc
    # No nsimplify, denominator guessing, string evaluation or float promotion.
    # Matrix construction sympifies Fraction and integer inputs exactly.
    if any(not isinstance(entry, sp.Rational) for entry in matrix):
        raise ValidationError(f"{name} entries must be exact rational numbers, never floats or symbols")
    return sp.ImmutableMatrix(matrix)


def _square(value: Any, name: str) -> Any:
    matrix = _matrix(value, name)
    if matrix.rows != matrix.cols:
        raise DimensionError(f"{name} must be square")
    return matrix


def _projector(value: Any, dimension: int) -> Any:
    matrix = _matrix(value, "retained_projector")
    if matrix.shape != (dimension, dimension):
        raise DimensionError("retained projector dimension does not match the operator")
    if matrix * matrix != matrix or matrix != matrix.T:
        raise ValidationError("retained_projector must be an exact orthogonal projector")
    return matrix


def _zero(matrix: Any) -> bool:
    return all(entry == 0 for entry in matrix)


def _size(matrix: Any) -> Fraction:
    value = max((abs(entry) for entry in matrix), default=0)
    return Fraction(int(value.p), int(value.q)) if value else Fraction(0)


def _certificate(
    identifier: str,
    residuals: Mapping[str, Any],
    *,
    valid: bool = True,
    checked: Sequence[str] = (),
    unchecked: Sequence[str] = (),
    diagnostics: Mapping[str, Any] | None = None,
    message: str,
) -> Certificate:
    return Certificate(
        valid=valid,
        exactness=Exactness.EXACT,
        residuals={name: _size(value) for name, value in residuals.items()},
        assumptions_checked=("exact rational finite matrices", *checked),
        assumptions_unchecked=tuple(unchecked),
        diagnostics=dict(diagnostics or {}),
        theorem_id=identifier,
        message=message,
    )


def _row_basis(matrix: Any) -> Any:
    sp = _sympy()
    reduced, pivots = matrix.rref()
    return sp.ImmutableMatrix(reduced[: len(pivots), :])


def _coordinates(rows: Any, basis: Any) -> Any:
    """Return canonical candidate coordinates and let callers check residuals."""
    sp = _sympy()
    if basis.rows == 0:
        return sp.ImmutableMatrix(sp.zeros(rows.rows, 0))
    _, pivot_columns = basis.rref()
    pivot_matrix = basis[:, list(pivot_columns)]
    return sp.ImmutableMatrix(rows[:, list(pivot_columns)] * pivot_matrix.inv())


@dataclass(frozen=True, slots=True)
class CompressionClosure:
    """A computed obstruction; a valid identity certificate can be nonclosed."""

    closed: bool
    closure_defect: Any
    composition_defect: Any
    certificate: Certificate

    def require_closed(self) -> CompressionClosure:
        self.certificate.require()
        if not self.closed:
            raise ValidationError("the retained state is not closed: PL(I-P) is nonzero")
        return self


@documented(
    ProvenanceRecord(
        identifier="KLSM-0.8-P07",
        title="Linear closure and compositional defect",
        status=ClaimStatus.FINITE_CERTIFICATE,
        source="Large Signal Model Foundations 0.8.0, Section 5.1, Proposition 7",
        assumptions=("supplied rational finite operators", "orthogonal retained projection"),
        limitations=("checks the supplied operators, not an unknown generating process",),
        verification_suite="signals.foundations",
    )
)
def compression_closure(
    operator: Any, retained_projector: Any, *, inner_operator: Any | None = None
) -> CompressionClosure:
    """Compute ``PLQ`` and verify ``PLMP - PLPMP = PLQMP``.

    ``operator`` is L and ``inner_operator`` is M (default L). The certificate
    certifies the defect identity. Use ``require_closed`` to require PLQ=0.
    """
    sp = _sympy()
    left = _square(operator, "operator")
    projection = _projector(retained_projector, left.rows)
    right = left if inner_operator is None else _square(inner_operator, "inner_operator")
    if right.shape != left.shape:
        raise DimensionError("inner and outer operators must have equal dimensions")
    unresolved = sp.eye(left.rows) - projection
    closure = sp.ImmutableMatrix(projection * left * unresolved)
    composition = sp.ImmutableMatrix(
        projection * left * right * projection
        - (projection * left * projection) * (projection * right * projection)
    )
    identity = composition - projection * left * unresolved * right * projection
    cert = _certificate(
        "KLSM-0.8-P07",
        {"composition_identity": identity},
        valid=_zero(identity),
        checked=("orthogonal projector", "composition defect identity"),
        diagnostics={"closed": _zero(closure), "ambient_dimension": left.rows},
        message="finite compression defect computed exactly",
    )
    return CompressionClosure(_zero(closure), closure, composition, cert)


@dataclass(frozen=True, slots=True)
class PredictiveCompletion:
    """Constructed closed row space, or a partial space after a round cap."""

    basis: Any
    original_decoder: Any
    reduced_operators: Mapping[str, Any]
    dimension_chain: tuple[int, ...]
    rounds: int
    round_limit: int
    complete: bool
    certificate: Certificate

    def require_complete(self) -> PredictiveCompletion:
        self.certificate.require()
        if not self.complete:
            raise ResourceLimitError("predictive completion stopped before exact invariance was established")
        return self


@documented(
    ProvenanceRecord(
        identifier="KLSM-0.8-P08",
        title="Minimal linear predictive completion",
        status=ClaimStatus.FINITE_CERTIFICATE,
        source="Large Signal Model Foundations 0.8.0, Section 6, Proposition 8",
        assumptions=("known finite rational operator family",),
        limitations=("does not identify unknown dynamics or make added observables accessible",),
        verification_suite="signals.foundations",
    )
)
def minimal_predictive_completion(
    summary: Any, operators: Mapping[str, Any], *, max_rounds: int | None = None
) -> PredictiveCompletion:
    """Construct ``W <- W + sum(W L_u)`` by exact row reduction.

    The default round cap is the dimension bound plus one stability check.
    An explicit smaller cap returns an invalid incomplete certificate; it never
    silently claims minimality of a truncated computation.
    """
    initial = _matrix(summary, "summary")
    if (
        not isinstance(operators, Mapping)
        or not operators
        or any(not isinstance(name, str) or not name for name in operators)
    ):
        raise ValidationError("operators must be a nonempty mapping with nonempty string names")
    if len(operators) > 1_024 or (len(operators) + 1) * initial.cols**2 > 1_000_000:
        raise ResourceLimitError("operator family exceeds the exact refinement storage limits")
    transitions = {name: _square(value, f"operator {name!r}") for name, value in operators.items()}
    if any(matrix.shape != (initial.cols, initial.cols) for matrix in transitions.values()):
        raise DimensionError("every operator must act on the summary's ambient space")
    basis = _row_basis(initial)
    limit = (
        initial.cols - basis.rows + 1 if max_rounds is None else _integer(max_rounds, "max_rounds", minimum=1)
    )
    chain = [basis.rows]
    complete = False
    rounds = 0
    for rounds in range(1, limit + 1):  # noqa: B007 - rounds is retained in the returned record.
        candidate = basis
        for matrix in transitions.values():
            candidate = candidate.col_join(basis * matrix)
        enlarged = _row_basis(candidate)
        if enlarged.rows == basis.rows:
            complete = True
            break
        basis = enlarged
        chain.append(basis.rows)
        if basis.rows == initial.cols:
            complete = True
            break
    decoder = _coordinates(initial, basis)
    reduced = {name: _coordinates(basis * matrix, basis) for name, matrix in transitions.items()}
    residuals = {"original_summary": initial - decoder * basis}
    residuals.update(
        {f"invariance:{name}": basis * transitions[name] - reduced[name] * basis for name in transitions}
    )
    valid = complete and all(_zero(value) for value in residuals.values())
    cert = _certificate(
        "KLSM-0.8-P08",
        residuals,
        valid=valid,
        checked=(
            "row-space refinement from the initial summary",
            "exact retained-summary factorization",
            "every declared operator intertwines",
        ),
        unchecked=(
            "supplied operators describe the target process",
            "added observables can be measured or initialized",
        ),
        diagnostics={"dimension_chain": chain, "rounds": rounds, "round_limit": limit, "complete": complete},
        message="minimal closed row space constructed exactly"
        if valid
        else "round cap reached before exact closure",
    )
    return PredictiveCompletion(
        basis, decoder, MappingProxyType(reduced), tuple(chain), rounds, limit, complete, cert
    )


@dataclass(frozen=True, slots=True)
class ProjectedMemory:
    """Exact full-ambient memory blocks through one declared finite horizon."""

    retained_projector: Any
    unresolved_projector: Any
    retained_operator: Any
    unresolved_to_retained: Any
    retained_to_unresolved: Any
    unresolved_operator: Any
    kernels: tuple[Any, ...]
    forcing_operators: tuple[Any, ...]
    horizon: int
    certificate: Certificate

    def next_state(self, retained_history: Sequence[Any], initial_unresolved: Any) -> Any:
        """Evaluate the exact memory formula at t=len(history)-1.

        Histories are oldest first and embedded in the ambient space. This
        evaluates on the supplied history; it does not certify a rollout from
        approximate histories or a bound for an omitted infinite tail.
        """
        self.certificate.require()
        if len(retained_history) == 0:
            raise ValidationError("retained_history must contain at least z_0")
        time = len(retained_history) - 1
        if time > self.horizon:
            raise ResourceLimitError("history exceeds the declared memory horizon")
        dimension = self.retained_projector.rows
        history = tuple(_matrix(value, "retained history vector") for value in retained_history)
        unresolved = _matrix(initial_unresolved, "initial_unresolved")
        if any(value.shape != (dimension, 1) for value in (*history, unresolved)):
            raise DimensionError("memory states must be ambient column vectors")
        if any(self.retained_projector * value != value for value in history):
            raise ValidationError("a retained history vector lies outside the retained subspace")
        if self.unresolved_projector * unresolved != unresolved:
            raise ValidationError("initial_unresolved lies outside the unresolved subspace")
        value = self.retained_operator * history[-1] + self.forcing_operators[time] * unresolved
        for index in range(time):
            value += self.kernels[index] * history[time - 1 - index]
        return _sympy().ImmutableMatrix(value)


@documented(
    ProvenanceRecord(
        identifier="KLSM-0.8-P10",
        title="Exact projected memory",
        status=ClaimStatus.FINITE_CERTIFICATE,
        source="Large Signal Model Foundations 0.8.0, Section 8, Proposition 10",
        assumptions=("known rational finite linear dynamics", "orthogonal retained projection"),
        limitations=("finite evaluated horizon; no geometric-tail or rollout-stability claim",),
        verification_suite="signals.foundations",
    )
)
def projected_memory(operator: Any, retained_projector: Any, *, horizon: int) -> ProjectedMemory:
    """Return kernels ``BD^kC`` and initial forcing operators ``BD^t``."""
    sp = _sympy()
    count = _integer(horizon, "horizon")
    if count > 10_000:
        raise ResourceLimitError("horizon exceeds the 10000-step exact memory limit")
    matrix = _square(operator, "operator")
    if (2 * count + 1) * matrix.rows * matrix.cols > 1_000_000:
        raise ResourceLimitError("stored memory blocks exceed the 1000000-entry exact memory limit")
    projection = _projector(retained_projector, matrix.rows)
    unresolved = sp.ImmutableMatrix(sp.eye(matrix.rows) - projection)
    a = sp.ImmutableMatrix(projection * matrix * projection)
    b = sp.ImmutableMatrix(projection * matrix * unresolved)
    c = sp.ImmutableMatrix(unresolved * matrix * projection)
    d = sp.ImmutableMatrix(unresolved * matrix * unresolved)
    power = sp.eye(matrix.rows)
    kernels, forcing = [], []
    for time in range(count + 1):
        forcing.append(sp.ImmutableMatrix(b * power))
        if time < count:
            kernels.append(sp.ImmutableMatrix(b * power * c))
        power = d * power
    cert = _certificate(
        "KLSM-0.8-P10",
        {"block_reconstruction": matrix - (a + b + c + d)},
        checked=("orthogonal split", "full block reconstruction", "exact finite matrix powers"),
        diagnostics={"horizon": count, "kernel_count": len(kernels)},
        message="finite projection-memory kernels constructed exactly",
    )
    return ProjectedMemory(projection, unresolved, a, b, c, d, tuple(kernels), tuple(forcing), count, cert)


@dataclass(frozen=True, slots=True)
class DelayClosure:
    """Known-operator delay criterion; not an empirical rank-plateau claim."""

    observation_map: Any
    successor_observation_map: Any
    rank: int
    extended_rank: int
    delay: int
    closed: bool
    transition: Any | None
    certificate: Certificate

    def require_transition(self) -> Any:
        self.certificate.require()
        if not self.closed or self.transition is None:
            raise ValidationError("the declared delay is not predictively closed")
        return self.transition


@documented(
    ProvenanceRecord(
        identifier="KLSM-0.8-P14",
        title="Exact causal delay closure",
        status=ClaimStatus.FINITE_CERTIFICATE,
        source="Large Signal Model Foundations 0.8.0, Section 15, Proposition 14",
        assumptions=("known autonomous rational finite operator and output map",),
        limitations=("uses true supplied observation maps, not a sampled data rank plateau",),
        verification_suite="signals.foundations",
    )
)
def delay_closure(operator: Any, observation: Any, *, delay: int) -> DelayClosure:
    """Check rank O_h=rank O_(h+1) and construct the closed delay transition."""
    sp = _sympy()
    matrix = _square(operator, "operator")
    output = _matrix(observation, "observation")
    length = _integer(delay, "delay", minimum=1)
    if output.cols != matrix.cols:
        raise DimensionError("observation must have the operator's ambient input dimension")
    if length * output.rows * output.cols > 65_536:
        raise ResourceLimitError("delay observation map exceeds the 65536-entry exact matrix limit")
    if (length * output.rows) ** 2 > 65_536:
        raise ResourceLimitError("ambient delay transition exceeds the 65536-entry exact matrix limit")
    rows, current = [], output
    for _ in range(length):
        rows.append(current)
        current = current * matrix
    observation_map = sp.ImmutableMatrix.vstack(*rows)
    extended_rank = observation_map.col_join(current).rank()
    rank = observation_map.rank()
    successor = sp.ImmutableMatrix(observation_map * matrix)
    closed = rank == extended_rank
    transition = sp.ImmutableMatrix(successor * observation_map.pinv()) if closed else None
    residuals = (
        {"delay_intertwining": successor - transition * observation_map}
        if closed
        else {"kernel_obstruction": successor * sp.Matrix.hstack(*observation_map.nullspace())}
    )
    cert = _certificate(
        "KLSM-0.8-P14",
        residuals,
        valid=closed,
        checked=("known observation-map ranks", "delay factorization"),
        unchecked=("supplied operator and output map describe the target process",),
        diagnostics={"delay": length, "rank": rank, "extended_rank": extended_rank},
        message="known-operator delay closes exactly" if closed else "known-operator delay does not close",
    )
    return DelayClosure(observation_map, successor, rank, extended_rank, length, closed, transition, cert)


@dataclass(frozen=True, slots=True)
class DataSpanIdentification:
    """Finite sample fit with span closure kept separate from global coverage."""

    input_states: Any
    successor_states: Any
    data_rank: int
    fit_consistent: bool
    span_closed: bool
    ambient_span_complete: bool
    fitted_operator: Any | None
    dependency_obstruction: Any
    certificate: Certificate

    @property
    def global_coverage(self) -> str:
        return "not_established"

    def require_fit(self) -> Any:
        self.certificate.require()
        if not self.fit_consistent or self.fitted_operator is None:
            raise ValidationError("sample dependencies do not admit a linear fit")
        return self.fitted_operator

    def require_closed_span(self) -> Any:
        fitted = self.require_fit()
        if not self.span_closed:
            raise ValidationError("successor states leave the sampled input span")
        return fitted


@documented(
    ProvenanceRecord(
        identifier="KLSM-0.8-P15",
        title="Data-span identification and its boundary",
        status=ClaimStatus.FINITE_CERTIFICATE,
        source="Large Signal Model Foundations 0.8.0, Section 16, Proposition 15",
        assumptions=("supplied finite rational paired states",),
        limitations=("fit and span closure do not establish true-process coverage",),
        verification_suite="signals.foundations",
    )
)
def identify_data_span(input_states: Any, successor_states: Any) -> DataSpanIdentification:
    """Check exact linear fit consistency and observed-span closure separately.

    A returned fit agrees on im(Z_-). No assertion about an unknown larger
    realizable state space follows, even when the supplied ambient span is full.
    """
    sp = _sympy()
    inputs = _matrix(input_states, "input_states")
    successors = _matrix(successor_states, "successor_states")
    if inputs.shape != successors.shape:
        raise DimensionError("paired input and successor matrices must have equal shapes")
    if inputs.rows**2 > 65_536:
        raise ResourceLimitError("ambient fitted operator exceeds the 65536-entry exact matrix limit")
    nullspace = inputs.nullspace()
    kernel = sp.Matrix.hstack(*nullspace) if nullspace else sp.zeros(inputs.cols, 0)
    obstruction = sp.ImmutableMatrix(successors * kernel)
    consistent = _zero(obstruction)
    rank = inputs.rank()
    span_closed = inputs.row_join(successors).rank() == rank
    fitted = sp.ImmutableMatrix(successors * inputs.pinv()) if consistent else None
    residuals = {"dependency_obstruction": obstruction}
    if fitted is not None:
        residuals["sample_fit"] = fitted * inputs - successors
    cert = _certificate(
        "KLSM-0.8-P15",
        residuals,
        valid=consistent,
        checked=("all exact input-column dependencies", "finite sample factorization"),
        unchecked=(
            "sampled states span the true realizable state space",
            "supplied successors describe the target process",
        ),
        diagnostics={
            "data_rank": rank,
            "span_closed": span_closed,
            "ambient_span_complete": rank == inputs.rows,
            "global_coverage": "not_established",
        },
        message="exact operator identified on the sampled span"
        if consistent
        else "sample dependencies contradict a linear fit",
    )
    return DataSpanIdentification(
        inputs, successors, rank, consistent, span_closed, rank == inputs.rows, fitted, obstruction, cert
    )


__all__ = [
    "CompressionClosure",
    "DataSpanIdentification",
    "DelayClosure",
    "PredictiveCompletion",
    "ProjectedMemory",
    "compression_closure",
    "delay_closure",
    "identify_data_span",
    "minimal_predictive_completion",
    "projected_memory",
]
