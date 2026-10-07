"""Exact state-space coordinate changes and canonical finite realizations."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from .basis import FiniteBasis
from .exact import (
    Matrix,
    Scalar,
    Vector,
    as_matrix,
    as_vector,
    identity_matrix,
    inverse_matrix,
    matrix_from_columns,
    matrix_multiply,
    matrix_power,
    matrix_rank,
    matrix_shape,
    matrix_vector_multiply,
)
from .tensor import transport_operator
from .transform import CoordinateTransform


@dataclass(frozen=True, slots=True)
class StateSpaceModel:
    r"""A finite exact state-space model ``x'=Ax+Bu``, ``y=Cx+Du``."""

    name: str
    transition: Matrix
    input_matrix: Matrix
    output_matrix: Matrix
    feedthrough: Matrix

    def __post_init__(self) -> None:
        transition = as_matrix(self.transition)
        inputs = as_matrix(self.input_matrix)
        outputs = as_matrix(self.output_matrix)
        feedthrough = as_matrix(self.feedthrough)
        states, transition_columns = matrix_shape(transition)
        if states == 0 or states != transition_columns:
            raise DimensionError("state transition matrix must be nonempty and square")
        input_rows, input_count = matrix_shape(inputs)
        output_count, output_columns = matrix_shape(outputs)
        feed_rows, feed_columns = matrix_shape(feedthrough)
        if input_rows != states:
            raise DimensionError("input matrix row count must match the state dimension")
        if output_columns != states:
            raise DimensionError("output matrix column count must match the state dimension")
        if feed_rows != output_count or feed_columns != input_count:
            raise DimensionError("feedthrough shape must match output and input counts")
        object.__setattr__(self, "transition", transition)
        object.__setattr__(self, "input_matrix", inputs)
        object.__setattr__(self, "output_matrix", outputs)
        object.__setattr__(self, "feedthrough", feedthrough)
        object.__setattr__(self, "name", str(self.name).strip() or "state-space model")

    @property
    def state_dimension(self) -> int:
        """Return the state dimension."""

        return matrix_shape(self.transition)[0]

    @property
    def input_dimension(self) -> int:
        """Return the input dimension."""

        return matrix_shape(self.input_matrix)[1]

    @property
    def output_dimension(self) -> int:
        """Return the output dimension."""

        return matrix_shape(self.output_matrix)[0]

    def change_coordinates(
        self,
        transform: CoordinateTransform,
        *,
        name: str | None = None,
    ) -> "StateSpaceModel":
        r"""Return the model in target state coordinates ``z=Cx``."""

        if not transform.square or transform.source_dimension != self.state_dimension:
            raise DimensionError("state coordinate transform has the wrong dimension")
        inverse = inverse_matrix(transform.matrix)
        transition = transport_operator(self.transition, transform)
        input_matrix = matrix_multiply(transform.matrix, self.input_matrix)
        output_matrix = matrix_multiply(self.output_matrix, inverse)
        return StateSpaceModel(
            name or f"{self.name} in {transform.target}",
            transition,
            input_matrix,
            output_matrix,
            self.feedthrough,
        )

    def markov_parameter(self, index: int) -> Matrix:
        """Return ``D`` for index zero and ``C A^(index-1) B`` afterward."""

        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValidationError("Markov-parameter index must be nonnegative")
        if index == 0:
            return self.feedthrough
        return matrix_multiply(
            matrix_multiply(self.output_matrix, matrix_power(self.transition, index - 1)),
            self.input_matrix,
        )

    def transfer_equivalent(
        self,
        other: "StateSpaceModel",
        *,
        count: int | None = None,
    ) -> bool:
        """Compare a finite prefix of exact Markov parameters."""

        if not isinstance(other, StateSpaceModel):
            return False
        if self.input_dimension != other.input_dimension or self.output_dimension != other.output_dimension:
            return False
        bound = max(self.state_dimension, other.state_dimension) * 2 + 1 if count is None else count
        if isinstance(bound, bool) or not isinstance(bound, int) or bound < 1:
            raise ValidationError("comparison count must be positive")
        return all(self.markov_parameter(index) == other.markov_parameter(index) for index in range(bound))


def controllability_matrix(
    transition: Iterable[Iterable[Scalar]],
    input_matrix: Iterable[Iterable[Scalar]],
    *,
    steps: int | None = None,
) -> Matrix:
    """Return the finite controllability block matrix ``[B,AB,...]``."""

    a = as_matrix(transition)
    b = as_matrix(input_matrix)
    states, columns = matrix_shape(a)
    input_rows, input_count = matrix_shape(b)
    if states == 0 or states != columns:
        raise DimensionError("transition matrix must be nonempty and square")
    if input_rows != states:
        raise DimensionError("input matrix has the wrong state dimension")
    count = states if steps is None else steps
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValidationError("controllability steps must be positive")
    block_columns: list[Vector] = []
    for power in range(count):
        block = matrix_multiply(matrix_power(a, power), b)
        for column in range(input_count):
            block_columns.append(tuple(block[row][column] for row in range(states)))
    return matrix_from_columns(block_columns)


def observability_block_matrix(
    transition: Iterable[Iterable[Scalar]],
    output_matrix: Iterable[Iterable[Scalar]],
    *,
    steps: int | None = None,
) -> Matrix:
    """Return the finite observability block matrix ``[C;CA;...]``."""

    a = as_matrix(transition)
    c = as_matrix(output_matrix)
    states, columns = matrix_shape(a)
    output_count, output_columns = matrix_shape(c)
    if states == 0 or states != columns:
        raise DimensionError("transition matrix must be nonempty and square")
    if output_columns != states:
        raise DimensionError("output matrix has the wrong state dimension")
    count = states if steps is None else steps
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValidationError("observability steps must be positive")
    rows: list[tuple[Fraction, ...]] = []
    for power in range(count):
        block = matrix_multiply(c, matrix_power(a, power))
        rows.extend(block[row] for row in range(output_count))
    return tuple(rows)


def controllability_basis(model: StateSpaceModel) -> FiniteBasis:
    """Return a maximal independent basis extracted from controllability columns."""

    matrix = controllability_matrix(model.transition, model.input_matrix)
    _, columns = matrix_shape(matrix)
    selected: list[Vector] = []
    current_rank = 0
    for column in range(columns):
        candidate = tuple(matrix[row][column] for row in range(model.state_dimension))
        trial = matrix_from_columns(tuple(selected) + (candidate,))
        rank = matrix_rank(trial)
        if rank > current_rank:
            selected.append(candidate)
            current_rank = rank
        if current_rank == model.state_dimension:
            break
    return FiniteBasis(
        f"reachable basis of {model.name}",
        tuple(selected),
        tuple(f"r{index}" for index in range(len(selected))),
        "independent columns of the controllability matrix",
    )


def controllability_coordinate_transform(model: StateSpaceModel) -> CoordinateTransform:
    """Map original state coordinates to a full controllability basis."""

    basis = controllability_basis(model)
    if not basis.square:
        raise ValidationError("model is not controllable")
    synthesis = CoordinateTransform(
        f"controllability coordinates of {model.name}",
        f"state coordinates of {model.name}",
        basis.matrix,
        "CC.R1",
        "state synthesis from controllability coordinates",
    )
    return synthesis.inverse(theorem="CC.R1")


def observability_coordinate_transform(model: StateSpaceModel) -> CoordinateTransform:
    """Map state coordinates to a full output-jet coordinate system."""

    matrix = observability_block_matrix(model.transition, model.output_matrix)
    rows, columns = matrix_shape(matrix)
    if rows != columns:
        if rows < columns:
            raise ValidationError("available output jet has too few coordinates")
        independent_rows: list[tuple[Fraction, ...]] = []
        current_rank = 0
        for row in matrix:
            trial = tuple(independent_rows) + (row,)
            rank = matrix_rank(trial)
            if rank > current_rank:
                independent_rows.append(row)
                current_rank = rank
            if current_rank == columns:
                break
        matrix = tuple(independent_rows)
        rows, columns = matrix_shape(matrix)
    if rows != columns or matrix_rank(matrix) != columns:
        raise ValidationError("model is not observable")
    return CoordinateTransform(
        f"state coordinates of {model.name}",
        f"observable output-jet coordinates of {model.name}",
        matrix,
        "CC.R2",
        "state analysis by an exact observability matrix",
    )


@dataclass(frozen=True, slots=True)
class StateSpaceCoordinateCertificate:
    """Exact invariance record for one state coordinate change."""

    source: StateSpaceModel
    target: StateSpaceModel
    transform: CoordinateTransform
    markov_count: int
    valid: bool

    def certificate(self) -> Certificate:
        """Return a machine-readable exact transfer-invariance certificate."""

        return Certificate(
            valid=self.valid,
            exactness=Exactness.EXACT,
            assumptions_checked=("invertible state transform", "exact Markov parameters"),
            diagnostics={
                "source": self.source.name,
                "target": self.target.name,
                "markov_count": self.markov_count,
            },
            theorem_id="CC.R3",
            message="state coordinate change preserves transfer behavior"
            if self.valid
            else "state coordinate transfer check failed",
        )


def certify_state_coordinate_change(
    model: StateSpaceModel,
    transform: CoordinateTransform,
    *,
    markov_count: int | None = None,
) -> StateSpaceCoordinateCertificate:
    """Certify transfer invariance under an exact state coordinate change."""

    target = model.change_coordinates(transform)
    count = 2 * model.state_dimension + 1 if markov_count is None else markov_count
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValidationError("Markov comparison count must be positive")
    valid = model.transfer_equivalent(target, count=count)
    return StateSpaceCoordinateCertificate(model, target, transform, count, valid)


def siso_state_space(
    transition: Iterable[Iterable[Scalar]],
    input_vector: Sequence[Scalar],
    output_covector: Sequence[Scalar],
    *,
    feedthrough: Scalar = 0,
    name: str = "SISO state-space model",
) -> StateSpaceModel:
    """Construct a single-input single-output exact state-space model."""

    a = as_matrix(transition)
    states, columns = matrix_shape(a)
    if states == 0 or states != columns:
        raise DimensionError("transition matrix must be nonempty and square")
    b = as_vector(input_vector)
    c = as_vector(output_covector)
    if len(b) != states or len(c) != states:
        raise DimensionError("SISO input and output vectors must match the state dimension")
    return StateSpaceModel(
        name,
        a,
        tuple((value,) for value in b),
        (c,),
        ((Fraction(feedthrough),),),
    )


__all__ = [
    "StateSpaceCoordinateCertificate",
    "StateSpaceModel",
    "certify_state_coordinate_change",
    "controllability_basis",
    "controllability_coordinate_transform",
    "controllability_matrix",
    "observability_block_matrix",
    "observability_coordinate_transform",
    "siso_state_space",
]
