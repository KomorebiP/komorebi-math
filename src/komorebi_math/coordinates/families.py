"""Named and custom recurrence coordinate families."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from ..exceptions import DimensionError, ValidationError
from .basis import FiniteBasis
from .exact import Scalar, Vector, as_vector
from .recurrence import (
    RecurrenceCoordinateRecord,
    analyze_recurrence_coordinates,
    companion_matrix,
    coordinate_change_between_recurrences,
    synthesize_recurrence_coordinates,
)
from .transform import CoordinateTransform


@dataclass(frozen=True, slots=True)
class RecurrenceModel:
    r"""A named constant-coefficient recurrence and optional initial state.

    The convention is
    ``a_(n+d)=sum_(j=0)^(d-1) coefficients[j] a_(n+j)``.
    """

    name: str
    coefficients: Vector
    initial_values: Vector = ()
    description: str = ""

    def __post_init__(self) -> None:
        coefficients = as_vector(self.coefficients)
        if not coefficients:
            raise DimensionError("a recurrence model needs at least one coefficient")
        initial = as_vector(self.initial_values)
        if initial and len(initial) != len(coefficients):
            raise DimensionError("initial values must match the recurrence order")
        if not str(self.name).strip():
            raise ValidationError("recurrence-model names must be nonempty")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "initial_values", initial)
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "description", str(self.description))

    @property
    def order(self) -> int:
        """Return the recurrence order."""

        return len(self.coefficients)

    @property
    def companion(self):
        """Return the exact companion matrix."""

        return companion_matrix(self.coefficients)

    def sequence(self, count: int, *, initial_values: Sequence[Scalar] | None = None) -> Vector:
        """Generate a homogeneous recurrence prefix."""

        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValidationError("sequence count must be a nonnegative integer")
        initial = self.initial_values if initial_values is None else as_vector(initial_values)
        if len(initial) != self.order:
            raise DimensionError("a complete initial state is required")
        if count <= self.order:
            return initial[:count]
        values = list(initial)
        while len(values) < count:
            offset = len(values) - self.order
            values.append(
                sum(
                    (
                        self.coefficients[index] * values[offset + index]
                        for index in range(self.order)
                    ),
                    Fraction(0),
                )
            )
        return tuple(values)

    def analyze(self, values: Sequence[Scalar]) -> RecurrenceCoordinateRecord:
        """Analyze a finite sequence into initial-state and defect coordinates."""

        return analyze_recurrence_coordinates(values, self.coefficients, name=self.name)

    def synthesize(self, coordinates: Sequence[Scalar]) -> Vector:
        """Synthesize a finite sequence from recurrence coordinates."""

        values = as_vector(coordinates)
        if len(values) < self.order:
            raise DimensionError("coordinate vector is shorter than the recurrence order")
        record = RecurrenceCoordinateRecord(
            self.name,
            self.coefficients,
            values[: self.order],
            values[self.order :],
        )
        return synthesize_recurrence_coordinates(record)

    def coordinate_transform_to(self, target: "RecurrenceModel", length: int) -> CoordinateTransform:
        """Return the finite coordinate map to another recurrence model."""

        if not isinstance(target, RecurrenceModel):
            raise ValidationError("target must be a RecurrenceModel")
        transform = coordinate_change_between_recurrences(
            self.coefficients,
            target.coefficients,
            length,
        )
        return CoordinateTransform(
            f"{self.name} recurrence coordinates of length {length}",
            f"{target.name} recurrence coordinates of length {length}",
            transform.matrix,
            "CC.Q1",
            "exact finite change between recurrence models",
        )

    def homogeneous_basis(self, length: int) -> FiniteBasis:
        """Return a basis for homogeneous prefixes of the declared length."""

        if isinstance(length, bool) or not isinstance(length, int) or length < self.order:
            raise DimensionError("homogeneous basis length must be at least the recurrence order")
        vectors = []
        for coordinate in range(self.order):
            initial = tuple(Fraction(int(index == coordinate)) for index in range(self.order))
            vectors.append(self.sequence(length, initial_values=initial))
        return FiniteBasis(
            f"homogeneous {self.name} prefixes of length {length}",
            tuple(vectors),
            tuple(f"initial state {index}" for index in range(self.order)),
            "fundamental solution basis of a constant-coefficient recurrence",
        )


def custom_recurrence_model(
    coefficients: Sequence[Scalar],
    *,
    initial_values: Sequence[Scalar] = (),
    name: str = "custom recurrence",
    description: str = "",
) -> RecurrenceModel:
    """Construct an arbitrary exact constant-coefficient recurrence model."""

    return RecurrenceModel(name, as_vector(coefficients), as_vector(initial_values), description)


def fibonacci_recurrence_model() -> RecurrenceModel:
    """Return the Fibonacci recurrence model."""

    return RecurrenceModel("Fibonacci", (1, 1), (0, 1), "a_(n+2)=a_n+a_(n+1)")


def lucas_recurrence_model() -> RecurrenceModel:
    """Return the Lucas recurrence model."""

    return RecurrenceModel("Lucas", (1, 1), (2, 1), "same companion dynamics as Fibonacci with a different state")


def pell_recurrence_model() -> RecurrenceModel:
    """Return the Pell recurrence model."""

    return RecurrenceModel("Pell", (1, 2), (0, 1), "a_(n+2)=a_n+2a_(n+1)")


def jacobsthal_recurrence_model() -> RecurrenceModel:
    """Return the Jacobsthal recurrence model."""

    return RecurrenceModel("Jacobsthal", (2, 1), (0, 1), "a_(n+2)=2a_n+a_(n+1)")


def tribonacci_recurrence_model() -> RecurrenceModel:
    """Return the Tribonacci recurrence model."""

    return RecurrenceModel("Tribonacci", (1, 1, 1), (0, 0, 1), "a_(n+3)=a_n+a_(n+1)+a_(n+2)")


def padovan_recurrence_model() -> RecurrenceModel:
    """Return a standard Padovan recurrence model."""

    return RecurrenceModel("Padovan", (1, 1, 0), (1, 1, 1), "a_(n+3)=a_n+a_(n+1)")


def recurrence_model_catalog() -> tuple[RecurrenceModel, ...]:
    """Return the built-in named recurrence models."""

    return (
        fibonacci_recurrence_model(),
        lucas_recurrence_model(),
        pell_recurrence_model(),
        jacobsthal_recurrence_model(),
        tribonacci_recurrence_model(),
        padovan_recurrence_model(),
    )



def get_recurrence_model(name: str) -> RecurrenceModel:
    """Return one built-in recurrence model by normalized name."""

    query = "".join(character for character in str(name).lower() if character.isalnum())
    for model in recurrence_model_catalog():
        key = "".join(character for character in model.name.lower() if character.isalnum())
        if key == query:
            return model
    raise ValidationError(f"unknown recurrence model {name!r}")

def sequence_coordinates_between_models(
    values: Sequence[Scalar],
    source: RecurrenceModel,
    target: RecurrenceModel,
) -> tuple[RecurrenceCoordinateRecord, RecurrenceCoordinateRecord, CoordinateTransform]:
    """Analyze one sequence in two recurrence systems and return the exact bridge."""

    source_record = source.analyze(values)
    target_record = target.analyze(values)
    transform = source.coordinate_transform_to(target, len(values))
    if transform.apply(source_record.coordinates()) != target_record.coordinates():
        raise ValidationError("recurrence-model coordinate bridge failed exact verification")
    return source_record, target_record, transform


__all__ = [
    "RecurrenceModel",
    "custom_recurrence_model",
    "fibonacci_recurrence_model",
    "get_recurrence_model",
    "jacobsthal_recurrence_model",
    "lucas_recurrence_model",
    "padovan_recurrence_model",
    "pell_recurrence_model",
    "recurrence_model_catalog",
    "sequence_coordinates_between_models",
    "tribonacci_recurrence_model",
]
