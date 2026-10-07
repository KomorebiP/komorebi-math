"""Trusted finite snapshots and portable codecs for generating-function objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Mapping

from ..exceptions import ValidationError
from ..export.persistence import register_portable_codec, to_portable_data
from .basis import (
    BasisDescriptor,
    WeightedMonomialBasis,
    basis_from_descriptor,
    basis_from_values,
)
from .budget import GeneratingBudget
from .domains import domain_from_descriptor
from .finite_series import FiniteNormalizedSeries
from .cartier import CartierAutomaton, CartierLinearRepresentation
from .functions import CustomFunctionSpec
from .masks import (
    CarryLayerConvolution,
    FiniteMaskCarryPlan,
    KummerLayerDistribution,
    compile_finite_mask_carry_plan,
    kummer_layer_distribution,
)
from .series import FormalSeries


@dataclass(frozen=True, slots=True)
class FrozenGeneratingFunction:
    """Provider-independent finite snapshot of a custom generating function."""

    name: str
    basis_name: str
    normalization: tuple[Fraction, ...]
    coefficients: tuple[Fraction, ...]
    source_kind: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        basis_name = str(self.basis_name).strip()
        source_kind = str(self.source_kind).strip()
        normalization = tuple(Fraction(value) for value in self.normalization)
        coefficients = tuple(Fraction(value) for value in self.coefficients)
        if not name or not basis_name or not source_kind:
            raise ValidationError("frozen function names and source kind must be nonempty")
        if not coefficients:
            raise ValidationError("frozen function must contain at least one coefficient")
        if len(normalization) != len(coefficients) or normalization[0] != 1:
            raise ValidationError(
                "frozen function normalization must start with one and match its coefficients"
            )
        if any(value == 0 for value in normalization):
            raise ValidationError("frozen function normalization cannot contain zero")
        try:
            to_portable_data(dict(self.metadata))
        except ValidationError as exc:
            raise ValidationError("frozen function metadata is not portable") from exc
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "basis_name", basis_name)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def degree(self) -> int:
        """Return the largest captured degree."""

        return len(self.coefficients) - 1

    def series(self, degree: int | None = None) -> FormalSeries:
        """Reconstruct a finite exact series without the original callback."""

        selected = self.degree if degree is None else degree
        if isinstance(selected, bool) or not isinstance(selected, int) or selected < 0:
            raise ValidationError("degree must be a nonnegative integer")
        if selected > self.degree:
            raise ValidationError(
                f"frozen function is captured only through degree {self.degree}"
            )
        basis = basis_from_values(self.basis_name, self.normalization[: selected + 1])
        return FormalSeries(basis, self.coefficients[: selected + 1])


def freeze_custom_function(
    function: CustomFunctionSpec,
    degree: int,
    *,
    include_metadata: bool = True,
) -> FrozenGeneratingFunction:
    """Capture one custom function into a finite provider-independent record."""

    if not isinstance(function, CustomFunctionSpec):
        raise ValidationError("function must be a CustomFunctionSpec")
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
        raise ValidationError("degree must be a nonnegative integer")
    series = function.series(degree)
    normalization = tuple(function.basis.gamma(index) for index in range(degree + 1))
    metadata: Mapping[str, Any] = dict(function.metadata) if include_metadata else {}
    return FrozenGeneratingFunction(
        function.name,
        function.basis.name,
        normalization,
        series.coefficients,
        function.source_kind,
        metadata,
    )


def _basis_snapshot_descriptor(
    basis: WeightedMonomialBasis,
    degree: int,
) -> BasisDescriptor:
    descriptor = basis.descriptor
    if descriptor is not None:
        return descriptor
    normalization = tuple(basis.gamma(index) for index in range(degree + 1))
    return BasisDescriptor.create(
        basis.name,
        "table",
        {"values": normalization},
    )


def _decode_basis_descriptor(data: Mapping[str, Any]) -> BasisDescriptor:
    if set(data) != {"name", "kind", "parameters"}:
        raise ValidationError("basis descriptor portable data does not match its schema")
    parameters = data["parameters"]
    if not isinstance(parameters, Mapping):
        raise ValidationError("basis descriptor parameters must be a mapping")
    return BasisDescriptor.create(str(data["name"]), str(data["kind"]), parameters)


def _decode_weighted_basis(data: Mapping[str, Any]) -> WeightedMonomialBasis:
    if set(data) != {"descriptor"} or not isinstance(data["descriptor"], BasisDescriptor):
        raise ValidationError("weighted-basis portable data does not match its schema")
    return basis_from_descriptor(data["descriptor"])


def _decode_budget(data: Mapping[str, Any]) -> GeneratingBudget:
    required = {
        "maximum_degree",
        "maximum_coefficient_bits",
        "maximum_matrix_dimension",
        "maximum_reachable_states",
        "maximum_local_combinations",
        "maximum_modulus",
        "maximum_serialized_bytes",
        "maximum_recursion_depth",
    }
    if set(data) != required:
        raise ValidationError("generating budget portable data does not match its schema")
    return GeneratingBudget(**{key: int(data[key]) for key in required})


def _decode_formal_series(data: Mapping[str, Any]) -> FormalSeries:
    required = {"basis", "coefficients"}
    if set(data) != required or not isinstance(data["basis"], BasisDescriptor):
        raise ValidationError("formal-series portable data does not match its schema")
    coefficients = tuple(Fraction(value) for value in data["coefficients"])
    if not coefficients:
        raise ValidationError("formal-series portable data cannot be empty")
    basis = basis_from_descriptor(data["basis"])
    return FormalSeries(basis, coefficients)


def _decode_finite_normalized_series(
    data: Mapping[str, Any],
) -> FiniteNormalizedSeries:
    required = {"basis", "domain", "coefficients", "budget"}
    if set(data) != required:
        raise ValidationError("finite normalized series data does not match its schema")
    if not isinstance(data["basis"], BasisDescriptor):
        raise ValidationError("finite normalized series basis descriptor is invalid")
    if not isinstance(data["domain"], Mapping):
        raise ValidationError("finite normalized series domain descriptor is invalid")
    if not isinstance(data["budget"], GeneratingBudget):
        raise ValidationError("finite normalized series budget is invalid")
    return FiniteNormalizedSeries(
        basis_from_descriptor(data["basis"]),
        domain_from_descriptor(dict(data["domain"])),
        tuple(data["coefficients"]),
        budget=data["budget"],
    )

def _decode_frozen_function(data: Mapping[str, Any]) -> FrozenGeneratingFunction:
    required = {
        "name",
        "basis_name",
        "normalization",
        "coefficients",
        "source_kind",
        "metadata",
    }
    if set(data) != required:
        raise ValidationError("frozen-function portable data does not match its schema")
    if not isinstance(data["metadata"], Mapping):
        raise ValidationError("frozen-function metadata must be a mapping")
    return FrozenGeneratingFunction(
        str(data["name"]),
        str(data["basis_name"]),
        tuple(data["normalization"]),
        tuple(data["coefficients"]),
        str(data["source_kind"]),
        dict(data["metadata"]),
    )


def _decode_cartier_automaton(data: Mapping[str, Any]) -> CartierAutomaton:
    required = {
        "prime",
        "numerator",
        "denominator",
        "degree_bound",
        "transitions",
        "outputs",
        "state_numerators",
        "state_members",
        "start_state",
    }
    if set(data) != required:
        raise ValidationError("Cartier automaton portable data does not match its schema")
    automaton = CartierAutomaton(
        prime=int(data["prime"]),
        numerator=tuple(int(value) for value in data["numerator"]),
        denominator=tuple(int(value) for value in data["denominator"]),
        degree_bound=int(data["degree_bound"]),
        transitions=tuple(tuple(int(value) for value in row) for row in data["transitions"]),
        outputs=tuple(int(value) for value in data["outputs"]),
        state_numerators=tuple(
            tuple(int(value) for value in row) for row in data["state_numerators"]
        ),
        state_members=tuple(
            tuple(int(value) for value in row) for row in data["state_members"]
        ),
        start_state=int(data["start_state"]),
    )
    if not automaton.transitions or not 0 <= automaton.start_state < automaton.number_states:
        raise ValidationError("Cartier automaton start state is invalid")
    if len(automaton.outputs) != automaton.number_states:
        raise ValidationError("Cartier automaton outputs do not match its states")
    if any(len(row) != automaton.prime for row in automaton.transitions):
        raise ValidationError("Cartier automaton transition rows have invalid length")
    if any(
        target < 0 or target >= automaton.number_states
        for row in automaton.transitions
        for target in row
    ):
        raise ValidationError("Cartier automaton transition target is invalid")
    return automaton



def _decode_cartier_linear_representation(
    data: Mapping[str, Any],
) -> CartierLinearRepresentation:
    required = {
        "prime",
        "numerator",
        "denominator",
        "degree_bound",
        "transition_matrices",
        "start_vector",
        "output_scale",
    }
    if set(data) != required:
        raise ValidationError(
            "Cartier linear-representation portable data does not match its schema"
        )
    representation = CartierLinearRepresentation(
        prime=int(data["prime"]),
        numerator=tuple(int(value) for value in data["numerator"]),
        denominator=tuple(int(value) for value in data["denominator"]),
        degree_bound=int(data["degree_bound"]),
        transition_matrices=tuple(
            tuple(tuple(int(value) for value in row) for row in matrix)
            for matrix in data["transition_matrices"]
        ),
        start_vector=tuple(int(value) for value in data["start_vector"]),
        output_scale=int(data["output_scale"]),
    )
    if len(representation.start_vector) != representation.dimension:
        raise ValidationError("Cartier linear representation has an invalid start vector")
    if len(representation.transition_matrices) != representation.prime:
        raise ValidationError("Cartier linear representation has the wrong digit count")
    for matrix in representation.transition_matrices:
        if len(matrix) != representation.dimension or any(
            len(row) != representation.dimension for row in matrix
        ):
            raise ValidationError("Cartier linear representation matrix has invalid dimensions")
    return representation

def _decode_mask_plan(data: Mapping[str, Any]) -> FiniteMaskCarryPlan:
    if set(data) != {"radix", "mask"}:
        raise ValidationError("finite-mask plan portable data does not match its schema")
    return compile_finite_mask_carry_plan(tuple(data["mask"]), int(data["radix"]))


def _decode_kummer_distribution(data: Mapping[str, Any]) -> KummerLayerDistribution:
    if set(data) != {"index", "prime", "counts"}:
        raise ValidationError("Kummer distribution portable data does not match its schema")
    expected = kummer_layer_distribution(int(data["index"]), int(data["prime"]))
    if expected.counts != tuple(int(value) for value in data["counts"]):
        raise ValidationError("Kummer distribution counts do not match their defining index")
    return expected


def _decode_carry_convolution(data: Mapping[str, Any]) -> CarryLayerConvolution:
    required = {
        "index",
        "prime",
        "exponent",
        "factorial_power",
        "modulus",
        "layer_sums",
    }
    if set(data) != required:
        raise ValidationError("carry-layer convolution portable data does not match its schema")
    result = CarryLayerConvolution(
        int(data["index"]),
        int(data["prime"]),
        int(data["exponent"]),
        int(data["factorial_power"]),
        int(data["modulus"]),
        tuple(int(value) for value in data["layer_sums"]),
    )
    if result.modulus != result.prime**result.exponent:
        raise ValidationError("carry-layer convolution modulus is inconsistent")
    if any(value < 0 or value >= result.modulus for value in result.layer_sums):
        raise ValidationError("carry-layer convolution contains a noncanonical residue")
    return result


def register_generating_codecs() -> None:
    """Register the trusted data-only generating-function codecs."""

    register_portable_codec(
        BasisDescriptor,
        tag="generating.basis_descriptor",
        version=1,
        encoder=lambda value: {
            "name": value.name,
            "kind": value.kind,
            "parameters": value.parameter_map(),
        },
        decoder=_decode_basis_descriptor,
        replace=True,
    )
    register_portable_codec(
        WeightedMonomialBasis,
        tag="generating.weighted_monomial_basis",
        version=1,
        encoder=lambda value: {"descriptor": value.require_descriptor()},
        decoder=_decode_weighted_basis,
        replace=True,
    )
    register_portable_codec(
        GeneratingBudget,
        tag="generating.budget",
        version=1,
        encoder=lambda value: {
            "maximum_degree": value.maximum_degree,
            "maximum_coefficient_bits": value.maximum_coefficient_bits,
            "maximum_matrix_dimension": value.maximum_matrix_dimension,
            "maximum_reachable_states": value.maximum_reachable_states,
            "maximum_local_combinations": value.maximum_local_combinations,
            "maximum_modulus": value.maximum_modulus,
            "maximum_serialized_bytes": value.maximum_serialized_bytes,
            "maximum_recursion_depth": value.maximum_recursion_depth,
        },
        decoder=_decode_budget,
        replace=True,
    )
    register_portable_codec(
        FormalSeries,
        tag="generating.formal_series",
        version=1,
        encoder=lambda value: {
            "basis": _basis_snapshot_descriptor(value.basis, value.degree),
            "coefficients": value.coefficients,
        },
        decoder=_decode_formal_series,
        replace=True,
    )
    register_portable_codec(
        FiniteNormalizedSeries,
        tag="generating.finite_normalized_series",
        version=1,
        encoder=lambda value: {
            "basis": _basis_snapshot_descriptor(value.basis, value.degree),
            "domain": value.domain.descriptor(),
            "coefficients": value.coefficients,
            "budget": value.budget,
        },
        decoder=_decode_finite_normalized_series,
        replace=True,
    )
    register_portable_codec(
        FrozenGeneratingFunction,
        tag="generating.frozen_function",
        version=1,
        encoder=lambda value: {
            "name": value.name,
            "basis_name": value.basis_name,
            "normalization": value.normalization,
            "coefficients": value.coefficients,
            "source_kind": value.source_kind,
            "metadata": dict(value.metadata),
        },
        decoder=_decode_frozen_function,
        replace=True,
    )
    register_portable_codec(
        FiniteMaskCarryPlan,
        tag="generating.finite_mask_carry_plan",
        version=1,
        encoder=lambda value: {"radix": value.radix, "mask": value.mask},
        decoder=_decode_mask_plan,
        replace=True,
    )
    register_portable_codec(
        CartierAutomaton,
        tag="generating.cartier_automaton",
        version=1,
        encoder=lambda value: {
            "prime": value.prime,
            "numerator": value.numerator,
            "denominator": value.denominator,
            "degree_bound": value.degree_bound,
            "transitions": value.transitions,
            "outputs": value.outputs,
            "state_numerators": value.state_numerators,
            "state_members": value.state_members,
            "start_state": value.start_state,
        },
        decoder=_decode_cartier_automaton,
        replace=True,
    )

    register_portable_codec(
        CartierLinearRepresentation,
        tag="generating.cartier_linear_representation",
        version=1,
        encoder=lambda value: {
            "prime": value.prime,
            "numerator": value.numerator,
            "denominator": value.denominator,
            "degree_bound": value.degree_bound,
            "transition_matrices": value.transition_matrices,
            "start_vector": value.start_vector,
            "output_scale": value.output_scale,
        },
        decoder=_decode_cartier_linear_representation,
        replace=True,
    )
    register_portable_codec(
        KummerLayerDistribution,
        tag="generating.kummer_layer_distribution",
        version=1,
        encoder=lambda value: {
            "index": value.index,
            "prime": value.prime,
            "counts": value.counts,
        },
        decoder=_decode_kummer_distribution,
        replace=True,
    )
    register_portable_codec(
        CarryLayerConvolution,
        tag="generating.carry_layer_convolution",
        version=1,
        encoder=lambda value: {
            "index": value.index,
            "prime": value.prime,
            "exponent": value.exponent,
            "factorial_power": value.factorial_power,
            "modulus": value.modulus,
            "layer_sums": value.layer_sums,
        },
        decoder=_decode_carry_convolution,
        replace=True,
    )



__all__ = [
    "FrozenGeneratingFunction",
    "freeze_custom_function",
    "register_generating_codecs",
]
