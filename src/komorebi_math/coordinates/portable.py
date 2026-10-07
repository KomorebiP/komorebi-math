"""Trusted portable codecs for finite coordinate-calculus records."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..exceptions import ValidationError
from ..export.persistence import (
    load_portable,
    register_portable_codec,
    save_portable,
)
from .affine import AffineCoordinateTransform, BarycentricCoordinateSystem
from .analysis_synthesis import AnalysisSynthesisSystem
from .basis import CoordinateAtlas, CoordinateSystem, FiniteBasis
from .budget import CoordinateBudget
from .calculus import PolynomialCoordinateOperator
from .exact import as_matrix, as_vector
from .families import RecurrenceModel
from .formal import FormalCoordinateChange
from .modular import FiniteFieldCoordinateTransform
from .numeric import ModalCoordinateSystem, NumericCoordinateTransform
from .operators import BezoutDefectIdentity, ShiftPolynomial
from .realization import IntertwinerSpace, LinearRealization, StateRecovery
from .recurrence import RecurrenceCoordinateRecord
from .state_space import StateSpaceModel
from .transform import CoordinateTransform


def _require_fields(data: Mapping[str, Any], fields: set[str], *, kind: str) -> None:
    if set(data) != fields:
        raise ValidationError(f"{kind} portable data does not match its schema")


def _decode_budget(data: Mapping[str, Any]) -> CoordinateBudget:
    fields = {
        "maximum_matrix_dimension",
        "maximum_polynomial_degree",
        "maximum_horizon",
        "maximum_exact_entries",
    }
    _require_fields(data, fields, kind="coordinate budget")
    return CoordinateBudget(**{key: int(data[key]) for key in fields})


def _decode_transform(data: Mapping[str, Any]) -> CoordinateTransform:
    fields = {"source", "target", "matrix", "theorem", "description"}
    _require_fields(data, fields, kind="coordinate transform")
    return CoordinateTransform(
        str(data["source"]),
        str(data["target"]),
        as_matrix(data["matrix"]),
        str(data["theorem"]),
        str(data["description"]),
    )


def _decode_recurrence_record(data: Mapping[str, Any]) -> RecurrenceCoordinateRecord:
    fields = {"name", "coefficients", "initial", "defects"}
    _require_fields(data, fields, kind="recurrence coordinates")
    record = RecurrenceCoordinateRecord(
        str(data["name"]),
        as_vector(data["coefficients"]),
        as_vector(data["initial"]),
        as_vector(data["defects"]),
    )
    record.certificate().require()
    return record


def _decode_realization(data: Mapping[str, Any]) -> LinearRealization:
    fields = {"transition", "initial", "output", "name"}
    _require_fields(data, fields, kind="linear realization")
    return LinearRealization(
        as_matrix(data["transition"]),
        as_vector(data["initial"]),
        as_vector(data["output"]),
        str(data["name"]),
    )


def _decode_state_recovery(data: Mapping[str, Any]) -> StateRecovery:
    fields = {"exact", "unique", "rank", "horizon", "state", "predicted", "residual"}
    _require_fields(data, fields, kind="state recovery")
    raw_state = data["state"]
    state = None if raw_state is None else as_vector(raw_state)
    return StateRecovery(
        bool(data["exact"]),
        bool(data["unique"]),
        int(data["rank"]),
        int(data["horizon"]),
        state,
        as_vector(data["predicted"]),
        as_vector(data["residual"]),
    )


def _decode_intertwiner(data: Mapping[str, Any]) -> IntertwinerSpace:
    fields = {"source_dimension", "target_dimension", "basis"}
    _require_fields(data, fields, kind="intertwiner space")
    return IntertwinerSpace(
        int(data["source_dimension"]),
        int(data["target_dimension"]),
        tuple(as_matrix(matrix) for matrix in data["basis"]),
    )


def _decode_shift_polynomial(data: Mapping[str, Any]) -> ShiftPolynomial:
    _require_fields(data, {"coefficients", "name"}, kind="shift polynomial")
    return ShiftPolynomial(as_vector(data["coefficients"]), str(data["name"]))


def _decode_bezout(data: Mapping[str, Any]) -> BezoutDefectIdentity:
    fields = {
        "gcd",
        "left_multiplier",
        "right_multiplier",
        "reconstructed",
        "expected",
        "valid",
    }
    _require_fields(data, fields, kind="Bezout defect identity")
    result = BezoutDefectIdentity(
        as_vector(data["gcd"]),
        as_vector(data["left_multiplier"]),
        as_vector(data["right_multiplier"]),
        as_vector(data["reconstructed"]),
        as_vector(data["expected"]),
        bool(data["valid"]),
    )
    if bool(data["valid"]):
        result.certificate().require()
    return result


def _decode_coordinate_system(data: Mapping[str, Any]) -> CoordinateSystem:
    fields = {"name", "dimension", "kind", "description"}
    _require_fields(data, fields, kind="coordinate system")
    return CoordinateSystem(
        str(data["name"]),
        int(data["dimension"]),
        str(data["kind"]),
        str(data["description"]),
    )



def _decode_coordinate_atlas(data: Mapping[str, Any]) -> CoordinateAtlas:
    fields = {"name", "systems", "transforms"}
    _require_fields(data, fields, kind="coordinate atlas")
    atlas = CoordinateAtlas(str(data["name"]))
    for system in data["systems"]:
        if not isinstance(system, CoordinateSystem):
            raise ValidationError("coordinate atlas systems must decode to CoordinateSystem records")
        atlas.register_system(system)
    for transform in data["transforms"]:
        if not isinstance(transform, CoordinateTransform):
            raise ValidationError("coordinate atlas edges must decode to CoordinateTransform records")
        atlas.register_transform(transform, include_inverse=False)
    return atlas


def _decode_finite_basis(data: Mapping[str, Any]) -> FiniteBasis:
    fields = {"name", "vectors", "labels", "description"}
    _require_fields(data, fields, kind="finite basis")
    return FiniteBasis(
        str(data["name"]),
        tuple(as_vector(vector) for vector in data["vectors"]),
        tuple(str(label) for label in data["labels"]),
        str(data["description"]),
    )


def _decode_affine_transform(data: Mapping[str, Any]) -> AffineCoordinateTransform:
    fields = {"source", "target", "matrix", "offset", "theorem", "description"}
    _require_fields(data, fields, kind="affine coordinate transform")
    return AffineCoordinateTransform(
        str(data["source"]),
        str(data["target"]),
        as_matrix(data["matrix"]),
        as_vector(data["offset"]),
        str(data["theorem"]),
        str(data["description"]),
    )


def _decode_barycentric(data: Mapping[str, Any]) -> BarycentricCoordinateSystem:
    fields = {"name", "vertices"}
    _require_fields(data, fields, kind="barycentric coordinate system")
    return BarycentricCoordinateSystem(
        str(data["name"]),
        tuple(as_vector(vertex) for vertex in data["vertices"]),
    )


def _decode_formal_change(data: Mapping[str, Any]) -> FormalCoordinateChange:
    fields = {"name", "coefficients", "source", "target"}
    _require_fields(data, fields, kind="formal coordinate change")
    return FormalCoordinateChange(
        str(data["name"]),
        as_vector(data["coefficients"]),
        str(data["source"]),
        str(data["target"]),
    )


def _decode_finite_field_transform(data: Mapping[str, Any]) -> FiniteFieldCoordinateTransform:
    fields = {"source", "target", "matrix", "prime", "theorem", "description"}
    _require_fields(data, fields, kind="finite-field coordinate transform")
    return FiniteFieldCoordinateTransform(
        str(data["source"]),
        str(data["target"]),
        tuple(tuple(int(value) for value in row) for row in data["matrix"]),
        int(data["prime"]),
        str(data["theorem"]),
        str(data["description"]),
    )




def _decode_analysis_synthesis(data: Mapping[str, Any]) -> AnalysisSynthesisSystem:
    fields = {"analysis", "synthesis", "source", "coordinates", "tolerance", "theorem"}
    _require_fields(data, fields, kind="analysis--synthesis system")
    system = AnalysisSynthesisSystem(
        data["analysis"],
        data["synthesis"],
        str(data["source"]),
        str(data["coordinates"]),
        float(data["tolerance"]),
        str(data["theorem"]),
    )
    system.certificate().require()
    return system

def _decode_numeric_transform(data: Mapping[str, Any]) -> NumericCoordinateTransform:
    fields = {"source", "target", "matrix", "theorem", "description", "tolerance"}
    _require_fields(data, fields, kind="numeric coordinate transform")
    return NumericCoordinateTransform(
        str(data["source"]),
        str(data["target"]),
        data["matrix"],
        str(data["theorem"]),
        str(data["description"]),
        float(data["tolerance"]),
    )


def _decode_modal_system(data: Mapping[str, Any]) -> ModalCoordinateSystem:
    fields = {"eigenvalues", "synthesis", "analysis", "diagonal_residual"}
    _require_fields(data, fields, kind="modal coordinate system")
    synthesis = data["synthesis"]
    analysis = data["analysis"]
    if not isinstance(synthesis, NumericCoordinateTransform) or not isinstance(analysis, NumericCoordinateTransform):
        raise ValidationError("modal coordinate transforms decoded with incompatible types")
    return ModalCoordinateSystem(
        data["eigenvalues"],
        synthesis,
        analysis,
        float(data["diagonal_residual"]),
    )


def _decode_recurrence_model(data: Mapping[str, Any]) -> RecurrenceModel:
    fields = {"name", "coefficients", "initial_values", "description"}
    _require_fields(data, fields, kind="recurrence model")
    return RecurrenceModel(
        str(data["name"]),
        as_vector(data["coefficients"]),
        as_vector(data["initial_values"]),
        str(data["description"]),
    )


def _decode_state_space_model(data: Mapping[str, Any]) -> StateSpaceModel:
    fields = {"name", "transition", "input_matrix", "output_matrix", "feedthrough"}
    _require_fields(data, fields, kind="state-space model")
    return StateSpaceModel(
        str(data["name"]),
        as_matrix(data["transition"]),
        as_matrix(data["input_matrix"]),
        as_matrix(data["output_matrix"]),
        as_matrix(data["feedthrough"]),
    )


def _decode_polynomial_operator(data: Mapping[str, Any]) -> PolynomialCoordinateOperator:
    fields = {
        "name",
        "source_family",
        "target_family",
        "source_degree",
        "target_degree",
        "matrix",
        "theorem",
    }
    _require_fields(data, fields, kind="polynomial coordinate operator")
    return PolynomialCoordinateOperator(
        str(data["name"]),
        str(data["source_family"]),
        str(data["target_family"]),
        int(data["source_degree"]),
        int(data["target_degree"]),
        as_matrix(data["matrix"]),
        str(data["theorem"]),
    )


def register_coordinate_codecs() -> None:
    """Register the closed data-only coordinate codecs."""

    register_portable_codec(
        CoordinateBudget,
        tag="coordinates.budget",
        version=1,
        encoder=lambda value: {
            "maximum_matrix_dimension": value.maximum_matrix_dimension,
            "maximum_polynomial_degree": value.maximum_polynomial_degree,
            "maximum_horizon": value.maximum_horizon,
            "maximum_exact_entries": value.maximum_exact_entries,
        },
        decoder=_decode_budget,
        replace=True,
    )
    register_portable_codec(
        CoordinateTransform,
        tag="coordinates.transform",
        version=1,
        encoder=lambda value: {
            "source": value.source,
            "target": value.target,
            "matrix": value.matrix,
            "theorem": value.theorem,
            "description": value.description,
        },
        decoder=_decode_transform,
        replace=True,
    )
    register_portable_codec(
        RecurrenceCoordinateRecord,
        tag="coordinates.recurrence_record",
        version=1,
        encoder=lambda value: {
            "name": value.name,
            "coefficients": value.coefficients,
            "initial": value.initial,
            "defects": value.defects,
        },
        decoder=_decode_recurrence_record,
        replace=True,
    )
    register_portable_codec(
        LinearRealization,
        tag="coordinates.linear_realization",
        version=1,
        encoder=lambda value: {
            "transition": value.transition,
            "initial": value.initial,
            "output": value.output,
            "name": value.name,
        },
        decoder=_decode_realization,
        replace=True,
    )
    register_portable_codec(
        StateRecovery,
        tag="coordinates.state_recovery",
        version=1,
        encoder=lambda value: {
            "exact": value.exact,
            "unique": value.unique,
            "rank": value.rank,
            "horizon": value.horizon,
            "state": value.state,
            "predicted": value.predicted,
            "residual": value.residual,
        },
        decoder=_decode_state_recovery,
        replace=True,
    )
    register_portable_codec(
        IntertwinerSpace,
        tag="coordinates.intertwiner_space",
        version=1,
        encoder=lambda value: {
            "source_dimension": value.source_dimension,
            "target_dimension": value.target_dimension,
            "basis": value.basis,
        },
        decoder=_decode_intertwiner,
        replace=True,
    )
    register_portable_codec(
        ShiftPolynomial,
        tag="coordinates.shift_polynomial",
        version=1,
        encoder=lambda value: {
            "coefficients": value.coefficients,
            "name": value.name,
        },
        decoder=_decode_shift_polynomial,
        replace=True,
    )
    register_portable_codec(
        BezoutDefectIdentity,
        tag="coordinates.bezout_defect_identity",
        version=1,
        encoder=lambda value: {
            "gcd": value.gcd,
            "left_multiplier": value.left_multiplier,
            "right_multiplier": value.right_multiplier,
            "reconstructed": value.reconstructed,
            "expected": value.expected,
            "valid": value.valid,
        },
        decoder=_decode_bezout,
        replace=True,
    )

    register_portable_codec(
        CoordinateSystem,
        tag="coordinates.system",
        version=1,
        encoder=lambda value: {
            "name": value.name,
            "dimension": value.dimension,
            "kind": value.kind,
            "description": value.description,
        },
        decoder=_decode_coordinate_system,
        replace=True,
    )
    register_portable_codec(
        CoordinateAtlas,
        tag="coordinates.atlas",
        version=1,
        encoder=lambda value: {
            "name": value.name,
            "systems": value.systems,
            "transforms": value.transforms,
        },
        decoder=_decode_coordinate_atlas,
        replace=True,
    )
    register_portable_codec(
        FiniteBasis,
        tag="coordinates.finite_basis",
        version=1,
        encoder=lambda value: {
            "name": value.name,
            "vectors": value.vectors,
            "labels": value.labels,
            "description": value.description,
        },
        decoder=_decode_finite_basis,
        replace=True,
    )
    register_portable_codec(
        AffineCoordinateTransform,
        tag="coordinates.affine_transform",
        version=1,
        encoder=lambda value: {
            "source": value.source,
            "target": value.target,
            "matrix": value.matrix,
            "offset": value.offset,
            "theorem": value.theorem,
            "description": value.description,
        },
        decoder=_decode_affine_transform,
        replace=True,
    )
    register_portable_codec(
        BarycentricCoordinateSystem,
        tag="coordinates.barycentric_system",
        version=1,
        encoder=lambda value: {
            "name": value.name,
            "vertices": value.vertices,
        },
        decoder=_decode_barycentric,
        replace=True,
    )
    register_portable_codec(
        FormalCoordinateChange,
        tag="coordinates.formal_change",
        version=1,
        encoder=lambda value: {
            "name": value.name,
            "coefficients": value.coefficients,
            "source": value.source,
            "target": value.target,
        },
        decoder=_decode_formal_change,
        replace=True,
    )
    register_portable_codec(
        FiniteFieldCoordinateTransform,
        tag="coordinates.finite_field_transform",
        version=1,
        encoder=lambda value: {
            "source": value.source,
            "target": value.target,
            "matrix": value.matrix,
            "prime": value.prime,
            "theorem": value.theorem,
            "description": value.description,
        },
        decoder=_decode_finite_field_transform,
        replace=True,
    )
    register_portable_codec(
        AnalysisSynthesisSystem,
        tag="coordinates.analysis_synthesis",
        version=1,
        encoder=lambda value: {
            "analysis": value.analysis,
            "synthesis": value.synthesis,
            "source": value.source,
            "coordinates": value.coordinates,
            "tolerance": value.tolerance,
            "theorem": value.theorem,
        },
        decoder=_decode_analysis_synthesis,
        replace=True,
    )
    register_portable_codec(
        NumericCoordinateTransform,
        tag="coordinates.numeric_transform",
        version=1,
        encoder=lambda value: {
            "source": value.source,
            "target": value.target,
            "matrix": value.matrix,
            "theorem": value.theorem,
            "description": value.description,
            "tolerance": value.tolerance,
        },
        decoder=_decode_numeric_transform,
        replace=True,
    )
    register_portable_codec(
        ModalCoordinateSystem,
        tag="coordinates.modal_system",
        version=1,
        encoder=lambda value: {
            "eigenvalues": value.eigenvalues,
            "synthesis": value.synthesis,
            "analysis": value.analysis,
            "diagonal_residual": value.diagonal_residual,
        },
        decoder=_decode_modal_system,
        replace=True,
    )
    register_portable_codec(
        RecurrenceModel,
        tag="coordinates.recurrence_model",
        version=1,
        encoder=lambda value: {
            "name": value.name,
            "coefficients": value.coefficients,
            "initial_values": value.initial_values,
            "description": value.description,
        },
        decoder=_decode_recurrence_model,
        replace=True,
    )
    register_portable_codec(
        StateSpaceModel,
        tag="coordinates.state_space_model",
        version=1,
        encoder=lambda value: {
            "name": value.name,
            "transition": value.transition,
            "input_matrix": value.input_matrix,
            "output_matrix": value.output_matrix,
            "feedthrough": value.feedthrough,
        },
        decoder=_decode_state_space_model,
        replace=True,
    )
    register_portable_codec(
        PolynomialCoordinateOperator,
        tag="coordinates.polynomial_operator",
        version=1,
        encoder=lambda value: {
            "name": value.name,
            "source_family": value.source_family,
            "target_family": value.target_family,
            "source_degree": value.source_degree,
            "target_degree": value.target_degree,
            "matrix": value.matrix,
            "theorem": value.theorem,
        },
        decoder=_decode_polynomial_operator,
        replace=True,
    )


def save_transform(
    path: str | Path,
    transform: CoordinateTransform,
    *,
    overwrite: bool = False,
) -> Path:
    """Save one exact coordinate transform with the package persistence envelope."""

    if not isinstance(transform, CoordinateTransform):
        raise ValidationError("transform must be a CoordinateTransform")
    return save_portable(transform, path, overwrite=overwrite)


def load_transform(path: str | Path) -> CoordinateTransform:
    """Load and verify one exact coordinate transform."""

    value = load_portable(path)
    if not isinstance(value, CoordinateTransform):
        raise ValidationError("portable file does not contain a CoordinateTransform")
    return value


def save_coordinate_record(
    path: str | Path,
    record: RecurrenceCoordinateRecord,
    *,
    overwrite: bool = False,
) -> Path:
    """Save recurrence defect coordinates with the package persistence envelope."""

    if not isinstance(record, RecurrenceCoordinateRecord):
        raise ValidationError("record must be a RecurrenceCoordinateRecord")
    return save_portable(record, path, overwrite=overwrite)


def load_coordinate_record(path: str | Path) -> RecurrenceCoordinateRecord:
    """Load and verify recurrence defect coordinates."""

    value = load_portable(path)
    if not isinstance(value, RecurrenceCoordinateRecord):
        raise ValidationError("portable file does not contain recurrence coordinates")
    return value


def save_coordinate_object(
    path: str | Path,
    value: Any,
    *,
    overwrite: bool = False,
) -> Path:
    """Save any registered portable coordinate object."""

    allowed = (
        CoordinateAtlas,
        CoordinateSystem,
        AnalysisSynthesisSystem,
        FiniteBasis,
        CoordinateTransform,
        AffineCoordinateTransform,
        BarycentricCoordinateSystem,
        FormalCoordinateChange,
        FiniteFieldCoordinateTransform,
        NumericCoordinateTransform,
        ModalCoordinateSystem,
        RecurrenceModel,
        RecurrenceCoordinateRecord,
        LinearRealization,
        StateSpaceModel,
        StateRecovery,
        IntertwinerSpace,
        ShiftPolynomial,
        BezoutDefectIdentity,
        PolynomialCoordinateOperator,
        CoordinateBudget,
    )
    if not isinstance(value, allowed):
        raise ValidationError("value is not a registered portable coordinate object")
    return save_portable(value, path, overwrite=overwrite)


def load_coordinate_object(path: str | Path) -> Any:
    """Load and verify one registered portable coordinate object."""

    value = load_portable(path)
    allowed = (
        CoordinateAtlas,
        CoordinateSystem,
        AnalysisSynthesisSystem,
        FiniteBasis,
        CoordinateTransform,
        AffineCoordinateTransform,
        BarycentricCoordinateSystem,
        FormalCoordinateChange,
        FiniteFieldCoordinateTransform,
        NumericCoordinateTransform,
        ModalCoordinateSystem,
        RecurrenceModel,
        RecurrenceCoordinateRecord,
        LinearRealization,
        StateSpaceModel,
        StateRecovery,
        IntertwinerSpace,
        ShiftPolynomial,
        BezoutDefectIdentity,
        PolynomialCoordinateOperator,
        CoordinateBudget,
    )
    if not isinstance(value, allowed):
        raise ValidationError("portable file does not contain a registered coordinate object")
    return value


__all__ = [
    "load_coordinate_object",
    "load_coordinate_record",
    "load_transform",
    "register_coordinate_codecs",
    "save_coordinate_object",
    "save_coordinate_record",
    "save_transform",
]
