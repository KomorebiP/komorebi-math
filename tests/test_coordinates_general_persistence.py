from fractions import Fraction

import pytest

from komorebi_math.coordinates import (
    AffineCoordinateTransform,
    BarycentricCoordinateSystem,
    CoordinateAtlas,
    CoordinateSystem,
    CoordinateTransform,
    FiniteBasis,
    FiniteFieldCoordinateTransform,
    FormalCoordinateChange,
    NumericCoordinateTransform,
    discrete_fourier_transform,
    modal_coordinate_system,
    PolynomialCoordinateOperator,
    RecurrenceModel,
    StateSpaceModel,
    load_coordinate_object,
    save_coordinate_object,
)


def _roundtrip(tmp_path, name, value):
    path = tmp_path / f"{name}.json"
    save_coordinate_object(path, value)
    return path, load_coordinate_object(path)


def test_general_coordinate_records_roundtrip(tmp_path) -> None:
    basis = FiniteBasis("shear", ((1, 0), (1, 1)), ("u", "v"), "exact shear basis")
    _, restored_basis = _roundtrip(tmp_path, "basis", basis)
    assert restored_basis == basis

    affine = AffineCoordinateTransform("x", "y", ((2, 0), (0, 3)), (1, -2), "CC.P1")
    _, restored_affine = _roundtrip(tmp_path, "affine", affine)
    assert restored_affine == affine

    barycentric = BarycentricCoordinateSystem("triangle", ((0, 0), (1, 0), (0, 1)))
    _, restored_barycentric = _roundtrip(tmp_path, "barycentric", barycentric)
    assert restored_barycentric == barycentric

    formal = FormalCoordinateChange("quadratic chart", (0, 1, Fraction(1, 2)), "z", "w")
    _, restored_formal = _roundtrip(tmp_path, "formal", formal)
    assert restored_formal == formal

    modular = FiniteFieldCoordinateTransform("a", "b", ((1, 2), (3, 4)), 5, "CC.M1")
    _, restored_modular = _roundtrip(tmp_path, "modular", modular)
    assert restored_modular == modular

    numeric = discrete_fourier_transform(4)
    _, restored_numeric = _roundtrip(tmp_path, "numeric", numeric)
    assert isinstance(restored_numeric, NumericCoordinateTransform)
    assert restored_numeric.source == numeric.source
    assert restored_numeric.target == numeric.target
    assert restored_numeric.tolerance == numeric.tolerance
    assert (abs(restored_numeric.matrix - numeric.matrix) == 0).all()

    modal = modal_coordinate_system(((2.0, 0.0), (0.0, 3.0)))
    _, restored_modal = _roundtrip(tmp_path, "modal", modal)
    assert (abs(restored_modal.eigenvalues - modal.eigenvalues) == 0).all()
    assert (abs(restored_modal.synthesis.matrix - modal.synthesis.matrix) == 0).all()
    assert restored_modal.diagonal_residual == modal.diagonal_residual

    recurrence = RecurrenceModel("Pell", (1, 2), (0, 1), "Pell recurrence")
    _, restored_recurrence = _roundtrip(tmp_path, "recurrence", recurrence)
    assert restored_recurrence == recurrence

    state = StateSpaceModel("system", ((0, 1), (-2, -3)), ((0,), (1,)), ((1, 0),), ((0,),))
    _, restored_state = _roundtrip(tmp_path, "state", state)
    assert restored_state == state

    operator = PolynomialCoordinateOperator("D", "Legendre", "Legendre", 2, 2, ((0, 1, 0), (0, 0, 3), (0, 0, 0)))
    _, restored_operator = _roundtrip(tmp_path, "operator", operator)
    assert restored_operator == operator


def test_coordinate_atlas_roundtrip_preserves_graph(tmp_path) -> None:
    atlas = CoordinateAtlas("general atlas")
    atlas.register_system(CoordinateSystem("A", 2, "basis"))
    atlas.register_system(CoordinateSystem("B", 2, "basis"))
    atlas.register_system(CoordinateSystem("C", 2, "basis"))
    ab = CoordinateTransform("A", "B", ((1, 1), (0, 1)), "CC.A1")
    bc = CoordinateTransform("B", "C", ((1, 0), (1, 1)), "CC.A1")
    atlas.register_transform(ab, include_inverse=True)
    atlas.register_transform(bc, include_inverse=True)

    _, restored = _roundtrip(tmp_path, "atlas", atlas)
    assert isinstance(restored, CoordinateAtlas)
    assert restored.name == atlas.name
    assert restored.systems == atlas.systems
    assert restored.transforms == atlas.transforms
    assert restored.resolve("A", "C").transform == atlas.resolve("A", "C").transform


def test_general_coordinate_portable_tamper_rejection(tmp_path) -> None:
    formal = FormalCoordinateChange("chart", (0, 1, 1), "z", "w")
    path, _ = _roundtrip(tmp_path, "tamper", formal)
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("\"numerator\": 1", "\"numerator\": 2", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256|integrity"):
        load_coordinate_object(path)
