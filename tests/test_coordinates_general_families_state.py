from fractions import Fraction

from komorebi_math.coordinates.families import (
    fibonacci_recurrence_model,
    jacobsthal_recurrence_model,
    lucas_recurrence_model,
    padovan_recurrence_model,
    pell_recurrence_model,
    recurrence_model_catalog,
    sequence_coordinates_between_models,
    tribonacci_recurrence_model,
)
from komorebi_math.coordinates.state_space import (
    certify_state_coordinate_change,
    controllability_coordinate_transform,
    observability_coordinate_transform,
    siso_state_space,
)


def test_named_recurrence_models_and_general_coordinate_bridge() -> None:
    assert fibonacci_recurrence_model().sequence(10) == (0, 1, 1, 2, 3, 5, 8, 13, 21, 34)
    assert lucas_recurrence_model().sequence(8) == (2, 1, 3, 4, 7, 11, 18, 29)
    assert pell_recurrence_model().sequence(8) == (0, 1, 2, 5, 12, 29, 70, 169)
    assert jacobsthal_recurrence_model().sequence(8) == (0, 1, 1, 3, 5, 11, 21, 43)
    assert tribonacci_recurrence_model().sequence(9) == (0, 0, 1, 1, 2, 4, 7, 13, 24)
    assert padovan_recurrence_model().sequence(9) == (1, 1, 1, 2, 2, 3, 4, 5, 7)
    assert len(recurrence_model_catalog()) >= 6

    values = (3, -1, 4, 1, 5, -9, 2, 6, 5, 3, 5, 8)
    source_record, target_record, transform = sequence_coordinates_between_models(
        values,
        pell_recurrence_model(),
        tribonacci_recurrence_model(),
    )
    assert transform.apply(source_record.coordinates()) == target_record.coordinates()
    assert target_record.reconstruct() == tuple(Fraction(value) for value in values)


def test_recurrence_model_synthesize_roundtrips_defect_coordinates() -> None:
    model = fibonacci_recurrence_model()
    values = (3, -1, 4, 1, 5, -9, 2, 6)
    record = model.analyze(values)
    assert model.synthesize(record.coordinates()) == tuple(Fraction(value) for value in values)


def test_homogeneous_recurrence_bases_are_not_family_specific() -> None:
    for model in recurrence_model_catalog():
        basis = model.homogeneous_basis(14)
        assert basis.dimension == model.order
        assert basis.ambient_dimension == 14
        assert basis.certificate().valid
        for vector in basis.vectors:
            assert model.analyze(vector).defects == tuple(Fraction(0) for _ in range(14 - model.order))


def test_controllability_and_observability_coordinates() -> None:
    model = siso_state_space(
        ((0, 1, 0), (0, 0, 1), (2, -1, 3)),
        (0, 0, 1),
        (1, 2, -1),
        feedthrough=2,
        name="generic third-order model",
    )
    controllability = controllability_coordinate_transform(model)
    certificate = certify_state_coordinate_change(model, controllability)
    certificate.certificate().require()
    transformed = model.change_coordinates(controllability)
    assert model.transfer_equivalent(transformed, count=12)

    observability = observability_coordinate_transform(model)
    state = (Fraction(2), Fraction(-3), Fraction(5))
    assert observability.inverse().apply(observability.apply(state)) == state
