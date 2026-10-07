from fractions import Fraction

from komorebi_math.coordinates.combinatorial import binomial_transform
from komorebi_math.coordinates.exact import identity_matrix, matrix_multiply
from komorebi_math.coordinates.modular import reduce_exact_transform_mod_prime
from komorebi_math.coordinates.tensor import (
    coordinate_pairing,
    pairing_invariant,
    tensor_coordinate_transform,
    transport_bilinear_form,
    transport_operator,
    transport_tensor_components,
)
from komorebi_math.coordinates.transform import CoordinateTransform


def test_vector_covector_operator_and_form_transport() -> None:
    transform = CoordinateTransform("old", "new", ((2, 1), (1, 1)), "test")
    vector = (3, -2)
    covector = (5, 7)
    assert pairing_invariant(covector, vector, transform)
    assert coordinate_pairing(covector, vector) == 1

    operator = ((1, 2), (3, 4))
    target_operator = transport_operator(operator, transform)
    target_vector = transform.apply(vector)
    old_output = transform.apply((operator[0][0] * vector[0] + operator[0][1] * vector[1], operator[1][0] * vector[0] + operator[1][1] * vector[1]))
    new_output = (
        target_operator[0][0] * target_vector[0] + target_operator[0][1] * target_vector[1],
        target_operator[1][0] * target_vector[0] + target_operator[1][1] * target_vector[1],
    )
    assert new_output == old_output

    form = ((2, 1), (1, 3))
    target_form = transport_bilinear_form(form, transform)
    old_quadratic = sum(form[i][j] * vector[i] * vector[j] for i in range(2) for j in range(2))
    new_quadratic = sum(target_form[i][j] * target_vector[i] * target_vector[j] for i in range(2) for j in range(2))
    assert new_quadratic == old_quadratic


def test_higher_tensor_transport_and_prime_field_reduction() -> None:
    transform = CoordinateTransform("old", "new", ((1, 1), (0, 1)), "test")
    tensor_transform = tensor_coordinate_transform(transform, ("contravariant", "covariant"))
    components = (1, 2, 3, 4)
    assert transport_tensor_components(components, transform, ("contravariant", "covariant")) == tensor_transform.apply(components)
    assert tensor_transform.roundtrip_valid()

    rational = binomial_transform(9)
    modular = reduce_exact_transform_mod_prime(rational, 7)
    inverse = modular.inverse()
    values = tuple(range(9))
    assert inverse.apply(modular.apply(values)) == tuple(value % 7 for value in values)
    modular.certificate().require()
