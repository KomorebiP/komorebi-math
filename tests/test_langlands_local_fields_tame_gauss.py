import numpy as np
import pytest

from komorebi_math.exceptions import ValidationError
from komorebi_math.langlands.local import (
    FiniteField,
    FrobeniusOrbit,
    SpecialMonodromyBlock,
    TameWeilPacket,
    gauss_sum,
    is_irreducible_polynomial,
    regular_orbit_count,
    tame_tensor_refinement,
)


def test_prime_power_field_arithmetic() -> None:
    field = FiniteField(2, 3, (1, 1, 0, 1))
    assert field.order == 8
    for value in range(1, field.order):
        assert field.multiply(value, field.inverse(value)) == 1
    assert len(field.logarithm_table) == 7
    assert {field.trace(value) for value in range(field.order)} <= {0, 1}
    assert {field.norm(value) for value in range(field.order)} <= {0, 1}


def test_reducible_modulus_rejected() -> None:
    assert not is_irreducible_polynomial((1, 0, 1), 2)
    with pytest.raises(ValidationError):
        FiniteField(2, 2, (1, 0, 1))


def test_gauss_sum_magnitude() -> None:
    field = FiniteField(3, 2, (1, 0, 1))
    result = gauss_sum(field, exponent=1)
    result.certificate.require()
    assert abs(result.magnitude - 3.0) < 1e-9


def test_tame_orbit_and_matrix_relation() -> None:
    orbit = FrobeniusOrbit(2, 3, 1)
    orbit.certificate().require()
    assert orbit.elements == (1, 2, 4)
    assert orbit.regular
    packet = TameWeilPacket(orbit)
    packet.relation_certificate().require()
    assert abs(np.linalg.det(packet.frobenius_matrix()) - packet.determinant_orientation) < 1e-10


def test_regular_orbit_count_by_enumeration() -> None:
    for q, degree in ((2, 2), (2, 3), (3, 2), (3, 3)):
        modulus = q**degree - 1
        regular = sum(FrobeniusOrbit(q, degree, exponent).regular for exponent in range(modulus))
        assert regular_orbit_count(q, degree) * degree == regular


def test_tame_tensor_refinement() -> None:
    result = tame_tensor_refinement(6, 4)
    result.certificate.require()
    assert result.packet_count == 2
    assert result.packet_dimension == 12
    assert result.total_dimension == 24


def test_special_monodromy_block() -> None:
    block = SpecialMonodromyBlock(4, 5.0)
    block.certificate().require()
    assert np.linalg.matrix_power(block.nilpotent(), 4).sum() == 0
