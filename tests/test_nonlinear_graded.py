from __future__ import annotations

import numpy as np
import pytest

from komorebi_math import Exactness, PacketSystem
from komorebi_math.exceptions import ValidationError
from komorebi_math.nonlinear.graded import (
    FiniteAbelianGroup,
    GradedPacketOperation,
    divided_square_packets,
    higher_maurer_cartan_packets,
    maurer_cartan_packets,
)


def test_finite_abelian_group_table_and_cyclic_sums() -> None:
    group = FiniteAbelianGroup.cyclic(5)
    group.certificate().require()
    assert group.add("3", "4") == "2"
    assert group.sum(("1", "4", "3")) == "3"
    assert group.inverse("2") == "3"
    with pytest.raises(ValidationError, match="every ordered label pair"):
        FiniteAbelianGroup(("0", "1"), {("0", "0"): "0"}, zero="0")


def matrix_difference_packets():
    sp = pytest.importorskip("sympy")
    group = FiniteAbelianGroup.cyclic(2)
    grade_zero = sp.diag(1, 0, 0, 1)
    grade_one = sp.diag(0, 1, 1, 0)
    packets = PacketSystem((grade_zero, grade_one), names=group.labels, backend="sympy")

    def multiply(values):
        left = sp.Matrix(values[0]).reshape(2, 2)
        right = sp.Matrix(values[1]).reshape(2, 2)
        return (left * right).reshape(4, 1)

    return sp, group, packets, multiply


def test_matrix_difference_grading_and_convolution_exact() -> None:
    sp, group, packets, multiply = matrix_difference_packets()
    operation = GradedPacketOperation(
        group,
        packets,
        2,
        multiply,
        exactness=Exactness.EXACT,
    )
    operation.grading_certificate().certificate.require()
    left = sp.Matrix([1, 2, 3, 4])
    right = sp.Matrix([5, 6, 7, 8])
    for output in group.labels:
        result = operation.packet_convolution((left, right), output)
        result.certificate.require()
        assert result.value == result.direct_value
        assert len(result.terms) == 2


def test_packet_maurer_cartan_reconstruction_exact() -> None:
    sp = pytest.importorskip("sympy")
    group = FiniteAbelianGroup.cyclic(3)
    components = {
        "0": sp.Matrix([1, 0]),
        "1": sp.Matrix([0, 2]),
        "2": sp.Matrix([3, 4]),
    }
    differential = lambda value: 2 * value
    bracket = lambda left, right: sp.Matrix(
        [left[0] * right[0], left[1] * right[1]]
    )
    result = maurer_cartan_packets(
        group,
        components,
        differential,
        bracket,
        exactness=Exactness.EXACT,
    )
    result.certificate.require()
    assert result.reconstructed == result.direct
    assert set(result.equations) == set(group.labels)


def test_higher_maurer_cartan_finite_truncation() -> None:
    group = FiniteAbelianGroup.cyclic(2)
    components = {"0": np.array([1.0, 0.5]), "1": np.array([-0.2, 0.3])}
    operations = {
        1: lambda values: 2.0 * values[0],
        2: lambda values: values[0] * values[1],
        3: lambda values: values[0] * values[1] * values[2],
    }
    result = higher_maurer_cartan_packets(group, components, operations)
    result.certificate.require()
    assert np.allclose(result.reconstructed, result.direct)
    assert result.max_order == 3


def test_divided_square_packet_identity() -> None:
    group = FiniteAbelianGroup.cyclic(3)
    components = {
        "0": np.array([1.0, 2.0]),
        "1": np.array([-0.5, 0.25]),
        "2": np.array([0.3, -0.7]),
    }
    result = divided_square_packets(
        group,
        components,
        differential=lambda value: 3.0 * value,
        bracket=lambda left, right: 2.0 * left * right,
        divided_square=lambda value: value * value,
    )
    result.certificate.require()
    assert np.allclose(result.reconstructed, result.direct)
