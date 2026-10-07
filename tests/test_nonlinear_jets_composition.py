from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from komorebi_math import Exactness, PacketSystem
from komorebi_math.exceptions import ValidationError
from komorebi_math.nonlinear.composition import (
    ComposedDerivativeProvider,
    faa_di_bruno,
    labeled_composition_term_count,
    packet_faa_di_bruno,
    set_partitions,
    stirling_second,
)
from komorebi_math.nonlinear.derivatives import (
    CallableDerivativeProvider,
    FiniteDifferenceDerivativeProvider,
    PolynomialDerivativeProvider,
    SympyDerivativeProvider,
)
from komorebi_math.nonlinear.jets import PacketJetAddress, PacketJetFamily, packet_jet_support, packet_taylor


def coordinate_packets(*, exact: bool = False) -> PacketSystem:
    if exact:
        sp = pytest.importorskip("sympy")
        return PacketSystem(
            [sp.diag(1, 0), sp.diag(0, 1)],
            names=("x", "y"),
            backend="sympy",
        )
    return PacketSystem(
        [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])],
        names=("x", "y"),
    )


def test_set_partition_and_labeled_term_counts() -> None:
    assert tuple(len(set_partitions(order)) for order in range(7)) == (1, 1, 2, 5, 15, 52, 203)
    assert stirling_second(5, 2) == 15
    assert stirling_second(5, 3) == 25
    assert labeled_composition_term_count(4, 3) == 3 + 7 * 9 + 6 * 27 + 81
    assert len(set(set_partitions(6))) == 203


def test_polynomial_derivative_provider_exact_scaling_and_symmetry() -> None:
    # F(x) = 1 + 2x + 3x^2, encoded as T_0=1, T_1=2, T_2=6.
    provider = PolynomialDerivativeProvider(
        {
            0: np.array([1], dtype=object),
            1: np.array([[2]], dtype=object),
            2: np.array([[[6]]], dtype=object),
        },
        exactness=Exactness.EXACT,
    )
    value = provider.value(np.array([Fraction(2, 3)], dtype=object))[0]
    derivative = provider.derivative(
        1,
        np.array([Fraction(2, 3)], dtype=object),
        (np.array([Fraction(3, 5)], dtype=object),),
    )[0]
    assert value == Fraction(11, 3)
    assert derivative == Fraction(18, 5)
    provider.symmetry_certificate().require()


def test_declared_and_finite_difference_derivative_providers() -> None:
    provider = CallableDerivativeProvider(
        function=lambda x: np.array([x[0] ** 3 + x[1] ** 2]),
        derivatives={
            1: lambda x, d: np.array([(3 * x[0] ** 2) * d[0][0] + 2 * x[1] * d[0][1]]),
            2: lambda x, d: np.array([6 * x[0] * d[0][0] * d[1][0] + 2 * d[0][1] * d[1][1]]),
            3: lambda x, d: np.array([6 * d[0][0] * d[1][0] * d[2][0]]),
        },
        source_dimension=2,
        target_dimension=1,
    )
    certificate = provider.validation_certificate(
        np.array([0.4, -0.2]),
        (np.array([1.0, 2.0]), np.array([-0.5, 1.0]), np.array([0.2, -0.4])),
    )
    certificate.require()

    numerical = FiniteDifferenceDerivativeProvider(
        lambda x: np.array([x[0] ** 3 + np.sin(x[1])]),
        source_dimension=2,
        target_dimension=1,
        step=2e-4,
        max_order=2,
    )
    point = np.array([0.7, -0.3])
    u = np.array([0.4, -0.8])
    v = np.array([-0.2, 0.5])
    expected = 6 * point[0] * u[0] * v[0] - np.sin(point[1]) * u[1] * v[1]
    assert np.allclose(numerical.derivative(2, point, (u, v)), [expected], rtol=2e-4, atol=2e-5)


def test_exact_packet_jet_reconstruction_support_and_taylor() -> None:
    sp = pytest.importorskip("sympy")
    x, y = sp.symbols("x y")
    provider = SympyDerivativeProvider((x**2, y**2), (x, y))
    packets = coordinate_packets(exact=True)
    point = sp.Matrix([1, 2])
    directions = (sp.Matrix([2, 3]), sp.Matrix([-1, 4]))

    family = PacketJetFamily(provider, point, packets, packets, 2)
    family.reconstruction_certificate(directions).require()
    family.jet("x", ("x", "x")).symmetry_certificate(directions).require()

    support = packet_jet_support(family)
    support.certificate.require()
    assert support.complete
    assert set(support.nonzero) == {
        PacketJetAddress("x", ("x", "x")),
        PacketJetAddress("y", ("y", "y")),
    }
    assert len(support.zero) == 6

    taylor = packet_taylor(provider, point, sp.Matrix([3, -1]), packets, packets, 2)
    taylor.certificate.require()
    assert taylor.truncation_residual == 0.0
    assert taylor.approximation == taylor.observed


def test_packet_support_honors_evaluation_limit_without_materializing_product() -> None:
    packets = coordinate_packets()
    tensor = np.ones((2, 2, 2, 2, 2), dtype=float)
    provider = PolynomialDerivativeProvider({4: tensor}, source_dimension=2, target_dimension=2)
    family = PacketJetFamily(provider, np.ones(2), packets, packets, 4)
    support = packet_jet_support(family, max_evaluations=1)
    assert support.evaluations == 1
    assert support.untested
    assert support.certificate.exactness is Exactness.CONDITIONAL


def symbolic_composition_providers():
    sp = pytest.importorskip("sympy")
    x, y = sp.symbols("x y")
    u, v = sp.symbols("u v")
    inner_expressions = (x**2 + y, x * y)
    outer_expressions = (u**2 + v, u * v**2)
    direct_expressions = tuple(
        expression.subs({u: inner_expressions[0], v: inner_expressions[1]})
        for expression in outer_expressions
    )
    return (
        sp,
        SympyDerivativeProvider(outer_expressions, (u, v), name="outer"),
        SympyDerivativeProvider(inner_expressions, (x, y), name="inner"),
        SympyDerivativeProvider(direct_expressions, (x, y), name="direct"),
    )


def test_exact_faa_di_bruno_through_order_four() -> None:
    sp, outer, inner, direct = symbolic_composition_providers()
    point = sp.Matrix([2, -1])
    directions = (
        sp.Matrix([1, 2]),
        sp.Matrix([-1, 3]),
        sp.Matrix([2, 1]),
        sp.Matrix([3, -2]),
    )
    for order in range(1, 5):
        result = faa_di_bruno(
            outer,
            inner,
            point,
            directions[:order],
            direct_provider=direct,
        )
        result.certificate.require()
        assert result.value == direct.derivative(order, point, directions[:order])
        assert result.raw_term_count == len(set_partitions(order))

    composed = ComposedDerivativeProvider(outer, inner)
    assert composed.value(point) == direct.value(point)
    assert composed.derivative(4, point, directions) == direct.derivative(4, point, directions)


def test_exact_packet_faa_di_bruno_reconstructs_projected_composition() -> None:
    sp, outer, inner, direct = symbolic_composition_providers()
    packets = coordinate_packets(exact=True)
    point = sp.Matrix([1, 2])
    directions = (sp.Matrix([2, 3]), sp.Matrix([-1, 4]), sp.Matrix([5, 2]))
    result = packet_faa_di_bruno(
        outer,
        inner,
        point,
        directions,
        packets,
        packets,
        packets,
        input_packets=("x", "y", "x"),
        output_packet="y",
    )
    result.certificate.require()

    projected_directions = (
        packets.projectors[0] * directions[0],
        packets.projectors[1] * directions[1],
        packets.projectors[0] * directions[2],
    )
    expected = packets.projectors[1] * direct.derivative(3, point, projected_directions)
    assert result.value == expected
    assert result.raw_term_count == labeled_composition_term_count(3, 2)


def test_faa_di_bruno_requires_all_lower_orders() -> None:
    inner = CallableDerivativeProvider(
        lambda x: np.asarray(x),
        {2: lambda x, directions: np.zeros(1)},
        source_dimension=1,
        target_dimension=1,
    )
    outer = CallableDerivativeProvider(
        lambda x: np.asarray(x),
        {2: lambda x, directions: np.zeros(1)},
        source_dimension=1,
        target_dimension=1,
    )
    with pytest.raises(ValidationError, match="through the requested order"):
        faa_di_bruno(outer, inner, np.array([0.0]), (np.array([1.0]), np.array([1.0])))
