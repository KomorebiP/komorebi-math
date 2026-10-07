from __future__ import annotations

from fractions import Fraction
from math import comb, cos, factorial

import numpy as np
import pytest

from komorebi_math.analytic import (
    CyclicSectionFamily,
    cyclic_packets,
    cyclic_product_packets,
    cyclic_project_coefficients,
    cyclic_project_value,
    cyclic_projection_certificate,
    cyclic_reconstruct,
    denominator,
    denominator_root_sum,
    logarithmic_coefficients,
    reciprocal_coefficients,
    reciprocal_polynomial,
    reciprocal_semigroup_certificate,
    section_certificate,
    signed_section,
    signed_shift_matrix,
    transfer_section,
    transfer_tangent,
)
from komorebi_math.exceptions import ValidationError


def test_cyclic_projection_and_reconstruction() -> None:
    coefficients = np.arange(17)
    packets = cyclic_packets(coefficients, 5)
    assert np.array_equal(cyclic_reconstruct(packets), coefficients)
    assert np.array_equal(cyclic_project_coefficients(coefficients, 5, 3), packets[3])
    cyclic_projection_certificate(coefficients, 5).certificate.require()


def test_root_average_matches_direct_polynomial_packet() -> None:
    coefficients = [1.0, -2.0, 3.0, 0.5, -0.25, 2.0]

    def polynomial(z: complex) -> complex:
        return sum(value * z**index for index, value in enumerate(coefficients))

    z = 0.2 + 0.1j
    for residue in range(3):
        direct = sum(value * z**index for index, value in enumerate(coefficients) if index % 3 == residue)
        assert abs(cyclic_project_value(polynomial, z, 3, residue) - direct) < 1e-12


def test_cyclic_packet_product_matches_full_convolution() -> None:
    left = np.array([1, 2, 3, 4], dtype=float)
    right = np.array([2, -1, 5], dtype=float)
    result = cyclic_product_packets(cyclic_packets(left, 3), cyclic_packets(right, 3))
    assert np.allclose(sum(result), np.convolve(left, right))


def test_transfer_and_signed_sections() -> None:
    assert abs(transfer_section(2, 0, 0.4) - np.cosh(0.4)) < 1e-13
    assert abs(transfer_section(2, 1, 0.4) - np.sinh(0.4)) < 1e-13
    assert abs(signed_section(2, 0, 0.4) - cos(0.4)) < 1e-13
    assert abs(denominator(5, 0.3 + 0.1j) - denominator_root_sum(5, 0.3 + 0.1j)) < 1e-12
    section_certificate(4, 0.2 - 0.1j, family="transfer").certificate.require()
    section_certificate(4, 0.2 - 0.1j, family="signed").certificate.require()


def test_signed_shift_matrix_power_and_fundamental_matrix() -> None:
    for order in range(1, 7):
        shift = signed_shift_matrix(order)
        assert np.allclose(np.linalg.matrix_power(shift, order), -np.eye(order))
    family = CyclicSectionFamily(4, "signed")
    matrix = family.fundamental_matrix(0.2)
    assert np.allclose(matrix[:, 0], np.asarray(family.vector(0.2)), atol=1e-12)


def test_reciprocal_coefficients_and_semigroup() -> None:
    assert reciprocal_coefficients(2, 1, 5) == (
        Fraction(1),
        Fraction(1),
        Fraction(5),
        Fraction(61),
        Fraction(1385),
        Fraction(50521),
    )
    reciprocal_semigroup_certificate(4, Fraction(2, 3), Fraction(5, 4), 6).certificate.require()


def test_reciprocal_polynomial_sympy() -> None:
    sympy = pytest.importorskip("sympy")
    r = sympy.Symbol("r")
    polynomial = reciprocal_polynomial(3, 2)
    expected = r * (comb(6, 3) * (r + 1) / 2 - 1)
    assert sympy.simplify(polynomial - expected) == 0


def test_logarithmic_coefficients_reconstruct_tangent() -> None:
    kappa = logarithmic_coefficients(3, 5)
    z = 0.1
    approximation = sum(3 * kappa[index + 1] * z ** (3 * index + 2) for index in range(5))
    assert abs(complex(approximation) - transfer_tangent(3, z)) < 1e-10


def test_invalid_cyclic_inputs() -> None:
    with pytest.raises(ValidationError):
        cyclic_project_coefficients([1, 2], 0, 0)
    with pytest.raises(ValidationError):
        cyclic_project_coefficients([1, 2], 2, 2)
    with pytest.raises(ValidationError):
        cyclic_reconstruct([])
