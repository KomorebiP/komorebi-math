"""Episode 007 tests for convex/Finsler packets and ellipsoidal compression."""

from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.nonlinear.convex_packets import (
    joint_operator_bound,
    parallelotope_norm_packet,
    quadratic_operator_norm,
)


def test_parallelotope_packet_is_coordinate_covariant() -> None:
    packet = parallelotope_norm_packet(np.array([[2.0, 0.3], [0.0, 0.5]]))
    operator = np.array([[0.5, -0.2], [0.4, 0.25]])
    vector = np.array([0.7, -1.1])
    gauge = np.array([[3.0, 0.4], [0.2, 0.6]])

    transported = packet.transport_coordinates(gauge)
    transformed_operator = gauge @ operator @ np.linalg.inv(gauge)
    transformed_vector = gauge @ vector

    assert transported.vector_norm(transformed_vector) == pytest.approx(packet.vector_norm(vector))
    assert transported.operator_norm(transformed_operator) == pytest.approx(packet.operator_norm(operator))


def test_john_ellipsoid_compression_bound_for_parallelotope() -> None:
    packet = parallelotope_norm_packet(np.eye(4))
    operator = np.zeros((4, 4))
    operator[:, 0] = 1.0

    compression = packet.compress_operator(operator)
    assert compression.convex_operator_norm == pytest.approx(1.0)
    assert compression.john_quadratic_operator_norm == pytest.approx(2.0)
    assert compression.john_theorem_upper_bound == pytest.approx(2.0)
    assert compression.compression_ratio == pytest.approx(2.0)


def test_ando_shih_pair_is_exactly_nonexpansive_in_infinity_norm() -> None:
    first = np.array([[1.0, 0.0], [1.0, 0.0]])
    second = np.array([[0.0, 1.0], [0.0, -1.0]])
    packet = parallelotope_norm_packet(np.eye(2))

    assert packet.operator_norm(first) == pytest.approx(1.0)
    assert packet.operator_norm(second) == pytest.approx(1.0)
    assert joint_operator_bound(packet, [first, second]) == pytest.approx(1.0)

    # The John ellipsoid is the Euclidean disk.  Both operators saturate the
    # sqrt(2) convex-to-quadratic compression loss.
    assert packet.john_operator_norm(first) == pytest.approx(np.sqrt(2.0))
    assert packet.john_operator_norm(second) == pytest.approx(np.sqrt(2.0))


def test_ando_shih_quadratic_lower_bound_identity() -> None:
    # For Q=[[a,c],[c,b]], positivity means ab-c^2>0.  If both induced
    # quadratic norms were < sqrt(2), adding the two strict inequalities would
    # force (a-b-c)^2 + 3c^2 < 0, impossible.  Here we freeze the algebraic
    # identity numerically over an adversarial collection of positive metrics.
    first = np.array([[1.0, 0.0], [1.0, 0.0]])
    second = np.array([[0.0, 1.0], [0.0, -1.0]])
    rng = np.random.default_rng(20260922)
    observed = []
    for _ in range(400):
        raw = rng.normal(size=(2, 2))
        q = raw.T @ raw + 1e-3 * np.eye(2)
        observed.append(max(quadratic_operator_norm(first, q), quadratic_operator_norm(second, q)))
    assert min(observed) >= np.sqrt(2.0) * (1.0 - 2e-12)
    assert max(quadratic_operator_norm(first, np.eye(2)), quadratic_operator_norm(second, np.eye(2))) == pytest.approx(np.sqrt(2.0))


def test_unipotent_parallelotope_escape_approaches_one_without_attainment() -> None:
    holonomy = np.array([[1.0, 1.0], [0.0, 1.0]])
    values = []
    conditions = []
    for scale in (1e-1, 1e-2, 1e-3, 1e-4):
        packet = parallelotope_norm_packet(np.diag([scale, 1.0]), tolerance=1e-14)
        values.append(packet.operator_norm(holonomy))
        conditions.append(packet.coordinate_condition_number)
    assert values == pytest.approx([1.1, 1.01, 1.001, 1.0001])
    assert values[-1] < values[0]
    assert conditions[-1] > conditions[0]
    assert values[-1] > 1.0
