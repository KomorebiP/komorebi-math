"""Episode 006 tests for the metric-completion/distortion-cocycle calculus."""

from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.nonlinear.distortion import (
    compose_distortion_bound,
    determinant_normalized,
    metric_distortion_packet,
    spectral_distortion_lower_bound,
)


def _transport_metric(metric: np.ndarray, gauge: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(gauge)
    return inverse.conj().T @ metric @ inverse


def test_metric_distortion_is_covariant_under_paired_coordinate_changes() -> None:
    transport = np.array([[2.0, 0.5], [0.25, 1.25]])
    source_metric = np.array([[2.0, 0.2], [0.2, 1.0]])
    target_metric = np.array([[1.5, -0.1], [-0.1, 3.0]])
    original = metric_distortion_packet(transport, source_metric, target_metric)

    source_gauge = np.array([[1e3, 2.0], [0.0, 1e-2]])
    target_gauge = np.array([[3e-2, -1.0], [0.0, 2e2]])
    transformed_transport = target_gauge @ transport @ np.linalg.inv(source_gauge)
    transformed = metric_distortion_packet(
        transformed_transport,
        _transport_metric(source_metric, source_gauge),
        _transport_metric(target_metric, target_gauge),
        tolerance=1e-14,
    )

    assert transformed.distortion == pytest.approx(original.distortion, rel=2e-8, abs=2e-10)
    assert transformed.volume_scale == pytest.approx(original.volume_scale, rel=2e-8, abs=2e-10)


def test_log_distortion_is_subadditive_under_composition() -> None:
    q0 = np.array([[2.0, 0.1], [0.1, 1.0]])
    q1 = np.array([[1.0, 0.2], [0.2, 2.0]])
    q2 = np.array([[3.0, -0.3], [-0.3, 1.0]])
    first = np.array([[1.5, 0.4], [0.2, 0.8]])
    second = np.array([[0.9, -0.25], [0.35, 1.4]])

    d1 = metric_distortion_packet(first, q0, q1).distortion
    d2 = metric_distortion_packet(second, q1, q2).distortion
    composed = metric_distortion_packet(second @ first, q0, q2).distortion

    assert composed <= compose_distortion_bound(d2, d1) * (1.0 + 2e-12)


def test_joint_word_obstruction_has_exact_minimax_example() -> None:
    t = 9.0
    a = np.array([[0.0, -1.0], [1.0, 0.0]])
    b = np.array([[0.0, -t], [1.0 / t, 0.0]])
    product = a @ b

    # Each generator is individually unitarizable, but the mixed word AB is
    # hyperbolic.  Its eigenvalue-modulus spread is t^2.
    assert spectral_distortion_lower_bound(a) == pytest.approx(1.0)
    assert spectral_distortion_lower_bound(b) == pytest.approx(1.0)
    assert spectral_distortion_lower_bound(product) == pytest.approx(t * t)

    # Q with coordinate ratio 1/t attains the minimax lower bound t on both
    # generators, so inf_Q max(D_Q(A), D_Q(B)) = t.
    q = np.diag([t ** -0.5, t ** 0.5])
    da = metric_distortion_packet(a, q, q).distortion
    db = metric_distortion_packet(b, q, q).distortion
    dab = metric_distortion_packet(product, q, q).distortion
    assert da == pytest.approx(t, rel=2e-12)
    assert db == pytest.approx(t, rel=2e-12)
    assert dab == pytest.approx(t * t, rel=2e-12)
    assert max(da, db) == pytest.approx(t, rel=2e-12)


def test_unipotent_has_zero_spectral_gap_but_nonattained_unit_distortion() -> None:
    holonomy = np.array([[1.0, 1.0], [0.0, 1.0]])
    assert spectral_distortion_lower_bound(holonomy) == pytest.approx(1.0)

    distortions = []
    for scale in (1e-1, 1e-2, 1e-3, 1e-4):
        metric = np.diag([scale * scale, 1.0])
        distortion = metric_distortion_packet(holonomy, metric, metric, tolerance=1e-14).distortion
        distortions.append(distortion)
        assert distortion > 1.0
    assert distortions[-1] < distortions[0]
    assert distortions[-1] < 1.001


def test_determinant_normalization_removes_only_volume_scale() -> None:
    matrix = np.array([[6.0, 0.0], [0.0, 1.5]])
    normalized = determinant_normalized(matrix)
    assert abs(np.linalg.det(normalized)) == pytest.approx(1.0)
    packet_original = metric_distortion_packet(matrix, np.eye(2), np.eye(2))
    packet_normalized = metric_distortion_packet(normalized, np.eye(2), np.eye(2))
    assert packet_original.distortion == pytest.approx(packet_normalized.distortion)
    assert packet_normalized.volume_scale == pytest.approx(1.0)
