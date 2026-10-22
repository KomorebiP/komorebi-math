"""Metric distortion packets for finite-dimensional transport maps.

This module is intentionally not re-exported from the aggregate nonlinear
namespace. The API is an experimental research surface.

For an invertible map ``T : (V,Q_source) -> (W,Q_target)`` define the metric
representative

    B = R_target T R_source^{-1},   Q = R^* R.

The anisotropy packet is the singular-value condition number of ``B``.  It is
coordinate-covariant when the metrics are transported with the coordinates,
and it is submultiplicative under composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
from typing import Any

import numpy as np

from ..exceptions import DimensionError, ValidationError


def _matrix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=complex)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise DimensionError(f"{name} must be a nonempty matrix")
    if not np.all(np.isfinite(array)):
        raise ValidationError(f"{name} must contain finite values")
    return array


def _positive_metric(value: Any, *, dimension: int, name: str, tolerance: float) -> np.ndarray:
    metric = _matrix(value, name=name)
    if metric.shape != (dimension, dimension):
        raise DimensionError(f"{name} has the wrong dimension")
    scale = max(1.0, float(np.linalg.norm(metric, ord=2)))
    hermitian_error = float(np.linalg.norm(metric - metric.conj().T, ord="fro"))
    if hermitian_error > tolerance * scale:
        raise ValidationError(f"{name} must be Hermitian within tolerance")
    hermitian = 0.5 * (metric + metric.conj().T)
    eigenvalues = np.linalg.eigvalsh(hermitian)
    if float(eigenvalues[0]) <= tolerance * scale:
        raise ValidationError(f"{name} must be positive definite and numerically resolved")
    return hermitian


def _metric_factor(metric: np.ndarray) -> np.ndarray:
    """Return R with metric = R^* R using a Cholesky factor."""

    lower = np.linalg.cholesky(metric)
    return lower.conj().T


@dataclass(frozen=True, slots=True)
class MetricDistortionPacket:
    """Scale/anisotropy decomposition of one invertible transport map."""

    dimension: int
    volume_scale: float
    distortion: float
    log_distortion: float
    smallest_metric_singular_value: float
    largest_metric_singular_value: float

    @property
    def conformal(self) -> bool:
        return bool(np.isclose(self.distortion, 1.0, rtol=1e-10, atol=1e-12))


def metric_representative(
    matrix: Any,
    source_metric: Any,
    target_metric: Any,
    *,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Represent a transport between metric coordinates in Euclidean form.

    If ``Q_s = R_s^* R_s`` and ``Q_t = R_t^* R_t``, the returned matrix is
    ``R_t T R_s^{-1}``.  Its singular values are intrinsic to the metric map:
    paired coordinate changes of ``T``, ``Q_s``, and ``Q_t`` alter the matrix
    only by unitary factors and therefore preserve those singular values.
    """

    transport = _matrix(matrix, name="transport")
    rows, columns = transport.shape
    if rows != columns:
        raise DimensionError("metric distortion currently requires a square invertible transport")
    source = _positive_metric(
        source_metric, dimension=columns, name="source metric", tolerance=tolerance
    )
    target = _positive_metric(
        target_metric, dimension=rows, name="target metric", tolerance=tolerance
    )
    if abs(np.linalg.det(transport)) <= tolerance:
        raise ValidationError("transport must be invertible")
    source_factor = _metric_factor(source)
    target_factor = _metric_factor(target)
    try:
        representative = target_factor @ transport @ np.linalg.inv(source_factor)
    except np.linalg.LinAlgError as exc:
        raise ValidationError("metric factorization produced a singular factor") from exc
    return representative


def metric_distortion_packet(
    matrix: Any,
    source_metric: Any,
    target_metric: Any,
    *,
    tolerance: float = 1e-12,
) -> MetricDistortionPacket:
    """Return intrinsic scale and anisotropy data for one metric transport."""

    representative = metric_representative(
        matrix,
        source_metric,
        target_metric,
        tolerance=tolerance,
    )
    singular_values = np.linalg.svd(representative, compute_uv=False)
    smallest = float(singular_values[-1])
    largest = float(singular_values[0])
    if smallest <= tolerance * max(1.0, largest):
        raise ValidationError("transport is numerically singular in the supplied metrics")
    distortion = largest / smallest
    dimension = representative.shape[0]
    determinant = abs(complex(np.linalg.det(representative)))
    volume_scale = float(determinant ** (1.0 / dimension))
    return MetricDistortionPacket(
        dimension=dimension,
        volume_scale=volume_scale,
        distortion=float(distortion),
        log_distortion=float(log(distortion)),
        smallest_metric_singular_value=smallest,
        largest_metric_singular_value=largest,
    )


def spectral_distortion_lower_bound(matrix: Any, *, tolerance: float = 1e-12) -> float:
    """Conjugacy-invariant lower bound for metric anisotropy of a closed word.

    For every positive-definite metric Q,

        cond_2(Q^{1/2} H Q^{-1/2}) >= max |lambda(H)| / min |lambda(H)|.

    The bound can be 1 even when exact conformal completion is impossible
    (for example, a nontrivial unipotent Jordan block).
    """

    holonomy = _matrix(matrix, name="holonomy")
    if holonomy.shape[0] != holonomy.shape[1]:
        raise DimensionError("holonomy must be square")
    eigenvalues = np.linalg.eigvals(holonomy)
    moduli = np.abs(eigenvalues)
    largest = float(np.max(moduli))
    smallest = float(np.min(moduli))
    if smallest <= tolerance * max(1.0, largest):
        return float("inf")
    return largest / smallest


def determinant_normalized(matrix: Any, *, tolerance: float = 1e-12) -> np.ndarray:
    """Remove positive volume scale from an invertible complex matrix.

    The positive scalar ``|det(T)|^(1/n)`` is removed.  The remaining matrix
    has determinant modulus one and carries only anisotropy/phase information.
    """

    transport = _matrix(matrix, name="transport")
    if transport.shape[0] != transport.shape[1]:
        raise DimensionError("transport must be square")
    determinant = abs(complex(np.linalg.det(transport)))
    if determinant <= tolerance:
        raise ValidationError("transport must be invertible")
    scale = determinant ** (1.0 / transport.shape[0])
    result = np.array(transport / scale, copy=True)
    result.setflags(write=False)
    return result


def compose_distortion_bound(*distortions: float) -> float:
    """Conservative anisotropy bound for a composed transport path."""

    total = 1.0
    for value in distortions:
        current = float(value)
        if not isfinite(current) or current < 1.0:
            raise ValidationError("distortion factors must be finite and at least one")
        total *= current
    return total


__all__ = [
    "MetricDistortionPacket",
    "compose_distortion_bound",
    "determinant_normalized",
    "metric_distortion_packet",
    "metric_representative",
    "spectral_distortion_lower_bound",
]
