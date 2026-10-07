"""Cyclic projection and residue packet operations for analytic series."""

from __future__ import annotations

import cmath
from collections.abc import Callable, Sequence
from math import tau
from typing import Any

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._series import validate_order


def _residue(order: int, residue: int) -> tuple[int, int]:
    n = validate_order(order)
    j = int(residue)
    if isinstance(residue, bool) or j != residue or not 0 <= j < n:
        raise ValidationError("residue must lie between zero and order minus one")
    return n, j


@documented(
    ProvenanceRecord(
        identifier="PSC-IV-T01",
        title="Cyclic coefficient projection",
        status=ClaimStatus.PROVED,
        source="Paper IV",
        assumptions=("finite or convergent coefficient sequence",),
        verification_suite="analytic.cyclic",
    )
)
def cyclic_project_coefficients(
    coefficients: Sequence[Any],
    order: int,
    residue: int,
) -> np.ndarray:
    """Keep coefficients whose indices equal one residue modulo ``order``."""

    n, j = _residue(order, residue)
    values = np.asarray(tuple(coefficients))
    if values.ndim != 1:
        raise ValidationError("coefficients must be one dimensional")
    result = np.zeros_like(values)
    result[j::n] = values[j::n]
    return result


def cyclic_packets(coefficients: Sequence[Any], order: int) -> tuple[np.ndarray, ...]:
    """Split one coefficient vector into all cyclic residue packets."""

    n = validate_order(order)
    return tuple(cyclic_project_coefficients(coefficients, n, residue) for residue in range(n))


def cyclic_reconstruct(packets: Sequence[Sequence[Any]]) -> np.ndarray:
    """Reconstruct a coefficient vector from equally shaped residue packets."""

    arrays = tuple(np.asarray(packet) for packet in packets)
    if not arrays:
        raise ValidationError("at least one packet is required")
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValidationError("all packets must have one common shape")
    return np.sum(np.stack(arrays, axis=0), axis=0)


def cyclic_project_value(
    function: Callable[[complex], complex],
    z: complex,
    order: int,
    residue: int,
) -> complex:
    """Evaluate the root of unity cyclic projection of an analytic function."""

    n, j = _residue(order, residue)
    omega = cmath.exp(1j * tau / n)
    return sum(omega ** (-j * ell) * function(omega**ell * z) for ell in range(n)) / n


def cyclic_product_packets(
    left: Sequence[Sequence[complex]],
    right: Sequence[Sequence[complex]],
) -> tuple[np.ndarray, ...]:
    """Multiply finite residue packet coefficient arrays by cyclic convolution."""

    lhs = tuple(np.asarray(packet) for packet in left)
    rhs = tuple(np.asarray(packet) for packet in right)
    if not lhs or len(lhs) != len(rhs):
        raise ValidationError("left and right packet families must have one common nonzero order")
    if any(array.ndim != 1 for array in (*lhs, *rhs)):
        raise ValidationError("packet coefficient arrays must be one dimensional")
    order = len(lhs)
    output_length = max(len(array) for array in lhs) + max(len(array) for array in rhs) - 1
    result = [np.zeros(output_length, dtype=np.result_type(*lhs, *rhs, complex)) for _ in range(order)]
    for a, left_packet in enumerate(lhs):
        for b, right_packet in enumerate(rhs):
            result[(a + b) % order][: len(left_packet) + len(right_packet) - 1] += np.convolve(
                left_packet,
                right_packet,
            )
    return tuple(result)


def cyclic_projection_certificate(
    coefficients: Sequence[complex],
    order: int,
    *,
    tolerance: float = 1e-12,
) -> ComputationResult[tuple[np.ndarray, ...]]:
    """Return all residue packets with an exact finite reconstruction check."""

    values = np.asarray(tuple(coefficients))
    packets = cyclic_packets(values, order)
    reconstructed = cyclic_reconstruct(packets)
    residual = float(np.linalg.norm(reconstructed - values))
    return ComputationResult(
        packets,
        Certificate(
            valid=residual <= tolerance,
            exactness=Exactness.EXACT,
            residuals={"reconstruction": residual},
            tolerance=tolerance,
            assumptions_checked=("finite coefficient vector", "complete residue family"),
            theorem_id="PSC-IV-T01",
            message="cyclic coefficient packets reconstruct exactly",
        ),
    )


__all__ = [
    "cyclic_packets",
    "cyclic_product_packets",
    "cyclic_project_coefficients",
    "cyclic_project_value",
    "cyclic_projection_certificate",
    "cyclic_reconstruct",
]
