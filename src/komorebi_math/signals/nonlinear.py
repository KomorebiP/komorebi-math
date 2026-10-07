"""Nonlinear residue packet identities for finite signal sequences."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented


def _positive_order(order: int) -> int:
    if isinstance(order, bool) or int(order) != order or order < 1:
        raise ValidationError("order must be a positive integer")
    return int(order)


def coefficient_packets(coefficients: Sequence[complex], order: int) -> tuple[np.ndarray, ...]:
    """Split a finite coefficient sequence into residue supported packets."""

    packet_order = _positive_order(order)
    values = np.asarray(coefficients)
    if values.ndim != 1:
        raise DimensionError("coefficient packets require a one dimensional sequence")
    indices = np.arange(values.size)
    return tuple(np.where(indices % packet_order == residue, values, 0) for residue in range(packet_order))


def _as_packet_channels(values: Any, order: int, name: str) -> tuple[np.ndarray, ...] | None:
    """Return explicit packet channels when ``values`` is a nested sequence."""

    if isinstance(values, np.ndarray) and values.ndim == 1:
        return None
    try:
        items = tuple(values)
    except TypeError:
        return None
    if len(items) != order:
        return None
    channels: list[np.ndarray] = []
    for index, item in enumerate(items):
        channel = np.asarray(item)
        if channel.ndim != 1:
            raise DimensionError(f"{name} packet {index} must be one dimensional")
        if not channel.size:
            raise DimensionError(f"{name} packet {index} must be nonempty")
        channels.append(channel)
    return tuple(channels)


@documented(
    ProvenanceRecord(
        identifier="PSC-V-T51",
        title="Cyclic product packet law",
        status=ClaimStatus.PROVED,
        source="Paper V",
        assumptions=("finite coefficient sequences or explicit residue channels",),
        verification_suite="signals.nonlinear",
    )
)
def packet_product(a: Any, b: Any, order: int) -> ComputationResult[tuple[np.ndarray, ...]]:
    """Multiply finite sequences or already separated cyclic packet channels.

    Flat inputs are first split into residue supported coefficient packets. Nested
    inputs of length ``order`` are interpreted as compact packet channels and are
    combined by cyclic channel convolution. The returned value always contains
    one output channel for every residue address.
    """

    packet_order = _positive_order(order)
    explicit_a = _as_packet_channels(a, packet_order, "left")
    explicit_b = _as_packet_channels(b, packet_order, "right")

    if (explicit_a is None) != (explicit_b is None):
        raise ValidationError("both packet product inputs must use the same representation")

    if explicit_a is not None and explicit_b is not None:
        dtype = np.result_type(*(channel.dtype for channel in (*explicit_a, *explicit_b)), complex)
        output_lengths = [0] * packet_order
        terms: list[tuple[int, int, np.ndarray]] = []
        for left in range(packet_order):
            for right in range(packet_order):
                target = (left + right) % packet_order
                carry = (left + right) // packet_order
                term = np.convolve(explicit_a[left], explicit_b[right])
                terms.append((target, carry, term))
                output_lengths[target] = max(output_lengths[target], carry + term.size)
        output = [np.zeros(length, dtype=dtype) for length in output_lengths]
        for target, carry, term in terms:
            output[target][carry : carry + term.size] += term
        result = tuple(np.real_if_close(channel, tol=1000) for channel in output)

        def reconstruct(channels: tuple[np.ndarray, ...]) -> np.ndarray:
            length = max(
                (packet_order * (channel.size - 1) + residue + 1 for residue, channel in enumerate(channels)),
                default=0,
            )
            full = np.zeros(length, dtype=np.result_type(*(channel.dtype for channel in channels), complex))
            for residue, channel in enumerate(channels):
                full[residue : residue + packet_order * channel.size : packet_order] = channel
            return full

        direct = np.convolve(reconstruct(explicit_a), reconstruct(explicit_b))
        reconstructed = reconstruct(result)[: direct.size]
        residual = float(np.max(np.abs(reconstructed - direct))) if direct.size else 0.0
        valid = residual <= 1e-10
        return ComputationResult(
            result,
            Certificate(
                valid=valid,
                exactness=Exactness.NUMERICAL,
                residuals={"cyclic_channel_law": residual},
                tolerance=1e-10,
                assumptions_checked=("explicit compact residue channels", "radix carry propagation"),
                theorem_id="PSC-V-T51",
                message="cyclic packet channel product verified" if valid else "cyclic packet channel product failed",
            ),
        )

    first = np.asarray(a)
    second = np.asarray(b)
    if first.ndim != 1 or second.ndim != 1 or not first.size or not second.size:
        raise DimensionError("packet products require nonempty one dimensional coefficient sequences")
    first_packets = coefficient_packets(first, packet_order)
    second_packets = coefficient_packets(second, packet_order)
    length = first.size + second.size - 1
    output = [np.zeros(length, dtype=np.result_type(first, second, complex)) for _ in range(packet_order)]
    for left in range(packet_order):
        for right in range(packet_order):
            output[(left + right) % packet_order] += np.convolve(first_packets[left], second_packets[right])
    result = tuple(np.real_if_close(packet, tol=1000) for packet in output)
    direct = np.convolve(first, second)
    reconstructed = sum(result, np.zeros(length, dtype=np.result_type(*result)))
    residual = float(np.max(np.abs(reconstructed - direct))) if direct.size else 0.0
    valid = residual <= 1e-10
    return ComputationResult(
        result,
        Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"product_reconstruction": residual},
            tolerance=1e-10,
            assumptions_checked=("finite Cauchy product",),
            theorem_id="PSC-V-T51",
            message="cyclic packet product verified" if valid else "cyclic packet product failed",
        ),
    )


def nonlinear_shift_filter(
    signal: Sequence[float],
    alpha: float = 0.25,
    *,
    shift: int = 1,
    shifts: Sequence[int] | None = None,
    coefficients: Sequence[float] | None = None,
) -> np.ndarray:
    """Apply a polynomial shift interaction to a one dimensional signal.

    The legacy form is ``x + alpha * x * roll(x, shift)``. A generalized form
    accepts matching ``shifts`` and ``coefficients`` and uses their weighted sum
    as the second nonlinear factor.
    """

    values = np.asarray(signal, dtype=float)
    if values.ndim != 1:
        raise DimensionError("the nonlinear shift filter requires a one dimensional signal")
    if shifts is None and coefficients is None:
        interaction = np.roll(values, int(shift))
    else:
        if shifts is None or coefficients is None:
            raise ValidationError("shifts and coefficients must be supplied together")
        shift_values = tuple(int(value) for value in shifts)
        coefficient_values = tuple(float(value) for value in coefficients)
        if not shift_values or len(shift_values) != len(coefficient_values):
            raise ValidationError("shifts and coefficients must be nonempty and have the same length")
        interaction = np.zeros_like(values)
        for offset, coefficient in zip(shift_values, coefficient_values, strict=True):
            interaction += coefficient * np.roll(values, offset)
    return values + float(alpha) * values * interaction


__all__ = ["coefficient_packets", "nonlinear_shift_filter", "packet_product"]
