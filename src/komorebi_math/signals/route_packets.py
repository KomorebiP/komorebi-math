"""Research-tier additive route packets for finite polyphase filters.

Block 14 tests the continuation-state packet idea outside max/convex semantics.
For linear time-invariant signal routing, alternative histories contribute by
*addition* rather than by an idempotent maximum.  A causal FIR filter therefore
induces an ``order x order`` matrix of coarse-lag kernels.  Cascading filters is
matrix multiplication in which scalar multiplication is convolution and scalar
addition is coefficientwise addition.

This module is intentionally not re-exported from the stable aggregate API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..exceptions import DimensionError, ValidationError
from .polyphase import polyphase_filter_kernels


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValidationError(f"{name} must be a positive integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{name} must be a positive integer") from exc
    if normalized != value or normalized <= 0:
        raise ValidationError(f"{name} must be a positive integer")
    return normalized


def _vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.size == 0:
        raise DimensionError(f"{name} must be a nonempty one dimensional array")
    if array.dtype.kind in "biu":
        # Casting integer arrays to object materializes Python integers, so
        # intermediate convolution products cannot wrap in fixed-width dtypes.
        array = array.astype(object)
    elif array.dtype == object:
        array = np.asarray(
            [int(item) if isinstance(item, (np.integer, np.bool_)) else item for item in array],
            dtype=object,
        )
    return array


def _convolve(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Convolution that preserves exact object/integer arithmetic."""

    dtype = np.result_type(left.dtype, right.dtype)
    if left.dtype == object or right.dtype == object:
        dtype = np.dtype(object)
    result = np.zeros(left.size + right.size - 1, dtype=dtype)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] += a * b
    return result


def _add(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    dtype = np.result_type(left.dtype, right.dtype)
    if left.dtype == object or right.dtype == object:
        dtype = np.dtype(object)
    length = max(left.size, right.size)
    result = np.zeros(length, dtype=dtype)
    result[: left.size] += left
    result[: right.size] += right
    return result


def _freeze(array: np.ndarray) -> np.ndarray:
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class PolyphaseRoutePacket:
    """Additive continuation-state packet for one causal FIR transfer.

    ``kernels[target][source][q]`` is the coefficient multiplying source
    residue ``source`` at coarse lag ``q`` in target residue ``target``.
    Intermediate tap/phase histories have already been summed.
    """

    kernels: tuple[tuple[np.ndarray, ...], ...]
    order: int
    kernel_length: int

    def __post_init__(self) -> None:
        order = _positive_int(self.order, "order")
        length = _positive_int(self.kernel_length, "kernel_length")
        if len(self.kernels) != order or any(len(row) != order for row in self.kernels):
            raise DimensionError("kernels must form an order by order matrix")
        frozen_rows: list[tuple[np.ndarray, ...]] = []
        for row in self.kernels:
            frozen_row: list[np.ndarray] = []
            for coefficients in row:
                array = _vector(coefficients, "polyphase route kernel")
                frozen_row.append(_freeze(array))
            frozen_rows.append(tuple(frozen_row))
        object.__setattr__(self, "kernels", tuple(frozen_rows))
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "kernel_length", length)

    @classmethod
    def from_filter(cls, kernel: Any, order: int) -> "PolyphaseRoutePacket":
        """Compile one FIR filter into its additive polyphase route packet."""

        h = _vector(kernel, "kernel")
        packet_order = _positive_int(order, "order")
        rows = polyphase_filter_kernels(h, packet_order)
        kernels = tuple(
            tuple(np.asarray(entry.coefficients) for entry in row)
            for row in rows
        )
        return cls(kernels=kernels, order=packet_order, kernel_length=int(h.size))

    def coefficient(self, target: int, source: int, coarse_lag: int) -> Any:
        target_i = int(target)
        source_i = int(source)
        lag = int(coarse_lag)
        if not 0 <= target_i < self.order or not 0 <= source_i < self.order:
            raise ValidationError("source and target must lie in the packet residue range")
        if lag < 0:
            raise ValidationError("coarse_lag must be nonnegative")
        coefficients = self.kernels[target_i][source_i]
        return coefficients[lag] if lag < coefficients.size else 0

    def compose(self, next_packet: "PolyphaseRoutePacket") -> "PolyphaseRoutePacket":
        """Cascade ``self`` followed by ``next_packet`` exactly.

        The update is the additive analogue of an observer-conditioned Bellman
        packet:

            C[target, source] = sum_mid next[target, mid] * self[mid, source],

        where ``*`` is convolution of coarse-lag sequences.
        """

        if not isinstance(next_packet, PolyphaseRoutePacket):
            raise ValidationError("next_packet must be a PolyphaseRoutePacket")
        if self.order != next_packet.order:
            raise DimensionError("polyphase route orders must match")

        rows: list[tuple[np.ndarray, ...]] = []
        for target in range(self.order):
            row: list[np.ndarray] = []
            for source in range(self.order):
                total = np.zeros(1, dtype=np.result_type(
                    self.kernels[0][0].dtype,
                    next_packet.kernels[0][0].dtype,
                ))
                if self.kernels[0][0].dtype == object or next_packet.kernels[0][0].dtype == object:
                    total = np.zeros(1, dtype=object)
                for middle in range(self.order):
                    contribution = _convolve(
                        next_packet.kernels[target][middle],
                        self.kernels[middle][source],
                    )
                    total = _add(total, contribution)
                row.append(total)
            rows.append(tuple(row))

        return PolyphaseRoutePacket(
            kernels=tuple(rows),
            order=self.order,
            kernel_length=self.kernel_length + next_packet.kernel_length - 1,
        )

    def reconstruct_filter(self) -> np.ndarray:
        """Recover the represented scalar FIR kernel from source residue zero."""

        dtype = np.result_type(*(self.kernels[target][0].dtype for target in range(self.order)))
        if any(self.kernels[target][0].dtype == object for target in range(self.order)):
            dtype = np.dtype(object)
        result = np.zeros(self.kernel_length, dtype=dtype)
        for lag in range(self.kernel_length):
            target = lag % self.order
            coarse_lag = lag // self.order
            result[lag] = self.coefficient(target, 0, coarse_lag)
        return result

    def nonzero_coefficient_count(self) -> int:
        return sum(
            int(np.count_nonzero(np.asarray(coefficients, dtype=object)))
            for row in self.kernels
            for coefficients in row
        )


def polyphase_route_packet(kernel: Any, order: int) -> PolyphaseRoutePacket:
    """Compile a causal FIR filter into a research-tier route packet."""

    return PolyphaseRoutePacket.from_filter(kernel, order)


__all__ = ["PolyphaseRoutePacket", "polyphase_route_packet"]
