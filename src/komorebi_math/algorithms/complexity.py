"""Declared arithmetic, storage, data movement, and roofline cost models."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Mapping

from ..exceptions import ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._validation import strict_integer
from .layout import PacketLayout

Address = tuple[int, int]


def _checked_addresses(
    source_layout: PacketLayout,
    target_layout: PacketLayout,
    support: Iterable[Address],
) -> frozenset[Address]:
    checked: set[Address] = set()
    for address in support:
        try:
            target, source = address
        except (TypeError, ValueError) as exc:
            raise ValidationError("support addresses must be target and source pairs") from exc
        target_index = strict_integer(target, "target packet index", minimum=0)
        source_index = strict_integer(source, "source packet index", minimum=0)
        if not 0 <= target_index < len(target_layout) or not 0 <= source_index < len(source_layout):
            raise ValidationError("support address is outside the packet layouts")
        checked.add((target_index, source_index))
    return frozenset(checked)


def dense_apply_flops(rows: int, columns: int | None = None, rhs: int = 1) -> int:
    """Return the classical dense matrix application operation count."""

    row_count = strict_integer(rows, "rows", minimum=1)
    column_count = row_count if columns is None else strict_integer(columns, "columns", minimum=1)
    rhs_count = strict_integer(rhs, "rhs", minimum=1)
    return 2 * row_count * column_count * rhs_count


@documented(
    ProvenanceRecord(
        identifier="PSC-VIII-T04",
        title="Packet apply arithmetic count",
        status=ClaimStatus.PROVED,
        source="Paper VIII",
        assumptions=("dense active packet blocks", "classical matrix multiplication count"),
        verification_suite="algorithms.complexity",
    )
)
def packet_apply_flops(
    source_layout: PacketLayout,
    target_layout: PacketLayout,
    support: Iterable[Address],
    rhs: int = 1,
) -> int:
    """Return the dense active block operation count."""

    rhs_count = strict_integer(rhs, "rhs", minimum=1)
    total = 0
    for target, source in _checked_addresses(source_layout, target_layout, support):
        total += 2 * target_layout.sizes[target] * source_layout.sizes[source] * rhs_count
    return int(total)


def packet_dense_storage_entries(
    source_layout: PacketLayout,
    target_layout: PacketLayout,
    support: Iterable[Address],
) -> int:
    """Count scalar entries stored by dense active packet blocks."""

    checked = _checked_addresses(source_layout, target_layout, support)
    return int(
        sum(
            target_layout.sizes[target] * source_layout.sizes[source]
            for target, source in checked
        )
    )


def block_diagonal_low_rank_storage_entries(layout: PacketLayout, rank: int) -> int:
    """Count entries in a dense block diagonal plus rank bounded representation."""

    rank_value = strict_integer(rank, "rank", minimum=0)
    return int(sum(size * size for size in layout.sizes) + 2 * layout.dimension * rank_value)


def block_diagonal_low_rank_flops(layout: PacketLayout, rank: int, rhs: int = 1) -> int:
    """Return the modeled application count for a square BDLR operator."""

    rank_value = strict_integer(rank, "rank", minimum=0)
    rhs_count = strict_integer(rhs, "rhs", minimum=1)
    block_flops = 2 * rhs_count * sum(size * size for size in layout.sizes)
    low_rank_flops = 4 * layout.dimension * rank_value * rhs_count
    return int(block_flops + low_rank_flops)


def packet_streaming_bytes(
    source_layout: PacketLayout,
    target_layout: PacketLayout,
    support: Iterable[Address],
    *,
    rhs: int = 1,
    itemsize: int = 8,
) -> int:
    """One pass upper model with no cache reuse assumption."""

    rhs_count = strict_integer(rhs, "rhs", minimum=1)
    itemsize_value = strict_integer(itemsize, "itemsize", minimum=1)
    scalar_count = 0
    for target, source in _checked_addresses(source_layout, target_layout, support):
        target_size = target_layout.sizes[target]
        source_size = source_layout.sizes[source]
        scalar_count += target_size * source_size
        scalar_count += source_size * rhs_count
        scalar_count += 2 * target_size * rhs_count
    return int(scalar_count * itemsize_value)


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """Return modeled arithmetic work per byte moved."""

    if not isfinite(float(flops)) or not isfinite(float(bytes_moved)):
        raise ValidationError("flops and bytes_moved must be finite")
    if flops < 0 or bytes_moved < 0:
        raise ValidationError("flops and bytes_moved must be nonnegative")
    if bytes_moved == 0:
        return 0.0 if flops == 0 else float("inf")
    return float(flops) / float(bytes_moved)


def roofline_time_lower_bound(
    flops: float,
    bytes_moved: float,
    peak_flops_per_second: float,
    bandwidth_bytes_per_second: float,
) -> float:
    """Return the elementary roofline lower bound for one operation."""

    values = (flops, bytes_moved, peak_flops_per_second, bandwidth_bytes_per_second)
    if any(not isfinite(float(value)) for value in values):
        raise ValidationError("roofline inputs must be finite")
    if min(flops, bytes_moved) < 0:
        raise ValidationError("work and data movement must be nonnegative")
    if peak_flops_per_second <= 0 or bandwidth_bytes_per_second <= 0:
        raise ValidationError("device rates must be positive")
    return max(
        float(flops) / float(peak_flops_per_second),
        float(bytes_moved) / float(bandwidth_bytes_per_second),
    )


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Declared work, storage, traffic, and optional roofline estimate."""

    flops: int
    storage_entries: int
    streaming_bytes: int
    arithmetic_intensity: float
    roofline_seconds: float | None
    metadata: Mapping[str, object]


def packet_cost_estimate(
    source_layout: PacketLayout,
    target_layout: PacketLayout,
    support: Iterable[Address],
    *,
    rhs: int = 1,
    itemsize: int = 8,
    peak_flops_per_second: float | None = None,
    bandwidth_bytes_per_second: float | None = None,
) -> CostEstimate:
    """Build one cost estimate for a dense packet support pattern."""

    support_tuple = tuple(_checked_addresses(source_layout, target_layout, support))
    flops = packet_apply_flops(source_layout, target_layout, support_tuple, rhs)
    storage = packet_dense_storage_entries(source_layout, target_layout, support_tuple)
    moved = packet_streaming_bytes(
        source_layout,
        target_layout,
        support_tuple,
        rhs=rhs,
        itemsize=itemsize,
    )
    intensity = arithmetic_intensity(flops, moved)
    if (peak_flops_per_second is None) != (bandwidth_bytes_per_second is None):
        raise ValidationError("both device rates must be supplied together")
    roofline = None
    if peak_flops_per_second is not None and bandwidth_bytes_per_second is not None:
        roofline = roofline_time_lower_bound(
            flops,
            moved,
            peak_flops_per_second,
            bandwidth_bytes_per_second,
        )
    return CostEstimate(
        flops,
        storage,
        moved,
        intensity,
        roofline,
        {
            "rhs": int(rhs),
            "itemsize": int(itemsize),
            "active_blocks": len(support_tuple),
        },
    )


__all__ = [
    "Address",
    "CostEstimate",
    "arithmetic_intensity",
    "block_diagonal_low_rank_flops",
    "block_diagonal_low_rank_storage_entries",
    "dense_apply_flops",
    "packet_apply_flops",
    "packet_cost_estimate",
    "packet_dense_storage_entries",
    "packet_streaming_bytes",
    "roofline_time_lower_bound",
]
