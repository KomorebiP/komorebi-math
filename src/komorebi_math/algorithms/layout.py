"""Packet layouts for structured numerical operators."""

from __future__ import annotations

from dataclasses import dataclass, field
from operator import index as integer_index
from typing import Any, Sequence

import numpy as np

from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented


@documented(
    ProvenanceRecord(
        identifier="PSC-VIII-D01",
        title="Finite packet layout",
        status=ClaimStatus.DEFINITION,
        source="Paper VIII",
        assumptions=("positive finite packet sizes",),
        verification_suite="algorithms.layout",
    )
)
@dataclass(frozen=True, slots=True)
class PacketLayout:
    """A contiguous finite packet decomposition of one coordinate space.

    Parameters
    ----------
    sizes:
        Positive packet dimensions in coordinate order.
    names:
        Optional packet names. Names must be unique when supplied.
    """

    sizes: tuple[int, ...]
    names: tuple[str, ...] | None = None
    slices: tuple[slice, ...] = field(init=False, repr=False)
    offsets: tuple[int, ...] = field(init=False, repr=False)
    dimension: int = field(init=False)

    def __post_init__(self) -> None:
        try:
            sizes = tuple(integer_index(size) for size in self.sizes)
        except TypeError as exc:
            raise ValidationError("packet sizes must be integers") from exc
        if not sizes:
            raise ValidationError("a packet layout must contain at least one packet")
        if any(isinstance(value, (bool, np.bool_)) for value in self.sizes) or any(
            size <= 0 for size in sizes
        ):
            raise ValidationError("packet sizes must be positive integers")
        if self.names is None:
            names = tuple(str(index) for index in range(len(sizes)))
        else:
            names = tuple(str(name) for name in self.names)
            if len(names) != len(sizes):
                raise DimensionError("packet names must match the number of packet sizes")
            if any(not name.strip() for name in names):
                raise ValidationError("packet names must be nonempty")
            if len(set(names)) != len(names):
                raise ValidationError("packet names must be unique")
        offsets_list = [0]
        for size in sizes:
            offsets_list.append(offsets_list[-1] + size)
        slices = tuple(
            slice(offsets_list[index], offsets_list[index + 1])
            for index in range(len(sizes))
        )
        object.__setattr__(self, "sizes", sizes)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "offsets", tuple(offsets_list))
        object.__setattr__(self, "slices", slices)
        object.__setattr__(self, "dimension", offsets_list[-1])

    @classmethod
    def equal(cls, packet_count: int, packet_size: int, *, prefix: str = "p") -> "PacketLayout":
        try:
            count = integer_index(packet_count)
            size = integer_index(packet_size)
        except TypeError as exc:
            raise ValidationError("packet_count and packet_size must be integers") from exc
        if isinstance(packet_count, bool) or isinstance(packet_size, bool) or count <= 0 or size <= 0:
            raise ValidationError("packet_count and packet_size must be positive integers")
        return cls(
            (size,) * count,
            tuple(f"{prefix}{packet_index}" for packet_index in range(count)),
        )

    @classmethod
    def from_packet_system(cls, packet_system: Any) -> "PacketLayout":
        """Construct a layout from a finite ``PacketSystem``.

        The ranges of abstract projectors need not be contiguous in the original
        coordinates. This method records their dimensions for use in packet
        coordinate space.
        """

        try:
            sizes = tuple(int(value) for value in packet_system.packet_dimensions())
            names = tuple(str(value) for value in packet_system.names)
        except (AttributeError, TypeError) as exc:
            raise ValidationError("value does not provide packet dimensions and names") from exc
        return cls(sizes, names)

    def __len__(self) -> int:
        return len(self.sizes)

    def index(self, packet: int | str) -> int:
        if not isinstance(packet, (str, bool)):
            try:
                packet_index = integer_index(packet)
            except TypeError:
                packet_index = None
            if packet_index is not None:
                if not 0 <= packet_index < len(self):
                    raise IndexError("packet index is outside the layout")
                return packet_index
        try:
            return self.names.index(str(packet))
        except ValueError as exc:
            raise KeyError(f"unknown packet name: {packet}") from exc

    def packet_slice(self, packet: int | str) -> slice:
        return self.slices[self.index(packet)]

    def packet_size(self, packet: int | str) -> int:
        return self.sizes[self.index(packet)]

    def split(self, value: Any, *, copy: bool = False) -> tuple[np.ndarray, ...]:
        """Split a vector or a stack of right hand sides into contiguous packets."""

        array = np.asarray(value)
        if array.ndim not in (1, 2):
            raise DimensionError("a packet value must be a vector or a two dimensional right hand side")
        if array.shape[0] != self.dimension:
            raise DimensionError(
                f"leading dimension must be {self.dimension}, received {array.shape[0]}"
            )
        parts = tuple(array[packet_slice] for packet_slice in self.slices)
        if copy:
            return tuple(np.array(part, copy=True) for part in parts)
        return parts

    def merge(self, packets: Sequence[Any]) -> np.ndarray:
        """Merge packet vectors or packet right hand sides into one array."""

        if len(packets) != len(self):
            raise DimensionError("packet count does not match the layout")
        arrays = tuple(np.asarray(packet) for packet in packets)
        if not arrays:
            raise ValidationError("at least one packet is required")
        dimensions = {array.ndim for array in arrays}
        if dimensions not in ({1}, {2}):
            raise DimensionError("all packets must be vectors or all packets must be matrices")
        if any(array.shape[0] != size for array, size in zip(arrays, self.sizes, strict=True)):
            raise DimensionError("a packet has the wrong leading dimension")
        if arrays[0].ndim == 2:
            rhs_counts = {array.shape[1] for array in arrays}
            if len(rhs_counts) != 1:
                raise DimensionError("all packet right hand sides must have the same column count")
        return np.concatenate(arrays, axis=0)

    def crossed(self, other: "PacketLayout", *, separator: str = "⊗") -> "PacketLayout":
        """Return the tensor product packet layout."""

        sizes = tuple(left * right for left in self.sizes for right in other.sizes)
        names = tuple(
            f"{left}{separator}{right}"
            for left in self.names
            for right in other.names
        )
        return PacketLayout(sizes, names)

    def permutation(self, order: Sequence[int]) -> np.ndarray:
        """Return the coordinate permutation for a reordered packet list."""

        try:
            order_tuple = tuple(integer_index(packet_index) for packet_index in order)
        except TypeError as exc:
            raise ValidationError("order must contain integer packet indices") from exc
        if sorted(order_tuple) != list(range(len(self))):
            raise ValidationError("order must be a permutation of all packet indices")
        indices = np.concatenate(
            [np.arange(self.offsets[index], self.offsets[index + 1]) for index in order_tuple]
        )
        return indices.astype(int, copy=False)

    @property
    def stored_coordinate_count(self) -> int:
        return self.dimension


__all__ = ["PacketLayout"]
