"""Packet operator intermediate representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._validation import strict_integer
from .layout import PacketLayout
from .structured import DenseOperator, KroneckerOperator, StructuredOperator, as_structured_operator

Address = tuple[int, int]


@documented(
    ProvenanceRecord(
        identifier="PSC-VIII-T01",
        title="Exact packet operator intermediate representation",
        status=ClaimStatus.PROVED,
        source="Paper VIII",
        assumptions=("finite source and target packet layouts", "declared block dimensions"),
        verification_suite="algorithms.ir",
    )
)
@dataclass(frozen=True, slots=True)
class PacketIR:
    """A finite operator as a map from source packets to target packets.

    Block addresses use ``(target, source)`` order throughout the algorithms
    package.
    """

    source_layout: PacketLayout
    target_layout: PacketLayout
    blocks: Mapping[Address, StructuredOperator | Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _blocks: Mapping[Address, StructuredOperator] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        checked: dict[Address, StructuredOperator] = {}
        for address, value in self.blocks.items():
            try:
                if len(address) != 2:
                    raise ValidationError("a packet address must be a target and source pair")
                target = strict_integer(address[0], "target packet index", minimum=0)
                source = strict_integer(address[1], "source packet index", minimum=0)
            except TypeError as exc:
                raise ValidationError("a packet address must be a target and source pair") from exc
            if not 0 <= target < len(self.target_layout):
                raise DimensionError("target packet index is outside the target layout")
            if not 0 <= source < len(self.source_layout):
                raise DimensionError("source packet index is outside the source layout")
            block = as_structured_operator(value)
            expected = (
                self.target_layout.sizes[target],
                self.source_layout.sizes[source],
            )
            if block.shape != expected:
                raise DimensionError(
                    f"block {(target, source)} has shape {block.shape}, expected {expected}"
                )
            checked[(target, source)] = block
        object.__setattr__(self, "_blocks", MappingProxyType(checked))
        object.__setattr__(self, "blocks", MappingProxyType(checked))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def square(
        cls,
        layout: PacketLayout,
        blocks: Mapping[Address, StructuredOperator | Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "PacketIR":
        return cls(layout, layout, blocks, {} if metadata is None else metadata)

    @classmethod
    def from_dense(
        cls,
        matrix: Any,
        source_layout: PacketLayout,
        target_layout: PacketLayout | None = None,
        *,
        atol: float = 0.0,
        rtol: float = 0.0,
        keep_zeros: bool = False,
    ) -> "PacketIR":
        target = source_layout if target_layout is None else target_layout
        if atol < 0 or rtol < 0 or not np.isfinite(atol) or not np.isfinite(rtol):
            raise ValidationError("atol and rtol must be finite and nonnegative")
        array = np.asarray(matrix)
        expected = (target.dimension, source_layout.dimension)
        if array.ndim != 2 or array.shape != expected:
            raise DimensionError(f"matrix shape must be {expected}")
        scale = float(np.linalg.norm(array, 2)) if array.size else 0.0
        threshold = float(atol) + float(rtol) * scale
        blocks: dict[Address, DenseOperator] = {}
        for target_index, target_slice in enumerate(target.slices):
            for source_index, source_slice in enumerate(source_layout.slices):
                block = np.array(array[target_slice, source_slice], copy=True)
                if keep_zeros or float(np.linalg.norm(block, 2)) > threshold:
                    blocks[(target_index, source_index)] = DenseOperator(block)
        return cls(source_layout, target, blocks, {"support_threshold": threshold})

    @property
    def shape(self) -> tuple[int, int]:
        return (self.target_layout.dimension, self.source_layout.dimension)

    @property
    def dtype(self) -> np.dtype[Any]:
        if not self._blocks:
            return np.dtype(float)
        return np.result_type(*(block.dtype for block in self._blocks.values()))

    @property
    def support(self) -> frozenset[Address]:
        return frozenset(self._blocks)

    @property
    def stored_entries(self) -> int:
        return int(sum(block.storage_entries() for block in self._blocks.values()))

    def flop_count(self, rhs: int = 1) -> int:
        rhs_count = strict_integer(rhs, "rhs", minimum=1)
        return int(sum(block.flop_count(rhs_count) for block in self._blocks.values()))

    def block(self, target: int | str, source: int | str) -> StructuredOperator | None:
        address = (self.target_layout.index(target), self.source_layout.index(source))
        return self._blocks.get(address)

    def materialize(self) -> np.ndarray:
        result = np.zeros(self.shape, dtype=self.dtype)
        for (target, source), block in self._blocks.items():
            result[
                self.target_layout.slices[target],
                self.source_layout.slices[source],
            ] += block.materialize()
        return result

    def apply(self, value: Any) -> np.ndarray:
        array = np.asarray(value)
        vector = array.ndim == 1
        if vector:
            array = array[:, None]
        if array.ndim != 2 or array.shape[0] != self.source_layout.dimension:
            raise DimensionError("right hand side has an incompatible shape")
        source_parts = self.source_layout.split(array)
        dtype = np.result_type(array.dtype, self.dtype)
        target_parts = [
            np.zeros((size, array.shape[1]), dtype=dtype)
            for size in self.target_layout.sizes
        ]
        for (target, source), block in self._blocks.items():
            target_parts[target] += block.apply(source_parts[source])
        result = self.target_layout.merge(target_parts)
        return result[:, 0] if vector else result

    def adjoint(self) -> "PacketIR":
        return PacketIR(
            self.target_layout,
            self.source_layout,
            {
                (source, target): block.adjoint()
                for (target, source), block in self._blocks.items()
            },
            {**self.metadata, "operation": "adjoint"},
        )

    def compose(self, right: "PacketIR", *, atol: float = 0.0) -> "PacketIR":
        """Return ``self`` after ``right``."""

        if atol < 0 or not np.isfinite(atol):
            raise ValidationError("atol must be finite and nonnegative")
        if right.target_layout != self.source_layout:
            raise DimensionError("intermediate packet layouts must agree")
        result: dict[Address, DenseOperator] = {}
        for target in range(len(self.target_layout)):
            for source in range(len(right.source_layout)):
                accumulator: np.ndarray | None = None
                for middle in range(len(self.source_layout)):
                    left_block = self._blocks.get((target, middle))
                    right_block = right._blocks.get((middle, source))
                    if left_block is None or right_block is None:
                        continue
                    term = left_block.materialize() @ right_block.materialize()
                    accumulator = term if accumulator is None else accumulator + term
                if accumulator is not None and (
                    atol <= 0.0 or float(np.linalg.norm(accumulator, 2)) > atol
                ):
                    result[(target, source)] = DenseOperator(accumulator)
        return PacketIR(
            right.source_layout,
            self.target_layout,
            result,
            {"operation": "composition"},
        )

    def power(self, exponent: int, *, atol: float = 0.0) -> "PacketIR":
        if self.source_layout != self.target_layout:
            raise DimensionError("only square packet operators have powers")
        exponent_value = strict_integer(exponent, "exponent", minimum=0)
        identity_blocks = {
            (index, index): np.eye(size, dtype=self.dtype)
            for index, size in enumerate(self.source_layout.sizes)
        }
        result = PacketIR.square(self.source_layout, identity_blocks)
        base = self
        power = exponent_value
        while power:
            if power & 1:
                result = base.compose(result, atol=atol)
            power >>= 1
            if power:
                base = base.compose(base, atol=atol)
        return result

    def tensor(self, other: "PacketIR") -> "PacketIR":
        source = self.source_layout.crossed(other.source_layout)
        target = self.target_layout.crossed(other.target_layout)
        blocks: dict[Address, StructuredOperator] = {}
        other_source_count = len(other.source_layout)
        other_target_count = len(other.target_layout)
        for (target_left, source_left), left_block in self._blocks.items():
            for (target_right, source_right), right_block in other._blocks.items():
                target_index = target_left * other_target_count + target_right
                source_index = source_left * other_source_count + source_right
                blocks[(target_index, source_index)] = KroneckerOperator(
                    left_block.materialize(),
                    right_block.materialize(),
                )
        return PacketIR(source, target, blocks, {"operation": "tensor"})

    def diagonal_part(self) -> "PacketIR":
        if self.source_layout != self.target_layout:
            raise DimensionError("a diagonal packet part requires one square layout")
        return PacketIR.square(
            self.source_layout,
            {
                address: block
                for address, block in self._blocks.items()
                if address[0] == address[1]
            },
            metadata={"operation": "packet_diagonal"},
        )

    def certificate(self, *, tolerance: float = 1e-10) -> Certificate:
        if tolerance < 0 or not np.isfinite(tolerance):
            raise ValidationError("tolerance must be finite and nonnegative")
        matrix = self.materialize()
        sample = np.eye(self.source_layout.dimension, dtype=self.dtype)
        residual = float(np.linalg.norm(self.apply(sample) - matrix))
        valid = residual <= tolerance
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"apply_materialize": residual},
            tolerance=tolerance,
            assumptions_checked=("block dimensions", "packet address ranges"),
            diagnostics={
                "shape": self.shape,
                "active_blocks": len(self._blocks),
                "stored_entries": self.stored_entries,
                "dense_entries": self.shape[0] * self.shape[1],
            },
            theorem_id="PSC-VIII-T01",
            message="valid packet intermediate representation" if valid else "packet IR failed reconstruction",
        )


__all__ = ["Address", "PacketIR"]
