"""Finite convolution operators and their exact polyphase blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .polyphase import PolyphaseCoordinates, polyphase_analyze

ConvolutionMode = Literal["full", "same", "valid", "circular"]


def _as_vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise DimensionError(f"{name} must be one dimensional")
    if array.size == 0:
        raise ValidationError(f"{name} cannot be empty")
    return array


def convolution_output_length(input_length: int, kernel_length: int, mode: ConvolutionMode) -> int:
    """Return the finite one dimensional convolution output length."""

    n = int(input_length)
    k = int(kernel_length)
    if n <= 0 or k <= 0:
        raise ValidationError("input and kernel lengths must be positive")
    if mode == "full":
        return n + k - 1
    if mode == "same" or mode == "circular":
        return n
    if mode == "valid":
        return max(n - k + 1, 0)
    raise ValidationError(f"unknown convolution mode: {mode}")


@documented(
    ProvenanceRecord(
        identifier="PSC-V-T05",
        title="Finite convolution matrix",
        status=ClaimStatus.PROVED,
        source="Paper V",
        assumptions=("finite one dimensional signal", "declared boundary mode"),
        verification_suite="signals.convolution",
    )
)
def convolution_matrix(
    kernel: Any,
    input_length: int,
    *,
    mode: ConvolutionMode = "full",
) -> np.ndarray:
    """Return the matrix implementing a finite convolution convention."""

    h = _as_vector(kernel, "kernel")
    n = int(input_length)
    if n <= 0:
        raise ValidationError("input length must be positive")
    dtype = np.result_type(h.dtype, float)
    if mode == "circular":
        matrix = np.zeros((n, n), dtype=dtype)
        for output in range(n):
            for tap, coefficient in enumerate(h):
                matrix[output, (output - tap) % n] += coefficient
        return matrix

    full = np.zeros((n + h.size - 1, n), dtype=dtype)
    for column in range(n):
        full[column : column + h.size, column] = h
    if mode == "full":
        return full
    if mode == "same":
        start = (h.size - 1) // 2
        return full[start : start + n]
    if mode == "valid":
        length = max(n - h.size + 1, 0)
        if length == 0:
            return np.zeros((0, n), dtype=dtype)
        start = h.size - 1
        return full[start : start + length]
    raise ValidationError(f"unknown convolution mode: {mode}")


def convolve_signal(signal: Any, kernel: Any, *, mode: ConvolutionMode = "full") -> np.ndarray:
    """Apply the declared finite convolution convention."""

    x = _as_vector(signal, "signal")
    return convolution_matrix(kernel, x.size, mode=mode) @ x


@documented(
    ProvenanceRecord(
        identifier="PSC-V-D06",
        title="Finite polyphase convolution transfer matrix",
        status=ClaimStatus.DEFINITION,
        source="Paper V",
        assumptions=("finite convolution matrix", "common packet order"),
        verification_suite="signals.convolution",
    )
)
@dataclass(frozen=True, slots=True)
class PolyphaseConvolution:
    """Exact packet block representation of a finite convolution operator."""

    kernel: np.ndarray
    input_length: int
    order: int
    mode: ConvolutionMode = "full"

    def __post_init__(self) -> None:
        kernel = _as_vector(self.kernel, "kernel")
        n = int(self.input_length)
        order = int(self.order)
        if n <= 0:
            raise ValidationError("input length must be positive")
        if order <= 0:
            raise ValidationError("packet order must be positive")
        object.__setattr__(self, "kernel", kernel)
        object.__setattr__(self, "input_length", n)
        object.__setattr__(self, "order", order)

    @property
    def matrix(self) -> np.ndarray:
        return convolution_matrix(self.kernel, self.input_length, mode=self.mode)

    @property
    def output_length(self) -> int:
        return self.matrix.shape[0]

    def source_indices(self, residue: int) -> np.ndarray:
        return np.arange(residue, self.input_length, self.order, dtype=int)

    def target_indices(self, residue: int) -> np.ndarray:
        return np.arange(residue, self.output_length, self.order, dtype=int)

    def blocks(self) -> tuple[tuple[np.ndarray, ...], ...]:
        """Return blocks indexed by target residue and source residue."""

        matrix = self.matrix
        return tuple(
            tuple(
                matrix[np.ix_(self.target_indices(target), self.source_indices(source))]
                for source in range(self.order)
            )
            for target in range(self.order)
        )

    def apply_coordinates(self, coordinates: PolyphaseCoordinates) -> PolyphaseCoordinates:
        if coordinates.order != self.order:
            raise DimensionError("coordinate order does not match the convolution transfer order")
        if coordinates.original_shape != (self.input_length,):
            raise DimensionError("coordinate shape does not match the declared input length")
        block_rows = self.blocks()
        channels: list[np.ndarray] = []
        for target, row in enumerate(block_rows):
            length = self.target_indices(target).size
            value = np.zeros(length, dtype=np.result_type(self.kernel.dtype, coordinates.dtype))
            for block, channel in zip(row, coordinates.channels, strict=True):
                value = value + block @ channel
            channels.append(value)
        return PolyphaseCoordinates(
            tuple(channels),
            self.order,
            0,
            (self.output_length,),
        )

    def apply(self, signal: Any) -> np.ndarray:
        x = _as_vector(signal, "signal")
        if x.size != self.input_length:
            raise DimensionError("signal length does not match the declared convolution input")
        coordinates = polyphase_analyze(x, self.order)
        return self.apply_coordinates(coordinates).reconstruct()

    @documented(
        ProvenanceRecord(
            identifier="PSC-V-T07",
            title="Exact finite convolution transfer law",
            status=ClaimStatus.PROVED,
            source="Paper V",
            assumptions=("finite convolution", "residue packet coordinates"),
            verification_suite="signals.convolution",
        )
    )
    def certificate(self, signal: Any, *, tolerance: float = 1e-11) -> ComputationResult[np.ndarray]:
        x = _as_vector(signal, "signal")
        direct = self.matrix @ x
        packet = self.apply(x)
        residual = float(np.linalg.norm(packet - direct))
        valid = residual <= tolerance
        return ComputationResult(
            packet,
            Certificate(
                valid=valid,
                exactness=Exactness.NUMERICAL,
                residuals={"transfer_law": residual},
                tolerance=tolerance,
                assumptions_checked=("finite convolution matrix", "exact residue partition"),
                theorem_id="PSC-V-T07",
                message="polyphase convolution transfer verified" if valid else "polyphase convolution transfer failed",
            ),
        )


@dataclass(frozen=True, slots=True)
class PolyphaseBlockMatrix:
    """Finite packet blocks of a convolution matrix."""

    blocks: tuple[tuple[np.ndarray, ...], ...]
    order: int
    input_length: int
    output_length: int


def finite_polyphase_blocks(
    kernel: Any,
    input_length: int,
    order: int,
    *,
    mode: ConvolutionMode = "full",
) -> PolyphaseBlockMatrix:
    """Return the exact finite packet blocks of a convolution operator."""

    transfer = PolyphaseConvolution(np.asarray(kernel), input_length, order, mode)
    return PolyphaseBlockMatrix(transfer.blocks(), transfer.order, transfer.input_length, transfer.output_length)


def polyphase_convolution(
    signal: Any,
    kernel: Any,
    order: int,
    *,
    mode: ConvolutionMode = "full",
    tolerance: float = 1e-11,
) -> ComputationResult[np.ndarray]:
    """Apply a convolution through exact finite polyphase blocks."""

    x = _as_vector(signal, "signal")
    return PolyphaseConvolution(np.asarray(kernel), x.size, order, mode).certificate(x, tolerance=tolerance)


__all__ = [
    "ConvolutionMode",
    "PolyphaseConvolution",
    "PolyphaseBlockMatrix",
    "convolution_matrix",
    "convolution_output_length",
    "convolve_signal",
    "polyphase_convolution",
    "finite_polyphase_blocks",
]
