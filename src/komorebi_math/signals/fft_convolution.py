"""FFT convolution and bounded-buffer overlap-add execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .filtering import FIRMode


def _vector(values: Any, name: str, *, allow_empty: bool = False) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1:
        raise DimensionError(f"{name} must be one dimensional")
    if not allow_empty and result.size == 0:
        raise DimensionError(f"{name} cannot be empty")
    if not np.all(np.isfinite(result)):
        raise ValidationError(f"{name} must contain only finite values")
    return result


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, int(value - 1).bit_length())


def _crop(full: np.ndarray, signal_length: int, kernel_length: int, mode: FIRMode) -> np.ndarray:
    if mode == "full":
        return full
    if mode == "causal":
        return full[:signal_length]
    if mode == "same":
        start = (kernel_length - 1) // 2
        return full[start : start + signal_length]
    if mode == "valid":
        if signal_length < kernel_length:
            return np.zeros(0, dtype=full.dtype)
        start = kernel_length - 1
        return full[start : start + signal_length - kernel_length + 1]
    raise ValidationError(f"unsupported convolution mode: {mode}")


@documented(
    ProvenanceRecord(
        identifier="KM-DSP-D28",
        title="Zero-padded finite FFT convolution",
        status=ClaimStatus.CLASSICAL,
        source="Block 12",
        assumptions=("finite one-dimensional inputs", "zero padding to linear-convolution length"),
        verification_suite="signals.fft_convolution",
    )
)
def fft_convolve(signal: Any, kernel: Any, *, mode: FIRMode = "full") -> np.ndarray:
    """Compute finite linear convolution by a zero-padded FFT."""

    source = _vector(signal, "signal")
    impulse = _vector(kernel, "kernel")
    full_length = source.size + impulse.size - 1
    fft_length = _next_power_of_two(full_length)
    if np.isrealobj(source) and np.isrealobj(impulse):
        transformed = np.fft.rfft(source, fft_length) * np.fft.rfft(impulse, fft_length)
        full = np.fft.irfft(transformed, fft_length)[:full_length]
    else:
        transformed = np.fft.fft(source, fft_length) * np.fft.fft(impulse, fft_length)
        full = np.fft.ifft(transformed, fft_length)[:full_length]
    return np.real_if_close(_crop(full, source.size, impulse.size, mode), tol=1000)


@dataclass(slots=True)
class OverlapAddConvolver:
    """Stateful bounded-input-buffer FFT overlap-add convolver."""

    kernel: np.ndarray
    block_length: int = 1024
    fft_length: int = field(init=False)
    _kernel_spectrum: np.ndarray = field(init=False, repr=False)
    _overlap: np.ndarray = field(init=False, repr=False)
    _buffer: np.ndarray = field(init=False, repr=False)
    _complex_mode: bool = field(init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    processed_samples: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        impulse = _vector(self.kernel, "kernel")
        if isinstance(self.block_length, bool) or int(self.block_length) != self.block_length or int(self.block_length) <= 0:
            raise ValidationError("block_length must be a positive integer")
        self.kernel = impulse.copy()
        self.block_length = int(self.block_length)
        self.fft_length = _next_power_of_two(self.block_length + impulse.size - 1)
        self._complex_mode = not np.isrealobj(impulse)
        self._kernel_spectrum = (
            np.fft.fft(impulse, self.fft_length)
            if self._complex_mode
            else np.fft.rfft(impulse, self.fft_length)
        )
        dtype = np.result_type(impulse.dtype, complex if self._complex_mode else float)
        self._overlap = np.zeros(impulse.size - 1, dtype=dtype)
        self._buffer = np.zeros(0, dtype=dtype)

    @property
    def buffered_samples(self) -> int:
        """Return the number of accepted input samples not yet transformed."""

        return int(self._buffer.size)

    def _promote_complex(self) -> None:
        if not self._complex_mode:
            self._complex_mode = True
            self._kernel_spectrum = np.fft.fft(self.kernel, self.fft_length)
            self._overlap = self._overlap.astype(complex)
            self._buffer = self._buffer.astype(complex)

    def _transform_block(self, values: np.ndarray) -> np.ndarray:
        if not np.isrealobj(values):
            self._promote_complex()
        if self._complex_mode:
            spectrum = np.fft.fft(values, self.fft_length) * self._kernel_spectrum
            convolution = np.fft.ifft(spectrum, self.fft_length)
        else:
            spectrum = np.fft.rfft(values, self.fft_length) * self._kernel_spectrum
            convolution = np.fft.irfft(spectrum, self.fft_length)
        required = values.size + self.kernel.size - 1
        convolution = np.asarray(convolution[:required])
        overlap_length = self._overlap.size
        if overlap_length:
            convolution[:overlap_length] += self._overlap
            self._overlap = convolution[values.size : values.size + overlap_length].copy()
        return np.real_if_close(convolution[: values.size], tol=1000)

    def process(self, chunk: Any) -> np.ndarray:
        """Accept one arbitrary chunk and emit every complete transform block."""

        if self._closed:
            raise ValidationError("cannot process samples after overlap-add flush")
        values = _vector(chunk, "chunk", allow_empty=True)
        self.processed_samples += values.size
        if values.size:
            if not np.isrealobj(values):
                self._promote_complex()
            dtype = np.result_type(self._buffer.dtype, values.dtype)
            self._buffer = np.concatenate(
                (self._buffer.astype(dtype, copy=False), values.astype(dtype, copy=False))
            )
        emitted: list[np.ndarray] = []
        while self._buffer.size >= self.block_length:
            block = self._buffer[: self.block_length]
            self._buffer = self._buffer[self.block_length :].copy()
            emitted.append(self._transform_block(block))
        if not emitted:
            return np.zeros(0, dtype=np.result_type(self._buffer.dtype, self.kernel.dtype))
        return np.concatenate(emitted)

    def flush(self) -> np.ndarray:
        """Transform the final partial block and emit the remaining convolution tail."""

        if self._closed:
            return np.zeros(0, dtype=np.result_type(self.kernel.dtype, float))
        self._closed = True
        emitted: list[np.ndarray] = []
        if self._buffer.size:
            emitted.append(self._transform_block(self._buffer))
            self._buffer = np.zeros(0, dtype=self._buffer.dtype)
        if self.processed_samples > 0 and self._overlap.size:
            emitted.append(np.real_if_close(self._overlap.copy(), tol=1000))
        if self._overlap.size:
            self._overlap = np.zeros_like(self._overlap)
        if not emitted:
            return np.zeros(0, dtype=np.result_type(self.kernel.dtype, float))
        return np.concatenate(emitted)

    def reset(self) -> None:
        """Restore the open zero-overlap state."""

        dtype = np.result_type(self.kernel.dtype, complex if not np.isrealobj(self.kernel) else float)
        self._complex_mode = not np.isrealobj(self.kernel)
        self._kernel_spectrum = (
            np.fft.fft(self.kernel, self.fft_length)
            if self._complex_mode
            else np.fft.rfft(self.kernel, self.fft_length)
        )
        self._overlap = np.zeros(self.kernel.size - 1, dtype=dtype)
        self._buffer = np.zeros(0, dtype=dtype)
        self._closed = False
        self.processed_samples = 0


@documented(
    ProvenanceRecord(
        identifier="KM-DSP-T10",
        title="Finite overlap-add convolution identity",
        status=ClaimStatus.PROVED,
        source="Block 12",
        assumptions=("ordered contiguous chunks", "zero-padded FFT blocks"),
        verification_suite="signals.fft_convolution",
    )
)
def overlap_add_convolve(
    signal: Any,
    kernel: Any,
    *,
    block_length: int = 1024,
    mode: FIRMode = "full",
) -> np.ndarray:
    """Compute finite convolution through bounded-buffer FFT overlap-add."""

    source = _vector(signal, "signal")
    impulse = _vector(kernel, "kernel")
    processor = OverlapAddConvolver(impulse, block_length=block_length)
    head = processor.process(source)
    tail = processor.flush()
    full = np.concatenate((head, tail))
    return np.real_if_close(_crop(full, source.size, impulse.size, mode), tol=1000)


def certified_overlap_add_convolve(
    signal: Any,
    kernel: Any,
    *,
    block_length: int = 1024,
    mode: FIRMode = "full",
    tolerance: float = 1e-11,
) -> ComputationResult[np.ndarray]:
    """Return overlap-add output with a direct-convolution equality certificate."""

    source = _vector(signal, "signal")
    impulse = _vector(kernel, "kernel")
    value = overlap_add_convolve(source, impulse, block_length=block_length, mode=mode)
    reference_full = np.convolve(source, impulse, mode="full")
    reference = _crop(reference_full, source.size, impulse.size, mode)
    residual = float(np.max(np.abs(value - reference))) if reference.size else 0.0
    scale = max(1.0, float(np.max(np.abs(reference))) if reference.size else 1.0)
    certificate = Certificate(
        valid=bool(value.shape == reference.shape and residual <= tolerance * scale),
        exactness=Exactness.NUMERICAL,
        residuals={"overlap_add_direct_max_abs": residual},
        tolerance=tolerance,
        assumptions_checked=("finite ordered input", "linear zero-padded convolution"),
        theorem_id="KM-DSP-T10",
        message="overlap-add convolution equals direct finite convolution",
    )
    return ComputationResult(value, certificate)


__all__ = [
    "OverlapAddConvolver",
    "certified_overlap_add_convolve",
    "fft_convolve",
    "overlap_add_convolve",
]
