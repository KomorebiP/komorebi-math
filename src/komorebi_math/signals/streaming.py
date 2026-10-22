"""Stateful chunked short-time Fourier analysis with batch equivalence certificates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .time_frequency import DetrendMode, stft
from .windows import WindowSpec, get_window


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValidationError(f"{name} must be a positive integer")
    return int(value)


def _finite_chunk(values: Any, *, allow_empty: bool = True) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1:
        raise DimensionError("streaming STFT chunks must be one dimensional")
    if not allow_empty and result.size == 0:
        raise ValidationError("streaming STFT chunk cannot be empty")
    if not np.all(np.isfinite(result)):
        raise ValidationError("streaming STFT chunks must contain only finite values")
    return result


@dataclass(frozen=True, slots=True)
class StreamingSTFTBatch:
    """Zero or more consecutive STFT frames emitted by one streaming operation."""

    coefficients: np.ndarray
    frequencies: np.ndarray
    times: np.ndarray
    starts: np.ndarray
    sample_rate: float
    frame_length: int
    hop_length: int
    fft_length: int
    onesided: bool

    @property
    def frame_count(self) -> int:
        """Return the number of emitted frames."""

        return int(self.coefficients.shape[0])


@dataclass(slots=True)
class StreamingSTFT:
    """Incremental finite STFT analysis with bounded overlap memory."""

    frame_length: int = 256
    hop_length: int | None = None
    fft_length: int | None = None
    window: WindowSpec = "hann"
    sample_rate: float = 1.0
    onesided: bool = True
    detrend: DetrendMode = "none"
    _analysis_window: np.ndarray = field(init=False, repr=False)
    _buffer: np.ndarray = field(init=False, repr=False)
    _buffer_start: int = field(default=0, init=False, repr=False)
    _next_frame_start: int = field(default=0, init=False, repr=False)
    _total_samples: int = field(default=0, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.frame_length = _positive_integer(self.frame_length, "frame_length")
        self.hop_length = (
            max(1, self.frame_length // 4)
            if self.hop_length is None
            else _positive_integer(self.hop_length, "hop_length")
        )
        if self.hop_length > self.frame_length:
            raise ValidationError("hop_length cannot exceed frame_length")
        self.fft_length = (
            self.frame_length
            if self.fft_length is None
            else _positive_integer(self.fft_length, "fft_length")
        )
        if self.fft_length < self.frame_length:
            raise ValidationError("fft_length cannot be smaller than frame_length")
        self.sample_rate = float(self.sample_rate)
        if not np.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValidationError("sample_rate must be positive and finite")
        if self.detrend not in ("none", "constant"):
            raise ValidationError("detrend must be 'none' or 'constant'")
        self._analysis_window = np.asarray(get_window(self.window, self.frame_length, periodic=True))
        self._buffer = np.zeros(0, dtype=float)

    @property
    def frequencies(self) -> np.ndarray:
        """Return the fixed Fourier-bin coordinates."""

        if self.onesided:
            return np.fft.rfftfreq(self.fft_length, d=1.0 / self.sample_rate)
        return np.fft.fftfreq(self.fft_length, d=1.0 / self.sample_rate)

    @property
    def buffered_samples(self) -> int:
        """Return the retained overlap and unframed sample count."""

        return int(self._buffer.size)

    @property
    def total_samples(self) -> int:
        """Return the number of input samples accepted so far."""

        return self._total_samples

    def reset(self) -> None:
        """Reset this analyzer to its deterministic empty state."""

        self._buffer = np.zeros(0, dtype=float)
        self._buffer_start = 0
        self._next_frame_start = 0
        self._total_samples = 0
        self._closed = False

    def _transform(self, frame: np.ndarray) -> np.ndarray:
        values = frame
        if self.detrend == "constant":
            values = values - np.mean(values)
        windowed = values * self._analysis_window
        if self.onesided:
            if not np.isrealobj(windowed):
                raise ValidationError("one-sided streaming STFT requires real-valued chunks")
            return np.fft.rfft(windowed, n=self.fft_length)
        return np.fft.fft(windowed, n=self.fft_length)

    def _extract(self, *, pad_end: bool) -> StreamingSTFTBatch:
        coefficient_rows: list[np.ndarray] = []
        starts: list[int] = []
        while self._total_samples - self._next_frame_start >= self.frame_length:
            offset = self._next_frame_start - self._buffer_start
            if offset < 0 or offset + self.frame_length > self._buffer.size:
                raise DimensionError("streaming STFT buffer lost required samples")
            frame = self._buffer[offset : offset + self.frame_length]
            coefficient_rows.append(self._transform(frame))
            starts.append(self._next_frame_start)
            self._next_frame_start += self.hop_length
            trim = min(max(0, self._next_frame_start - self._buffer_start), self._buffer.size)
            if trim:
                self._buffer = self._buffer[trim:].copy()
                self._buffer_start += trim

        # Batch framing with pad_end=True appends at most one terminal frame.
        # Once that frame reaches beyond the accepted sample range, no further
        # hop-shifted padded frames belong to the finite framing.
        available = self._total_samples - self._next_frame_start
        if self._total_samples <= self.frame_length:
            final_frame_start = 0
        else:
            final_frame_start = (
                (self._total_samples - self.frame_length + self.hop_length - 1)
                // self.hop_length
            ) * self.hop_length
        if pad_end and available > 0 and self._next_frame_start <= final_frame_start:
            offset = self._next_frame_start - self._buffer_start
            if offset < 0 or offset + available > self._buffer.size:
                raise DimensionError("streaming STFT buffer lost required terminal samples")
            frame = self._buffer[offset : offset + available]
            frame = np.pad(frame, (0, self.frame_length - available), mode="constant")
            coefficient_rows.append(self._transform(frame))
            starts.append(self._next_frame_start)
            self._next_frame_start += self.hop_length
            trim = min(max(0, self._next_frame_start - self._buffer_start), self._buffer.size)
            if trim:
                self._buffer = self._buffer[trim:].copy()
                self._buffer_start += trim
        frequencies = self.frequencies
        coefficients = (
            np.stack(coefficient_rows, axis=0)
            if coefficient_rows
            else np.zeros((0, frequencies.size), dtype=complex)
        )
        starts_array = np.asarray(starts, dtype=int)
        times = (starts_array + 0.5 * self.frame_length) / self.sample_rate
        return StreamingSTFTBatch(
            coefficients,
            frequencies,
            np.asarray(times, dtype=float),
            starts_array,
            self.sample_rate,
            self.frame_length,
            self.hop_length,
            self.fft_length,
            self.onesided,
        )

    def process(self, chunk: Any) -> StreamingSTFTBatch:
        """Accept one chunk and emit every newly complete frame."""

        if self._closed:
            raise ValidationError("cannot process samples after streaming STFT flush")
        values = _finite_chunk(chunk)
        if values.size:
            dtype = np.result_type(self._buffer.dtype, values.dtype)
            self._buffer = np.concatenate(
                (self._buffer.astype(dtype, copy=False), values.astype(dtype, copy=False))
            )
            self._total_samples += values.size
        return self._extract(pad_end=False)

    def flush(self, *, pad_end: bool = True) -> StreamingSTFTBatch:
        """Emit padded terminal frames or discard the incomplete tail and close."""

        if self._closed:
            frequencies = self.frequencies
            return StreamingSTFTBatch(
                np.zeros((0, frequencies.size), dtype=complex),
                frequencies,
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=int),
                self.sample_rate,
                self.frame_length,
                self.hop_length,
                self.fft_length,
                self.onesided,
            )
        result = self._extract(pad_end=bool(pad_end))
        self._closed = True
        self._buffer = np.zeros(0, dtype=self._buffer.dtype)
        self._buffer_start = self._total_samples
        return result


def _combine_batches(batches: Iterable[StreamingSTFTBatch], analyzer: StreamingSTFT) -> StreamingSTFTBatch:
    values = tuple(batches)
    nonempty = tuple(batch for batch in values if batch.frame_count)
    frequencies = analyzer.frequencies
    coefficients = (
        np.concatenate([batch.coefficients for batch in nonempty], axis=0)
        if nonempty
        else np.zeros((0, frequencies.size), dtype=complex)
    )
    times = (
        np.concatenate([batch.times for batch in nonempty]) if nonempty else np.zeros(0, dtype=float)
    )
    starts = (
        np.concatenate([batch.starts for batch in nonempty]) if nonempty else np.zeros(0, dtype=int)
    )
    return StreamingSTFTBatch(
        coefficients,
        frequencies,
        times,
        starts,
        analyzer.sample_rate,
        analyzer.frame_length,
        analyzer.hop_length,
        analyzer.fft_length,
        analyzer.onesided,
    )


@documented(
    ProvenanceRecord(
        identifier="KM-DSP-T07",
        title="Chunk-invariant finite STFT framing",
        status=ClaimStatus.PROVED,
        source="finite signal processing",
        assumptions=("contiguous complete chunk partition", "zero-padded terminal framing"),
        verification_suite="signals.streaming",
    )
)
def stream_stft(
    signal: Any,
    chunk_sizes: Iterable[int],
    *,
    frame_length: int = 256,
    hop_length: int | None = None,
    fft_length: int | None = None,
    window: WindowSpec = "hann",
    sample_rate: float = 1.0,
    onesided: bool = True,
    detrend: DetrendMode = "none",
    pad_end: bool = True,
    tolerance: float = 1e-11,
) -> ComputationResult[StreamingSTFTBatch]:
    """Analyze declared chunks and certify equality with the matching batch STFT."""

    source = _finite_chunk(signal, allow_empty=False)
    sizes = tuple(_positive_integer(size, "chunk size") for size in chunk_sizes)
    if sum(sizes) != source.size:
        raise DimensionError("chunk sizes must sum to the signal length")
    analyzer = StreamingSTFT(
        frame_length=frame_length,
        hop_length=hop_length,
        fft_length=fft_length,
        window=window,
        sample_rate=sample_rate,
        onesided=onesided,
        detrend=detrend,
    )
    batches: list[StreamingSTFTBatch] = []
    start = 0
    for size in sizes:
        batches.append(analyzer.process(source[start : start + size]))
        start += size
    batches.append(analyzer.flush(pad_end=pad_end))
    streamed = _combine_batches(batches, analyzer)
    batch = stft(
        source,
        sample_rate=sample_rate,
        frame_length=analyzer.frame_length,
        hop_length=analyzer.hop_length,
        fft_length=analyzer.fft_length,
        window=window,
        center=False,
        pad_end=pad_end,
        boundary="zeros",
        onesided=onesided,
        detrend=detrend,
    )
    if streamed.coefficients.shape != batch.coefficients.shape:
        residual = float("inf")
        time_residual = float("inf")
        valid = False
    else:
        residual = float(np.max(np.abs(streamed.coefficients - batch.coefficients))) if batch.coefficients.size else 0.0
        time_residual = float(np.max(np.abs(streamed.times - batch.times))) if batch.times.size else 0.0
        scale = max(1.0, float(np.max(np.abs(batch.coefficients))) if batch.coefficients.size else 1.0)
        valid = residual <= tolerance * scale and time_residual <= tolerance
    certificate = Certificate(
        valid=valid,
        exactness=Exactness.NUMERICAL,
        residuals={"stream_batch_coefficients": residual, "stream_batch_times": time_residual},
        tolerance=tolerance,
        assumptions_checked=("contiguous declared chunks", "matching finite STFT parameters"),
        theorem_id="KM-DSP-T07",
        message="streaming STFT equals batch STFT" if valid else "streaming STFT differs from batch STFT",
    )
    return ComputationResult(streamed, certificate)


__all__ = [
    "StreamingSTFT",
    "StreamingSTFTBatch",
    "stream_stft",
]
