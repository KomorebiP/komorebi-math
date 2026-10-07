"""Finite polyphase coordinates, residue projectors, and transfer kernels."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented


def _strict_positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValidationError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{name} must be a positive integer") from exc
    if result != value or result <= 0:
        raise ValidationError(f"{name} must be a positive integer")
    return result


def _normalize_axis(axis: int, ndim: int) -> int:
    result = int(axis)
    if result < 0:
        result += ndim
    if result < 0 or result >= ndim:
        raise DimensionError("axis is outside the signal dimensions")
    return result


@documented(
    ProvenanceRecord(
        identifier="PSC-V-D01",
        title="One dimensional polyphase coordinates",
        status=ClaimStatus.DEFINITION,
        source="Paper V",
        assumptions=("finite array", "positive packet order"),
        verification_suite="signals.polyphase",
    )
)
@dataclass(frozen=True, slots=True)
class PolyphaseCoordinates:
    """Residue channels of a finite array along one axis."""

    channels: tuple[np.ndarray, ...]
    order: int
    axis: int
    original_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        order = _strict_positive_int(self.order, "order")
        if len(self.channels) != order:
            raise DimensionError("the number of channels must equal the packet order")
        if not self.original_shape:
            raise DimensionError("a scalar has no polyphase axis")
        axis = _normalize_axis(self.axis, len(self.original_shape))
        object.__setattr__(self, "channels", tuple(np.asarray(channel) for channel in self.channels))
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "axis", axis)

    @property
    def dtype(self) -> np.dtype[Any]:
        if not self.channels:
            return np.dtype(float)
        return np.result_type(*(channel.dtype for channel in self.channels))

    def reconstruct(self) -> np.ndarray:
        """Interleave the channels into the original finite array."""

        result = np.empty(self.original_shape, dtype=self.dtype)
        for residue, channel in enumerate(self.channels):
            index = [slice(None)] * len(self.original_shape)
            index[self.axis] = slice(residue, self.original_shape[self.axis], self.order)
            expected_shape = result[tuple(index)].shape
            if channel.shape != expected_shape:
                raise DimensionError(
                    f"channel {residue} has shape {channel.shape}, expected {expected_shape}"
                )
            result[tuple(index)] = channel
        return result

    def channel_lengths(self) -> tuple[int, ...]:
        return tuple(channel.shape[self.axis] for channel in self.channels)

    def refine(self, fine_order: int, *, copy: bool = True) -> "PolyphaseCoordinates":
        """Refine these coordinates to a divisible finer packet order."""

        from ..refinement.polyphase import refine_polyphase

        refined = refine_polyphase(self, fine_order, copy=copy)
        if not isinstance(refined, PolyphaseCoordinates):
            raise ValidationError("internal signal refinement returned the wrong coordinate type")
        return refined

    def coarsen(self, coarse_order: int, *, copy: bool = True) -> "PolyphaseCoordinates":
        """Coarsen these coordinates to a divisor of the current order."""

        from ..refinement.polyphase import coarsen_polyphase

        coarsened = coarsen_polyphase(self, coarse_order, copy=copy)
        if not isinstance(coarsened, PolyphaseCoordinates):
            raise ValidationError("internal signal coarsening returned the wrong coordinate type")
        return coarsened

    def certificate(self, original: Any, *, tolerance: float = 1e-12) -> Certificate:
        """Certify reconstruction against a supplied original array."""

        expected = np.asarray(original)
        if expected.shape != self.original_shape:
            raise DimensionError("original shape does not match the polyphase coordinate record")
        reconstructed = self.reconstruct()
        residual = float(np.linalg.norm((reconstructed - expected).reshape(-1)))
        scale = max(1.0, float(np.linalg.norm(expected.reshape(-1))))
        valid = residual <= float(tolerance) * scale
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"polyphase_reconstruction": residual},
            tolerance=float(tolerance),
            assumptions_checked=("finite residue partition", "declared original shape"),
            theorem_id="PSC-V-T02",
            message="polyphase reconstruction verified" if valid else "polyphase reconstruction failed",
        )


@dataclass(frozen=True, slots=True)
class PolyphaseSystem:
    """Reusable one axis polyphase analysis and synthesis system."""

    order: int
    axis: int = -1

    def __post_init__(self) -> None:
        object.__setattr__(self, "order", _strict_positive_int(self.order, "order"))
        object.__setattr__(self, "axis", int(self.axis))

    def analyze(self, signal: Any, *, copy: bool = True) -> PolyphaseCoordinates:
        return polyphase_analyze(signal, self.order, axis=self.axis, copy=copy)

    def synthesize(self, coordinates: PolyphaseCoordinates) -> np.ndarray:
        if coordinates.order != self.order:
            raise DimensionError("coordinate order does not match this polyphase system")
        return coordinates.reconstruct()

    def certificate(self, signal: Any, *, tolerance: float = 1e-12) -> Certificate:
        coordinates = self.analyze(signal)
        return coordinates.certificate(signal, tolerance=tolerance)


@documented(
    ProvenanceRecord(
        identifier="PSC-V-T02",
        title="Exact finite polyphase decomposition",
        status=ClaimStatus.PROVED,
        source="Paper V",
        assumptions=("finite array", "positive packet order"),
        verification_suite="signals.polyphase",
    )
)
def polyphase_analyze(
    signal: Any,
    order: int,
    *,
    axis: int = -1,
    copy: bool = True,
) -> PolyphaseCoordinates:
    """Split a finite array into residue channels along ``axis``."""

    array = np.asarray(signal)
    if array.ndim == 0:
        raise DimensionError("a scalar has no polyphase decomposition")
    packet_order = _strict_positive_int(order, "order")
    normalized_axis = _normalize_axis(axis, array.ndim)
    channels: list[np.ndarray] = []
    for residue in range(packet_order):
        index = [slice(None)] * array.ndim
        index[normalized_axis] = slice(residue, array.shape[normalized_axis], packet_order)
        channel = array[tuple(index)]
        channels.append(channel.copy() if copy else channel)
    return PolyphaseCoordinates(tuple(channels), packet_order, normalized_axis, tuple(array.shape))


def polyphase_synthesize(coordinates: PolyphaseCoordinates) -> np.ndarray:
    """Reconstruct a finite array from polyphase coordinates."""

    if not isinstance(coordinates, PolyphaseCoordinates):
        raise ValidationError("coordinates must be a PolyphaseCoordinates object")
    return coordinates.reconstruct()


def residue_project(signal: Any, order: int, residue: int, *, axis: int = -1) -> np.ndarray:
    """Return the full shaped signal supported on one residue class."""

    array = np.asarray(signal)
    if array.ndim == 0:
        raise DimensionError("a scalar has no residue projector")
    packet_order = _strict_positive_int(order, "order")
    value = int(residue)
    if value < 0 or value >= packet_order:
        raise ValidationError("residue must lie between zero and order minus one")
    normalized_axis = _normalize_axis(axis, array.ndim)
    result = np.zeros_like(array)
    index = [slice(None)] * array.ndim
    index[normalized_axis] = slice(value, array.shape[normalized_axis], packet_order)
    result[tuple(index)] = array[tuple(index)]
    return result


def refinement_children(coarse_order: int, fine_order: int, residue: int) -> tuple[int, ...]:
    """Return fine residues whose reduction modulo the coarse order is ``residue``."""

    coarse = _strict_positive_int(coarse_order, "coarse_order")
    fine = _strict_positive_int(fine_order, "fine_order")
    if fine % coarse:
        raise ValidationError("fine_order must be divisible by coarse_order")
    value = int(residue)
    if value < 0 or value >= coarse:
        raise ValidationError("residue must lie in the coarse residue range")
    return tuple(range(value, fine, coarse))


@dataclass(frozen=True, slots=True)
class PolyphaseKernel:
    """One source to target filter kernel in an infinite polyphase law."""

    target: int
    source: int
    coefficients: np.ndarray
    order: int

    def __post_init__(self) -> None:
        packet_order = _strict_positive_int(self.order, "order")
        target = int(self.target)
        source = int(self.source)
        if not 0 <= target < packet_order or not 0 <= source < packet_order:
            raise ValidationError("source and target residues must lie in the packet range")
        coefficients = np.asarray(self.coefficients)
        if coefficients.ndim != 1:
            raise DimensionError("polyphase kernel coefficients must be one dimensional")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "order", packet_order)


def polyphase_filter_kernels(kernel: Any, order: int) -> tuple[tuple[PolyphaseKernel, ...], ...]:
    """Return the causal kernels ``h[Nm + target - source]`` for every residue pair.

    Coefficients with negative or out of range tap indices are omitted. The
    resulting objects describe the usual infinite zero extended polyphase law.
    """

    h = np.asarray(kernel)
    if h.ndim != 1 or h.size == 0:
        raise DimensionError("kernel must be a nonempty one dimensional array")
    packet_order = _strict_positive_int(order, "order")
    rows: list[tuple[PolyphaseKernel, ...]] = []
    for target in range(packet_order):
        row: list[PolyphaseKernel] = []
        for source in range(packet_order):
            taps: list[Any] = []
            shift = target - source
            maximum_lag = int(np.floor((h.size - 1 - shift) / packet_order))
            if maximum_lag < 0:
                taps.append(0)
            else:
                for coarse_lag in range(maximum_lag + 1):
                    tap = packet_order * coarse_lag + shift
                    taps.append(h[tap] if 0 <= tap < h.size else 0)
            row.append(PolyphaseKernel(target, source, np.asarray(taps, dtype=h.dtype), packet_order))
        rows.append(tuple(row))
    return tuple(rows)


def polyphase_convolution(signal: Any, kernel: Any, order: int) -> ComputationResult[np.ndarray]:
    """Apply full convolution through exact finite polyphase transfer blocks."""

    from .convolution import PolyphaseConvolution

    x = np.asarray(signal)
    if x.ndim != 1 or x.size == 0:
        raise DimensionError("signal must be a nonempty one dimensional array")
    transfer = PolyphaseConvolution(np.asarray(kernel), x.size, _strict_positive_int(order, "order"), mode="full")
    return transfer.certificate(x)


@documented(
    ProvenanceRecord(
        identifier="PSC-V-D03",
        title="Rectangular lattice packet coordinates",
        status=ClaimStatus.DEFINITION,
        source="Paper V",
        assumptions=("finite array", "positive stride in every dimension"),
        verification_suite="signals.polyphase",
    )
)
@dataclass(frozen=True, slots=True)
class LatticePolyphaseCoordinates:
    """Rectangular lattice residue channels of a finite multidimensional array."""

    channels: dict[tuple[int, ...], np.ndarray]
    strides: tuple[int, ...]
    original_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.strides) != len(self.original_shape):
            raise DimensionError("one lattice stride is required for every array dimension")
        normalized = tuple(_strict_positive_int(value, "stride") for value in self.strides)
        expected = set(product(*(range(value) for value in normalized)))
        if set(self.channels) != expected:
            raise ValidationError("lattice channels do not contain every residue address exactly once")
        object.__setattr__(self, "channels", {address: np.asarray(channel) for address, channel in self.channels.items()})
        object.__setattr__(self, "strides", normalized)

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.result_type(*(channel.dtype for channel in self.channels.values()))

    def reconstruct(self) -> np.ndarray:
        result = np.empty(self.original_shape, dtype=self.dtype)
        for address, channel in self.channels.items():
            index = tuple(
                slice(residue, length, stride)
                for residue, length, stride in zip(address, self.original_shape, self.strides, strict=True)
            )
            expected = result[index].shape
            if channel.shape != expected:
                raise DimensionError(f"channel {address} has shape {channel.shape}, expected {expected}")
            result[index] = channel
        return result

    def refine(
        self,
        fine_strides: Iterable[int],
        *,
        copy: bool = True,
    ) -> "LatticePolyphaseCoordinates":
        """Refine these coordinates componentwise to finer lattice strides."""

        from ..refinement.polyphase import refine_lattice_polyphase

        return refine_lattice_polyphase(self, fine_strides, copy=copy)

    def coarsen(
        self,
        coarse_strides: Iterable[int],
        *,
        copy: bool = True,
    ) -> "LatticePolyphaseCoordinates":
        """Coarsen these coordinates componentwise to coarser lattice strides."""

        from ..refinement.polyphase import coarsen_lattice_polyphase

        return coarsen_lattice_polyphase(self, coarse_strides, copy=copy)

    def certificate(self, original: Any, *, tolerance: float = 1e-12) -> Certificate:
        expected = np.asarray(original)
        if expected.shape != self.original_shape:
            raise DimensionError("original shape does not match lattice coordinates")
        residual = float(np.linalg.norm((self.reconstruct() - expected).reshape(-1)))
        scale = max(1.0, float(np.linalg.norm(expected.reshape(-1))))
        valid = residual <= tolerance * scale
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"lattice_polyphase_reconstruction": residual},
            tolerance=tolerance,
            assumptions_checked=("rectangular lattice residue partition",),
            theorem_id="PSC-V-T04",
            message="lattice polyphase reconstruction verified" if valid else "lattice reconstruction failed",
        )


@documented(
    ProvenanceRecord(
        identifier="PSC-V-T04",
        title="Exact rectangular lattice polyphase decomposition",
        status=ClaimStatus.PROVED,
        source="Paper V",
        assumptions=("finite multidimensional array", "positive rectangular strides"),
        verification_suite="signals.polyphase",
    )
)
def lattice_polyphase_analyze(
    signal: Any,
    strides: Iterable[int],
    *,
    copy: bool = True,
) -> LatticePolyphaseCoordinates:
    """Split an array into channels indexed by a rectangular lattice quotient."""

    array = np.asarray(signal)
    if array.ndim == 0:
        raise DimensionError("a scalar has no lattice packet decomposition")
    stride_values = tuple(_strict_positive_int(value, "stride") for value in strides)
    if len(stride_values) != array.ndim:
        raise DimensionError("one stride is required for every array dimension")
    channels: dict[tuple[int, ...], np.ndarray] = {}
    for address in product(*(range(value) for value in stride_values)):
        index = tuple(
            slice(residue, length, stride)
            for residue, length, stride in zip(address, array.shape, stride_values, strict=True)
        )
        channel = array[index]
        channels[address] = channel.copy() if copy else channel
    return LatticePolyphaseCoordinates(channels, stride_values, tuple(array.shape))


def lattice_polyphase_synthesize(coordinates: LatticePolyphaseCoordinates) -> np.ndarray:
    """Reconstruct an array from rectangular lattice packet coordinates."""

    return coordinates.reconstruct()


def mixed_radix_address(digits: Iterable[int], radices: Iterable[int]) -> int:
    """Return the least significant first mixed radix address."""

    digit_values = tuple(int(value) for value in digits)
    radix_values = tuple(_strict_positive_int(value, "radix") for value in radices)
    if len(digit_values) != len(radix_values):
        raise DimensionError("digits and radices must have the same length")
    address = 0
    place = 1
    for digit, radix in zip(digit_values, radix_values, strict=True):
        if digit < 0 or digit >= radix:
            raise ValidationError("a mixed radix digit lies outside its radix")
        address += place * digit
        place *= radix
    return address


def split_mixed_radix_address(address: int, radices: Iterable[int]) -> tuple[int, ...]:
    """Recover least significant first mixed radix digits."""

    value = int(address)
    if value < 0:
        raise ValidationError("address must be nonnegative")
    digits: list[int] = []
    total = 1
    for radix in tuple(_strict_positive_int(item, "radix") for item in radices):
        digits.append(value % radix)
        value //= radix
        total *= radix
    if value:
        raise ValidationError(f"address lies outside the mixed radix range 0 through {total - 1}")
    return tuple(digits)


def polyphase_roundtrip_certificate(
    signal: Any,
    order: int,
    *,
    axis: int = -1,
    tolerance: float = 1e-12,
) -> ComputationResult[np.ndarray]:
    """Analyze and reconstruct a finite signal with a numerical certificate."""

    coordinates = polyphase_analyze(signal, order, axis=axis)
    reconstructed = coordinates.reconstruct()
    return ComputationResult(reconstructed, coordinates.certificate(signal, tolerance=tolerance))


__all__ = [
    "LatticePolyphaseCoordinates",
    "PolyphaseCoordinates",
    "PolyphaseKernel",
    "PolyphaseSystem",
    "lattice_polyphase_analyze",
    "lattice_polyphase_synthesize",
    "mixed_radix_address",
    "polyphase_analyze",
    "polyphase_convolution",
    "polyphase_filter_kernels",
    "polyphase_roundtrip_certificate",
    "polyphase_synthesize",
    "refinement_children",
    "residue_project",
    "split_mixed_radix_address",
]
