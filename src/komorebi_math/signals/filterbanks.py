"""Finite periodic filter banks, frames, and lifting schemes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .convolution import convolution_matrix


def _filter_tuple(filters: Iterable[Any], name: str) -> tuple[np.ndarray, ...]:
    result = tuple(np.asarray(value) for value in filters)
    if not result:
        raise ValidationError(f"{name} cannot be empty")
    if any(value.ndim != 1 or value.size == 0 for value in result):
        raise DimensionError(f"every {name} entry must be a nonempty one dimensional filter")
    if any(not np.all(np.isfinite(value)) for value in result):
        raise ValidationError(f"every {name} entry must contain only finite values")
    return result


def _downsample_matrix(length: int, decimation: int) -> np.ndarray:
    if length % decimation:
        raise DimensionError("periodic signal length must be divisible by the decimation factor")
    result = np.zeros((length // decimation, length))
    result[np.arange(length // decimation), np.arange(0, length, decimation)] = 1.0
    return result


def _upsample_matrix(length: int, decimation: int) -> np.ndarray:
    return _downsample_matrix(length, decimation).T


@documented(
    ProvenanceRecord(
        identifier="PSC-V-D08",
        title="Finite periodic filter bank",
        status=ClaimStatus.DEFINITION,
        source="Paper V",
        assumptions=("finite periodic signal", "length divisible by decimation"),
        verification_suite="signals.filterbanks",
    )
)
@dataclass(frozen=True, slots=True)
class PeriodicFilterBank:
    """A finite periodic analysis and synthesis bank with explicit matrices."""

    analysis_filters: tuple[np.ndarray, ...]
    synthesis_filters: tuple[np.ndarray, ...]
    decimation: int

    def __init__(
        self,
        analysis_filters: Iterable[Any],
        synthesis_filters: Iterable[Any],
        decimation: int,
    ) -> None:
        analysis = _filter_tuple(analysis_filters, "analysis filter")
        synthesis = _filter_tuple(synthesis_filters, "synthesis filter")
        factor = int(decimation)
        if factor <= 0:
            raise ValidationError("decimation must be positive")
        if len(analysis) != len(synthesis):
            raise DimensionError("analysis and synthesis banks must have the same number of channels")
        object.__setattr__(self, "analysis_filters", analysis)
        object.__setattr__(self, "synthesis_filters", synthesis)
        object.__setattr__(self, "decimation", factor)

    @property
    def channels(self) -> int:
        return len(self.analysis_filters)

    def analysis_matrix(self, length: int) -> np.ndarray:
        down = _downsample_matrix(int(length), self.decimation)
        blocks = [down @ convolution_matrix(h, int(length), mode="circular") for h in self.analysis_filters]
        return np.vstack(blocks)

    def synthesis_matrix(self, length: int) -> np.ndarray:
        up = _upsample_matrix(int(length), self.decimation)
        blocks = [convolution_matrix(g, int(length), mode="circular") @ up for g in self.synthesis_filters]
        return np.hstack(blocks)

    def analyze(self, signal: Any) -> tuple[np.ndarray, ...]:
        x = np.asarray(signal)
        if x.ndim != 1:
            raise DimensionError("filter bank signals must be one dimensional")
        matrix = self.analysis_matrix(x.size)
        flat = matrix @ x
        block_length = x.size // self.decimation
        return tuple(flat[index * block_length : (index + 1) * block_length] for index in range(self.channels))

    def synthesize(self, coefficients: Iterable[Any], *, length: int | None = None) -> np.ndarray:
        values = tuple(np.asarray(value) for value in coefficients)
        if len(values) != self.channels:
            raise DimensionError("one coefficient channel is required for every synthesis filter")
        if any(value.ndim != 1 for value in values):
            raise DimensionError("filter bank coefficient channels must be one dimensional")
        channel_lengths = {value.size for value in values}
        if len(channel_lengths) != 1:
            raise DimensionError("all coefficient channels must have the same length")
        inferred = next(iter(channel_lengths)) * self.decimation
        signal_length = inferred if length is None else int(length)
        if signal_length != inferred:
            raise DimensionError("declared output length does not match coefficient channel lengths")
        return self.synthesis_matrix(signal_length) @ np.concatenate(values)

    def reconstruction_operator(self, length: int) -> np.ndarray:
        return self.synthesis_matrix(length) @ self.analysis_matrix(length)

    @documented(
        ProvenanceRecord(
            identifier="PSC-V-T09",
            title="Finite perfect reconstruction criterion",
            status=ClaimStatus.PROVED,
            source="Paper V",
            assumptions=("finite periodic bank", "matching analysis and synthesis matrices"),
            verification_suite="signals.filterbanks",
        )
    )
    def perfect_reconstruction_certificate(
        self,
        length: int,
        *,
        tolerance: float = 1e-10,
    ) -> Certificate:
        operator = self.reconstruction_operator(length)
        residual = float(np.linalg.norm(operator - np.eye(int(length)), ord=2))
        valid = residual <= tolerance
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"reconstruction_operator": residual},
            tolerance=tolerance,
            assumptions_checked=("finite periodic convention", "matching channel count"),
            theorem_id="PSC-V-T09",
            message="perfect reconstruction verified" if valid else "filter bank is not perfectly reconstructing",
        )

    def frame_bounds(self, length: int) -> tuple[float, float]:
        analysis = self.analysis_matrix(length)
        singular_values = np.linalg.svd(analysis, compute_uv=False)
        if singular_values.size == 0:
            return 0.0, 0.0
        lower = 0.0 if analysis.shape[0] < analysis.shape[1] else float(singular_values[-1] ** 2)
        return lower, float(singular_values[0] ** 2)

    def canonical_synthesis_matrix(self, length: int, *, rcond: float | None = None) -> np.ndarray:
        """Return the Moore Penrose canonical dual synthesis matrix."""

        analysis = self.analysis_matrix(length)
        if rcond is None:
            return np.linalg.pinv(analysis)
        return np.linalg.pinv(analysis, rcond=rcond)

    def reconstruction_error_bound(self, length: int) -> float:
        """Return the operator norm of the reconstruction defect."""

        return float(np.linalg.norm(self.reconstruction_operator(length) - np.eye(length), ord=2))


@documented(
    ProvenanceRecord(
        identifier="PSC-V-T10",
        title="Orthonormal Haar reconstruction",
        status=ClaimStatus.PROVED,
        source="Paper V",
        assumptions=("even finite signal length",),
        verification_suite="signals.filterbanks",
    )
)
@dataclass(frozen=True, slots=True)
class HaarFilterBank:
    """Exact pairwise orthonormal Haar analysis and synthesis."""

    normalized: bool = True

    def analyze(self, signal: Any) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(signal)
        if x.ndim != 1 or x.size % 2:
            raise DimensionError("Haar analysis requires an even one dimensional signal")
        if not np.all(np.isfinite(x)):
            raise ValidationError("Haar signal entries must be finite")
        x = x.astype(np.result_type(x.dtype, float), copy=False)
        even = x[0::2]
        odd = x[1::2]
        scale = np.sqrt(2.0) if self.normalized else 1.0
        return (even + odd) / scale, (even - odd) / scale

    def synthesize(self, coefficients: Iterable[Any]) -> np.ndarray:
        channels = tuple(np.asarray(value) for value in coefficients)
        if len(channels) != 2:
            raise DimensionError("Haar synthesis requires two coefficient channels")
        low, high = channels
        if low.ndim != 1 or high.ndim != 1 or low.shape != high.shape:
            raise DimensionError("Haar coefficient channels must be equal one dimensional arrays")
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
            raise ValidationError("Haar coefficient entries must be finite")
        dtype = np.result_type(low.dtype, high.dtype, float)
        low = low.astype(dtype, copy=False)
        high = high.astype(dtype, copy=False)
        scale = np.sqrt(2.0) if self.normalized else 2.0
        result = np.empty(low.size * 2, dtype=dtype)
        result[0::2] = (low + high) / scale
        result[1::2] = (low - high) / scale
        return result

    def certificate(self, signal: Any, *, tolerance: float = 1e-12) -> ComputationResult[np.ndarray]:
        x = np.asarray(signal)
        x = x.astype(np.result_type(x.dtype, float), copy=False)
        reconstructed = self.synthesize(self.analyze(x))
        residual = float(np.linalg.norm(reconstructed - x))
        energy = float(np.vdot(x, x).real)
        energy_residual = abs(
            energy
            - sum(float(np.vdot(channel, channel).real) for channel in self.analyze(x))
        )
        reconstruction_scale = max(1.0, float(np.linalg.norm(x)))
        energy_scale = max(1.0, energy)
        valid = (
            np.isfinite(residual)
            and np.isfinite(energy_residual)
            and np.isfinite(reconstruction_scale)
            and np.isfinite(energy_scale)
            and residual <= tolerance * reconstruction_scale
            and (not self.normalized or energy_residual <= tolerance * energy_scale)
        )
        return ComputationResult(
            reconstructed,
            Certificate(
                valid=valid,
                exactness=Exactness.NUMERICAL,
                residuals={"reconstruction": residual, "energy": energy_residual},
                tolerance=tolerance,
                assumptions_checked=("even finite signal",),
                theorem_id="PSC-V-T10",
                message="Haar reconstruction verified" if valid else "Haar reconstruction failed",
            ),
        )


def _circular_filter(signal: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    result = np.zeros_like(signal, dtype=np.result_type(signal.dtype, coefficients.dtype, float))
    for offset, coefficient in enumerate(coefficients):
        result += coefficient * np.roll(signal, offset)
    return result


@documented(
    ProvenanceRecord(
        identifier="PSC-V-T11",
        title="Exact periodic lifting update",
        status=ClaimStatus.PROVED,
        source="Paper V",
        assumptions=("even finite periodic signal",),
        verification_suite="signals.filterbanks",
    )
)
@dataclass(frozen=True, slots=True)
class PeriodicLiftingScheme:
    """A reversible predict and update lifting scheme on periodic pairs."""

    predict: np.ndarray
    update: np.ndarray

    def __init__(self, predict: Any, update: Any) -> None:
        predict_array = np.asarray(predict)
        update_array = np.asarray(update)
        if predict_array.ndim != 1 or predict_array.size == 0:
            raise DimensionError("predict filter must be a nonempty one dimensional array")
        if update_array.ndim != 1 or update_array.size == 0:
            raise DimensionError("update filter must be a nonempty one dimensional array")
        object.__setattr__(self, "predict", predict_array)
        object.__setattr__(self, "update", update_array)

    def analyze(self, signal: Any) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(signal)
        if x.ndim != 1 or x.size % 2:
            raise DimensionError("lifting analysis requires an even one dimensional signal")
        even = x[0::2].astype(np.result_type(x.dtype, float), copy=True)
        odd = x[1::2].astype(np.result_type(x.dtype, float), copy=True)
        detail = odd - _circular_filter(even, self.predict)
        smooth = even + _circular_filter(detail, self.update)
        return smooth, detail

    def synthesize(self, coefficients: Iterable[Any] | Any, detail_channel: Any | None = None) -> np.ndarray:
        values = (coefficients, detail_channel) if detail_channel is not None else tuple(coefficients)
        smooth, detail = tuple(np.asarray(value) for value in values)
        if smooth.ndim != 1 or detail.ndim != 1 or smooth.shape != detail.shape:
            raise DimensionError("lifting coefficient channels must be equal one dimensional arrays")
        even = smooth - _circular_filter(detail, self.update)
        odd = detail + _circular_filter(even, self.predict)
        result = np.empty(even.size * 2, dtype=np.result_type(even.dtype, odd.dtype))
        result[0::2] = even
        result[1::2] = odd
        return result

    def certificate(self, signal: Any, *, tolerance: float = 1e-11) -> ComputationResult[np.ndarray]:
        x = np.asarray(signal)
        reconstructed = self.synthesize(self.analyze(x))
        residual = float(np.linalg.norm(reconstructed - x))
        valid = residual <= tolerance
        return ComputationResult(
            reconstructed,
            Certificate(
                valid=valid,
                exactness=Exactness.NUMERICAL,
                residuals={"lifting_roundtrip": residual},
                tolerance=tolerance,
                assumptions_checked=("periodic boundary convention", "even finite signal"),
                theorem_id="PSC-V-T11",
                message="lifting roundtrip verified" if valid else "lifting roundtrip failed",
            ),
        )


@dataclass(frozen=True, slots=True)
class FilterBankReport:
    """Finite reconstruction and frame diagnostics for a filter bank."""

    lower_frame_bound: float
    upper_frame_bound: float
    reconstruction_residual: float
    certificate: Certificate


@dataclass(frozen=True, slots=True)
class FiniteFilterBank:
    """A finite analysis matrix together with a declared synthesis matrix."""

    analysis_matrix: np.ndarray
    synthesis_matrix: np.ndarray

    def __init__(self, analysis_matrix: Any, synthesis_matrix: Any) -> None:
        analysis = np.asarray(analysis_matrix)
        synthesis = np.asarray(synthesis_matrix)
        if analysis.ndim != 2 or synthesis.ndim != 2:
            raise DimensionError("analysis and synthesis operators must be matrices")
        if synthesis.shape != (analysis.shape[1], analysis.shape[0]):
            raise DimensionError("synthesis matrix shape must be the transpose shape of analysis")
        if not np.all(np.isfinite(analysis)) or not np.all(np.isfinite(synthesis)):
            raise ValidationError("analysis and synthesis matrices must contain only finite values")
        object.__setattr__(self, "analysis_matrix", analysis)
        object.__setattr__(self, "synthesis_matrix", synthesis)

    @property
    def input_dimension(self) -> int:
        return int(self.analysis_matrix.shape[1])

    @property
    def coefficient_dimension(self) -> int:
        return int(self.analysis_matrix.shape[0])

    def analyze(self, signal: Any) -> np.ndarray:
        value = np.asarray(signal)
        if value.ndim != 1 or value.size != self.input_dimension:
            raise DimensionError("signal dimension does not match the filter bank")
        return self.analysis_matrix @ value

    def synthesize(self, coefficients: Any) -> np.ndarray:
        value = np.asarray(coefficients)
        if value.ndim != 1 or value.size != self.coefficient_dimension:
            raise DimensionError("coefficient dimension does not match the filter bank")
        return self.synthesis_matrix @ value

    def reconstruction_operator(self) -> np.ndarray:
        return self.synthesis_matrix @ self.analysis_matrix

    def report(self, *, tolerance: float = 1e-10) -> FilterBankReport:
        singular_values = np.linalg.svd(self.analysis_matrix, compute_uv=False)
        lower = (
            float(singular_values[-1] ** 2)
            if singular_values.size and self.coefficient_dimension >= self.input_dimension
            else 0.0
        )
        upper = float(singular_values[0] ** 2) if singular_values.size else 0.0
        residual = float(np.linalg.norm(self.reconstruction_operator() - np.eye(self.input_dimension), 2))
        valid = residual <= tolerance
        return FilterBankReport(
            lower,
            upper,
            residual,
            Certificate(
                valid=valid,
                exactness=Exactness.NUMERICAL,
                residuals={"reconstruction": residual},
                tolerance=tolerance,
                assumptions_checked=("finite analysis matrix", "declared synthesis matrix"),
                theorem_id="PSC-V-T09",
                message="finite filter bank reconstructs exactly" if valid else "finite filter bank has a reconstruction defect",
            ),
        )

    def canonical_dual(self, *, rcond: float | None = None) -> "FiniteFilterBank":
        if np.linalg.matrix_rank(self.analysis_matrix) < self.input_dimension:
            raise ValidationError("canonical dual requires a full column rank analysis matrix")
        if rcond is None:
            synthesis = np.linalg.pinv(self.analysis_matrix)
        else:
            synthesis = np.linalg.pinv(self.analysis_matrix, rcond=rcond)
        return FiniteFilterBank(self.analysis_matrix, synthesis)

    def perturbation_bound(self, analysis_error: Any, synthesis_error: Any) -> float:
        delta_e = np.asarray(analysis_error)
        delta_r = np.asarray(synthesis_error)
        if delta_e.shape != self.analysis_matrix.shape or delta_r.shape != self.synthesis_matrix.shape:
            raise DimensionError("perturbation matrices must match the analysis and synthesis shapes")
        base = float(np.linalg.norm(self.reconstruction_operator() - np.eye(self.input_dimension), 2))
        return float(
            base
            + np.linalg.norm(delta_r, 2) * np.linalg.norm(self.analysis_matrix, 2)
            + np.linalg.norm(self.synthesis_matrix, 2) * np.linalg.norm(delta_e, 2)
            + np.linalg.norm(delta_r, 2) * np.linalg.norm(delta_e, 2)
        )


def haar_analyze(signal: Any, *, normalized: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Split a finite even length signal into one level Haar low and high channels."""

    return HaarFilterBank(normalized=normalized).analyze(signal)


def haar_synthesize(low: Any, high: Any, *, normalized: bool = True) -> np.ndarray:
    """Reconstruct a signal from one level Haar low and high channels."""

    return HaarFilterBank(normalized=normalized).synthesize((low, high))


def haar_matrix(length: int, *, normalized: bool = True) -> np.ndarray:
    """Return the finite one level Haar analysis matrix for an even signal length."""

    size = int(length)
    if size <= 0 or size % 2:
        raise DimensionError("Haar matrix length must be a positive even integer")
    bank = HaarFilterBank(normalized=normalized)
    columns = []
    for index in range(size):
        basis = np.zeros(size)
        basis[index] = 1.0
        low, high = bank.analyze(basis)
        columns.append(np.concatenate((low, high)))
    return np.column_stack(columns)


def haar_filter_bank(length: int, *, normalized: bool = True) -> FiniteFilterBank:
    """Construct the finite one level Haar analysis and synthesis matrices."""

    analysis = haar_matrix(length, normalized=normalized)
    synthesis = np.linalg.inv(analysis)
    return FiniteFilterBank(analysis, synthesis)


def reconstruction_error(bank: FiniteFilterBank) -> float:
    """Return the spectral norm of the finite filter bank reconstruction defect."""

    return float(np.linalg.norm(bank.reconstruction_operator() - np.eye(bank.input_dimension), 2))


def periodic_five_three_lifting() -> PeriodicLiftingScheme:
    """Return the floating point periodic five three lifting scheme."""

    return PeriodicLiftingScheme(np.array([0.5, 0.5]), np.array([0.25, 0.25]))


__all__ = [
    "FilterBankReport",
    "FiniteFilterBank",
    "HaarFilterBank",
    "PeriodicFilterBank",
    "PeriodicLiftingScheme",
    "haar_analyze",
    "haar_filter_bank",
    "haar_matrix",
    "haar_synthesize",
    "periodic_five_three_lifting",
    "reconstruction_error",
]
