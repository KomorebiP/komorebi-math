"""Finite LMS, normalized LMS, and recursive least-squares adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented

AdaptiveMethod = Literal["lms", "nlms", "rls"]


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValidationError(f"{name} must be a positive integer")
    return int(value)


def _paired(input_signal: Any, desired_signal: Any) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(input_signal)
    desired = np.asarray(desired_signal)
    if source.ndim != 1 or desired.ndim != 1 or source.size == 0 or desired.size == 0:
        raise DimensionError("adaptive filtering requires nonempty one dimensional signals")
    if source.shape != desired.shape:
        raise DimensionError("input and desired signals must have equal length")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(desired)):
        raise ValidationError("adaptive-filter signal entries must be finite")
    return source, desired


def _initial_weights(initial_weights: Any | None, taps: int, dtype: np.dtype[Any]) -> np.ndarray:
    if initial_weights is None:
        return np.zeros(taps, dtype=dtype)
    value = np.asarray(initial_weights, dtype=dtype)
    if value.ndim != 1 or value.size != taps:
        raise DimensionError("initial_weights must have length num_taps")
    if not np.all(np.isfinite(value)):
        raise ValidationError("initial_weights entries must be finite")
    return value.copy()


def _regressor(source: np.ndarray, index: int, taps: int, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.zeros(taps, dtype=dtype)
    available = min(taps, index + 1)
    result[:available] = source[index - np.arange(available)]
    return result


@dataclass(frozen=True, slots=True)
class AdaptiveFilterResult:
    """Finite adaptive-filter outputs, errors, and coefficient trajectory."""

    output: np.ndarray
    error: np.ndarray
    final_weights: np.ndarray
    weight_history: np.ndarray
    method: AdaptiveMethod
    certificate: Certificate

    @property
    def mean_squared_error(self) -> float:
        """Return the finite mean squared adaptation error."""

        return float(np.mean(np.abs(self.error) ** 2))

    def tail_mean_squared_error(self, fraction: float = 0.25) -> float:
        """Return the mean squared error over the final positive fraction."""

        portion = float(fraction)
        if not np.isfinite(portion) or portion <= 0 or portion > 1:
            raise ValidationError("fraction must lie in (0, 1]")
        count = max(1, int(np.ceil(self.error.size * portion)))
        return float(np.mean(np.abs(self.error[-count:]) ** 2))


def _result(
    output: np.ndarray,
    error: np.ndarray,
    weights: np.ndarray,
    history: np.ndarray,
    method: AdaptiveMethod,
    *,
    diagnostics: dict[str, float | int],
) -> AdaptiveFilterResult:
    finite = bool(
        np.all(np.isfinite(output))
        and np.all(np.isfinite(error))
        and np.all(np.isfinite(weights))
        and np.all(np.isfinite(history))
    )
    initial_count = max(1, error.size // 4)
    final_count = max(1, error.size // 4)
    initial_mse = float(np.mean(np.abs(error[:initial_count]) ** 2))
    final_mse = float(np.mean(np.abs(error[-final_count:]) ** 2))
    certificate = Certificate(
        valid=finite,
        exactness=Exactness.NUMERICAL,
        residuals={"nonfinite_entries": 0.0 if finite else 1.0},
        assumptions_checked=("finite recursion", "declared parameter ranges"),
        assumptions_unchecked=(
            "persistent excitation",
            "stochastic independence assumptions",
            "convergence for the supplied step or forgetting factor",
        ),
        diagnostics={
            **diagnostics,
            "initial_quarter_mse": initial_mse,
            "final_quarter_mse": final_mse,
        },
        theorem_id="KM-DSP-D12",
        message=f"finite {method.upper()} recursion completed" if finite else f"{method.upper()} recursion diverged",
    )
    return AdaptiveFilterResult(output, error, weights, history, method, certificate)


@documented(
    ProvenanceRecord(
        identifier="KM-DSP-D12",
        title="Finite least-mean-squares adaptation",
        status=ClaimStatus.CLASSICAL,
        source="Block 12",
        assumptions=("finite paired signals", "positive step size"),
        limitations=("the finite routine does not assert stochastic convergence",),
        verification_suite="signals.adaptive",
    )
)
def lms_filter(
    input_signal: Any,
    desired_signal: Any,
    num_taps: int,
    *,
    step_size: float = 0.01,
    leakage: float = 0.0,
    initial_weights: Any | None = None,
) -> AdaptiveFilterResult:
    """Run finite real or complex LMS system identification."""

    source, desired = _paired(input_signal, desired_signal)
    taps = _positive_integer(num_taps, "num_taps")
    step = float(step_size)
    leak = float(leakage)
    if not np.isfinite(step) or step <= 0:
        raise ValidationError("step_size must be positive and finite")
    if not np.isfinite(leak) or leak < 0:
        raise ValidationError("leakage must be finite and nonnegative")
    dtype = np.dtype(np.result_type(source.dtype, desired.dtype, float))
    weights = _initial_weights(initial_weights, taps, dtype)
    output = np.zeros(source.size, dtype=dtype)
    error = np.zeros(source.size, dtype=dtype)
    history = np.zeros((source.size + 1, taps), dtype=dtype)
    history[0] = weights
    retention = 1.0 - step * leak
    for index in range(source.size):
        regressor = _regressor(source, index, taps, dtype)
        output[index] = np.vdot(weights, regressor)
        error[index] = desired[index] - output[index]
        weights = retention * weights + step * regressor * np.conjugate(error[index])
        history[index + 1] = weights
    return _result(
        output,
        error,
        weights,
        history,
        "lms",
        diagnostics={"step_size": step, "leakage": leak, "num_taps": taps},
    )


@documented(
    ProvenanceRecord(
        identifier="KM-DSP-D13",
        title="Finite normalized least-mean-squares adaptation",
        status=ClaimStatus.CLASSICAL,
        source="Block 12",
        assumptions=("finite paired signals", "positive normalization floor"),
        limitations=("the finite routine does not assert stochastic convergence",),
        verification_suite="signals.adaptive",
    )
)
def nlms_filter(
    input_signal: Any,
    desired_signal: Any,
    num_taps: int,
    *,
    step_size: float = 0.5,
    epsilon: float = 1e-8,
    leakage: float = 0.0,
    initial_weights: Any | None = None,
) -> AdaptiveFilterResult:
    """Run finite normalized LMS with an explicit denominator floor."""

    source, desired = _paired(input_signal, desired_signal)
    taps = _positive_integer(num_taps, "num_taps")
    step = float(step_size)
    floor = float(epsilon)
    leak = float(leakage)
    if not np.isfinite(step) or step <= 0:
        raise ValidationError("step_size must be positive and finite")
    if not np.isfinite(floor) or floor <= 0:
        raise ValidationError("epsilon must be positive and finite")
    if not np.isfinite(leak) or leak < 0:
        raise ValidationError("leakage must be finite and nonnegative")
    dtype = np.dtype(np.result_type(source.dtype, desired.dtype, float))
    weights = _initial_weights(initial_weights, taps, dtype)
    output = np.zeros(source.size, dtype=dtype)
    error = np.zeros(source.size, dtype=dtype)
    history = np.zeros((source.size + 1, taps), dtype=dtype)
    history[0] = weights
    for index in range(source.size):
        regressor = _regressor(source, index, taps, dtype)
        output[index] = np.vdot(weights, regressor)
        error[index] = desired[index] - output[index]
        denominator = floor + float(np.vdot(regressor, regressor).real)
        retention = 1.0 - step * leak / denominator
        weights = retention * weights + (step / denominator) * regressor * np.conjugate(error[index])
        history[index + 1] = weights
    return _result(
        output,
        error,
        weights,
        history,
        "nlms",
        diagnostics={
            "step_size": step,
            "epsilon": floor,
            "leakage": leak,
            "num_taps": taps,
        },
    )


@documented(
    ProvenanceRecord(
        identifier="KM-DSP-D14",
        title="Finite recursive least-squares adaptation",
        status=ClaimStatus.CLASSICAL,
        source="Block 12",
        assumptions=("finite paired signals", "positive inverse-covariance initialization", "forgetting factor in (0, 1]"),
        limitations=("the finite routine does not assert a stochastic identification theorem",),
        verification_suite="signals.adaptive",
    )
)
def rls_filter(
    input_signal: Any,
    desired_signal: Any,
    num_taps: int,
    *,
    forgetting_factor: float = 0.99,
    delta: float = 1.0,
    initial_weights: Any | None = None,
) -> AdaptiveFilterResult:
    """Run finite real or complex recursive least squares."""

    source, desired = _paired(input_signal, desired_signal)
    taps = _positive_integer(num_taps, "num_taps")
    forgetting = float(forgetting_factor)
    initialization = float(delta)
    if not np.isfinite(forgetting) or forgetting <= 0 or forgetting > 1:
        raise ValidationError("forgetting_factor must lie in (0, 1]")
    if not np.isfinite(initialization) or initialization <= 0:
        raise ValidationError("delta must be positive and finite")
    dtype = np.dtype(np.result_type(source.dtype, desired.dtype, float))
    weights = _initial_weights(initial_weights, taps, dtype)
    inverse_covariance = np.eye(taps, dtype=dtype) * initialization
    output = np.zeros(source.size, dtype=dtype)
    error = np.zeros(source.size, dtype=dtype)
    history = np.zeros((source.size + 1, taps), dtype=dtype)
    history[0] = weights
    maximum_hermitian_residual = 0.0
    for index in range(source.size):
        regressor = _regressor(source, index, taps, dtype)
        projected = inverse_covariance @ regressor
        denominator = forgetting + np.vdot(regressor, projected)
        if abs(complex(denominator)) <= np.finfo(float).eps:
            raise ValidationError("RLS gain denominator vanished")
        gain = projected / denominator
        output[index] = np.vdot(weights, regressor)
        error[index] = desired[index] - output[index]
        weights = weights + gain * np.conjugate(error[index])
        row = np.conjugate(regressor) @ inverse_covariance
        inverse_covariance = (inverse_covariance - np.outer(gain, row)) / forgetting
        inverse_covariance = 0.5 * (inverse_covariance + np.conjugate(inverse_covariance.T))
        maximum_hermitian_residual = max(
            maximum_hermitian_residual,
            float(np.max(np.abs(inverse_covariance - np.conjugate(inverse_covariance.T)))),
        )
        history[index + 1] = weights
    return _result(
        output,
        error,
        weights,
        history,
        "rls",
        diagnostics={
            "forgetting_factor": forgetting,
            "delta": initialization,
            "num_taps": taps,
            "maximum_hermitian_residual": maximum_hermitian_residual,
        },
    )


__all__ = [
    "AdaptiveFilterResult",
    "AdaptiveMethod",
    "lms_filter",
    "nlms_filter",
    "rls_filter",
]
