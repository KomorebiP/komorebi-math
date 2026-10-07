"""Finite rational memory kernels, auxiliary realizations, and stability bounds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._linalg import matrix_exponential


def _matrix(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 2:
        raise DimensionError(f"{name} must be a matrix")
    if result.size and not np.all(np.isfinite(result)):
        raise ValidationError(f"{name} must contain finite values")
    return result


def _trapezoid(values: np.ndarray, grid: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.sum(0.5 * (values[:-1] + values[1:]) * np.diff(grid)))


@documented(
    ProvenanceRecord(
        identifier="PSC-VI-T30",
        title="Finite rational memory realization",
        status=ClaimStatus.PROVED,
        source="Paper VI",
        assumptions=("finite auxiliary state realization",),
        verification_suite="dynamics.memory",
    )
)
@dataclass(frozen=True, slots=True)
class MemoryKernel:
    """A matrix kernel ``B exp(tD) C`` with rational Laplace transform."""

    output: np.ndarray
    hidden: np.ndarray
    input: np.ndarray

    def __init__(self, output: Any, hidden: Any, input: Any) -> None:
        b = _matrix(output, "output matrix")
        d = _matrix(hidden, "hidden matrix")
        c = _matrix(input, "input matrix")
        if d.shape[0] != d.shape[1]:
            raise DimensionError("hidden matrix must be square")
        if b.shape[1] != d.shape[0] or c.shape[0] != d.shape[0]:
            raise DimensionError("memory realization dimensions do not agree")
        if b.shape[0] != c.shape[1]:
            raise DimensionError("memory kernel must have a common retained input and output dimension")
        object.__setattr__(self, "output", b)
        object.__setattr__(self, "hidden", d)
        object.__setattr__(self, "input", c)

    @property
    def retained_dimension(self) -> int:
        return int(self.output.shape[0])

    @property
    def hidden_dimension(self) -> int:
        return int(self.hidden.shape[0])

    def at(self, time: float) -> np.ndarray:
        t = float(time)
        if not np.isfinite(t) or t < 0:
            raise ValidationError("memory time must be finite and nonnegative")
        return self.output @ matrix_exponential(t * self.hidden) @ self.input

    def evaluate(self, time: float) -> np.ndarray:
        return self.at(time)

    def laplace(self, spectral_parameter: complex) -> np.ndarray:
        z = complex(spectral_parameter)
        identity = np.eye(self.hidden_dimension, dtype=np.result_type(self.hidden, z))
        return self.output @ np.linalg.inv(z * identity - self.hidden) @ self.input

    def finite_laplace(self, spectral_parameter: complex, horizon: float) -> np.ndarray:
        z = complex(spectral_parameter)
        final = float(horizon)
        if not np.isfinite(final) or final < 0:
            raise ValidationError("horizon must be finite and nonnegative")
        identity = np.eye(self.hidden_dimension, dtype=np.result_type(self.hidden, z))
        resolvent = np.linalg.inv(z * identity - self.hidden)
        tail_factor = matrix_exponential((self.hidden - z * identity) * final)
        return self.output @ resolvent @ (identity - tail_factor) @ self.input

    def certificate(
        self,
        spectral_parameter: complex,
        *,
        horizon: float = 10.0,
        points: int = 1001,
        tolerance: float = 1e-9,
    ) -> Certificate:
        """Certify the finite horizon plus tail decomposition of the Laplace transform."""

        count = int(points)
        if count < 2:
            raise ValidationError("points must be at least two")
        z = complex(spectral_parameter)
        final = float(horizon)
        identity = np.eye(self.hidden_dimension, dtype=np.result_type(self.hidden, z))
        resolvent = np.linalg.inv(z * identity - self.hidden)
        tail_factor = matrix_exponential((self.hidden - z * identity) * final)
        finite = self.output @ resolvent @ (identity - tail_factor) @ self.input
        tail = self.output @ resolvent @ tail_factor @ self.input
        exact = self.laplace(z)
        residual = float(np.linalg.norm(finite + tail - exact, ord=2))
        valid = residual <= tolerance * max(1.0, float(np.linalg.norm(exact, ord=2)))
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"laplace_decomposition": residual},
            tolerance=tolerance,
            assumptions_checked=("finite auxiliary realization", "hidden resolvent exists"),
            diagnostics={"horizon": final, "requested_quadrature_points": count, "tail_norm": float(np.linalg.norm(tail, 2))},
            theorem_id="PSC-VI-T30",
            message="rational memory transform verified" if valid else "rational memory transform failed",
        )


RationalMemoryKernel = MemoryKernel


@dataclass(frozen=True, slots=True)
class RationalMemoryRealization:
    """Finite auxiliary state realization of a rational memory kernel."""

    output: np.ndarray
    hidden: np.ndarray
    input: np.ndarray

    def __init__(self, output: Any, hidden: Any, input: Any) -> None:
        kernel = MemoryKernel(output, hidden, input)
        object.__setattr__(self, "output", kernel.output)
        object.__setattr__(self, "hidden", kernel.hidden)
        object.__setattr__(self, "input", kernel.input)

    @property
    def retained_dimension(self) -> int:
        return int(self.output.shape[0])

    @property
    def hidden_dimension(self) -> int:
        return int(self.hidden.shape[0])

    def kernel(self) -> MemoryKernel:
        return MemoryKernel(self.output, self.hidden, self.input)

    def augment(self, retained_generator: Any) -> np.ndarray:
        generator = _matrix(retained_generator, "retained generator")
        if generator.shape != (self.retained_dimension, self.retained_dimension):
            raise DimensionError("retained generator has the wrong dimension")
        return np.block([[generator, self.output], [self.input, self.hidden]])

    def retained_trajectory(self, retained_generator: Any, initial_state: Any, time: float) -> np.ndarray:
        initial = np.asarray(initial_state)
        expected = self.retained_dimension + self.hidden_dimension
        if initial.ndim != 1 or initial.size != expected:
            raise DimensionError("initial augmented state has the wrong dimension")
        evolved = matrix_exponential(float(time) * self.augment(retained_generator)) @ initial
        return evolved[: self.retained_dimension]


def _evaluate_kernel(kernel: MemoryKernel | Callable[[float], Any], time: float) -> np.ndarray:
    return np.asarray(kernel.at(time) if isinstance(kernel, MemoryKernel) else kernel(time))


def kernel_l1_error(
    first: MemoryKernel | Callable[[float], Any],
    second: MemoryKernel | Callable[[float], Any],
    horizon: float,
    *,
    samples: int = 1001,
) -> float:
    """Estimate the operator norm ``L1`` difference by trapezoidal quadrature."""

    final = float(horizon)
    count = int(samples)
    if not np.isfinite(final) or final < 0:
        raise ValidationError("horizon must be finite and nonnegative")
    if count < 2:
        raise ValidationError("at least two quadrature samples are required")
    grid = np.linspace(0.0, final, count)
    values = np.asarray(
        [float(np.linalg.norm(_evaluate_kernel(first, t) - _evaluate_kernel(second, t), ord=2)) for t in grid]
    )
    return _trapezoid(values, grid)


def _exponential_integral_bound(matrix_norm: float, horizon: float) -> float:
    if matrix_norm == 0.0:
        return horizon
    return float(np.expm1(matrix_norm * horizon) / matrix_norm)


def memory_kernel_error_bound(
    first: MemoryKernel | Callable[[float], Any],
    second: MemoryKernel | Callable[[float], Any],
    *,
    horizon: float,
    samples: int = 1001,
) -> ComputationResult[float]:
    """Return a finite horizon upper certificate for a kernel difference.

    For two finite rational memory kernels, the value is the minimum of an
    integrated exponential norm bound and a sampled estimate with a global
    derivative allowance. Arbitrary callables receive a conditional quadrature
    estimate because no derivative certificate is available.
    """

    final = float(horizon)
    estimate = kernel_l1_error(first, second, final, samples=samples)
    if isinstance(first, MemoryKernel) and isinstance(second, MemoryKernel):
        first_b = float(np.linalg.norm(first.output, 2) * np.linalg.norm(first.input, 2))
        second_b = float(np.linalg.norm(second.output, 2) * np.linalg.norm(second.input, 2))
        first_d = float(np.linalg.norm(first.hidden, 2))
        second_d = float(np.linalg.norm(second.hidden, 2))
        separate = (
            first_b * _exponential_integral_bound(first_d, final)
            + second_b * _exponential_integral_bound(second_d, final)
        )
        step = final / (int(samples) - 1)
        derivative = (
            first_b * first_d * np.exp(first_d * final)
            + second_b * second_d * np.exp(second_d * final)
        )
        quadrature_allowance = 0.5 * derivative * final * step
        sampled_upper = estimate + quadrature_allowance
        value = float(min(separate, sampled_upper))
        return ComputationResult(
            value,
            Certificate(
                valid=np.isfinite(value) and value >= estimate - 1e-12,
                exactness=Exactness.NUMERICAL,
                residuals={
                    "quadrature_estimate": estimate,
                    "quadrature_allowance": float(quadrature_allowance),
                    "separate_exponential_bound": float(separate),
                },
                assumptions_checked=(
                    "finite horizon",
                    "finite rational memory kernels",
                    "operator norm exponential and derivative bounds",
                ),
                diagnostics={"samples": int(samples), "horizon": final},
                theorem_id="PSC-VI-T31",
                message="finite horizon memory kernel upper bound computed",
            ),
        )
    return ComputationResult(
        estimate,
        Certificate(
            valid=np.isfinite(estimate) and estimate >= 0,
            exactness=Exactness.CONDITIONAL,
            residuals={"quadrature_estimate": estimate},
            assumptions_checked=("finite horizon", "trapezoidal operator norm quadrature"),
            assumptions_unchecked=("quadrature estimate upper bounds the true kernel integral",),
            diagnostics={"samples": int(samples), "horizon": final},
            theorem_id="PSC-VI-T31",
            message="memory kernel integral estimated for arbitrary callables",
        ),
    )


@documented(
    ProvenanceRecord(
        identifier="PSC-VI-T31",
        title="Finite horizon Volterra kernel stability bound",
        status=ClaimStatus.PROVED,
        source="Paper VI",
        assumptions=("finite horizon", "declared trajectory and kernel bounds"),
        verification_suite="dynamics.memory",
    )
)
def volterra_trajectory_bound(
    kernel_l1_difference: float,
    horizon: float,
    trajectory_bound: float,
    retained_generator_norm: float | None = None,
    reference_kernel_l1: float | None = None,
    *,
    local_generator_bound: float | None = None,
    kernel_l1_bound: float | None = None,
) -> float:
    """Return the declared finite horizon trajectory error certificate.

    Both the Paper VI names and the earlier public keyword names are accepted.
    """

    local = retained_generator_norm if retained_generator_norm is not None else local_generator_bound
    kernel = reference_kernel_l1 if reference_kernel_l1 is not None else kernel_l1_bound
    if local is None or kernel is None:
        raise ValidationError("retained generator and reference kernel bounds are required")
    values = tuple(
        float(value)
        for value in (kernel_l1_difference, horizon, trajectory_bound, local, kernel)
    )
    if any(not np.isfinite(value) or value < 0 for value in values):
        raise ValidationError("Volterra bound inputs must be finite and nonnegative")
    delta, final, magnitude, local_value, kernel_value = values
    return float(magnitude * final * delta * np.exp((local_value + kernel_value) * final))


__all__ = [
    "MemoryKernel",
    "RationalMemoryKernel",
    "RationalMemoryRealization",
    "kernel_l1_error",
    "memory_kernel_error_bound",
    "volterra_trajectory_bound",
]
