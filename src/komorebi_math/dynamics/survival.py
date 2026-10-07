"""Finite phase type survival and competing risk packets."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Any

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .semigroups import matrix_exponential


def _initial_probability(value: Any, size: int) -> np.ndarray:
    probability = np.asarray(value, dtype=float)
    if probability.ndim != 1 or probability.size != size:
        raise DimensionError("initial probability does not match the transient generator")
    if np.any(probability < 0) or not np.all(np.isfinite(probability)):
        raise ValidationError("initial probability must be finite and nonnegative")
    total = float(np.sum(probability))
    if total <= 0:
        raise ValidationError("initial probability must have positive mass")
    return probability / total


@documented(
    ProvenanceRecord(
        identifier="PSC-VI-T32",
        title="Finite state dependent killing survival model",
        status=ClaimStatus.PROVED,
        source="Paper VI",
        assumptions=("finite transient column subgenerator",),
        verification_suite="dynamics.survival",
    )
)
@dataclass(frozen=True, slots=True)
class PhaseTypeSurvival:
    """A finite transient subgenerator acting on column probabilities."""

    subgenerator: np.ndarray
    initial: np.ndarray
    cause_rates: np.ndarray | None = None
    tolerance: float = 1e-12

    def __init__(
        self,
        subgenerator: Any,
        initial: Any,
        *,
        cause_rates: Any | None = None,
        tolerance: float = 1e-12,
        validate: bool = True,
    ) -> None:
        q = np.asarray(subgenerator, dtype=float)
        if q.ndim != 2 or q.shape[0] != q.shape[1]:
            raise DimensionError("transient subgenerator must be square")
        p = _initial_probability(initial, q.shape[0])
        causes = None if cause_rates is None else np.asarray(cause_rates, dtype=float)
        if causes is not None:
            if causes.ndim != 2 or causes.shape[1] != q.shape[0]:
                raise DimensionError("cause rates must have shape (causes, transient states)")
            if np.any(causes < 0) or not np.all(np.isfinite(causes)):
                raise ValidationError("cause rates must be finite and nonnegative")
        tol = float(tolerance)
        if not np.isfinite(tol) or tol <= 0:
            raise ValidationError("tolerance must be finite and positive")
        object.__setattr__(self, "subgenerator", q)
        object.__setattr__(self, "initial", p)
        object.__setattr__(self, "cause_rates", causes)
        object.__setattr__(self, "tolerance", tol)
        if validate:
            self.certificate().require()

    @property
    def states(self) -> int:
        return self.subgenerator.shape[0]

    @property
    def absorption_rates(self) -> np.ndarray:
        return -np.sum(self.subgenerator, axis=0)

    def certificate(self) -> Certificate:
        q = self.subgenerator
        off_diagonal = q.copy()
        np.fill_diagonal(off_diagonal, 0.0)
        negative = max(0.0, -float(np.min(off_diagonal)))
        rates = self.absorption_rates
        rate_negative = max(0.0, -float(np.min(rates)))
        cause_residual = 0.0
        if self.cause_rates is not None:
            cause_residual = float(np.max(np.abs(np.sum(self.cause_rates, axis=0) - rates)))
        spectral_abscissa = float(np.max(np.real(np.linalg.eigvals(q))))
        stability_failure = max(0.0, spectral_abscissa + self.tolerance)
        valid = (
            max(negative, rate_negative, cause_residual) <= self.tolerance
            and spectral_abscissa < -self.tolerance
        )
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={
                "negative_off_diagonal": negative,
                "negative_absorption_rate": rate_negative,
                "cause_rate_sum": cause_residual,
                "spectral_abscissa_failure": stability_failure,
            },
            tolerance=self.tolerance,
            assumptions_checked=("column transient convention", "nonnegative off diagonal rates", "strictly stable transient generator"),
            theorem_id="PSC-VI-T32",
            message="valid finite phase type survival model" if valid else "invalid transient survival model",
        )

    def transient_probability(self, time: float) -> np.ndarray:
        t = float(time)
        if not np.isfinite(t) or t < 0:
            raise ValidationError("survival time must be finite and nonnegative")
        return np.real_if_close(matrix_exponential(t * self.subgenerator) @ self.initial).astype(float)

    def survival(self, time: float) -> float:
        return float(np.sum(self.transient_probability(time)))

    def density(self, time: float) -> float:
        return float(np.dot(self.absorption_rates, self.transient_probability(time)))

    def hazard(self, time: float) -> float:
        survival = self.survival(time)
        if survival <= self.tolerance:
            return float("inf")
        return self.density(time) / survival

    def laplace_transform(self, parameter: float) -> float:
        s = float(parameter)
        if not np.isfinite(s) or s < 0:
            raise ValidationError("Laplace parameter must be finite and nonnegative")
        resolvent = np.linalg.inv(s * np.eye(self.states) - self.subgenerator)
        return float(self.absorption_rates @ resolvent @ self.initial)

    def moment(self, order: int) -> float:
        value = int(order)
        if value != order or value < 0:
            raise ValidationError("moment order must be a nonnegative integer")
        if value == 0:
            return 1.0
        inverse = np.linalg.inv(-self.subgenerator)
        power = np.linalg.matrix_power(inverse, value)
        return float(factorial(value) * np.ones(self.states) @ power @ self.initial)

    def cause_density(self, time: float) -> np.ndarray:
        if self.cause_rates is None:
            raise ValidationError("cause specific rates were not supplied")
        return self.cause_rates @ self.transient_probability(time)

    def residual_initial(self, time: float) -> np.ndarray:
        transient = self.transient_probability(time)
        survival = float(np.sum(transient))
        if survival <= self.tolerance:
            raise ValidationError("survival probability is zero at the requested restart time")
        return transient / survival

    def residual_model(self, time: float) -> "PhaseTypeSurvival":
        return PhaseTypeSurvival(
            self.subgenerator,
            self.residual_initial(time),
            cause_rates=self.cause_rates,
            tolerance=self.tolerance,
        )

    def identities_certificate(self, time: float, *, step: float = 1e-6) -> Certificate:
        t = float(time)
        h = float(step)
        if t < h or not np.isfinite(t) or not np.isfinite(h) or h <= 0:
            raise ValidationError("identity check requires finite time greater than the positive step")
        derivative = (self.survival(t + h) - self.survival(t - h)) / (2.0 * h)
        density_residual = abs(-derivative - self.density(t))
        cause_residual = 0.0
        if self.cause_rates is not None:
            cause_residual = abs(float(np.sum(self.cause_density(t))) - self.density(t))
        valid = max(density_residual, cause_residual) <= 1e-5
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"density_derivative": density_residual, "cause_sum": cause_residual},
            tolerance=1e-5,
            assumptions_checked=("finite difference derivative check", "valid survival model"),
            theorem_id="PSC-VI-T33",
            message="survival identities verified" if valid else "survival identity check failed",
        )


__all__ = ["PhaseTypeSurvival"]
