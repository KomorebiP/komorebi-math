"""Typed certificates and result wrappers.

The package uses these classes to keep exact mathematics, numerical checks,
conditional constructions, heuristics, and planned features visibly distinct.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Rational, Real
from types import MappingProxyType
from typing import Any, Generic, TypeVar

import numpy as np

from .exceptions import ValidationError

T = TypeVar("T")


def _finite_real(value: Any) -> bool:
    """Check finite scalar evidence without rounding unbounded integers."""
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (Rational, np.integer)):
        return True
    if isinstance(value, np.floating):
        # math.isfinite converts through a C double and can overflow a finite
        # extended-precision NumPy scalar before testing it.
        return bool(np.isfinite(value))
    if isinstance(value, Decimal):
        return value.is_finite()
    if not isinstance(value, Real):
        # In particular, reject mutable zero-dimensional and length-one arrays.
        # Their implicit float conversion does not make owned scalar evidence.
        return False
    try:
        return math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False


class Exactness(StrEnum):
    EXACT = "exact"
    NUMERICAL = "numerical"
    CONDITIONAL = "conditional"
    HEURISTIC = "heuristic"
    PLANNED = "planned"


@dataclass(frozen=True, slots=True, eq=False)
class Certificate:
    """Machine readable validation record."""

    valid: bool
    exactness: Exactness
    residuals: dict[str, float] = field(default_factory=dict)
    tolerance: float | None = None
    assumptions_checked: tuple[str, ...] = ()
    assumptions_unchecked: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)
    theorem_id: str | None = None
    message: str = ""
    _recorded_valid: bool | None = field(default=None, init=False, repr=False, compare=False)
    _revalidation_required: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # NumPy predicates are genuine booleans, not truthy arbitrary objects.
        # Normalize them once so every retained validity field is a Python bool.
        if isinstance(self.valid, np.bool_):
            object.__setattr__(self, "valid", bool(self.valid))
        elif type(self.valid) is not bool:
            raise ValidationError("certificate valid must be a boolean")
        # The frozen record owns its residuals, so changing a caller's mapping
        # cannot silently change the evidence under an old true validity flag.
        object.__setattr__(self, "residuals", MappingProxyType(dict(self.residuals)))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))
        object.__setattr__(self, "assumptions_checked", tuple(self.assumptions_checked))
        object.__setattr__(self, "assumptions_unchecked", tuple(self.assumptions_unchecked))
        failures = [
            f"residual {name!r} must be finite"
            for name, value in self.residuals.items()
            if not _finite_real(value)
        ]
        if self.tolerance is not None and (
            not _finite_real(self.tolerance) or self.tolerance < 0
        ):
            failures.append("tolerance must be finite and nonnegative")
        if self.valid and failures:
            object.__setattr__(self, "valid", False)
            reason = "; ".join(failures)
            object.__setattr__(self, "message", f"{self.message}; {reason}" if self.message else reason)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Certificate):
            return NotImplemented
        return (
            self.recorded_valid == other.recorded_valid
            and self.exactness == other.exactness
            and self.residuals == other.residuals
            and self.tolerance == other.tolerance
            and self.assumptions_checked == other.assumptions_checked
            and self.assumptions_unchecked == other.assumptions_unchecked
            and self.diagnostics == other.diagnostics
            and self.theorem_id == other.theorem_id
            and self.message == other.message
        )

    @property
    def recorded_valid(self) -> bool:
        """Return the validity flag recorded by the producing computation.

        Fresh in-memory certificates report their live validity here. A
        certificate reconstructed from portable persistence retains the flag
        that was recorded at serialization time even though it is not treated
        as a freshly revalidated certificate.
        """

        return self.valid if self._recorded_valid is None else self._recorded_valid

    @property
    def revalidation_required(self) -> bool:
        """Whether this certificate was loaded as historical evidence."""

        return self._revalidation_required

    @property
    def current_valid(self) -> bool:
        """Return validity only when the certificate is live in this process."""

        return self.valid is True and not self._revalidation_required

    def require(self) -> "Certificate":
        """Require a live true boolean flag, not merely a persisted record.

        Conditional assumptions and producer-specific residual checks retain
        their own contracts. This method does not recompute a witness. Portable
        persistence intentionally reconstructs certificates as historical
        records; their producer must be rerun to obtain a live certificate.
        """
        if self._revalidation_required:
            raise ValidationError(
                "persisted certificate requires producer-specific revalidation"
            )
        if self.valid is not True:
            detail = self.message or "certificate validation failed"
            raise ValidationError(detail)
        return self

    def to_dict(self) -> dict[str, Any]:
        # Keep the established public dictionary schema. Private persistence
        # lifecycle fields are runtime provenance and are deliberately omitted.
        return {
            "valid": self.valid,
            "exactness": self.exactness.value,
            "residuals": dict(self.residuals),
            "tolerance": self.tolerance,
            "assumptions_checked": tuple(self.assumptions_checked),
            "assumptions_unchecked": tuple(self.assumptions_unchecked),
            "diagnostics": dict(self.diagnostics),
            "theorem_id": self.theorem_id,
            "message": self.message,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)


@dataclass(frozen=True, slots=True)
class ComputationResult(Generic[T]):
    """A value paired with its execution certificate."""

    value: T
    certificate: Certificate

    @property
    def valid(self) -> bool:
        return self.certificate.current_valid

    @property
    def exact(self) -> bool:
        return self.certificate.exactness is Exactness.EXACT

    def require(self) -> T:
        self.certificate.require()
        return self.value


@dataclass(frozen=True, slots=True)
class ConditionalResult(Generic[T]):
    """A value whose conclusion depends on explicit unchecked assumptions."""

    value: T
    certificate: Certificate

    def __post_init__(self) -> None:
        if self.certificate.exactness is not Exactness.CONDITIONAL:
            raise ValueError("ConditionalResult requires a conditional certificate")
