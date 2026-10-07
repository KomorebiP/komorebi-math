"""Research-tier carrier-change obstruction packets.

Episode 011 separates *state compression* from *carrier translation*.  A packet
calculus may aggregate alternative histories with addition, maximum, convex
hull, Boolean OR, or another operation.  A map between carriers is semantics
preserving only when it commutes with the operations required by the downstream
query.

This module provides two small exact completions used by the Episode 011 audit:

* ``SignedMassPacket`` retains positive and negative mass separately, exposing
  cancellation that any zerosumfree/idempotent target necessarily forgets.
* ``TropicalResidualPacket`` refines max-plus data with the exact log-sum-exp
  correction.  Its tropical projection composes by max/plus while its full
  observation reconstructs finite-temperature sum-product semantics.

The module is intentionally research-tier and is not re-exported from the
stable aggregate API.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Iterable

from ..exceptions import ValidationError


def _finite_real(value: Any, name: str) -> Real:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValidationError(f"{name} must be a finite real number")
    if not math.isfinite(float(value)):
        raise ValidationError(f"{name} must be a finite real number")
    return value


def _finite_nonnegative(value: Any, name: str) -> Real:
    result = _finite_real(value, name)
    if result < 0:
        raise ValidationError(f"{name} must be nonnegative")
    return result


def boolean_support_nonnegative(value: Any) -> bool:
    """Boolean support map on a nonnegative scalar carrier.

    On a zerosumfree domain such as the nonnegative reals, this map preserves
    ordinary addition/multiplication as Boolean OR/AND.  Negative inputs are
    rejected because cancellation destroys that property.
    """

    result = _finite_nonnegative(value, "value")
    return bool(result != 0)


@dataclass(frozen=True, slots=True)
class SignedMassPacket:
    """Two-channel exact completion for a finite signed scalar decomposition.

    ``positive`` is the total positive mass and ``negative`` is the magnitude
    of the total negative mass.  The exact signed observation is
    ``positive - negative``.  The packet deliberately remembers cancellation
    potential that unsigned/idempotent projections discard.
    """

    positive: Real
    negative: Real

    def __post_init__(self) -> None:
        object.__setattr__(self, "positive", _finite_nonnegative(self.positive, "positive"))
        object.__setattr__(self, "negative", _finite_nonnegative(self.negative, "negative"))

    @classmethod
    def from_scalar(cls, value: Any) -> "SignedMassPacket":
        scalar = _finite_real(value, "value")
        if scalar >= 0:
            return cls(scalar, 0)
        return cls(0, -scalar)

    @classmethod
    def from_values(cls, values: Iterable[Any]) -> "SignedMassPacket":
        positive: Any = 0
        negative: Any = 0
        seen = False
        for index, value in enumerate(values):
            scalar = _finite_real(value, f"values[{index}]")
            seen = True
            if scalar >= 0:
                positive += scalar
            else:
                negative += -scalar
        if not seen:
            return cls(0, 0)
        return cls(positive, negative)

    @property
    def value(self) -> Real:
        return self.positive - self.negative

    @property
    def total_variation(self) -> Real:
        return self.positive + self.negative

    @property
    def cancellation_mass(self) -> Real:
        return min(self.positive, self.negative)

    @property
    def cancellation_defect(self) -> Real:
        """Gap between unsigned mass and the magnitude of the exact sum."""

        return self.total_variation - abs(self.value)

    def merge(self, other: "SignedMassPacket") -> "SignedMassPacket":
        if not isinstance(other, SignedMassPacket):
            raise ValidationError("other must be a SignedMassPacket")
        return SignedMassPacket(
            self.positive + other.positive,
            self.negative + other.negative,
        )

    def multiply(self, other: "SignedMassPacket") -> "SignedMassPacket":
        """Cartesian-product multiplication of signed decompositions.

        Positive products arise from equal signs; negative products arise from
        opposite signs.  The exact signed observation therefore multiplies.
        """

        if not isinstance(other, SignedMassPacket):
            raise ValidationError("other must be a SignedMassPacket")
        return SignedMassPacket(
            self.positive * other.positive + self.negative * other.negative,
            self.positive * other.negative + self.negative * other.positive,
        )



@dataclass(frozen=True, slots=True)
class FactorizationWitness:
    """Finite witness that target semantics do not factor through source semantics."""

    source_value: Any
    first_target: Any
    second_target: Any
    first_index: int
    second_index: int


@dataclass(frozen=True, slots=True)
class FiniteFactorizationAudit:
    """Finite audit of whether target labels are constant on source fibers.

    A successful finite audit is only evidence on the supplied sample.  The
    universal factorization theorem is the elementary kernel-inclusion
    statement documented in the Block 15 contract.
    """

    exact_on_sample: bool
    source_class_count: int
    joint_class_count: int
    witness: FactorizationWitness | None = None


def finite_factorization_audit(
    source_values: Iterable[Any],
    target_values: Iterable[Any],
) -> FiniteFactorizationAudit:
    """Check whether target values are constant on equal source values.

    The finite implementation intentionally uses equality rather than hashing so
    that small unhashable semantic objects can be audited as well.
    """

    source = list(source_values)
    target = list(target_values)
    if len(source) != len(target):
        raise ValidationError("source_values and target_values must have equal length")

    representatives: list[tuple[Any, Any, int]] = []
    joint_representatives: list[tuple[Any, Any]] = []
    witness: FactorizationWitness | None = None
    for index, (s_value, t_value) in enumerate(zip(source, target, strict=True)):
        matched_source = False
        for old_source, old_target, old_index in representatives:
            if s_value == old_source:
                matched_source = True
                if t_value != old_target and witness is None:
                    witness = FactorizationWitness(
                        source_value=s_value,
                        first_target=old_target,
                        second_target=t_value,
                        first_index=old_index,
                        second_index=index,
                    )
                break
        if not matched_source:
            representatives.append((s_value, t_value, index))

        if not any(s_value == old_s and t_value == old_t for old_s, old_t in joint_representatives):
            joint_representatives.append((s_value, t_value))

    return FiniteFactorizationAudit(
        exact_on_sample=witness is None,
        source_class_count=len(representatives),
        joint_class_count=len(joint_representatives),
        witness=witness,
    )

def _logaddexp(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    high = max(left, right)
    low = min(left, right)
    return high + math.log1p(math.exp(low - high))


@dataclass(frozen=True, slots=True)
class TropicalResidualPacket:
    """Exact finite-temperature completion of a max-plus packet.

    A nonzero packet represents finitely many real ``scores`` x_i at
    temperature tau > 0 through

        dominant = max_i x_i
        log_residual_mass = log(sum_i exp((x_i-dominant)/tau)).

    The tropical projection is ``dominant``.  The exact dequantized observation
    is ``free_energy = dominant + tau * log_residual_mass``.  Merge corresponds
    to addition of the underlying positive weights; multiply corresponds to the
    Cartesian product of terms, hence multiplication of those weights.
    """

    dominant: float
    log_residual_mass: float
    term_count: int
    temperature: float = 1.0

    def __post_init__(self) -> None:
        temperature = float(self.temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValidationError("temperature must be a positive finite real number")
        count = int(self.term_count)
        if count != self.term_count or count < 0:
            raise ValidationError("term_count must be a nonnegative integer")

        dominant = float(self.dominant)
        log_residual = float(self.log_residual_mass)
        if count == 0:
            if dominant != -math.inf or log_residual != -math.inf:
                raise ValidationError("the zero packet requires -inf dominant and residual")
        else:
            if not math.isfinite(dominant):
                raise ValidationError("a nonzero packet requires a finite dominant score")
            if not math.isfinite(log_residual) or log_residual < -1e-14:
                raise ValidationError("a nonzero packet requires finite nonnegative log residual mass")
            # Every nonzero packet contains at least one dominant term, so r >= 1.
            if log_residual < 0:
                log_residual = 0.0
            upper = math.log(count)
            if log_residual > upper + 1e-12:
                raise ValidationError("log residual mass exceeds the finite-term bound")

        object.__setattr__(self, "dominant", dominant)
        object.__setattr__(self, "log_residual_mass", log_residual)
        object.__setattr__(self, "term_count", count)
        object.__setattr__(self, "temperature", temperature)

    @classmethod
    def zero(cls, temperature: float = 1.0) -> "TropicalResidualPacket":
        return cls(-math.inf, -math.inf, 0, temperature)

    @classmethod
    def from_scores(
        cls,
        scores: Iterable[Any],
        *,
        temperature: float = 1.0,
    ) -> "TropicalResidualPacket":
        tau = float(temperature)
        if not math.isfinite(tau) or tau <= 0:
            raise ValidationError("temperature must be a positive finite real number")
        finite_scores: list[float] = []
        for index, value in enumerate(scores):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValidationError(f"scores[{index}] must be real or -inf")
            score = float(value)
            if math.isnan(score) or score == math.inf:
                raise ValidationError(f"scores[{index}] must be finite or -inf")
            if score != -math.inf:
                finite_scores.append(score)
        if not finite_scores:
            return cls.zero(tau)
        dominant = max(finite_scores)
        shifted = [(score - dominant) / tau for score in finite_scores]
        log_residual = -math.inf
        for value in shifted:
            log_residual = _logaddexp(log_residual, value)
        return cls(dominant, log_residual, len(finite_scores), tau)

    @classmethod
    def from_positive_weights(
        cls,
        weights: Iterable[Any],
        *,
        temperature: float = 1.0,
    ) -> "TropicalResidualPacket":
        tau = float(temperature)
        scores: list[float] = []
        for index, value in enumerate(weights):
            weight = _finite_nonnegative(value, f"weights[{index}]")
            if weight > 0:
                scores.append(tau * math.log(float(weight)))
        return cls.from_scores(scores, temperature=tau)

    @property
    def is_zero(self) -> bool:
        return self.term_count == 0

    @property
    def tropical_value(self) -> float:
        return self.dominant

    @property
    def correction(self) -> float:
        if self.is_zero:
            return 0.0
        return self.temperature * self.log_residual_mass

    @property
    def free_energy(self) -> float:
        if self.is_zero:
            return -math.inf
        return self.dominant + self.correction

    @property
    def effective_multiplicity(self) -> float:
        """Soft number of terms surviving the tropical projection.

        This is ``exp(correction / temperature)`` and lies in ``[1, n]`` for
        a nonzero packet with ``n`` represented terms.
        """

        if self.is_zero:
            return 0.0
        return math.exp(self.log_residual_mass)

    @property
    def correction_upper_bound(self) -> float:
        if self.is_zero:
            return 0.0
        return self.temperature * math.log(self.term_count)

    def merge(self, other: "TropicalResidualPacket") -> "TropicalResidualPacket":
        """Exact addition of represented positive sum-product expressions."""

        self._check_compatible(other)
        if self.is_zero:
            return other
        if other.is_zero:
            return self
        dominant = max(self.dominant, other.dominant)
        left = self.log_residual_mass + (self.dominant - dominant) / self.temperature
        right = other.log_residual_mass + (other.dominant - dominant) / self.temperature
        return TropicalResidualPacket(
            dominant,
            _logaddexp(left, right),
            self.term_count + other.term_count,
            self.temperature,
        )

    def multiply(self, other: "TropicalResidualPacket") -> "TropicalResidualPacket":
        """Exact multiplication via pairwise score addition."""

        self._check_compatible(other)
        if self.is_zero or other.is_zero:
            return TropicalResidualPacket.zero(self.temperature)
        return TropicalResidualPacket(
            self.dominant + other.dominant,
            self.log_residual_mass + other.log_residual_mass,
            self.term_count * other.term_count,
            self.temperature,
        )

    def _check_compatible(self, other: "TropicalResidualPacket") -> None:
        if not isinstance(other, TropicalResidualPacket):
            raise ValidationError("other must be a TropicalResidualPacket")
        if not math.isclose(self.temperature, other.temperature, rel_tol=0.0, abs_tol=0.0):
            raise ValidationError("carrier temperatures must match")


__all__ = [
    "FactorizationWitness",
    "FiniteFactorizationAudit",
    "SignedMassPacket",
    "TropicalResidualPacket",
    "boolean_support_nonnegative",
    "finite_factorization_audit",
]
