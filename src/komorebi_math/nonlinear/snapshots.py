"""Portable finite snapshots of packet-jet blocks.

A :class:`PacketJet` contains an executable derivative provider and therefore
cannot be reconstructed from JSON safely.  A snapshot evaluates the finite
multilinear block on the ambient standard basis and stores its nonzero
coefficients.  The resulting object is provider-independent, serializable, and
can evaluate the same fixed packet block at its recorded base point.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from math import isfinite
from typing import Any

import numpy as np

from .._residuals import (value_audit, finite_tolerance, coefficient_storage_decision,
                          python_integer_entries)
from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .jets import PacketJet, PacketJetAddress


def _positive_integer(value: Any, *, name: str) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a positive integer") from exc
    if isinstance(value, bool) or normalized != value or normalized < 1:
        raise ValidationError(f"{name} must be a positive integer")
    return normalized


def _vector_entries(value: Any, *, dimension: int, name: str) -> tuple[Any, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        normalized_shape = tuple(int(entry) for entry in shape)
        if normalized_shape == (dimension,):
            return tuple(value[index] for index in range(dimension))
        if normalized_shape == (dimension, 1):
            return tuple(value[index, 0] for index in range(dimension))
        if normalized_shape == (1, dimension):
            return tuple(value[0, index] for index in range(dimension))
    array = np.asarray(value, dtype=object)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1 or array.shape[0] != dimension:
        raise DimensionError(f"{name} must have dimension {dimension}")
    return tuple(array.tolist())


def _numeric_vector(value: Any, *, dimension: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1 or array.shape[0] != dimension:
        raise DimensionError(f"{name} must have dimension {dimension}")
    return array


def _norm(value: Any) -> float:
    try:
        return float(np.linalg.norm(np.asarray(value, dtype=complex).reshape(-1)))
    except (TypeError, ValueError):
        entries = np.asarray(value, dtype=object).reshape(-1)
        return float(sum(abs(complex(entry)) ** 2 for entry in entries) ** 0.5)


@dataclass(frozen=True, slots=True)
class PacketJetCoefficient:
    """One standard-basis coefficient of a persisted packet jet."""

    indices: tuple[int, ...]
    value: tuple[Any, ...]


@documented(
    ProvenanceRecord(
        identifier="PSC-III-D58",
        title="Finite packet-jet snapshot",
        status=ClaimStatus.FINITE_CERTIFICATE,
        source="Block 10 runtime",
        assumptions=("finite dimensional multilinear packet jet",),
        limitations=("snapshot represents one fixed order, address, and base point",),
        verification_suite="nonlinear.snapshots",
    )
)
@dataclass(frozen=True, slots=True)
class PacketJetSnapshot:
    """A provider-independent sparse coefficient record of one packet jet."""

    address: PacketJetAddress
    source_dimension: int
    target_dimension: int
    provider_name: str
    exactness: Exactness
    point: tuple[Any, ...]
    coefficients: tuple[PacketJetCoefficient, ...]
    capture_certificate: Certificate

    def __post_init__(self) -> None:
        source_dimension = _positive_integer(self.source_dimension, name="source dimension")
        target_dimension = _positive_integer(self.target_dimension, name="target dimension")
        object.__setattr__(self, "source_dimension", source_dimension)
        object.__setattr__(self, "target_dimension", target_dimension)
        object.__setattr__(self, "exactness", Exactness(self.exactness))
        point = tuple(self.point)
        if len(point) != source_dimension:
            raise DimensionError("snapshot point does not match the source dimension")
        object.__setattr__(self, "point", point)
        normalized: list[PacketJetCoefficient] = []
        seen: set[tuple[int, ...]] = set()
        for raw in self.coefficients:
            coefficient = (
                raw
                if isinstance(raw, PacketJetCoefficient)
                else PacketJetCoefficient(tuple(raw.indices), tuple(raw.value))
            )
            indices = tuple(int(value) for value in coefficient.indices)
            if len(indices) != self.address.order:
                raise DimensionError("snapshot coefficient order does not match its packet address")
            if any(value < 0 or value >= source_dimension for value in indices):
                raise DimensionError("snapshot coefficient index is outside the source dimension")
            if indices in seen:
                raise ValidationError("snapshot contains duplicate basis coefficients")
            seen.add(indices)
            values = tuple(coefficient.value)
            if len(values) != target_dimension:
                raise DimensionError("snapshot coefficient value has the wrong target dimension")
            normalized.append(PacketJetCoefficient(indices, values))
        normalized.sort(key=lambda entry: entry.indices)
        object.__setattr__(self, "coefficients", tuple(normalized))

    @property
    def order(self) -> int:
        """Return the multilinear derivative order."""

        return self.address.order

    @property
    def nnz(self) -> int:
        """Return the number of stored standard-basis coefficient vectors."""

        return len(self.coefficients)

    @property
    def dense_coefficient_count(self) -> int:
        """Return the number of possible input coefficient addresses."""

        return self.source_dimension**self.order

    def coefficient(self, indices: Sequence[int]) -> tuple[Any, ...]:
        """Return one coefficient vector, or a zero vector when unstored."""

        key = tuple(int(value) for value in indices)
        if len(key) != self.order:
            raise DimensionError(f"an order {self.order} snapshot requires {self.order} indices")
        if any(value < 0 or value >= self.source_dimension for value in key):
            raise DimensionError("snapshot coefficient index is outside the source dimension")
        for coefficient in self.coefficients:
            if coefficient.indices == key:
                return coefficient.value
        return (0,) * self.target_dimension

    def evaluate(self, directions: Sequence[Any]) -> np.ndarray:
        """Evaluate the persisted multilinear block on ordered directions."""

        direction_tuple = tuple(
            _numeric_vector(
                direction,
                dimension=self.source_dimension,
                name="snapshot direction",
            )
            for direction in directions
        )
        if len(direction_tuple) != self.order:
            raise ValidationError(
                f"order {self.order} snapshot requires exactly {self.order} directions"
            )
        if self.exactness is Exactness.EXACT:
            direction_tuple = tuple(python_integer_entries(v) for v in direction_tuple)
        value_dtypes = [
            np.asarray(value).dtype
            for coefficient in self.coefficients
            for value in coefficient.value
        ]
        direction_dtypes = [direction.dtype for direction in direction_tuple]
        try:
            dtype = np.result_type(*value_dtypes, *direction_dtypes)
        except TypeError:
            dtype = np.dtype(object)
        if dtype.kind in {"O", "S", "U", "V"} or self.exactness is Exactness.EXACT:
            dtype = np.dtype(object)
        total = np.zeros(self.target_dimension, dtype=dtype)
        for coefficient in self.coefficients:
            factor: Any = 1
            for slot, index in enumerate(coefficient.indices):
                factor = factor * direction_tuple[slot][index]
            for output, value in enumerate(coefficient.value):
                if self.exactness is Exactness.EXACT and isinstance(value, np.integer):
                    value = int(value)
                total[output] = total[output] + factor * value
        return total

    __call__ = evaluate

    def to_dense(self, *, max_elements: int = 10_000_000) -> np.ndarray:
        """Expand the stored block into an ordered dense multilinear tensor."""

        limit = _positive_integer(max_elements, name="maximum dense elements")
        element_count = self.target_dimension * self.dense_coefficient_count
        if element_count > limit:
            raise ValidationError(
                f"dense packet-jet expansion requires {element_count} elements, exceeding {limit}"
            )
        dtype = object if self.exactness is Exactness.EXACT else complex
        tensor = np.zeros(
            (self.target_dimension, *(self.source_dimension for _ in range(self.order))),
            dtype=dtype,
        )
        for coefficient in self.coefficients:
            for output, value in enumerate(coefficient.value):
                tensor[(output, *coefficient.indices)] = value
        return tensor

    def comparison_certificate(
        self,
        jet: PacketJet,
        directions: Sequence[Any],
        *,
        tolerance: float | None = None,
    ) -> Certificate:
        """Compare the snapshot with its executable jet on one supplied probe."""

        if jet.address != self.address:
            raise ValidationError("snapshot and executable jet use different packet addresses")
        if jet.source.dimension != self.source_dimension or jet.target.dimension != self.target_dimension:
            raise DimensionError("snapshot and executable jet dimensions differ")
        tol = finite_tolerance(max(jet.source.tolerance, jet.target.tolerance)
                               if tolerance is None else tolerance)
        exactness = (Exactness.EXACT if self.exactness is Exactness.EXACT
                     and jet.exactness is Exactness.EXACT else Exactness.NUMERICAL)
        audit = value_audit(exact=exactness is Exactness.EXACT, tolerance=tol)
        observed = jet.target.backend.as_vector(jet.evaluate(directions))
        persisted = jet.target.backend.as_vector(self.evaluate(directions))
        residual = audit.check_value("snapshot_reconstruction",
                                    jet.target.backend.subtract(observed, persisted))
        valid = audit.valid
        return Certificate(
            valid=valid,
            exactness=exactness,
            residuals={"snapshot_reconstruction": residual},
            tolerance=None if exactness is Exactness.EXACT else tol,
            assumptions_checked=("matching packet address", "matching finite dimensions"),
            assumptions_unchecked=("probes not supplied to this finite comparison",
                                   "global correctness and multilinearity of the derivative provider") + audit.unresolved,
            diagnostics={"order": self.order, "stored_coefficients": self.nnz,
                         "residual_decisions": audit.decisions},
            theorem_id="PSC-III-D58",
            message=(
                "packet-jet snapshot matches the executable block"
                if valid
                else "packet-jet snapshot comparison failed"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic snapshot data suitable for report exports."""

        return {
            "address": self.address.to_dict(),
            "source_dimension": self.source_dimension,
            "target_dimension": self.target_dimension,
            "provider_name": self.provider_name,
            "exactness": self.exactness.value,
            "point": self.point,
            "coefficients": tuple(
                {"indices": coefficient.indices, "value": coefficient.value}
                for coefficient in self.coefficients
            ),
            "capture_certificate": self.capture_certificate,
        }


def snapshot_packet_jet(
    jet: PacketJet,
    *,
    tolerance: float | None = None,
    max_evaluations: int = 100_000,
) -> PacketJetSnapshot:
    """Capture one packet jet by exhaustive standard-basis evaluation.

    The safety limit is checked before evaluation.  Exact backends retain every
    nonzero coefficient.  Numerical backends omit coefficients whose output
    norm does not exceed the declared tolerance and report the largest omitted
    basis norm in the certificate.
    """

    limit = _positive_integer(max_evaluations, name="maximum evaluations")
    evaluation_count = jet.source.dimension**jet.order
    if evaluation_count > limit:
        raise ValidationError(
            f"packet-jet snapshot requires {evaluation_count} evaluations, exceeding {limit}"
        )
    tol = (
        max(jet.source.tolerance, jet.target.tolerance)
        if tolerance is None
        else float(tolerance)
    )
    if tol < 0 or not isfinite(tol):
        raise ValidationError("snapshot tolerance must be finite and nonnegative")
    identity = jet.source.backend.eye(jet.source.dimension)
    basis = tuple(identity[:, index] for index in range(jet.source.dimension))
    coefficients: list[PacketJetCoefficient] = []
    maximum_omitted_norm = 0.0
    for indices in product(range(jet.source.dimension), repeat=jet.order):
        value = jet.evaluate(tuple(basis[index] for index in indices))
        zero, value_norm, _ = coefficient_storage_decision(
            value, exact=jet.exactness is Exactness.EXACT, tolerance=tol)
        if zero:
            maximum_omitted_norm = max(maximum_omitted_norm, value_norm)
            continue
        coefficients.append(
            PacketJetCoefficient(
                tuple(indices),
                _vector_entries(
                    value,
                    dimension=jet.target.dimension,
                    name="packet-jet coefficient",
                ),
            )
        )
    certificate = Certificate(
        valid=True,
        exactness=jet.exactness,
        residuals={"maximum_omitted_basis_norm": maximum_omitted_norm},
        tolerance=None if jet.exactness is Exactness.EXACT else tol,
        assumptions_checked=(
            "complete ambient standard basis",
            "fixed packet address and base point",
        ),
        assumptions_unchecked=(
            "global correctness and multilinearity of the underlying derivative provider",
        ),
        diagnostics={
            "order": jet.order,
            "basis_evaluations": evaluation_count,
            "stored_coefficients": len(coefficients),
            "dense_coefficient_addresses": evaluation_count,
            "provider": jet.provider.name,
        },
        theorem_id="PSC-III-D58",
        message="packet-jet block captured on the complete standard basis",
    )
    return PacketJetSnapshot(
        address=jet.address,
        source_dimension=jet.source.dimension,
        target_dimension=jet.target.dimension,
        provider_name=jet.provider.name,
        exactness=jet.exactness,
        point=_vector_entries(
            jet.point,
            dimension=jet.source.dimension,
            name="packet-jet point",
        ),
        coefficients=tuple(coefficients),
        capture_certificate=certificate,
    )


__all__ = [
    "PacketJetCoefficient",
    "PacketJetSnapshot",
    "snapshot_packet_jet",
]
