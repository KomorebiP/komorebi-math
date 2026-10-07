"""Sparse symmetric derivative tensors and polynomial derivative providers.

The dense :class:`~komorebi_math.nonlinear.PolynomialDerivativeProvider`
stores every ordered tensor entry.  This module stores one coefficient per
symmetric index orbit and expands only the contractions required by an
actual directional derivative.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import combinations_with_replacement
from math import factorial
from typing import Any

import numpy as np

from .._residuals import value_audit, finite_tolerance, python_integer_entries
from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .derivatives import PolynomialDerivativeProvider


@lru_cache(maxsize=None)
def _unique_permutations(indices: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Return the distinct permutations of one sorted integer tuple."""

    if len(indices) <= 1:
        return (indices,)
    permutations: list[tuple[int, ...]] = []
    for value in sorted(set(indices)):
        position = indices.index(value)
        remaining = indices[:position] + indices[position + 1 :]
        permutations.extend((value, *tail) for tail in _unique_permutations(remaining))
    return tuple(permutations)


def _as_vector(value: Any, dimension: int, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1 or int(array.shape[0]) != dimension:
        raise DimensionError(f"{name} must have dimension {dimension}")
    return array


def _scalar_is_zero(value: Any, tolerance: float) -> bool:
    try:
        if tolerance == 0:
            return bool(value == 0)
        return abs(complex(value)) <= tolerance
    except (TypeError, ValueError):
        return bool(value == 0)


def _scalar_close(left: Any, right: Any, tolerance: float) -> bool:
    if tolerance == 0:
        return bool(left == right)
    try:
        return abs(complex(left - right)) <= tolerance
    except (TypeError, ValueError):
        return bool(left == right)


def _result_dtype(values: Sequence[Any], vectors: Sequence[np.ndarray], *, exact: bool) -> np.dtype[Any]:
    if exact:
        return np.dtype(object)
    dtypes: list[np.dtype[Any]] = [np.dtype(float)]
    dtypes.extend(vector.dtype for vector in vectors)
    for value in values:
        dtype = np.asarray(value).dtype
        if dtype == object:
            return np.dtype(object)
        dtypes.append(dtype)
    return np.result_type(*dtypes)


@dataclass(frozen=True, slots=True)
class SparseTensorEntry:
    """One coefficient of a symmetric tensor orbit.

    ``inputs`` is stored in nondecreasing order.  The coefficient represents
    every distinct ordered permutation of those source indices.
    """

    output: int
    inputs: tuple[int, ...]
    value: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", int(self.output))
        object.__setattr__(self, "inputs", tuple(int(index) for index in self.inputs))


@documented(
    ProvenanceRecord(
        identifier="PSC-III-D04S",
        title="Sparse symmetric derivative tensor",
        status=ClaimStatus.DEFINITION,
        source="Paper III scalable runtime",
        assumptions=("finite dimensions", "symmetric source axes"),
        verification_suite="nonlinear.sparse",
    )
)
@dataclass(frozen=True, slots=True)
class SparseSymmetricTensor:
    """One finite symmetric tensor stored by source-index orbits."""

    order: int
    source_dimension: int
    target_dimension: int
    entries: tuple[SparseTensorEntry, ...] = ()

    def __post_init__(self) -> None:
        order = int(self.order)
        source_dimension = int(self.source_dimension)
        target_dimension = int(self.target_dimension)
        if isinstance(self.order, bool) or order != self.order or order < 0:
            raise ValidationError("sparse tensor order must be a nonnegative integer")
        if source_dimension < 1 or target_dimension < 1:
            raise DimensionError("sparse tensor dimensions must be positive")

        normalized: list[SparseTensorEntry] = []
        seen: set[tuple[int, tuple[int, ...]]] = set()
        for raw_entry in self.entries:
            if not isinstance(raw_entry, SparseTensorEntry):
                raise ValidationError("sparse tensor entries must be SparseTensorEntry instances")
            entry = SparseTensorEntry(raw_entry.output, raw_entry.inputs, raw_entry.value)
            if entry.output < 0 or entry.output >= target_dimension:
                raise DimensionError("sparse tensor output index is outside the target dimension")
            if len(entry.inputs) != order:
                raise DimensionError("sparse tensor input index count does not match its order")
            if any(index < 0 or index >= source_dimension for index in entry.inputs):
                raise DimensionError("sparse tensor input index is outside the source dimension")
            if tuple(sorted(entry.inputs)) != entry.inputs:
                raise ValidationError("sparse symmetric tensor inputs must be nondecreasing")
            key = (entry.output, entry.inputs)
            if key in seen:
                raise ValidationError("sparse tensor orbit keys must be unique")
            seen.add(key)
            if not _scalar_is_zero(entry.value, 0.0):
                normalized.append(entry)

        normalized.sort(key=lambda entry: (entry.output, entry.inputs))
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "source_dimension", source_dimension)
        object.__setattr__(self, "target_dimension", target_dimension)
        object.__setattr__(self, "entries", tuple(normalized))

    @classmethod
    def from_mapping(
        cls,
        order: int,
        source_dimension: int,
        target_dimension: int,
        entries: Mapping[tuple[int, ...], Any],
    ) -> "SparseSymmetricTensor":
        """Construct from ``(output, *sorted_inputs) -> coefficient`` data."""

        normalized: list[SparseTensorEntry] = []
        expected_length = int(order) + 1
        for key, value in entries.items():
            index = tuple(int(part) for part in key)
            if len(index) != expected_length:
                raise DimensionError(
                    f"an order {order} sparse tensor key must contain {expected_length} indices"
                )
            normalized.append(SparseTensorEntry(index[0], index[1:], value))
        return cls(order, source_dimension, target_dimension, tuple(normalized))

    @classmethod
    def from_dense(
        cls,
        tensor: Any,
        *,
        source_dimension: int | None = None,
        tolerance: float = 0.0,
        validate_symmetry: bool = True,
    ) -> "SparseSymmetricTensor":
        """Compress a dense symmetric tensor into orbit storage."""

        tol = float(tolerance)
        if not np.isfinite(tol) or tol < 0:
            raise ValidationError("sparse tensor tolerance must be finite and nonnegative")
        array = np.asarray(tensor)
        if array.ndim < 1:
            raise DimensionError("a derivative tensor must have at least one target axis")
        order = array.ndim - 1
        target_dimension = int(array.shape[0])
        if order == 0:
            inferred_source = 1 if source_dimension is None else int(source_dimension)
            if inferred_source < 1:
                raise DimensionError("source dimension must be positive")
        else:
            source_axes = tuple(int(length) for length in array.shape[1:])
            if len(set(source_axes)) != 1:
                raise DimensionError("every source axis must have one common dimension")
            inferred_source = source_axes[0]
            if source_dimension is not None and int(source_dimension) != inferred_source:
                raise DimensionError("declared source dimension does not match the dense tensor")
        entries: list[SparseTensorEntry] = []
        input_orbits: tuple[tuple[int, ...], ...] = (
            ((),)
            if order == 0
            else tuple(combinations_with_replacement(range(inferred_source), order))
        )
        for output in range(target_dimension):
            for inputs in input_orbits:
                value = array[(output, *inputs)]
                if validate_symmetry:
                    for permutation in _unique_permutations(inputs):
                        candidate = array[(output, *permutation)]
                        if not _scalar_close(candidate, value, tol):
                            raise ValidationError(
                                "dense tensor source axes are not symmetric within tolerance"
                            )
                if not _scalar_is_zero(value, tol):
                    entries.append(SparseTensorEntry(output, inputs, value.item() if isinstance(value, np.generic) else value))
        return cls(order, inferred_source, target_dimension, tuple(entries))

    @property
    def stored_entry_count(self) -> int:
        """Number of stored symmetric-orbit coefficients."""

        return len(self.entries)

    @property
    def dense_entry_count(self) -> int:
        """Number of entries in the equivalent dense tensor."""

        return self.target_dimension * self.source_dimension**self.order

    @property
    def expanded_nonzero_count(self) -> int:
        """Number of nonzero ordered entries represented by the orbit storage."""

        return sum(len(_unique_permutations(entry.inputs)) for entry in self.entries)

    @property
    def storage_ratio(self) -> float:
        """Stored orbit count divided by the dense tensor size."""

        return self.stored_entry_count / self.dense_entry_count

    def contract(self, vectors: Sequence[Any], *, exact: bool = False) -> np.ndarray:
        """Contract the tensor against one vector per source axis."""

        direction_tuple = tuple(
            _as_vector(vector, self.source_dimension, name="tensor contraction vector")
            for vector in vectors
        )
        if len(direction_tuple) != self.order:
            raise ValidationError(f"order {self.order} contraction requires {self.order} vectors")
        dtype = _result_dtype(
            tuple(entry.value for entry in self.entries),
            direction_tuple,
            exact=exact or any(np.asarray(entry.value).dtype == object for entry in self.entries),
        )
        if dtype.kind == "O":
            direction_tuple = tuple(python_integer_entries(v) for v in direction_tuple)
        total = np.zeros(self.target_dimension, dtype=dtype)
        for entry in self.entries:
            orbit_total: Any = 0
            for permutation in _unique_permutations(entry.inputs):
                product_value: Any = (int(entry.value) if dtype.kind == "O"
                                      and isinstance(entry.value, np.integer) else entry.value)
                for position, source_index in enumerate(permutation):
                    product_value = product_value * direction_tuple[position][source_index]
                orbit_total = orbit_total + product_value
            total[entry.output] = total[entry.output] + orbit_total
        return total

    def to_dense(self, *, dtype: Any | None = None) -> np.ndarray:
        """Materialize the full symmetric tensor."""

        if dtype is None:
            values = tuple(entry.value for entry in self.entries)
            if not values:
                resolved_dtype: Any = float
            elif any(np.asarray(value).dtype == object for value in values):
                resolved_dtype = object
            else:
                resolved_dtype = np.result_type(*values)
        else:
            resolved_dtype = dtype
        shape = (self.target_dimension,) + (self.source_dimension,) * self.order
        dense = np.zeros(shape, dtype=resolved_dtype)
        for entry in self.entries:
            for permutation in _unique_permutations(entry.inputs):
                dense[(entry.output, *permutation)] = entry.value
        return dense

    def certificate(self, *, tolerance: float = 0.0) -> Certificate:
        """Certify that orbit expansion and dense compression agree."""

        tol = finite_tolerance(tolerance)
        dense = self.to_dense()
        roundtrip = SparseSymmetricTensor.from_dense(
            dense,
            source_dimension=self.source_dimension,
            tolerance=tol,
        )
        left = self.to_dense(dtype=object if dense.dtype == object else None)
        right = roundtrip.to_dense(dtype=object if dense.dtype == object else None)
        if dense.dtype == object:
            valid = bool(np.array_equal(left, right))
            residual = 0.0 if valid else 1.0
            exactness = Exactness.EXACT
        else:
            audit = value_audit(exact=False, tolerance=tol)
            residual = audit.check_value("dense_roundtrip", (left-right).reshape(-1))
            valid = audit.valid
            exactness = Exactness.NUMERICAL
        return Certificate(
            valid=valid,
            exactness=exactness,
            residuals={"dense_roundtrip": residual},
            tolerance=None if exactness is Exactness.EXACT else tol,
            assumptions_checked=("canonical symmetric orbit keys", "dense expansion", "dense recompression"),
            diagnostics={
                "order": self.order,
                "stored_entries": self.stored_entry_count,
                "expanded_nonzero_entries": self.expanded_nonzero_count,
                "dense_entries": self.dense_entry_count,
                "storage_ratio": self.storage_ratio,
            },
            theorem_id="PSC-III-D04S",
            message="sparse symmetric tensor storage is internally consistent" if valid else "sparse tensor roundtrip failed",
        )

    def to_dict(self) -> dict[str, Any]:
        """Return stable sparse tensor metadata and orbit entries."""

        return {
            "order": self.order,
            "source_dimension": self.source_dimension,
            "target_dimension": self.target_dimension,
            "entries": tuple(
                {"output": entry.output, "inputs": entry.inputs, "value": entry.value}
                for entry in self.entries
            ),
        }


@documented(
    ProvenanceRecord(
        identifier="PSC-III-D04P",
        title="Sparse polynomial derivative provider",
        status=ClaimStatus.DEFINITION,
        source="Paper III scalable runtime",
        assumptions=("finite sparse symmetric coefficient tensors",),
        verification_suite="nonlinear.sparse",
    )
)
class SparsePolynomialDerivativeProvider:
    """Polynomial derivatives evaluated from sparse symmetric tensor orbits.

    The coefficient convention is identical to
    :class:`PolynomialDerivativeProvider`:

    ``F(x) = sum_r T_r[x, ..., x] / r!``.
    """

    def __init__(
        self,
        tensors: Mapping[int, SparseSymmetricTensor | Mapping[tuple[int, ...], Any] | Any],
        *,
        source_dimension: int | None = None,
        target_dimension: int | None = None,
        exactness: Exactness = Exactness.NUMERICAL,
        tolerance: float = 1e-10,
        name: str = "sparse polynomial derivative provider",
    ) -> None:
        if not tensors:
            raise ValidationError("a sparse polynomial provider requires at least one tensor")
        normalized_exactness = Exactness(exactness)
        tol = finite_tolerance(tolerance)
        compression_tolerance = 0.0 if normalized_exactness is Exactness.EXACT else tol
        declared_source = None if source_dimension is None else int(source_dimension)
        declared_target = None if target_dimension is None else int(target_dimension)
        if declared_source is not None and declared_source < 1:
            raise DimensionError("source dimension must be positive")
        if declared_target is not None and declared_target < 1:
            raise DimensionError("target dimension must be positive")

        ordered_tensors: list[tuple[int, Any]] = []
        for raw_order, raw_tensor in tensors.items():
            order = int(raw_order)
            if isinstance(raw_order, bool) or order != raw_order or order < 0:
                raise ValidationError("sparse polynomial tensor orders must be nonnegative integers")
            ordered_tensors.append((order, raw_tensor))
        ordered_tensors.sort(key=lambda item: item[0])

        # Infer dimensions from every informative tensor before converting the
        # order-zero term.  A constant coefficient has no source axis and must
        # inherit the source dimension from a positive-order term when present.
        for order, raw_tensor in ordered_tensors:
            if isinstance(raw_tensor, SparseSymmetricTensor):
                if raw_tensor.order != order:
                    raise ValidationError("sparse tensor order does not match its mapping key")
                candidate_source = raw_tensor.source_dimension
                candidate_target = raw_tensor.target_dimension
            elif isinstance(raw_tensor, Mapping):
                continue
            else:
                array = np.asarray(raw_tensor)
                if array.ndim < 1 or array.ndim - 1 != order:
                    raise DimensionError("dense sparse-polynomial tensor rank does not match its order")
                candidate_target = int(array.shape[0])
                candidate_source = None
                if order > 0:
                    source_axes = tuple(int(length) for length in array.shape[1:])
                    if len(set(source_axes)) != 1:
                        raise DimensionError("every source axis must have one common dimension")
                    candidate_source = source_axes[0]
            if declared_target is None:
                declared_target = candidate_target
            elif declared_target != candidate_target:
                raise DimensionError("all sparse polynomial tensors must share target dimensions")
            if candidate_source is not None:
                if declared_source is None:
                    declared_source = candidate_source
                elif declared_source != candidate_source:
                    raise DimensionError("all sparse polynomial tensors must share source dimensions")

        normalized: dict[int, SparseSymmetricTensor] = {}
        for order, raw_tensor in ordered_tensors:
            if isinstance(raw_tensor, SparseSymmetricTensor):
                tensor = raw_tensor
            elif isinstance(raw_tensor, Mapping):
                if declared_source is None or declared_target is None:
                    raise DimensionError(
                        "source and target dimensions are required for sparse tensor mappings"
                    )
                tensor = SparseSymmetricTensor.from_mapping(
                    order,
                    declared_source,
                    declared_target,
                    raw_tensor,
                )
            else:
                tensor = SparseSymmetricTensor.from_dense(
                    np.asarray(raw_tensor),
                    source_dimension=declared_source,
                    tolerance=compression_tolerance,
                )
            declared_source = tensor.source_dimension if declared_source is None else declared_source
            declared_target = tensor.target_dimension if declared_target is None else declared_target
            if tensor.source_dimension != declared_source or tensor.target_dimension != declared_target:
                raise DimensionError("all sparse polynomial tensors must share source and target dimensions")
            normalized[order] = tensor

        if declared_source is None or declared_target is None:
            raise DimensionError("sparse polynomial dimensions could not be inferred")
        self.source_dimension = declared_source
        self.target_dimension = declared_target
        self.tensors = normalized
        self.exactness = normalized_exactness
        self.tolerance = tol
        self.name = str(name)
        if not np.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValidationError("sparse polynomial tolerance must be finite and nonnegative")

    @classmethod
    def from_dense_provider(
        cls,
        provider: PolynomialDerivativeProvider,
        *,
        tolerance: float | None = None,
        name: str = "sparse polynomial derivative provider",
    ) -> "SparsePolynomialDerivativeProvider":
        """Compress an existing dense polynomial derivative provider."""

        tol = finite_tolerance(provider.tolerance if tolerance is None else tolerance)
        compression_tolerance = 0.0 if provider.exactness is Exactness.EXACT else tol
        tensors = {
            order: SparseSymmetricTensor.from_dense(
                tensor,
                source_dimension=provider.source_dimension,
                tolerance=compression_tolerance,
            )
            for order, tensor in provider.tensors.items()
        }
        return cls(
            tensors,
            source_dimension=provider.source_dimension,
            target_dimension=provider.target_dimension,
            exactness=provider.exactness,
            tolerance=tol,
            name=name,
        )

    @property
    def degree(self) -> int:
        return max(self.tensors)

    @property
    def stored_entry_count(self) -> int:
        return sum(tensor.stored_entry_count for tensor in self.tensors.values())

    @property
    def dense_entry_count(self) -> int:
        return sum(tensor.dense_entry_count for tensor in self.tensors.values())

    @property
    def storage_ratio(self) -> float:
        return self.stored_entry_count / self.dense_entry_count

    def supports_order(self, order: int) -> bool:
        try:
            normalized = int(order)
        except (TypeError, ValueError):
            return False
        return (
            not isinstance(order, bool)
            and normalized == order
            and 1 <= normalized <= self.degree
        )

    def _vectors_dtype(self, vectors: Sequence[np.ndarray]) -> np.dtype[Any]:
        values = tuple(entry.value for tensor in self.tensors.values() for entry in tensor.entries)
        return _result_dtype(values, vectors, exact=self.exactness is Exactness.EXACT)

    def value(self, point: Any) -> np.ndarray:
        x = _as_vector(point, self.source_dimension, name="point")
        total = np.zeros(self.target_dimension, dtype=self._vectors_dtype((x,)))
        for order, tensor in self.tensors.items():
            term = tensor.contract((x,) * order, exact=self.exactness is Exactness.EXACT)
            scalar: Any = Fraction(1, factorial(order)) if self.exactness is Exactness.EXACT else 1 / factorial(order)
            total = total + scalar * term
        return total

    def derivative(self, order: int, point: Any, directions: Sequence[Any]) -> np.ndarray:
        derivative_order = int(order)
        if not self.supports_order(derivative_order):
            raise ValidationError(
                f"sparse polynomial derivative order must lie between 1 and {self.degree}"
            )
        x = _as_vector(point, self.source_dimension, name="point")
        direction_tuple = tuple(
            _as_vector(direction, self.source_dimension, name="direction")
            for direction in directions
        )
        if len(direction_tuple) != derivative_order:
            raise ValidationError(
                f"order {derivative_order} requires exactly {derivative_order} directions"
            )
        total = np.zeros(
            self.target_dimension,
            dtype=self._vectors_dtype((x, *direction_tuple)),
        )
        for tensor_order, tensor in self.tensors.items():
            if tensor_order < derivative_order:
                continue
            remaining = tensor_order - derivative_order
            term = tensor.contract(
                (*direction_tuple, *((x,) * remaining)),
                exact=self.exactness is Exactness.EXACT,
            )
            scalar: Any = Fraction(1, factorial(remaining)) if self.exactness is Exactness.EXACT else 1 / factorial(remaining)
            total = total + scalar * term
        return total

    def to_dense_provider(self) -> PolynomialDerivativeProvider:
        """Materialize an equivalent dense provider."""

        dense_dtype: Any | None = object if self.exactness is Exactness.EXACT else None
        return PolynomialDerivativeProvider(
            {
                order: tensor.to_dense(dtype=dense_dtype)
                for order, tensor in self.tensors.items()
            },
            source_dimension=self.source_dimension,
            target_dimension=self.target_dimension,
            exactness=self.exactness,
            tolerance=self.tolerance,
            name=f"dense realization of {self.name}",
        )

    def storage_certificate(
        self,
        points: Sequence[Any] = (),
        directions: Sequence[Any] = (),
        *,
        max_order: int | None = None,
        tolerance: float | None = None,
    ) -> Certificate:
        """Compare sparse evaluation with the dense realization at finite probes."""

        tol = finite_tolerance(self.tolerance if tolerance is None else tolerance)
        dense = self.to_dense_provider()
        probes = tuple(points)
        direction_tuple = tuple(directions)
        exact = self.exactness is Exactness.EXACT
        audit = value_audit(exact=exact, tolerance=tol)
        residual = 0.0
        checked_values = 0
        checked_derivatives = 0
        for index, point in enumerate(probes):
            sparse_value = self.value(point)
            dense_value = dense.value(point)
            residual = max(residual, audit.check_value(f"value:{index}", sparse_value-dense_value))
            checked_values += 1
            order_limit = self.degree if max_order is None else min(self.degree, int(max_order))
            for derivative_order in range(1, order_limit + 1):
                if len(direction_tuple) < derivative_order:
                    break
                selected = direction_tuple[:derivative_order]
                sparse_derivative = self.derivative(derivative_order, point, selected)
                dense_derivative = dense.derivative(derivative_order, point, selected)
                residual = max(residual, audit.check_value(
                    f"derivative:{index}:{derivative_order}", sparse_derivative-dense_derivative))
                checked_derivatives += 1
        valid = audit.valid
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT if exact else Exactness.NUMERICAL,
            residuals={"sparse_dense_equivalence": residual},
            tolerance=None if exact else tol,
            assumptions_checked=(
                "symmetric orbit expansion",
                "finite value probes",
                "finite directional derivative probes",
            ),
            assumptions_unchecked=(("global equality beyond the supplied probes",)
                                   if probes else ("no value or derivative probes were supplied",)) + audit.unresolved,
            diagnostics={
                "degree": self.degree,
                "stored_entries": self.stored_entry_count,
                "dense_entries": self.dense_entry_count,
                "storage_ratio": self.storage_ratio,
                "residual_decisions": audit.decisions,
                "value_probes": checked_values,
                "derivative_probes": checked_derivatives,
            },
            theorem_id="PSC-III-D04P",
            message="sparse and dense polynomial providers agree on all probes" if valid else "sparse and dense polynomial providers disagree",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_dimension": self.source_dimension,
            "target_dimension": self.target_dimension,
            "degree": self.degree,
            "exactness": self.exactness.value,
            "stored_entries": self.stored_entry_count,
            "dense_entries": self.dense_entry_count,
            "storage_ratio": self.storage_ratio,
            "tensors": {order: tensor.to_dict() for order, tensor in sorted(self.tensors.items())},
        }


__all__ = [
    "SparsePolynomialDerivativeProvider",
    "SparseSymmetricTensor",
    "SparseTensorEntry",
]
