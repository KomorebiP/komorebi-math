"""Finite dimensional derivative providers for nonlinear packet calculus.

The runtime separates the source of derivative data from packet projection.
Providers may be exact, symbolic, analytic, or numerical.  Every provider uses
one directional interface::

    derivative(order, point, directions)

where ``directions`` contains exactly ``order`` ambient source vectors.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations
from math import factorial
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .._residuals import value_audit, finite_tolerance, coefficient_storage_decision
from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented

DirectionalDerivative = Callable[[Any, tuple[Any, ...]], Any]


def _vector_length(value: Any) -> int:
    """Return the length of a vector-like value without changing its type."""

    shape = getattr(value, "shape", None)
    if shape is not None:
        normalized = tuple(int(entry) for entry in shape)
        if len(normalized) == 1:
            return normalized[0]
        if len(normalized) == 2 and 1 in normalized:
            return max(normalized)
    array = np.asarray(value, dtype=object)
    if array.ndim == 0:
        return 1
    if array.ndim == 1:
        return int(array.shape[0])
    if array.ndim == 2 and 1 in array.shape:
        return int(max(array.shape))
    raise DimensionError("value must be vector-like")


def _as_numeric_vector(value: Any, *, dimension: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1 or array.shape[0] != dimension:
        raise DimensionError(f"{name} must have dimension {dimension}")
    if not np.all(np.isfinite(array.astype(complex))):
        raise ValidationError(f"{name} must contain finite values")
    return array


def _norm(value: Any) -> float:
    try:
        return float(np.linalg.norm(np.asarray(value, dtype=complex).reshape(-1)))
    except (TypeError, ValueError):
        array = np.asarray(value, dtype=object).reshape(-1)
        return float(sum(abs(complex(entry)) ** 2 for entry in array) ** 0.5)


def _subtract(left: Any, right: Any) -> Any:
    try:
        return left - right
    except TypeError:
        return np.asarray(left, dtype=object) - np.asarray(right, dtype=object)


@runtime_checkable
class DerivativeProvider(Protocol):
    """Protocol implemented by finite dimensional derivative sources."""

    source_dimension: int
    target_dimension: int
    exactness: Exactness
    name: str

    def value(self, point: Any) -> Any:
        """Evaluate the underlying map."""

    def derivative(self, order: int, point: Any, directions: Sequence[Any]) -> Any:
        """Evaluate one directional Fréchet derivative."""

    def supports_order(self, order: int) -> bool:
        """Return whether the requested derivative order is available."""


@documented(
    ProvenanceRecord(
        identifier="PSC-III-D02",
        title="Declared finite dimensional derivative provider",
        status=ClaimStatus.DEFINITION,
        source="Paper III runtime",
        assumptions=("declared directional derivatives",),
        verification_suite="nonlinear.derivatives",
    )
)
@dataclass(slots=True)
class CallableDerivativeProvider:
    """A map with explicitly supplied directional derivative callables.

    Each derivative callable receives ``(point, directions)``.  The provider
    validates dimensions but does not infer mathematical correctness from the
    declaration.  ``validation_certificate`` can test symmetry and an optional
    Taylor identity at selected probes.
    """

    function: Callable[[Any], Any]
    derivatives: Mapping[int, DirectionalDerivative]
    source_dimension: int
    target_dimension: int
    exactness: Exactness = Exactness.NUMERICAL
    name: str = "declared derivative provider"

    def __post_init__(self) -> None:
        self.source_dimension = int(self.source_dimension)
        self.target_dimension = int(self.target_dimension)
        if self.source_dimension < 1 or self.target_dimension < 1:
            raise DimensionError("source and target dimensions must be positive")
        normalized: dict[int, DirectionalDerivative] = {}
        for raw_order, derivative in self.derivatives.items():
            order = int(raw_order)
            if isinstance(raw_order, bool) or order != raw_order or order < 1:
                raise ValidationError("derivative orders must be positive integers")
            if not callable(derivative):
                raise ValidationError("every declared derivative must be callable")
            normalized[order] = derivative
        self.derivatives = normalized
        if not callable(self.function):
            raise ValidationError("function must be callable")

    @property
    def available_orders(self) -> tuple[int, ...]:
        return tuple(sorted(self.derivatives))

    def supports_order(self, order: int) -> bool:
        try:
            n = int(order)
        except (TypeError, ValueError):
            return False
        return not isinstance(order, bool) and n == order and n in self.derivatives

    def _validate_point(self, point: Any) -> None:
        if _vector_length(point) != self.source_dimension:
            raise DimensionError("point dimension does not match the derivative provider")

    def _validate_output(self, value: Any) -> Any:
        if _vector_length(value) != self.target_dimension:
            raise DimensionError("derivative output dimension does not match the provider")
        return value

    def value(self, point: Any) -> Any:
        self._validate_point(point)
        return self._validate_output(self.function(point))

    def derivative(self, order: int, point: Any, directions: Sequence[Any]) -> Any:
        n = int(order)
        if isinstance(order, bool) or n != order or n < 1:
            raise ValidationError("derivative order must be a positive integer")
        try:
            derivative = self.derivatives[n]
        except KeyError as exc:
            raise ValidationError(f"derivative order {n} is not available") from exc
        self._validate_point(point)
        direction_tuple = tuple(directions)
        if len(direction_tuple) != n:
            raise ValidationError(f"order {n} requires exactly {n} directions")
        if any(_vector_length(direction) != self.source_dimension for direction in direction_tuple):
            raise DimensionError("direction dimension does not match the derivative provider")
        return self._validate_output(derivative(point, direction_tuple))

    def validation_certificate(
        self,
        point: Any,
        directions: Sequence[Any],
        *,
        tolerance: float = 1e-9,
        check_taylor: bool = False,
        increment: Any | None = None,
    ) -> Certificate:
        """Check derivative symmetry and, optionally, a finite Taylor identity.

        This is a finite validation at the supplied point and directions.  It is
        not a proof that user supplied derivative callables are globally valid.
        """

        self._validate_point(point)
        probes = tuple(directions)
        tolerance = finite_tolerance(tolerance)
        audit = value_audit(exact=self.exactness is Exactness.EXACT, tolerance=tolerance)
        symmetry_residual = 0.0
        checked_orders: list[int] = []
        for order in self.available_orders:
            if order > len(probes):
                continue
            selected = probes[:order]
            reference = self.derivative(order, point, selected)
            for permutation in permutations(range(order)):
                candidate = self.derivative(order, point, tuple(selected[index] for index in permutation))
                symmetry_residual = max(symmetry_residual, audit.check_value(
                    f"symmetry:{order}:{permutation}", _subtract(candidate, reference)))
            checked_orders.append(order)

        residuals: dict[str, float] = {"derivative_symmetry": symmetry_residual}
        unchecked = ["global correctness of declared derivatives"]
        if check_taylor:
            if increment is None:
                raise ValidationError("Taylor validation requires an increment")
            if _vector_length(increment) != self.source_dimension:
                raise DimensionError("increment dimension does not match the derivative provider")
            approximation = self.value(point)
            for order in self.available_orders:
                term = self.derivative(order, point, (increment,) * order)
                approximation = approximation + term / factorial(order)
            point_array = np.asarray(point, dtype=complex).reshape(-1)
            increment_array = np.asarray(increment, dtype=complex).reshape(-1)
            observed = self.value(point_array + increment_array)
            residuals["truncated_taylor"] = _norm(_subtract(observed, approximation))
            unchecked.append("Taylor remainder beyond the highest declared order")

        valid = audit.valid
        return Certificate(
            valid=valid,
            exactness=self.exactness,
            residuals=residuals,
            tolerance=None if self.exactness is Exactness.EXACT else tolerance,
            assumptions_checked=("source and target dimensions", "finite derivative symmetry probes"),
            assumptions_unchecked=tuple(unchecked) + audit.unresolved,
            diagnostics={"checked_orders": tuple(checked_orders), "provider": self.name,
                         "residual_decisions": audit.decisions,
                         "taylor_residual_is_diagnostic_only": True},
            theorem_id="PSC-III-D02",
            message="declared derivatives passed finite validation" if valid else "declared derivatives failed symmetry validation",
        )


@documented(
    ProvenanceRecord(
        identifier="PSC-III-N03",
        title="Recursive central directional differences",
        status=ClaimStatus.FINITE_CERTIFICATE,
        source="Numerical runtime",
        assumptions=("sufficient smoothness near the evaluation point", "finite difference scale resolves the derivative"),
        limitations=("roundoff and truncation error grow with derivative order",),
        verification_suite="nonlinear.derivatives",
    )
)
@dataclass(slots=True)
class FiniteDifferenceDerivativeProvider:
    """Numerical directional derivatives by recursive central differences."""

    function: Callable[[np.ndarray], Any]
    source_dimension: int
    target_dimension: int
    step: float = 1e-4
    max_order: int = 4
    name: str = "central finite difference provider"
    exactness: Exactness = Exactness.NUMERICAL

    def __post_init__(self) -> None:
        self.source_dimension = int(self.source_dimension)
        self.target_dimension = int(self.target_dimension)
        self.max_order = int(self.max_order)
        self.step = float(self.step)
        if self.source_dimension < 1 or self.target_dimension < 1:
            raise DimensionError("source and target dimensions must be positive")
        if self.max_order < 1:
            raise ValidationError("max_order must be positive")
        if not np.isfinite(self.step) or self.step <= 0:
            raise ValidationError("finite difference step must be positive and finite")

    def supports_order(self, order: int) -> bool:
        n = int(order)
        return not isinstance(order, bool) and n == order and 1 <= n <= self.max_order

    def value(self, point: Any) -> np.ndarray:
        x = _as_numeric_vector(point, dimension=self.source_dimension, name="point")
        value = np.asarray(self.function(x))
        if value.ndim == 2 and 1 in value.shape:
            value = value.reshape(-1)
        if value.ndim != 1 or value.shape[0] != self.target_dimension:
            raise DimensionError("function output dimension does not match the provider")
        if not np.all(np.isfinite(value.astype(complex))):
            raise ValidationError("function output must be finite")
        return value

    def derivative(self, order: int, point: Any, directions: Sequence[Any]) -> np.ndarray:
        n = int(order)
        if not self.supports_order(n):
            raise ValidationError(f"finite difference order must lie between 1 and {self.max_order}")
        x = _as_numeric_vector(point, dimension=self.source_dimension, name="point").astype(complex)
        direction_tuple = tuple(
            _as_numeric_vector(direction, dimension=self.source_dimension, name="direction").astype(complex)
            for direction in directions
        )
        if len(direction_tuple) != n:
            raise ValidationError(f"order {n} requires exactly {n} directions")

        def recurse(base: np.ndarray, remaining: tuple[np.ndarray, ...]) -> np.ndarray:
            if not remaining:
                return np.asarray(self.value(base), dtype=complex)
            direction = remaining[-1]
            scale = self.step / max(1.0, float(np.linalg.norm(direction)))
            return (
                recurse(base + scale * direction, remaining[:-1])
                - recurse(base - scale * direction, remaining[:-1])
            ) / (2.0 * scale)

        result = recurse(x, direction_tuple)
        return np.real_if_close(result, tol=1000)


@documented(
    ProvenanceRecord(
        identifier="PSC-III-D04",
        title="Symmetric polynomial derivative tensors",
        status=ClaimStatus.DEFINITION,
        source="Paper III runtime",
        assumptions=("declared symmetric coefficient tensors",),
        verification_suite="nonlinear.derivatives",
    )
)
class PolynomialDerivativeProvider:
    """A finite polynomial map represented by symmetric coefficient tensors.

    For tensors ``T_r`` the map is

    ``F(x) = sum_r T_r[x, ..., x] / r!``.

    ``T_0`` is a target vector.  A tensor of order ``r`` has shape
    ``(target_dimension, source_dimension, ..., source_dimension)`` with ``r``
    source axes.
    """

    def __init__(
        self,
        tensors: Mapping[int, Any],
        *,
        source_dimension: int | None = None,
        target_dimension: int | None = None,
        exactness: Exactness = Exactness.NUMERICAL,
        tolerance: float = 1e-10,
        name: str = "polynomial derivative provider",
    ) -> None:
        if not tensors:
            raise ValidationError("a polynomial provider requires at least one tensor")
        normalized_exactness = Exactness(exactness)
        tol = finite_tolerance(tolerance)
        normalized: dict[int, np.ndarray] = {}
        for raw_order, value in tensors.items():
            order = int(raw_order)
            if isinstance(raw_order, bool) or order != raw_order or order < 0:
                raise ValidationError("polynomial tensor orders must be nonnegative integers")
            tensor = np.asarray(
                value,
                dtype=object if normalized_exactness is Exactness.EXACT else None,
            )
            if normalized_exactness is Exactness.EXACT:
                tensor = tensor.copy()
                for index in np.ndindex(tensor.shape):
                    try:
                        if bool(tensor[index] == 0):
                            tensor[index] = 0
                    except (TypeError, ValueError):
                        pass
            normalized[order] = tensor
        inferred_target: int | None = target_dimension
        inferred_source: int | None = source_dimension
        for order, tensor in normalized.items():
            expected_rank = order + 1
            if tensor.ndim != expected_rank:
                raise DimensionError(f"tensor T_{order} must have rank {expected_rank}")
            if inferred_target is None:
                inferred_target = int(tensor.shape[0])
            if int(tensor.shape[0]) != int(inferred_target):
                raise DimensionError("all polynomial tensors must share one target dimension")
            if order:
                axes = tuple(int(length) for length in tensor.shape[1:])
                if len(set(axes)) != 1:
                    raise DimensionError("every source axis of a polynomial tensor must have one dimension")
                if inferred_source is None:
                    inferred_source = axes[0]
                if any(length != int(inferred_source) for length in axes):
                    raise DimensionError("all polynomial tensors must share one source dimension")
        if inferred_target is None or inferred_source is None:
            raise DimensionError("source and target dimensions could not be inferred")
        self.source_dimension = int(inferred_source)
        self.target_dimension = int(inferred_target)
        self.tensors = normalized
        self.exactness = normalized_exactness
        self.tolerance = tol
        self.name = str(name)

    @property
    def degree(self) -> int:
        return max(self.tensors)

    def supports_order(self, order: int) -> bool:
        n = int(order)
        return not isinstance(order, bool) and n == order and 1 <= n <= self.degree

    @staticmethod
    def _contract(tensor: np.ndarray, vectors: Sequence[Any]) -> Any:
        result: Any = tensor
        for vector in vectors:
            result = np.tensordot(result, np.asarray(vector), axes=(1, 0))
        return result

    def _scale_factorial(self, value: Any, order: int) -> Any:
        scalar: Any = Fraction(1, factorial(order)) if self.exactness is Exactness.EXACT else 1 / factorial(order)
        return scalar * value

    def value(self, point: Any) -> Any:
        x = np.asarray(point)
        if x.ndim == 2 and 1 in x.shape:
            x = x.reshape(-1)
        if x.ndim != 1 or x.shape[0] != self.source_dimension:
            raise DimensionError("point dimension does not match the polynomial provider")
        dtype = object if any(tensor.dtype == object for tensor in self.tensors.values()) else np.result_type(*self.tensors.values(), x)
        total = np.zeros(self.target_dimension, dtype=dtype)
        for order, tensor in self.tensors.items():
            total = total + self._scale_factorial(self._contract(tensor, (x,) * order), order)
        return total

    def derivative(self, order: int, point: Any, directions: Sequence[Any]) -> Any:
        n = int(order)
        if not self.supports_order(n):
            raise ValidationError(f"polynomial derivative order must lie between 1 and {self.degree}")
        x = np.asarray(point)
        if x.ndim == 2 and 1 in x.shape:
            x = x.reshape(-1)
        if x.ndim != 1 or x.shape[0] != self.source_dimension:
            raise DimensionError("point dimension does not match the polynomial provider")
        direction_tuple = tuple(np.asarray(direction).reshape(-1) for direction in directions)
        if len(direction_tuple) != n:
            raise ValidationError(f"order {n} requires exactly {n} directions")
        if any(direction.shape != (self.source_dimension,) for direction in direction_tuple):
            raise DimensionError("direction dimension does not match the polynomial provider")
        dtype = object if any(tensor.dtype == object for tensor in self.tensors.values()) else np.result_type(*self.tensors.values(), x, *direction_tuple)
        total = np.zeros(self.target_dimension, dtype=dtype)
        for tensor_order in range(n, self.degree + 1):
            tensor = self.tensors.get(tensor_order)
            if tensor is None:
                continue
            vectors = direction_tuple + (x,) * (tensor_order - n)
            total = total + self._scale_factorial(
                self._contract(tensor, vectors), tensor_order - n
            )
        return total

    def symmetry_certificate(self) -> Certificate:
        audit = value_audit(exact=self.exactness is Exactness.EXACT, tolerance=self.tolerance)
        residual = 0.0
        nonfinite_orders = []
        for order, tensor in self.tensors.items():
            reference = np.asarray(tensor)
            try:
                coefficient_storage_decision(reference.reshape(-1),
                    exact=self.exactness is Exactness.EXACT, tolerance=self.tolerance)
            except ValidationError:
                nonfinite_orders.append(order)
            if order < 2:
                continue
            for left in range(1, order):
                swapped = np.swapaxes(reference, left, left + 1)
                residual = max(residual, audit.check_value(
                    f"tensor:{order}:swap:{left}", (reference-swapped).reshape(-1)))
        valid = audit.valid and not nonfinite_orders
        return Certificate(
            valid=bool(valid),
            exactness=self.exactness,
            residuals={"tensor_symmetry": residual},
            tolerance=None if self.exactness is Exactness.EXACT else self.tolerance,
            assumptions_checked=("tensor dimensions", "adjacent source-axis symmetry"),
            assumptions_unchecked=audit.unresolved,
            diagnostics={"residual_decisions": audit.decisions,
                         "nonfinite_tensor_orders": nonfinite_orders},
            theorem_id="PSC-III-D04",
            message="polynomial tensors are symmetric" if valid else "polynomial tensors are not symmetric",
        )


@documented(
    ProvenanceRecord(
        identifier="PSC-III-E05",
        title="Exact symbolic directional derivatives",
        status=ClaimStatus.FINITE_CERTIFICATE,
        source="SymPy adapter",
        assumptions=("SymPy expressions and substitution values are exact",),
        verification_suite="nonlinear.derivatives",
    )
)
class SympyDerivativeProvider:
    """Exact directional derivatives of a finite SymPy expression vector."""

    exactness = Exactness.EXACT

    def __init__(self, expressions: Sequence[Any], variables: Sequence[Any], *, name: str = "SymPy derivative provider") -> None:
        try:
            import sympy as sp
        except ImportError as exc:  # pragma: no cover - optional dependency boundary
            from ..exceptions import BackendUnavailableError

            raise BackendUnavailableError("SymPy is required for SympyDerivativeProvider") from exc
        self._sp = sp
        self.expressions = tuple(sp.sympify(expression) for expression in expressions)
        self.variables = tuple(variables)
        if not self.expressions or not self.variables:
            raise ValidationError("symbolic derivative providers require expressions and variables")
        if len(set(self.variables)) != len(self.variables):
            raise ValidationError("symbolic variables must be distinct")
        self.source_dimension = len(self.variables)
        self.target_dimension = len(self.expressions)
        self.name = str(name)

    def supports_order(self, order: int) -> bool:
        n = int(order)
        return not isinstance(order, bool) and n == order and n >= 1

    def _vector(self, value: Any, *, name: str) -> tuple[Any, ...]:
        matrix = self._sp.Matrix(value)
        entries = tuple(matrix)
        if len(entries) != self.source_dimension:
            raise DimensionError(f"{name} dimension does not match the symbolic provider")
        return entries

    def value(self, point: Any) -> Any:
        entries = self._vector(point, name="point")
        substitutions = dict(zip(self.variables, entries, strict=True))
        return self._sp.Matrix([expression.subs(substitutions) for expression in self.expressions])

    def derivative(self, order: int, point: Any, directions: Sequence[Any]) -> Any:
        n = int(order)
        if not self.supports_order(n):
            raise ValidationError("derivative order must be a positive integer")
        direction_tuple = tuple(self._vector(direction, name="direction") for direction in directions)
        if len(direction_tuple) != n:
            raise ValidationError(f"order {n} requires exactly {n} directions")
        current = self.expressions
        for direction in direction_tuple:
            current = tuple(
                sum(self._sp.diff(expression, variable) * coefficient for variable, coefficient in zip(self.variables, direction, strict=True))
                for expression in current
            )
        entries = self._vector(point, name="point")
        substitutions = dict(zip(self.variables, entries, strict=True))
        return self._sp.Matrix([expression.subs(substitutions) for expression in current])


__all__ = [
    "CallableDerivativeProvider",
    "DerivativeProvider",
    "DirectionalDerivative",
    "FiniteDifferenceDerivativeProvider",
    "PolynomialDerivativeProvider",
    "SympyDerivativeProvider",
]
