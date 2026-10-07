"""Optional automatic-differentiation adapters for directional derivatives.

Every adapter implements the existing :class:`DerivativeProvider` protocol.
Imports of external frameworks are lazy, so the base NumPy installation does
not acquire JAX or PyTorch as import-time dependencies.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from itertools import permutations
import json
import subprocess
import sys
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..certificates import Certificate, Exactness
from ..exceptions import BackendUnavailableError, DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented


@runtime_checkable
class DirectionalJVP(Protocol):
    """One forward-mode Jacobian-vector product primitive."""

    def __call__(
        self,
        function: Callable[[Any], Any],
        point: Any,
        direction: Any,
    ) -> tuple[Any, Any]:
        """Return ``(function(point), D function(point)[direction])``."""


def _vector_length(value: Any) -> int:
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
    raise DimensionError("automatic-differentiation values must be vector-like")


def _norm(value: Any) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=complex).reshape(-1)))


def _subtract(left: Any, right: Any) -> Any:
    try:
        return left - right
    except TypeError:
        return np.asarray(left, dtype=object) - np.asarray(right, dtype=object)


@dataclass(frozen=True, slots=True)
class _ADFrameworkProbe:
    """Result of one crash-isolated optional-backend capability probe."""

    framework: str
    discovered: bool
    available: bool
    version: str | None = None
    returncode: int | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "discovered": self.discovered,
            "available": self.available,
            "version": self.version,
            "returncode": self.returncode,
            "detail": self.detail,
        }


_FRAMEWORK_PROBE_PROGRAMS = {
    "jax": r"""
import json
import jax
import jax.numpy as jnp
available = callable(getattr(jax, "jvp", None)) and callable(getattr(jnp, "asarray", None))
print(json.dumps({
    "available": bool(available),
    "version": str(getattr(jax, "__version__", "unknown")),
    "detail": "jax.jvp and jax.numpy.asarray are available" if available else "required JAX capabilities are absent",
}))
raise SystemExit(0 if available else 3)
""",
    "torch": r"""
import json
import torch
available = hasattr(torch, "func") and callable(getattr(torch.func, "jvp", None))
print(json.dumps({
    "available": bool(available),
    "version": str(getattr(torch, "__version__", "unknown")),
    "detail": "torch.func.jvp is available" if available else "required PyTorch capability torch.func.jvp is absent",
}))
raise SystemExit(0 if available else 3)
""",
}


def _short_probe_detail(stdout: str, stderr: str, *, limit: int = 600) -> str:
    """Return a bounded one-line diagnostic from subprocess output."""

    combined = " | ".join(
        part.strip().replace("\r", " ").replace("\n", " ")
        for part in (stderr, stdout)
        if part.strip()
    )
    if not combined:
        return "backend probe terminated without diagnostic output"
    return combined[-limit:]


@lru_cache(maxsize=2)
def _probe_ad_framework(framework: str) -> _ADFrameworkProbe:
    """Probe one optional AD runtime in a separate interpreter process.

    A native extension can terminate Python during import.  Running the probe
    out of process prevents a broken optional JAX or PyTorch installation from
    terminating Komorebi Math, pytest, or an application that only wants to
    inspect backend availability.
    """

    name = str(framework).strip().lower()
    if name not in _FRAMEWORK_PROBE_PROGRAMS:
        raise ValidationError(f"unknown automatic-differentiation framework: {framework}")
    try:
        discovered = find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        discovered = False
    if not discovered:
        return _ADFrameworkProbe(
            framework=name,
            discovered=False,
            available=False,
            detail=f"{name} is not installed in the active Python environment",
        )

    try:
        completed = subprocess.run(
            (sys.executable, "-c", _FRAMEWORK_PROBE_PROGRAMS[name]),
            check=False,
            capture_output=True,
            text=True,
            timeout=60.0,
        )
    except subprocess.TimeoutExpired as exc:
        return _ADFrameworkProbe(
            framework=name,
            discovered=True,
            available=False,
            detail=f"{name} import probe exceeded {exc.timeout} seconds",
        )
    except OSError as exc:
        return _ADFrameworkProbe(
            framework=name,
            discovered=True,
            available=False,
            detail=f"could not start isolated {name} probe: {exc}",
        )

    payload: dict[str, Any] = {}
    if completed.stdout.strip():
        try:
            candidate = json.loads(completed.stdout.strip().splitlines()[-1])
            if isinstance(candidate, dict):
                payload = candidate
        except json.JSONDecodeError:
            payload = {}
    available = bool(completed.returncode == 0 and payload.get("available", False))
    detail = str(payload.get("detail", "")).strip()
    if not detail:
        detail = _short_probe_detail(completed.stdout, completed.stderr)
    return _ADFrameworkProbe(
        framework=name,
        discovered=True,
        available=available,
        version=(None if payload.get("version") is None else str(payload.get("version"))),
        returncode=int(completed.returncode),
        detail=detail,
    )


def ad_framework_diagnostics(*, refresh: bool = False) -> dict[str, dict[str, Any]]:
    """Return crash-isolated discovery, import, and capability diagnostics."""

    if refresh:
        _probe_ad_framework.cache_clear()
    return {
        name: _probe_ad_framework(name).to_dict()
        for name in ("jax", "torch")
    }


def available_ad_frameworks() -> dict[str, bool]:
    """Report usable optional AD frameworks without importing them in-process.

    ``find_spec`` alone is insufficient because an installed native extension
    may still fail or abort while importing.  Availability therefore means that
    a clean child interpreter imported the framework and found the directional
    JVP capability required by the adapter.
    """

    diagnostics = ad_framework_diagnostics()
    return {name: bool(status["available"]) for name, status in diagnostics.items()}


@documented(
    ProvenanceRecord(
        identifier="PSC-III-N04A",
        title="Directional automatic-differentiation adapter",
        status=ClaimStatus.DEFINITION,
        source="Paper III scalable runtime",
        assumptions=("declared forward-mode JVP primitive",),
        verification_suite="nonlinear.autodiff",
    )
)
@dataclass(slots=True)
class AutomaticDifferentiationProvider:
    """Framework-neutral higher derivatives obtained by nested JVP calls.

    The supplied ``jvp`` primitive must preserve its framework's tracing or
    dual-number state when it calls the function.  Nested JVPs then compute
    higher directional Fréchet derivatives without changing the common
    derivative-provider protocol.
    """

    function: Callable[[Any], Any]
    jvp: DirectionalJVP
    source_dimension: int
    target_dimension: int
    max_order: int = 4
    exactness: Exactness = Exactness.NUMERICAL
    name: str = "automatic differentiation provider"

    def __post_init__(self) -> None:
        self.source_dimension = int(self.source_dimension)
        self.target_dimension = int(self.target_dimension)
        self.max_order = int(self.max_order)
        self.exactness = Exactness(self.exactness)
        self.name = str(self.name)
        if self.source_dimension < 1 or self.target_dimension < 1:
            raise DimensionError("source and target dimensions must be positive")
        if self.max_order < 1:
            raise ValidationError("automatic-differentiation max_order must be positive")
        if not callable(self.function) or not callable(self.jvp):
            raise ValidationError("function and jvp must be callable")

    def supports_order(self, order: int) -> bool:
        try:
            normalized = int(order)
        except (TypeError, ValueError):
            return False
        return (
            not isinstance(order, bool)
            and normalized == order
            and 1 <= normalized <= self.max_order
        )

    def _validate_point(self, point: Any, *, name: str = "point") -> None:
        if _vector_length(point) != self.source_dimension:
            raise DimensionError(f"{name} dimension does not match the AD provider")

    def _validate_output(self, value: Any) -> Any:
        if _vector_length(value) != self.target_dimension:
            raise DimensionError("AD output dimension does not match the provider")
        return value

    def value(self, point: Any) -> Any:
        self._validate_point(point)
        return self._validate_output(self.function(point))

    def _directional(
        self,
        function: Callable[[Any], Any],
        point: Any,
        directions: tuple[Any, ...],
    ) -> Any:
        if not directions:
            return function(point)
        prefix = directions[:-1]
        final_direction = directions[-1]

        def lower_derivative(base: Any) -> Any:
            return self._directional(function, base, prefix)

        result = self.jvp(lower_derivative, point, final_direction)
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValidationError("a directional JVP primitive must return a (primal, tangent) pair")
        return result[1]

    def derivative(self, order: int, point: Any, directions: Sequence[Any]) -> Any:
        derivative_order = int(order)
        if not self.supports_order(derivative_order):
            raise ValidationError(
                f"automatic-differentiation order must lie between 1 and {self.max_order}"
            )
        self._validate_point(point)
        direction_tuple = tuple(directions)
        if len(direction_tuple) != derivative_order:
            raise ValidationError(
                f"order {derivative_order} requires exactly {derivative_order} directions"
            )
        for direction in direction_tuple:
            self._validate_point(direction, name="direction")
        return self._validate_output(
            self._directional(self.function, point, direction_tuple)
        )

    def validation_certificate(
        self,
        point: Any,
        directions: Sequence[Any],
        *,
        tolerance: float = 1e-8,
    ) -> Certificate:
        """Check finite permutation symmetry of available derivative orders."""

        self._validate_point(point)
        probes = tuple(directions)
        residual = 0.0
        checked_orders: list[int] = []
        for order in range(1, min(self.max_order, len(probes)) + 1):
            selected = probes[:order]
            reference = self.derivative(order, point, selected)
            for permutation in permutations(range(order)):
                candidate = self.derivative(
                    order,
                    point,
                    tuple(selected[index] for index in permutation),
                )
                residual = max(residual, _norm(_subtract(candidate, reference)))
            checked_orders.append(order)
        threshold = 0.0 if self.exactness is Exactness.EXACT else float(tolerance)
        valid = residual <= threshold
        return Certificate(
            valid=valid,
            exactness=self.exactness,
            residuals={"directional_symmetry": residual},
            tolerance=None if self.exactness is Exactness.EXACT else threshold,
            assumptions_checked=("finite source and target dimensions", "permutation symmetry probes"),
            assumptions_unchecked=(
                "global correctness of the supplied JVP primitive",
                "framework tracing outside the tested points",
            ),
            diagnostics={"provider": self.name, "checked_orders": tuple(checked_orders)},
            theorem_id="PSC-III-N04A",
            message="nested JVP derivatives passed finite symmetry checks" if valid else "nested JVP derivative symmetry failed",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_dimension": self.source_dimension,
            "target_dimension": self.target_dimension,
            "max_order": self.max_order,
            "exactness": self.exactness.value,
        }


class _FrameworkProviderBase:
    source_dimension: int
    target_dimension: int
    max_order: int
    exactness: Exactness
    name: str

    def supports_order(self, order: int) -> bool:
        try:
            normalized = int(order)
        except (TypeError, ValueError):
            return False
        return (
            not isinstance(order, bool)
            and normalized == order
            and 1 <= normalized <= self.max_order
        )

    def _validate_dimensions(self) -> None:
        self.source_dimension = int(self.source_dimension)
        self.target_dimension = int(self.target_dimension)
        self.max_order = int(self.max_order)
        if self.source_dimension < 1 or self.target_dimension < 1:
            raise DimensionError("source and target dimensions must be positive")
        if self.max_order < 1:
            raise ValidationError("automatic-differentiation max_order must be positive")
        if not callable(self.function):
            raise ValidationError("automatic-differentiation function must be callable")

    def _validate_input(self, value: Any, *, name: str) -> None:
        if _vector_length(value) != self.source_dimension:
            raise DimensionError(f"{name} dimension does not match the AD provider")

    def _validate_output_array(self, value: Any) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim == 2 and 1 in array.shape:
            array = array.reshape(-1)
        if array.ndim != 1 or int(array.shape[0]) != self.target_dimension:
            raise DimensionError("AD output dimension does not match the provider")
        return array

    def validation_certificate(
        self,
        point: Any,
        directions: Sequence[Any],
        *,
        tolerance: float = 1e-8,
    ) -> Certificate:
        """Check finite permutation symmetry through the available probe order."""

        self._validate_input(point, name="point")
        probes = tuple(directions)
        residual = 0.0
        checked_orders: list[int] = []
        for order in range(1, min(self.max_order, len(probes)) + 1):
            selected = probes[:order]
            reference = self.derivative(order, point, selected)
            for permutation in permutations(range(order)):
                candidate = self.derivative(
                    order,
                    point,
                    tuple(selected[index] for index in permutation),
                )
                residual = max(residual, _norm(_subtract(candidate, reference)))
            checked_orders.append(order)
        threshold = float(tolerance)
        if not np.isfinite(threshold) or threshold < 0:
            raise ValidationError("AD validation tolerance must be finite and nonnegative")
        valid = residual <= threshold
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"directional_symmetry": residual},
            tolerance=threshold,
            assumptions_checked=(
                "finite source and target dimensions",
                "permutation symmetry probes",
            ),
            assumptions_unchecked=(
                "global correctness of the external automatic-differentiation runtime",
                "framework tracing outside the tested points",
            ),
            diagnostics={"provider": self.name, "checked_orders": tuple(checked_orders)},
            theorem_id="PSC-III-N04A",
            message=(
                "automatic-differentiation derivatives passed finite symmetry checks"
                if valid
                else "automatic-differentiation derivative symmetry failed"
            ),
        )


@documented(
    ProvenanceRecord(
        identifier="PSC-III-N04J",
        title="Lazy JAX derivative adapter",
        status=ClaimStatus.FINITE_CERTIFICATE,
        source="External automatic-differentiation adapter",
        assumptions=("compatible JAX installation", "JAX-traceable function"),
        limitations=("device and precision behavior are controlled by the external JAX runtime",),
        verification_suite="nonlinear.autodiff",
    )
)
@dataclass(slots=True)
class JaxDerivativeProvider(_FrameworkProviderBase):
    """Nested ``jax.jvp`` derivatives with lazy framework import."""

    function: Callable[[Any], Any]
    source_dimension: int
    target_dimension: int
    max_order: int = 4
    dtype: str | None = None
    name: str = "JAX derivative provider"
    exactness: Exactness = Exactness.NUMERICAL

    def __post_init__(self) -> None:
        self._validate_dimensions()
        self.exactness = Exactness.NUMERICAL

    @staticmethod
    def available() -> bool:
        return _probe_ad_framework("jax").available

    def _framework(self) -> tuple[Any, Any]:
        if not self.available():
            status = _probe_ad_framework("jax")
            raise BackendUnavailableError(
                f"JAX is not installed or unavailable: {status.detail}"
            )
        try:
            jax = import_module("jax")
            jnp = import_module("jax.numpy")
        except Exception as exc:
            raise BackendUnavailableError(f"JAX could not be imported: {exc}") from exc
        return jax, jnp

    def _array(self, value: Any, jnp: Any) -> Any:
        return jnp.asarray(value, dtype=self.dtype) if self.dtype is not None else jnp.asarray(value)

    def _raw_directional(
        self,
        jax: Any,
        function: Callable[[Any], Any],
        point: Any,
        directions: tuple[Any, ...],
    ) -> Any:
        if not directions:
            return function(point)
        prefix = directions[:-1]
        direction = directions[-1]
        return jax.jvp(
            lambda base: self._raw_directional(jax, function, base, prefix),
            (point,),
            (direction,),
        )[1]

    def value(self, point: Any) -> np.ndarray:
        self._validate_input(point, name="point")
        _, jnp = self._framework()
        value = self.function(self._array(point, jnp))
        return self._validate_output_array(value)

    def derivative(self, order: int, point: Any, directions: Sequence[Any]) -> np.ndarray:
        derivative_order = int(order)
        if not self.supports_order(derivative_order):
            raise ValidationError(f"JAX derivative order must lie between 1 and {self.max_order}")
        self._validate_input(point, name="point")
        direction_tuple = tuple(directions)
        if len(direction_tuple) != derivative_order:
            raise ValidationError(
                f"order {derivative_order} requires exactly {derivative_order} directions"
            )
        for direction in direction_tuple:
            self._validate_input(direction, name="direction")
        jax, jnp = self._framework()
        x = self._array(point, jnp)
        tangent_tuple = tuple(self._array(direction, jnp) for direction in direction_tuple)
        value = self._raw_directional(jax, self.function, x, tangent_tuple)
        return self._validate_output_array(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": "jax",
            "available": self.available(),
            "source_dimension": self.source_dimension,
            "target_dimension": self.target_dimension,
            "max_order": self.max_order,
            "dtype": self.dtype,
        }


@documented(
    ProvenanceRecord(
        identifier="PSC-III-N04T",
        title="Lazy PyTorch derivative adapter",
        status=ClaimStatus.FINITE_CERTIFICATE,
        source="External automatic-differentiation adapter",
        assumptions=("compatible PyTorch installation", "torch.func.jvp-compatible function"),
        limitations=("device and precision behavior are controlled by the external PyTorch runtime",),
        verification_suite="nonlinear.autodiff",
    )
)
@dataclass(slots=True)
class TorchDerivativeProvider(_FrameworkProviderBase):
    """Nested ``torch.func.jvp`` derivatives with lazy framework import."""

    function: Callable[[Any], Any]
    source_dimension: int
    target_dimension: int
    max_order: int = 4
    dtype: str = "float64"
    device: str | None = None
    name: str = "PyTorch derivative provider"
    exactness: Exactness = Exactness.NUMERICAL

    def __post_init__(self) -> None:
        self._validate_dimensions()
        self.exactness = Exactness.NUMERICAL

    @staticmethod
    def available() -> bool:
        return _probe_ad_framework("torch").available

    def _framework(self) -> Any:
        if not self.available():
            status = _probe_ad_framework("torch")
            raise BackendUnavailableError(
                f"PyTorch is not installed or unavailable: {status.detail}"
            )
        try:
            torch = import_module("torch")
        except Exception as exc:
            raise BackendUnavailableError(f"PyTorch could not be imported: {exc}") from exc
        if not hasattr(torch, "func") or not hasattr(torch.func, "jvp"):
            raise BackendUnavailableError("the installed PyTorch build does not provide torch.func.jvp")
        return torch

    def _array(self, value: Any, torch: Any) -> Any:
        try:
            dtype = getattr(torch, self.dtype)
        except AttributeError as exc:
            raise ValidationError(f"unknown PyTorch dtype: {self.dtype}") from exc
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _raw_directional(
        self,
        torch: Any,
        function: Callable[[Any], Any],
        point: Any,
        directions: tuple[Any, ...],
    ) -> Any:
        if not directions:
            return function(point)
        prefix = directions[:-1]
        direction = directions[-1]
        return torch.func.jvp(
            lambda base: self._raw_directional(torch, function, base, prefix),
            (point,),
            (direction,),
        )[1]

    @staticmethod
    def _host(value: Any) -> np.ndarray:
        return np.asarray(value.detach().cpu().numpy())

    def value(self, point: Any) -> np.ndarray:
        self._validate_input(point, name="point")
        torch = self._framework()
        value = self.function(self._array(point, torch))
        return self._validate_output_array(self._host(value))

    def derivative(self, order: int, point: Any, directions: Sequence[Any]) -> np.ndarray:
        derivative_order = int(order)
        if not self.supports_order(derivative_order):
            raise ValidationError(
                f"PyTorch derivative order must lie between 1 and {self.max_order}"
            )
        self._validate_input(point, name="point")
        direction_tuple = tuple(directions)
        if len(direction_tuple) != derivative_order:
            raise ValidationError(
                f"order {derivative_order} requires exactly {derivative_order} directions"
            )
        for direction in direction_tuple:
            self._validate_input(direction, name="direction")
        torch = self._framework()
        x = self._array(point, torch)
        tangent_tuple = tuple(self._array(direction, torch) for direction in direction_tuple)
        value = self._raw_directional(torch, self.function, x, tangent_tuple)
        return self._validate_output_array(self._host(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": "torch",
            "available": self.available(),
            "source_dimension": self.source_dimension,
            "target_dimension": self.target_dimension,
            "max_order": self.max_order,
            "dtype": self.dtype,
            "device": self.device,
        }


def autodiff_provider(
    framework: str,
    function: Callable[[Any], Any],
    source_dimension: int,
    target_dimension: int,
    *,
    max_order: int = 4,
    **options: Any,
) -> JaxDerivativeProvider | TorchDerivativeProvider:
    """Construct one supported external automatic-differentiation adapter."""

    name = str(framework).strip().lower()
    if name == "jax":
        return JaxDerivativeProvider(
            function,
            source_dimension,
            target_dimension,
            max_order=max_order,
            **options,
        )
    if name in {"torch", "pytorch"}:
        return TorchDerivativeProvider(
            function,
            source_dimension,
            target_dimension,
            max_order=max_order,
            **options,
        )
    raise ValidationError(f"unknown automatic-differentiation framework: {framework}")


# Descriptive aliases retained for the Block 10 public API.
available_autodiff_backends = available_ad_frameworks
automatic_derivative_provider = autodiff_provider


__all__ = [
    "AutomaticDifferentiationProvider",
    "DirectionalJVP",
    "JaxDerivativeProvider",
    "TorchDerivativeProvider",
    "ad_framework_diagnostics",
    "autodiff_provider",
    "automatic_derivative_provider",
    "available_ad_frameworks",
    "available_autodiff_backends",
]
