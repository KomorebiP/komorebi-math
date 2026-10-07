"""Backend discovery and selection."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any

from ..exceptions import BackendUnavailableError
from .base import LinearAlgebraBackend
from .numpy import NumpyBackend

_BACKENDS: dict[str, LinearAlgebraBackend] = {"numpy": NumpyBackend()}


def register_backend(backend: LinearAlgebraBackend, *, replace: bool = False) -> None:
    if backend.name in _BACKENDS and not replace:
        raise ValueError(f"backend already registered: {backend.name}")
    _BACKENDS[backend.name] = backend


def get_backend(backend: str | LinearAlgebraBackend | None = None) -> LinearAlgebraBackend:
    if isinstance(backend, LinearAlgebraBackend):
        return backend
    name = "numpy" if backend is None else backend
    if name == "sympy" and name not in _BACKENDS:
        if find_spec("sympy") is None:
            raise BackendUnavailableError(
                "the SymPy backend requires the 'exact' optional dependency"
            )
        from .sympy import SympyBackend

        _BACKENDS[name] = SympyBackend()
    try:
        return _BACKENDS[name]
    except KeyError as exc:
        raise BackendUnavailableError(f"unknown or unavailable backend: {name}") from exc


def available_backends() -> dict[str, bool]:
    return {
        "numpy": True,
        "sympy": find_spec("sympy") is not None,
        "scipy": find_spec("scipy") is not None,
        "mpmath": find_spec("mpmath") is not None,
        "cupy": find_spec("cupy") is not None,
    }


def backend_name(value: Any) -> str:
    module = type(value).__module__.split(".", 1)[0]
    if module == "sympy":
        return "sympy"
    return "numpy"


__all__ = [
    "LinearAlgebraBackend",
    "NumpyBackend",
    "available_backends",
    "backend_name",
    "get_backend",
    "register_backend",
]
