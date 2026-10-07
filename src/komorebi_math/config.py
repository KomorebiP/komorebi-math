"""Runtime settings for backend and validation defaults."""

from __future__ import annotations

from ._residuals import finite_tolerance

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Iterator


@dataclass(frozen=True, slots=True)
class Settings:
    backend: str = "numpy"
    tolerance: float = 1e-10
    validate_on_construction: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "tolerance", finite_tolerance(self.tolerance))


_SETTINGS: ContextVar[Settings] = ContextVar("komorebi_math_settings", default=Settings())


def get_settings() -> Settings:
    """Return the active runtime settings for the current context."""

    return _SETTINGS.get()


def configure(
    *,
    backend: str | None = None,
    tolerance: float | None = None,
    validate_on_construction: bool | None = None,
) -> Settings:
    """Update and return the runtime settings in the current context."""

    current = get_settings()
    updated = replace(
        current,
        backend=current.backend if backend is None else backend,
        tolerance=current.tolerance if tolerance is None else float(tolerance),
        validate_on_construction=(
            current.validate_on_construction
            if validate_on_construction is None
            else bool(validate_on_construction)
        ),
    )
    _SETTINGS.set(updated)
    return updated


@contextmanager
def settings_context(**changes: object) -> Iterator[Settings]:
    """Temporarily apply runtime setting changes inside a context manager."""

    current = get_settings()
    updated = replace(current, **changes)
    token = _SETTINGS.set(updated)
    try:
        yield updated
    finally:
        _SETTINGS.reset(token)
