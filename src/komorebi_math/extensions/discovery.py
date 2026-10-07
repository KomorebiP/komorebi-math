"""Extension discovery through standard Python entry points."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points

from .api import ExtensionSpec

ENTRY_POINT_GROUP = "komorebi_math.extensions"


@dataclass(frozen=True, slots=True)
class ExtensionLoadResult:
    entry_point: str
    loaded: bool
    extension: ExtensionSpec | None = None
    error: str | None = None


def _group_entry_points() -> tuple[EntryPoint, ...]:
    discovered = entry_points()
    if hasattr(discovered, "select"):
        return tuple(discovered.select(group=ENTRY_POINT_GROUP))
    return tuple(discovered.get(ENTRY_POINT_GROUP, ()))


def discover_extensions(*, activate: bool = False) -> tuple[ExtensionLoadResult, ...]:
    results: list[ExtensionLoadResult] = []
    for point in _group_entry_points():
        try:
            loaded = point.load()
            extension = loaded() if callable(loaded) and not isinstance(loaded, ExtensionSpec) else loaded
            if not isinstance(extension, ExtensionSpec):
                raise TypeError("extension entry point did not return ExtensionSpec")
            if activate:
                extension.activate()
            results.append(ExtensionLoadResult(point.name, True, extension=extension))
        except Exception as exc:
            results.append(ExtensionLoadResult(point.name, False, error=f"{type(exc).__name__}: {exc}"))
    return tuple(results)
