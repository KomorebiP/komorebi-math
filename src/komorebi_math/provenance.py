"""Mathematical provenance registry for public objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from importlib import import_module
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class ClaimStatus(StrEnum):
    DEFINITION = "definition"
    PROVED = "proved"
    CLASSICAL = "classical input"
    CONDITIONAL = "conditional theorem"
    FINITE_CERTIFICATE = "finite certificate"
    HEURISTIC = "heuristic"
    CONJECTURE = "conjecture"
    PLANNED = "planned"


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    identifier: str
    title: str
    status: ClaimStatus
    source: str
    assumptions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    verification_suite: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


_RECORDS: dict[str, ProvenanceRecord] = {}


def documented(record: ProvenanceRecord) -> Callable[[F], F]:
    """Attach a provenance record to a function or class."""

    def decorator(obj: F) -> F:
        key = f"{obj.__module__}.{obj.__qualname__}"
        _RECORDS[key] = record
        setattr(obj, "__komorebi_provenance__", record)
        return obj

    return decorator


def register_record(qualified_name: str, record: ProvenanceRecord) -> None:
    _RECORDS[qualified_name] = record


def describe(obj: Any) -> ProvenanceRecord | None:
    """Return the attached record for an object or qualified name."""
    if isinstance(obj, str):
        if obj in _RECORDS:
            return _RECORDS[obj]
        module_name, _, attribute_path = obj.rpartition(".")
        if not module_name:
            return None
        try:
            current: Any = import_module(module_name)
            for part in attribute_path.split("."):
                current = getattr(current, part)
        except (ImportError, AttributeError):
            return None
        return getattr(current, "__komorebi_provenance__", None)
    return getattr(obj, "__komorebi_provenance__", None)


def records() -> dict[str, ProvenanceRecord]:
    return dict(_RECORDS)
