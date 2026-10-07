"""Normalization helpers for portable data export."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, TypeVar, runtime_checkable

import numpy as np

from ..certificates import Certificate, ComputationResult, ConditionalResult
from ..exceptions import ValidationError

T = TypeVar("T")
ExportAdapter = Callable[[Any], Any]


@runtime_checkable
class Exportable(Protocol):
    """Protocol for objects that provide their own portable export data."""

    def to_export_data(self) -> Any:
        """Return a recursively normalizable representation."""


_ADAPTERS: dict[type[Any], ExportAdapter] = {}


def register_export_adapter(type_: type[T], adapter: Callable[[T], Any], *, replace: bool = False) -> None:
    """Register a normalization adapter for one Python type."""

    if type_ in _ADAPTERS and not replace:
        raise ValidationError(f"an export adapter is already registered for {type_.__name__}")
    _ADAPTERS[type_] = adapter


def export_adapters() -> dict[type[Any], ExportAdapter]:
    """Return a copy of the active export adapter registry."""

    return dict(_ADAPTERS)


def _adapter_for(value: Any) -> ExportAdapter | None:
    for type_ in type(value).__mro__:
        adapter = _ADAPTERS.get(type_)
        if adapter is not None:
            return adapter
    return None


def to_serializable(
    value: Any,
    *,
    preserve_types: bool = True,
    max_depth: int = 64,
    _depth: int = 0,
) -> Any:
    """Convert package and scientific Python objects to JSON compatible data."""

    if _depth > max_depth:
        raise ValidationError("maximum export normalization depth exceeded")
    next_depth = _depth + 1

    adapter = _adapter_for(value)
    if adapter is not None:
        return to_serializable(
            adapter(value),
            preserve_types=preserve_types,
            max_depth=max_depth,
            _depth=next_depth,
        )

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return to_serializable(
            value.item(),
            preserve_types=preserve_types,
            max_depth=max_depth,
            _depth=next_depth,
        )
    if isinstance(value, Fraction):
        if preserve_types:
            return {
                "__type__": "Fraction",
                "numerator": value.numerator,
                "denominator": value.denominator,
            }
        return str(value)
    if isinstance(value, complex):
        if preserve_types:
            return {"__type__": "complex", "real": value.real, "imag": value.imag}
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return to_serializable(
            value.value,
            preserve_types=preserve_types,
            max_depth=max_depth,
            _depth=next_depth,
        )
    if isinstance(value, np.ndarray):
        data = value.tolist()
        if preserve_types:
            return {
                "__type__": "ndarray",
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": to_serializable(
                    data,
                    preserve_types=preserve_types,
                    max_depth=max_depth,
                    _depth=next_depth,
                ),
            }
        return to_serializable(
            data,
            preserve_types=preserve_types,
            max_depth=max_depth,
            _depth=next_depth,
        )
    if isinstance(value, Certificate):
        return to_serializable(
            value.to_dict(),
            preserve_types=preserve_types,
            max_depth=max_depth,
            _depth=next_depth,
        )
    if isinstance(value, ComputationResult):
        return {
            "value": to_serializable(
                value.value,
                preserve_types=preserve_types,
                max_depth=max_depth,
                _depth=next_depth,
            ),
            "certificate": to_serializable(
                value.certificate,
                preserve_types=preserve_types,
                max_depth=max_depth,
                _depth=next_depth,
            ),
        }
    if isinstance(value, ConditionalResult):
        return {
            "value": to_serializable(
                value.value,
                preserve_types=preserve_types,
                max_depth=max_depth,
                _depth=next_depth,
            ),
            "certificate": to_serializable(
                value.certificate,
                preserve_types=preserve_types,
                max_depth=max_depth,
                _depth=next_depth,
            ),
        }
    if isinstance(value, Exportable):
        return to_serializable(
            value.to_export_data(),
            preserve_types=preserve_types,
            max_depth=max_depth,
            _depth=next_depth,
        )
    if is_dataclass(value) and not isinstance(value, type):
        data = {
            field.name: to_serializable(
                getattr(value, field.name),
                preserve_types=preserve_types,
                max_depth=max_depth,
                _depth=next_depth,
            )
            for field in fields(value)
        }
        if preserve_types:
            return {"__type__": f"{type(value).__module__}.{type(value).__qualname__}", **data}
        return data
    if isinstance(value, Mapping):
        return {
            str(key): to_serializable(
                item,
                preserve_types=preserve_types,
                max_depth=max_depth,
                _depth=next_depth,
            )
            for key, item in value.items()
        }
    if isinstance(value, (set, frozenset)):
        ordered = sorted(value, key=repr)
        return [
            to_serializable(
                item,
                preserve_types=preserve_types,
                max_depth=max_depth,
                _depth=next_depth,
            )
            for item in ordered
        ]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            to_serializable(
                item,
                preserve_types=preserve_types,
                max_depth=max_depth,
                _depth=next_depth,
            )
            for item in value
        ]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return to_serializable(
            value.to_dict(),
            preserve_types=preserve_types,
            max_depth=max_depth,
            _depth=next_depth,
        )
    raise ValidationError(
        f"object of type {type(value).__module__}.{type(value).__qualname__} is not exportable"
    )


def from_serializable(
    value: Any,
    *,
    max_depth: int = 64,
    _depth: int = 0,
) -> Any:
    """Decode the safe built-in type markers emitted by :func:`to_serializable`.

    The decoder intentionally does not import or instantiate arbitrary classes
    named by ``__type__``.  Unknown markers remain ordinary mappings so schema
    specific decoders can validate them explicitly.
    """

    if _depth > max_depth:
        raise ValidationError("maximum export decoding depth exceeded")
    next_depth = _depth + 1
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, list):
        return [
            from_serializable(item, max_depth=max_depth, _depth=next_depth)
            for item in value
        ]
    if not isinstance(value, Mapping):
        raise ValidationError(
            f"serialized value of type {type(value).__module__}.{type(value).__qualname__} is invalid"
        )

    marker = value.get("__type__")
    if marker == "Fraction":
        try:
            return Fraction(int(value["numerator"]), int(value["denominator"]))
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValidationError("invalid serialized Fraction") from exc
    if marker == "complex":
        try:
            return complex(float(value["real"]), float(value["imag"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("invalid serialized complex value") from exc
    if marker == "ndarray":
        try:
            dtype = np.dtype(str(value["dtype"]))
            shape = tuple(int(length) for length in value["shape"])
            data = from_serializable(
                value["data"],
                max_depth=max_depth,
                _depth=next_depth,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("invalid serialized ndarray metadata") from exc
        if any(length < 0 for length in shape):
            raise ValidationError("serialized ndarray shape must be nonnegative")
        if dtype.kind not in {"b", "i", "u", "f", "c", "O", "S", "U"}:
            raise ValidationError(f"serialized ndarray dtype is not supported: {dtype}")
        try:
            array = np.asarray(data, dtype=dtype)
            if array.shape != shape:
                array = array.reshape(shape)
        except (TypeError, ValueError) as exc:
            raise ValidationError("serialized ndarray data does not match its shape or dtype") from exc
        return array

    return {
        str(key): from_serializable(
            item,
            max_depth=max_depth,
            _depth=next_depth,
        )
        for key, item in value.items()
    }


def table_rows(value: Any) -> tuple[tuple[str, ...], tuple[tuple[Any, ...], ...]]:
    """Convert a table shaped value to headers and rows."""

    if isinstance(value, np.ndarray):
        if value.ndim == 1:
            return ("index", "value"), tuple((index, item) for index, item in enumerate(value.tolist()))
        if value.ndim == 2:
            return tuple(f"column_{index}" for index in range(value.shape[1])), tuple(
                tuple(row) for row in value.tolist()
            )
        raise ValidationError("CSV and table exports support only one or two dimensional arrays")

    normalized = to_serializable(value, preserve_types=False)
    if isinstance(normalized, Mapping):
        if normalized and all(
            isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray))
            for item in normalized.values()
        ):
            lengths = {len(item) for item in normalized.values()}
            if len(lengths) != 1:
                raise ValidationError("mapping columns must have equal lengths")
            headers = tuple(normalized)
            rows = tuple(tuple(normalized[key][index] for key in headers) for index in range(lengths.pop()))
            return headers, rows
        return ("key", "value"), tuple((key, item) for key, item in normalized.items())

    if isinstance(normalized, list):
        if not normalized:
            return ("value",), ()
        if all(isinstance(item, Mapping) for item in normalized):
            headers: list[str] = []
            for item in normalized:
                for key in item:
                    if key not in headers:
                        headers.append(key)
            rows = tuple(tuple(item.get(key, "") for key in headers) for item in normalized)
            return tuple(headers), rows
        if all(isinstance(item, list) for item in normalized):
            width = max(len(item) for item in normalized)
            headers = tuple(f"column_{index}" for index in range(width))
            rows = tuple(tuple(item[index] if index < len(item) else "" for index in range(width)) for item in normalized)
            return headers, rows
        return ("index", "value"), tuple((index, item) for index, item in enumerate(normalized))

    return ("value",), ((normalized,),)


__all__ = [
    "Exportable",
    "from_serializable",
    "export_adapters",
    "register_export_adapter",
    "table_rows",
    "to_serializable",
]
