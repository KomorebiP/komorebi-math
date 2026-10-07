"""Internal validation helpers for structured algorithm APIs."""

from __future__ import annotations

from math import isfinite
from operator import index as integer_index
from typing import Any

import numpy as np

from ..exceptions import ValidationError


def strict_integer(
    value: Any,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return an integer index without silently truncating real values."""

    if isinstance(value, (bool, np.bool_)):
        raise ValidationError(f"{name} must be an integer")
    try:
        result = integer_index(value)
    except TypeError as exc:
        raise ValidationError(f"{name} must be an integer") from exc
    if minimum is not None and result < minimum:
        raise ValidationError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValidationError(f"{name} must be at most {maximum}")
    return int(result)


def finite_float(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    """Return a finite float satisfying an optional lower bound."""

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{name} must be a finite real number") from exc
    if not isfinite(result):
        raise ValidationError(f"{name} must be a finite real number")
    if minimum is not None:
        invalid = result <= minimum if strict_minimum else result < minimum
        if invalid:
            relation = "greater than" if strict_minimum else "at least"
            raise ValidationError(f"{name} must be {relation} {minimum}")
    return result


__all__: list[str] = []
