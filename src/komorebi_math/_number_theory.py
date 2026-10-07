"""Small exact number theoretic validation helpers used across the package."""

from __future__ import annotations

from math import isqrt
from numbers import Integral

from .exceptions import ValidationError


def require_integer(value: int, *, name: str = "value") -> int:
    """Return an exact integer input and reject booleans and numeric coercions."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValidationError(f"{name} must be an integer")
    return int(value)


def is_prime(value: int) -> bool:
    """Return whether ``value`` is a positive rational prime."""

    try:
        integer = require_integer(value)
    except ValidationError:
        return False
    if integer < 2:
        return False
    if integer in (2, 3):
        return True
    if integer % 2 == 0 or integer % 3 == 0:
        return False
    limit = isqrt(integer)
    divisor = 5
    step = 2
    while divisor <= limit:
        if integer % divisor == 0:
            return False
        divisor += step
        step = 6 - step
    return True


def require_prime(value: int, *, name: str = "prime") -> int:
    """Return ``value`` as an integer or raise when it is not prime."""

    integer = require_integer(value, name=name)
    if not is_prime(integer):
        raise ValidationError(f"{name} must be a positive prime integer")
    return integer


__all__ = ["is_prime", "require_integer", "require_prime"]
