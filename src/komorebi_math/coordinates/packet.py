"""Exact packet, residue, and tensor-index coordinate permutations."""

from __future__ import annotations

from math import gcd, prod
from numbers import Integral
from typing import Callable, Sequence

from ..exceptions import DimensionError, ValidationError
from .basis import permutation_transform
from .transform import CoordinateTransform


def _positive(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError(f"{name} must be a positive integer")
    return value


def residue_class_order(length: int, modulus: int) -> tuple[int, ...]:
    """Return indices grouped by residue modulo ``modulus``."""

    count = _positive(length, name="length")
    radix = _positive(modulus, name="modulus")
    # Empty lanes beyond the finite prefix need no enumeration.
    return tuple(index for residue in range(min(radix, count))
                 for index in range(residue, count, radix))


def residue_class_transform(length: int, modulus: int) -> CoordinateTransform:
    """Return the exact polyphase permutation into residue-lane order."""

    order = residue_class_order(length, modulus)
    return permutation_transform(
        order,
        source=f"sample order of length {length}",
        target=f"residue-lane order modulo {modulus}",
    )


def digit_reversal_order(length: int, radix: int = 2) -> tuple[int, ...]:
    """Return the fixed-width digit-reversal order for an integer radix >= 2."""

    count = _positive(length, name="length")
    base = _positive(radix, name="radix")
    if base < 2:
        raise ValidationError("radix must be at least two")
    power = 1
    digits = 0
    while power < count:
        power *= base
        digits += 1
    if power != count:
        raise DimensionError("length must be an exact power of the radix")

    def reverse(index: int) -> int:
        value = index
        result = 0
        for _ in range(digits):
            result = result * base + value % base
            value //= base
        return result

    return tuple(reverse(index) for index in range(count))


def digit_reversal_transform(length: int, radix: int = 2) -> CoordinateTransform:
    """Return the exact fixed-width digit-reversal permutation."""

    return permutation_transform(
        digit_reversal_order(length, radix),
        source=f"natural order of length {length}",
        target=f"base-{radix} digit-reversed order",
    )


def _unravel(index: int, shape: Sequence[int]) -> tuple[int, ...]:
    coordinates = [0 for _ in shape]
    value = index
    for axis in range(len(shape) - 1, -1, -1):
        coordinates[axis] = value % shape[axis]
        value //= shape[axis]
    return tuple(coordinates)


def _ravel(coordinates: Sequence[int], shape: Sequence[int]) -> int:
    index = 0
    for coordinate, size in zip(coordinates, shape, strict=True):
        index = index * size + coordinate
    return index


def tensor_axis_permutation_order(
    shape: Sequence[int],
    axes: Sequence[int],
) -> tuple[int, ...]:
    """Return the flat-index order induced by a tensor-axis permutation."""

    dimensions = tuple(_positive(value, name="shape entry") for value in shape)
    if not dimensions:
        raise DimensionError("tensor shape must be nonempty")
    permutation = tuple(axes)
    if any(isinstance(axis, bool) or not isinstance(axis, Integral) for axis in permutation):
        raise ValidationError("axes must be integer tensor dimensions")
    permutation = tuple(int(axis) for axis in permutation)
    if sorted(permutation) != list(range(len(dimensions))):
        raise ValidationError("axes must be a permutation of tensor dimensions")
    target_shape = tuple(dimensions[axis] for axis in permutation)
    count = prod(dimensions)
    order: list[int] = []
    for target_index in range(count):
        target_coordinates = _unravel(target_index, target_shape)
        source_coordinates = [0 for _ in dimensions]
        for target_axis, source_axis in enumerate(permutation):
            source_coordinates[source_axis] = target_coordinates[target_axis]
        order.append(_ravel(source_coordinates, dimensions))
    return tuple(order)


def tensor_axis_permutation_transform(
    shape: Sequence[int],
    axes: Sequence[int],
) -> CoordinateTransform:
    """Return the exact coordinate permutation for transposed tensor axes."""

    dimensions = tuple(shape)
    permutation = tuple(axes)
    return permutation_transform(
        tensor_axis_permutation_order(dimensions, permutation),
        source=f"row-major tensor coordinates with shape {dimensions}",
        target=f"row-major tensor coordinates with axes {permutation}",
    )


def chinese_remainder_order(moduli: Sequence[int]) -> tuple[int, ...]:
    """Return flat indices ordered lexicographically by CRT residues."""

    values = tuple(_positive(value, name="modulus") for value in moduli)
    if not values:
        raise DimensionError("at least one modulus is required")
    for left_index, left in enumerate(values):
        for right in values[left_index + 1 :]:
            if gcd(left, right) != 1:
                raise ValidationError("CRT moduli must be pairwise coprime")
    count = prod(values)
    keyed = sorted(range(count), key=lambda index: tuple(index % modulus for modulus in values))
    return tuple(keyed)


def chinese_remainder_transform(moduli: Sequence[int]) -> CoordinateTransform:
    """Return the permutation from integer order to CRT-residue order."""

    values = tuple(moduli)
    count = prod(values)
    return permutation_transform(
        chinese_remainder_order(values),
        source=f"integer index order modulo {count}",
        target=f"CRT residue order for moduli {values}",
    )


def sorted_coordinate_transform(
    length: int,
    key: Callable[[int], object],
    *,
    source: str = "original index order",
    target: str = "key-sorted index order",
) -> CoordinateTransform:
    """Return a custom exact index permutation obtained from a deterministic key."""

    count = _positive(length, name="length")
    order = tuple(sorted(range(count), key=key))
    return permutation_transform(order, source=source, target=target)


__all__ = [
    "chinese_remainder_order",
    "chinese_remainder_transform",
    "digit_reversal_order",
    "digit_reversal_transform",
    "residue_class_order",
    "residue_class_transform",
    "sorted_coordinate_transform",
    "tensor_axis_permutation_order",
    "tensor_axis_permutation_transform",
]
