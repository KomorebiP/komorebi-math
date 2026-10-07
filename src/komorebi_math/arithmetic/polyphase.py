"""Exact discrete and Mahler polyphase refinement for arithmetic sequences."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import Iterable

from .._number_theory import require_integer
from ..exceptions import ValidationError
from .mahler import MahlerSeries


@dataclass(frozen=True, slots=True)
class ArithmeticPolyphase:
    """Exact residue channels of one finite arithmetic sequence."""

    modulus: int
    channels: tuple[tuple[Number, ...], ...]
    length: int

    def reconstruct(self) -> tuple[Number, ...]:
        """Interleave the channels into the original finite sequence."""

        result: list[Number] = [0 for _ in range(self.length)]
        for residue, channel in enumerate(self.channels):
            positions = tuple(range(residue, self.length, self.modulus))
            if len(channel) != len(positions):
                raise ValidationError("polyphase channel length does not match its residue positions")
            for position, value in zip(positions, channel, strict=True):
                result[position] = value
        return tuple(result)

    def nested(self, inner_modulus: int) -> tuple["ArithmeticPolyphase", ...]:
        """Refine every channel by a second residue modulus."""

        return tuple(polyphase_analyze(channel, inner_modulus) for channel in self.channels)

    def refine(self, fine_modulus: int) -> "ArithmeticPolyphase":
        """Refine these exact channels to a divisible finer modulus."""

        from ..refinement.polyphase import refine_polyphase

        refined = refine_polyphase(self, fine_modulus)
        if not isinstance(refined, ArithmeticPolyphase):
            raise ValidationError("internal arithmetic refinement returned the wrong coordinate type")
        return refined

    def coarsen(self, coarse_modulus: int) -> "ArithmeticPolyphase":
        """Coarsen these exact channels to a divisor of the current modulus."""

        from ..refinement.polyphase import coarsen_polyphase

        coarsened = coarsen_polyphase(self, coarse_modulus)
        if not isinstance(coarsened, ArithmeticPolyphase):
            raise ValidationError("internal arithmetic coarsening returned the wrong coordinate type")
        return coarsened


def polyphase_analyze(values: Iterable[Number], modulus: int) -> ArithmeticPolyphase:
    """Split a finite sequence into exact residue channels."""

    data = tuple(values)
    m = require_integer(modulus, name="modulus")
    if m < 1:
        raise ValidationError("modulus must be positive")
    channels = tuple(tuple(data[residue::m]) for residue in range(m))
    return ArithmeticPolyphase(m, channels, len(data))


def polyphase_synthesize(channels: Iterable[Iterable[Number]], length: int | None = None) -> tuple[Number, ...]:
    """Interleave exact residue channels."""

    packets = tuple(tuple(channel) for channel in channels)
    if not packets:
        raise ValidationError("at least one channel is required")
    inferred = sum(len(channel) for channel in packets)
    target = inferred if length is None else require_integer(length, name="length")
    if target < 0 or target != inferred:
        raise ValidationError("length must equal the total number of channel entries")
    return ArithmeticPolyphase(len(packets), packets, target).reconstruct()


def mixed_radix_address(digits: Iterable[int], radices: Iterable[int]) -> int:
    """Encode nested residue digits in least significant first order."""

    values = tuple(require_integer(value, name="digit") for value in digits)
    bases = tuple(require_integer(value, name="radix") for value in radices)
    if len(values) != len(bases):
        raise ValidationError("digits and radices must have the same length")
    address = 0
    place = 1
    for digit, radix in zip(values, bases, strict=True):
        if radix < 1 or digit < 0 or digit >= radix:
            raise ValidationError("every digit must lie in its radix range")
        address += place * digit
        place *= radix
    return address


def mixed_radix_digits(address: int, radices: Iterable[int]) -> tuple[int, ...]:
    """Decode one mixed radix address in least significant first order."""

    value = require_integer(address, name="address")
    bases = tuple(require_integer(radix, name="radix") for radix in radices)
    if value < 0 or any(radix < 1 for radix in bases):
        raise ValidationError("address must be nonnegative and radices positive")
    digits: list[int] = []
    remaining = value
    for radix in bases:
        digits.append(remaining % radix)
        remaining //= radix
    if remaining:
        raise ValidationError("address exceeds the declared mixed radix range")
    return tuple(digits)


def mahler_polyphase(series: MahlerSeries, modulus: int, address: int) -> MahlerSeries:
    """Return the exact Mahler coordinates of ``series(modulus*x + address)``."""

    m = require_integer(modulus, name="modulus")
    residue = require_integer(address, name="address")
    if m < 1 or residue < 0 or residue >= m:
        raise ValidationError("modulus must be positive and address must lie in its residue range")
    return series.affine_subsequence(m, residue)


__all__ = [
    "ArithmeticPolyphase",
    "mahler_polyphase",
    "mixed_radix_address",
    "mixed_radix_digits",
    "polyphase_analyze",
    "polyphase_synthesize",
]
