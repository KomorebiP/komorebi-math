"""Residue and polyphase packet coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..backends import LinearAlgebraBackend, get_backend
from ..core.packets import PacketCoordinates, PacketSystem
from ..exceptions import DimensionError
from ..provenance import ClaimStatus, ProvenanceRecord, documented


@documented(
    ProvenanceRecord(
        identifier="PSC-II-D05",
        title="Residue packet system",
        status=ClaimStatus.DEFINITION,
        source="Paper II",
        assumptions=("finite coordinate length", "positive modulus"),
        verification_suite="refinement.residue",
    )
)
def residue_packet_system(
    length: int,
    modulus: int,
    *,
    backend: str | LinearAlgebraBackend | None = None,
    names: tuple[str, ...] | None = None,
) -> PacketSystem:
    """Return diagonal projectors selecting coordinate residues modulo ``modulus``."""

    if length < 0:
        raise ValueError("length must be nonnegative")
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    algebra = get_backend(backend)
    projectors: list[Any] = []
    for residue in range(modulus):
        diagonal = [1 if index % modulus == residue else 0 for index in range(length)]
        if algebra.name == "sympy":
            import sympy as sp

            projectors.append(sp.diag(*diagonal) if diagonal else sp.zeros(0, 0))
        else:
            projectors.append(np.diag(np.asarray(diagonal, dtype=float)))
    packet_names = names if names is not None else tuple(f"{index} mod {modulus}" for index in range(modulus))
    if len(packet_names) != modulus:
        raise DimensionError("residue packet names must match the modulus")
    return PacketSystem(projectors, names=packet_names, backend=algebra)


@dataclass(frozen=True, slots=True)
class ResidueCoordinates:
    """Ordinary polyphase channels for one finite sequence."""

    modulus: int
    channels: tuple[np.ndarray, ...]
    length: int

    def __post_init__(self) -> None:
        if self.modulus <= 0:
            raise ValueError("modulus must be positive")
        if len(self.channels) != self.modulus:
            raise DimensionError("channel count must equal the modulus")
        if self.length < 0:
            raise ValueError("length must be nonnegative")

    def reconstruct(self) -> np.ndarray:
        output_dtype = np.result_type(*(channel.dtype for channel in self.channels)) if self.channels else float
        result = np.zeros(self.length, dtype=output_dtype)
        for residue, channel in enumerate(self.channels):
            expected = len(range(residue, self.length, self.modulus))
            if channel.size != expected:
                raise DimensionError("channel length does not match its residue positions")
            result[residue : self.length : self.modulus] = channel
        return result

    def nested(self, inner_modulus: int) -> tuple["ResidueCoordinates", ...]:
        return tuple(residue_analyze(channel, inner_modulus) for channel in self.channels)

    def refine(self, fine_modulus: int, *, copy: bool = True) -> "ResidueCoordinates":
        """Refine these residue channels to a divisible finer modulus."""

        from .polyphase import refine_polyphase

        refined = refine_polyphase(self, fine_modulus, copy=copy)
        if not isinstance(refined, ResidueCoordinates):
            raise DimensionError("internal residue refinement returned the wrong coordinate type")
        return refined

    def coarsen(self, coarse_modulus: int, *, copy: bool = True) -> "ResidueCoordinates":
        """Coarsen these residue channels to a divisor of the current modulus."""

        from .polyphase import coarsen_polyphase

        coarsened = coarsen_polyphase(self, coarse_modulus, copy=copy)
        if not isinstance(coarsened, ResidueCoordinates):
            raise DimensionError("internal residue coarsening returned the wrong coordinate type")
        return coarsened


@documented(
    ProvenanceRecord(
        identifier="PSC-II-T06",
        title="Exact residue analysis and synthesis",
        status=ClaimStatus.PROVED,
        source="Paper II",
        assumptions=("finite sequence", "positive modulus"),
        verification_suite="refinement.residue",
    )
)
def residue_analyze(values: Any, modulus: int) -> ResidueCoordinates:
    """Split a one dimensional sequence into residue channels."""

    array = np.asarray(values)
    if array.ndim != 1:
        raise DimensionError("residue analysis requires a one dimensional sequence")
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    channels = tuple(np.array(array[residue::modulus], copy=True) for residue in range(modulus))
    return ResidueCoordinates(modulus, channels, int(array.size))


def residue_synthesize(channels: ResidueCoordinates | tuple[Any, ...], *, length: int | None = None) -> np.ndarray:
    """Reconstruct a sequence from residue channels."""

    if isinstance(channels, ResidueCoordinates):
        return channels.reconstruct()
    channel_arrays = tuple(np.asarray(channel) for channel in channels)
    modulus = len(channel_arrays)
    if modulus == 0:
        return np.asarray([], dtype=float)
    inferred = sum(channel.size for channel in channel_arrays)
    target_length = inferred if length is None else int(length)
    return ResidueCoordinates(modulus, channel_arrays, target_length).reconstruct()


@documented(
    ProvenanceRecord(
        identifier="PSC-II-T07",
        title="Nested residue refinement",
        status=ClaimStatus.PROVED,
        source="Paper II",
        assumptions=("positive outer and inner moduli",),
        verification_suite="refinement.residue",
    )
)
def nested_residue_address(outer_residue: int, outer_modulus: int, inner_residue: int, inner_modulus: int) -> int:
    """Return the combined address ``outer_residue + outer_modulus * inner_residue``."""

    if outer_modulus <= 0 or inner_modulus <= 0:
        raise ValueError("moduli must be positive")
    if not 0 <= outer_residue < outer_modulus:
        raise ValueError("outer residue is outside its standard range")
    if not 0 <= inner_residue < inner_modulus:
        raise ValueError("inner residue is outside its standard range")
    return outer_residue + outer_modulus * inner_residue


def nested_residue_analysis(values: Any, outer_modulus: int, inner_modulus: int) -> ResidueCoordinates:
    """Analyze twice and return the equivalent channels modulo the product."""

    outer = residue_analyze(values, outer_modulus)
    combined_channels: list[np.ndarray] = [np.asarray([])] * (outer_modulus * inner_modulus)
    for outer_residue, channel in enumerate(outer.channels):
        inner = residue_analyze(channel, inner_modulus)
        for inner_residue, subchannel in enumerate(inner.channels):
            address = nested_residue_address(outer_residue, outer_modulus, inner_residue, inner_modulus)
            combined_channels[address] = subchannel
    return ResidueCoordinates(outer_modulus * inner_modulus, tuple(combined_channels), outer.length)


def packet_coordinates_from_residues(values: Any, modulus: int, *, backend: str | LinearAlgebraBackend | None = None) -> PacketCoordinates:
    """Return embedded packet coordinates for a finite vector."""

    array = np.asarray(values)
    packets = residue_packet_system(int(array.size), modulus, backend=backend)
    return packets.analyze(array)


__all__ = [
    "ResidueCoordinates",
    "nested_residue_address",
    "nested_residue_analysis",
    "packet_coordinates_from_residues",
    "residue_analyze",
    "residue_packet_system",
    "residue_synthesize",
]
