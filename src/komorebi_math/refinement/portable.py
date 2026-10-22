"""Trusted portable codecs for finite polyphase and refinement records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..arithmetic.polyphase import ArithmeticPolyphase
from ..exceptions import ValidationError
from ..export.persistence import register_portable_codec
from ..signals.polyphase import LatticePolyphaseCoordinates, PolyphaseCoordinates
from .polyphase import (
    LatticePolyphaseRefinement,
    PolyphaseRefinement,
    PolyphaseRefinementTower,
)
from .residue import ResidueCoordinates


def _require_fields(data: Mapping[str, Any], fields: set[str], *, kind: str) -> None:
    if set(data) != fields:
        raise ValidationError(f"{kind} portable data does not match its schema")


def _decode_signal_coordinates(data: Mapping[str, Any]) -> PolyphaseCoordinates:
    fields = {"channels", "order", "axis", "original_shape"}
    _require_fields(data, fields, kind="signal polyphase coordinates")
    return PolyphaseCoordinates(
        tuple(data["channels"]),
        int(data["order"]),
        int(data["axis"]),
        tuple(int(value) for value in data["original_shape"]),
    )


def _decode_lattice_coordinates(data: Mapping[str, Any]) -> LatticePolyphaseCoordinates:
    fields = {"channels", "strides", "original_shape"}
    _require_fields(data, fields, kind="lattice polyphase coordinates")
    raw_channels = data["channels"]
    if not isinstance(raw_channels, Mapping):
        raise ValidationError("lattice polyphase channels must be a mapping")
    channels = {
        tuple(int(value) for value in address): channel
        for address, channel in raw_channels.items()
    }
    return LatticePolyphaseCoordinates(
        channels,
        tuple(int(value) for value in data["strides"]),
        tuple(int(value) for value in data["original_shape"]),
    )


def _decode_residue_coordinates(data: Mapping[str, Any]) -> ResidueCoordinates:
    fields = {"modulus", "channels", "length"}
    _require_fields(data, fields, kind="residue coordinates")
    return ResidueCoordinates(
        int(data["modulus"]),
        tuple(data["channels"]),
        int(data["length"]),
    )


def _decode_arithmetic_coordinates(data: Mapping[str, Any]) -> ArithmeticPolyphase:
    fields = {"modulus", "channels", "length"}
    _require_fields(data, fields, kind="arithmetic polyphase coordinates")
    return ArithmeticPolyphase(
        int(data["modulus"]),
        tuple(tuple(channel) for channel in data["channels"]),
        int(data["length"]),
    )


def _decode_refinement(data: Mapping[str, Any]) -> PolyphaseRefinement:
    fields = {"coarse_order", "fine_order"}
    _require_fields(data, fields, kind="polyphase refinement")
    return PolyphaseRefinement(int(data["coarse_order"]), int(data["fine_order"]))


def _decode_refinement_tower(data: Mapping[str, Any]) -> PolyphaseRefinementTower:
    _require_fields(data, {"orders"}, kind="polyphase refinement tower")
    return PolyphaseRefinementTower(tuple(int(value) for value in data["orders"]))


def _decode_lattice_refinement(data: Mapping[str, Any]) -> LatticePolyphaseRefinement:
    fields = {"coarse_strides", "fine_strides"}
    _require_fields(data, fields, kind="lattice polyphase refinement")
    return LatticePolyphaseRefinement(
        tuple(int(value) for value in data["coarse_strides"]),
        tuple(int(value) for value in data["fine_strides"]),
    )


def register_refinement_codecs() -> None:
    """Register all finite polyphase codecs idempotently."""

    register_portable_codec(
        PolyphaseCoordinates,
        tag="signals.polyphase_coordinates",
        version=1,
        encoder=lambda value: {
            "channels": value.channels,
            "order": value.order,
            "axis": value.axis,
            "original_shape": value.original_shape,
        },
        decoder=_decode_signal_coordinates,
        replace=True,
    )
    register_portable_codec(
        LatticePolyphaseCoordinates,
        tag="signals.lattice_polyphase_coordinates",
        version=1,
        encoder=lambda value: {
            "channels": value.channels,
            "strides": value.strides,
            "original_shape": value.original_shape,
        },
        decoder=_decode_lattice_coordinates,
        replace=True,
    )
    register_portable_codec(
        ResidueCoordinates,
        tag="refinement.residue_coordinates",
        version=1,
        encoder=lambda value: {
            "modulus": value.modulus,
            "channels": value.channels,
            "length": value.length,
        },
        decoder=_decode_residue_coordinates,
        replace=True,
    )
    register_portable_codec(
        ArithmeticPolyphase,
        tag="arithmetic.polyphase_coordinates",
        version=1,
        encoder=lambda value: {
            "modulus": value.modulus,
            "channels": value.channels,
            "length": value.length,
        },
        decoder=_decode_arithmetic_coordinates,
        replace=True,
    )
    register_portable_codec(
        PolyphaseRefinement,
        tag="refinement.polyphase",
        version=1,
        encoder=lambda value: {
            "coarse_order": value.coarse_order,
            "fine_order": value.fine_order,
        },
        decoder=_decode_refinement,
        replace=True,
    )
    register_portable_codec(
        PolyphaseRefinementTower,
        tag="refinement.polyphase_tower",
        version=1,
        encoder=lambda value: {"orders": value.orders},
        decoder=_decode_refinement_tower,
        replace=True,
    )
    register_portable_codec(
        LatticePolyphaseRefinement,
        tag="refinement.lattice_polyphase",
        version=1,
        encoder=lambda value: {
            "coarse_strides": value.coarse_strides,
            "fine_strides": value.fine_strides,
        },
        decoder=_decode_lattice_refinement,
        replace=True,
    )


__all__ = ["register_refinement_codecs"]
