from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from komorebi_math import stable
from komorebi_math.arithmetic.polyphase import polyphase_analyze as arithmetic_analyze
from komorebi_math.compatibility import check_stable_api
from komorebi_math.exceptions import ValidationError
from komorebi_math.export.persistence import dumps_portable, loads_portable
from komorebi_math.refinement import (
    LatticePolyphaseRefinement,
    PolyphaseRefinement,
    PolyphaseRefinementTower,
    coarsen_lattice_polyphase,
    coarsen_polyphase,
    polyphase_refinement_certificate,
    refine_lattice_polyphase,
    refine_polyphase,
    residue_analyze,
)
from komorebi_math.signals import lattice_polyphase_analyze, polyphase_analyze


@pytest.mark.parametrize(
    ("coarse", "fine", "residue", "digit", "address"),
    [
        (1, 1, 0, 0, 0),
        (2, 4, 0, 1, 2),
        (2, 8, 1, 3, 7),
        (4, 12, 1, 2, 9),
        (5, 20, 4, 3, 19),
    ],
)
def test_address_law(coarse: int, fine: int, residue: int, digit: int, address: int) -> None:
    refinement = PolyphaseRefinement(coarse, fine)
    assert refinement.address(residue, digit) == address
    assert refinement.split_address(address) == (residue, digit)
    assert refinement.parent(address) == residue
    assert refinement.child_digit(address) == digit


@pytest.mark.parametrize(
    ("coarse", "fine", "residue", "children"),
    [
        (1, 4, 0, (0, 1, 2, 3)),
        (2, 6, 0, (0, 2, 4)),
        (2, 6, 1, (1, 3, 5)),
        (4, 12, 1, (1, 5, 9)),
    ],
)
def test_children_and_parent_map(
    coarse: int,
    fine: int,
    residue: int,
    children: tuple[int, ...],
) -> None:
    refinement = PolyphaseRefinement(coarse, fine)
    assert refinement.children(residue) == children
    assert tuple(refinement.parent(child) for child in children) == (residue,) * len(children)
    assert len(refinement.parent_map) == fine


@pytest.mark.parametrize(
    ("coarse", "fine"),
    [(0, 2), (-1, 2), (2, 0), (3, 8)],
)
def test_invalid_refinement_orders(coarse: int, fine: int) -> None:
    with pytest.raises(ValidationError):
        PolyphaseRefinement(coarse, fine)


@pytest.mark.parametrize(
    ("shape", "axis", "coarse", "fine"),
    [
        ((17,), 0, 2, 6),
        ((18,), -1, 3, 9),
        ((3, 17), 1, 2, 8),
        ((17, 3), 0, 4, 12),
        ((2, 3, 19), 2, 3, 12),
        ((2, 19, 3), 1, 5, 10),
    ],
)
def test_signal_refinement_matches_direct_analysis(
    shape: tuple[int, ...],
    axis: int,
    coarse: int,
    fine: int,
) -> None:
    signal = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    coarse_coordinates = polyphase_analyze(signal, coarse, axis=axis)
    refined = refine_polyphase(coarse_coordinates, fine)
    direct = polyphase_analyze(signal, fine, axis=axis)
    assert refined.order == fine
    assert refined.axis == direct.axis
    assert all(
        np.array_equal(left, right)
        for left, right in zip(refined.channels, direct.channels, strict=True)
    )
    restored = coarsen_polyphase(refined, coarse)
    assert np.array_equal(restored.reconstruct(), signal)
    assert polyphase_refinement_certificate(coarse_coordinates, fine).valid


@pytest.mark.parametrize(
    ("length", "coarse", "fine"),
    [(1, 1, 4), (17, 2, 6), (31, 4, 12), (64, 5, 20)],
)
def test_residue_refinement_roundtrip(length: int, coarse: int, fine: int) -> None:
    values = np.arange(length, dtype=np.int64)
    coordinates = residue_analyze(values, coarse)
    refined = coordinates.refine(fine)
    direct = residue_analyze(values, fine)
    assert all(
        np.array_equal(left, right)
        for left, right in zip(refined.channels, direct.channels, strict=True)
    )
    assert np.array_equal(refined.coarsen(coarse).reconstruct(), values)


@pytest.mark.parametrize(
    ("length", "coarse", "fine"),
    [(7, 1, 7), (13, 2, 6), (29, 4, 12)],
)
def test_exact_arithmetic_refinement(length: int, coarse: int, fine: int) -> None:
    values = tuple(Fraction(index + 1, index + 2) for index in range(length))
    coordinates = arithmetic_analyze(values, coarse)
    refined = coordinates.refine(fine)
    direct = arithmetic_analyze(values, fine)
    assert refined == direct
    assert refined.coarsen(coarse) == coordinates
    assert PolyphaseRefinement(coarse, fine).certificate(coordinates).valid


@pytest.mark.parametrize("orders", [(1, 2, 8), (2, 6, 24), (3, 12, 36)])
def test_refinement_tower_coherence(orders: tuple[int, ...]) -> None:
    values = np.arange(71, dtype=np.int64)
    tower = PolyphaseRefinementTower(orders)
    coordinates = residue_analyze(values, orders[0])
    sequential = tower.refine(coordinates)
    direct = PolyphaseRefinement(orders[0], orders[-1]).refine(coordinates)
    assert all(
        np.array_equal(left, right)
        for left, right in zip(sequential.channels, direct.channels, strict=True)
    )
    restored = tower.coarsen(sequential)
    assert np.array_equal(restored.reconstruct(), coordinates.reconstruct())
    assert tower.coherence_certificate(coordinates).valid


@pytest.mark.parametrize(
    ("shape", "coarse", "fine"),
    [
        ((5, 7), (1, 1), (2, 3)),
        ((7, 11), (1, 2), (3, 6)),
        ((6, 9), (2, 3), (4, 9)),
        ((3, 5, 7), (1, 1, 1), (2, 3, 2)),
    ],
)
def test_lattice_refinement_matches_direct_analysis(
    shape: tuple[int, ...],
    coarse: tuple[int, ...],
    fine: tuple[int, ...],
) -> None:
    signal = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    coordinates = lattice_polyphase_analyze(signal, coarse)
    refined = refine_lattice_polyphase(coordinates, fine)
    direct = lattice_polyphase_analyze(signal, fine)
    assert refined.strides == fine
    assert set(refined.channels) == set(direct.channels)
    assert all(np.array_equal(refined.channels[key], direct.channels[key]) for key in direct.channels)
    restored = coarsen_lattice_polyphase(refined, coarse)
    assert np.array_equal(restored.reconstruct(), signal)
    assert LatticePolyphaseRefinement(coarse, fine).certificate(coordinates).valid


@pytest.mark.parametrize(
    "record",
    [
        PolyphaseRefinement(4, 12),
        PolyphaseRefinementTower((2, 4, 12)),
        LatticePolyphaseRefinement((1, 2), (2, 6)),
    ],
)
def test_refinement_records_have_portable_roundtrips(record: object) -> None:
    restored = loads_portable(dumps_portable(record))
    assert restored == record


@pytest.mark.parametrize("kind", ["signal", "residue", "arithmetic", "lattice"])
def test_coordinate_records_have_portable_roundtrips(kind: str) -> None:
    if kind == "signal":
        value = polyphase_analyze(np.arange(30).reshape(5, 6), 4, axis=1)
    elif kind == "residue":
        value = residue_analyze(np.arange(23), 5)
    elif kind == "arithmetic":
        value = arithmetic_analyze((Fraction(i, i + 1) for i in range(1, 12)), 4)
    else:
        value = lattice_polyphase_analyze(np.arange(35).reshape(5, 7), (2, 3))
    restored = loads_portable(dumps_portable(value))
    assert type(restored) is type(value)
    if kind == "arithmetic":
        assert restored == value
    else:
        assert np.array_equal(restored.reconstruct(), value.reconstruct())


def test_curated_aliases_are_single_implementations() -> None:
    assert stable.polyphase_refine is stable.refine_polyphase
    assert stable.polyphase_coarsen is stable.coarsen_polyphase
    assert stable.lattice_polyphase_refine is stable.refine_lattice_polyphase
    assert stable.lattice_polyphase_coarsen is stable.coarsen_lattice_polyphase


def test_versioned_namespaces_match_curated_surface() -> None:
    import komorebi_math.api.v1 as api_v1
    import komorebi_math.v1 as v1

    assert api_v1.__all__ == stable.__all__
    assert v1.__all__ == stable.__all__
    for name in stable.__all__:
        assert getattr(api_v1, name) is getattr(stable, name)
        assert getattr(v1, name) is getattr(stable, name)


def test_committed_stable_api_manifest_matches_runtime() -> None:
    report = check_stable_api()
    assert report.compatible
    assert report.added == ()
    assert report.removed == ()
    assert report.kind_changed == ()
    assert report.signature_changed == ()


def test_convenience_methods_delegate_to_canonical_refinement() -> None:
    signal = np.arange(41)
    signal_coordinates = polyphase_analyze(signal, 4)
    assert np.array_equal(signal_coordinates.refine(12).coarsen(4).reconstruct(), signal)

    residue_coordinates = residue_analyze(signal, 4)
    assert np.array_equal(residue_coordinates.refine(12).coarsen(4).reconstruct(), signal)

    arithmetic = arithmetic_analyze(tuple(range(41)), 4)
    assert arithmetic.refine(12).coarsen(4).reconstruct() == tuple(range(41))
