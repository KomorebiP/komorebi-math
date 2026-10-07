from __future__ import annotations

import numpy as np
import pytest

from komorebi_math import PacketSystem
from komorebi_math.exceptions import ValidationError
from komorebi_math.refinement import PacketRefinement, RefinementTower, common_refinement


def singleton_packets(size: int) -> PacketSystem:
    return PacketSystem([np.diag([1 if i == j else 0 for i in range(size)]) for j in range(size)])


def two_level_refinement() -> tuple[PacketSystem, PacketSystem, PacketRefinement]:
    coarse = PacketSystem([np.diag([1, 1, 0, 0]), np.diag([0, 0, 1, 1])], names=["left", "right"])
    fine = singleton_packets(4)
    refinement = PacketRefinement(coarse, fine, (0, 0, 1, 1))
    return coarse, fine, refinement


def test_refinement_certificate_and_children() -> None:
    _, _, refinement = two_level_refinement()
    refinement.certificate().require()
    assert refinement.children("left") == (0, 1)
    assert refinement.children("right") == (2, 3)
    assert refinement.parent(3) == 1


def test_exact_coordinate_refine_and_coarsen() -> None:
    coarse, fine, refinement = two_level_refinement()
    vector = np.array([1, 2, 3, 4], dtype=float)
    fine_coordinates = refinement.refine(coarse.analyze(vector))
    assert fine_coordinates.system.equivalent(fine)
    assert np.allclose(fine_coordinates.reconstruct(), vector)
    coarse_coordinates = refinement.coarsen(fine_coordinates)
    assert coarse_coordinates.system.equivalent(coarse)
    assert np.allclose(coarse_coordinates.reconstruct(), vector)


def test_operator_blocks_refine_and_coarsen() -> None:
    coarse, fine, refinement = two_level_refinement()
    matrix = np.arange(16, dtype=float).reshape(4, 4)
    assert np.allclose(refinement.refine_operator(matrix).reconstruct(), matrix)
    assert np.allclose(refinement.coarsen_operator(matrix).reconstruct(), matrix)
    assert len(refinement.refine_operator(matrix).blocks) == len(fine)
    assert len(refinement.coarsen_operator(matrix).blocks) == len(coarse)


def test_support_reflection() -> None:
    coarse, _, refinement = two_level_refinement()
    vector = np.array([0, 2, 0, 4], dtype=float)
    support = refinement.support_reflection(coarse.analyze(vector))
    assert support == {"left": ("P1",), "right": ("P3",)}


def test_three_level_tower_coherence() -> None:
    coarse = PacketSystem([np.eye(4)], names=["all"])
    middle = PacketSystem([np.diag([1, 1, 0, 0]), np.diag([0, 0, 1, 1])])
    fine = singleton_packets(4)
    first = PacketRefinement(coarse, middle, (0, 0))
    second = PacketRefinement(middle, fine, (0, 0, 1, 1))
    tower = RefinementTower([first, second])
    tower.certificate().require()
    vector = np.array([5, 6, 7, 8], dtype=float)
    fine_coordinates = tower.refine(coarse.analyze(vector), source_level=0, target_level=2)
    assert np.allclose(fine_coordinates.reconstruct(), vector)
    back = tower.coarsen(fine_coordinates, source_level=2, target_level=0)
    assert np.allclose(back.reconstruct(), vector)
    assert tower.descendant_address(0, 3, coarse_level=0, fine_level=2)


def test_common_refinement_of_commuting_packets() -> None:
    left = PacketSystem([np.diag([1, 1, 0, 0]), np.diag([0, 0, 1, 1])])
    right = PacketSystem([np.diag([1, 0, 1, 0]), np.diag([0, 1, 0, 1])])
    common, left_map, right_map = common_refinement(left, right)
    assert len(common) == 4
    left_map.certificate().require()
    right_map.certificate().require()
    vector = np.arange(4, dtype=float)
    assert np.allclose(left_map.coarsen(common.analyze(vector)).reconstruct(), vector)


def test_noncommuting_packets_rejected() -> None:
    p = PacketSystem([np.diag([1, 0]), np.diag([0, 1])])
    q0 = 0.5 * np.array([[1, 1], [1, 1]], dtype=float)
    q = PacketSystem([q0, np.eye(2) - q0])
    with pytest.raises(ValidationError, match="commuting"):
        common_refinement(p, q)
