from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from komorebi_math import SeedSystem, semisimple_seed_rank, single_operator_seed_rank
from komorebi_math.exceptions import ValidationError


def cyclic_shift(size: int) -> np.ndarray:
    matrix = np.zeros((size, size))
    for index in range(size):
        matrix[(index + 1) % size, index] = 1
    return matrix


def test_cyclic_shift_seed_generates() -> None:
    seed = np.eye(5)[:, 0]
    system = SeedSystem({"S": cyclic_shift(5)}, [seed])
    assert system.hull_dimension() == 5
    assert system.is_generating()
    assert [len(layer) for layer in system.orbit_layers()] == [1, 1, 1, 1, 1]


def test_proper_seed_hull() -> None:
    diagonal = np.diag([1, 2, 3])
    seed = np.array([1, 0, 0])
    system = SeedSystem([diagonal], [seed])
    assert system.hull_dimension() == 1
    assert not system.generation_certificate().valid


def test_free_presentation() -> None:
    operator = np.array([[0, 1], [1, 0]], dtype=float)
    seed = np.array([1, 0], dtype=float)
    system = SeedSystem([operator], [seed])
    presentation = system.presentation([np.eye(2), operator])
    assert presentation.rank == 2
    assert presentation.nullity == 0


def test_standard_basis_search() -> None:
    diagonal = np.diag([1, 2, 3])
    system = SeedSystem([diagonal], [np.ones(3)])
    result = system.minimum_standard_basis_seeds()
    assert result.count == 3
    assert result.indices == (0, 1, 2)


def test_standard_basis_search_guard() -> None:
    system = SeedSystem([], [np.ones(13)])
    with pytest.raises(ValidationError):
        system.minimum_standard_basis_seeds()


def test_semisimple_seed_rank_formula() -> None:
    assert semisimple_seed_rank([3], [7]) == 3
    assert semisimple_seed_rank([2, 4], [5, 9]) == 3
    assert semisimple_seed_rank([1, 1, 1], [2, 5, 3]) == 5


def test_single_operator_seed_rank_over_q() -> None:
    jordan2 = sp.Matrix([[4, 1], [0, 4]])
    jordan1 = sp.Matrix([[4]])
    matrix = sp.diag(jordan2, jordan1)
    assert single_operator_seed_rank(matrix) == 2


def test_irreducible_quadratic_primary_rank() -> None:
    rotation = sp.Matrix([[0, -1], [1, 0]])
    matrix = sp.diag(rotation, rotation)
    assert single_operator_seed_rank(matrix) == 2


def test_single_operator_requires_rational_entries() -> None:
    with pytest.raises(ValidationError):
        single_operator_seed_rank(sp.Matrix([[sp.sqrt(2)]]))
