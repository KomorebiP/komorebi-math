from __future__ import annotations

import numpy as np

from komorebi_math import OperatorMorphism, PacketMorphism, PacketSystem, SeedMorphism, SeedSystem
from komorebi_math.backends import get_backend


def test_operator_morphism_intertwining() -> None:
    backend = get_backend("numpy")
    source = {"T": np.diag([1, 2])}
    target = {"S": np.diag([1, 2, 3])}
    phi = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)
    morphism = OperatorMorphism.create(
        source,
        target,
        phi,
        {"T": "S"},
        backend=backend,
    )
    assert morphism.certificate().valid


def test_operator_morphism_detects_failure() -> None:
    backend = get_backend("numpy")
    morphism = OperatorMorphism.create(
        {"T": np.diag([1, 2])},
        {"S": np.diag([1, 3])},
        np.eye(2),
        {"T": "S"},
        backend=backend,
    )
    assert not morphism.certificate().valid


def test_packet_morphism_aggregation() -> None:
    source = PacketSystem(
        [np.diag([1, 0, 0]), np.diag([0, 1, 0]), np.diag([0, 0, 1])],
        names=["a", "b", "c"],
    )
    target = source.aggregate({"ab": ["a", "b"], "c": ["c"]})
    morphism = PacketMorphism(
        source,
        target,
        np.eye(3),
        {"a": ("ab",), "b": ("ab",), "c": ("c",)},
    )
    assert morphism.certificate().valid


def test_seed_morphism() -> None:
    source_operator = np.array([[0, 1], [1, 0]], dtype=float)
    target_operator = np.diag([1, -1])
    change = np.array([[1, 1], [1, -1]], dtype=float)
    source = SeedSystem({"T": source_operator}, [np.array([1, 0])])
    target = SeedSystem({"D": target_operator}, [change @ np.array([1, 0])])
    morphism = SeedMorphism(
        source,
        target,
        change,
        {"T": "D"},
        seed_images=[change @ np.array([1, 0])],
    )
    assert morphism.certificate().valid
