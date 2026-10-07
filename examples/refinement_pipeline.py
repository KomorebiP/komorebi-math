"""Packet refinement, residue coordinates, and weighted transfer."""

from __future__ import annotations

import numpy as np

from komorebi_math import PacketRefinement, PacketSystem, WeightedTransferEdge, WeightedTransferGraph
from komorebi_math.refinement import RefinementTower, crt_many, residue_analyze


coarse = PacketSystem(
    [
        np.diag([1, 1, 0, 0]),
        np.diag([0, 0, 1, 1]),
    ],
    names=["low", "high"],
)
fine = PacketSystem(
    [
        np.diag([1, 0, 0, 0]),
        np.diag([0, 1, 0, 0]),
        np.diag([0, 0, 1, 0]),
        np.diag([0, 0, 0, 1]),
    ],
    names=["0", "1", "2", "3"],
)

step = PacketRefinement(coarse, fine, (0, 0, 1, 1))
tower = RefinementTower([step])
vector = np.array([1.0, 2.0, 3.0, 4.0])
coarse_coordinates = coarse.analyze(vector)
fine_coordinates = tower.refine(coarse_coordinates, source_level=0, target_level=1)
coarse_again = tower.coarsen(fine_coordinates, source_level=1, target_level=0)
assert np.allclose(coarse_again.reconstruct(), vector)

residues = residue_analyze(np.arange(12), modulus=3)
assert np.array_equal(residues.reconstruct(), np.arange(12))
address = crt_many((1, 2, 4), (2, 3, 5))
assert address.residue == 29

edge_01 = WeightedTransferEdge(
    0,
    1,
    1,
    np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    ),
)
edge_12 = WeightedTransferEdge(
    1,
    2,
    2,
    np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    ),
)
graph = WeightedTransferGraph(fine, np.eye(4), [edge_01, edge_12])
assert graph.minimum_path_weight(0, 2) == 3
exact_inverse = graph.exact_inverse_if_acyclic().require()
assert np.allclose(exact_inverse, np.linalg.inv(graph.operator))

print("refinement levels:", len(tower.levels))
print("CRT address:", address.residue)
print("minimum transfer weight 0 to 2:", graph.minimum_path_weight(0, 2))
