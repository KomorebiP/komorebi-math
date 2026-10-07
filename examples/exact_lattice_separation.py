"""Exact rational packet separation and Smith congruence solving."""

from __future__ import annotations

import sympy as sp

from komorebi_math import LatticePacketData, SmithCongruenceProblem


P0 = sp.Matrix([[1, sp.Rational(1, 4)], [0, 0]])
P1 = sp.eye(2) - P0
split_basis = sp.Matrix([[1, -1], [0, 4]])

data = LatticePacketData(
    sp.eye(2),
    [P0, P1],
    prime=2,
    split_basis=split_basis,
)

data.gluing_weight_certificate().require()
assert data.projector_weights() == (2, 2)
assert data.gluing_module().invariant_factors == (1, 4)

problem = SmithCongruenceProblem(
    [[2, 0, 0], [0, 4, 0]],
    [2, 4],
    prime=2,
    depth=3,
)
solution = problem.solve().require()
assert solution.solution_count == 64

print("projector weights:", data.projector_weights())
print("gluing factors:", data.gluing_module().invariant_factors)
print("congruence solutions:", solution.solution_count)
