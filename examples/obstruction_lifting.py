"""Cokernel obstruction and exact prime power lifting."""

from __future__ import annotations

import sympy as sp

from komorebi_math import LinearObstructionProblem, PolynomialLiftProblem
from komorebi_math.nonlinear import build_lift_tree


obstruction = LinearObstructionProblem(
    sp.Matrix([[1, 0], [0, 0]]),
    sp.Matrix([0, 1]),
    backend="sympy",
).solve()
assert not obstruction.vanishes
assert obstruction.cokernel_dimension == 1

F = lambda x: (x[0] * x[0] - 1,)
J = lambda x: [[2 * x[0]]]


def factory(point: tuple[int, ...], depth: int) -> PolynomialLiftProblem:
    return PolynomialLiftProblem(F, J, point, prime=3, depth=depth)


tree = build_lift_tree(factory, [(1,), (2,)], prime=3, start_depth=1, max_depth=5)
branches = tree.terminal_branches()
assert len(branches) == 2
assert all(len(branch) == 5 for branch in branches)

print("obstruction coordinates:", obstruction.obstruction_coordinates)
print("lift levels:", len(tree.levels))
print("terminal branches:", len(branches))
