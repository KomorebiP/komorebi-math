from __future__ import annotations

import sympy as sp

from komorebi_math import OperatorAlgebra, PacketSystem, single_operator_seed_rank

nilpotent = sp.Matrix([[0, 1], [0, 0]])
algebra = OperatorAlgebra([nilpotent], backend="sympy")
print("algebra dimension:", algebra.dimension)

packets = PacketSystem(
    [sp.diag(1, 0), sp.diag(0, 1)],
    names=["first", "second"],
    backend="sympy",
)
print(packets.certificate().to_json())

repeated = sp.diag(sp.Matrix([[4, 1], [0, 4]]), sp.Matrix([[4]]))
print("single operator seed rank:", single_operator_seed_rank(repeated))
