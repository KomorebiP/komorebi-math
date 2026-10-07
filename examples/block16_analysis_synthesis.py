"""Certified finite analysis--synthesis and operator-block example."""

from __future__ import annotations

import numpy as np

from komorebi_math.stable import AnalysisSynthesisSystem, dumps_portable, loads_portable

analysis = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, -1.0],
    ]
)
system = AnalysisSynthesisSystem.canonical_dual(
    analysis,
    source="signal",
    coordinates="redundant analysis coordinates",
)
system.certificate().require()

value = np.array([3.0, -2.0])
reconstructed = system.reconstruct(value)
assert np.allclose(reconstructed, value)

operator = np.array([[2.0, -1.0], [3.0, 4.0]])
blocks = system.operator_blocks(operator, groups=((0, 2), (1, 3)))
blocks.certificate().require()
assert np.allclose(blocks.reconstruct(), operator)

restored = loads_portable(dumps_portable(system))
restored.certificate().require()
assert np.allclose(restored.reconstruct(value), value)

print("analysis-synthesis certificate", system.certificate().to_dict())
print("operator-block certificate", blocks.certificate().to_dict())
print("reconstructed", reconstructed)
