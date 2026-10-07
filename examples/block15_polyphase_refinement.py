"""Minimal certified Block 15 refinement example."""

import numpy as np

from komorebi_math.refinement import PolyphaseRefinement, PolyphaseRefinementTower
from komorebi_math.signals import polyphase_analyze

signal = np.arange(29)
coarse = polyphase_analyze(signal, 4)
refinement = PolyphaseRefinement(4, 12)
fine = refinement.refine(coarse)
certificate = refinement.certificate(coarse)
certificate.require()

assert refinement.children(1) == (1, 5, 9)
assert np.array_equal(fine.reconstruct(), signal)
assert np.array_equal(refinement.coarsen(fine).reconstruct(), signal)

# Sequential refinement agrees with direct refinement.
tower = PolyphaseRefinementTower((2, 6, 12))
tower_certificate = tower.coherence_certificate(polyphase_analyze(signal, 2))
tower_certificate.require()

print("Block 15 polyphase refinement verified")
