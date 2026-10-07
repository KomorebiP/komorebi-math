"""Polyphase coordinates, convolution transfer, and Haar reconstruction."""

from __future__ import annotations

import numpy as np

from komorebi_math.signals import HaarPacketTree, haar_analyze, haar_synthesize, polyphase_analyze, polyphase_convolution

signal = np.arange(16, dtype=float)
coordinates = polyphase_analyze(signal, 4)
coordinates.certificate(signal).require()

filtered = polyphase_convolution(signal, [0.25, 0.5, 0.25], 4)
filtered.certificate.require()

low, high = haar_analyze(signal)
assert np.allclose(haar_synthesize(low, high), signal)

packet_tree = HaarPacketTree.analyze(signal, depth=3)
packet_tree.certificate().require()
assert np.allclose(packet_tree.prune(("0", "10", "110", "111")).reconstruct(), signal)

print("polyphase channels:", tuple(channel.size for channel in coordinates.channels))
print("filtered length:", filtered.value.size)
print("packet tree nodes:", len(packet_tree.nodes))
