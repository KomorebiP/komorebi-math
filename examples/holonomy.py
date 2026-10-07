"""Finite matrix holonomy and gauge invariance."""

from __future__ import annotations

import numpy as np

from komorebi_math import TransportEdge, TransportGraph


A = np.array([[1.0, 1.0], [0.0, 1.0]])
B = np.array([[1.0, 0.0], [1.0, 1.0]])
C = B @ A

graph = TransportGraph(
    ["a", "b", "c"],
    [
        TransportEdge("a", "b", A, "ab"),
        TransportEdge("b", "c", B, "bc"),
        TransportEdge("a", "c", C, "ac"),
    ],
)
graph.gluing_certificate().require()

gauges = {
    "a": np.eye(2),
    "b": np.array([[2.0, 0.0], [0.0, 1.0]]),
    "c": np.array([[1.0, 0.0], [0.0, 3.0]]),
}
transformed = graph.gauge_transform(gauges)
transformed.gluing_certificate().require()

broken = TransportGraph(
    ["a", "b", "c"],
    [
        TransportEdge("a", "b", A, "ab"),
        TransportEdge("b", "c", B, "bc"),
        TransportEdge("a", "c", 2.0 * C, "ac"),
    ],
)
assert not broken.gluing_certificate().valid

print("fundamental cycles:", len(graph.fundamental_holonomies()))
print("gauge invariant gluing:", transformed.gluing_certificate().valid)
print("broken cycle detected:", not broken.gluing_certificate().valid)
