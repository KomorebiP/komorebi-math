"""Stable jump recovery and retained packet reconstruction."""

from __future__ import annotations

import numpy as np

from komorebi_math.signals import SingularPacket1D, recover_separated_jumps, retained_packet_reconstruction

count = 512
spacing = 1.0 / count
grid = np.arange(count + 1) * spacing
locations = np.array([0.23, 0.71])
amplitudes = np.array([0.45, -0.35])
smooth = 0.04 * np.sin(2.0 * np.pi * grid)
signal = smooth.copy()
for location, amplitude in zip(locations, amplitudes, strict=True):
    signal += amplitude * (grid >= location)

recovery = recover_separated_jumps(
    signal,
    spacing=spacing,
    lipschitz_bound=0.08 * np.pi,
    noise_bound=0.0,
    minimum_jump=float(np.min(np.abs(amplitudes))),
    minimum_separation=0.4,
)
recovery.certificate.require()

packet = SingularPacket1D.from_jumps(recovery.locations, recovery.amplitudes).evaluate(grid)
identity = retained_packet_reconstruction(
    signal,
    packet,
    np.eye(signal.size),
    true_smooth=signal - packet,
    true_packet=packet,
)
identity.certificate.require()

print("detected cells:", recovery.cells)
print("estimated locations:", recovery.locations)
print("estimated amplitudes:", recovery.amplitudes)
