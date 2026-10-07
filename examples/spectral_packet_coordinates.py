"""Combine numerical spectral coordinates with exact packet permutations."""

import numpy as np

from komorebi_math.coordinates import (
    discrete_fourier_transform,
    haar_wavelet_transform,
    residue_class_transform,
)

samples = np.arange(8, dtype=float)
fourier = discrete_fourier_transform(8)
haar = haar_wavelet_transform(8)

fourier.certificate().require()
haar.certificate().require()
assert np.allclose(fourier.inverse().apply(fourier.apply(samples)), samples)
assert np.allclose(haar.inverse().apply(haar.apply(samples)), samples)

residue = residue_class_transform(8, 2)
packet_order = residue.apply(tuple(range(8)))
assert residue.inverse().apply(packet_order) == tuple(range(8))

print("Fourier coordinates:", fourier.apply(samples))
print("Haar coordinates:", haar.apply(samples))
print("residue-lane order:", packet_order)
