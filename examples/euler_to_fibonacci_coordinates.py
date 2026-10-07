from __future__ import annotations

from komorebi_math.coordinates import euler_fibonacci_coordinates, variation_of_constants_prefix

record = euler_fibonacci_coordinates(18)
record.certificate().require()
assert record.reconstruct() == variation_of_constants_prefix(record)
print("initial", tuple(str(value) for value in record.initial))
print("defects", tuple(str(value) for value in record.defects))
print("reconstructed", tuple(str(value) for value in record.reconstruct()))
