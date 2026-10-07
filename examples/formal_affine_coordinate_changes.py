"""Use affine, barycentric, and formal coordinate changes."""

from fractions import Fraction

from komorebi_math.coordinates import (
    BarycentricCoordinateSystem,
    FormalCoordinateChange,
    affine_basis_transform,
)

chart = affine_basis_transform(
    (3, -2),
    ((1, 1), (1, -1)),
    local_name="diagonal local",
    ambient_name="Cartesian",
)
local = (Fraction(5, 2), Fraction(-3, 2))
assert chart.inverse().apply(chart.apply(local)) == local

simplex = BarycentricCoordinateSystem("triangle", ((0, 0), (2, 0), (0, 3)))
weights = (Fraction(1, 6), Fraction(1, 3), Fraction(1, 2))
assert simplex.from_cartesian(simplex.to_cartesian(weights)) == weights

formal = FormalCoordinateChange("near identity", (0, 1, 1, -2, 3), "u", "z")
formal.certificate().require()
series = (3, 4, -1, 2, 5)
pulled = formal.pullback_coefficients(series)
assert formal.pushforward_coefficients(pulled) == tuple(Fraction(value) for value in series)

print("Cartesian point:", chart.apply(local))
print("barycentric point:", simplex.to_cartesian(weights))
print("formal pullback:", pulled)
