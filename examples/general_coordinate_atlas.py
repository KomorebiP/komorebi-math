"""Build an exact coordinate atlas from arbitrary finite bases."""

from fractions import Fraction

from komorebi_math.coordinates import (
    CoordinateAtlas,
    CoordinateSystem,
    basis_from_matrix,
    standard_basis,
)

standard = standard_basis(3, name="standard")
shear = basis_from_matrix(
    "shear",
    (
        (1, 1, 0),
        (0, 1, 1),
        (0, 0, 1),
    ),
)
scaled = basis_from_matrix(
    "scaled",
    (
        (2, 0, 0),
        (0, 3, 0),
        (0, 0, 5),
    ),
)

atlas = CoordinateAtlas("three-basis atlas")
for basis in (standard, shear, scaled):
    atlas.register_system(CoordinateSystem(basis.name, basis.dimension, "basis"))

atlas.register_transform(standard.connection_to(shear), include_inverse=True)
atlas.register_transform(shear.connection_to(scaled), include_inverse=True)
atlas.register_transform(standard.connection_to(scaled), include_inverse=True)

commuting = atlas.compare_paths(
    ("standard", "scaled"),
    ("standard", "shear", "scaled"),
)
commuting.certificate().require()

vector = (Fraction(7), Fraction(-3), Fraction(11))
coordinates = atlas.resolve("standard", "scaled").transform.apply(vector)
assert scaled.synthesize(coordinates) == vector

print("path:", atlas.resolve("standard", "scaled").systems)
print("scaled coordinates:", coordinates)
