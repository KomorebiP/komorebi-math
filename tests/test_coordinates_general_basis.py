from fractions import Fraction

from komorebi_math.coordinates.basis import (
    CoordinateAtlas,
    CoordinateSystem,
    basis_from_matrix,
    direct_sum_basis,
    standard_basis,
    tensor_product_basis,
)
from komorebi_math.coordinates.transform import CoordinateTransform


def test_arbitrary_finite_basis_analysis_and_connection() -> None:
    standard = standard_basis(2, name="standard")
    shear = basis_from_matrix("shear", ((1, 1), (0, 1)), labels=("u", "v"))
    transform = standard.connection_to(shear)
    assert transform.apply((3, 5)) == (Fraction(-2), Fraction(5))
    assert shear.synthesize(transform.apply((3, 5))) == (Fraction(3), Fraction(5))
    assert transform.inverse().apply(transform.apply((3, 5))) == (Fraction(3), Fraction(5))
    assert shear.certificate().valid
    assert shear.orthogonal_projector() == ((1, 0), (0, 1))


def test_rectangular_subspace_basis_and_projector() -> None:
    plane = basis_from_matrix("plane", ((1, 0), (0, 1), (0, 0)))
    assert plane.dimension == 2
    assert plane.ambient_dimension == 3
    assert plane.contains((2, 3, 0))
    assert not plane.contains((2, 3, 1))
    assert plane.orthogonal_projector() == ((1, 0, 0), (0, 1, 0), (0, 0, 0))


def test_coordinate_atlas_resolves_and_checks_commuting_paths() -> None:
    atlas = CoordinateAtlas("test atlas")
    for name in ("A", "B", "C", "D"):
        atlas.register_system(CoordinateSystem(name, 2))
    ab = CoordinateTransform("A", "B", ((1, 1), (0, 1)), "test")
    bc = CoordinateTransform("B", "C", ((1, 0), (1, 1)), "test")
    ac = ab.then(bc)
    cd = CoordinateTransform("C", "D", ((2, 0), (0, 3)), "test")
    atlas.register_transform(ab, include_inverse=True)
    atlas.register_transform(bc, include_inverse=True)
    atlas.register_transform(ac)
    atlas.register_transform(cd, include_inverse=True)
    resolved = atlas.resolve("A", "D")
    assert resolved.systems in (("A", "C", "D"), ("A", "B", "C", "D"))
    check = atlas.compare_paths(("A", "C", "D"), ("A", "B", "C", "D"))
    assert check.valid
    check.certificate().require()


def test_direct_sum_and_tensor_product_bases() -> None:
    left = standard_basis(2, name="L")
    right = basis_from_matrix("R", ((1, 1), (1, -1)))
    direct = direct_sum_basis(left, right)
    tensor = tensor_product_basis(left, right)
    assert direct.dimension == 4
    assert direct.ambient_dimension == 4
    assert tensor.dimension == 4
    assert tensor.ambient_dimension == 4
    assert tensor.certificate().valid
