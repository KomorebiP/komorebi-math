"""Exact controls for the machinery audited against Papers I--IV."""

from itertools import product

import numpy as np
import pytest

from komorebi_math.core.paper_machinery import (
    finite_mapping_cone,
    rational_primary_packets,
    rational_spectral_projectors,
)
from komorebi_math.deformation.complexes import FiniteCochainComplex
from komorebi_math.exceptions import ValidationError
from komorebi_math.nonlinear.filtered import (
    FilteredCorrectionProblem,
    FilteredVectorSpace,
    FiniteFilteredCochainComplex,
)

LARGE_PRIME = 4294967291


def _scalar_rank_2_by_2(entries: tuple[int, ...], prime: int) -> int:
    """Independent exact scalar determinant/rank oracle, with no array product."""
    a, b, c, d = (entry % prime for entry in entries)
    if (a * d - b * c) % prime:
        return 2
    return int(any((a, b, c, d)))


def test_rank_one_f0_cannot_receive_exact_full_space_certificate() -> None:
    p = LARGE_PRIME
    rank_one = [[1, p - 1], [p - 1, 1]]
    assert _scalar_rank_2_by_2((1, p - 1, p - 1, 1), p) == 1
    with pytest.raises(ValidationError, match=r"F.0 must be the full"):
        FilteredVectorSpace(2, {0: rank_one, 1: np.zeros((2, 0), dtype=object)}, prime=p)


@pytest.mark.parametrize("prime", [2, 7, LARGE_PRIME])
def test_public_filtered_cohomology_matches_scalar_rank_oracle(prime: int) -> None:
    spaces = {degree: FilteredVectorSpace.from_weights([0, 0], prime=prime) for degree in (0, 1)}
    # Every 2x2 matrix with coefficients in {-1,0,1}, including dependent
    # columns and pivots equal to p-1. These are exact bounded controls.
    for entries in product((-1, 0, 1), repeat=4):
        matrix = np.array(entries, dtype=object).reshape(2, 2)
        complex_ = FiniteFilteredCochainComplex({0: 2, 1: 2}, {0: matrix}, spaces, prime=prime)
        expected_rank = _scalar_rank_2_by_2(entries, prime)
        assert complex_.total_cohomology_dimensions() == {0: 2 - expected_rank, 1: 2 - expected_rank}


def test_noncanonical_large_prime_filtration_coordinates_are_exact() -> None:
    p = LARGE_PRIME
    space = FilteredVectorSpace(
        2,
        {0: np.eye(2, dtype=object), 1: [[p - 1], [p - 1]], 2: np.zeros((2, 0), dtype=object)},
        prime=p,
    )
    assert space.dimension_at(1) == 1
    assert space.contains((1, 1), 1)
    assert not space.contains((1, 2), 1)
    coordinates = space.graded_coordinates((1, 1), 1)
    assert tuple(coordinates) == (p - 1,)
    assert tuple(space.lift_graded(coordinates, 1)) == (1, 1)
    assert space.certificate().valid


def test_large_prime_filtered_correction_improves_defect_exactly() -> None:
    p = LARGE_PRIME
    space = FilteredVectorSpace.from_weights([0], prime=p)
    problem = FilteredCorrectionProblem(space, space, [[p - 1]], [p - 1], 0)
    result = problem.solve()
    assert result.vanishes
    assert result.correction_representatives == ((p - 1,),)
    assert result.certificate.valid
    assert ((p - 1) + (p - 1) * result.correction_representatives[0][0]) % p == 0


def test_large_prime_nonzero_filtered_obstruction_stays_nonzero() -> None:
    p = LARGE_PRIME
    source = FilteredVectorSpace.from_weights([0], prime=p)
    target = FilteredVectorSpace.from_weights([0, 0], prime=p)
    result = FilteredCorrectionProblem(source, target, [[p - 1], [p - 1]], [1, 2], 0).solve()
    assert not result.vanishes
    assert not result.correction_representatives
    assert result.certificate.valid  # A correct computed obstruction, not approval of a lift.


def test_large_prime_filtered_complex_and_pages_match_independent_complex() -> None:
    p = LARGE_PRIME
    dimensions = {0: 1, 1: 2, 2: 1}
    differentials = {0: [[p - 1], [p - 1]], 1: [[p - 1, 1]]}
    assert ((p - 1) * (p - 1) + (p - 1)) % p == 0
    spaces = {
        degree: FilteredVectorSpace.from_weights([0] * dimension, prime=p)
        for degree, dimension in dimensions.items()
    }
    filtered = FiniteFilteredCochainComplex(dimensions, differentials, spaces, prime=p)
    independent = FiniteCochainComplex(dimensions, differentials, prime=p)
    expected = {degree: independent.cohomology(degree).cohomology_dimension for degree in dimensions}
    assert expected == {0: 0, 1: 0, 2: 0}
    assert filtered.total_cohomology_dimensions() == expected
    result = filtered.spectral_sequence()
    assert result.certificate.valid
    assert result.limiting_dimensions == expected
    assert all(page.certificate.valid for page in result.pages)

    bad_differentials = {0: differentials[0], 1: [[p - 1, 2]]}
    with pytest.raises(ValidationError, match="do not compose to zero"):
        FiniteFilteredCochainComplex(dimensions, bad_differentials, spaces, prime=p)


def test_huge_input_coefficients_are_reduced_before_filtered_arithmetic() -> None:
    p = LARGE_PRIME
    huge = p * (1 << 200)
    spaces = {degree: FilteredVectorSpace.from_weights([0, 0], prime=p) for degree in (0, 1)}
    complex_ = FiniteFilteredCochainComplex(
        {0: 2, 1: 2},
        {0: [[huge + 1, -huge - 1], [huge - 1, -huge + 1]]},
        spaces,
        prime=p,
    )
    assert complex_.total_cohomology_dimensions() == {0: 1, 1: 1}


@pytest.mark.parametrize("prime", [2, 7, LARGE_PRIME])
def test_mapping_cone_identity_is_acyclic_and_uses_negative_source_differential(prime: int) -> None:
    differential = np.array([[1, 0], [0, 0]], dtype=object)
    complex_ = FiniteCochainComplex({0: 2, 1: 2}, {0: differential}, prime=prime)
    cone = finite_mapping_cone(complex_, complex_, {0: np.eye(2, dtype=object), 1: np.eye(2, dtype=object)})
    assert cone.dimensions == {-1: 2, 0: 4, 1: 2}
    assert np.array_equal(cone.differential(-1)[2:, :], (-differential) % prime)
    assert np.array_equal(cone.differential(0)[:, :2], differential)
    assert all(cone.cohomology(degree).cohomology_dimension == 0 for degree in cone.dimensions)
    assert cone.certificate().valid
    # The two paths around the cochain square cancel entry by entry.
    assert all(int(entry) % prime == 0 for entry in (cone.differential(0) @ cone.differential(-1)).flat)


def test_mapping_cone_degree_zero_recovers_kernel_and_cokernel() -> None:
    prime = LARGE_PRIME
    source = FiniteCochainComplex({0: 1}, {}, prime=prime)
    target = FiniteCochainComplex({0: 2}, {}, prime=prime)
    cone = finite_mapping_cone(source, target, {0: [[prime - 1], [prime - 1]]})
    assert cone.cohomology(-1).cohomology_dimension == 0
    assert cone.cohomology(0).cohomology_dimension == 1
    assert tuple(cone.differential(-1)[:, 0]) == (prime - 1, prime - 1)
    # The functional (1,-1) detects the remaining one-dimensional cokernel.
    assert ((prime - 1) - (prime - 1)) % prime == 0


def test_mapping_cone_rejects_a_noncommuting_square() -> None:
    source = FiniteCochainComplex({0: 1, 1: 1}, {0: [[1]]}, prime=7)
    target = FiniteCochainComplex({0: 1, 1: 1}, {0: [[1]]}, prime=7)
    with pytest.raises(ValidationError, match="does not commute"):
        finite_mapping_cone(source, target, {0: [[1]], 1: [[2]]})


def test_mapping_cone_owns_input_data_and_supports_sparse_negative_degrees() -> None:
    source = FiniteCochainComplex({-3: 1, -1: 1}, {}, prime=7)
    target = FiniteCochainComplex({-3: 1}, {}, prime=7)
    component = np.array([[3]], dtype=object)
    cone = finite_mapping_cone(source, target, {-3: component})
    component[0, 0] = 6
    source.dimensions[-3] = 4
    assert cone.dimensions[-4] == 1
    assert cone.differential(-4)[0, 0] == 3
    assert cone.certificate().valid
    assert cone.cohomology(-2).cohomology_dimension == 1


def test_mapping_cone_rejects_field_mismatch_and_invalid_component_inputs() -> None:
    source = FiniteCochainComplex({0: 1}, {}, prime=7)
    other_field = FiniteCochainComplex({0: 1}, {}, prime=5)
    with pytest.raises(ValidationError, match="same prime"):
        finite_mapping_cone(source, other_field, {})
    with pytest.raises(ValidationError, match="outside both"):
        finite_mapping_cone(source, source, {2: [[1]]})
    with pytest.raises(ValidationError, match="integer"):
        finite_mapping_cone(source, source, {0: [[0.5]]})


def test_rational_spectral_projectors_retain_primary_jordan_blocks() -> None:
    sp = pytest.importorskip("sympy")
    x = sp.Symbol("x")
    operator = sp.Matrix([[0, 1, 0], [0, 0, 0], [0, 0, 2]])
    change = sp.Matrix([[1, 0, 1], [0, 1, 2], [0, 0, 1]])
    transformed = change * operator * change.inv()
    packets = rational_spectral_projectors(transformed, [x**2, x - 2], names=["nilpotent", "eigenvalue_two"])
    expected_first = change * sp.diag(1, 1, 0) * change.inv()
    expected_second = change * sp.diag(0, 0, 1) * change.inv()
    assert packets.projectors == (expected_first, expected_second)
    assert packets.packet_dimensions() == {"nilpotent": 2, "eigenvalue_two": 1}
    assert packets.certificate().valid
    assert packets.certificate().exactness.value == "exact"
    assert transformed**2 * packets.projectors[0] == sp.zeros(3)
    assert (transformed - 2 * sp.eye(3)) * packets.projectors[1] == sp.zeros(3)


def test_rational_spectral_projectors_allow_an_irreducible_quadratic() -> None:
    sp = pytest.importorskip("sympy")
    x = sp.Symbol("x")
    operator = sp.Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 2]])
    packets = rational_spectral_projectors(operator, [sp.Poly(x**2 + 1, x), sp.Poly(x - 2, x)])
    assert packets.projectors == (sp.diag(1, 1, 0), sp.diag(0, 0, 1))
    assert packets.certificate().valid


def test_rational_spectral_projectors_accept_a_single_characteristic_packet() -> None:
    sp = pytest.importorskip("sympy")
    x = sp.Symbol("x")
    packets = rational_spectral_projectors([[0, 1], [0, 0]], [x**2])
    assert packets.projectors == (sp.eye(2),)


def test_rational_spectral_projectors_reject_nonmatching_or_noncoprime_factors() -> None:
    sp = pytest.importorskip("sympy")
    x = sp.Symbol("x")
    with pytest.raises(ValidationError, match="characteristic polynomial"):
        rational_spectral_projectors(sp.diag(0, 2), [x, x - 3])
    with pytest.raises(ValidationError, match="coprime"):
        rational_spectral_projectors(sp.zeros(2), [x, x])
    with pytest.raises(ValidationError, match="monic"):
        rational_spectral_projectors(sp.diag(0, 2), [2 * x, x - 2])
    with pytest.raises(ValidationError, match="nonconstant"):
        rational_spectral_projectors(sp.diag(0, 2), [1, x * (x - 2)])


def test_rational_spectral_projectors_reject_inexact_coefficients() -> None:
    sp = pytest.importorskip("sympy")
    x = sp.Symbol("x")
    with pytest.raises(ValidationError, match="exact rational entries"):
        rational_spectral_projectors([[0.0, 0], [0, 2]], [x, x - 2])
    with pytest.raises(ValidationError, match="exact rational coefficients"):
        rational_spectral_projectors(sp.diag(0, 2), [x, x - 2.0])


def test_rational_primary_packets_discover_exact_characteristic_primary_factors() -> None:
    sp = pytest.importorskip("sympy")
    operator = sp.Matrix([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]])
    change = sp.eye(4)
    change[0, 3] = sp.Rational(3, 7)
    transformed = change * operator * change.inv()
    packets = rational_primary_packets(transformed)
    # Primary packet ordering follows the exact factorizer; compare the full
    # projector set with a direct rational change-of-basis construction.
    expected = (change * sp.diag(1, 1, 0, 0) * change.inv(), change * sp.diag(0, 0, 1, 1) * change.inv())
    assert len(packets.projectors) == 2
    assert all(any(actual == projector for actual in packets.projectors) for projector in expected)
    assert packets.certificate().valid
    assert all(transformed * projector == projector * transformed for projector in packets.projectors)


def test_rational_primary_packets_keep_repeated_nilpotent_factor_in_one_packet() -> None:
    sp = pytest.importorskip("sympy")
    assert rational_primary_packets([[0, 1], [0, 0]]).projectors == (sp.eye(2),)
    with pytest.raises(ValidationError, match="exact rational entries"):
        rational_primary_packets([[0.0]])
