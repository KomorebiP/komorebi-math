import numpy as np
import pytest

from komorebi_math.exceptions import ValidationError
from komorebi_math.nonlinear.filtered import (
    FilteredCorrectionProblem,
    FilteredVectorSpace,
    FiniteFilteredCochainComplex,
    conditional_projective_limit_completion,
    verify_finite_filtered_branch,
)


def test_weight_filtration_and_graded_coordinates_round_trip() -> None:
    space = FilteredVectorSpace.from_weights([0, 1], prime=5)

    assert tuple(space.dimension_at(depth) for depth in range(3)) == (2, 1, 0)
    assert tuple(space.graded_dimension(depth) for depth in range(2)) == (1, 1)
    assert np.array_equal(space.graded_coordinates([3, 4], 0), np.array([3]))
    assert np.array_equal(space.graded_coordinates([0, 4], 1), np.array([4]))
    assert np.array_equal(space.lift_graded([3], 0), np.array([3, 0]))
    assert np.array_equal(space.lift_graded([4], 1), np.array([0, 4]))
    assert space.certificate().valid


def test_one_step_obstruction_detects_failure_and_constructs_correction() -> None:
    source = FilteredVectorSpace.from_weights([0, 1], prime=5)
    target = FilteredVectorSpace.from_weights([0, 1], prime=5)

    obstructed = FilteredCorrectionProblem(
        source,
        target,
        np.diag([1, 0]),
        [0, 1],
        1,
    ).solve()
    assert not obstructed.vanishes
    assert obstructed.correction_representatives == ()
    assert obstructed.certificate.valid

    solved = FilteredCorrectionProblem(source, target, np.eye(2, dtype=int), [0, 1], 1).solve()
    assert solved.vanishes
    assert solved.correction_space.solution_count == 1
    assert solved.correction_representatives == ((0, 4),)
    corrected = (np.array([0, 1]) + np.asarray(solved.correction_representatives[0])) % 5
    assert target.contains(corrected, 2)
    assert solved.certificate.valid


def test_finite_filtered_branch_reaches_exact_terminal_solution() -> None:
    source = FilteredVectorSpace.from_weights([0, 1], prime=5)
    target = FilteredVectorSpace.from_weights([0, 1], prime=5)

    result = verify_finite_filtered_branch(
        source,
        target,
        lambda point: (point[0] - 1, point[1] - 2),
        [(0, 0), (1, 0), (1, 2)],
    )

    assert result.limit == (1, 2)
    assert result.exact_solution
    assert result.certificate.valid


def test_projective_limit_record_keeps_external_hypotheses_visible() -> None:
    completed = conditional_projective_limit_completion(
        7,
        separated=True,
        complete=True,
        continuous=True,
        compatible_infinite_branch=True,
    )
    assert completed.certificate.valid
    assert completed.certificate.exactness.value == "conditional"
    assert "external inputs" in completed.certificate.assumptions_unchecked[0]

    incomplete = conditional_projective_limit_completion(
        7,
        separated=True,
        complete=False,
        continuous=True,
        compatible_infinite_branch=True,
    )
    assert not incomplete.certificate.valid
    assert "complete" in incomplete.certificate.assumptions_unchecked


def _zero_terminal_space(dimension: int, prime: int, terminal_depth: int) -> FilteredVectorSpace:
    levels = {0: np.eye(dimension, dtype=int)}
    for depth in range(1, terminal_depth + 1):
        levels[depth] = np.zeros((dimension, 0), dtype=int)
    return FilteredVectorSpace(dimension, levels, prime=prime)


def test_spectral_sequence_records_a_nonzero_d1_cancellation() -> None:
    prime = 5
    c0 = _zero_terminal_space(1, prime, terminal_depth=2)
    c1 = FilteredVectorSpace.from_weights([1], prime=prime)
    complex_ = FiniteFilteredCochainComplex(
        {0: 1, 1: 1},
        {0: [[1]]},
        {0: c0, 1: c1},
        prime=prime,
    )

    e0 = complex_.page(0)
    e1 = complex_.page(1)
    e2 = complex_.page(2)
    result = complex_.spectral_sequence()

    assert e0.entry(0, 0).dimension == 1
    assert e0.entry(1, 0).dimension == 1
    assert e1.entry(0, 0).dimension == 1
    assert e1.entry(1, 0).dimension == 1
    assert e1.entry(0, 0).outgoing_rank == 1
    assert e2.total_dimension == 0
    assert result.total_cohomology_dimensions == {0: 0, 1: 0}
    assert result.limiting_dimensions == {0: 0, 1: 0}
    assert result.certificate.valid


def test_zero_filtered_differential_degenerates_and_converges() -> None:
    c0 = FilteredVectorSpace.from_weights([0, 1], prime=3)
    c1 = FilteredVectorSpace.from_weights([0, 1], prime=3)
    complex_ = FiniteFilteredCochainComplex(
        {0: 2, 1: 2},
        {0: np.zeros((2, 2), dtype=int)},
        {0: c0, 1: c1},
        prime=3,
    )
    result = complex_.spectral_sequence()

    assert all(page.total_differential_rank == 0 for page in result.pages)
    assert result.pages[0].total_dimension == 4
    assert result.pages[-1].total_dimension == 4
    assert result.total_cohomology_dimensions == {0: 2, 1: 2}
    assert result.certificate.valid


def test_filtered_complex_rejects_non_preserving_differential() -> None:
    source = FilteredVectorSpace.from_weights([1, 0], prime=5)
    target = FilteredVectorSpace.from_weights([1, 0], prime=5)
    with pytest.raises(ValidationError, match="does not preserve"):
        FiniteFilteredCochainComplex(
            {0: 2, 1: 2},
            {0: [[0, 0], [1, 0]]},
            {0: source, 1: target},
            prime=5,
        )
