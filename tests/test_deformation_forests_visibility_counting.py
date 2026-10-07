from fractions import Fraction

import pytest

from komorebi_math.deformation import (
    FiniteLiftForest,
    VisibilityTree,
    nilpotent_chart_count,
    nilpotent_chart_series,
    normal_crossing_count,
    normal_crossing_series,
    residue_forest,
    smooth_chart_count,
    smooth_chart_series,
    weighted_chart_count,
)
from komorebi_math.exceptions import ValidationError


def test_finite_lift_forest_stable_core_and_branches() -> None:
    forest = residue_forest(
        {
            1: (0, 1),
            2: (0, 1, 2, 3),
            3: (0, 1, 4, 5),
        },
        prime=2,
    )
    report = forest.report()
    report.certificate.require()
    assert forest.stable_core(1) == (0, 1)
    assert len(report.branches) == 4
    assert all(branch.depth == 3 for branch in report.branches)


def test_forest_detects_invalid_reduction() -> None:
    with pytest.raises(ValidationError):
        FiniteLiftForest({1: (0,), 2: (1,)}, {2: {1: 9}})


def test_visibility_partitions_refine() -> None:
    tree = VisibilityTree((0, 4, 2, 1), prime=2)
    assert tree.separation_depth(0, 1) == 3
    assert tree.partition(1) == ((0, 1, 2), (3,))
    assert tree.partition(2) == ((0, 1), (2,), (3,))
    assert tree.partition(3) == ((0,), (1,), (2,), (3,))
    assert tree.ultrametric_certificate()


def test_visibility_exact_duplicates_never_separate() -> None:
    tree = VisibilityTree((Fraction(1, 3), Fraction(1, 3)), prime=2)
    assert tree.partition(20) == ((0, 1),)


def test_counting_formulas_and_series() -> None:
    assert smooth_chart_count(3, 2, 4) == 3**6
    assert weighted_chart_count(3, (1, 3), 4) == 3**4
    assert nilpotent_chart_count(5, 3, 7) == 5 ** (7 - 3)
    assert normal_crossing_count(3, 2) == 21
    smooth = smooth_chart_series(3, 2)
    assert smooth.first(5) == (0, 1, 9, 81, 729)
    nilpotent = nilpotent_chart_series(5, 3)
    assert tuple(int(value) for value in nilpotent.first(8)[1:]) == tuple(
        nilpotent_chart_count(5, 3, depth) for depth in range(1, 8)
    )
    crossing = normal_crossing_series(3)
    assert tuple(int(value) for value in crossing.first(6)[1:]) == tuple(
        normal_crossing_count(3, depth) for depth in range(1, 6)
    )
