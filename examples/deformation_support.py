"""Build a finite residual forest and verify a surviving occupancy packet."""

from __future__ import annotations

import numpy as np

from komorebi_math.deformation import (
    nested_subspace_occupancy,
    normal_crossing_count,
    residue_forest,
)


def main() -> None:
    """Run the finite deformation and occupancy example."""

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

    occupancy = nested_subspace_occupancy(
        (
            np.eye(3),
            np.array([[1, 0], [0, 1], [0, 0]], dtype=float),
            np.array([[1], [0], [0]], dtype=float),
        )
    )
    occupancy.certificate.require()

    print(
        {
            "level_sizes": tuple(len(level) for level in report.levels),
            "stable_core_sizes": tuple(len(core) for core in report.stable_cores),
            "branch_count": len(report.branches),
            "intersection_dimension": occupancy.intersection_dimension,
            "normal_crossing_count_q3_depth2": normal_crossing_count(3, 2),
        }
    )


if __name__ == "__main__":
    main()
