from __future__ import annotations

import numpy as np

from komorebi_math.nonlinear import (
    CallableDerivativeProvider,
    LyapunovSchmidtReduction,
    NonlinearSchurElimination,
    PolynomialFixedPointProblem,
    majorant_convergence,
)


def main() -> None:
    fixed_point = PolynomialFixedPointProblem(
        np.array([0.1]),
        {2: lambda values: np.array([values[0][0] * values[1][0]])},
    )
    trees = fixed_point.tree_expansion(4)
    trees.certificate.require()
    convergence = majorant_convergence(0.1, {2: 1.0}, radius=0.2)
    convergence.certificate.require()
    print("finite tree sum:", trees.value[0])
    print("Picard tail bound at depth 4:", convergence.picard_tail_bound(4))

    retained = CallableDerivativeProvider(
        lambda point: np.array([3 * point[0] + 2 * point[1]]),
        {1: lambda point, directions: np.array([3 * directions[0][0] + 2 * directions[0][1]])},
        source_dimension=2,
        target_dimension=1,
    )
    hidden = CallableDerivativeProvider(
        lambda point: np.array([point[1] - 4 * point[0]]),
        {1: lambda point, directions: np.array([directions[0][1] - 4 * directions[0][0]])},
        source_dimension=2,
        target_dimension=1,
    )
    schur = NonlinearSchurElimination(retained, hidden, 1, 1, [0.0], [0.0])
    derivatives = schur.first_derivatives()
    derivatives.certificate.require()
    print("implicit derivative:", derivatives.implicit_derivative)
    print("reduced derivative:", derivatives.reduced_derivative)

    provider = CallableDerivativeProvider(
        lambda point: np.array([point[1], point[0] ** 2]),
        {
            1: lambda point, directions: np.array(
                [directions[0][1], 2 * point[0] * directions[0][0]]
            ),
            2: lambda point, directions: np.array(
                [0.0, 2 * directions[0][0] * directions[1][0]]
            ),
        },
        source_dimension=2,
        target_dimension=2,
    )
    reduction = LyapunovSchmidtReduction(provider, [0.0, 0.0], tolerance=1e-12)
    point = reduction.solve_range([0.3])
    point.certificate.require()
    print("reduced obstruction:", point.reduced_obstruction)


if __name__ == "__main__":
    main()
