"""Worked exact reconstruction, irrational discrepancy and inexact-solve checks."""

from fractions import Fraction as F

from komorebi_math.signals.tikhonov import (
    bracket_discrepancy,
    candidate_noise_acceptance,
    certify_squared_discrepancy,
    normal_residual_correction,
    protected_discrepancy_window,
    select_discrepancy,
    tikhonov_system,
)


def main() -> None:
    # Sufficient Resolution, Example 11.5: repeated source coordinates,
    # two response directions that cannot be fitted, and positive image bias.
    system = tikhonov_system([[1, 1], [0, 0], [0, 0]], 1)
    solution = system.solve([3, 4, 5])
    solution.certificate.require()
    system.whole_map_bounds().require()
    print("Source:", list(solution.source))
    print("Squared fit:", solution.fitting_squared, "objective:", solution.objective)
    print("Unregularized source objective:", system.objective_increment([3, 4, 5], [F(3, 2), F(3, 2)]).candidate_objective)

    # Rank-deficient matrix: attainable squared floor 1, target 3/2. The
    # positive root is 1+sqrt(2); only a rational bracket is returned.
    bracket = bracket_discrepancy([[1], [0]], [1, 1], F(3, 2), width=F(1, 1 << 12))
    bracket.certificate.require()
    print("Discrepancy parameter bracket:", bracket.lower, bracket.upper)
    print("Endpoint squared residuals:", bracket.lower_residual_squared, bracket.upper_residual_squared)
    print("Exact rational root found:", bracket.exact_parameter is not None)
    selected = select_discrepancy([[1]], [1], F(1, 10), 2)
    selected.certificate.require()
    print("Accepted rational parameter:", selected.parameter, "squared fit:", selected.fitting_squared)

    # Computed 81/100 differs from the actual regularized 4/5. The fitting
    # residual alone is insufficient; the actual normal residual is 1/80.
    scalar = tikhonov_system([[1]], F(1, 4))
    computed = [F(81, 100)]
    correction = normal_residual_correction(scalar, [1], computed)
    print("Normal residual squared:", correction.normal_squared)
    print("Source/image squared allowances:", correction.source_bound_squared, correction.image_bound_squared)
    window = protected_discrepancy_window(F(1, 10), 2, F(1, 100), F(1, 100)).require_nonempty()
    print("Protected scalar window:", window.lower, window.upper, "midpoint:", window.midpoint)
    certified = certify_squared_discrepancy(
        scalar, [1], computed, eta=F(1, 10), tau=3,
        rho=F(1, 80), xi=F(1, 80), kappa=F(1, 20), measured=F(19, 100), omega=0,
    )
    certified.certificate.require()
    print("Squared protected inexact certificate:", certified.certificate.valid)
    forged = certify_squared_discrepancy(
        scalar, [1], computed, eta=F(1, 10), tau=3,
        rho=0, xi=0, kappa=0, measured=F(19, 100), omega=0,
    )
    print("Falsely reported zero normal residual accepted:", forged.certificate.valid)

    # SR-60: valid truth/model allowances alone do not imply feasibility.
    zero_model = tikhonov_system([[0]], F(1, 4))
    rejected = candidate_noise_acceptance(zero_model, [1], [0], eta=F(1, 10), delta=1, tau=2)
    print("Zero-model generalized acceptance:", rejected.accepted)
    print("Noise truth and infinite-family convergence remain separate premises.")


if __name__ == "__main__":
    main()
