"""Cartier Chapter 24: a nonfree quotient and a noncommuting witness."""

from komorebi_math.generating.residual_modules import (
    RingLinearRealization,
    canonical_residual_module,
    finite_word_comparison,
)


def main() -> None:
    # Example 24.4 is printed with row states: transpose for column states.
    realization = RingLinearRealization(2, 2, (((1, 0), (2, 1)),), (1, 0), (2, 0))
    residual = canonical_residual_module(realization)
    assert residual.ambient_cardinality == 16
    assert len(residual.reachable_states) == 8
    assert residual.carrier_cardinality == 2
    assert residual.additive_p_exponents() == (1,)
    assert residual.verify_laws().valid
    assert finite_word_comparison(realization, residual).equivalent

    print("Ring: Z/4Z")
    print("Ambient / reachable / residual cardinalities:", 16, 8, 2)
    print("Residual additive factors: Z/2Z; not a free Z/4Z-module")
    print("Residual addition:", residual.addition)
    print("Output:", residual.outputs)

    upper = ((0, 1), (0, 0))
    lower = ((0, 0), (1, 0))
    left = RingLinearRealization(2, 1, (upper, lower), (1, 0), (1, 0))
    right = RingLinearRealization(2, 1, (lower, upper), (1, 0), (1, 0))
    comparison = finite_word_comparison(left, right)
    assert comparison.witness == (0, 1)
    assert comparison.left_output == 0
    assert comparison.right_output == 1
    print("Shortest noncommuting distinguishing word:", comparison.witness)
    print("Outputs:", comparison.left_output, comparison.right_output)
    print("Theorem length condition: strictly below", comparison.length_horizon)


if __name__ == "__main__":
    main()
