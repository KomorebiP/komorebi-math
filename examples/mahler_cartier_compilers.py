"""Finite-mask carry and rational Cartier compilers."""

from __future__ import annotations

from komorebi_math.generating import (
    compile_cartier_linear_representation,
    compile_finite_mask_carry_plan,
    solve_polynomial_coboundary,
)

stern = compile_finite_mask_carry_plan((1, 1, 1), 2)
stern.certificate(500).require()

cartier = compile_cartier_linear_representation((1, 1), (1, -1), 5)
large_index_value = cartier.evaluate(10**18 + 12345)

coboundary = solve_polynomial_coboundary((1, 2), (1, 0, 2), 2, 4)
assert coboundary.require() == (1, 2)

print(
    {
        "stern_shift_prefix": stern.sequence(20),
        "cartier_dimension": cartier.dimension,
        "large_index_value_mod_5": large_index_value,
        "coboundary": coboundary.coefficients,
    }
)
