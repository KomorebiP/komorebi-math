"""Prime-power kernels, Bell reductions, and Gaussian valuations."""

from __future__ import annotations

from math import comb

from komorebi_math.generating import (
    PrimePowerKernelCompiler,
    generalized_bell_digit_reduction,
    q_binomial_valuation_decomposition,
)

kernel = PrimePowerKernelCompiler(3, 5, 2)
entry = kernel.kernel(137, 52)
assert entry.residue == pow(comb(137, 52), 2, 3**5)

bell = generalized_bell_digit_reduction(137, 5, 9)
bell.require()

q_value = q_binomial_valuation_decomposition(48, 22, 2, 5)

print(
    {
        "kernel_residue": entry.residue,
        "kernel_valuation": entry.valuation,
        "bell_digit_sum": bell.digit_sum,
        "q_binomial_valuation": q_value.valuation,
    }
)
