"""Worked exact cancellation and primitive-content example from finite shells."""

from __future__ import annotations

import json

from komorebi_math.arithmetic.lehmer_shells import (
    auxiliary_shell_polynomial,
    enumerate_five_core_shell,
    primitive_residuals_from_tau,
    tau_from_primitive_residuals,
    tau_from_q_product,
)


def main() -> None:
    tau = tau_from_q_product(128)
    primitive = primitive_residuals_from_tau(tau)
    recovered = tau_from_primitive_residuals(primitive)
    if recovered != tau:
        raise RuntimeError("finite primitive transform failed its exact inverse check")
    shell = enumerate_five_core_shell(121)
    polynomial = auxiliary_shell_polynomial(shell)
    if shell.weighted_sum != tau[121] or shell.primitive_residual != primitive[121]:
        raise RuntimeError("complete shell and independent q-product values disagree")
    prime_shell = enumerate_five_core_shell(2411)
    units = prime_shell.prime_unit_certificate()
    if not units.valid:
        raise RuntimeError("source index-prime unit condition failed")
    result = {
        "arithmetic": "exact Python integers",
        "q_product_degree": 128,
        "shell_q_product_comparison_index": 121,
        "square_transform_inverse_checked": recovered == tau,
        "shell_index": shell.index,
        "packet_count": len(shell.packets),
        "positive_packets": sum(packet.weight > 0 for packet in shell.packets),
        "negative_packets": sum(packet.weight < 0 for packet in shell.packets),
        "tau": shell.weighted_sum,
        "primitive_residual": shell.primitive_residual,
        "content_masses": shell.content_masses,
        "P_at_5": polynomial.evaluate(5),
        "P_at_0": polynomial.evaluate(0),
        "five_adic_content": polynomial.five_adic_content,
        "termwise_unit_index": prime_shell.index,
        "termwise_unit_packet_count": len(prime_shell.packets),
        "all_index_prime_weights_are_units": units.valid,
        "unit_shell_sum_is_divisible_by_index": prime_shell.weighted_sum % prime_shell.index == 0,
        "universal_nonvanishing_established": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
