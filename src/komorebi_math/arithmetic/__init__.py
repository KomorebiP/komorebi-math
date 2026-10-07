"""Exact local arithmetic packets, Euler coordinates, Mahler transfer, and recurrences."""

from .._planned import development_status
from ..registry import FeatureSpec
from .domains import ModularRing, PrimeField, TruncatedPadic
from .euler import (
    EulerLocalCoordinates,
    EulerTruncation,
    bernoulli_numbers,
    dyadic_beta,
    dyadic_kernel,
    dyadic_kernel_valuation,
    dyadic_row_weight,
    dyadic_source,
    euler_zigzag_number,
    euler_zigzag_numbers,
    odd_kernel,
    odd_power_sum,
    signed_euler_number,
    signed_euler_numbers,
)
from .hurwitz import (
    HurwitzSeries,
    compressed_hurwitz_inverse,
    compressed_period,
    hurwitz_inverse,
    hurwitz_product,
    subgroup_roots,
    subgroup_selector,
)
from .local_modes import (
    DifferenceCongruence,
    certify_difference_congruence,
    finite_difference_value,
    local_unit_jump,
    simultaneous_unit_difference,
    unit_mode_for_precision,
)
from .mahler import (
    MahlerSeries,
    UnitModeMahler,
    affine_kernel_composition_certificate,
    affine_kernel_matrix,
    affine_mahler_kernel,
    difference_table,
    evaluate_matrix_mahler,
    generalized_binomial,
    mahler_coefficients,
    matrix_mahler_coefficients,
    prime_power_kernel_certificate,
    prime_power_kernel_residue,
)
from .polyphase import (
    ArithmeticPolyphase,
    mahler_polyphase,
    mixed_radix_address,
    mixed_radix_digits,
    polyphase_analyze,
    polyphase_synthesize,
)
from .recurrence import (
    LinearRecurrence,
    OrbitPeriod,
    RecurrencePacket,
    determinant_integer,
    fibonacci_recurrence,
    lucas_recurrence,
    matrix_power_exact,
    matrix_power_mod,
    second_order_jump,
    second_order_trace,
)
from .standard_windows import (
    PeriodicGeneratingFunction,
    StandardWindow,
    rational_mod,
    standard_window_inverse,
    standard_window_period,
    subgroup_standard_window_inverse,
)
from .valuation import (
    binomial_valuation,
    binomial_valuation_from_digits,
    carmichael_exponent_prime_power,
    digit_sum,
    exact_binomial,
    factorial_valuation,
    falling_factorial,
    padic_valuation,
    unit_difference_value,
    unit_difference_valuation,
    verify_binomial_digit_formula,
)

FEATURE_KEY = "arithmetic"


def status() -> FeatureSpec:
    """Return the registered implementation status for this domain."""

    return development_status(FEATURE_KEY)


def require() -> None:
    """Confirm that this package block is implemented."""

    return None


__all__ = [
    "FEATURE_KEY",
    "require",
    "status",
    "ArithmeticPolyphase",
    "DifferenceCongruence",
    "EulerLocalCoordinates",
    "EulerTruncation",
    "HurwitzSeries",
    "LinearRecurrence",
    "MahlerSeries",
    "ModularRing",
    "OrbitPeriod",
    "PeriodicGeneratingFunction",
    "PrimeField",
    "RecurrencePacket",
    "StandardWindow",
    "TruncatedPadic",
    "UnitModeMahler",
    "affine_kernel_composition_certificate",
    "affine_kernel_matrix",
    "affine_mahler_kernel",
    "bernoulli_numbers",
    "binomial_valuation",
    "binomial_valuation_from_digits",
    "carmichael_exponent_prime_power",
    "certify_difference_congruence",
    "compressed_hurwitz_inverse",
    "compressed_period",
    "determinant_integer",
    "difference_table",
    "digit_sum",
    "dyadic_beta",
    "dyadic_kernel",
    "dyadic_kernel_valuation",
    "dyadic_row_weight",
    "dyadic_source",
    "euler_zigzag_number",
    "euler_zigzag_numbers",
    "evaluate_matrix_mahler",
    "exact_binomial",
    "factorial_valuation",
    "falling_factorial",
    "fibonacci_recurrence",
    "finite_difference_value",
    "generalized_binomial",
    "hurwitz_inverse",
    "hurwitz_product",
    "local_unit_jump",
    "lucas_recurrence",
    "mahler_coefficients",
    "mahler_polyphase",
    "matrix_mahler_coefficients",
    "matrix_power_exact",
    "matrix_power_mod",
    "mixed_radix_address",
    "mixed_radix_digits",
    "odd_kernel",
    "odd_power_sum",
    "padic_valuation",
    "polyphase_analyze",
    "polyphase_synthesize",
    "prime_power_kernel_certificate",
    "prime_power_kernel_residue",
    "rational_mod",
    "second_order_jump",
    "second_order_trace",
    "signed_euler_number",
    "signed_euler_numbers",
    "simultaneous_unit_difference",
    "standard_window_inverse",
    "standard_window_period",
    "subgroup_roots",
    "subgroup_selector",
    "subgroup_standard_window_inverse",
    "unit_difference_value",
    "unit_difference_valuation",
    "unit_mode_for_precision",
    "verify_binomial_digit_formula",
]
