"""Cyclic analytic functions, reciprocal families, and zero diagnostics."""

from .._planned import development_status
from ..registry import FeatureSpec
from .appell import (
    appell_coefficients,
    appell_polynomial,
    appell_polynomial_symbolic,
    congruential_inverse,
    congruential_reciprocal_coefficients,
    theta_polynomial,
    theta_polynomial_symbolic,
)
from .cyclic import (
    cyclic_packets,
    cyclic_product_packets,
    cyclic_project_coefficients,
    cyclic_project_value,
    cyclic_projection_certificate,
    cyclic_reconstruct,
)
from .reciprocal import (
    logarithmic_coefficients,
    reciprocal_coefficients,
    reciprocal_polynomial,
    reciprocal_semigroup_certificate,
    reciprocal_value,
    transfer_tangent,
    transfer_tangent_coefficients,
)
from .sections import (
    CyclicSectionFamily,
    cyclic_addition,
    denominator,
    denominator_derivative,
    denominator_root_sum,
    section_certificate,
    signed_section,
    signed_sections,
    signed_shift_matrix,
    transfer_section,
    transfer_sections,
    transfer_shift_matrix,
)
from .zeros import (
    CyclicDenominator,
    ZeroStringResult,
    asymptotic_positive_zero,
    hadamard_partial_product,
    positive_zero_string,
    reciprocal_coefficient_from_zeros,
    reduced_zero_zeta,
    zero_count_asymptotic,
)

FEATURE_KEY = "analytic"


def status() -> FeatureSpec:
    """Return the registered implementation status for this domain."""

    return development_status(FEATURE_KEY)


def require() -> None:
    """Confirm that this package block is implemented."""

    return None


__all__ = [
    "FEATURE_KEY",
    "CyclicDenominator",
    "CyclicSectionFamily",
    "ZeroStringResult",
    "appell_coefficients",
    "appell_polynomial",
    "appell_polynomial_symbolic",
    "asymptotic_positive_zero",
    "congruential_inverse",
    "congruential_reciprocal_coefficients",
    "cyclic_addition",
    "cyclic_packets",
    "cyclic_product_packets",
    "cyclic_project_coefficients",
    "cyclic_project_value",
    "cyclic_projection_certificate",
    "cyclic_reconstruct",
    "denominator",
    "denominator_derivative",
    "denominator_root_sum",
    "hadamard_partial_product",
    "logarithmic_coefficients",
    "positive_zero_string",
    "reciprocal_coefficient_from_zeros",
    "reciprocal_coefficients",
    "reciprocal_polynomial",
    "reciprocal_semigroup_certificate",
    "reciprocal_value",
    "reduced_zero_zeta",
    "section_certificate",
    "signed_section",
    "signed_sections",
    "signed_shift_matrix",
    "require",
    "status",
    "theta_polynomial",
    "theta_polynomial_symbolic",
    "transfer_section",
    "transfer_sections",
    "transfer_shift_matrix",
    "transfer_tangent",
    "transfer_tangent_coefficients",
    "zero_count_asymptotic",
]
