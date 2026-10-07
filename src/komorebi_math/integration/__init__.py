"""General finite measure integration for the research API."""

from .adapters import (
    AtomicIntegratedDerivativeProvider,
    analysis_synthesis_integration_certificate,
    integrate_formal_series,
    integrate_polynomial_coordinates,
    polynomial_from_family_coordinates,
    polynomial_from_formal_series,
)
from .coordinates import (
    PolynomialIntegrationCovector,
    coordinate_fundamental_theorem,
    coordinate_transport_identity,
    polynomial_integration_covector,
)
from .identities import (
    IntegralIdentity,
    affine_interval_change_of_variables,
    all_fubini_orders,
    fubini_identity,
    fundamental_theorem,
    integration_by_parts,
    pushforward_identity,
)
from .measures import (
    AtomicMeasure,
    ExactAffineMap,
    IntervalMeasure,
    Measure,
    ProductMeasure,
    PushforwardMeasure,
    product_measure,
    pushforward_measure,
)
from .polynomial import ExactPolynomial
from .quadrature import (
    CompositeSimpson,
    DerivativeBound,
    ErrorGuarantee,
    IntegralResult,
    IntegrationError,
    integrate,
    integrate_derivative_provider,
    integrate_polynomial_oriented,
    polynomial_derivative_bound,
)
from .rules import ExactQuadratureRule, QuadratureExactness, QuadratureRule

__all__ = [
    "AtomicIntegratedDerivativeProvider",
    "AtomicMeasure",
    "CompositeSimpson",
    "DerivativeBound",
    "ErrorGuarantee",
    "ExactAffineMap",
    "ExactPolynomial",
    "ExactQuadratureRule",
    "IntegralIdentity",
    "IntegralResult",
    "IntegrationError",
    "IntervalMeasure",
    "Measure",
    "PolynomialIntegrationCovector",
    "ProductMeasure",
    "PushforwardMeasure",
    "QuadratureExactness",
    "QuadratureRule",
    "affine_interval_change_of_variables",
    "all_fubini_orders",
    "analysis_synthesis_integration_certificate",
    "coordinate_fundamental_theorem",
    "coordinate_transport_identity",
    "fubini_identity",
    "fundamental_theorem",
    "integrate",
    "integrate_derivative_provider",
    "integrate_formal_series",
    "integrate_polynomial_coordinates",
    "integrate_polynomial_oriented",
    "integration_by_parts",
    "polynomial_derivative_bound",
    "polynomial_from_family_coordinates",
    "polynomial_from_formal_series",
    "polynomial_integration_covector",
    "product_measure",
    "pushforward_identity",
    "pushforward_measure",
]
