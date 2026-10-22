"""Exact finite measure and integration example."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from typing import Any

from komorebi_math.coordinates import (
    connection_transform,
    euler_polynomial_family,
    fibonacci_polynomial_family,
    monomial_polynomial_family,
)
from komorebi_math.exceptions import ValidationError
from komorebi_math.integration import (
    CompositeSimpson,
    ExactAffineMap,
    ExactPolynomial,
    ExactQuadratureRule,
    IntervalMeasure,
    PushforwardMeasure,
    affine_interval_change_of_variables,
    coordinate_transport_identity,
    fundamental_theorem,
    integrate,
    polynomial_derivative_bound,
    polynomial_integration_covector,
    pushforward_identity,
)

F = Fraction


def _text(value: Any) -> Any:
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, tuple):
        return [_text(entry) for entry in value]
    if isinstance(value, list):
        return [_text(entry) for entry in value]
    if isinstance(value, dict):
        return {key: _text(entry) for key, entry in value.items()}
    return value


def run() -> dict[str, Any]:
    base = IntervalMeasure(0, 1, name="Lebesgue measure on x in [0,1]")
    chart = ExactAffineMap(((2,),), (1,), name="y = 2x + 1")
    pushed = PushforwardMeasure(base, chart, name="chart pushforward")
    image = base.pushforward_affine(chart)
    if not isinstance(image, IntervalMeasure):
        raise AssertionError("nonzero affine chart did not produce an interval measure")

    integrand = ExactPolynomial.univariate((1, -2, 0, 0, 1))
    pulled = chart.pullback(integrand)
    pushforward_result = integrate(integrand, pushed)
    density_result = integrate(integrand, image)
    pullback_result = integrate(pulled, base)
    expected = F(106, 5)
    assert pushforward_result.require() == density_result.require() == pullback_result.require() == expected

    pushforward_identity(integrand, base, chart).require()
    affine_interval_change_of_variables(integrand, base, chart).require()
    fundamental_theorem(integrand, image).require()

    monomial = monomial_polynomial_family()
    families = (monomial, euler_polynomial_family(), fibonacci_polynomial_family())
    monomial_coefficients = integrand.to_univariate_coefficients()
    coordinate_rows: dict[str, Any] = {}
    for family in families:
        coordinates = connection_transform(monomial, family, 4).apply(monomial_coefficients)
        covector = polynomial_integration_covector(family, 4, image)
        value = covector.apply(coordinates)
        assert value == expected
        coordinate_rows[family.name] = {
            "coordinates": coordinates,
            "integration_covector": covector.weights,
            "pairing": value,
        }
    coordinate_transport_identity(
        euler_polynomial_family(),
        fibonacci_polynomial_family(),
        4,
        image,
    ).require()

    derivative_bound = polynomial_derivative_bound(integrand, image)
    numerical = integrate(
        integrand,
        image,
        provider=CompositeSimpson(16, derivative_bound),
    )
    if not isinstance(numerical.value, Fraction):
        raise AssertionError("the certified Simpson path lost exact rational arithmetic")
    numerical_error = abs(numerical.value - expected)
    assert numerical.error.absolute_bound is not None
    assert numerical_error <= numerical.error.absolute_bound

    adversary = ExactPolynomial.univariate(
        (0, 0, F(1, 4), F(-3, 2), F(13, 4), -3, 1)
    )
    simpson_nodes = ExactQuadratureRule(
        (0, F(1, 2), 1),
        (F(1, 6), F(2, 3), F(1, 6)),
        name="three-node exact Simpson functional",
    )
    sampled_value = simpson_nodes.integrate_polynomial(adversary)
    true_value = integrate(adversary, base)
    false_certificate_blocked = False
    try:
        simpson_nodes.certify_polynomial_integral(adversary, base)
    except ValidationError:
        false_certificate_blocked = True
    assert sampled_value.require() == 0
    assert true_value.require() == F(1, 840)
    assert false_certificate_blocked

    return {
        "example_scope": "exact finite integration",
        "base_measure": {"interval": (0, 1), "density": 1},
        "chart": {"formula": "y = 2x + 1", "absolute_jacobian": 2},
        "pushed_measure": {
            "interval": (image.lower, image.upper),
            "density": image.density,
        },
        "integrand": {
            "formula": "f(y) = y^4 - 2y + 1",
            "coefficients": integrand.to_univariate_coefficients(),
        },
        "pullback": {
            "formula": "f(2x+1) = 16x^4 + 32x^3 + 24x^2 + 4x",
            "coefficients": pulled.to_univariate_coefficients(),
        },
        "exact_integral": expected,
        "coordinate_pairings": coordinate_rows,
        "certified_simpson": {
            "value": numerical.value,
            "absolute_error": numerical_error,
            "rigorous_bound": numerical.error.absolute_bound,
            "guarantee": numerical.error.guarantee.value,
        },
        "adversarial_guard": {
            "sampled_quadrature": sampled_value.value,
            "true_integral": true_value.value,
            "false_exactness_certificate_blocked": false_certificate_blocked,
        },
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()
    report = _text(run())
    if arguments.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    print("Exact finite integration example")
    print("  chart: y = 2x + 1, x in [0,1]")
    print("  pushed measure: (1/2) dy on [1,3]")
    print("  integrand: f(y) = y^4 - 2y + 1")
    print("  pullback: f(2x+1) = 16x^4 + 32x^3 + 24x^2 + 4x")
    print(f"  exact integral: {report['exact_integral']}")
    for family, row in report["coordinate_pairings"].items():
        print(f"  {family} pairing: {row['pairing']}")
    numerical = report["certified_simpson"]
    numerical_value = float(Fraction(numerical["value"]))
    numerical_error = float(Fraction(numerical["absolute_error"]))
    rigorous_bound = float(Fraction(numerical["rigorous_bound"]))
    print(
        "  Simpson: "
        f"{numerical_value:.15f}, error {numerical_error:.3e}, "
        f"rigorous bound {rigorous_bound:.3e}"
    )
    guard = report["adversarial_guard"]
    print(
        "  node adversary: sampled "
        f"{guard['sampled_quadrature']}, true {guard['true_integral']}, "
        f"false certificate blocked={guard['false_exactness_certificate_blocked']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
