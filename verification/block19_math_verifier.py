#!/usr/bin/env python3
"""Independent, pytest-free mathematical verifier for Komorebi Math Block 19.

The verifier exercises only package APIs.  It can be run from an installed
distribution or directly from a source checkout::

    python verification/block19_math_verifier.py
    python verification/block19_math_verifier.py --output report.json

The process exits with status 0 only when every check passes.  ``--output``
writes a machine-readable report without changing the human-readable console
summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable


def _import_from_source_checkout_if_needed() -> None:
    """Make a neighboring ``src`` tree importable when no wheel is installed."""

    try:
        __import__("komorebi_math")
    except ModuleNotFoundError:
        source = Path(__file__).resolve().parents[1] / "src"
        if source.is_dir():
            sys.path.insert(0, str(source))


_import_from_source_checkout_if_needed()

import numpy as np

import komorebi_math
from komorebi_math.certificates import Exactness
from komorebi_math.coordinates import (
    AnalysisSynthesisSystem,
    euler_polynomial_family,
    fibonacci_polynomial_family,
    legendre_polynomial_family,
    monomial_polynomial_family,
)
from komorebi_math.exceptions import ValidationError
from komorebi_math.integration import (
    AtomicIntegratedDerivativeProvider,
    AtomicMeasure,
    CompositeSimpson,
    ErrorGuarantee,
    ExactAffineMap,
    ExactPolynomial,
    ExactQuadratureRule,
    IntervalMeasure,
    ProductMeasure,
    PushforwardMeasure,
    affine_interval_change_of_variables,
    all_fubini_orders,
    analysis_synthesis_integration_certificate,
    coordinate_fundamental_theorem,
    coordinate_transport_identity,
    fundamental_theorem,
    integrate,
    integrate_polynomial_oriented,
    integration_by_parts,
    polynomial_derivative_bound,
    pushforward_identity,
)

F = Fraction


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Serializable outcome of one independent verification check."""

    name: str
    passed: bool
    detail: str
    error: str | None = None


Check = Callable[[], str]
CHECKS: list[tuple[str, Check]] = []


def check(name: str) -> Callable[[Check], Check]:
    """Register one check while keeping its implementation directly callable."""

    def register(function: Check) -> Check:
        CHECKS.append((name, function))
        return function

    return register


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_raises(
    exception_type: type[BaseException],
    function: Callable[[], Any],
    *,
    message_contains: str | None = None,
) -> BaseException:
    """Require a precise rejection rather than accepting any exception."""

    try:
        function()
    except exception_type as exc:
        if message_contains is not None:
            require(
                message_contains in str(exc),
                f"expected error containing {message_contains!r}, received {str(exc)!r}",
            )
        return exc
    except BaseException as exc:  # pragma: no cover - diagnostic failure path
        raise AssertionError(
            f"expected {exception_type.__name__}, received {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"expected {exception_type.__name__}, but no exception was raised")


@check("exact interval moments and polynomial contraction")
def verify_interval_moments() -> str:
    measure = IntervalMeasure(-1, 2, density=F(3, 2))
    observed = tuple(measure.moment((degree,)).require() for degree in range(5))
    expected = (F(9, 2), F(9, 4), F(9, 2), F(45, 8), F(99, 10))
    require(observed == expected, f"moment mismatch: {observed!r}")
    polynomial = ExactPolynomial.univariate((4, -2, 1))
    result = integrate(polynomial, measure)
    require(result.require() == 18 and result.exact, "exact moment contraction did not return 18")
    return "moments 0..4 match exactly; integral(4-2x+x^2)=18"


@check("exact affine pushforward and change of variables")
def verify_pushforward() -> str:
    base = IntervalMeasure(0, 1)
    mapping = ExactAffineMap(((2,),), (1,), name="y = 2x + 1")
    polynomial = ExactPolynomial.univariate((1, -2, 0, 0, 1))
    pushed = PushforwardMeasure(base, mapping)
    image = base.pushforward_affine(mapping)
    require(isinstance(image, IntervalMeasure), "affine image was not materialized as an interval")
    require(
        (image.lower, image.upper, image.density) == (F(1), F(3), F(1, 2)),
        "absolute-Jacobian density is incorrect",
    )
    values = (
        integrate(polynomial, pushed).require(),
        integrate(polynomial, image).require(),
        pushforward_identity(polynomial, base, mapping).require().left,
        affine_interval_change_of_variables(polynomial, base, mapping).require().left,
    )
    require(values == (F(106, 5),) * 4, f"pushforward values disagree: {values!r}")
    return "pushforward, pullback, and |Jacobian| density all give 106/5"


@check("noninjective atomic pushforward coalesces equal images")
def verify_noninjective_pushforward() -> str:
    base = AtomicMeasure(
        (-2, -1, 1, 2),
        (F(1, 6), F(1, 3), F(1, 4), F(1, 4)),
    )
    pushed = PushforwardMeasure(base, lambda x: x * x, target_dimension=1).materialize_atomic()
    observed = dict(zip(pushed.points, pushed.weights, strict=True))
    expected = {(F(1),): F(7, 12), (F(4),): F(5, 12)}
    require(observed == expected, f"coalesced atomic weights are wrong: {observed!r}")
    return "x -> x^2 coalesces four atoms into weights 7/12 and 5/12"


@check("exact Fubini identity in every variable order")
def verify_fubini() -> str:
    box = ProductMeasure((IntervalMeasure(0, 1), IntervalMeasure(1, 2)))
    polynomial = ExactPolynomial(2, {(2, 1): 1, (1, 2): 3})
    identities = all_fubini_orders(polynomial, box)
    require(len(identities) == 2, f"expected two integration orders, received {len(identities)}")
    values = tuple(identity.require().left for identity in identities)
    require(values == (F(4), F(4)), f"Fubini order values disagree: {values!r}")
    return "both iterated orders and the joint moment contraction equal 4"


@check("fundamental theorem of calculus")
def verify_ftc() -> str:
    primitive = ExactPolynomial.univariate((7, -5, F(3, 2), F(-2, 3), F(1, 4)))
    identity = fundamental_theorem(primitive.derivative(), IntervalMeasure(-1, 2)).require()
    require(identity.left == identity.right == F(-51, 4), "FTC endpoint identity mismatch")
    return "integral F' = F(2)-F(-1) = -51/4"


@check("integration by parts retains the boundary term")
def verify_integration_by_parts() -> str:
    left_factor = ExactPolynomial.univariate((1, 0, 1))
    right_factor = ExactPolynomial.univariate((0, -2, 0, 1))
    identity = integration_by_parts(
        left_factor,
        right_factor,
        IntervalMeasure(-1, 2),
    ).require()
    require(identity.left == identity.right == F(84, 5), "integration-by-parts mismatch")
    return "integral u v' and [uv]-integral u'v both equal 84/5"


@check("coordinate FTC and contragredient covariance")
def verify_coordinate_covariance() -> str:
    measure = IntervalMeasure(-1, 2)
    families = (
        monomial_polynomial_family(),
        euler_polynomial_family(),
        fibonacci_polynomial_family(),
        legendre_polynomial_family(),
    )
    for family in families:
        coordinate_fundamental_theorem(family, 7, measure).require()
    transported = coordinate_transport_identity(families[1], families[2], 7, measure).require()
    require(transported.left == transported.right, "integration covector did not transport contragrediently")
    return "ell D = ev_b-ev_a in four bases; Euler-to-Fibonacci covector transport agrees"


@check("exact Simpson degree boundary blocks a quartic claim")
def verify_simpson_degree_boundary() -> str:
    rule = ExactQuadratureRule(
        (-1, F(1, 2), 2),
        (F(1, 2), 2, F(1, 2)),
        name="exact Simpson functional",
    )
    measure = IntervalMeasure(-1, 2)
    degree = rule.degree_of_exactness(measure, max_degree=7)
    require(degree.exact and degree.degree == 3, f"certified degree was {degree.degree}")
    cubic = ExactPolynomial.univariate((-5, 3, -2, 1))
    require(rule.certify_polynomial_integral(cubic, measure).require() == F(-51, 4), "cubic failed")
    quartic = ExactPolynomial.univariate((0, 0, 0, 0, 1))
    quadrature = rule.integrate_polynomial(quartic).require()
    truth = integrate(quartic, measure).require()
    require((quadrature, truth) == (F(69, 8), F(33, 5)), "quartic boundary values changed")
    require_raises(
        ValidationError,
        lambda: rule.certify_polynomial_integral(quartic, measure),
        message_contains="not exactly certified",
    )
    return "degree=3; quartic 69/8 differs from 33/5 and certification is rejected"


@check("sampled-zero polynomial defeats three-node inference")
def verify_sampled_zero_adversary() -> str:
    adversary = ExactPolynomial.univariate(
        (0, 0, F(1, 4), F(-3, 2), F(13, 4), -3, 1)
    )
    rule = ExactQuadratureRule(
        (0, F(1, 2), 1),
        (F(1, 6), F(2, 3), F(1, 6)),
        name="three-node Simpson functional",
    )
    sampled = rule.integrate_polynomial(adversary)
    truth = integrate(adversary, IntervalMeasure(0, 1))
    require(sampled.require() == 0, "adversarial polynomial did not vanish at all rule nodes")
    require(sampled.error.guarantee is ErrorGuarantee.UNKNOWN, "uncertified rule claimed an error bound")
    require(truth.require() == F(1, 840), "true adversarial integral is not 1/840")
    require_raises(
        ValidationError,
        lambda: rule.certify_polynomial_integral(adversary, IntervalMeasure(0, 1)),
    )
    return "all three samples are zero, but the exact integral is 1/840"


@check("signed quadrature functional is not a positive measure")
def verify_signed_rule_boundary() -> str:
    rule = ExactQuadratureRule(
        (0, F(1, 2), 1, 2),
        (F(-1, 12), F(4, 3), F(-1, 3), F(1, 12)),
        name="signed functional",
    )
    require(not rule.positive, "signed rule was mislabeled positive")
    require_raises(
        ValidationError,
        rule.as_atomic_measure,
        message_contains="negative weights",
    )
    require(rule.certificate().valid, "signed functional itself should remain valid")
    return "negative weights remain a valid functional but cannot become AtomicMeasure"


@check("rigorous Simpson bound encloses the actual error")
def verify_rigorous_simpson_bound() -> str:
    polynomial = ExactPolynomial.univariate((0, 0, 0, 0, 1))
    measure = IntervalMeasure(0, 1)
    bound = polynomial_derivative_bound(polynomial, measure)
    result = integrate(polynomial, measure, provider=CompositeSimpson(8, bound))
    require(result.error.guarantee is ErrorGuarantee.RIGOROUS, "bound was not labeled rigorous")
    require(result.error.absolute_bound is not None, "rigorous result omitted its bound")
    actual = abs(result.value - F(1, 5))
    require(actual <= result.error.absolute_bound, "actual error exceeded rigorous bound")
    return f"exact actual error {actual} <= exact bound {result.error.absolute_bound}"


@check("oriented polynomial integral changes sign without a signed measure")
def verify_oriented_integral() -> str:
    constant = ExactPolynomial.univariate((2,))
    forward = integrate_polynomial_oriented(constant, -1, 1).require()
    reverse = integrate_polynomial_oriented(constant, 1, -1).require()
    require((forward, reverse) == (F(4), F(-4)), "oriented integral did not reverse sign")
    return "forward integral is 4 and reversed bounds give -4"


@check("joint atomic law is distinguished from product marginals")
def verify_joint_vs_product() -> str:
    joint = AtomicMeasure(((0, 0), (1, 1)), (F(1, 2), F(1, 2)))
    marginal = AtomicMeasure((0, 1), (F(1, 2), F(1, 2)))
    independent = ProductMeasure((marginal, marginal))
    product_xy = ExactPolynomial.monomial(1, (1, 1))
    joint_value = integrate(product_xy, joint).require()
    independent_value = integrate(product_xy, independent).require()
    require((joint_value, independent_value) == (F(1, 2), F(1, 4)), "dependence was erased")
    return "E_joint[XY]=1/2 while product marginals give 1/4"


@check("one-dimensional derivative bound cannot certify a tensor rule")
def verify_bound_scope_guard() -> str:
    quartic = ExactPolynomial.univariate((0, 0, 0, 0, 1))
    bound = polynomial_derivative_bound(quartic, IntervalMeasure(0, 1))
    square = ProductMeasure((IntervalMeasure(0, 1), IntervalMeasure(0, 1)))
    require_raises(
        ValidationError,
        lambda: integrate(
            lambda point: point[0] ** 4 + point[1],
            square,
            provider=CompositeSimpson(8, bound),
        ),
        message_contains="one-dimensional derivative bound",
    )
    return "tensor certification rejects an unrelated one-dimensional derivative bound"


@check("fixed-weight derivative providers commute with finite integration")
def verify_derivative_provider_adapter() -> str:
    class QuadraticProvider:
        source_dimension = 1
        target_dimension = 1
        exactness = Exactness.EXACT

        def __init__(self, scale: int) -> None:
            self.scale = scale

        def supports_order(self, order: int) -> bool:
            return order in (1, 2)

        def value(self, point: np.ndarray) -> np.ndarray:
            return np.asarray([self.scale * point[0] ** 2], dtype=object)

        def derivative(
            self,
            order: int,
            point: np.ndarray,
            directions: tuple[np.ndarray, ...],
        ) -> np.ndarray:
            if order == 1:
                return np.asarray(
                    [2 * self.scale * point[0] * directions[0][0]],
                    dtype=object,
                )
            return np.asarray(
                [2 * self.scale * directions[0][0] * directions[1][0]],
                dtype=object,
            )

    provider = AtomicIntegratedDerivativeProvider(
        (QuadraticProvider(1), QuadraticProvider(3)),
        (F(1, 4), F(3, 4)),
    )
    provider.certificate().require()
    point = np.asarray([2], dtype=object)
    direction = np.asarray([5], dtype=object)
    observed = tuple(provider.derivative(1, point, (direction,)))
    require(observed == (50,), f"integrated directional derivative was {observed!r}")
    return "weighted directional derivative equals 50 exactly"


@check("analysis and synthesis commute with atomic integration")
def verify_frame_adapter() -> str:
    system = AnalysisSynthesisSystem(
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        np.array([[0.5, -0.5, 0.5], [-0.25, 0.75, 0.25]]),
    )
    measure = AtomicMeasure((0, 1, 2), (F(1, 4), F(1, 2), F(1, 4)))
    values = (
        np.array([1.0, -2.0]),
        np.array([3.0, 4.0]),
        np.array([-1.0, 5.0]),
    )
    certificate = analysis_synthesis_integration_certificate(system, measure, values)
    certificate.require()
    maximum = max(float(value) for value in certificate.residuals.values())
    require(maximum <= 1e-10, f"frame commutation residual was {maximum}")
    return f"analysis/synthesis commutation residual <= {maximum:.3g}"


def run_checks(*, verbose: bool = True) -> dict[str, Any]:
    """Execute every registered check and return a JSON-compatible report."""

    outcomes: list[CheckResult] = []
    for name, function in CHECKS:
        try:
            detail = function()
        except BaseException as exc:  # keep running to expose all independent failures
            outcomes.append(
                CheckResult(
                    name=name,
                    passed=False,
                    detail="check failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            if verbose:
                print(f"FAIL  {name}")
                print(f"      {type(exc).__name__}: {exc}")
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    traceback.print_exc()
        else:
            outcomes.append(CheckResult(name=name, passed=True, detail=detail))
            if verbose:
                print(f"PASS  {name}")
                print(f"      {detail}")

    passed = sum(outcome.passed for outcome in outcomes)
    total = len(outcomes)
    report = {
        "block": 19,
        "package": "komorebi-math",
        "package_version": komorebi_math.__version__,
        "verifier": "verification/block19_math_verifier.py",
        "pytest_required": False,
        "passed": passed,
        "failed": total - passed,
        "total": total,
        "success": passed == total,
        "checks": [asdict(outcome) for outcome in outcomes],
    }
    if verbose:
        print()
        print(f"Block 19 independent verification: {passed}/{total} passed")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional path for the complete JSON verification report",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress individual check lines (the final summary is still printed)",
    )
    arguments = parser.parse_args(argv)
    report = run_checks(verbose=not arguments.quiet)
    if arguments.quiet:
        print(
            "Block 19 independent verification: "
            f"{report['passed']}/{report['total']} passed"
        )
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"JSON report: {arguments.output}")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
