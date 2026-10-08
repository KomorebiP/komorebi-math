"""Positive Gamma windows and normalized Meixner shape refinement.

These are scalar numerical evaluations of closed formulas, not interval
certificates. Each call uses its own mpmath precision context. The Fourier
convention is (2*pi)**(-1/2) * integral f(y)*exp(-i*omega*y) dy.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..exceptions import DimensionError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._validation import context, integer, real


def _log_cosh(ctx, value):
    # The expression also avoids overflow in implementations with finite exponents.
    absolute = abs(value)
    return absolute + ctx.log1p(ctx.exp(-2 * absolute)) - ctx.log(2)


def log_gamma_window(shape: Any, y: Any, *, dps: int = 50):
    """Return Gamma(shape)**(-1/2)*exp(shape*y/2-exp(y)/2), shape>0."""
    ctx = context(dps)
    s = real(ctx, shape, "shape", positive=True)
    point = real(ctx, y, "y")
    return ctx.exp((s * point - ctx.exp(point) - ctx.loggamma(s)) / 2)


def log_gamma_fourier(shape: Any, omega: Any, *, dps: int = 50):
    """Evaluate the unitary Fourier transform of ``log_gamma_window``."""
    ctx = context(dps)
    s = real(ctx, shape, "shape", positive=True)
    frequency = real(ctx, omega, "omega")
    z = s / 2 - ctx.j * frequency
    return ctx.exp(z * ctx.log(2) + ctx.loggamma(z) - (ctx.log(2 * ctx.pi) + ctx.loggamma(s)) / 2)


@documented(ProvenanceRecord(
    identifier="special-functions.cross-shape-gamma-sech",
    title="Cross-shape Gamma–sech ambiguity formula",
    status=ClaimStatus.PROVED,
    source="Special Functions Consolidation v0.1, shape_core.tex, Cross-shape Gamma–sech identity; Paper VI",
    assumptions=("alpha and gamma are positive", "h and eta are real"),
    limitations=("Numerical formula evaluation is not an interval enclosure.",),
    verification_suite="tests/test_special_functions_shape.py",
))
def gamma_sech_ambiguity(alpha: Any, gamma: Any, h: Any, eta: Any, *, dps: int = 50):
    """Return integral g_alpha(y+h/2)*g_gamma(y-h/2)*exp(i*eta*y) dy."""
    ctx = context(dps)
    a = real(ctx, alpha, "alpha", positive=True)
    g = real(ctx, gamma, "gamma", positive=True)
    shift = real(ctx, h, "h")
    frequency = real(ctx, eta, "eta")
    z = (a + g) / 2 + ctx.j * frequency
    return ctx.exp(
        (a - g) * shift / 4 + ctx.loggamma(z)
        - (ctx.loggamma(a) + ctx.loggamma(g)) / 2
        - z * _log_cosh(ctx, shift / 2)
    )


def shape_affinity(alpha: Any, gamma: Any, *, dps: int = 50):
    """Return the real positive zero-shift overlap of two unit Gamma windows."""
    ctx = context(dps)
    a = real(ctx, alpha, "alpha", positive=True)
    g = real(ctx, gamma, "gamma", positive=True)
    return ctx.exp(ctx.loggamma((a + g) / 2) - (ctx.loggamma(a) + ctx.loggamma(g)) / 2)


def meixner_density(shape: Any, omega: Any, *, dps: int = 50):
    """Evaluate rho_s(omega)=2**(s-1)*|Gamma(s/2+i*omega)|**2/(pi*Gamma(s))."""
    ctx = context(dps)
    s = real(ctx, shape, "shape", positive=True)
    frequency = real(ctx, omega, "omega")
    return ctx.exp((s - 1) * ctx.log(2) - ctx.log(ctx.pi) - ctx.loggamma(s)
                   + 2 * ctx.re(ctx.loggamma(s / 2 + ctx.j * frequency)))


def meixner_characteristic(shape: Any, h: Any, *, dps: int = 50):
    """Return sech(h/2)**shape, the characteristic function of rho_shape."""
    ctx = context(dps)
    s = real(ctx, shape, "shape", positive=True)
    shift = real(ctx, h, "h")
    return ctx.exp(-s * _log_cosh(ctx, shift / 2))


def beta_logit_density(alpha: Any, gamma: Any, h: Any, *, dps: int = 50):
    """Evaluate the density of log(U/(1-U)) for U~Beta(alpha,gamma)."""
    ctx = context(dps)
    a = real(ctx, alpha, "alpha", positive=True)
    g = real(ctx, gamma, "gamma", positive=True)
    shift = real(ctx, h, "h")
    return ctx.exp(ctx.loggamma(a + g) - ctx.loggamma(a) - ctx.loggamma(g)
                   - (a + g) * ctx.log(2) + (a - g) * shift / 2
                   - (a + g) * _log_cosh(ctx, shift / 2))


def meixner_pollaczek(shape: Any, degree: int, x: Any, *, dps: int = 50):
    """Evaluate p_(n,s) normalized in L2(rho_s), using its three-term recurrence.

    Its unnormalized generating function is
    (1-i*z)**(-s/2+i*x)*(1+i*z)**(-s/2-i*x).
    """
    ctx = context(dps)
    s = real(ctx, shape, "shape", positive=True)
    n = integer(degree, "degree")
    point = real(ctx, x, "x")
    previous = ctx.mpf(1)
    if n == 0:
        return previous
    current = 2 * point / ctx.sqrt(s)
    for k in range(1, n):
        following = (2 * point * current - ctx.sqrt(k * (k + s - 1)) * previous) / ctx.sqrt((k + 1) * (k + s))
        previous, current = current, following
    return current


def _refinement(ctx, shapes, degrees):
    total = sum(degrees)
    log_weight = ctx.loggamma(total + 1) - ctx.loggamma(sum(shapes) + total) + ctx.loggamma(sum(shapes))
    for s, n in zip(shapes, degrees):
        log_weight += ctx.loggamma(s + n) - ctx.loggamma(s) - ctx.loggamma(n + 1)
    return ctx.exp(log_weight / 2)


@documented(ProvenanceRecord(
    identifier="special-functions.dirichlet-multinomial-refinement",
    title="Normalized Gamma-shape polynomial refinement coefficient",
    status=ClaimStatus.PROVED,
    source="Special Functions Consolidation v0.1, shape_core.tex, Normalized two-child refinement; Paper VI",
    assumptions=("all shapes are positive", "all degrees are nonnegative integers"),
    limitations=("Coefficients are evaluated numerically at the requested precision.",),
    verification_suite="tests/test_special_functions_shape.py",
))
def shape_refinement_coefficient(shapes: Iterable[Any], degrees: Iterable[int], *, dps: int = 50):
    """Return sqrt(n!*prod((s_j)_(r_j)/r_j!)/(sum(s_j))_n), n=sum(r_j)."""
    ctx = context(dps)
    values = tuple(real(ctx, s, "shape", positive=True) for s in shapes)
    orders = tuple(integer(n, "degree") for n in degrees)
    if not values or len(values) != len(orders):
        raise DimensionError("shapes and degrees must have the same positive length")
    return _refinement(ctx, values, orders)


def beta_binomial_refinement(degree: int, alpha: Any, gamma: Any, *, dps: int = 50) -> tuple:
    """Return all n+1 positive two-child amplitudes; their squares sum to one."""
    ctx = context(dps)
    n = integer(degree, "degree")
    a = real(ctx, alpha, "alpha", positive=True)
    g = real(ctx, gamma, "gamma", positive=True)
    return tuple(_refinement(ctx, (a, g), (r, n - r)) for r in range(n + 1))


def shape_transfer_singular_value(shape: Any, increment: Any, degree: int, *, dps: int = 50):
    """Return sqrt((shape)_degree/(shape+increment)_degree), shape and increment>0."""
    ctx = context(dps)
    s = real(ctx, shape, "shape", positive=True)
    t = real(ctx, increment, "increment", positive=True)
    n = integer(degree, "degree")
    return ctx.exp((ctx.loggamma(s + n) - ctx.loggamma(s)
                    - ctx.loggamma(s + t + n) + ctx.loggamma(s + t)) / 2)
