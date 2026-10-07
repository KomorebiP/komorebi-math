"""Odd and dyadic Euler coordinate systems with exact local certificates."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb
from numbers import Integral

from .._number_theory import require_prime
from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .valuation import binomial_valuation, digit_sum, padic_valuation


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValidationError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValidationError(f"{name} must be nonnegative")
    return result


def bernoulli_numbers(degree: int) -> tuple[Fraction, ...]:
    """Return Bernoulli numbers through ``degree`` with ``B_1 = -1/2``."""

    nmax = _nonnegative_integer(degree, name="degree")
    values: list[Fraction] = [Fraction(1)]
    for n in range(1, nmax + 1):
        total = sum(Fraction(comb(n + 1, k)) * values[k] for k in range(n))
        values.append(-total / Fraction(n + 1))
    return tuple(values)


def signed_euler_numbers(degree: int) -> tuple[int, ...]:
    """Return coefficients of ``sech(t)`` through ``degree``."""

    nmax = _nonnegative_integer(degree, name="degree")
    values = [0] * (nmax + 1)
    values[0] = 1
    for n in range(2, nmax + 1, 2):
        values[n] = -sum(comb(n, k) * values[k] for k in range(0, n, 2))
    return tuple(values)


def signed_euler_number(index: int) -> int:
    """Return one signed Euler number from the ``sech`` convention."""

    n = _nonnegative_integer(index, name="index")
    return signed_euler_numbers(n)[n]


def euler_zigzag_numbers(degree: int) -> tuple[int, ...]:
    """Return coefficients of ``sec(t) + tan(t)`` through ``degree``."""

    nmax = _nonnegative_integer(degree, name="degree")
    values = [0] * (nmax + 1)
    values[0] = 1
    for n in range(nmax):
        square_coefficient = sum(comb(n, k) * values[k] * values[n - k] for k in range(n + 1))
        values[n + 1] = (square_coefficient + (1 if n == 0 else 0)) // 2
    return tuple(values)


def euler_zigzag_number(index: int) -> int:
    """Return one Euler zigzag number."""

    n = _nonnegative_integer(index, name="index")
    return euler_zigzag_numbers(n)[n]


def odd_power_sum(modulus: int, degree: int) -> int:
    """Return ``sum (-1)^a (2a+1)^degree`` over one odd modulus window."""

    q = _nonnegative_integer(modulus, name="modulus")
    n = _nonnegative_integer(degree, name="degree")
    if q < 1 or q % 2 == 0:
        raise ValidationError("modulus must be a positive odd integer")
    return sum((-1) ** a * (2 * a + 1) ** n for a in range(q))


def dyadic_source(index: int) -> Fraction:
    """Return the dyadic Euler source coefficient ``R_index``."""

    n = _nonnegative_integer(index, name="index")
    if n % 2:
        return Fraction(0)
    return Fraction(3 ** (n + 2) - 1, 4 * (n + 1) * (n + 2))


def dyadic_beta(index: int) -> Fraction:
    """Return the coefficient of ``z**(2*index)`` in ``z**2 / sinh(z)**2``."""

    r = _nonnegative_integer(index, name="index")
    if r == 0:
        return Fraction(1)
    bernoulli = bernoulli_numbers(2 * r)[2 * r]
    from math import factorial

    return -Fraction(2 ** (2 * r) * (2 * r - 1), factorial(2 * r)) * bernoulli


def dyadic_kernel(index: int) -> Fraction:
    """Return the normalized dyadic reconstruction kernel ``kappa[2,index]``."""

    r = _nonnegative_integer(index, name="index")
    if r == 0:
        return Fraction(1)
    return -Fraction(2 ** (4 * r) * (2 * r - 1)) * bernoulli_numbers(2 * r)[2 * r]


def odd_kernel(prime: int, index: int) -> Fraction:
    """Return the normalized odd prime Euler kernel."""

    p = require_prime(prime)
    if p == 2:
        raise ValidationError("odd_kernel requires an odd prime")
    r = _nonnegative_integer(index, name="index")
    return Fraction((-1) ** r * euler_zigzag_number(2 * r + 1) * p ** (2 * r))


@dataclass(frozen=True, slots=True)
class EulerTruncation:
    """A finite local Euler reconstruction with an exact congruence certificate."""

    prime: int
    index: int
    depth: int
    value: Fraction
    exact_value: int
    modulus: int | None
    error: Fraction
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="PSC-IX-T19",
        title="Unified local Euler transfer",
        status=ClaimStatus.PROVED,
        source="Paper IX",
        assumptions=("prime local chart", "even Euler index"),
        limitations=("odd and dyadic integral charts remain distinct",),
        verification_suite="arithmetic.euler",
    )
)
@dataclass(frozen=True, slots=True)
class EulerLocalCoordinates:
    """Exact odd or dyadic local coordinates for signed Euler numbers."""

    prime: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "prime", require_prime(self.prime))

    @property
    def is_dyadic(self) -> bool:
        """Return whether this is the dyadic chart."""

        return self.prime == 2

    def source(self, index: int) -> Fraction:
        """Return the local source coefficient ``A[prime,index]``."""

        n = _nonnegative_integer(index, name="index")
        if self.is_dyadic:
            return dyadic_source(n)
        return Fraction(odd_power_sum(self.prime, n))

    def kernel(self, index: int) -> Fraction:
        """Return the normalized reconstruction kernel coefficient."""

        return dyadic_kernel(index) if self.is_dyadic else odd_kernel(self.prime, index)

    def inverse_kernel(self, index: int) -> Fraction:
        """Return the normalized inverse local transfer coefficient."""

        r = _nonnegative_integer(index, name="index")
        if self.is_dyadic:
            return Fraction(2 ** (4 * r + 1), (2 * r + 1) * (2 * r + 2))
        if r == 0:
            return Fraction(1)
        return Fraction(2 ** (2 * r - 1) * self.prime ** (2 * r))

    def reconstruct(self, index: int, *, depth: int | None = None) -> Fraction:
        """Reconstruct one signed Euler number, optionally through finite depth."""

        n = _nonnegative_integer(index, name="index")
        if n % 2:
            raise ValidationError("Euler local reconstruction requires an even index")
        maximum = n // 2 if depth is None else min(_nonnegative_integer(depth, name="depth"), n // 2)
        return sum(
            Fraction(comb(n, 2 * r) * self.kernel(r)) * self.source(n - 2 * r)
            for r in range(maximum + 1)
        )

    def inverse_reconstruct_source(self, index: int) -> Fraction:
        """Recover the local source from signed Euler coefficients."""

        n = _nonnegative_integer(index, name="index")
        if n % 2:
            raise ValidationError("source reconstruction requires an even index")
        return sum(
            Fraction(comb(n, 2 * r))
            * self.inverse_kernel(r)
            * signed_euler_number(n - 2 * r)
            for r in range(n // 2 + 1)
        )

    def kernel_valuation(self, index: int) -> int | float:
        """Return the exact local valuation of one normalized kernel coefficient."""

        r = _nonnegative_integer(index, name="index")
        return padic_valuation(self.kernel(r), self.prime)

    def row_weight(self, index: int, transfer_index: int) -> int | float:
        """Return the exact valuation of one reconstruction row term."""

        n = _nonnegative_integer(index, name="index")
        r = _nonnegative_integer(transfer_index, name="transfer_index")
        if n % 2 or 2 * r > n:
            raise ValidationError("row indices must satisfy an even index and 2 * transfer_index <= index")
        term = Fraction(comb(n, 2 * r) * self.kernel(r)) * self.source(n - 2 * r)
        return padic_valuation(term, self.prime)

    def formula_row_weight(self, index: int, transfer_index: int) -> int | float:
        """Return the separated valuation formula for one row term."""

        n = _nonnegative_integer(index, name="index")
        r = _nonnegative_integer(transfer_index, name="transfer_index")
        if n % 2 or 2 * r > n:
            raise ValidationError("row indices must satisfy an even index and 2 * transfer_index <= index")
        if self.is_dyadic and r > 0:
            return 4 * r - 1 + binomial_valuation(n, 2 * r, 2)
        return (
            padic_valuation(self.kernel(r), self.prime)
            + binomial_valuation(n, 2 * r, self.prime)
            + padic_valuation(self.source(n - 2 * r), self.prime)
        )

    def sharp_modulus(self, depth: int) -> int:
        """Return the proved uniform modulus after truncation at ``depth``."""

        level = _nonnegative_integer(depth, name="depth")
        if self.is_dyadic:
            return 2 ** (4 * level + 3)
        return self.prime ** (2 * level + 2)

    def truncate(self, index: int, depth: int) -> EulerTruncation:
        """Return a finite reconstruction and its exact local congruence certificate."""

        n = _nonnegative_integer(index, name="index")
        level = _nonnegative_integer(depth, name="depth")
        if n % 2:
            raise ValidationError("Euler local reconstruction requires an even index")
        approximate = self.reconstruct(n, depth=level)
        exact = signed_euler_number(n)
        error = Fraction(exact) - approximate
        complete = level >= n // 2
        modulus = None if complete else self.sharp_modulus(level)
        valid = error == 0 if complete else padic_valuation(error, self.prime) >= padic_valuation(modulus, self.prime)
        return EulerTruncation(
            prime=self.prime,
            index=n,
            depth=level,
            value=approximate,
            exact_value=exact,
            modulus=modulus,
            error=error,
            certificate=Certificate(
                valid=valid,
                exactness=Exactness.EXACT,
                assumptions_checked=("prime local chart", "even Euler index", "finite exact arithmetic"),
                diagnostics={
                    "complete": complete,
                    "modulus": modulus,
                    "error_valuation": padic_valuation(error, self.prime),
                    "sharp_uniform_bound": self.is_dyadic,
                },
                theorem_id="PSC-IX-T19",
                message="local Euler reconstruction satisfies the declared exact or congruential identity",
            ),
        )

    def certificate(self, maximum_index: int = 20) -> ComputationResult[tuple[Fraction, ...]]:
        """Validate the exact coordinate pair through an even index bound."""

        nmax = _nonnegative_integer(maximum_index, name="maximum_index")
        reconstructions: list[Fraction] = []
        valid = True
        inverse_valid = True
        weight_valid = True
        for n in range(0, nmax + 1, 2):
            reconstructed = self.reconstruct(n)
            reconstructions.append(reconstructed)
            valid = valid and reconstructed == signed_euler_number(n)
            inverse_valid = inverse_valid and self.inverse_reconstruct_source(n) == self.source(n)
            for r in range(n // 2 + 1):
                weight_valid = weight_valid and self.row_weight(n, r) == self.formula_row_weight(n, r)
        return ComputationResult(
            tuple(reconstructions),
            Certificate(
                valid=valid and inverse_valid and weight_valid,
                exactness=Exactness.EXACT,
                assumptions_checked=("prime local chart", "finite exact coefficient range"),
                diagnostics={
                    "forward_reconstruction": valid,
                    "inverse_reconstruction": inverse_valid,
                    "row_weight_formula": weight_valid,
                    "maximum_index": nmax,
                },
                theorem_id="PSC-IX-T19",
                message="forward, inverse, and valuation formulas agree through the requested range",
            ),
        )


def dyadic_kernel_valuation(index: int) -> int:
    """Return the exact valuation ``4 * index - 1`` for positive dyadic kernel depth."""

    r = _nonnegative_integer(index, name="index")
    return 0 if r == 0 else 4 * r - 1


def dyadic_row_weight(index: int, transfer_index: int) -> int:
    """Return the exact dyadic row weight formula."""

    n = _nonnegative_integer(index, name="index")
    r = _nonnegative_integer(transfer_index, name="transfer_index")
    if n % 2 or 2 * r > n:
        raise ValidationError("row indices must satisfy an even index and 2 * transfer_index <= index")
    if r == 0:
        return int(padic_valuation(dyadic_source(n), 2))
    return 4 * r - 1 + digit_sum(r, 2) + digit_sum(n - 2 * r, 2) - digit_sum(n, 2)


__all__ = [
    "EulerLocalCoordinates",
    "EulerTruncation",
    "bernoulli_numbers",
    "dyadic_beta",
    "dyadic_kernel",
    "dyadic_kernel_valuation",
    "dyadic_row_weight",
    "dyadic_source",
    "euler_zigzag_number",
    "euler_zigzag_numbers",
    "odd_kernel",
    "odd_power_sum",
    "signed_euler_number",
    "signed_euler_numbers",
]
