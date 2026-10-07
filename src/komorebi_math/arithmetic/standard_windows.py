"""Finite field standard windows and congruential inverse coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb
from numbers import Integral

from .._number_theory import require_integer, require_prime
from ..analytic import congruential_inverse
from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .hurwitz import compressed_hurwitz_inverse, compressed_period


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValidationError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValidationError(f"{name} must be positive")
    return result


def rational_mod(value: Fraction | int, prime: int) -> int:
    """Reduce a rational with prime unit denominator modulo ``prime``."""

    p = require_prime(prime)
    q = Fraction(value)
    if q.denominator % p == 0:
        raise ValidationError("rational denominator is not a unit at the declared prime")
    return q.numerator * pow(q.denominator, -1, p) % p


@dataclass(frozen=True, slots=True)
class PeriodicGeneratingFunction:
    """A positive tail periodic ordinary generating function modulo a prime."""

    prime: int
    period: int
    constant: int
    numerator: tuple[int, ...]

    def coefficient(self, index: int) -> int:
        """Return one coefficient of the represented periodic series."""

        n = require_integer(index, name="index")
        if n < 0:
            raise ValidationError("index must be nonnegative")
        if n == 0:
            return self.constant % self.prime
        residue = (n - 1) % self.period
        return self.numerator[residue] % self.prime

    def first(self, count: int) -> tuple[int, ...]:
        """Return the first ``count`` coefficients."""

        total = require_integer(count, name="count")
        if total < 0:
            raise ValidationError("count must be nonnegative")
        return tuple(self.coefficient(index) for index in range(total))


@documented(
    ProvenanceRecord(
        identifier="PSC-IX-T09",
        title="Uniform standard window period",
        status=ClaimStatus.PROVED,
        source="Paper IX",
        assumptions=("odd prime", "compression divides prime minus one", "standard address"),
        limitations=("period is proved modulo the residue prime",),
        verification_suite="arithmetic.standard_windows",
    )
)
@dataclass(frozen=True, slots=True)
class StandardWindow:
    """A congruential inverse window with exact rational and residual coordinates."""

    prime: int
    compression: int

    def __post_init__(self) -> None:
        prime = require_prime(self.prime)
        compression = _positive_integer(self.compression, name="compression")
        object.__setattr__(self, "prime", prime)
        object.__setattr__(self, "compression", compression)

    @property
    def order(self) -> int:
        """Return the congruential order ``compression * prime``."""

        return self.compression * self.prime

    @property
    def has_uniform_period(self) -> bool:
        """Return whether the proved subgroup period applies."""

        return self.prime % 2 == 1 and (self.prime - 1) % self.compression == 0

    @property
    def period(self) -> int:
        """Return the proved positive tail period modulo the prime."""

        if not self.has_uniform_period:
            raise ValidationError("a uniform period requires an odd prime and compression dividing prime - 1")
        return compressed_period(self.prime, self.compression)

    def reduced_inverse(self, degree: int) -> tuple[int, ...]:
        """Return the reduced standard inverse modulo the prime."""

        nmax = require_integer(degree, name="degree")
        if nmax < 0:
            raise ValidationError("degree must be nonnegative")
        values = [1]
        for n in range(1, nmax + 1):
            values.append(-sum(comb(self.compression * n, self.compression * q) * values[q] for q in range(n)) % self.prime)
        return tuple(values)

    def exact_coefficients(self, address: int, degree: int) -> tuple[Fraction, ...]:
        """Return exact congruential inverse coefficients for one address."""

        j = require_integer(address, name="address")
        if j < 0 or j >= self.prime:
            raise ValidationError("address must satisfy 0 <= address < prime")
        nmax = require_integer(degree, name="degree")
        if nmax < 0:
            raise ValidationError("degree must be nonnegative")
        return tuple(Fraction(value) for value in congruential_inverse(self.order, j, nmax))

    def residues(self, address: int, degree: int) -> tuple[int, ...]:
        """Return one exact address reduced modulo the prime."""

        return tuple(rational_mod(value, self.prime) for value in self.exact_coefficients(address, degree))

    def rational_generating_function(self) -> PeriodicGeneratingFunction:
        """Return the residual rational generating function of the standard window."""

        h = self.period
        values = self.reduced_inverse(h)
        return PeriodicGeneratingFunction(self.prime, h, values[0], values[1 : h + 1])

    def minimum_period(self) -> int:
        """Return the minimum positive period of the computed residual tail."""

        h = self.period
        block = self.reduced_inverse(h)[1:]
        for divisor in range(1, h + 1):
            if h % divisor == 0 and all(block[index] == block[index % divisor] for index in range(h)):
                return divisor
        return h

    def certificate(self, address: int, periods: int = 4) -> ComputationResult[tuple[int, ...]]:
        """Validate address independence and periodicity through several periods."""

        repetitions = require_integer(periods, name="periods")
        if repetitions < 1:
            raise ValidationError("periods must be positive")
        h = self.period
        degree = repetitions * h + 2
        reduced = self.reduced_inverse(degree)
        address_values = self.residues(address, degree)
        address_ok = address_values == reduced
        periodic = all(reduced[n + h] == reduced[n] for n in range(1, degree - h + 1))
        recurrence = True
        for n in range(degree + 1):
            total = sum(
                comb(self.compression * n, self.compression * q) * reduced[q]
                for q in range(n + 1)
            ) % self.prime
            recurrence = recurrence and total == (1 if n == 0 else 0)
        valid = address_ok and periodic and recurrence
        return ComputationResult(
            reduced,
            Certificate(
                valid=valid,
                exactness=Exactness.EXACT,
                assumptions_checked=(
                    "prime field arithmetic",
                    "standard address",
                    "compression divides prime minus one",
                ),
                diagnostics={
                    "address": require_integer(address, name="address"),
                    "period": h,
                    "minimum_period": self.minimum_period(),
                    "address_independence": address_ok,
                    "periodic": periodic,
                    "inverse_recurrence": recurrence,
                },
                theorem_id="PSC-IX-T09",
                message="standard window recurrence, address independence, and residual period agree",
            ),
        )


def standard_window_inverse(prime: int, compression: int, degree: int) -> tuple[int, ...]:
    """Return the reduced standard window inverse modulo ``prime``."""

    return StandardWindow(prime, compression).reduced_inverse(degree)


def standard_window_period(prime: int, compression: int) -> int:
    """Return the uniform standard window period."""

    return StandardWindow(prime, compression).period


def subgroup_standard_window_inverse(prime: int, compression: int, degree: int) -> tuple[int, ...]:
    """Return the same inverse through the finite Hurwitz subgroup construction."""

    return compressed_hurwitz_inverse(prime, compression, degree)


__all__ = [
    "PeriodicGeneratingFunction",
    "StandardWindow",
    "rational_mod",
    "standard_window_inverse",
    "standard_window_period",
    "subgroup_standard_window_inverse",
]
