"""Exact truncated formal-series operations in a custom normalized basis."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

from .basis import WeightedMonomialBasis
from .domains import _native_fraction


def _fraction_tuple(values: Iterable[int | Fraction]) -> tuple[Fraction, ...]:
    return tuple(_native_fraction(value) for value in values)


def _ordinary_multiply(left: Sequence[Fraction], right: Sequence[Fraction], degree: int) -> list[Fraction]:
    out = [Fraction(0) for _ in range(degree + 1)]
    for i, a in enumerate(left):
        if a == 0:
            continue
        for j, b in enumerate(right):
            if i + j > degree:
                break
            if b != 0:
                out[i + j] += a * b
    return out


def _ordinary_reciprocal(values: Sequence[Fraction], degree: int) -> list[Fraction]:
    if not values or values[0] == 0:
        raise ZeroDivisionError("formal reciprocal requires a nonzero constant coefficient")
    out = [Fraction(0) for _ in range(degree + 1)]
    out[0] = 1 / values[0]
    for n in range(1, degree + 1):
        out[n] = -sum(values[k] * out[n - k] for k in range(1, min(n, len(values) - 1) + 1)) / values[0]
    return out


@dataclass(frozen=True, slots=True)
class PacketIdentity:
    """Exact finite identity for one residue packet transfer."""

    modulus: int
    residue: int
    source_basis: str
    target_basis: str
    source_degree: int
    packet_degree: int
    valid: bool


class FormalSeries:
    r"""A finite exact series ``sum a_n z^n/gamma_n``.

    The stored values are normalized coefficients ``a_n``.  Conversion to
    ordinary monomial coefficients is exact and explicit.
    """

    def __init__(self, basis: WeightedMonomialBasis, coefficients: Iterable[int | Fraction]) -> None:
        if not isinstance(basis, WeightedMonomialBasis):
            raise TypeError("basis must be a WeightedMonomialBasis")
        self.basis = basis
        values = _fraction_tuple(coefficients)
        if not values:
            values = (Fraction(0),)
        self.coefficients = values

    @property
    def degree(self) -> int:
        return len(self.coefficients) - 1

    def truncate(self, degree: int) -> "FormalSeries":
        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
            raise ValueError("degree must be nonnegative")
        values = list(self.coefficients[: degree + 1])
        values.extend(Fraction(0) for _ in range(degree + 1 - len(values)))
        return FormalSeries(self.basis, values)

    def ordinary_coefficients(self, degree: int | None = None) -> tuple[Fraction, ...]:
        bound = self.degree if degree is None else degree
        if isinstance(bound, bool) or not isinstance(bound, int) or bound < 0:
            raise ValueError("degree must be nonnegative")
        values = self.truncate(bound).coefficients
        return tuple(values[n] / self.basis.gamma(n) for n in range(bound + 1))

    @classmethod
    def from_ordinary_coefficients(
        cls,
        basis: WeightedMonomialBasis,
        coefficients: Iterable[int | Fraction],
    ) -> "FormalSeries":
        values = _fraction_tuple(coefficients)
        return cls(basis, (value * basis.gamma(n) for n, value in enumerate(values)))

    def convert(self, basis: WeightedMonomialBasis) -> "FormalSeries":
        return FormalSeries.from_ordinary_coefficients(basis, self.ordinary_coefficients())

    def _coerce(self, other: "FormalSeries") -> tuple["FormalSeries", "FormalSeries"]:
        if not isinstance(other, FormalSeries):
            raise TypeError("formal-series operation requires another FormalSeries")
        if other.basis is self.basis:
            return self, other
        if other.basis.name == self.basis.name and other.basis.equivalent_through(
            self.basis, max(self.degree, other.degree)
        ):
            return self, other
        return self, other.convert(self.basis)

    def add(self, other: "FormalSeries", *, degree: int | None = None) -> "FormalSeries":
        left, right = self._coerce(other)
        bound = max(left.degree, right.degree) if degree is None else degree
        a = left.truncate(bound).coefficients
        b = right.truncate(bound).coefficients
        return FormalSeries(self.basis, (a[n] + b[n] for n in range(bound + 1)))

    def subtract(self, other: "FormalSeries", *, degree: int | None = None) -> "FormalSeries":
        return self.add(other.scale(-1), degree=degree)

    def scale(self, scalar: int | Fraction) -> "FormalSeries":
        factor = _native_fraction(scalar)
        return FormalSeries(self.basis, (factor * value for value in self.coefficients))

    def multiply(self, other: "FormalSeries", *, degree: int | None = None) -> "FormalSeries":
        left, right = self._coerce(other)
        bound = left.degree + right.degree if degree is None else degree
        a = left.truncate(bound).coefficients
        b = right.truncate(bound).coefficients
        out = [Fraction(0) for _ in range(bound + 1)]
        for n in range(bound + 1):
            out[n] = sum(
                self.basis.structure_constant(n, k) * a[k] * b[n - k]
                for k in range(n + 1)
            )
        return FormalSeries(self.basis, out)

    def hadamard(self, other: "FormalSeries", *, degree: int | None = None) -> "FormalSeries":
        """Return the coefficientwise product in the current normalized basis."""

        left, right = self._coerce(other)
        bound = max(left.degree, right.degree) if degree is None else degree
        a = left.truncate(bound).coefficients
        b = right.truncate(bound).coefficients
        return FormalSeries(self.basis, (a[n] * b[n] for n in range(bound + 1)))

    def ordinary_derivative(self) -> "FormalSeries":
        if self.degree == 0:
            return FormalSeries(self.basis, [0])
        out = []
        for n in range(self.degree):
            out.append(Fraction(n + 1) * self.basis.gamma(n) * self.coefficients[n + 1] / self.basis.gamma(n + 1))
        return FormalSeries(self.basis, out)

    def ordinary_integral(self, constant: int | Fraction = 0) -> "FormalSeries":
        out = [_native_fraction(constant)]
        for n, value in enumerate(self.coefficients):
            out.append(value * self.basis.gamma(n + 1) / (Fraction(n + 1) * self.basis.gamma(n)))
        return FormalSeries(self.basis, out)

    def custom_lower(self) -> "FormalSeries":
        r"""Apply ``D_gamma``; this shifts normalized coefficients exactly."""

        if self.degree == 0:
            return FormalSeries(self.basis, [0])
        return FormalSeries(self.basis, self.coefficients[1:])

    def reciprocal(self, *, degree: int | None = None) -> "FormalSeries":
        bound = self.degree if degree is None else degree
        ordinary = self.ordinary_coefficients(bound)
        return FormalSeries.from_ordinary_coefficients(self.basis, _ordinary_reciprocal(ordinary, bound))

    def compose(self, inner: "FormalSeries", *, degree: int | None = None) -> "FormalSeries":
        r"""Return the ordinary formal composition ``self(inner(z))``.

        Composition is basis-independent semantically.  The implementation
        converts to ordinary coefficients, performs exact truncated
        composition, and converts back to the outer basis.
        """

        bound = inner.degree if degree is None else degree
        f = self.ordinary_coefficients(bound)
        g = inner.ordinary_coefficients(bound)
        if g[0] != 0:
            raise ValueError("formal composition requires an inner series with zero constant term")
        out = [Fraction(0) for _ in range(bound + 1)]
        power = [Fraction(0) for _ in range(bound + 1)]
        power[0] = 1
        for k in range(bound + 1):
            if k < len(f) and f[k] != 0:
                for n in range(bound + 1):
                    out[n] += f[k] * power[n]
            power = _ordinary_multiply(power, g, bound)
        return FormalSeries.from_ordinary_coefficients(self.basis, out)

    def exp(self, *, degree: int | None = None) -> "FormalSeries":
        r"""Return the exact formal exponential when the constant term is zero."""

        bound = self.degree if degree is None else degree
        g = self.ordinary_coefficients(bound)
        if g[0] != 0:
            raise ValueError("exact rational formal exp requires zero constant term")
        out = [Fraction(0) for _ in range(bound + 1)]
        out[0] = 1
        # n f_n = sum_(k=1)^n k g_k f_(n-k)
        for n in range(1, bound + 1):
            out[n] = sum(Fraction(k) * g[k] * out[n - k] for k in range(1, n + 1)) / n
        return FormalSeries.from_ordinary_coefficients(self.basis, out)

    def log(self, *, degree: int | None = None) -> "FormalSeries":
        r"""Return the exact formal logarithm when the constant term is one."""

        bound = self.degree if degree is None else degree
        f = self.ordinary_coefficients(bound)
        if f[0] != 1:
            raise ValueError("formal log requires constant coefficient one")
        derivative = [Fraction(n + 1) * f[n + 1] for n in range(bound) if n + 1 < len(f)]
        derivative.extend(Fraction(0) for _ in range(bound - len(derivative)))
        reciprocal = _ordinary_reciprocal(f, bound)
        quotient = _ordinary_multiply(derivative, reciprocal, max(bound - 1, 0))
        out = [Fraction(0)]
        out.extend(quotient[n - 1] / n for n in range(1, bound + 1))
        return FormalSeries.from_ordinary_coefficients(self.basis, out)

    def power(self, exponent: int, *, degree: int | None = None) -> "FormalSeries":
        if isinstance(exponent, bool) or not isinstance(exponent, int):
            raise TypeError("formal integer power requires an integer exponent")
        bound = self.degree if degree is None else degree
        if exponent < 0:
            return self.reciprocal(degree=bound).power(-exponent, degree=bound)
        result = FormalSeries(self.basis, [1]).truncate(bound)
        base = self.truncate(bound)
        power = exponent
        while power:
            if power & 1:
                result = result.multiply(base, degree=bound)
            base = base.multiply(base, degree=bound)
            power >>= 1
        return result

    def project(self, modulus: int, residue: int) -> "FormalSeries":
        if modulus < 1 or not 0 <= residue < modulus:
            raise ValueError("invalid residue projector")
        return FormalSeries(
            self.basis,
            (value if n % modulus == residue else 0 for n, value in enumerate(self.coefficients)),
        )

    def packet(
        self,
        modulus: int,
        residue: int,
        *,
        target_basis: WeightedMonomialBasis,
    ) -> "FormalSeries":
        if modulus < 1 or not 0 <= residue < modulus:
            raise ValueError("invalid packet address")
        count = 0 if self.degree < residue else (self.degree - residue) // modulus + 1
        values = []
        for q in range(count):
            n = modulus * q + residue
            values.append(
                self.coefficients[n]
                * self.basis.packet_transfer_weight(modulus, residue, q, target_basis)
            )
        return FormalSeries(target_basis, values or [0])

    def packet_identity(
        self,
        modulus: int,
        residue: int,
        *,
        target_basis: WeightedMonomialBasis,
    ) -> PacketIdentity:
        packet = self.packet(modulus, residue, target_basis=target_basis)
        reconstructed = [Fraction(0) for _ in range(self.degree + 1)]
        ordinary_packet = packet.ordinary_coefficients()
        for q, coefficient in enumerate(ordinary_packet):
            n = modulus * q + residue
            if n <= self.degree:
                reconstructed[n] = coefficient
        projected = self.project(modulus, residue).ordinary_coefficients()
        return PacketIdentity(
            modulus=modulus,
            residue=residue,
            source_basis=self.basis.name,
            target_basis=target_basis.name,
            source_degree=self.degree,
            packet_degree=packet.degree,
            valid=tuple(reconstructed) == projected,
        )

    def evaluate(self, z: complex) -> complex:
        return sum(complex(value / self.basis.gamma(n)) * z**n for n, value in enumerate(self.coefficients))

    def __add__(self, other: "FormalSeries") -> "FormalSeries":
        return self.add(other)

    def __sub__(self, other: "FormalSeries") -> "FormalSeries":
        return self.subtract(other)

    def __mul__(self, other: "FormalSeries" | int | Fraction) -> "FormalSeries":
        if isinstance(other, FormalSeries):
            return self.multiply(other)
        return self.scale(other)

    def __rmul__(self, other: int | Fraction) -> "FormalSeries":
        return self.scale(other)

    def __pow__(self, exponent: int) -> "FormalSeries":
        return self.power(exponent)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FormalSeries):
            return NotImplemented
        bound = max(self.degree, other.degree)
        return self.ordinary_coefficients(bound) == other.ordinary_coefficients(bound)

    def __getitem__(self, n: int) -> Fraction:
        return self.coefficients[n]

    def __repr__(self) -> str:
        return f"FormalSeries(basis={self.basis.name!r}, degree={self.degree})"


__all__ = ["FormalSeries", "PacketIdentity"]
