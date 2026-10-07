"""Exact normalized monomial bases for formal generating functions.

A basis is determined by nonzero normalization factors ``gamma_n`` through
``e_n(z) = z**n / gamma_n``.  The class accepts arbitrary callable rules for
live user-defined bases, while :class:`BasisDescriptor` records the closed, declarative
subset that can be validated and persisted safely.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import factorial
from typing import Any, Callable, Iterable, Mapping


GammaRule = Callable[[int], int | Fraction]
BasisParameter = int | Fraction | str | tuple[Fraction, ...]


@dataclass(frozen=True, slots=True)
class BasisDescriptor:
    """Declarative, portable description of one normalized monomial basis.

    ``kind`` selects a closed constructor.  ``parameters`` is an ordered tuple
    rather than an arbitrary mapping so descriptors are immutable, hashable,
    and canonically serializable.  No Python import path or callback name is
    accepted from persisted data.
    """

    name: str
    kind: str
    parameters: tuple[tuple[str, BasisParameter], ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("basis descriptor name must be nonempty")
        if not self.kind.strip():
            raise ValueError("basis descriptor kind must be nonempty")
        keys = tuple(key for key, _ in self.parameters)
        if len(set(keys)) != len(keys):
            raise ValueError("basis descriptor parameters must have unique keys")
        if tuple(sorted(keys)) != keys:
            raise ValueError("basis descriptor parameters must be sorted by key")

    def parameter_map(self) -> dict[str, BasisParameter]:
        """Return a mutable copy of the descriptor parameter mapping."""

        return dict(self.parameters)

    @classmethod
    def create(
        cls,
        name: str,
        kind: str,
        parameters: Mapping[str, BasisParameter] | None = None,
    ) -> "BasisDescriptor":
        """Construct a descriptor with deterministic parameter ordering."""

        items = () if parameters is None else tuple(sorted(parameters.items()))
        return cls(str(name), str(kind), items)


@dataclass(frozen=True, slots=True)
class BasisCertificate:
    """Finite exact diagnostics for one normalized monomial basis."""

    name: str
    degree: int
    nonzero_normalization: bool
    integral_structure_constants: bool
    maximum_denominator: int
    maximum_numerator_bits: int


class WeightedMonomialBasis:
    r"""A basis ``e_n(z)=z^n/gamma_n`` with ``gamma_0=1``.

    The associated product has structure constants

    ``C(n,k)=gamma_n/(gamma_k*gamma_(n-k))``.

    The custom lowering operator is defined by

    ``D_gamma z^n = (gamma_n/gamma_(n-1)) z^(n-1)``.

    Consequently ``D_gamma e_n=e_(n-1)`` for every positive degree.
    """

    def __init__(
        self,
        name: str,
        gamma: GammaRule,
        *,
        descriptor: BasisDescriptor | None = None,
    ) -> None:
        if not name:
            raise ValueError("basis name must be nonempty")
        if not callable(gamma):
            raise TypeError("gamma must be callable")
        if descriptor is not None and descriptor.name != str(name):
            raise ValueError("basis name and descriptor name must agree")
        self.name = str(name)
        self._gamma_rule = gamma
        self._descriptor = descriptor
        if self.gamma(0) != 1:
            raise ValueError("gamma(0) must equal one")

    @property
    def descriptor(self) -> BasisDescriptor | None:
        """Return the portable descriptor, or ``None`` for a live callback."""

        return self._descriptor

    @property
    def portable(self) -> bool:
        """Report whether the basis has a closed declarative descriptor."""

        return self._descriptor is not None

    def require_descriptor(self) -> BasisDescriptor:
        """Return the descriptor or reject an opaque callback basis."""

        if self._descriptor is None:
            raise ValueError(
                f"basis {self.name!r} is callback-defined and cannot be persisted portably"
            )
        return self._descriptor

    def semantically_equivalent(self, other: "WeightedMonomialBasis") -> bool:
        """Test declared basis equivalence without comparing Python callbacks."""

        if self is other:
            return True
        if not isinstance(other, WeightedMonomialBasis):
            return False
        return self._descriptor is not None and self._descriptor == other._descriptor

    def equivalent_through(
        self,
        other: "WeightedMonomialBasis",
        degree: int,
    ) -> bool:
        """Test exact normalization equality through a finite degree.

        Declarative bases are compared by descriptor first.  Callback-defined
        bases can still participate in finite operations when their explicitly
        evaluated normalization tables agree on the requested range.
        """

        if not isinstance(other, WeightedMonomialBasis):
            return False
        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
            raise ValueError("degree must be a nonnegative integer")
        if self.semantically_equivalent(other):
            return True
        return all(self.gamma(index) == other.gamma(index) for index in range(degree + 1))

    @lru_cache(maxsize=None)
    def gamma(self, n: int) -> Fraction:
        """Return the exact basis normalization ``gamma_n``."""

        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("basis degree must be a nonnegative integer")
        value = Fraction(self._gamma_rule(n))
        if value == 0:
            raise ZeroDivisionError(f"basis normalization gamma({n}) vanishes")
        return value

    def lowering_weight(self, n: int) -> Fraction:
        """Return ``gamma_n/gamma_(n-1)`` for positive ``n``."""

        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("lowering degree must be a positive integer")
        return self.gamma(n) / self.gamma(n - 1)

    def structure_constant(self, n: int, k: int) -> Fraction:
        """Return the exact generalized binomial coefficient ``C(n,k)``."""

        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("n must be a nonnegative integer")
        if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= n:
            raise ValueError("k must lie between zero and n")
        return self.gamma(n) / (self.gamma(k) * self.gamma(n - k))

    def packet_transfer_weight(
        self,
        modulus: int,
        residue: int,
        packet_index: int,
        target_basis: "WeightedMonomialBasis",
    ) -> Fraction:
        r"""Return the coefficient renormalization for one residue packet.

        If ``F(z)=sum a_n z^n/gamma_n`` and the packet is represented as
        ``G(t)=sum b_q t^q/eta_q``, then

        ``P_(m,r)F(z)=z^r G(z^m)`` with
        ``b_q=eta_q*a_(mq+r)/gamma_(mq+r)``.
        """

        if modulus < 1 or not 0 <= residue < modulus or packet_index < 0:
            raise ValueError("invalid packet address")
        return target_basis.gamma(packet_index) / self.gamma(
            modulus * packet_index + residue
        )

    def finite_certificate(self, degree: int) -> BasisCertificate:
        """Audit normalization and product integrality through one degree."""

        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
            raise ValueError("degree must be nonnegative")
        constants = [
            self.structure_constant(n, k)
            for n in range(degree + 1)
            for k in range(n + 1)
        ]
        return BasisCertificate(
            name=self.name,
            degree=degree,
            nonzero_normalization=all(
                self.gamma(n) != 0 for n in range(degree + 1)
            ),
            integral_structure_constants=all(
                value.denominator == 1 for value in constants
            ),
            maximum_denominator=max(
                (value.denominator for value in constants), default=1
            ),
            maximum_numerator_bits=max(
                (abs(value.numerator).bit_length() for value in constants),
                default=0,
            ),
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, WeightedMonomialBasis) and self.semantically_equivalent(other)

    def __hash__(self) -> int:
        return object.__hash__(self) if self._descriptor is None else hash(self._descriptor)

    def __repr__(self) -> str:
        status = "portable" if self.portable else "callback"
        return f"WeightedMonomialBasis(name={self.name!r}, mode={status!r})"


def ordinary_basis(*, name: str = "ordinary") -> WeightedMonomialBasis:
    """Return the ordinary generating-function basis ``z^n``."""

    descriptor = BasisDescriptor.create(name, "ordinary")
    return WeightedMonomialBasis(name, lambda _n: 1, descriptor=descriptor)


def exponential_basis(*, name: str = "exponential") -> WeightedMonomialBasis:
    """Return the exponential generating-function basis ``z^n/n!``."""

    descriptor = BasisDescriptor.create(name, "exponential")
    return WeightedMonomialBasis(name, factorial, descriptor=descriptor)


def factorial_power_basis(
    power: int,
    *,
    name: str | None = None,
) -> WeightedMonomialBasis:
    r"""Return ``z^n/(n!)^power`` for a positive integer power."""

    if isinstance(power, bool) or not isinstance(power, int) or power < 1:
        raise ValueError("power must be a positive integer")
    basis_name = f"factorial-power-{power}" if name is None else str(name)
    descriptor = BasisDescriptor.create(
        basis_name,
        "factorial-power",
        {"power": power},
    )
    return WeightedMonomialBasis(
        basis_name,
        lambda n: factorial(n) ** power,
        descriptor=descriptor,
    )


def q_factorial_basis(q: int, *, name: str | None = None) -> WeightedMonomialBasis:
    r"""Return the integer ``q``-factorial basis.

    ``gamma_n=[n]_q!`` where ``[j]_q=(q^j-1)/(q-1)``.  At ``q=1`` this is
    the ordinary factorial basis.
    """

    if isinstance(q, bool) or not isinstance(q, int) or q < 1:
        raise ValueError("q must be a positive integer")

    @lru_cache(maxsize=None)
    def gamma(n: int) -> int:
        value = 1
        if q == 1:
            return factorial(n)
        for j in range(1, n + 1):
            value *= (q**j - 1) // (q - 1)
        return value

    basis_name = f"q-factorial-{q}" if name is None else str(name)
    descriptor = BasisDescriptor.create(basis_name, "q-factorial", {"q": q})
    return WeightedMonomialBasis(basis_name, gamma, descriptor=descriptor)


def fibonacci_factorial_basis(
    *,
    name: str = "fibonacci-factorial",
) -> WeightedMonomialBasis:
    """Return the fibonomial basis with ``gamma_n=prod_(j=1)^n F_j``."""

    @lru_cache(maxsize=None)
    def fibonacci(n: int) -> int:
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a

    @lru_cache(maxsize=None)
    def gamma(n: int) -> int:
        value = 1
        for j in range(1, n + 1):
            value *= fibonacci(j)
        return value

    descriptor = BasisDescriptor.create(name, "fibonacci-factorial")
    return WeightedMonomialBasis(name, gamma, descriptor=descriptor)


def pochhammer_basis(
    alpha: int | Fraction,
    *,
    name: str | None = None,
) -> WeightedMonomialBasis:
    """Return ``z^n/(alpha)_n`` for a nonterminating rational path."""

    scalar = Fraction(alpha)
    if scalar.denominator == 1 and scalar.numerator <= 0:
        raise ValueError("alpha cannot be a nonpositive integer")

    @lru_cache(maxsize=None)
    def gamma(n: int) -> Fraction:
        value = Fraction(1)
        for j in range(n):
            value *= scalar + j
        return value

    basis_name = f"pochhammer-{scalar}" if name is None else str(name)
    descriptor = BasisDescriptor.create(
        basis_name,
        "pochhammer",
        {"alpha": scalar},
    )
    return WeightedMonomialBasis(basis_name, gamma, descriptor=descriptor)


def basis_from_values(
    name: str,
    values: Iterable[int | Fraction],
) -> WeightedMonomialBasis:
    """Create a finite portable basis from an explicit normalization table."""

    table = tuple(Fraction(value) for value in values)
    if not table or table[0] != 1:
        raise ValueError("the normalization table must start with one")
    if any(value == 0 for value in table):
        raise ValueError("the normalization table cannot contain zero")

    def gamma(n: int) -> Fraction:
        if n >= len(table):
            raise IndexError(f"basis {name!r} has no normalization at degree {n}")
        return table[n]

    descriptor = BasisDescriptor.create(
        name,
        "table",
        {"values": table},
    )
    return WeightedMonomialBasis(name, gamma, descriptor=descriptor)


def basis_from_descriptor(descriptor: BasisDescriptor) -> WeightedMonomialBasis:
    """Rebuild a basis from the closed declarative descriptor registry."""

    if not isinstance(descriptor, BasisDescriptor):
        raise TypeError("descriptor must be a BasisDescriptor")
    parameters = descriptor.parameter_map()
    kind = descriptor.kind
    expected: set[str]
    if kind == "ordinary":
        expected = set()
        constructor = ordinary_basis(name=descriptor.name)
    elif kind == "exponential":
        expected = set()
        constructor = exponential_basis(name=descriptor.name)
    elif kind == "factorial-power":
        expected = {"power"}
        constructor = factorial_power_basis(
            int(parameters.get("power", 0)),
            name=descriptor.name,
        )
    elif kind == "q-factorial":
        expected = {"q"}
        constructor = q_factorial_basis(
            int(parameters.get("q", 0)),
            name=descriptor.name,
        )
    elif kind == "fibonacci-factorial":
        expected = set()
        constructor = fibonacci_factorial_basis(name=descriptor.name)
    elif kind == "pochhammer":
        expected = {"alpha"}
        constructor = pochhammer_basis(
            Fraction(parameters.get("alpha", 0)),
            name=descriptor.name,
        )
    elif kind == "table":
        expected = {"values"}
        raw_values = parameters.get("values")
        if not isinstance(raw_values, tuple):
            raise ValueError("table descriptor values must be a tuple")
        constructor = basis_from_values(descriptor.name, raw_values)
    else:
        raise ValueError(f"unknown basis descriptor kind {kind!r}")
    if set(parameters) != expected:
        raise ValueError(
            f"basis descriptor {kind!r} requires parameters {sorted(expected)}"
        )
    if constructor.require_descriptor() != descriptor:
        raise ValueError("basis descriptor did not round-trip canonically")
    return constructor


__all__ = [
    "BasisCertificate",
    "BasisDescriptor",
    "WeightedMonomialBasis",
    "basis_from_descriptor",
    "basis_from_values",
    "exponential_basis",
    "factorial_power_basis",
    "fibonacci_factorial_basis",
    "ordinary_basis",
    "pochhammer_basis",
    "q_factorial_basis",
]
