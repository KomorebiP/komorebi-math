"""Exact prime-power unit-factorial and custom kernel compilers."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd, isqrt
from typing import Callable


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor <= isqrt(value):
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def factorial_valuation(index: int, prime: int) -> int:
    """Return ``v_prime(index!)`` by Legendre's formula."""

    if index < 0:
        raise ValueError("index must be nonnegative")
    if not _is_prime(prime):
        raise ValueError("prime must be prime")
    total = 0
    remaining = index
    while remaining:
        remaining //= prime
        total += remaining
    return total


def integer_valuation(value: int, prime: int) -> int:
    """Return the finite p-adic valuation of a nonzero integer."""

    if value == 0:
        raise ValueError("zero has no finite valuation")
    if not _is_prime(prime):
        raise ValueError("prime must be prime")
    remaining = abs(value)
    total = 0
    while remaining % prime == 0:
        remaining //= prime
        total += 1
    return total


def binomial_valuation(index: int, subindex: int, prime: int) -> int:
    """Return ``v_prime(binomial(index,subindex))``."""

    if subindex < 0 or subindex > index:
        raise ValueError("subindex must lie between zero and index")
    return (
        factorial_valuation(index, prime)
        - factorial_valuation(subindex, prime)
        - factorial_valuation(index - subindex, prime)
    )


def borrow_count(index: int, subindex: int, prime: int) -> int:
    """Count base-p borrows in subtracting ``subindex`` from ``index``."""

    if subindex < 0 or subindex > index:
        raise ValueError("subindex must lie between zero and index")
    if not _is_prime(prime):
        raise ValueError("prime must be prime")
    n = index
    k = subindex
    borrow = 0
    count = 0
    while n or k or borrow:
        n, n_digit = divmod(n, prime)
        k, k_digit = divmod(k, prime)
        needed = k_digit + borrow
        if n_digit < needed:
            count += 1
            borrow = 1
        else:
            borrow = 0
    return count


@dataclass(frozen=True)
class PrimePowerBinomial:
    """Valuation, unit part, and residue of a binomial coefficient."""

    index: int
    subindex: int
    prime: int
    exponent: int
    modulus: int
    valuation: int
    unit: int
    residue: int

    @property
    def survives(self) -> bool:
        """Whether the residue is nonzero modulo ``p**r``."""

        return self.valuation < self.exponent


@dataclass
class PrimePowerUnitCompiler:
    """Compile p-free factorials and binomial residues modulo ``p**r``.

    The prefix table costs ``O(p**r)`` memory. Each query then uses
    ``O(log_p(n))`` iterations, with memoization of completed unit factorials.
    """

    prime: int
    exponent: int
    max_modulus: int = 2_000_000
    modulus: int = field(init=False)
    prefix_units: tuple[int, ...] = field(init=False, repr=False)
    block_unit: int = field(init=False)
    _cache: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not _is_prime(self.prime):
            raise ValueError("prime must be prime")
        if self.exponent < 1:
            raise ValueError("exponent must be positive")
        modulus = self.prime**self.exponent
        if modulus > self.max_modulus:
            raise ValueError(
                f"modulus {modulus} exceeds max_modulus={self.max_modulus}"
            )
        prefix = [1] * (modulus + 1)
        running = 1
        for value in range(1, modulus + 1):
            if value % self.prime:
                running = running * value % modulus
            prefix[value] = running
        self.modulus = modulus
        self.prefix_units = tuple(prefix)
        self.block_unit = prefix[modulus]
        self._cache = {0: 1}

    def unit_factorial(self, index: int) -> int:
        """Return ``index!/p**v_p(index!)`` modulo the configured modulus."""

        if index < 0:
            raise ValueError("index must be nonnegative")
        if index in self._cache:
            return self._cache[index]
        remaining = index
        result = 1
        while remaining:
            result = (
                result
                * pow(self.block_unit, remaining // self.modulus, self.modulus)
                * self.prefix_units[remaining % self.modulus]
            ) % self.modulus
            remaining //= self.prime
        self._cache[index] = result
        return result

    def binomial_data(self, index: int, subindex: int) -> PrimePowerBinomial:
        """Return the complete prime-power decomposition of one binomial."""

        if subindex < 0 or subindex > index:
            raise ValueError("subindex must lie between zero and index")
        valuation = binomial_valuation(index, subindex, self.prime)
        if valuation >= self.exponent:
            return PrimePowerBinomial(
                index,
                subindex,
                self.prime,
                self.exponent,
                self.modulus,
                valuation,
                0,
                0,
            )
        numerator_unit = self.unit_factorial(index)
        denominator_unit = (
            self.unit_factorial(subindex) * self.unit_factorial(index - subindex)
        ) % self.modulus
        unit = numerator_unit * pow(denominator_unit, -1, self.modulus) % self.modulus
        residue = pow(self.prime, valuation, self.modulus) * unit % self.modulus
        return PrimePowerBinomial(
            index,
            subindex,
            self.prime,
            self.exponent,
            self.modulus,
            valuation,
            unit,
            residue,
        )

    def binomial(self, index: int, subindex: int) -> int:
        """Return ``binomial(index,subindex)`` modulo the configured modulus."""

        return self.binomial_data(index, subindex).residue


@dataclass(frozen=True)
class NormalizedKernelResidue:
    """Prime-power residue of a carry-separable normalized-basis kernel."""

    index: int
    subindex: int
    factorial_power: int
    prime: int
    exponent: int
    valuation: int
    unit: int
    residue: int

    @property
    def survives(self) -> bool:
        """Whether the kernel survives modulo the selected prime power."""

        return self.valuation < self.exponent


def constant_unit_sequence(_index: int) -> int:
    """Return the multiplicative identity for the untwisted kernel family."""

    return 1


@dataclass
class PrimePowerKernelCompiler:
    """Compile kernels for ``gamma_n=c**n (n!)**s u_n`` modulo ``p**r``.

    The geometric factor cancels from every structure constant. The supplied
    ``unit_sequence`` must take values coprime to ``p``.
    """

    prime: int
    exponent: int
    factorial_power: int
    unit_sequence: Callable[[int], int] | None = None
    max_modulus: int = 2_000_000
    binomial_compiler: PrimePowerUnitCompiler = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.factorial_power < 0:
            raise ValueError("factorial_power must be nonnegative")
        if self.unit_sequence is None:
            self.unit_sequence = constant_unit_sequence
        elif not callable(self.unit_sequence):
            raise TypeError("unit_sequence must be callable or None")
        self.binomial_compiler = PrimePowerUnitCompiler(
            self.prime, self.exponent, self.max_modulus
        )

    @property
    def modulus(self) -> int:
        """Configured prime-power modulus."""

        return self.binomial_compiler.modulus

    def kernel(self, index: int, subindex: int) -> NormalizedKernelResidue:
        """Return one complete normalized structure constant modulo ``p**r``."""

        data = self.binomial_compiler.binomial_data(index, subindex)
        valuation = self.factorial_power * data.valuation
        if valuation >= self.exponent:
            return NormalizedKernelResidue(
                index,
                subindex,
                self.factorial_power,
                self.prime,
                self.exponent,
                valuation,
                0,
                0,
            )
        unit_sequence = self.unit_sequence
        if unit_sequence is None:  # pragma: no cover - normalized in __post_init__
            raise RuntimeError("unit_sequence was not initialized")
        units = tuple(
            int(unit_sequence(value)) % self.modulus
            for value in (index, subindex, index - subindex)
        )
        if any(gcd(value, self.prime) != 1 for value in units):
            raise ValueError("unit_sequence must return p-adic units")
        custom_ratio = units[0] * pow(units[1] * units[2] % self.modulus, -1, self.modulus)
        custom_ratio %= self.modulus
        unit = pow(data.unit, self.factorial_power, self.modulus)
        unit = unit * custom_ratio % self.modulus
        residue = pow(self.prime, valuation, self.modulus) * unit % self.modulus
        return NormalizedKernelResidue(
            index,
            subindex,
            self.factorial_power,
            self.prime,
            self.exponent,
            valuation,
            unit,
            residue,
        )
