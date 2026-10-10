"""Exact fixed-precision Cartier states for rational Mahler masks.

This is the finite rectangle of Cartier Compilers, Chapter 23, Theorems
23.3-23.4.  It represents the unique unit series ``B F = A F(z**p)`` over
``Z/p**r Z``.  A state is ``sum(P_l / B**l) F``; its coordinates may be
redundant.  No reachable/observable minimization, nonfree module quotient,
analytic continuation or compatibility between minimized precisions is claimed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import comb
from typing import Any

from .._number_theory import require_integer, require_prime
from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ResourceLimitError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .budget import DEFAULT_GENERATING_BUDGET, GeneratingBudget
from .cartier import polynomial_multiply_mod, polynomial_power_mod

Polynomial = tuple[int, ...]
PrimePowerCartierState = tuple[Polynomial, ...]
Branch = tuple[int, Polynomial]


def _nonnegative(value: Any, name: str) -> int:
    result = require_integer(value, name=name)
    if result < 0:
        raise ValidationError(f"{name} must be nonnegative")
    return result


def _polynomial(values: Iterable[int], name: str, modulus: int, budget: GeneratingBudget) -> Polynomial:
    # Enforce the slot budget while consuming an iterable, before copying it.
    data = []
    for value in values:
        budget.require_degree(len(data))
        integer = require_integer(value, name=f"{name} coefficient")
        budget.require_coefficient_bits(abs(integer).bit_length())
        data.append(integer % modulus)
    if not data:
        raise ValidationError(f"{name} must have at least one coefficient")
    while len(data) > 1 and data[-1] == 0:
        data.pop()
    return tuple(data)


def _frobenius_defect(denominator: Polynomial, prime: int, modulus: int) -> Polynomial:
    """Return ``(B**p-B(z**p))/p`` modulo modulus, with the source sign.

    Compute the numerator modulo p*modulus before division.  Reducing it
    modulo modulus first would discard a p-adic digit needed by the quotient.
    """
    lifted_modulus = prime * modulus
    powered = polynomial_power_mod(denominator, prime, lifted_modulus)
    size = max(len(powered), prime * (len(denominator) - 1) + 1)
    values = []
    for index in range(size):
        substituted = (
            denominator[index // prime] if index % prime == 0 and index // prime < len(denominator) else 0
        )
        difference = ((powered[index] if index < len(powered) else 0) - substituted) % lifted_modulus
        if difference % prime:
            raise ArithmeticError("integral Frobenius defect lost its divisibility")
        values.append(difference // prime)
    while len(values) > 1 and values[-1] == 0:
        values.pop()
    return tuple(values)


def _mahler_prefix(
    numerator: Polynomial, denominator: Polynomial, prime: int, modulus: int, count: int
) -> Polynomial:
    """Independent triangular coefficient recurrence for a finite replay."""
    inverse = pow(denominator[0], -1, modulus)
    values = [1]
    for index in range(1, count):
        right = sum(
            value * values[(index - degree) // prime]
            for degree, value in enumerate(numerator)
            if degree <= index and (index - degree) % prime == 0
        )
        left_tail = sum(
            denominator[degree] * values[index - degree]
            for degree in range(1, min(index + 1, len(denominator)))
        )
        values.append((right - left_tail) * inverse % modulus)
    return tuple(values[:count])


@documented(
    ProvenanceRecord(
        identifier="CC-23-PRIME-POWER-CARTIER",
        title="Fixed-precision rational-mask Cartier rectangle",
        status=ClaimStatus.FINITE_CERTIFICATE,
        source="Cartier Compilers, Chapter 23, Theorems 23.3-23.4 and Corollary 23.5",
        assumptions=("prime p and fixed positive precision r", "unit B(0) and A(0)=B(0) modulo p**r"),
        limitations=(
            "redundant finite rectangle, no canonical ring-module minimization",
            "finite replay certificates are not universal proofs",
        ),
        verification_suite="generating.cartier_prime_power",
    )
)
@dataclass(frozen=True, slots=True, init=False)
class PrimePowerCartierRepresentation:
    """Immutable exact denominator-layer representation at one precision.

    Polynomial coefficients and digit words are integers.  The layer bound
    is ``p*r``, and the coefficient modulus is ``p**r``.  Transitions use
    Python integer arithmetic throughout.  ``branches[l]`` stores exact
    numerator multipliers paired with their target denominator layers.
    """

    prime: int
    exponent: int
    modulus: int
    numerator: Polynomial
    denominator: Polynomial
    layer_bound: int
    degree_bound: int
    frobenius_defect: Polynomial
    branches: tuple[tuple[Branch, ...], ...]
    initial_state: PrimePowerCartierState
    prune: bool
    budget: GeneratingBudget

    def __init__(
        self,
        numerator: Iterable[int],
        denominator: Iterable[int],
        prime: int,
        exponent: int,
        *,
        prune: bool = True,
        budget: GeneratingBudget = DEFAULT_GENERATING_BUDGET,
    ) -> None:
        if not isinstance(budget, GeneratingBudget):
            raise TypeError("budget must be a GeneratingBudget")
        p = require_integer(prime, name="prime")
        budget.require_modulus(p)
        p = require_prime(p)
        r = require_integer(exponent, name="exponent")
        if r < 1:
            raise ValidationError("exponent must be positive")
        if not isinstance(prune, bool):
            raise ValidationError("prune must be a boolean")
        # Capped multiplication rejects huge exponents without creating p**r.
        modulus = 1
        for _ in range(r):
            modulus *= p
            budget.require_modulus(modulus)
        a = _polynomial(numerator, "numerator", modulus, budget)
        b = _polynomial(denominator, "denominator", modulus, budget)
        if b[0] % p == 0:
            raise ValidationError("denominator constant must be a unit modulo prime")
        if a[0] != b[0]:
            raise ValidationError("A(0)=B(0) is required modulo p**r")
        layer_bound = p * r
        forcing_degree = len(a) - 1 + (layer_bound - 1) * (len(b) - 1)
        degree_bound = (forcing_degree + p - 2) // (p - 1)
        budget.require_degree(degree_bound + forcing_degree)
        slots = (layer_bound + 1) * (degree_bound + 1)
        budget.require_matrix_dimension(slots)
        budget.require_local_combinations((layer_bound + 1) * r * (degree_bound + 1) * (forcing_degree + 1))
        defect = _frobenius_defect(b, p, modulus)
        b_powers = [(1,)]
        for _ in range(p - 1):
            b_powers.append(polynomial_multiply_mod(b_powers[-1], b, modulus))
        defect_powers = [(1,)]
        for _ in range(r - 1):
            defect_powers.append(polynomial_multiply_mod(defect_powers[-1], defect, modulus))
        layers = []
        for layer in range(layer_bound + 1):
            padding_quotient = (layer + p) // p  # ceil((layer+1)/p)
            padding_power = p * padding_quotient - layer - 1
            base = polynomial_multiply_mod(a, b_powers[padding_power], modulus)
            branches = []
            for order in range(r):
                target = padding_quotient + order
                scalar = pow(-p, order, modulus) * comb(padding_quotient + order - 1, order) % modulus
                if prune and scalar == 0:
                    continue
                multiplier = polynomial_multiply_mod(base, defect_powers[order], modulus)
                multiplier = tuple(scalar * value % modulus for value in multiplier)
                if prune and not any(multiplier):
                    continue
                if target > layer_bound or len(multiplier) - 1 > forcing_degree:
                    raise ArithmeticError("prime-power Cartier branch left its invariant rectangle")
                branches.append((target, multiplier))
            layers.append(tuple(branches))
        zero = (0,) * (degree_bound + 1)
        initial = ((1,) + (0,) * degree_bound,) + (zero,) * layer_bound
        for name, value in (
            ("prime", p),
            ("exponent", r),
            ("modulus", modulus),
            ("numerator", a),
            ("denominator", b),
            ("layer_bound", layer_bound),
            ("degree_bound", degree_bound),
            ("frobenius_defect", defect),
            ("branches", tuple(layers)),
            ("initial_state", initial),
            ("prune", prune),
            ("budget", budget),
        ):
            object.__setattr__(self, name, value)

    @property
    def coordinate_slots(self) -> int:
        return (self.layer_bound + 1) * (self.degree_bound + 1)

    @property
    def theoretical_state_bound(self) -> int:
        """Ambient rectangle cardinality, not minimum automaton size."""
        return self.modulus**self.coordinate_slots

    def _state(self, state: Sequence[Sequence[int]]) -> PrimePowerCartierState:
        if len(state) != self.layer_bound + 1:
            raise DimensionError("state must have layer_bound+1 denominator layers")
        result = []
        for polynomial in state:
            if len(polynomial) != self.degree_bound + 1:
                raise DimensionError("every state polynomial must have degree_bound+1 coefficient slots")
            coefficients = []
            for value in polynomial:
                integer = require_integer(value, name="state coefficient")
                self.budget.require_coefficient_bits(abs(integer).bit_length())
                coefficients.append(integer % self.modulus)
            result.append(tuple(coefficients))
        return tuple(result)

    def transition(self, state: Sequence[Sequence[int]], digit: int) -> PrimePowerCartierState:
        """Apply the exact section ``Lambda_digit`` to any rectangle state."""
        selected = require_integer(digit, name="digit")
        if not 0 <= selected < self.prime:
            raise ValidationError("digit must lie in 0,...,prime-1")
        source = self._state(state)
        output = [[0] * (self.degree_bound + 1) for _ in range(self.layer_bound + 1)]
        for layer, polynomial in enumerate(source):
            if not any(polynomial):
                continue
            for target, multiplier in self.branches[layer]:
                product = polynomial_multiply_mod(polynomial, multiplier, self.modulus)
                for degree, coefficient in enumerate(product[selected :: self.prime]):
                    if degree > self.degree_bound:
                        raise ArithmeticError("Cartier section left its invariant degree bound")
                    output[target][degree] = (output[target][degree] + coefficient) % self.modulus
        return tuple(tuple(polynomial) for polynomial in output)

    def output(self, state: Sequence[Sequence[int]]) -> int:
        """Evaluate the represented series at zero, using F(0)=1."""
        source = self._state(state)
        inverse_b0 = pow(self.denominator[0], -1, self.modulus)
        return (
            sum(
                polynomial[0] * pow(inverse_b0, layer, self.modulus)
                for layer, polynomial in enumerate(source)
            )
            % self.modulus
        )

    def evaluate(self, index: int) -> int:
        """Read least-significant-first base-p digits and return one residue."""
        remaining = _nonnegative(index, "index")
        # Count the actual word before doing state transitions. Exponentiating
        # a caller's depth allowance can be enormous even for index zero.
        digits = []
        while remaining:
            if len(digits) >= self.budget.maximum_recursion_depth:
                raise ResourceLimitError("index exceeds the configured digit-word length")
            remaining, digit = divmod(remaining, self.prime)
            digits.append(digit)
        state = self.initial_state
        for digit in digits:
            state = self.transition(state, digit)
        return self.output(state)

    def sequence(self, count: int) -> Polynomial:
        """Evaluate a bounded finite coefficient prefix."""
        size = _nonnegative(count, "count")
        self.budget.require_terms(size, label="coefficient prefix")
        return tuple(self.evaluate(index) for index in range(size))

    def certificate(self, count: int = 64) -> Certificate:
        """Compare a finite prefix with the triangular Mahler recurrence.

        The comparison range is explicit.  Rectangle construction follows the
        displayed source formula; this finite replay is additional software
        evidence, not a proof inferred from the prefix.
        """
        size = _nonnegative(count, "count")
        if size == 0:
            raise ValidationError("certificate comparison count must be positive")
        self.budget.require_terms(size, label="certificate comparison")
        expected = _mahler_prefix(self.numerator, self.denominator, self.prime, self.modulus, size)
        actual = self.sequence(size)
        failures = tuple(
            (index, observed, expected[index])
            for index, observed in enumerate(actual)
            if observed != expected[index]
        )
        return Certificate(
            valid=not failures,
            exactness=Exactness.EXACT,
            residuals={"prefix_mismatches": len(failures)},
            assumptions_checked=(
                "prime and fixed positive precision",
                "unit denominator constant",
                "equal constants modulo p**r",
                "source rectangle layer/degree bounds",
                "exact finite Mahler recurrence comparison",
            ),
            diagnostics={
                "compared_terms": size,
                "failures": failures,
                "prime": self.prime,
                "exponent": self.exponent,
                "layer_bound": self.layer_bound,
                "degree_bound": self.degree_bound,
                "coordinate_slots": self.coordinate_slots,
            },
            theorem_id="CC-23-PRIME-POWER-CARTIER",
            message="fixed-precision Cartier prefix agrees with the Mahler recurrence"
            if not failures
            else "prime-power Cartier prefix comparison failed",
        )


def compile_prime_power_cartier_representation(
    numerator: Iterable[int],
    denominator: Iterable[int],
    prime: int,
    exponent: int,
    *,
    prune: bool = True,
    budget: GeneratingBudget = DEFAULT_GENERATING_BUDGET,
) -> PrimePowerCartierRepresentation:
    """Compile the Chapter 23 finite rectangle without enumerating its states."""
    return PrimePowerCartierRepresentation(
        numerator, denominator, prime, exponent, prune=prune, budget=budget
    )


__all__ = [
    "PrimePowerCartierRepresentation",
    "PrimePowerCartierState",
    "compile_prime_power_cartier_representation",
]
