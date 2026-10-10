"""Cartier-section compilers for finite and rational Mahler masks.

For a prime ``p`` and polynomials ``A`` and ``B`` with equal nonzero
constant coefficient, this module studies the unique formal unit series
``F`` satisfying ``B(z)F(z)=A(z)F(z**p)`` modulo ``p``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import isqrt

from ..exceptions import ValidationError
from .budget import DEFAULT_GENERATING_BUDGET, GeneratingBudget


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


def _trim(values: Iterable[int], modulus: int) -> tuple[int, ...]:
    result = [int(value) % modulus for value in values]
    if not result:
        result = [0]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return tuple(result)


def polynomial_multiply_mod(
    left: Sequence[int], right: Sequence[int], modulus: int
) -> tuple[int, ...]:
    """Multiply coefficient tuples modulo ``modulus``."""

    if modulus < 2:
        raise ValueError("modulus must be at least two")
    if not left or not right:
        return (0,)
    values = [0] * (len(left) + len(right) - 1)
    for i, left_value in enumerate(left):
        for j, right_value in enumerate(right):
            values[i + j] = (values[i + j] + left_value * right_value) % modulus
    return _trim(values, modulus)


def polynomial_power_mod(
    polynomial: Sequence[int], exponent: int, modulus: int
) -> tuple[int, ...]:
    """Raise a polynomial to a nonnegative power modulo ``modulus``."""

    if exponent < 0:
        raise ValueError("exponent must be nonnegative")
    result = (1,)
    base = _trim(polynomial, modulus)
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = polynomial_multiply_mod(result, base, modulus)
        remaining >>= 1
        if remaining:
            base = polynomial_multiply_mod(base, base, modulus)
    return result


def polynomial_cartier(
    polynomial: Sequence[int], digit: int, prime: int
) -> tuple[int, ...]:
    """Apply ``Lambda_digit`` to a polynomial modulo ``prime``."""

    if not _is_prime(prime):
        raise ValueError("prime must be prime")
    if not 0 <= digit < prime:
        raise ValueError("digit must lie in 0,...,prime-1")
    selected = [polynomial[index] for index in range(digit, len(polynomial), prime)]
    return _trim(selected or [0], prime)


def rational_mask_coefficients_mod_prime(
    numerator: Sequence[int], denominator: Sequence[int], prime: int, count: int
) -> tuple[int, ...]:
    """Return the first ``count`` coefficients of the Mahler solution modulo ``prime``."""

    if not _is_prime(prime):
        raise ValueError("prime must be prime")
    if count < 0:
        raise ValueError("count must be nonnegative")
    if count == 0:
        return ()
    a = _trim(numerator, prime)
    b = _trim(denominator, prime)
    if b[0] == 0:
        raise ValueError("denominator constant coefficient must be nonzero")
    if a[0] != b[0]:
        raise ValueError("numerator and denominator must have equal constants")
    inverse_b0 = pow(b[0], -1, prime)
    coefficients = [0] * count
    coefficients[0] = 1
    for n in range(1, count):
        right = 0
        for index, value in enumerate(a):
            difference = n - index
            if difference >= 0 and difference % prime == 0:
                right += value * coefficients[difference // prime]
        left_tail = sum(
            b[index] * coefficients[n - index]
            for index in range(1, min(len(b) - 1, n) + 1)
        )
        coefficients[n] = (right - left_tail) * inverse_b0 % prime
    return tuple(coefficients)


def cartier_degree_bound(
    numerator: Sequence[int], denominator: Sequence[int], prime: int
) -> int:
    """Return a degree bound invariant under every rational-mask transition."""

    if not _is_prime(prime):
        raise ValueError("prime must be prime")
    a = _trim(numerator, prime)
    b = _trim(denominator, prime)
    forcing = (len(a) - 1) + (prime - 2) * (len(b) - 1)
    quotient, remainder = divmod(forcing, prime - 1)
    return max(len(b) - 1, quotient + bool(remainder))


@dataclass(frozen=True)
class AutomatonCertificate:
    """Finite comparison between compiled and recurrence coefficients."""

    compared_terms: int
    failures: tuple[tuple[int, int, int], ...]

    @property
    def valid(self) -> bool:
        """Whether every comparison passed."""

        return not self.failures

    def require(self) -> None:
        """Raise on the first discrepancy."""

        if self.failures:
            raise ValidationError(f"automaton mismatch: {self.failures[0]}")


@dataclass(frozen=True)
class CartierAutomaton:
    """Finite output automaton for a rational-mask Mahler series modulo a prime."""

    prime: int
    numerator: tuple[int, ...]
    denominator: tuple[int, ...]
    degree_bound: int
    transitions: tuple[tuple[int, ...], ...]
    outputs: tuple[int, ...]
    state_numerators: tuple[tuple[int, ...], ...]
    state_members: tuple[tuple[int, ...], ...]
    start_state: int = 0

    @property
    def number_states(self) -> int:
        """Number of reachable states."""

        return len(self.transitions)

    @property
    def theoretical_state_bound(self) -> int:
        """Bound obtained from all numerator vectors of bounded degree."""

        return self.prime ** (self.degree_bound + 1)

    def evaluate(self, index: int) -> int:
        """Evaluate a coefficient from least-significant-first base-p digits."""

        if index < 0:
            raise ValueError("index must be nonnegative")
        state = self.start_state
        remaining = index
        while remaining:
            remaining, digit = divmod(remaining, self.prime)
            state = self.transitions[state][digit]
        return self.outputs[state]

    def sequence(self, count: int) -> tuple[int, ...]:
        """Return a finite coefficient prefix."""

        if count < 0:
            raise ValueError("count must be nonnegative")
        return tuple(self.evaluate(index) for index in range(count))

    def certificate(self, count: int) -> AutomatonCertificate:
        """Compare a finite prefix with the defining recurrence."""

        expected = rational_mask_coefficients_mod_prime(
            self.numerator, self.denominator, self.prime, count
        )
        failures = tuple(
            (index, self.evaluate(index), expected[index])
            for index in range(count)
            if self.evaluate(index) != expected[index]
        )
        return AutomatonCertificate(count, failures)


def _reachable_automaton(
    numerator: Sequence[int], denominator: Sequence[int], prime: int, max_states: int
) -> CartierAutomaton:
    if not _is_prime(prime):
        raise ValueError("prime must be prime")
    if max_states < 1:
        raise ValueError("max_states must be positive")
    a = _trim(numerator, prime)
    b = _trim(denominator, prime)
    if b[0] == 0 or a[0] != b[0]:
        raise ValueError("A(0)=B(0) must be nonzero modulo prime")
    degree_bound = cartier_degree_bound(a, b, prime)
    stabilizer = polynomial_multiply_mod(
        a, polynomial_power_mod(b, prime - 2, prime), prime
    )

    def pad(polynomial: Sequence[int]) -> tuple[int, ...]:
        values = list(_trim(polynomial, prime))
        if len(values) > degree_bound + 1:
            raise ArithmeticError("transition exceeded the proven degree bound")
        return tuple(values + [0] * (degree_bound + 1 - len(values)))

    def transition(state: Sequence[int], digit: int) -> tuple[int, ...]:
        product = polynomial_multiply_mod(_trim(state, prime), stabilizer, prime)
        return pad(polynomial_cartier(product, digit, prime))

    start = pad(b)
    states = [start]
    indices = {start: 0}
    rows: list[tuple[int, ...]] = []
    cursor = 0
    while cursor < len(states):
        row: list[int] = []
        for digit in range(prime):
            target = transition(states[cursor], digit)
            if target not in indices:
                if len(states) >= max_states:
                    raise RuntimeError(f"state guard exceeded: {max_states}")
                indices[target] = len(states)
                states.append(target)
            row.append(indices[target])
        rows.append(tuple(row))
        cursor += 1
    inverse_b0 = pow(b[0], -1, prime)
    outputs = tuple(state[0] * inverse_b0 % prime for state in states)
    return CartierAutomaton(
        prime,
        a,
        b,
        degree_bound,
        tuple(rows),
        outputs,
        tuple(states),
        tuple((index,) for index in range(len(states))),
    )


def minimize_cartier_automaton(automaton: CartierAutomaton) -> CartierAutomaton:
    """Minimize an output automaton by exact partition refinement."""

    count = automaton.number_states
    output_map: dict[int, int] = {}
    classes = []
    for output in automaton.outputs:
        output_map.setdefault(output, len(output_map))
        classes.append(output_map[output])
    while True:
        signatures: dict[tuple[int, tuple[int, ...]], int] = {}
        refined: list[int] = []
        for state in range(count):
            signature = (
                automaton.outputs[state],
                tuple(classes[target] for target in automaton.transitions[state]),
            )
            signatures.setdefault(signature, len(signatures))
            refined.append(signatures[signature])
        if refined == classes:
            break
        classes = refined
    block_count = max(classes) + 1
    blocks = [tuple(i for i, value in enumerate(classes) if value == block) for block in range(block_count)]
    raw_rows = []
    raw_outputs = []
    raw_representatives = []
    for block in blocks:
        representative = block[0]
        raw_rows.append(tuple(classes[target] for target in automaton.transitions[representative]))
        raw_outputs.append(automaton.outputs[representative])
        raw_representatives.append(automaton.state_numerators[representative])
    start_block = classes[automaton.start_state]
    order: list[int] = []
    queue = [start_block]
    seen = {start_block}
    while queue:
        block = queue.pop(0)
        order.append(block)
        for target in raw_rows[block]:
            if target not in seen:
                seen.add(target)
                queue.append(target)
    order.extend(block for block in range(block_count) if block not in seen)
    remap = {old: new for new, old in enumerate(order)}
    return CartierAutomaton(
        automaton.prime,
        automaton.numerator,
        automaton.denominator,
        automaton.degree_bound,
        tuple(tuple(remap[target] for target in raw_rows[old]) for old in order),
        tuple(raw_outputs[old] for old in order),
        tuple(raw_representatives[old] for old in order),
        tuple(
            tuple(
                member
                for original in blocks[old]
                for member in automaton.state_members[original]
            )
            for old in order
        ),
        0,
    )


def compile_rational_mask_automaton(
    numerator: Sequence[int],
    denominator: Sequence[int],
    prime: int,
    *,
    minimize: bool = True,
    max_states: int = 100_000,
    budget: GeneratingBudget = DEFAULT_GENERATING_BUDGET,
) -> CartierAutomaton:
    """Compile a rational Mahler mask into a coefficient automaton modulo ``prime``.

    The degree-derived linear dimension and the reachable-state guard are
    checked before state exploration begins.
    """

    bound = cartier_degree_bound(numerator, denominator, prime)
    budget.require_matrix_dimension(bound + 1)
    budget.require_states(max_states)
    automaton = _reachable_automaton(numerator, denominator, prime, max_states)
    return minimize_cartier_automaton(automaton) if minimize else automaton


def finite_mask_carry_matrices(
    mask: Sequence[int],
    prime: int,
    *,
    budget: GeneratingBudget = DEFAULT_GENERATING_BUDGET,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Return finite-mask carry matrices ``T_d(q,q')=r[d+p*q'-q]``."""

    if not _is_prime(prime):
        raise ValueError("prime must be prime")
    coefficients = _trim(mask, prime)
    bound = (len(coefficients) - 1 + prime - 2) // (prime - 1)
    budget.require_matrix_dimension(bound + 1)
    matrices = []
    for digit in range(prime):
        rows = []
        for source in range(bound + 1):
            row = []
            for target in range(bound + 1):
                index = digit + prime * target - source
                row.append(coefficients[index] if 0 <= index < len(coefficients) else 0)
            rows.append(tuple(value % prime for value in row))
        matrices.append(tuple(rows))
    return tuple(matrices)


@dataclass(frozen=True)
class CartierCarryEquivalence:
    """Finite certificate of Cartier/carry matrix equality."""

    prime: int
    dimension: int
    failures: tuple[tuple[int, int, int, int, int], ...]

    @property
    def valid(self) -> bool:
        """Whether every entry agreed."""

        return not self.failures

    def require(self) -> None:
        """Raise on the first mismatch."""

        if self.failures:
            raise ValidationError(f"Cartier/carry mismatch: {self.failures[0]}")


def certify_cartier_carry_equivalence(
    mask: Sequence[int],
    prime: int,
    *,
    budget: GeneratingBudget = DEFAULT_GENERATING_BUDGET,
) -> CartierCarryEquivalence:
    """Certify that finite-mask carry matrices are Cartier multiplication matrices."""

    coefficients = _trim(mask, prime)
    matrices = finite_mask_carry_matrices(coefficients, prime, budget=budget)
    dimension = len(matrices[0])
    failures = []
    for digit in range(prime):
        for source in range(dimension):
            basis = [0] * dimension
            basis[source] = 1
            transitioned = polynomial_cartier(
                polynomial_multiply_mod(basis, coefficients, prime), digit, prime
            )
            padded = list(transitioned) + [0] * (dimension - len(transitioned))
            for target in range(dimension):
                observed = padded[target] % prime
                expected = matrices[digit][source][target]
                if observed != expected:
                    failures.append((digit, source, target, observed, expected))
    return CartierCarryEquivalence(prime, dimension, tuple(failures))

@dataclass(frozen=True)
class CartierLinearRepresentation:
    """Bounded-dimensional linear representation of a rational-mask series."""

    prime: int
    numerator: tuple[int, ...]
    denominator: tuple[int, ...]
    degree_bound: int
    transition_matrices: tuple[tuple[tuple[int, ...], ...], ...]
    start_vector: tuple[int, ...]
    output_scale: int

    @property
    def dimension(self) -> int:
        """Linear-state dimension."""

        return self.degree_bound + 1

    def transition(self, vector: Sequence[int], digit: int) -> tuple[int, ...]:
        """Apply one least-significant-first digit transition."""

        if len(vector) != self.dimension:
            raise ValueError("vector has the wrong dimension")
        if not 0 <= digit < self.prime:
            raise ValueError("digit lies outside the radix")
        matrix = self.transition_matrices[digit]
        return tuple(
            sum(vector[source] * matrix[source][target] for source in range(self.dimension))
            % self.prime
            for target in range(self.dimension)
        )

    def evaluate(self, index: int) -> int:
        """Evaluate one coefficient without enumerating the reachable automaton."""

        if index < 0:
            raise ValueError("index must be nonnegative")
        vector = self.start_vector
        remaining = index
        while remaining:
            remaining, digit = divmod(remaining, self.prime)
            vector = self.transition(vector, digit)
        return vector[0] * self.output_scale % self.prime

    def sequence(self, count: int) -> tuple[int, ...]:
        """Return a finite coefficient prefix."""

        if count < 0:
            raise ValueError("count must be nonnegative")
        return tuple(self.evaluate(index) for index in range(count))


def compile_cartier_linear_representation(
    numerator: Sequence[int],
    denominator: Sequence[int],
    prime: int,
    *,
    budget: GeneratingBudget = DEFAULT_GENERATING_BUDGET,
) -> CartierLinearRepresentation:
    """Compile the denominator-stabilized Cartier transition matrices."""

    if not _is_prime(prime):
        raise ValueError("prime must be prime")
    a = _trim(numerator, prime)
    b = _trim(denominator, prime)
    if b[0] == 0 or a[0] != b[0]:
        raise ValueError("A(0)=B(0) must be nonzero modulo prime")
    degree_bound = cartier_degree_bound(a, b, prime)
    stabilizer = polynomial_multiply_mod(
        a, polynomial_power_mod(b, prime - 2, prime), prime
    )
    dimension = degree_bound + 1
    budget.require_matrix_dimension(dimension)
    matrices = []
    for digit in range(prime):
        rows = []
        for source in range(dimension):
            row = []
            for target in range(dimension):
                index = digit + prime * target - source
                row.append(stabilizer[index] if 0 <= index < len(stabilizer) else 0)
            rows.append(tuple(value % prime for value in row))
        matrices.append(tuple(rows))
    start = tuple(list(b) + [0] * (dimension - len(b)))
    return CartierLinearRepresentation(
        prime,
        a,
        b,
        degree_bound,
        tuple(matrices),
        start,
        pow(b[0], -1, prime),
    )
