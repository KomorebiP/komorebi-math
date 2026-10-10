"""Exact, bounded five-core shells and primitive content for Ramanujan tau.

The conventions are those of the Lehmer reader edition 1.0, Chapters 6, 7 and
13: coordinate residues are indexed 0 through 4, the norm is ``10*n``, and
the *signed* Vandermonde is divided by 288. Complete finite enumeration does
not establish Lehmer's conjecture or universal nonvanishing.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice, product
from math import gcd, isqrt, prod
from numbers import Integral

from ..certificates import Certificate, Exactness
from ..exceptions import ResourceLimitError, ValidationError


def _integer(value: object, name: str, minimum: int = 0, *, bits: int = 32768) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValidationError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValidationError(f"{name} must be at least {minimum}")
    if abs(result).bit_length() > bits:
        raise ResourceLimitError(f"{name} exceeds the integer bit limit")
    return result


def quadratic_character_five(value: int) -> int:
    """Return chi_5 on any integral argument, including negative arguments."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValidationError("character argument must be an integer")
    residue = int(value) % 5
    return 0 if residue == 0 else (1 if residue in (1, 4) else -1)


@dataclass(frozen=True, slots=True)
class ShellLimits:
    """Preflight limits; exceeding one raises instead of returning a partial shell."""

    max_candidate_tuples: int = 2_000_000
    max_packets: int = 100_000
    max_integer_bits: int = 4096

    def __post_init__(self) -> None:
        for name in ("max_candidate_tuples", "max_packets", "max_integer_bits"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1))


@dataclass(frozen=True, slots=True)
class LehmerPacket:
    """One validated standard-residue shell point with a nonzero integral weight."""

    coordinates: tuple[int, ...] | Iterable[int]

    def __post_init__(self) -> None:
        try:
            coordinates = tuple(islice(iter(self.coordinates), 6))
        except TypeError as exc:
            raise ValidationError("coordinates must be an iterable of five integers") from exc
        if len(coordinates) != 5:
            raise ValidationError("a packet must have exactly five coordinates")
        for i, coordinate in enumerate(coordinates):
            if isinstance(coordinate, bool) or not isinstance(coordinate, Integral):
                raise ValidationError("packet coordinates must be integers")
            if abs(int(coordinate)).bit_length() > 4096:
                raise ResourceLimitError("packet coordinate exceeds the integer bit limit")
            if int(coordinate) % 5 != i:
                raise ValidationError("packet coordinate residues must be i modulo 5")
        owned = tuple(int(coordinate) for coordinate in coordinates)
        if sum(owned) != 0:
            raise ValidationError("packet coordinates must sum to zero")
        norm = sum(coordinate * coordinate for coordinate in owned)
        if norm == 0 or norm % 10:
            raise ValidationError("packet squared norm must be a positive multiple of 10")
        object.__setattr__(self, "coordinates", owned)

    @property
    def index(self) -> int:
        return sum(coordinate * coordinate for coordinate in self.coordinates) // 10

    @property
    def weight(self) -> int:
        numerator = prod(
            self.coordinates[i] - self.coordinates[j] for i in range(5) for j in range(i + 1, 5)
        )
        weight, remainder = divmod(numerator, 288)
        if remainder or weight == 0:
            raise ValidationError("Vandermonde weight must be nonzero and integral")
        return weight

    @property
    def content(self) -> int:
        return gcd(*self.coordinates)

    def primitive(self) -> LehmerPacket:
        """Divide by the coordinate gcd and restore residues by source (13.1)."""
        content = self.content
        return LehmerPacket(tuple(self.coordinates[(content * j) % 5] // content for j in range(5)))

    def scale_content(self, content: int) -> LehmerPacket:
        """Inverse residue transport; requires positive content coprime to five."""
        content = _integer(content, "content", 1, bits=4096)
        if content % 5 == 0:
            raise ValidationError("packet content must be coprime to five")
        inverse = pow(content, -1, 5)
        # Validate prospective output size before doing large integer products.
        if max(abs(x).bit_length() for x in self.coordinates) + content.bit_length() > 4096:
            raise ResourceLimitError("scaled packet exceeds the integer bit limit")
        return LehmerPacket(tuple(content * self.coordinates[(inverse * i) % 5] for i in range(5)))

    def involution(self) -> LehmerPacket:
        """The weight-preserving involution J(v)_i = -v_(-i modulo 5)."""
        return LehmerPacket(tuple(-self.coordinates[(-i) % 5] for i in range(5)))


def five_core_packet_count(index: int, *, max_divisor_trials: int = 1_000_000) -> int:
    """Source (6.7): ``sum_(d|n) chi5(d)*(n//d)``; not a weighted sum."""
    index = _integer(index, "index", 1)
    limit = _integer(max_divisor_trials, "max_divisor_trials", 1)
    root = isqrt(index)
    if root > limit:
        raise ResourceLimitError("packet count exceeds max_divisor_trials")
    result = 0
    for divisor in range(1, root + 1):
        if index % divisor == 0:
            other = index // divisor
            result += quadratic_character_five(divisor) * other
            if divisor != other:
                result += quadratic_character_five(other) * divisor
    return result


@dataclass(frozen=True, slots=True)
class FiveCoreShell:
    """Owned complete finite shell returned after enumeration and count checks."""

    index: int
    packets: tuple[LehmerPacket, ...]
    algorithm: str
    candidate_upper_bound: int

    def __post_init__(self) -> None:
        index = _integer(self.index, "index", 1, bits=4096)
        if self.algorithm not in ("quadratic", "four_coordinate"):
            raise ValidationError("shell algorithm must be quadratic or four_coordinate")
        _integer(self.candidate_upper_bound, "candidate_upper_bound", 0)
        if not isinstance(self.packets, (tuple, list)):
            raise ValidationError("shell packets must be an explicit finite tuple or list")
        expected_count = five_core_packet_count(index)
        if len(self.packets) != expected_count:
            raise ValidationError("shell cardinality disagrees with the complete five-core count")
        packets = tuple(self.packets)
        if any(not isinstance(packet, LehmerPacket) or packet.index != index for packet in packets):
            raise ValidationError("every packet must belong to the specified shell")
        if len({packet.coordinates for packet in packets}) != len(packets):
            raise ValidationError("a complete shell cannot contain duplicate packets")
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "packets", packets)

    @property
    def weighted_sum(self) -> int:
        return sum(packet.weight for packet in self.packets)

    @property
    def primitive_residual(self) -> int:
        return sum(packet.weight for packet in self.packets if packet.content == 1)

    @property
    def content_masses(self) -> dict[int, int]:
        """Return fresh exact-content signed sums; cancellation is retained."""
        masses: dict[int, int] = {}
        for packet in self.packets:
            masses[packet.content] = masses.get(packet.content, 0) + packet.weight
        return dict(sorted(masses.items()))

    def certificate(self) -> Certificate:
        return Certificate(
            valid=True,
            exactness=Exactness.EXACT,
            assumptions_checked=(
                "complete finite standard-residue shell",
                "exact integer Vandermonde weights and source five-core count",
            ),
            diagnostics={
                "index": self.index,
                "packet_count": len(self.packets),
                "candidate_upper_bound": self.candidate_upper_bound,
                "algorithm": self.algorithm,
                "weighted_sum": self.weighted_sum,
                "primitive_residual": self.primitive_residual,
                "content_masses": self.content_masses,
                "positive_packets": sum(packet.weight > 0 for packet in self.packets),
                "negative_packets": sum(packet.weight < 0 for packet in self.packets),
                "universal_nonvanishing_established": False,
            },
            message="Exact complete finite shell; Lehmer nonvanishing remains unestablished.",
        )

    def prime_unit_certificate(self) -> Certificate:
        """Check the source's termwise index-prime property, only for primes >20."""
        if self.index <= 20 or any(self.index % d == 0 for d in range(2, isqrt(self.index) + 1)):
            raise ValidationError("the source index-prime unit hypothesis requires a prime greater than 20")
        units = all(packet.weight % self.index for packet in self.packets)
        return Certificate(
            valid=units,
            exactness=Exactness.EXACT,
            assumptions_checked=("index is a prime greater than 20",),
            diagnostics={
                "index": self.index,
                "packet_count": len(self.packets),
                "all_weights_are_index_prime_units": units,
                "weighted_sum": self.weighted_sum,
                "weighted_sum_residue": self.weighted_sum % self.index,
                "nonvanishing_inferred_from_units": False,
            },
            message="Termwise unit check; opposite signs and sum divisibility remain possible.",
        )


def enumerate_five_core_shell(
    index: int, *, method: str = "quadratic", limits: ShellLimits | None = None
) -> FiveCoreShell:
    """Enumerate every V-coded packet, using exact squares or a four-coordinate loop.

    Candidate and output bounds are checked before allocating coordinate ranges.
    Both methods use exact integer arithmetic; no sampling or truncation occurs.
    """
    if limits is None:
        limits = ShellLimits()
    if not isinstance(limits, ShellLimits):
        raise ValidationError("limits must be a ShellLimits instance")
    index = _integer(index, "index", 1, bits=limits.max_integer_bits)
    if method not in ("quadratic", "four_coordinate"):
        raise ValidationError("method must be quadratic or four_coordinate")
    bound = isqrt(10 * index)
    starts = tuple(-bound + (i + bound) % 5 for i in range(5))
    counts = tuple(max(0, (bound - start) // 5 + 1) for start in starts)
    coordinate_count = 3 if method == "quadratic" else 4
    candidates = prod(counts[:coordinate_count])
    if candidates > limits.max_candidate_tuples:
        raise ResourceLimitError("complete shell enumeration exceeds max_candidate_tuples")
    if bound.bit_length() > 4096:
        raise ResourceLimitError("packet coordinates exceed the integer bit limit")
    packet_count = five_core_packet_count(index, max_divisor_trials=max(1, bound))
    if packet_count > limits.max_packets:
        raise ResourceLimitError("complete shell output exceeds max_packets")
    # A sum of at most packet_count products of ten differences is bounded here.
    if 10 * (2 * bound).bit_length() + packet_count.bit_length() > limits.max_integer_bits:
        raise ResourceLimitError("derived shell arithmetic exceeds max_integer_bits")
    ranges = tuple(range(starts[i], bound + 1, 5) for i in range(coordinate_count))
    packets = []
    norm = 10 * index
    if method == "four_coordinate":
        for prefix in product(*ranges):
            last = -sum(prefix)
            if last % 5 == 4 and sum(x * x for x in prefix) + last * last == norm:
                packets.append(LehmerPacket((*prefix, last)))
    else:
        for prefix in product(*ranges):
            remaining_sum = -sum(prefix)
            remaining_norm = norm - sum(x * x for x in prefix)
            discriminant = 2 * remaining_norm - remaining_sum * remaining_sum
            if discriminant < 0:
                continue
            root = isqrt(discriminant)
            if root * root != discriminant:
                continue
            for difference in ((root, -root) if root else (0,)):
                if (remaining_sum + difference) % 2:
                    continue
                x = (remaining_sum + difference) // 2
                y = remaining_sum - x
                if x % 5 == 3 and y % 5 == 4:
                    packets.append(LehmerPacket((*prefix, x, y)))
    packets.sort(key=lambda packet: packet.coordinates)
    return FiveCoreShell(index, tuple(packets), method, candidates)


@dataclass(frozen=True, slots=True)
class ShellPolynomial:
    """One fixed shell's degree-at-most-ten auxiliary polynomial and 5-adic content.

    Coefficients are in increasing degree order. The variable is the source's
    auxiliary T, not the shell index; normalization substitutes T=5*U.
    """

    coefficients: tuple[int, ...]

    def __post_init__(self) -> None:
        try:
            coefficients = tuple(islice(iter(self.coefficients), 12))
        except TypeError as exc:
            raise ValidationError("shell polynomial coefficients must be an iterable of integers") from exc
        if len(coefficients) != 11:
            raise ValidationError("shell polynomial requires eleven coefficients")
        if any(isinstance(c, bool) or not isinstance(c, Integral) for c in coefficients):
            raise ValidationError("shell polynomial coefficients must be integers")
        if coefficients[0] <= 0:
            raise ValidationError("shell polynomial requires a positive constant term")
        if any(abs(int(c)).bit_length() > 32768 for c in coefficients):
            raise ResourceLimitError("shell polynomial coefficients exceed the integer bit limit")
        object.__setattr__(self, "coefficients", tuple(int(c) for c in coefficients))

    @property
    def five_adic_content(self) -> int:
        weighted_valuations = []
        for degree, coefficient in enumerate(self.coefficients):
            if coefficient:
                valuation = 0
                while coefficient % 5 == 0:
                    coefficient //= 5
                    valuation += 1
                weighted_valuations.append(degree + valuation)
        return min(weighted_valuations)

    @property
    def normalized_coefficients(self) -> tuple[int, ...]:
        divisor = 5**self.five_adic_content
        return tuple(coefficient * 5**degree // divisor for degree, coefficient in enumerate(self.coefficients))

    def evaluate(self, value: int, *, normalized: bool = False) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValidationError("polynomial argument must be an integer")
        if abs(int(value)).bit_length() > 4096:
            raise ResourceLimitError("polynomial argument exceeds the integer bit limit")
        result = 0
        coefficients = self.normalized_coefficients if normalized else self.coefficients
        for coefficient in reversed(coefficients):
            result = result * int(value) + coefficient
        return result


def auxiliary_shell_polynomial(shell: FiveCoreShell) -> ShellPolynomial:
    """Build source (7.1) P_n(T) by exact products of ten linear factors."""
    if not isinstance(shell, FiveCoreShell):
        raise ValidationError("shell must be a complete FiveCoreShell")
    bound = max(abs(coordinate) for packet in shell.packets for coordinate in packet.coordinates)
    if 10 * (2 * bound).bit_length() + 10 + len(shell.packets).bit_length() > 32768:
        raise ResourceLimitError("derived shell polynomial exceeds the integer bit limit")
    aggregate = [0] * 11
    for packet in shell.packets:
        c = tuple((coordinate - i) // 5 for i, coordinate in enumerate(packet.coordinates))
        coefficients = [1]
        for i in range(5):
            for j in range(i + 1, 5):
                next_coefficients = [0] * (len(coefficients) + 1)
                for degree, coefficient in enumerate(coefficients):
                    next_coefficients[degree] += coefficient * (i - j)
                    next_coefficients[degree + 1] += coefficient * (c[i] - c[j])
                coefficients = next_coefficients
        for degree, coefficient in enumerate(coefficients):
            aggregate[degree] += coefficient
    return ShellPolynomial(tuple(aggregate))


def tau_from_q_product(
    max_index: int, *, max_updates: int = 5_000_000, max_integer_bits: int = 32768
) -> tuple[int, ...]:
    """Independent truncated ``q*prod(1-q**m)**24`` coefficients, starting at tau(0).

    Multiplication is by 24 individual binomials for each m. Exactly the
    degree <=max_index part is retained; factors m>=max_index cannot affect it.
    """
    bits = _integer(max_integer_bits, "max_integer_bits", 1)
    max_index = _integer(max_index, "max_index", bits=bits)
    updates = _integer(max_updates, "max_updates", 1)
    required_updates = 12 * max_index * max(0, max_index - 1)
    if required_updates > updates:
        raise ResourceLimitError("truncated q product exceeds max_updates")
    if 24 * max(0, max_index - 1) + 1 > bits:
        raise ResourceLimitError("derived q-product coefficients exceed max_integer_bits")
    if max_index == 0:
        return (0,)
    coefficients = [1] + [0] * (max_index - 1)
    for exponent in range(1, max_index):
        for _ in range(24):
            for degree in range(max_index - 1, exponent - 1, -1):
                coefficients[degree] -= coefficients[degree - exponent]
    return (0, *coefficients)


def _mobius(value: int) -> int:
    result = 1
    prime = 2
    while prime * prime <= value:
        if value % prime == 0:
            value //= prime
            result = -result
            if value % prime == 0:
                return 0
        prime += 1
    return -result if value > 1 else result


def _square_transform(
    coefficients: Iterable[int], *, inverse: bool, max_indices: int, max_integer_bits: int
) -> tuple[int, ...]:
    limit = _integer(max_indices, "max_indices", 1)
    bits = _integer(max_integer_bits, "max_integer_bits", 1)
    try:
        iterator = iter(coefficients)
    except TypeError as exc:
        raise ValidationError("coefficients must be an iterable of integers") from exc
    values = []
    for value in iterator:
        if len(values) > limit:
            raise ResourceLimitError("square transform exceeds max_indices")
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValidationError("coefficients must be integers")
        if abs(int(value)).bit_length() > bits:
            raise ResourceLimitError("coefficient exceeds max_integer_bits")
        values.append(int(value))
    if not values or values[0] != 0:
        raise ValidationError("coefficients must include a zero at index zero")
    degree = len(values) - 1
    if max((abs(x).bit_length() for x in values), default=0) + 11 * max(1, degree).bit_length() > bits:
        raise ResourceLimitError("derived square-transform coefficients exceed max_integer_bits")
    output = [0] * len(values)
    for index in range(1, len(values)):
        total = 0
        for divisor in range(1, isqrt(index) + 1):
            square = divisor * divisor
            if index % square == 0:
                character = quadratic_character_five(divisor)
                multiplier = (_mobius(divisor) if inverse else 1) * character * divisor**10
                total += multiplier * values[index // square]
        output[index] = total
    return tuple(output)


def primitive_residuals_from_tau(
    coefficients: Iterable[int], *, max_indices: int = 10_000, max_integer_bits: int = 32768
) -> tuple[int, ...]:
    """Finite source (13.5) transform R(n)=sum mu(d)chi5(d)d^10*tau(n/d^2)."""
    return _square_transform(
        coefficients, inverse=True, max_indices=max_indices, max_integer_bits=max_integer_bits
    )


def tau_from_primitive_residuals(
    coefficients: Iterable[int], *, max_indices: int = 10_000, max_integer_bits: int = 32768
) -> tuple[int, ...]:
    """Finite inverse source (13.4); no nonzero coefficient is assumed or inferred."""
    return _square_transform(
        coefficients, inverse=False, max_indices=max_indices, max_integer_bits=max_integer_bits
    )
