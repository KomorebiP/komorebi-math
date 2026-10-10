"""Independent exact controls for finite Lehmer five-core and content machinery."""

from fractions import Fraction
from itertools import permutations, product, repeat
from math import factorial, gcd, isqrt, prod

import pytest

from komorebi_math.arithmetic.lehmer_shells import (
    FiveCoreShell,
    LehmerPacket,
    ShellLimits,
    ShellPolynomial,
    auxiliary_shell_polynomial,
    enumerate_five_core_shell,
    five_core_packet_count,
    primitive_residuals_from_tau,
    quadratic_character_five,
    tau_from_primitive_residuals,
    tau_from_q_product,
)
from komorebi_math.exceptions import ResourceLimitError, ValidationError


def _independent_four_coordinate_shell(index):
    bound = isqrt(10 * index)
    ranges = [tuple(x for x in range(-bound, bound + 1) if x % 5 == i) for i in range(4)]
    points = set()
    for v0, v1, v2, v3 in product(*ranges):
        v4 = -(v0 + v1 + v2 + v3)
        if v4 % 5 == 4 and v0**2 + v1**2 + v2**2 + v3**2 + v4**2 == 10 * index:
            points.add((v0, v1, v2, v3, v4))
    return points


def _integer_valued_binomial_determinant(coordinates):
    # A separate determinant expansion in the binomial basis from source (6.5).
    rows = [
        [Fraction(prod(x - k for k in range(j)), factorial(j)) for j in range(5)]
        for x in coordinates
    ]
    result = Fraction(0)
    for permutation in permutations(range(5)):
        inversions = sum(permutation[i] > permutation[j] for i in range(5) for j in range(i + 1, 5))
        result += (-1) ** inversions * prod(rows[i][permutation[i]] for i in range(5))
    assert result.denominator == 1
    return result.numerator


def _tau_logarithmic_recurrence(max_index):
    # n*a_n=-24*sum_(k=1)^n sigma_1(k)*a_(n-k), with tau(n)=a_(n-1).
    divisor_sums = [sum(d for d in range(1, k + 1) if k % d == 0) for k in range(max_index)]
    product_coefficients = [1]
    for degree in range(1, max_index):
        numerator = -24 * sum(
            divisor_sums[k] * product_coefficients[degree - k] for k in range(1, degree + 1)
        )
        value, remainder = divmod(numerator, degree)
        assert remainder == 0
        product_coefficients.append(value)
    return (0, *product_coefficients)


def test_complete_shells_against_independent_four_coordinate_control_through_32():
    for index in range(1, 33):
        shell = enumerate_five_core_shell(index)
        control = _independent_four_coordinate_shell(index)
        assert {packet.coordinates for packet in shell.packets} == control
        second = enumerate_five_core_shell(index, method="four_coordinate")
        assert shell.packets == second.packets
        assert len(control) == five_core_packet_count(index)
        assert len(control) == sum(
            quadratic_character_five(d) * (index // d) for d in range(1, index + 1) if index % d == 0
        )
        assert shell.certificate().valid


def test_all_signed_shell_sums_and_primitive_transforms_through_128():
    tau = tau_from_q_product(128)
    assert tau == _tau_logarithmic_recurrence(128)
    assert tau[:11] == (0, 1, -24, 252, -1472, 4830, -6048, -16744, 84480, -113643, -115920)
    residuals = primitive_residuals_from_tau(tau)
    assert tau_from_primitive_residuals(residuals) == tau
    for index in range(1, 129):
        shell = enumerate_five_core_shell(index)
        assert shell.weighted_sum == tau[index]
        assert shell.primitive_residual == residuals[index]
        assert sum(shell.content_masses.values()) == tau[index]
        for content, mass in shell.content_masses.items():
            assert index % content**2 == 0 and content % 5 != 0
            assert mass == quadratic_character_five(content) * content**10 * residuals[index // content**2]


def test_packet_weights_against_binomial_determinants_including_negative_coordinates():
    for index in (1, 2, 7, 9, 31):
        for packet in enumerate_five_core_shell(index).packets:
            assert packet.weight == _integer_valued_binomial_determinant(packet.coordinates)
            assert packet.weight != 0 and packet.weight % 5 == 1


def test_content_maps_are_bijective_and_restore_coordinate_residues():
    for index in (9, 16, 36, 49, 81, 100, 121, 128):
        shell = enumerate_five_core_shell(index)
        for packet in shell.packets:
            primitive = packet.primitive()
            content = gcd(*packet.coordinates)
            assert primitive.content == 1
            assert primitive.index * content**2 == index
            assert primitive.scale_content(content) == packet
            assert packet.weight == quadratic_character_five(content) * content**10 * primitive.weight
        for content in shell.content_masses:
            primitive_shell = enumerate_five_core_shell(index // content**2)
            scaled = {packet.scale_content(content) for packet in primitive_shell.packets if packet.content == 1}
            assert scaled == {packet for packet in shell.packets if packet.content == content}


def test_source_opposed_content_masses_at_121_and_cancellation_at_7():
    shell = enumerate_five_core_shell(121)
    assert shell.content_masses == {1: -25_439_104_668, 11: 25_937_424_601}
    assert shell.weighted_sum == 498_319_933
    assert shell.primitive_residual != shell.weighted_sum
    seven = enumerate_five_core_shell(7)
    weights = [packet.weight for packet in seven.packets]
    assert len(weights) == 6 and min(weights) < 0 < max(weights)
    assert sum(abs(weight) for weight in weights) > abs(sum(weights))
    assert not shell.certificate().diagnostics["universal_nonvanishing_established"]


def test_source_prime_index_unit_hypotheses_and_weighted_sum_divisibility():
    # The p=2411 example enters a regime where every packet is a p-unit but the
    # signed total is divisible by p. This checks cancellation, not more passes.
    shell = enumerate_five_core_shell(2411)
    assert len(shell.packets) == 2412
    certificate = shell.prime_unit_certificate()
    assert certificate.valid
    assert all(packet.weight % 2411 != 0 for packet in shell.packets)
    assert shell.weighted_sum % 2411 == 0 and shell.weighted_sum % 2411**2 != 0
    assert not certificate.diagnostics["nonvanishing_inferred_from_units"]
    for index in (2, 19, 25, 121):
        with pytest.raises(ValidationError, match="prime greater than 20"):
            enumerate_five_core_shell(index).prime_unit_certificate()


def test_weight_preserving_involution_and_prime_primitive_identity():
    for index in (1, 7, 31, 121):
        shell = enumerate_five_core_shell(index)
        points = set(shell.packets)
        for packet in shell.packets:
            image = packet.involution()
            assert image in points and image.involution() == packet
            assert image.weight == packet.weight and image.content == packet.content
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 31, 127):
        shell = enumerate_five_core_shell(prime)
        assert shell.primitive_residual == shell.weighted_sum
        assert shell.content_masses == {1: shell.weighted_sum}


def test_square_transform_is_linear_and_invertible_on_supplied_coefficients():
    left = (0, *((-1) ** n * (n * n + 3) for n in range(1, 81)))
    right = (0, *(n % 7 - 3 for n in range(1, 81)))
    transform_left = primitive_residuals_from_tau(left)
    transform_right = primitive_residuals_from_tau(right)
    assert primitive_residuals_from_tau(tuple(a + b for a, b in zip(left, right, strict=True))) == tuple(
        a + b for a, b in zip(transform_left, transform_right, strict=True)
    )
    assert tau_from_primitive_residuals(transform_left) == left
    # Exact zero input stays zero: no nonzero assertion is smuggled into inversion.
    assert primitive_residuals_from_tau((0,) * 81) == (0,) * 81


def test_auxiliary_polynomial_and_normalization_including_source_499_fixture():
    for index in (1, 7, 31, 121, 499):
        shell = enumerate_five_core_shell(index)
        polynomial = auxiliary_shell_polynomial(shell)
        assert polynomial.coefficients[0] == 288 * len(shell.packets)
        assert polynomial.evaluate(5) == 288 * shell.weighted_sum
        normalized = polynomial.normalized_coefficients
        assert any(coefficient % 5 for coefficient in normalized)
        assert polynomial.evaluate(1, normalized=True) * 5**polynomial.five_adic_content == polynomial.evaluate(5)
        # Independent direct packet products at several auxiliary arguments.
        for argument in (-3, 0, 1, 6):
            direct = sum(
                prod((i - j) + argument * ((packet.coordinates[i] - i) // 5
                                         - (packet.coordinates[j] - j) // 5)
                     for i in range(5) for j in range(i + 1, 5))
                for packet in shell.packets
            )
            assert polynomial.evaluate(argument) == direct
    assert polynomial.coefficients == (
        144000, -360000, 1063040, 28418240, -148872808, 209474504,
        2422451240, 16445000, 8107402848, -11893316304, -1160632800,
    )
    assert polynomial.five_adic_content == 3
    assert tuple(c % 5 for c in polynomial.normalized_coefficients) == (2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0)
    assert polynomial.evaluate(1, normalized=True) % 25 == 10
    assert polynomial.evaluate(16, normalized=True) % 25 == 0
    assert sum(degree * coefficient for degree, coefficient in enumerate(polynomial.normalized_coefficients)) % 5 == 1
    for malformed in (repeat(1), ("wrong",) * 11, (0,) * 11, (1, 2)):
        with pytest.raises(ValidationError):
            ShellPolynomial(malformed)


@pytest.mark.parametrize("index", [0, -1, 1.5, True, "7"])
def test_bad_shell_indices(index):
    with pytest.raises(ValidationError):
        enumerate_five_core_shell(index)


def test_packet_validation_and_owned_coordinates():
    coordinates = [0, 1, 2, -2, -1]
    packet = LehmerPacket(coordinates)
    coordinates[0] = 100
    assert packet.coordinates == (0, 1, 2, -2, -1)
    assert packet.index == packet.weight == packet.content == 1
    for wrong in ([], [0, 1, 2, 3, 4], [0, 1, 2, -2, -1, 5], [0.0, 1, 2, -2, -1], repeat(0)):
        with pytest.raises(ValidationError):
            LehmerPacket(wrong)
    with pytest.raises(ValidationError, match="coprime"):
        packet.scale_content(5)
    with pytest.raises(ValidationError, match="cardinality"):
        FiveCoreShell(1, (packet, packet), "quadratic", 2)
    with pytest.raises(ValidationError, match="cardinality"):
        FiveCoreShell(1, (), "quadratic", 1)


def test_shell_constructor_rejects_unbounded_iterators_before_consumption():
    packet = LehmerPacket((0, 1, 2, -2, -1))
    infinite = repeat(packet)
    with pytest.raises(ValidationError, match="finite tuple or list"):
        FiveCoreShell(1, infinite, "quadratic", 1)
    assert next(infinite) is packet
    with pytest.raises(ValidationError, match="cardinality"):
        FiveCoreShell(1, [packet] * 100, "quadratic", 1)
    owned = [packet]
    shell = FiveCoreShell(1, owned, "quadratic", 1)
    owned.clear()
    assert shell.packets == (packet,)
    seven = enumerate_five_core_shell(7)
    duplicates = [seven.packets[0]] * len(seven.packets)
    with pytest.raises(ValidationError, match="duplicate"):
        FiveCoreShell(7, duplicates, "quadratic", 1)


def test_shell_constructor_preflights_index_arithmetic_before_packet_copy():
    packet = LehmerPacket((0, 1, 2, -2, -1))
    with pytest.raises(ResourceLimitError, match="max_divisor_trials"):
        FiveCoreShell(10**100, [packet], "quadratic", 1)
    with pytest.raises(ResourceLimitError, match="integer bit limit"):
        FiveCoreShell(1 << 4096, [packet], "quadratic", 1)


def test_resource_preflight_and_no_partial_results():
    with pytest.raises(ResourceLimitError, match="candidate"):
        enumerate_five_core_shell(10**100)
    with pytest.raises(ResourceLimitError, match="max_packets"):
        enumerate_five_core_shell(31, limits=ShellLimits(max_packets=1))
    with pytest.raises(ResourceLimitError, match="derived"):
        enumerate_five_core_shell(31, limits=ShellLimits(max_integer_bits=20))
    with pytest.raises(ResourceLimitError, match="candidate"):
        enumerate_five_core_shell(31, method="four_coordinate", limits=ShellLimits(max_candidate_tuples=1))
    with pytest.raises(ResourceLimitError, match="max_updates"):
        tau_from_q_product(10**100)
    with pytest.raises(ResourceLimitError, match="derived"):
        tau_from_q_product(5, max_integer_bits=10)
    with pytest.raises(ResourceLimitError, match="max_indices"):
        primitive_residuals_from_tau(repeat(0), max_indices=10)
    with pytest.raises(ResourceLimitError, match="derived"):
        primitive_residuals_from_tau((0, 1, 2, 3), max_integer_bits=8)
    with pytest.raises(ValidationError, match="zero at index zero"):
        primitive_residuals_from_tau((1, 2, 3))
    with pytest.raises(ValidationError):
        primitive_residuals_from_tau((0, 1.5))
    with pytest.raises(ValidationError):
        enumerate_five_core_shell(1, method="sample")
    assert tau_from_q_product(0) == primitive_residuals_from_tau((0,)) == (0,)
