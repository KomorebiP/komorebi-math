"""Independent finite controls for the Paper IX--XII executable boundary."""

from fractions import Fraction
from itertools import product
from math import comb, factorial, lcm

import numpy as np
import pytest
import sympy as sp

from komorebi_math.analytic.appell import congruential_inverse
from komorebi_math.arithmetic.euler import bernoulli_numbers, signed_euler_numbers
from komorebi_math.arithmetic.packet_differences import (
    certify_packet_difference,
    rational_mixed_difference,
)
from komorebi_math.arithmetic.polynomial_lifting import (
    IntegerPolynomialSystem,
    SparseIntegerPolynomial,
)
from komorebi_math.arithmetic.valuation import padic_valuation
from komorebi_math.deformation.support import nested_subspace_occupancy
from komorebi_math.exceptions import DimensionError, ValidationError
from komorebi_math.langlands.local.tame import FrobeniusOrbit
from komorebi_math.langlands.local.tame_tensor import tame_tensor_character_packets


def _polynomial(variables, terms):
    return SparseIntegerPolynomial(variables, terms)


def _brute_lifts(system, root, prime, depth, fixed=None):
    scale = prime**depth
    modulus = scale * prime
    result = set()
    for digits in product(range(prime), repeat=system.variable_count):
        if any(digits[index] != value for index, value in (fixed or {}).items()):
            continue
        point = tuple((value + scale * digit) % modulus for value, digit in zip(root, digits, strict=True))
        if all(value % modulus == 0 for value in system.evaluate(point)):
            result.add(point)
    return result


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_polynomial_lift_fibers_against_direct_evaluation(prime):
    """Quadratic rectangular, square, and singular systems at two depths."""
    fixtures = (
        IntegerPolynomialSystem((_polynomial(1, {(2,): 1, (0,): -1}),)),
        IntegerPolynomialSystem((_polynomial(2, {(1, 1): 1}),)),
        IntegerPolynomialSystem((_polynomial(2, {(2, 0): 1, (0, 1): -1}),)),
        IntegerPolynomialSystem(
            (
                _polynomial(2, {(2, 0): 1, (0, 1): -1}),
                _polynomial(2, {(1, 1): 1, (0, 0): -1}),
            )
        ),
    )
    for system in fixtures:
        for depth in (1, 2):
            modulus = prime**depth
            for root in product(range(modulus), repeat=system.variable_count):
                if any(value % modulus for value in system.evaluate(root)):
                    continue
                fiber = system.lift_fiber(root, prime=prime, depth=depth)
                expected = _brute_lifts(system, root, prime, depth)
                assert set(fiber.lifts()) == expected
                assert fiber.count == len(expected)
                assert fiber.certificate.valid == bool(expected)
                for digit in fiber.correction_digits():
                    assert all(
                        (
                            residual
                            + sum(value * correction for value, correction in zip(row, digit, strict=True))
                        )
                        % prime
                        == 0
                        for row, residual in zip(fiber.jacobian, fiber.residual, strict=True)
                    )


def test_shared_correction_detects_separate_but_incompatible_lifts():
    first = IntegerPolynomialSystem((_polynomial(1, {(1,): 1}),))
    second = IntegerPolynomialSystem((_polynomial(1, {(1,): 1, (0,): -2}),))
    assert first.lift_fiber((0,), prime=2, depth=1).lifts() == ((0,),)
    assert second.lift_fiber((0,), prime=2, depth=1).lifts() == ((2,),)
    joint = first.stack(second).lift_fiber((0,), prime=2, depth=1)
    assert not joint.solvable and joint.count == 0
    assert joint.lifts() == ()
    with pytest.raises(ValidationError):
        joint.certificate.require()


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_prescribed_parameter_fibers_against_direct_evaluation(prime):
    # Full Jacobian (0, 1) is onto; the solved block is zero.
    system = IntegerPolynomialSystem((_polynomial(2, {(0, 1): 1}),))
    for digit in range(prime):
        fixed = {1: digit}
        fiber = system.lift_fiber((0, 0), prime=prime, depth=1, fixed_digits=fixed)
        assert set(fiber.lifts()) == _brute_lifts(system, (0, 0), prime, 1, fixed)
        assert fiber.count == (prime if digit == 0 else 0)
        assert fiber.solved_jacobian_rank == 0
        assert fiber.obstruction_dimension == 1
    all_fixed = system.lift_fiber((0, 0), prime=prime, depth=1, fixed_digits={0: 1, 1: 0})
    assert all_fixed.lifts() == ((prime, 0),)
    assert all_fixed.kernel_basis == ()


def test_singular_success_does_not_imply_a_later_lift():
    system = IntegerPolynomialSystem((_polynomial(1, {(2,): 1, (0,): -5}),))
    first = system.lift_fiber((1,), prime=2, depth=1)
    assert set(first.lifts()) == {(1,), (3,)}
    for root in first.lifts():
        assert not system.lift_fiber(root, prime=2, depth=2).solvable
    with pytest.raises(ValidationError, match="obstruction"):
        system.finite_tower((1,), prime=2, depth=1, target_depth=3)


@pytest.mark.parametrize("prime", [3, 5, 7])
def test_smooth_finite_tower_retains_exact_residue_compatibility(prime):
    system = IntegerPolynomialSystem((_polynomial(1, {(2,): 1, (0,): -1}),))
    tower = system.finite_tower((prime - 1,), prime=prime, depth=1, target_depth=12)
    assert len(tower) == 12
    for depth, root in enumerate(tower, start=1):
        assert root == (prime**depth - 1,)
        assert system.evaluate(root)[0] % (prime**depth) == 0
        if depth > 1:
            assert root[0] % (prime ** (depth - 1)) == tower[depth - 2][0]


def test_lift_snapshot_representatives_and_resource_caps():
    terms = {(1, 0): 1}
    system = IntegerPolynomialSystem((_polynomial(2, terms),))
    terms[(1, 0)] = 2
    assert system.evaluate((3, 9)) == (3,)
    fiber = system.lift_fiber((0, 0), prime=5, depth=2)
    translated = system.lift_fiber((25, -50), prime=5, depth=2)
    assert set(fiber.lifts()) == set(translated.lifts())
    assert fiber.count == 5
    with pytest.raises(ValidationError, match="max_count"):
        fiber.lifts(max_count=4)
    with pytest.raises(ValidationError, match="max_steps"):
        system.finite_tower((0, 0), prime=5, depth=1, target_depth=4, max_steps=2)
    huge = IntegerPolynomialSystem((_polynomial(1, {(1,): 2**150 + 1, (0,): -(2**150 + 1)}),))
    assert huge.lift_fiber((1,), prime=3, depth=4).lifts() == ((1,),)


@pytest.mark.parametrize(
    "prime,depth,root,fixed",
    [
        (4, 1, (0,), None),
        (2, 0, (0,), None),
        (2, True, (0,), None),
        (2, 1, (1,), None),
        (2, 1, (0.0,), None),
        (2, 1, (0,), {1: 0}),
        (2, 1, (0,), {0: 2}),
    ],
)
def test_lift_domain_controls(prime, depth, root, fixed):
    system = IntegerPolynomialSystem((_polynomial(1, {(1,): 1}),))
    with pytest.raises(ValidationError):
        system.lift_fiber(root, prime=prime, depth=depth, fixed_digits=fixed)


def test_sparse_polynomial_domains():
    with pytest.raises(ValidationError):
        _polynomial(1, {(-1,): 1})
    with pytest.raises(ValidationError):
        _polynomial(1, {(1,): Fraction(1, 2)})
    with pytest.raises(DimensionError):
        _polynomial(2, {(1,): 1})
    with pytest.raises(DimensionError):
        IntegerPolynomialSystem((_polynomial(1, {}), _polynomial(2, {})))


def _triangular_packet(width, address, degree):
    coefficients = [Fraction(1)]
    for row in range(1, degree + 1):
        numerator = sum(
            Fraction(comb(width * row + address, width * previous)) * coefficients[previous]
            for previous in range(row)
        )
        coefficients.append(-numerator / comb(width * row + address, width * row))
    return tuple(coefficients)


def test_all_address_euler_packets_against_independent_triangular_recurrence():
    for width in range(1, 9):
        for address in range(width):
            coefficients = congruential_inverse(width, address, 16)
            assert coefficients == _triangular_packet(width, address, 16)
            clearing = 1
            for row, coefficient in enumerate(coefficients):
                if row:
                    clearing *= comb(width * row + address, width * row)
                assert (clearing * coefficient).denominator == 1


def test_euler_packet_classical_normalization_and_bernoulli_boundary():
    assert congruential_inverse(2, 0, 16) == signed_euler_numbers(32)[::2]
    bernoulli = bernoulli_numbers(32)
    assert congruential_inverse(2, 1, 16) == tuple(
        (2 - 2 ** (2 * row)) * bernoulli[2 * row] for row in range(17)
    )
    # Normalized C(0) is one, whereas the original coefficient U(0) is j!.
    for address in range(4):
        normalized = congruential_inverse(4, address, 4)
        assert factorial(address) * normalized[0] == factorial(address)


def test_four_dyadic_displacement_and_difference_controls():
    for address, gain in enumerate((1, 1, 4, 2)):
        coefficients = congruential_inverse(4, address, 40)
        original_gain = gain + padic_valuation(factorial(address), 2)
        for row in range(5):
            for step in (1, 2, 3, 4, 7, 8, 15, 16, 17):
                difference = coefficients[row + step] - coefficients[row]
                assert difference != 0
                assert padic_valuation(difference, 2) == padic_valuation(step, 2) + gain
                assert (
                    padic_valuation(factorial(address) * difference, 2)
                    == padic_valuation(step, 2) + original_gain
                )
            for order in range(1, 5):
                for step in (0, 1, 2, 3, 4, 8):
                    difference = sum(
                        (-1) ** (order - index) * comb(order, index) * coefficients[row + step * index]
                        for index in range(order + 1)
                    )
                    step_valuation = 0 if step == 0 else padic_valuation(step, 2)
                    assert padic_valuation(difference, 2) >= order * (1 + step_valuation)


def _mixed_difference(coefficients, row, steps):
    return rational_mixed_difference(coefficients, row, steps)


def _direct_mixed_difference(coefficients, row, steps):
    value = Fraction(0)
    for bits in product((0, 1), repeat=len(steps)):
        value += (-1) ** (len(steps) - sum(bits)) * coefficients[
            row + sum(bit * step for bit, step in zip(bits, steps, strict=True))
        ]
    return value


def test_rational_mixed_stencils_against_independent_subset_expansion():
    coefficients = tuple(Fraction((-1) ** row * (row**3 + 1), 2 * row + 1) for row in range(12))
    for order in range(5):
        for steps in product(range(3), repeat=order):
            for row in (0, 2):
                assert rational_mixed_difference(coefficients, row, steps) == _direct_mixed_difference(
                    coefficients, row, steps
                )
    assert rational_mixed_difference((Fraction(1, 8),), 0, ()) == Fraction(1, 8)
    assert rational_mixed_difference((Fraction(1, 8),), 0, (0,)) == 0


def test_rational_stencil_evidence_caps_and_false_bound_rejection():
    coefficients = congruential_inverse(4, 1, 8)
    passing = certify_packet_difference(coefficients, 0, (2, 2), prime=2, valuation_bound=5)
    passing.certificate.require()
    assert passing.certificate.diagnostics["finite_statement_only"]
    failing = certify_packet_difference(coefficients, 0, (2, 2), prime=2, valuation_bound=6)
    assert not failing.certificate.valid
    with pytest.raises(ValidationError):
        failing.certificate.require()
    with pytest.raises(ValidationError, match="max_stencil_size"):
        rational_mixed_difference(coefficients, 0, (1, 2), max_stencil_size=3)
    with pytest.raises(ValidationError, match="max_order"):
        rational_mixed_difference(coefficients, 0, (1, 1), max_order=1)
    with pytest.raises(ValidationError, match="finite packet prefix"):
        rational_mixed_difference(coefficients, 0, (9,))
    for values, steps in (((1.0, 2.0), (1,)), ((1, 2), (True,)), ((1, 2), (-1,))):
        with pytest.raises(ValidationError):
            rational_mixed_difference(values, 0, steps)


def test_latest_euler_factorial_precision_and_mixed_cancellation_controls():
    stencils = ((1,), (2,), (1, 2), (2, 4), (1, 1, 2), (1, 2, 4, 8))
    for address, offset in enumerate((0, 0, 3, 1)):
        coefficients = congruential_inverse(4, address, 40)
        for row in range(3):
            for steps in stencils:
                order = len(steps)
                baseline = (
                    order
                    + padic_valuation(factorial(order), 2)
                    + sum(padic_valuation(step, 2) for step in steps)
                )
                valuation = padic_valuation(_mixed_difference(coefficients, row, steps), 2)
                assert valuation >= baseline + offset
                if address in (0, 1):
                    even_count = sum(step % 2 == 0 for step in steps)
                    if order % 2 or even_count % 2 == 0:
                        assert valuation == baseline
                    else:
                        assert valuation >= baseline + 1
                if address == 2:
                    assert valuation >= 2 * order + padic_valuation(factorial(order), 2) + order // 2 + sum(
                        padic_valuation(step, 2) for step in steps
                    )
            for order in (1, 2, 4, 8):
                for step in (1, 2, 4):
                    steps = (step,) * order
                    valuation = padic_valuation(_mixed_difference(coefficients, row, steps), 2)
                    if address in (0, 1):
                        assert valuation == order + padic_valuation(
                            factorial(order), 2
                        ) + order * padic_valuation(step, 2)
                    if address == 2:
                        leading = {1: 4, 2: 6}.get(
                            order, 2 * order + padic_valuation(factorial(order), 2) + order // 2
                        )
                        assert valuation == leading + order * padic_valuation(step, 2)


def test_prime_power_mixed_transfer_sign_boundary_and_upper_half_counterexamples():
    for prime, widths in ((2, (1, 2, 4, 8)), (3, (3, 9)), (5, (5,))):
        for width in widths:
            for address in range(width):
                coefficients = congruential_inverse(width, address, 10)
                signed = tuple((-1) ** row * coefficient for row, coefficient in enumerate(coefficients))
                for steps in ((1,), (2,), (1, 2), (1, 3, 3)):
                    bound = len(steps) + sum(padic_valuation(step, prime) for step in steps)
                    for row in range(3):
                        assert padic_valuation(_mixed_difference(signed, row, steps), prime) >= bound
                        if prime == 2:
                            assert (
                                padic_valuation(_mixed_difference(coefficients, row, steps), prime) >= bound
                            )
                if prime != 2 and address == 0:
                    assert padic_valuation(coefficients[1] - coefficients[0], prime) == 0
    for width, address, first, second in ((16, 9, 5, 7), (32, 18, 6, 6), (32, 19, 5, 7)):
        coefficients = congruential_inverse(width, address, 2)
        assert padic_valuation(coefficients[1] - coefficients[0], 2) == first
        assert padic_valuation(coefficients[2] - coefficients[0], 2) == second
        assert second != first + 1


def test_nishibuchi_finite_degree_displacements_include_boundary_addresses():
    for prime in (2, 3, 5, 7, 11):
        for multiplier in sorted({1, 2, prime - 1}):
            if (prime - 1) % multiplier and multiplier != 2:
                continue
            width = multiplier * prime
            for address in sorted({0, prime - 1, width - 1}):
                for precision in (1, 2):
                    row_step = lcm(2, prime - 1) * prime ** (precision - 1) // multiplier
                    coefficients = congruential_inverse(width, address, precision + row_step + 1)
                    for row in (precision, precision + 1):
                        difference = factorial(address) * (coefficients[row + row_step] - coefficients[row])
                        assert padic_valuation(difference, prime) >= precision


def test_occupancy_leading_zero_columns_preserve_the_actual_span():
    duplicate_basis = np.array([[0, 0, 0], [0, 1, 2]], dtype=float)
    same_span = np.array([[0], [1]], dtype=float)
    report = nested_subspace_occupancy((duplicate_basis, same_span))
    assert report.dimensions == (1, 1)
    assert report.intersection_dimension == 1
    assert report.stabilized_at == 1


def test_occupancy_early_equality_does_not_claim_later_stability():
    line = np.array([[0], [1]], dtype=float)
    report = nested_subspace_occupancy((np.eye(2), np.eye(2), line))
    assert report.dimensions == (2, 2, 1)
    assert report.stabilized_at is None
    later = nested_subspace_occupancy((np.eye(2), np.eye(2), line, line))
    assert later.stabilized_at == 3
    assert later.certificate.diagnostics["stability_scope"] == "supplied finite prefix only"


def test_occupancy_all_coordinate_subspace_intersections_with_dependent_columns():
    for dimension in range(1, 5):
        transform = np.eye(dimension) + np.triu(np.ones((dimension, dimension)), 1)
        for left_mask in range(2**dimension):
            for right_mask in range(2**dimension):
                bases = []
                for mask in (left_mask, right_mask):
                    columns = [transform[:, index] for index in range(dimension) if mask & (1 << index)]
                    data = [np.zeros(dimension), *columns, *columns]
                    bases.append(np.column_stack(data))
                report = nested_subspace_occupancy(bases)
                assert report.dimensions == (left_mask.bit_count(), (left_mask & right_mask).bit_count())


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf"), 1])
def test_occupancy_tolerance_controls(tolerance):
    with pytest.raises(ValidationError):
        nested_subspace_occupancy((np.eye(2),), tolerance=tolerance)


def test_occupancy_matrix_and_nonfinite_controls():
    with pytest.raises(DimensionError):
        nested_subspace_occupancy((np.asarray(1),))
    with pytest.raises(ValidationError):
        nested_subspace_occupancy((np.array([[float("nan")]]),))


def _forward_cycle(degree, closing):
    matrix = sp.zeros(degree)
    for column in range(degree):
        matrix[(column + 1) % degree, column] = closing if column == degree - 1 else 1
    return matrix


@pytest.mark.parametrize("q", [2, 3, 4])
def test_exact_tame_tensor_cycles_and_polynomials_against_full_tensor_matrix(q):
    for left_degree in range(1, 5):
        for right_degree in range(1, 5):
            left = FrobeniusOrbit(q, left_degree, 1)
            right = FrobeniusOrbit(q, right_degree, -1)
            result = tame_tensor_character_packets(left, right, left_scalar=2, right_scalar=Fraction(1, 2))
            result.certificate.require()
            assert {pair for packet in result.packets for pair in packet.cycle_pairs} == set(
                product(range(left_degree), range(right_degree))
            )
            tensor = sp.kronecker_product(
                _forward_cycle(left_degree, 2), _forward_cycle(right_degree, sp.Rational(1, 2))
            )
            period = lcm(left_degree, right_degree)
            modulus = q**period - 1
            invariant_rows = [
                i * right_degree + j
                for i, j in product(range(left_degree), range(right_degree))
                if (
                    left.exponent * q**i * (modulus // left.modulus)
                    + right.exponent * q**j * (modulus // right.modulus)
                )
                % modulus
                == 0
            ]
            restricted = tensor.extract(invariant_rows, invariant_rows)
            assert result.inertia_fixed_dimension == len(invariant_rows)
            assert (
                result.weil_fixed_dimension
                == len(invariant_rows) - (restricted - sp.eye(len(invariant_rows))).rank()
            )
            u = sp.Symbol("u")
            assert sp.expand((sp.eye(len(invariant_rows)) - u * restricted).det()) == sum(
                sp.Rational(value.numerator, value.denominator) * u**degree
                for degree, value in enumerate(result.local_polynomial_coefficients)
            )
            assert result.tame_artin_conductor == left_degree * right_degree - len(invariant_rows)
            for packet in result.packets:
                # Multiply the actual tensor-edge coefficients around this cycle.
                closing = Fraction(1)
                for i, j in packet.cycle_pairs:
                    closing *= Fraction(2) if i == left_degree - 1 else 1
                    closing *= Fraction(1, 2) if j == right_degree - 1 else 1
                assert closing == packet.closing_scalar


def test_tame_tensor_interaction_cycle_can_have_trivial_inertia_without_being_atomic():
    result = tame_tensor_character_packets(FrobeniusOrbit(2, 2, 1), FrobeniusOrbit(2, 2, 2))
    assert len(result.packets) == 2
    assert result.inertia_fixed_dimension == 2
    assert result.weil_fixed_dimension == 1
    assert result.local_polynomial_coefficients == (Fraction(1), Fraction(0), Fraction(-1))
    assert result.certificate.diagnostics["interaction_packets_need_not_be_irreducible"]
    # A different closing scalar retains inertia invariants but removes Weil fixed vectors.
    twisted = tame_tensor_character_packets(FrobeniusOrbit(2, 2, 1), FrobeniusOrbit(2, 2, 2), left_scalar=-2)
    assert twisted.inertia_fixed_dimension == 2
    assert twisted.weil_fixed_dimension == 0
    assert twisted.local_polynomial_coefficients == (Fraction(1), Fraction(0), Fraction(2))


@pytest.mark.parametrize("scalar", [0, True, 1.0, complex(1)])
def test_tame_tensor_strict_nonzero_rational_scalar_domain(scalar):
    with pytest.raises(ValidationError):
        tame_tensor_character_packets(FrobeniusOrbit(2, 2, 1), FrobeniusOrbit(2, 2, 1), left_scalar=scalar)


def test_tame_tensor_input_conventions_and_resource_caps():
    with pytest.raises(ValidationError, match="same residue"):
        tame_tensor_character_packets(FrobeniusOrbit(2, 2, 1), FrobeniusOrbit(3, 2, 1))
    with pytest.raises(ValidationError, match="prime power"):
        tame_tensor_character_packets(FrobeniusOrbit(6, 1, 1), FrobeniusOrbit(6, 1, 1))
    with pytest.raises(ValidationError, match="regular"):
        tame_tensor_character_packets(FrobeniusOrbit(2, 2, 0), FrobeniusOrbit(2, 2, 1))
    with pytest.raises(ValidationError, match="max_dimension"):
        tame_tensor_character_packets(FrobeniusOrbit(2, 4, 1), FrobeniusOrbit(2, 4, 1), max_dimension=15)
    with pytest.raises(ValidationError, match="max_modulus_bits"):
        tame_tensor_character_packets(FrobeniusOrbit(2, 4, 1), FrobeniusOrbit(2, 4, 1), max_modulus_bits=7)
    with pytest.raises(ValidationError, match="max_residue_cardinality"):
        tame_tensor_character_packets(
            FrobeniusOrbit(9, 1, 1), FrobeniusOrbit(9, 1, 1), max_residue_cardinality=8
        )
