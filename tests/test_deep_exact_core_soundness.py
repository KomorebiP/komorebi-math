"""Independent exact-field and mutable-witness regression cases."""
from __future__ import annotations

from itertools import product
import numpy as np
import pytest

from komorebi_math.core.seeds import semisimple_seed_rank
from komorebi_math.exceptions import ValidationError
from komorebi_math.refinement import LatticePacketData, SmithCongruenceProblem, SmithDecomposition, refinement_weight_monotonicity

sp = pytest.importorskip("sympy")


@pytest.mark.parametrize("count,dimension", [(2**53 + 1, 1), (10**400 + 1, 3), (17, 5)])
def test_semisimple_seed_ceiling_uses_exact_integer_arithmetic(count, dimension) -> None:
    rank = semisimple_seed_rank([dimension], [count])
    assert rank * dimension >= count
    assert (rank - 1) * dimension < count


@pytest.mark.parametrize("dimensions,counts", [([1.5], [2]), ([1], [True]), ([1], [2.5])])
def test_semisimple_formula_rejects_lossy_integer_coercion(dimensions, counts) -> None:
    with pytest.raises(ValidationError):
        semisimple_seed_rank(dimensions, counts)


def test_cached_smith_witness_must_match_current_matrix() -> None:
    problem = SmithCongruenceProblem([[2]], [1], prime=2, depth=1)
    problem.matrix = sp.Matrix([[1]])
    assert (problem.matrix * sp.Matrix([1]) - problem.rhs)[0] % 2 == 0
    with pytest.raises(ValidationError, match="does not match"):
        problem.solve()


def test_cached_smith_witness_is_checked_before_claiming_kernel_completeness() -> None:
    problem = SmithCongruenceProblem([[0]], [0], prime=2, depth=1)
    old = problem.decomposition
    problem.decomposition = SmithDecomposition(old.matrix, old.diagonal, old.left, sp.zeros(1), old.invariant_factors)
    assert not problem.decomposition.certificate().valid
    with pytest.raises(ValidationError, match="unimodular"):
        problem.solve()


def test_produced_smith_matrices_are_owned_immutable_snapshots() -> None:
    matrix = sp.Matrix([[2]])
    rhs = sp.Matrix([0])
    problem = SmithCongruenceProblem(matrix, rhs, prime=2, depth=2)
    matrix[0, 0] = 1
    rhs[0] = 1
    assert problem.matrix == sp.Matrix([[2]])
    assert problem.rhs == sp.Matrix([0])
    for witness in (problem.matrix, problem.rhs, problem.decomposition.left, problem.decomposition.right):
        with pytest.raises(TypeError):
            witness[0, 0] = 0


@pytest.mark.parametrize("rows", [0, 1, 2])
def test_zero_column_smith_problem_has_one_empty_solution(rows) -> None:
    problem = SmithCongruenceProblem(sp.zeros(rows, 0), sp.zeros(rows, 1), prime=2, depth=2)
    result = problem.solve().require()
    assert result.solution_count == 1
    assert result.particular_solution == ()
    assert problem.enumerate_solutions() == ((),)


def test_no_variable_equation_with_nonzero_rhs_is_unsolvable() -> None:
    result = SmithCongruenceProblem(sp.zeros(1, 0), [1], prime=2, depth=1).solve()
    assert not result.solvable
    result.certificate.require()


@pytest.mark.parametrize("depth", [1.5, True])
def test_smith_depth_cannot_be_silently_truncated(depth) -> None:
    with pytest.raises(ValidationError):
        SmithCongruenceProblem([[1]], [0], prime=2, depth=depth)


def test_singular_replaced_lattice_basis_has_invalid_fresh_certificate() -> None:
    data = LatticePacketData(sp.eye(2), [sp.eye(2)], prime=2)
    data.lattice_basis = sp.diag(1, 0)
    assert not data.certificate().valid
    with pytest.raises(ValidationError):
        data.projector_weights()


def test_irrational_packet_projectors_do_not_enter_rational_valuation_calculus() -> None:
    x = sp.sqrt(2)
    first = sp.Matrix([[1, x], [0, 0]])
    second = sp.eye(2) - first
    assert first * first == first
    assert first + second == sp.eye(2)
    with pytest.raises(ValidationError):
        LatticePacketData(sp.eye(2), [first, second], prime=2)


def test_lattice_projector_ownership_preserves_certified_partition() -> None:
    first = sp.diag(1, 0)
    data = LatticePacketData(sp.eye(2), [first, sp.eye(2) - first], prime=2)
    first[0, 0] = 0
    data.certificate().require()
    with pytest.raises(TypeError):
        data.projectors[0][0, 0] = 0
    data.projectors = (sp.eye(2), sp.eye(2))
    assert not data.certificate().valid


def test_weight_monotonicity_revalidates_the_packet_systems() -> None:
    coarse = LatticePacketData(sp.eye(2), [sp.eye(2)], prime=2)
    fine = LatticePacketData(sp.eye(2), [sp.eye(2)], prime=2)
    coarse.projectors = (2 * sp.eye(2),)
    with pytest.raises(ValidationError):
        refinement_weight_monotonicity(coarse, fine)


def test_small_smith_solutions_match_independent_brute_force() -> None:
    random = np.random.default_rng(4003)
    for _ in range(40):
        matrix = random.integers(-3, 4, size=(2, 2)).tolist()
        rhs = random.integers(-3, 4, size=2).tolist()
        problem = SmithCongruenceProblem(matrix, rhs, prime=2, depth=2)
        expected = {
            vector for vector in product(range(4), repeat=2)
            if all((sum(matrix[row][column] * vector[column] for column in range(2)) - rhs[row]) % 4 == 0
                   for row in range(2))
        }
        result = problem.solve()
        result.certificate.require()
        assert result.solution_count == len(expected)
        assert set(problem.enumerate_solutions()) == expected
