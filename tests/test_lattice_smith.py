from __future__ import annotations

import pytest

from komorebi_math.exceptions import ValidationError
from komorebi_math.refinement import (
    LatticePacketData,
    SmithCongruenceProblem,
    padic_valuation,
    rational_padic_valuation,
    refinement_weight_monotonicity,
    smith_decomposition,
)

sp = pytest.importorskip("sympy")


def test_padic_valuations() -> None:
    assert padic_valuation(40, 2) == 3
    assert padic_valuation(-45, 3) == 2
    assert rational_padic_valuation(sp.Rational(3, 8), 2) == -3


def test_smith_decomposition_with_transforms() -> None:
    matrix = sp.Matrix([[12, 6, 4], [3, 9, 6], [2, 16, 14]])
    decomposition = smith_decomposition(matrix)
    decomposition.certificate().require()
    assert decomposition.invariant_factors == (1, 10, 30)


def test_smith_congruence_count_and_particular_solution() -> None:
    problem = SmithCongruenceProblem([[2, 0, 0], [0, 4, 0]], [2, 4], prime=2, depth=3)
    result = problem.solve()
    result.require()
    assert result.smith_exponents == (1, 2)
    assert result.solution_count == 64
    assert result.particular_solution is not None
    assert len(problem.enumerate_solutions(limit=100)) == 64


def test_smith_congruence_obstruction() -> None:
    problem = SmithCongruenceProblem([[8]], [4], prime=2, depth=5)
    result = problem.solve()
    assert not result.solvable
    assert result.obstruction_rows == (0,)
    assert result.solution_count == 0


def test_exact_lattice_separation_and_gluing() -> None:
    p0 = sp.Matrix([[1, sp.Rational(1, 4)], [0, 0]])
    p1 = sp.eye(2) - p0
    split_basis = sp.Matrix([[1, -1], [0, 4]])
    data = LatticePacketData(sp.eye(2), [p0, p1], prime=2, split_basis=split_basis)
    assert data.projector_weights() == (2, 2)
    assert data.separation_weight == 2
    gluing = data.gluing_module()
    assert gluing.invariant_factors == (1, 4)
    assert gluing.length == 2
    data.gluing_weight_certificate().require()


def test_precision_delay() -> None:
    p0 = sp.Matrix([[1, sp.Rational(1, 8)], [0, 0]])
    p1 = sp.eye(2) - p0
    data = LatticePacketData(sp.eye(2), [p0, p1], prime=2)
    delays = data.precision_delays()
    assert [delay.weight for delay in delays] == [3, 3]
    assert delays[0].input_depth(10) == 13


def test_invalid_split_basis_rejected_by_gluing_certificate() -> None:
    p0 = sp.diag(1, 0)
    p1 = sp.diag(0, 1)
    with pytest.raises(ValidationError, match="integrally contained"):
        data = LatticePacketData(sp.eye(2), [p0, p1], prime=2, split_basis=sp.diag(sp.Rational(1, 2), 1))
        data.gluing_module()


def test_smith_congruence_uses_signed_diagonal_entry() -> None:
    problem = SmithCongruenceProblem([[-2]], [2], prime=2, depth=3)
    result = problem.solve().require()
    assert result.particular_solution is not None
    x = result.particular_solution[0]
    assert (-2 * x - 2) % 8 == 0
    assert result.solution_count == 2
    assert set(problem.enumerate_solutions()) == {(3,), (7,)}


def test_prime_parameters_reject_composite_moduli() -> None:
    with pytest.raises(ValueError, match="prime"):
        SmithCongruenceProblem([[1]], [0], prime=4, depth=1)
    with pytest.raises(ValueError, match="prime"):
        LatticePacketData([[1]], [[[1]]], prime=6)


def test_refinement_weight_requires_actual_projector_refinement() -> None:
    sp = pytest.importorskip("sympy")
    lattice = sp.eye(2)
    coordinate_packets = [sp.diag(1, 0), sp.diag(0, 1)]
    rotated_packets = [
        sp.Matrix([[sp.Rational(1, 2), sp.Rational(1, 2)], [sp.Rational(1, 2), sp.Rational(1, 2)]]),
        sp.Matrix([[sp.Rational(1, 2), -sp.Rational(1, 2)], [-sp.Rational(1, 2), sp.Rational(1, 2)]]),
    ]
    coarse = LatticePacketData(lattice, coordinate_packets, prime=2)
    unrelated = LatticePacketData(lattice, rotated_packets, prime=2)
    with pytest.raises(ValidationError, match="parent|reconstruct"):
        refinement_weight_monotonicity(coarse, unrelated)
