from __future__ import annotations

import numpy as np
import pytest

from komorebi_math import OperatorGrammar, OperatorRealization, PacketOperator, PacketSystem
from komorebi_math.coordinates.analysis_synthesis import AnalysisSynthesisSystem
from komorebi_math.coordinates.numeric import NumericCoordinateTransform
from komorebi_math.nonlinear.correction import (
    HilbertObstructionProjector,
    LyapunovSchmidtReduction,
    RightInverseCorrectionProblem,
)
from komorebi_math.nonlinear.derivatives import CallableDerivativeProvider
from komorebi_math.refinement import WeightedTransferEdge, WeightedTransferGraph


def _singleton_packets() -> PacketSystem:
    return PacketSystem([np.array([[1.0]], dtype=np.complex128)])


def test_packet_system_snapshots_caller_projector() -> None:
    source = np.array([[1.0]], dtype=np.complex128)
    packets = PacketSystem([source])
    before = packets.certificate()
    source[0, 0] = 2.0
    assert before.valid
    assert packets.certificate().valid
    assert packets.projectors[0][0, 0] == 1.0


def test_packet_system_numpy_projector_is_not_mutable_through_public_reference() -> None:
    packets = _singleton_packets()
    with pytest.raises(ValueError):
        packets.projectors[0][0, 0] = 2.0
    assert packets.certificate().valid


def test_operator_realization_snapshots_caller_matrix() -> None:
    grammar = OperatorGrammar(["X"], {"T": ("X", "X")})
    grammar.relation("idempotent", [(1, grammar.word("T", "T")), (-1, grammar.word("T"))])
    matrix = np.array([[1.0]], dtype=np.complex128)
    realization = OperatorRealization(grammar, {"X": 1}, {"T": matrix})
    matrix[0, 0] = 2.0
    assert realization.relation_certificate().valid
    assert realization.operators["T"][0, 0] == 1.0


def test_packet_operator_snapshots_direct_block_arrays() -> None:
    packets = _singleton_packets()
    block = np.array([[3.0]], dtype=np.complex128)
    operator = PacketOperator(packets, packets, ((block,),))
    block[0, 0] = 9.0
    assert operator.reconstruct()[0, 0] == 3.0
    assert operator.certificate().valid


def test_hilbert_obstruction_projector_snapshots_matrix_before_svd_cache() -> None:
    matrix = np.array([[1.0]], dtype=np.complex128)
    model = HilbertObstructionProjector(matrix, tolerance=1e-12)
    matrix[0, 0] = 0.0
    result = model.solve([2.0])
    assert model.matrix[0, 0] == 1.0
    assert result.vanishes
    assert np.allclose(result.least_norm_correction, [-2.0])
    assert result.certificate.valid


def test_right_inverse_problem_snapshots_linear_data_and_target() -> None:
    linear = np.array([[1.0]], dtype=np.complex128)
    inverse = np.array([[1.0]], dtype=np.complex128)
    target = np.array([0.1], dtype=np.complex128)
    problem = RightInverseCorrectionProblem(
        linear,
        inverse,
        lambda x: np.zeros_like(x),
        target,
        radius=0.2,
        lipschitz_bound=0.0,
        tolerance=1e-12,
    )
    linear[0, 0] = 0.0
    inverse[0, 0] = 0.0
    target[0] = 10.0
    assert problem.hypothesis_certificate().valid
    result = problem.solve()
    assert result.certificate.valid
    assert np.allclose(result.solution, [0.1])


def test_lyapunov_schmidt_snapshots_base_point() -> None:
    provider = CallableDerivativeProvider(
        lambda point: np.array([point[0]]),
        {1: lambda point, directions: np.array([directions[0][0]])},
        source_dimension=1,
        target_dimension=1,
    )
    base = np.array([0.0], dtype=np.complex128)
    reduction = LyapunovSchmidtReduction(provider, base, tolerance=1e-12)
    base[0] = 7.0
    assert np.allclose(reduction.base_point, [0.0])
    assert reduction.decomposition.certificate.valid


def test_weighted_graph_snapshots_diagonal_before_inverse_cache() -> None:
    packets = _singleton_packets()
    diagonal = np.array([[2.0]], dtype=np.complex128)
    graph = WeightedTransferGraph(packets, diagonal, [])
    diagonal[0, 0] = 4.0
    assert graph.diagonal[0, 0] == 2.0
    assert graph.diagonal_inverse[0, 0] == 0.5
    assert graph.operator[0, 0] == 2.0


def test_weighted_graph_snapshots_edge_operator() -> None:
    packets = _singleton_packets()
    edge_matrix = np.array([[0.25]], dtype=np.complex128)
    edge = WeightedTransferEdge(0, 0, 1, edge_matrix)
    graph = WeightedTransferGraph(packets, np.eye(1), [edge])
    edge_matrix[0, 0] = 0.75
    assert graph.edges[0].operator[0, 0] == 0.25
    assert graph.operator[0, 0] == 1.25


def test_existing_immutable_coordinate_surfaces_are_controls() -> None:
    matrix = np.array([[2.0]], dtype=np.complex128)
    transform = NumericCoordinateTransform("x", "y", matrix)
    matrix[0, 0] = 9.0
    assert transform.matrix[0, 0] == 2.0
    with pytest.raises(ValueError):
        transform.matrix[0, 0] = 3.0

    analysis = np.array([[2.0]], dtype=np.complex128)
    synthesis = np.array([[0.5]])
    system = AnalysisSynthesisSystem(analysis, synthesis)
    analysis[0, 0] = 7.0
    synthesis[0, 0] = 8.0
    assert system.analysis[0, 0] == 2.0
    assert system.synthesis[0, 0] == 0.5
    with pytest.raises(ValueError):
        system.analysis[0, 0] = 1.0
