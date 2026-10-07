from __future__ import annotations

import numpy as np

from komorebi_math import PacketOperator, PacketSystem, block_from_paths, path_contributions


def singleton_packets(size: int) -> PacketSystem:
    return PacketSystem(
        [np.diag([1 if i == j else 0 for i in range(size)]) for j in range(size)],
        names=[str(index) for index in range(size)],
    )


def test_packet_operator_reconstruction_and_apply() -> None:
    packets = singleton_packets(3)
    matrix = np.array([[1, 2, 0], [0, 3, 4], [5, 0, 6]], dtype=float)
    operator = PacketOperator.from_operator(matrix, packets)
    assert np.allclose(operator.reconstruct(), matrix)
    vector = np.array([1, 2, 3], dtype=float)
    assert np.allclose(operator.apply(vector), matrix @ vector)
    coordinates = packets.analyze(vector)
    assert np.allclose(operator.apply_coordinates(coordinates).reconstruct(), matrix @ vector)


def test_packet_operator_composition() -> None:
    packets = singleton_packets(3)
    left_matrix = np.array([[1, 2, 0], [0, 1, 3], [4, 0, 1]], dtype=float)
    right_matrix = np.array([[2, 0, 1], [1, 2, 0], [0, 1, 2]], dtype=float)
    left = PacketOperator.from_operator(left_matrix, packets)
    right = PacketOperator.from_operator(right_matrix, packets)
    product = left.compose(right)
    assert np.allclose(product.reconstruct(), left_matrix @ right_matrix)


def test_packet_operator_adjoint() -> None:
    packets = singleton_packets(2)
    matrix = np.array([[1 + 1j, 2], [3j, 4]], dtype=complex)
    operator = PacketOperator.from_operator(matrix, packets)
    assert np.allclose(operator.adjoint().reconstruct(), matrix.conj().T)


def test_packet_operator_tensor() -> None:
    packets = singleton_packets(2)
    left_matrix = np.array([[1, 2], [3, 4]], dtype=float)
    right_matrix = np.array([[0, 1], [1, 0]], dtype=float)
    left = PacketOperator.from_operator(left_matrix, packets)
    right = PacketOperator.from_operator(right_matrix, packets)
    tensor = left.tensor(right)
    assert np.allclose(tensor.reconstruct(), np.kron(left_matrix, right_matrix))


def test_power_and_path_expansion() -> None:
    packets = singleton_packets(3)
    matrix = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0]], dtype=float)
    operator = PacketOperator.from_operator(matrix, packets)
    assert np.allclose(operator.power(2).reconstruct(), matrix @ matrix)
    contributions = path_contributions(operator, source=0, target=2, length=2)
    assert len(contributions) == 1
    assert contributions[0].vertices == (0, 1, 2)
    result = block_from_paths(operator, source=0, target=2, length=2)
    assert result.valid
    assert np.allclose(result.value, operator.power(2).blocks[2][0])


def test_zero_length_path_is_packet_identity() -> None:
    packets = singleton_packets(2)
    operator = PacketOperator.from_operator(np.eye(2), packets)
    result = block_from_paths(operator, source=1, target=1, length=0)
    assert result.valid
    assert np.allclose(result.value, packets.projectors[1])


def test_block_diagonal_detection() -> None:
    packets = singleton_packets(3)
    diagonal = PacketOperator.from_operator(np.diag([1, 2, 3]), packets)
    off_diagonal = PacketOperator.from_operator(np.ones((3, 3)), packets)
    assert diagonal.is_block_diagonal()
    assert not off_diagonal.is_block_diagonal()
