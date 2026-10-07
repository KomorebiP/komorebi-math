from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.algorithms import DenseOperator, PacketIR, PacketLayout
from komorebi_math.exceptions import DimensionError, ValidationError


def test_layout_split_merge_and_names() -> None:
    layout = PacketLayout((2, 1, 3), ("a", "b", "c"))
    x = np.arange(6.0)
    parts = layout.split(x)
    assert tuple(part.size for part in parts) == (2, 1, 3)
    assert np.array_equal(layout.merge(parts), x)
    assert layout.packet_slice("b") == slice(2, 3)


def test_layout_matrix_rhs_and_crossed() -> None:
    left = PacketLayout((1, 2), ("x", "y"))
    right = PacketLayout((2, 3), ("u", "v"))
    crossed = left.crossed(right)
    assert crossed.sizes == (2, 3, 4, 6)
    matrix = np.arange(18.0).reshape(3, 6)
    parts = left.split(matrix)
    assert np.array_equal(left.merge(parts), matrix)


def test_layout_rejects_bad_data() -> None:
    with pytest.raises(ValidationError):
        PacketLayout(())
    with pytest.raises(ValidationError):
        PacketLayout((1, 0))
    with pytest.raises(ValidationError):
        PacketLayout((1, 1), ("a", "a"))
    with pytest.raises(ValidationError):
        PacketLayout((1.5, 2))
    with pytest.raises(ValidationError):
        PacketLayout((True, 2))
    with pytest.raises(ValidationError):
        PacketLayout((1, 1), ("a", " "))
    with pytest.raises(DimensionError):
        PacketLayout((1, 2)).split(np.zeros(4))


def test_layout_accepts_numpy_integer_index() -> None:
    layout = PacketLayout((1, 2), ("a", "b"))
    assert layout.index(np.int64(1)) == 1


def test_packet_ir_apply_materialize_and_adjoint() -> None:
    layout = PacketLayout((2, 1))
    blocks = {
        (0, 0): np.array([[1.0, 2.0], [0.0, 1.0]]),
        (1, 0): np.array([[3.0, 4.0]]),
        (1, 1): np.array([[5.0]]),
    }
    ir = PacketIR.square(layout, blocks)
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(ir.apply(x), ir.materialize() @ x)
    assert np.allclose(ir.adjoint().materialize(), ir.materialize().conj().T)
    ir.certificate().require()


def test_packet_ir_from_dense_threshold() -> None:
    layout = PacketLayout((1, 1, 1))
    matrix = np.diag([1.0, 2.0, 3.0])
    matrix[1, 0] = 1e-14
    ir = PacketIR.from_dense(matrix, layout, atol=1e-12)
    assert ir.support == frozenset({(0, 0), (1, 1), (2, 2)})


def test_packet_ir_rejects_negative_thresholds() -> None:
    layout = PacketLayout((1,))
    with pytest.raises(ValidationError):
        PacketIR.from_dense([[1.0]], layout, atol=-1.0)
    ir = PacketIR.square(layout, {(0, 0): [[1.0]]})
    with pytest.raises(ValidationError):
        ir.compose(ir, atol=-1.0)


def test_packet_ir_composition_and_power() -> None:
    layout = PacketLayout((1, 1, 1))
    a = PacketIR.square(
        layout,
        {
            (1, 0): [[2.0]],
            (2, 1): [[3.0]],
        },
    )
    product = a.compose(a)
    assert np.allclose(product.materialize(), a.materialize() @ a.materialize())
    assert np.allclose(a.power(2).materialize(), a.materialize() @ a.materialize())
    assert np.allclose(a.power(0).materialize(), np.eye(3))


def test_packet_ir_tensor() -> None:
    left_layout = PacketLayout((1, 1))
    right_layout = PacketLayout((1, 2))
    left = PacketIR.square(left_layout, {(0, 0): [[2.0]], (1, 1): [[3.0]]})
    right = PacketIR.square(right_layout, {(0, 0): [[4.0]], (1, 1): np.eye(2)})
    tensor = left.tensor(right)
    assert np.allclose(tensor.materialize(), np.kron(left.materialize(), right.materialize()))


def test_packet_ir_rectangular_layouts() -> None:
    source = PacketLayout((2, 1))
    target = PacketLayout((1, 2))
    matrix = np.arange(9.0).reshape(3, 3)
    ir = PacketIR.from_dense(matrix, source, target, keep_zeros=True)
    assert ir.shape == (3, 3)
    assert np.allclose(ir.apply(np.eye(3)), matrix)


def test_packet_ir_rejects_wrong_block_shape() -> None:
    layout = PacketLayout((2, 1))
    with pytest.raises(DimensionError):
        PacketIR.square(layout, {(0, 1): DenseOperator(np.eye(2))})

def test_packet_ir_strict_integer_addresses_and_powers() -> None:
    layout = PacketLayout((1, 1))
    with pytest.raises(ValidationError):
        PacketIR.square(layout, {(0.0, 0): [[1.0]]})
    operator = PacketIR.square(layout, {(0, 0): [[1.0]]})
    with pytest.raises(ValidationError):
        operator.power(2.5)
    with pytest.raises(ValidationError):
        operator.flop_count(rhs=1.5)

