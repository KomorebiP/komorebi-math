import numpy as np

from komorebi_math.coordinates.numeric import (
    discrete_cosine_transform,
    discrete_fourier_transform,
    haar_wavelet_transform,
    modal_coordinate_system,
    walsh_hadamard_transform,
)
from komorebi_math.coordinates.packet import (
    chinese_remainder_transform,
    digit_reversal_transform,
    residue_class_transform,
    tensor_axis_permutation_transform,
)


def test_orthogonal_and_fourier_coordinate_roundtrips() -> None:
    rng = np.random.default_rng(12345)
    values = rng.normal(size=16)
    for transform in (
        discrete_fourier_transform(16),
        discrete_cosine_transform(16),
        walsh_hadamard_transform(16),
        haar_wavelet_transform(16),
    ):
        recovered = transform.inverse().apply(transform.apply(values))
        assert np.allclose(recovered, values, atol=1e-11, rtol=1e-11)
        transform.certificate().require()


def test_modal_coordinates_diagonalize_a_generic_matrix() -> None:
    operator = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [0.0, 0.0, 5.0]])
    modal = modal_coordinate_system(operator)
    modal.certificate().require()
    diagonal = modal.analysis.matrix @ operator @ modal.synthesis.matrix
    assert np.allclose(diagonal, np.diag(modal.eigenvalues), atol=1e-10, rtol=1e-10)


def test_packet_and_index_coordinate_permutations() -> None:
    values = tuple(range(12))
    residue = residue_class_transform(12, 4)
    assert residue.apply(values) == (0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11)
    assert residue.inverse().apply(residue.apply(values)) == values

    reversal = digit_reversal_transform(8)
    assert reversal.apply(tuple(range(8))) == (0, 4, 2, 6, 1, 5, 3, 7)

    array = np.arange(24).reshape(2, 3, 4)
    axes = tensor_axis_permutation_transform((2, 3, 4), (2, 0, 1))
    transformed = np.array(axes.apply(array.ravel())).reshape(4, 2, 3)
    assert np.array_equal(transformed, np.transpose(array, (2, 0, 1)))

    crt = chinese_remainder_transform((3, 4, 5))
    transformed_indices = crt.apply(tuple(range(60)))
    assert sorted(transformed_indices) == list(range(60))
    assert crt.inverse().apply(transformed_indices) == tuple(range(60))
