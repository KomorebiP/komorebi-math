"""Arbitrary-size integer controls for additive polyphase routes."""

import numpy as np
import pytest

from komorebi_math.signals.route_packets import PolyphaseRoutePacket


def integer_convolution(left, right):
    output = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            output[i + j] += int(a) * int(b)
    return output


@pytest.mark.parametrize("order", [1, 2, 3, 5])
@pytest.mark.parametrize("dtype,values", [(np.uint8, [255, 255]), (np.int64, [2**40, 1]),
                                        (object, [np.int64(2**40), np.int64(1)])])
def test_integer_route_composition_preserves_products_beyond_dtype_range(order, dtype, values):
    kernel = np.asarray(values, dtype=dtype)
    packet = PolyphaseRoutePacket.from_filter(kernel, order)
    result = packet.compose(packet).reconstruct_filter()
    assert result.tolist() == integer_convolution(values, values)
    assert result.dtype == object
    assert all(isinstance(item, int) for item in result)


@pytest.mark.parametrize("dtype", [float, complex])
def test_numerical_route_packet_types_keep_floating_and_complex_semantics(dtype):
    values = np.asarray([1.25, -.75, 2.0], dtype=dtype)
    if dtype is complex:
        values[1] += .2j
    packet = PolyphaseRoutePacket.from_filter(values, 3)
    output = packet.compose(packet).reconstruct_filter()
    assert output.dtype == values.dtype
    np.testing.assert_allclose(output, np.convolve(values, values))


def test_large_integer_route_cascade_associativity_matches_python_arithmetic():
    kernels = ([2**40, -2**40, 1], [2**40, 2**40], [1, -1, 2**40])
    packets = [PolyphaseRoutePacket.from_filter(kernel, 3) for kernel in kernels]
    expected = integer_convolution(integer_convolution(kernels[0], kernels[1]), kernels[2])
    assert packets[0].compose(packets[1]).compose(packets[2]).reconstruct_filter().tolist() == expected
    assert packets[0].compose(packets[1].compose(packets[2])).reconstruct_filter().tolist() == expected
