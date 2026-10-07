from __future__ import annotations

import math

import numpy as np
import pytest

from komorebi_math.signals.filterbanks import HaarFilterBank
from komorebi_math.signals.route_packets import PolyphaseRoutePacket, polyphase_route_packet


def test_route_packet_roundtrip_recovers_filter_exactly() -> None:
    kernel = np.array([2, -1, 3, 0, 4, -2, 5], dtype=object)
    for order in (2, 3, 5):
        packet = polyphase_route_packet(kernel, order)
        assert np.array_equal(packet.reconstruct_filter(), kernel)


def test_route_packet_composition_equals_direct_convolution() -> None:
    first = np.array([1, 2, -1, 3, 1], dtype=object)
    second = np.array([2, -1, 0, 4], dtype=object)
    for order in (2, 3, 4):
        left = PolyphaseRoutePacket.from_filter(first, order)
        right = PolyphaseRoutePacket.from_filter(second, order)
        cascade = left.compose(right)
        expected = np.convolve(first, second)
        assert np.array_equal(cascade.reconstruct_filter(), expected)
        direct = PolyphaseRoutePacket.from_filter(expected, order)
        for target in range(order):
            for source in range(order):
                a = cascade.kernels[target][source]
                b = direct.kernels[target][source]
                length = max(a.size, b.size)
                aa = np.pad(a, (0, length - a.size))
                bb = np.pad(b, (0, length - b.size))
                assert np.array_equal(aa, bb)


def test_route_packet_composition_is_associative() -> None:
    filters = (
        np.array([1, 1, -1], dtype=object),
        np.array([2, 0, 1, 3], dtype=object),
        np.array([-1, 2, 2], dtype=object),
    )
    packets = tuple(PolyphaseRoutePacket.from_filter(h, 3) for h in filters)
    left = packets[0].compose(packets[1]).compose(packets[2])
    right = packets[0].compose(packets[1].compose(packets[2]))
    assert np.array_equal(left.reconstruct_filter(), right.reconstruct_filter())
    assert np.array_equal(left.reconstruct_filter(), np.convolve(np.convolve(filters[0], filters[1]), filters[2]))


def test_convex_max_carrier_cannot_preserve_haar_reconstruction() -> None:
    x = np.array([1.0, 0.0])
    low, high = HaarFilterBank(normalized=True).analyze(x)
    scale = math.sqrt(2.0)

    even_contributions = (float(low[0] / scale), float(high[0] / scale))
    odd_contributions = (float(low[0] / scale), float(-high[0] / scale))

    # Reconstruction is additive: both even contributions are required and the
    # odd contributions cancel.  A max/idempotent carrier gives the wrong answer.
    assert sum(even_contributions) == pytest.approx(1.0)
    assert max(even_contributions) == pytest.approx(0.5)
    assert sum(odd_contributions) == pytest.approx(0.0)
    assert max(odd_contributions) == pytest.approx(0.5)


def test_duplicate_additive_routes_cannot_be_deduplicated() -> None:
    # Idempotent support packets can delete an exact duplicate atom.  Additive
    # signal routing cannot: 1 + 1 is not 1.
    contributions = np.array([1.0, 1.0])
    assert float(np.sum(contributions)) == 2.0
    assert float(contributions[0]) == 1.0
