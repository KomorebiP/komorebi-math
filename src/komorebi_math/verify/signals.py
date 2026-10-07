"""Installed package verification for the finite signal domain."""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..signals import (
    HaarPacketTree,
    PeriodicFilterBank,
    SingularPacket1D,
    lattice_polyphase_analyze,
    polyphase_filter_kernels,
    packet_product,
    polyphase_analyze,
    recover_separated_jumps,
    retained_packet_reconstruction,
)
from .core import VerificationCheck, VerificationReport


def _capture(name: str, function: Callable[[], None]) -> VerificationCheck:
    try:
        function()
        return VerificationCheck(name, True)
    except Exception as exc:
        return VerificationCheck(name, False, f"{type(exc).__name__}: {exc}")


def run_signals_verification() -> VerificationReport:
    """Run lightweight checks for the installed signal implementation."""

    checks: list[VerificationCheck] = []

    def polyphase_check() -> None:
        values = np.arange(29.0)
        coordinates = polyphase_analyze(values, 4)
        coordinates.certificate(values).require()
        assert np.array_equal(coordinates.reconstruct(), values)

    checks.append(_capture("finite polyphase reconstruction", polyphase_check))

    def lattice_check() -> None:
        values = np.arange(42).reshape(6, 7)
        coordinates = lattice_polyphase_analyze(values, (2, 3))
        coordinates.certificate(values).require()
        assert np.array_equal(coordinates.reconstruct(), values)

    checks.append(_capture("lattice polyphase reconstruction", lattice_check))

    def filter_bank_check() -> None:
        bank = PeriodicFilterBank(([1.0],), ([1.0],), 1)
        bank.perfect_reconstruction_certificate(19).require()
        values = np.arange(19.0)
        assert np.allclose(bank.synthesize(bank.analyze(values)), values)

    checks.append(_capture("periodic filter bank reconstruction", filter_bank_check))

    def wavelet_check() -> None:
        values = np.arange(16.0)
        tree = HaarPacketTree.analyze(values, depth=3)
        assert np.allclose(tree.reconstruct(), values)
        tree.certificate().require()

    checks.append(_capture("Haar packet tree reconstruction", wavelet_check))

    def singular_check() -> None:
        packet = SingularPacket1D.from_jumps([0.25, 0.75], [2.0, -1.0])
        points = np.linspace(0.0, 1.0, 65)
        smooth = np.sin(2.0 * np.pi * points)
        signal = smooth + packet.evaluate(points)
        result = retained_packet_reconstruction(signal, packet.evaluate(points), np.eye(signal.size))
        result.certificate.require()
        assert np.allclose(result.reconstruction, signal)

    checks.append(_capture("retained singular packet reconstruction", singular_check))

    def recovery_check() -> None:
        values = np.concatenate((np.zeros(12), 3.0 * np.ones(12), np.ones(12)))
        result = recover_separated_jumps(
            values,
            spacing=1.0,
            lipschitz_bound=0.0,
            noise_bound=0.0,
            minimum_jump=1.5,
        )
        result.certificate.require()
        assert result.cells == (11, 23)

    checks.append(_capture("separated jump recovery", recovery_check))

    def nonlinear_check() -> None:
        left = np.array([1.0, 2.0, 3.0])
        right = np.array([4.0, -1.0])
        packets = packet_product(left, right, 3).require()
        reconstructed = sum(packets, np.zeros(left.size + right.size - 1))
        assert np.allclose(reconstructed, np.convolve(left, right))

    checks.append(_capture("nonlinear residue product", nonlinear_check))


    def compact_packet_check() -> None:
        left = (np.array([1.0, 2.0]), np.array([3.0, -1.0]))
        right = (np.array([2.0, 1.0]), np.array([4.0]))
        result = packet_product(left, right, 2)
        result.certificate.require()

    checks.append(_capture("compact packet carry", compact_packet_check))

    def infinite_kernel_check() -> None:
        signal = np.arange(9.0)
        kernel = np.array([1.0, -2.0, 3.0, 0.5])
        order = 3
        source = polyphase_analyze(signal, order).channels
        kernels = polyphase_filter_kernels(kernel, order)
        channels: list[np.ndarray] = []
        for target in range(order):
            terms = [
                np.convolve(source[index], kernels[target][index].coefficients)
                if source[index].size
                else np.zeros(1)
                for index in range(order)
            ]
            length = max(term.size for term in terms)
            combined = np.zeros(length)
            for term in terms:
                combined[: term.size] += term
            channels.append(combined)
        reconstructed = np.zeros(max(order * (channel.size - 1) + residue + 1 for residue, channel in enumerate(channels)))
        for residue, channel in enumerate(channels):
            reconstructed[residue : residue + order * channel.size : order] = channel
        direct = np.convolve(signal, kernel)
        assert np.allclose(reconstructed[: direct.size], direct)

    checks.append(_capture("infinite polyphase kernel law", infinite_kernel_check))

    return VerificationReport(tuple(checks))


__all__ = ["run_signals_verification"]
