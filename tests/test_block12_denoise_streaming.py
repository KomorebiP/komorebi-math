from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.exceptions import DimensionError, ValidationError
from komorebi_math.signals import (
    StreamingSTFT,
    denoising_risk_certificate,
    estimate_white_noise_sigma,
    haar_decompose,
    median_absolute_deviation,
    stream_stft,
    sure_threshold,
    threshold_coefficients,
    universal_threshold,
    wavelet_denoise,
)


def test_multilevel_haar_decomposition_reconstructs_and_preserves_energy() -> None:
    rng = np.random.default_rng(12301)
    signal = rng.normal(size=512)
    decomposition = haar_decompose(signal, 5)
    decomposition.certificate(signal).require()
    assert decomposition.levels == 5
    assert decomposition.approximation.size == 16
    assert np.allclose(decomposition.reconstruct(), signal, atol=2e-15)


def test_haar_decomposition_rejects_incompatible_depth() -> None:
    with pytest.raises(DimensionError):
        haar_decompose(np.arange(30.0), 4)
    with pytest.raises(ValidationError):
        haar_decompose(np.arange(32.0), 0)


def test_threshold_modes_have_declared_scalar_behavior() -> None:
    coefficients = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    assert np.array_equal(
        threshold_coefficients(coefficients, 2.0, mode="hard"),
        np.array([-3.0, 0.0, 0.0, 0.0, 3.0]),
    )
    assert np.array_equal(
        threshold_coefficients(coefficients, 2.0, mode="soft"),
        np.array([-1.0, 0.0, 0.0, 0.0, 1.0]),
    )
    garrote = threshold_coefficients(coefficients, 2.0, mode="garrote")
    assert np.allclose(garrote, np.array([-5.0 / 3.0, 0.0, 0.0, 0.0, 5.0 / 3.0]))


def test_complex_soft_threshold_preserves_phase() -> None:
    coefficients = np.array([3.0 + 4.0j, 0.0j])
    reduced = threshold_coefficients(coefficients, 2.0, mode="soft")
    assert reduced[0] == pytest.approx(1.8 + 2.4j)
    assert reduced[1] == 0.0j


def test_noise_scale_and_threshold_utilities() -> None:
    details = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert median_absolute_deviation(details, center=0.0) == 1.0
    assert estimate_white_noise_sigma(details) == pytest.approx(1.0 / 0.6744897501960817)
    assert universal_threshold(0.5, 100) == pytest.approx(0.5 * np.sqrt(2.0 * np.log(100)))
    threshold = sure_threshold(details, sigma=1.0)
    assert 0.0 <= threshold <= 2.0


def test_fixed_zero_threshold_is_identity() -> None:
    rng = np.random.default_rng(12302)
    signal = rng.normal(size=256)
    result = wavelet_denoise(signal, levels=4, rule="fixed", threshold=0.0)
    result.certificate.require()
    assert np.allclose(result.signal, signal, atol=2e-15)
    assert result.removed_energy == pytest.approx(0.0, abs=1e-28)


def test_wavelet_denoising_improves_observed_piecewise_constant_risk() -> None:
    rng = np.random.default_rng(12303)
    clean = np.repeat(np.array([0.0, 2.0, -1.0, 1.5]), 256)
    noisy = clean + rng.normal(scale=0.35, size=clean.size)
    result = wavelet_denoise(noisy, levels=5, rule="universal", mode="soft")
    result.certificate.require()
    denoising_risk_certificate(clean, noisy, result.signal).require()


def test_sure_and_level_dependent_denoising_are_finite() -> None:
    rng = np.random.default_rng(12304)
    signal = np.sin(2 * np.pi * np.arange(512) / 64) + rng.normal(scale=0.2, size=512)
    result = wavelet_denoise(signal, levels=4, rule="sure", level_dependent=True)
    result.certificate.require()
    assert len(result.thresholds) == 4
    assert np.all(np.isfinite(result.signal))


def test_denoising_validation() -> None:
    with pytest.raises(ValidationError):
        wavelet_denoise(np.arange(32.0), levels=2, rule="fixed")
    with pytest.raises(ValidationError):
        wavelet_denoise(np.arange(32.0), levels=2, rule="universal", threshold=1.0)
    with pytest.raises(ValidationError):
        threshold_coefficients(np.ones(3), -1.0)
    with pytest.raises(DimensionError):
        denoising_risk_certificate(np.ones(4), np.ones(5), np.ones(4))


@pytest.mark.parametrize(
    "length,frame_length,hop_length,pad_end",
    [
        (1, 8, 2, True),
        (127, 128, 32, True),
        (128, 128, 32, False),
        (128, 128, 32, True),
        (129, 128, 32, True),
        (160, 128, 32, True),
        (1000, 128, 32, False),
        (1000, 128, 32, True),
    ],
)
def test_streaming_stft_matches_batch_at_framing_boundaries(
    length: int, frame_length: int, hop_length: int, pad_end: bool
) -> None:
    rng = np.random.default_rng(12400 + length + int(pad_end))
    signal = rng.normal(size=length)
    if length == 1:
        chunks = (1,)
    else:
        first = max(1, length // 7)
        second = max(1, length // 3)
        chunks = (first, second, length - first - second)
    result = stream_stft(
        signal,
        chunks,
        frame_length=frame_length,
        hop_length=hop_length,
        fft_length=2 * frame_length,
        pad_end=pad_end,
    )
    result.certificate.require()


def test_streaming_stft_complex_two_sided_matches_batch() -> None:
    rng = np.random.default_rng(12305)
    signal = rng.normal(size=511) + 1j * rng.normal(size=511)
    result = stream_stft(
        signal,
        (13, 200, 1, 297),
        frame_length=64,
        hop_length=16,
        onesided=False,
    )
    result.certificate.require()


def test_streaming_stft_state_machine_flush_and_reset() -> None:
    rng = np.random.default_rng(12306)
    signal = rng.normal(size=200)
    processor = StreamingSTFT(frame_length=64, hop_length=16)
    first = processor.process(signal[:100])
    second = processor.process(signal[100:])
    terminal = processor.flush()
    assert first.frame_count + second.frame_count + terminal.frame_count == 10
    assert processor.total_samples == 200
    empty = processor.flush()
    assert empty.frame_count == 0
    with pytest.raises(ValidationError):
        processor.process(np.ones(1))
    processor.reset()
    assert processor.total_samples == 0
    assert processor.process(signal).frame_count > 0


def test_streaming_stft_validation() -> None:
    with pytest.raises(DimensionError):
        stream_stft(np.ones(10), (3, 3), frame_length=4)
    with pytest.raises(ValidationError):
        StreamingSTFT(frame_length=8, hop_length=9)
    processor = StreamingSTFT(frame_length=8, hop_length=2, onesided=True)
    with pytest.raises(ValidationError):
        processor.process(np.ones(8) + 1j)
