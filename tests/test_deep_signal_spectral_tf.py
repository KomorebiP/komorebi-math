"""Independent regression checks for adversarial spectral and STFT inputs."""

from dataclasses import replace

import numpy as np
import pytest

from komorebi_math.exceptions import ValidationError
from komorebi_math.signals import (
    canonical_dual_window,
    cross_spectrum,
    get_window,
    overlap_add_certificate,
    parseval_certificate,
    periodogram,
    spectrogram,
    stft,
)
from komorebi_math.signals.approximation import (
    estimate_positive_packet_1d,
    fft_lowpass_1d,
    otsu_threshold,
)


@pytest.mark.parametrize("length,mode", [(10, 3), (14, 5), (22, 7), (33, 15)])
def test_fourier_cutoff_retains_exact_integer_boundary_mode(length, mode):
    signal = np.cos(2 * np.pi * mode * np.arange(length) / length)
    assert np.allclose(fft_lowpass_1d(signal, mode), signal, atol=1e-13)
    assert np.max(np.abs(fft_lowpass_1d(signal, mode - 1))) < 1e-13


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_stft_rejects_nonfinite_sources(nonfinite):
    with pytest.raises(ValidationError, match="finite"):
        stft([nonfinite, 1.0], frame_length=2, window="boxcar")


def test_inverse_rejects_mutated_nonfinite_coefficients():
    transform = stft(np.arange(16.0), frame_length=4)
    coefficients = transform.coefficients.copy()
    coefficients[0, 0] = np.nan
    with pytest.raises(ValidationError, match="finite"):
        replace(transform, coefficients=coefficients).inverse()
    with pytest.raises(ValidationError, match="finite"):
        transform.inverse(synthesis_window=np.full(4, np.inf))


def test_large_finite_parseval_energy_does_not_square_unscaled_fft():
    certificate = parseval_certificate(np.full(128, 1e153))
    assert certificate.valid
    assert np.isfinite(certificate.residuals["parseval"])


def test_unrepresentable_parseval_energy_is_not_certified():
    certificate = parseval_certificate(np.full(128, 1e155))
    assert not certificate.valid


def test_parseval_promotes_integer_energy_before_squaring():
    assert parseval_certificate(np.full(64, 2**40, dtype=np.int64)).valid


def test_wrong_large_source_fails_stft_roundtrip_with_finite_residual():
    transform = stft(np.zeros(128), frame_length=16, window="boxcar")
    certificate = transform.roundtrip_certificate(np.full(128, 1e155))
    assert not certificate.valid
    assert np.isfinite(certificate.residuals["stft_roundtrip"])
    assert certificate.residuals["stft_roundtrip"] == pytest.approx(np.sqrt(128) * 1e155)


@pytest.mark.parametrize("scale", [1e-80, 1e-100])
def test_coherence_retains_perfect_gain_under_small_common_scaling(scale):
    signal = np.random.default_rng(42).normal(size=512)
    report = cross_spectrum(signal * scale, 2 * signal * scale, frame_length=64, window="boxcar", detrend="none")
    report.certificate.require()
    assert np.all(report.input_power > 0)
    assert np.allclose(report.coherence, 1.0, atol=1e-14)
    assert np.allclose(report.transfer_h1, 2.0, atol=1e-14)


def test_otsu_packet_estimate_is_translation_invariant():
    signal = np.array([0.0, 2.0, 100.0])
    base = estimate_positive_packet_1d(signal)
    shifted = estimate_positive_packet_1d(signal + 1e10)
    assert np.array_equal(base.mask, np.array([False, False, True]))
    assert np.array_equal(base.mask, shifted.mask)
    assert shifted.amplitude == base.amplitude


@pytest.mark.parametrize("values", [[0.0, np.nan], [0.0, np.inf], [1.0 + 1.0j]])
def test_otsu_rejects_nonfinite_or_complex_values(values):
    with pytest.raises(ValidationError):
        otsu_threshold(values)
    with pytest.raises(ValidationError):
        estimate_positive_packet_1d(values)


def test_kaiser_overflow_is_a_clear_validation_failure():
    with pytest.raises(ValidationError, match="Kaiser.*nonfinite"):
        get_window(("kaiser", 1000.0), 8)


def test_single_sample_window_still_validates_name():
    with pytest.raises(ValidationError, match="unknown window"):
        get_window("not_a_window", 1)


def test_complex_window_uses_two_sided_transform_by_default():
    signal = np.arange(16.0)
    window = np.ones(4) * (1.0 + 1.0j)
    transform = stft(signal, frame_length=4, window=window)
    assert not transform.onesided
    assert np.allclose(transform.inverse().require(), signal)
    assert not periodogram(signal, window=np.ones(16) * (1 + 1j)).onesided
    with pytest.raises(ValidationError, match="real.*window"):
        stft(signal, frame_length=4, window=window, onesided=True)


def test_density_scaling_requires_nonzero_window_energy():
    with pytest.raises(ValidationError, match="normalization"):
        periodogram(np.arange(4.0), window=np.zeros(4))
    with pytest.raises(ValidationError, match="window energy"):
        spectrogram(np.arange(4.0), frame_length=4, window=np.zeros(4))


def test_small_window_scale_preserves_lattice_dual_and_stft_reconstruction():
    window = np.full(4, 1e-8)
    overlap_add_certificate(window, 2).require()
    assert canonical_dual_window(window, 2).valid
    signal = np.arange(16.0)
    assert np.allclose(stft(signal, frame_length=4, window=window).inverse().require(), signal)


def test_spectrum_normalization_does_not_reject_small_coherent_gain():
    signal = np.arange(8.0)
    normal = periodogram(signal, window=np.ones(8), detrend="none", scaling="spectrum")
    scaled = periodogram(signal, window=np.ones(8) * 1e-10, detrend="none", scaling="spectrum")
    assert scaled.certificate.valid
    assert np.allclose(scaled.power, normal.power)
