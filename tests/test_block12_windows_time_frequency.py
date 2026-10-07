from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.exceptions import AssumptionError, DimensionError, ValidationError
from komorebi_math.signals import (
    canonical_dual_window,
    frame_signal,
    get_window,
    istft,
    overlap_add,
    overlap_add_certificate,
    overlap_profile,
    spectrogram,
    stft,
    window_metrics,
)


def test_window_conventions_and_aliases() -> None:
    periodic = get_window("hann", 8, periodic=True)
    symmetric = get_window("hanning", 8, periodic=False)
    assert periodic.shape == symmetric.shape == (8,)
    assert periodic[-1] != 0.0
    assert symmetric[-1] == pytest.approx(0.0)
    assert np.array_equal(get_window("rectangular", 5), np.ones(5))
    assert np.array_equal(get_window([1.0, 2.0, 3.0], 3), np.array([1.0, 2.0, 3.0]))



def test_tukey_parameter_limits_and_periodic_shape() -> None:
    assert np.array_equal(get_window(("tukey", 0.0), 16), np.ones(16))
    assert np.allclose(get_window(("tukey", 1.0), 16), get_window("hann", 16))
    tapered = get_window(("tukey", 0.4), 17, periodic=False)
    assert tapered[0] == pytest.approx(0.0)
    assert tapered[-1] == pytest.approx(0.0)
    assert np.max(tapered) == pytest.approx(1.0)

def test_window_metrics_for_periodic_hann() -> None:
    metrics = window_metrics(get_window("hann", 128), sample_rate=1024.0)
    assert metrics.coherent_gain == pytest.approx(0.5)
    assert metrics.power_gain == pytest.approx(3.0 / 8.0)
    assert metrics.equivalent_noise_bandwidth_bins == pytest.approx(1.5)
    assert metrics.equivalent_noise_bandwidth_hz == pytest.approx(12.0)


def test_overlap_profiles_and_cola_nola_certificates() -> None:
    window = get_window("hann", 128)
    linear = overlap_profile(window, 64, squared=False)
    squared = overlap_profile(window, 32, squared=True)
    assert np.max(np.abs(linear - linear[0])) < 1e-12
    assert np.max(np.abs(squared - squared[0])) < 1e-12
    overlap_add_certificate(window, 64, squared=False, require_constant=True).require()
    overlap_add_certificate(window, 32, squared=True, require_constant=True).require()


def test_canonical_dual_window_has_unit_cross_overlap() -> None:
    window = get_window(("kaiser", 7.0), 63)
    result = canonical_dual_window(window, 17)
    dual = result.require()
    profile = overlap_profile(window * dual, 17, squared=False)
    assert np.allclose(profile, 1.0, atol=1e-12)


def test_zero_window_has_no_dual() -> None:
    result = canonical_dual_window(np.zeros(8), 4)
    assert not result.valid
    with pytest.raises(ValidationError):
        result.require()


def test_frame_signal_without_padding_leaves_only_complete_frames() -> None:
    signal = np.arange(13.0)
    batch = frame_signal(signal, 5, 3, center=False, pad_end=False, boundary="none")
    assert batch.frames.shape == (3, 5)
    assert np.array_equal(batch.starts, np.array([0, 3, 6]))
    assert np.array_equal(batch.frames[1], signal[3:8])
    assert batch.covered_length == 11
    assert batch.padded_length == 13
    batch.certificate().require()


def test_centered_frame_signal_pads_and_covers_original() -> None:
    signal = np.arange(17.0)
    batch = frame_signal(signal, 8, 3, center=True, pad_end=True, boundary="zeros")
    assert batch.pad_left == 4
    assert batch.covered_length == batch.padded_length
    assert batch.frames.shape[1] == 8
    batch.certificate().require()


def test_frame_signal_rejects_short_unpadded_record() -> None:
    with pytest.raises(DimensionError):
        frame_signal(np.arange(3.0), 8, center=False, pad_end=False)
    with pytest.raises(ValidationError):
        frame_signal(np.arange(10.0), 4, 5)


def test_overlap_add_reconstructs_rectangular_nonoverlapping_frames() -> None:
    signal = np.arange(24.0)
    batch = frame_signal(signal, 6, 6, boundary="none")
    reconstructed = overlap_add(batch.frames, 6)
    assert np.array_equal(reconstructed, signal)


def test_real_stft_roundtrip_odd_record_and_zero_padding() -> None:
    rng = np.random.default_rng(12001)
    signal = rng.normal(size=1001)
    transform = stft(signal, frame_length=127, hop_length=31, fft_length=256)
    reconstructed = transform.inverse().require()
    assert reconstructed.shape == signal.shape
    assert np.allclose(reconstructed, signal, atol=2e-15)
    transform.roundtrip_certificate(signal).require()


def test_complex_two_sided_stft_roundtrip() -> None:
    rng = np.random.default_rng(12002)
    signal = rng.normal(size=511) + 1j * rng.normal(size=511)
    transform = stft(signal, frame_length=64, hop_length=16, onesided=False)
    assert transform.coefficients.shape[1] == 64
    reconstructed = istft(transform).require()
    assert np.allclose(reconstructed, signal, atol=2e-15)


def test_stft_rejects_one_sided_complex_input() -> None:
    with pytest.raises(ValidationError):
        stft(np.arange(16.0) + 1j, frame_length=8, onesided=True)


def test_stft_detects_nonoverlap_window_singularity() -> None:
    signal = np.arange(16.0)
    transform = stft(
        signal,
        frame_length=8,
        hop_length=8,
        window="hann",
        center=False,
        pad_end=False,
        boundary="none",
    )
    result = transform.inverse()
    assert not result.valid


def test_detrended_stft_is_explicitly_noninvertible() -> None:
    transform = stft(np.arange(32.0), frame_length=8, detrend="constant")
    with pytest.raises(AssumptionError):
        transform.inverse()


def test_spectrogram_density_shape_and_nonnegativity() -> None:
    signal = np.sin(2.0 * np.pi * 0.125 * np.arange(128))
    result = spectrogram(
        signal,
        frame_length=32,
        hop_length=8,
        fft_length=64,
        scaling="density",
    )
    assert result.values.shape == result.transform.coefficients.shape
    assert np.all(result.values >= 0.0)
    assert result.frequencies.shape == (33,)
    assert result.times.shape == (result.transform.frame_count,)


def test_window_and_spectrogram_validation() -> None:
    with pytest.raises(DimensionError):
        get_window(np.ones(3), 4)
    with pytest.raises(ValidationError):
        get_window(("hann", 2.0), 8)
    with pytest.raises(ValidationError):
        spectrogram(np.arange(16.0), frame_length=8, scaling="unknown")
