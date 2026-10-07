from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.exceptions import DimensionError, ValidationError
from komorebi_math.signals import (
    StreamingFIR,
    cross_spectrum,
    design_bandpass,
    design_bandstop,
    design_fir,
    design_highpass,
    design_lowpass,
    fir_filter,
    fir_frequency_response,
    parseval_certificate,
    periodogram,
    stream_fir_filter,
    welch_psd,
)


def test_parseval_certificate_real_and_complex() -> None:
    rng = np.random.default_rng(12101)
    parseval_certificate(rng.normal(size=127)).require()
    parseval_certificate(rng.normal(size=127) + 1j * rng.normal(size=127)).require()


def test_periodogram_recovers_integer_bin_and_power() -> None:
    sample_rate = 1024.0
    index = np.arange(1024)
    signal = 2.0 * np.sin(2.0 * np.pi * 91.0 * index / sample_rate)
    estimate = periodogram(
        signal,
        sample_rate=sample_rate,
        window="boxcar",
        detrend="none",
        scaling="density",
    )
    estimate.certificate.require()
    assert estimate.dominant_frequency() == 91.0
    assert estimate.integrated_power() == pytest.approx(np.mean(signal**2), rel=1e-12)


def test_periodogram_spectrum_scaling_has_expected_sine_line_power() -> None:
    index = np.arange(256)
    signal = 3.0 * np.sin(2.0 * np.pi * 17.0 * index / 256.0)
    estimate = periodogram(signal, window="boxcar", detrend="none", scaling="spectrum")
    peak = estimate.power[int(np.argmax(estimate.power))]
    assert peak == pytest.approx(3.0**2 / 2.0, rel=1e-12)


def test_two_sided_complex_periodogram_frequency() -> None:
    length = 128
    frequency = -13
    index = np.arange(length)
    signal = np.exp(2.0j * np.pi * frequency * index / length)
    estimate = periodogram(signal, detrend="none", onesided=False)
    assert estimate.dominant_frequency() == pytest.approx(frequency / length)


def test_welch_boxcar_nonoverlap_integrates_exact_finite_mean_power() -> None:
    rng = np.random.default_rng(12102)
    signal = rng.normal(size=1024)
    estimate = welch_psd(
        signal,
        sample_rate=512.0,
        frame_length=128,
        hop_length=128,
        window="boxcar",
        detrend="none",
        scaling="density",
    )
    assert estimate.segment_count == 8
    assert estimate.integrated_power() == pytest.approx(np.mean(signal**2), rel=1e-12)


def test_welch_median_is_finite_and_nonnegative() -> None:
    rng = np.random.default_rng(12103)
    estimate = welch_psd(rng.normal(size=1000), frame_length=100, average="median")
    estimate.certificate.require()
    assert np.all(np.isfinite(estimate.power))
    assert np.all(estimate.power >= 0)


def test_cross_spectrum_recovers_deterministic_gain_and_coherence() -> None:
    rng = np.random.default_rng(12104)
    source = rng.normal(size=4096)
    gain = -2.25
    report = cross_spectrum(
        source,
        gain * source,
        frame_length=256,
        hop_length=128,
        detrend="none",
    )
    report.certificate.require()
    active = report.input_power > 1e-12
    assert np.allclose(report.coherence[active], 1.0, atol=2e-15)
    assert np.allclose(report.transfer_h1[active], gain, atol=2e-14)


def test_cross_spectrum_regularization_handles_zero_record() -> None:
    record = np.zeros(512)
    report = cross_spectrum(record, record, frame_length=128, regularization=1e-9)
    report.certificate.require()
    assert np.array_equal(report.coherence, np.zeros_like(report.coherence))
    assert np.array_equal(report.transfer_h1, np.zeros_like(report.transfer_h1))


def test_cross_spectrum_rejects_mismatched_lengths() -> None:
    with pytest.raises(DimensionError):
        cross_spectrum(np.ones(10), np.ones(11), frame_length=4)


def test_all_windowed_sinc_designs_are_symmetric_and_normalized() -> None:
    designs = (
        design_lowpass(100.0, num_taps=101, sample_rate=1000.0),
        design_highpass(100.0, num_taps=101, sample_rate=1000.0),
        design_bandpass((100.0, 200.0), num_taps=101, sample_rate=1000.0),
        design_bandstop((100.0, 200.0), num_taps=101, sample_rate=1000.0),
    )
    for design in designs:
        design.certificate(tolerance=1e-9).require()
        assert np.allclose(design.taps, design.taps[::-1], atol=1e-15)
        assert design.group_delay_samples == 50.0


def test_lowpass_and_highpass_frequency_selectivity() -> None:
    low = design_lowpass(100.0, num_taps=101, sample_rate=1000.0)
    high = design_highpass(100.0, num_taps=101, sample_rate=1000.0)
    low_response = fir_frequency_response(low.taps, sample_rate=1000.0, points=1001)
    high_response = fir_frequency_response(high.taps, sample_rate=1000.0, points=1001)
    near_dc = 0
    near_nyquist = -1
    assert low_response.magnitude[near_dc] > 0.999
    assert low_response.magnitude[near_nyquist] < 1e-3
    assert high_response.magnitude[near_dc] < 1e-3
    assert high_response.magnitude[near_nyquist] > 0.999


def test_bandpass_and_bandstop_complementary_centers() -> None:
    band = (100.0, 200.0)
    passed = design_bandpass(band, num_taps=121, sample_rate=1000.0)
    stopped = design_bandstop(band, num_taps=121, sample_rate=1000.0)
    passed_response = passed.response(points=1001)
    stopped_response = stopped.response(points=1001)
    center_index = 300
    assert passed_response.frequencies[center_index] == pytest.approx(150.0)
    assert passed_response.magnitude[center_index] > 0.99
    assert stopped_response.magnitude[center_index] < 0.02


def test_fir_filter_modes_match_finite_convolution_slices() -> None:
    signal = np.arange(1.0, 8.0)
    taps = np.array([0.25, 0.5, 0.25])
    full = np.convolve(signal, taps, mode="full")
    assert np.allclose(fir_filter(signal, taps, mode="full"), full)
    assert np.allclose(fir_filter(signal, taps, mode="causal"), full[: signal.size])
    assert np.allclose(fir_filter(signal, taps, mode="same"), full[1 : 1 + signal.size])
    assert np.allclose(fir_filter(signal, taps, mode="valid"), full[2:7])


def test_streaming_fir_matches_batch_for_irregular_chunks() -> None:
    rng = np.random.default_rng(12105)
    signal = rng.normal(size=1111)
    taps = design_lowpass(0.2, num_taps=31).taps
    result = stream_fir_filter(signal, taps, (1, 17, 300, 2, 791))
    result.certificate.require()
    assert np.allclose(result.value, fir_filter(signal, taps, mode="causal"), atol=1e-15)


def test_streaming_fir_supports_complex_chunks_and_reset() -> None:
    rng = np.random.default_rng(12106)
    signal = rng.normal(size=100) + 1j * rng.normal(size=100)
    taps = np.array([1.0, -0.2, 0.1])
    processor = StreamingFIR(taps)
    first = np.concatenate((processor.process(signal[:37]), processor.process(signal[37:])))
    assert np.allclose(first, fir_filter(signal, taps, mode="causal"))
    processor.reset()
    second = processor.process(signal)
    assert np.allclose(second, first)


def test_filter_design_validation() -> None:
    with pytest.raises(ValidationError):
        design_fir("lowpass", 0.2, num_taps=10)
    with pytest.raises(ValidationError):
        design_bandpass((0.3, 0.2))
    with pytest.raises(ValidationError):
        design_lowpass(0.6, sample_rate=1.0)
    with pytest.raises(DimensionError):
        fir_filter([], [1.0])
