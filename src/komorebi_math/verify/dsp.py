"""Installed-package verification for finite digital signal processing."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..signals import (
    canonical_dual_window,
    certified_overlap_add_convolve,
    cross_spectrum,
    denoising_risk_certificate,
    design_biquad,
    design_lowpass,
    fir_filter,
    get_window,
    nlms_filter,
    overlap_add_certificate,
    parseval_certificate,
    periodogram,
    resample_poly_result,
    stft,
    stream_fir_filter,
    stream_lfilter,
    stream_stft,
    wavelet_denoise,
    welch_psd,
)
from .core import VerificationCheck, VerificationReport


def _capture(name: str, function: Callable[[], None]) -> VerificationCheck:
    try:
        function()
        return VerificationCheck(name, True)
    except Exception as exc:
        return VerificationCheck(name, False, f"{type(exc).__name__}: {exc}")


def run_dsp_verification() -> VerificationReport:
    """Run deterministic installed-package checks for the production DSP layer."""

    checks: list[VerificationCheck] = []

    def window_check() -> None:
        window = get_window("hann", 128)
        overlap_add_certificate(window, 32, squared=True, require_constant=True).require()
        canonical_dual_window(window, 32).certificate.require()

    checks.append(_capture("window overlap and canonical dual", window_check))

    def stft_check() -> None:
        rng = np.random.default_rng(1201)
        signal = rng.normal(size=997)
        transform = stft(signal, frame_length=128, hop_length=32, fft_length=256)
        transform.roundtrip_certificate(signal, tolerance=1e-10).require()

    checks.append(_capture("normalized STFT roundtrip", stft_check))

    def spectrum_check() -> None:
        sample_rate = 1024.0
        index = np.arange(1024)
        signal = np.sin(2.0 * np.pi * 73.0 * index / sample_rate)
        estimate = periodogram(signal, sample_rate=sample_rate, detrend="none")
        estimate.certificate.require()
        assert estimate.dominant_frequency() == 73.0
        parseval_certificate(signal).require()
        welch_psd(
            signal,
            sample_rate=sample_rate,
            frame_length=256,
            detrend="none",
        ).certificate.require()

    checks.append(_capture("periodogram Welch and Parseval", spectrum_check))

    def coherence_check() -> None:
        rng = np.random.default_rng(1202)
        source = rng.normal(size=4096)
        report = cross_spectrum(
            source,
            1.75 * source,
            frame_length=256,
            hop_length=128,
            detrend="none",
        )
        report.certificate.require()
        active = report.input_power > 1e-10
        assert np.allclose(report.coherence[active], 1.0, atol=1e-12)
        assert np.allclose(report.transfer_h1[active], 1.75, atol=1e-12)

    checks.append(_capture("cross spectrum coherence and H1 transfer", coherence_check))

    def fir_check() -> None:
        rng = np.random.default_rng(1203)
        signal = rng.normal(size=1200)
        design = design_lowpass(0.12, num_taps=63, sample_rate=1.0)
        design.certificate(tolerance=1e-9).require()
        streamed = stream_fir_filter(signal, design.taps, (113, 287, 19, 781))
        streamed.certificate.require()
        assert np.allclose(streamed.value, fir_filter(signal, design.taps, mode="causal"))

    checks.append(_capture("FIR design and streaming equivalence", fir_check))

    def resampling_check() -> None:
        rng = np.random.default_rng(1204)
        signal = rng.normal(size=257)
        result = resample_poly_result(signal, 5, 3)
        result.certificate.require()
        assert result.signal.size == 429

    checks.append(_capture("rational polyphase resampling identity", resampling_check))

    def adaptive_check() -> None:
        rng = np.random.default_rng(1205)
        source = rng.normal(size=3000)
        target_weights = np.array([0.7, -0.35, 0.15])
        desired = fir_filter(source, target_weights, mode="causal")
        result = nlms_filter(source, desired, 3, step_size=0.8)
        result.certificate.require()
        assert np.allclose(result.final_weights, target_weights, atol=1e-8)

    checks.append(_capture("normalized LMS system identification", adaptive_check))

    def denoise_check() -> None:
        rng = np.random.default_rng(1206)
        clean = np.repeat(np.array([0.0, 2.0, -1.0, 1.5]), 128)
        noisy = clean + rng.normal(scale=0.3, size=clean.size)
        result = wavelet_denoise(noisy, levels=4, rule="universal", mode="soft")
        result.certificate.require()
        denoising_risk_certificate(clean, noisy, result.signal).require()

    checks.append(_capture("finite Haar shrinkage and observed risk", denoise_check))

    def streaming_stft_check() -> None:
        rng = np.random.default_rng(1207)
        signal = rng.normal(size=1000)
        result = stream_stft(
            signal,
            (123, 456, 421),
            frame_length=128,
            hop_length=32,
            fft_length=256,
        )
        result.certificate.require()

    checks.append(_capture("chunk-invariant streaming STFT", streaming_stft_check))

    def recursive_check() -> None:
        rng = np.random.default_rng(1208)
        signal = rng.normal(size=900)
        model = design_biquad("lowpass", 0.12, sample_rate=1.0)
        model.stability_certificate().require()
        result = stream_lfilter(model, signal, (1, 23, 400, 476))
        result.certificate.require()

    checks.append(_capture("stable biquad and recursive streaming", recursive_check))

    def overlap_add_check() -> None:
        rng = np.random.default_rng(1209)
        signal = rng.normal(size=777) + 1j * rng.normal(size=777)
        kernel = rng.normal(size=61)
        result = certified_overlap_add_convolve(signal, kernel, block_length=37)
        result.certificate.require()

    checks.append(_capture("FFT overlap-add convolution identity", overlap_add_check))

    return VerificationReport(tuple(checks))


__all__ = ["run_dsp_verification"]
