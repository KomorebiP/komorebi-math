"""End-to-end finite signal-processing example."""

from __future__ import annotations

import json

import numpy as np

from komorebi_math.signals import (
    certified_overlap_add_convolve,
    cross_spectrum,
    denoising_risk_certificate,
    design_biquad,
    design_lowpass,
    fir_filter,
    nlms_filter,
    periodogram,
    resample_poly_result,
    stft,
    stream_fir_filter,
    stream_lfilter,
    stream_stft,
    wavelet_denoise,
    welch_psd,
)


def main() -> None:
    """Run one deterministic end-to-end signal-processing example."""

    rng = np.random.default_rng(1212)
    sample_rate = 1024.0
    index = np.arange(1024)
    clean = np.sin(2.0 * np.pi * 73.0 * index / sample_rate)
    noisy = clean + rng.normal(scale=0.15, size=clean.size)

    transform = stft(noisy, sample_rate=sample_rate, frame_length=128, hop_length=32)
    transform.roundtrip_certificate(noisy).require()

    line = periodogram(clean, sample_rate=sample_rate, window="boxcar", detrend="none")
    welch = welch_psd(noisy, sample_rate=sample_rate, frame_length=256, hop_length=128)
    coherence = cross_spectrum(clean, 1.75 * clean, frame_length=256, hop_length=128)
    coherence.certificate.require()

    lowpass = design_lowpass(120.0, num_taps=63, sample_rate=sample_rate)
    lowpass.certificate(tolerance=1e-9).require()
    streamed = stream_fir_filter(noisy, lowpass, (113, 287, 19, 605))
    streamed.certificate.require()

    resampled = resample_poly_result(streamed.value, 5, 4)
    resampled.certificate.require()

    adaptive_source = rng.normal(size=3000)
    target_weights = np.array([0.7, -0.35, 0.15])
    desired = fir_filter(adaptive_source, target_weights, mode="causal")
    adaptive = nlms_filter(adaptive_source, desired, 3, step_size=0.8)
    adaptive.certificate.require()

    piecewise_clean = np.repeat(np.array([0.0, 2.0, -1.0, 1.5]), 256)
    piecewise_noisy = piecewise_clean + rng.normal(scale=0.3, size=piecewise_clean.size)
    denoised = wavelet_denoise(piecewise_noisy, levels=5, rule="universal", mode="soft")
    denoised.certificate.require()
    risk = denoising_risk_certificate(piecewise_clean, piecewise_noisy, denoised.signal)

    chunked_stft = stream_stft(
        noisy,
        (123, 456, 445),
        sample_rate=sample_rate,
        frame_length=128,
        hop_length=32,
        fft_length=256,
    )
    chunked_stft.certificate.require()

    biquad = design_biquad("lowpass", 120.0, q=0.8, sample_rate=sample_rate)
    biquad.stability_certificate().require()
    recursive = stream_lfilter(biquad, noisy, (1, 17, 300, 706))
    recursive.certificate.require()

    kernel = np.array([0.125, 0.375, 0.375, 0.125])
    overlap_add = certified_overlap_add_convolve(noisy, kernel, block_length=37)
    overlap_add.certificate.require()

    report = {
        "dominant_frequency_hz": line.dominant_frequency(),
        "welch_segments": welch.segment_count,
        "stft_frames": transform.frame_count,
        "streaming_stft_frames": chunked_stft.value.frame_count,
        "streaming_fir_samples": int(streamed.value.size),
        "resampled_samples": int(resampled.signal.size),
        "recursive_filter_samples": int(recursive.value.size),
        "overlap_add_samples": int(overlap_add.value.size),
        "adaptive_final_weights": adaptive.final_weights.tolist(),
        "denoising_observed_risk_improved": risk.valid,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
