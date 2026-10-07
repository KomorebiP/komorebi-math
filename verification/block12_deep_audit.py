"""Independent deterministic randomized audit for the Block 12 DSP layer."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from komorebi_math import __version__
from komorebi_math.signals import (
    LinearFilter,
    OverlapAddConvolver,
    canonical_dual_window,
    certified_overlap_add_convolve,
    cross_spectrum,
    denoising_risk_certificate,
    design_bandpass,
    design_biquad,
    design_highpass,
    design_lowpass,
    fft_convolve,
    fir_filter,
    get_window,
    haar_decompose,
    nlms_filter,
    overlap_add_certificate,
    parseval_certificate,
    periodogram,
    resample_poly_result,
    rls_filter,
    stft,
    stream_fir_filter,
    stream_lfilter,
    stream_stft,
    threshold_coefficients,
    wavelet_denoise,
    welch_psd,
)


@dataclass(slots=True)
class Audit:
    """Count scalar comparisons and retain compact failure diagnostics."""

    checks: int = 0
    failures: list[str] = field(default_factory=list)
    sections: dict[str, int] = field(default_factory=dict)

    def check(self, condition: bool, message: str, *, section: str) -> None:
        self.checks += 1
        self.sections[section] = self.sections.get(section, 0) + 1
        if not bool(condition) and len(self.failures) < 100:
            self.failures.append(message)

    def allclose(
        self,
        observed: Any,
        expected: Any,
        *,
        atol: float,
        rtol: float = 0.0,
        message: str,
        section: str,
    ) -> None:
        left = np.asarray(observed)
        right = np.asarray(expected)
        if left.shape != right.shape:
            self.check(False, f"{message}: shape {left.shape} != {right.shape}", section=section)
            return
        finite = np.isfinite(left) & np.isfinite(right)
        residual = np.abs(left - right)
        tolerance = atol + rtol * np.abs(right)
        matches = finite & (residual <= tolerance)
        flat = matches.ravel()
        self.checks += int(flat.size)
        self.sections[section] = self.sections.get(section, 0) + int(flat.size)
        if not bool(np.all(flat)) and len(self.failures) < 100:
            index = int(np.flatnonzero(~flat)[0])
            self.failures.append(
                f"{message}: first mismatch at flat index {index}, "
                f"observed={left.ravel()[index]!r}, expected={right.ravel()[index]!r}, "
                f"residual={residual.ravel()[index]!r}"
            )


def _partition(length: int, rng: np.random.Generator) -> tuple[int, ...]:
    sizes: list[int] = []
    remaining = length
    while remaining:
        size = int(rng.integers(1, min(remaining, 97) + 1))
        sizes.append(size)
        remaining -= size
    return tuple(sizes)


def _difference_equation(
    numerator: np.ndarray,
    denominator: np.ndarray,
    signal: np.ndarray,
) -> np.ndarray:
    """Evaluate a causal transfer function independently of the package recursion."""

    b = np.asarray(numerator) / denominator[0]
    a = np.asarray(denominator) / denominator[0]
    output = np.zeros(signal.size, dtype=np.result_type(b, a, signal))
    for index in range(signal.size):
        for offset, coefficient in enumerate(b):
            if index >= offset:
                output[index] += coefficient * signal[index - offset]
        for offset, coefficient in enumerate(a[1:], start=1):
            if index >= offset:
                output[index] -= coefficient * output[index - offset]
    return output


def run() -> dict[str, Any]:
    """Run the full deterministic audit and return its JSON-compatible report."""

    rng = np.random.default_rng(12012026)
    audit = Audit()

    # Window and dual identities.
    for name in ("boxcar", "hann", "hamming", "blackman", "bartlett", ("kaiser", 7.0), ("tukey", 0.4)):
        for length in (15, 16, 31, 32, 63, 64, 127, 128):
            window = get_window(name, length, periodic=True)
            audit.check(window.shape == (length,), f"window shape {name} {length}", section="windows")
            audit.check(bool(np.all(np.isfinite(window))), f"window finite {name} {length}", section="windows")
    for length, hop in ((32, 8), (64, 16), (128, 32), (63, 17), (127, 29)):
        window = get_window(("kaiser", 6.0), length)
        dual = canonical_dual_window(window, hop).require()
        certificate = overlap_add_certificate(
            window * dual, hop, squared=False, require_constant=True
        )
        audit.check(certificate.valid, f"canonical dual {length}/{hop}", section="windows")

    # STFT reconstruction and stream equality.
    for case in range(120):
        length = int(rng.integers(1, 1500))
        frame = int(rng.choice((16, 32, 64, 128)))
        hop = max(1, frame // int(rng.choice((2, 4))))
        fft_length = int(rng.choice((frame, 2 * frame)))
        signal = rng.normal(size=length)
        transform = stft(
            signal,
            frame_length=frame,
            hop_length=hop,
            fft_length=fft_length,
            center=True,
            pad_end=True,
        )
        reconstructed = transform.inverse().require()
        audit.allclose(
            reconstructed,
            signal,
            atol=5e-13,
            rtol=5e-13,
            message=f"real STFT reconstruction case {case}",
            section="stft",
        )
        streamed = stream_stft(
            signal,
            _partition(length, rng),
            frame_length=frame,
            hop_length=hop,
            fft_length=fft_length,
            pad_end=True,
        )
        audit.check(streamed.valid, f"stream STFT certificate case {case}", section="streaming_stft")

    for case in range(40):
        length = int(rng.integers(64, 800))
        frame = int(rng.choice((16, 32, 64)))
        hop = frame // 4
        signal = rng.normal(size=length) + 1j * rng.normal(size=length)
        transform = stft(
            signal,
            frame_length=frame,
            hop_length=hop,
            onesided=False,
            center=True,
            pad_end=True,
        )
        audit.allclose(
            transform.inverse().require(),
            signal,
            atol=7e-13,
            rtol=7e-13,
            message=f"complex STFT reconstruction case {case}",
            section="stft",
        )
        streamed = stream_stft(
            signal,
            _partition(length, rng),
            frame_length=frame,
            hop_length=hop,
            onesided=False,
        )
        audit.check(streamed.valid, f"complex stream STFT case {case}", section="streaming_stft")

    # Spectral identities and deterministic transfer estimates.
    for case in range(200):
        length = int(rng.integers(2, 600))
        signal = rng.normal(size=length)
        audit.check(parseval_certificate(signal).valid, f"real Parseval {case}", section="spectral")
        complex_signal = signal + 1j * rng.normal(size=length)
        audit.check(parseval_certificate(complex_signal).valid, f"complex Parseval {case}", section="spectral")

    for bin_index in range(1, 101):
        length = 512
        signal = np.sin(2.0 * np.pi * bin_index * np.arange(length) / length)
        estimate = periodogram(signal, sample_rate=float(length), window="boxcar", detrend="none")
        audit.check(
            estimate.dominant_frequency() == float(bin_index),
            f"periodogram bin {bin_index}",
            section="spectral",
        )
        audit.check(estimate.certificate.valid, f"periodogram certificate {bin_index}", section="spectral")

    for case in range(60):
        source = rng.normal(size=2048)
        gain = float(rng.uniform(-3.0, 3.0))
        if abs(gain) < 0.1:
            gain += 0.25
        report = cross_spectrum(source, gain * source, frame_length=128, hop_length=64, detrend="none")
        active = report.input_power > 1e-12
        audit.allclose(
            report.coherence[active],
            np.ones(np.count_nonzero(active)),
            atol=3e-13,
            message=f"coherence case {case}",
            section="spectral",
        )
        audit.allclose(
            report.transfer_h1[active],
            np.full(np.count_nonzero(active), gain),
            atol=5e-12,
            rtol=5e-12,
            message=f"H1 gain case {case}",
            section="spectral",
        )
        welch = welch_psd(source, frame_length=128, hop_length=64)
        audit.check(welch.certificate.valid, f"Welch case {case}", section="spectral")

    # FIR design, direct convolution, and chunk invariance.
    designs = (
        design_lowpass(0.12, num_taps=63),
        design_highpass(0.12, num_taps=63),
        design_bandpass((0.1, 0.2), num_taps=63),
    )
    for design in designs:
        audit.check(design.certificate(tolerance=1e-9).valid, f"FIR design {design.kind}", section="fir")
        audit.allclose(
            design.taps,
            design.taps[::-1],
            atol=2e-15,
            message=f"FIR symmetry {design.kind}",
            section="fir",
        )
    for case in range(160):
        length = int(rng.integers(1, 1200))
        tap_count = int(rng.choice((1, 3, 5, 9, 17, 31)))
        signal = rng.normal(size=length)
        taps = rng.normal(size=tap_count)
        streamed = stream_fir_filter(signal, taps, _partition(length, rng))
        audit.check(streamed.valid, f"stream FIR certificate {case}", section="fir")
        audit.allclose(
            streamed.value,
            fir_filter(signal, taps, mode="causal"),
            atol=3e-13,
            rtol=3e-13,
            message=f"stream FIR case {case}",
            section="fir",
        )

    # Rational resampling against the independent explicit reference.
    for case in range(180):
        length = int(rng.integers(1, 180))
        up = int(rng.integers(1, 9))
        down = int(rng.integers(1, 9))
        signal = rng.normal(size=length)
        result = resample_poly_result(signal, up, down)
        audit.check(result.certificate.valid, f"resampling certificate {case}", section="resampling")
        audit.allclose(
            result.signal,
            result.plan.explicit_reference(signal),
            atol=4e-13,
            rtol=4e-13,
            message=f"resampling reference {case}",
            section="resampling",
        )

    # Adaptive system identification under deterministic persistent excitation.
    for case in range(30):
        source = rng.normal(size=1800)
        target = rng.uniform(-0.8, 0.8, size=3)
        desired = fir_filter(source, target, mode="causal")
        nlms = nlms_filter(source, desired, 3, step_size=0.8)
        audit.check(nlms.certificate.valid, f"NLMS certificate {case}", section="adaptive")
        audit.allclose(
            nlms.final_weights,
            target,
            atol=2e-8,
            rtol=2e-8,
            message=f"NLMS identification {case}",
            section="adaptive",
        )
        rls = rls_filter(source, desired, 3, forgetting_factor=0.999, delta=10.0)
        audit.check(rls.certificate.valid, f"RLS certificate {case}", section="adaptive")
        audit.allclose(
            rls.final_weights,
            target,
            atol=5e-5,
            rtol=5e-5,
            message=f"RLS identification {case}",
            section="adaptive",
        )

    # Haar reconstruction and coefficient shrinkage.
    for case in range(120):
        levels = int(rng.integers(1, 7))
        length = int(2**levels * rng.integers(1, 12))
        signal = rng.normal(size=length)
        decomposition = haar_decompose(signal, levels)
        audit.check(decomposition.certificate(signal).valid, f"Haar certificate {case}", section="denoise")
        audit.allclose(
            decomposition.reconstruct(),
            signal,
            atol=5e-13,
            rtol=5e-13,
            message=f"Haar reconstruction {case}",
            section="denoise",
        )
        coefficients = rng.normal(size=100)
        threshold = float(rng.uniform(0.0, 2.0))
        for mode in ("hard", "soft", "garrote"):
            reduced = threshold_coefficients(coefficients, threshold, mode=mode)
            audit.check(
                bool(np.all(np.abs(reduced) <= np.abs(coefficients) + 1e-14)),
                f"threshold monotonicity {case}/{mode}",
                section="denoise",
            )

    clean = np.repeat(np.array([0.0, 2.0, -1.0, 1.5]), 512)
    for case in range(20):
        noisy = clean + rng.normal(scale=0.3, size=clean.size)
        result = wavelet_denoise(noisy, levels=5, rule="universal", mode="soft")
        audit.check(result.certificate.valid, f"denoise certificate {case}", section="denoise")
        audit.check(
            denoising_risk_certificate(clean, noisy, result.signal).valid,
            f"observed denoise risk {case}",
            section="denoise",
        )

    # Finite recursive filters, independent difference equations, and biquad constraints.
    kinds = ("lowpass", "highpass", "bandpass", "notch", "allpass")
    for case in range(120):
        kind = kinds[case % len(kinds)]
        sample_rate = float(rng.choice((800.0, 1000.0, 2048.0, 48000.0)))
        frequency = float(rng.uniform(0.01, 0.46) * sample_rate)
        quality = float(rng.uniform(0.25, 8.0))
        model = design_biquad(kind, frequency, q=quality, sample_rate=sample_rate)
        audit.check(model.stability_certificate().valid, f"biquad stability {case}", section="recursive")
        signal = rng.normal(size=int(rng.integers(1, 1200)))
        if case % 4 == 0:
            signal = signal + 1j * rng.normal(size=signal.size)
        batch = model.process(signal)
        independent = _difference_equation(model.numerator, model.denominator, signal)
        audit.allclose(
            batch.samples,
            independent,
            atol=8e-13,
            rtol=8e-13,
            message=f"recursive difference equation {case}",
            section="recursive",
        )
        streamed = stream_lfilter(model, signal, _partition(signal.size, rng))
        audit.check(streamed.certificate.valid, f"recursive stream certificate {case}", section="recursive")
        audit.allclose(
            streamed.value,
            batch.samples,
            atol=8e-13,
            rtol=8e-13,
            message=f"recursive stream equality {case}",
            section="recursive",
        )

    response_points = 1025
    response_rate = 2000.0
    response_frequency = 250.0
    low = design_biquad("lowpass", response_frequency, sample_rate=response_rate).response(
        points=response_points, sample_rate=response_rate
    )
    high = design_biquad("highpass", response_frequency, sample_rate=response_rate).response(
        points=response_points, sample_rate=response_rate
    )
    notch = design_biquad("notch", response_frequency, q=4.0, sample_rate=response_rate).response(
        points=response_points, sample_rate=response_rate
    )
    allpass = design_biquad("allpass", response_frequency, q=1.2, sample_rate=response_rate).response(
        points=response_points, sample_rate=response_rate
    )
    audit.check(abs(low.magnitude[0] - 1.0) < 2e-14, "lowpass DC gain", section="recursive")
    audit.check(low.magnitude[-1] < 2e-14, "lowpass Nyquist rejection", section="recursive")
    audit.check(high.magnitude[0] < 2e-14, "highpass DC rejection", section="recursive")
    audit.check(abs(high.magnitude[-1] - 1.0) < 2e-14, "highpass Nyquist gain", section="recursive")
    notch_index = int(np.argmin(np.abs(notch.frequencies - response_frequency)))
    audit.check(notch.magnitude[notch_index] < 2e-13, "notch center rejection", section="recursive")
    audit.allclose(
        allpass.magnitude,
        np.ones(response_points),
        atol=4e-14,
        message="allpass magnitude identity",
        section="recursive",
    )
    unstable = LinearFilter([1.0], [1.0, -1.01])
    audit.check(not unstable.stability_certificate().valid, "unstable pole rejection", section="recursive")

    # FFT convolution and stateful overlap-add against direct finite convolution.
    for case in range(180):
        signal_length = int(rng.integers(1, 1400))
        kernel_length = int(rng.integers(1, min(signal_length + 25, 160)))
        signal = rng.normal(size=signal_length)
        kernel = rng.normal(size=kernel_length)
        if case % 3 == 0:
            signal = signal + 1j * rng.normal(size=signal_length)
        if case % 5 == 0:
            kernel = kernel + 1j * rng.normal(size=kernel_length)
        reference = np.convolve(signal, kernel)
        audit.allclose(
            fft_convolve(signal, kernel),
            reference,
            atol=2e-12,
            rtol=2e-12,
            message=f"FFT convolution {case}",
            section="fft_convolution",
        )
        block_length = int(rng.choice((1, 2, 7, 31, 64, 257)))
        result = certified_overlap_add_convolve(signal, kernel, block_length=block_length)
        audit.check(result.certificate.valid, f"overlap-add certificate {case}", section="fft_convolution")
        audit.allclose(
            result.value,
            reference,
            atol=3e-12,
            rtol=3e-12,
            message=f"overlap-add convolution {case}",
            section="fft_convolution",
        )
        processor = OverlapAddConvolver(kernel, block_length=block_length)
        pieces: list[np.ndarray] = []
        offset = 0
        for size in _partition(signal_length, rng):
            pieces.append(processor.process(signal[offset : offset + size]))
            offset += size
        pieces.append(processor.flush())
        streamed = np.concatenate(pieces)
        audit.allclose(
            streamed,
            reference,
            atol=3e-12,
            rtol=3e-12,
            message=f"stateful overlap-add {case}",
            section="fft_convolution",
        )
        audit.check(processor.processed_samples == signal_length, f"overlap-add count {case}", section="fft_convolution")

    report = {
        "schema": "komorebi-math.block12-deep-audit",
        "package_version": __version__,
        "seed": 12012026,
        "passed": not audit.failures,
        "checks": audit.checks,
        "failures": audit.failures,
        "sections": dict(sorted(audit.sections.items())),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "counting_convention": "Each scalar Boolean or elementwise numerical comparison counts as one check.",
    }
    return report


def main() -> int:
    """Run the audit, write its report, and return a process status."""

    report = run()
    destination = Path(__file__).with_name("BLOCK12_DEEP_AUDIT.json")
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
