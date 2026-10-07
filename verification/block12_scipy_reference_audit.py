"""Optional SciPy reference audit for the Block 12 finite DSP conventions."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy import signal as scipy_signal

from komorebi_math import __version__
from komorebi_math.signals import (
    design_bandpass,
    design_bandstop,
    design_highpass,
    design_lowpass,
    design_resample_filter,
    fft_convolve,
    get_window,
    lfilter,
    overlap_add_convolve,
    periodogram,
    resample_poly,
    welch_psd,
)


@dataclass(slots=True)
class ReferenceAudit:
    """Accumulate elementwise comparisons against an independent library."""

    checks: int = 0
    failures: list[str] = field(default_factory=list)
    sections: dict[str, int] = field(default_factory=dict)

    def allclose(
        self,
        observed: Any,
        expected: Any,
        *,
        atol: float,
        rtol: float,
        message: str,
        section: str,
    ) -> None:
        """Compare two finite arrays and count every scalar relation."""

        left = np.asarray(observed)
        right = np.asarray(expected)
        if left.shape != right.shape:
            self.checks += 1
            self.sections[section] = self.sections.get(section, 0) + 1
            self.failures.append(f"{message}: shape {left.shape} != {right.shape}")
            return
        matches = np.isclose(left, right, atol=atol, rtol=rtol, equal_nan=False)
        count = int(matches.size)
        self.checks += count
        self.sections[section] = self.sections.get(section, 0) + count
        if not bool(np.all(matches)) and len(self.failures) < 100:
            index = int(np.flatnonzero(~matches.ravel())[0])
            self.failures.append(
                f"{message}: mismatch at flat index {index}; "
                f"observed={left.ravel()[index]!r}; expected={right.ravel()[index]!r}"
            )


def _partition(length: int, rng: np.random.Generator) -> tuple[int, ...]:
    sizes: list[int] = []
    remaining = length
    while remaining:
        size = int(rng.integers(1, min(remaining, 113) + 1))
        sizes.append(size)
        remaining -= size
    return tuple(sizes)


def run() -> dict[str, Any]:
    """Run deterministic convention-matched comparisons against SciPy."""

    rng = np.random.default_rng(12122026)
    audit = ReferenceAudit()

    window_specs: tuple[Any, ...] = (
        "boxcar",
        "hann",
        "hamming",
        "blackman",
        "bartlett",
        ("kaiser", 7.0),
        ("tukey", 0.4),
    )
    for spec in window_specs:
        for length in (1, 2, 7, 15, 16, 31, 32, 63, 64, 127, 128):
            for periodic in (False, True):
                audit.allclose(
                    get_window(spec, length, periodic=periodic),
                    scipy_signal.get_window(spec, length, fftbins=periodic),
                    atol=2e-14,
                    rtol=2e-14,
                    message=f"window {spec!r}/{length}/{periodic}",
                    section="windows",
                )

    designs = (
        (design_lowpass(0.12, num_taps=63), 0.12, True),
        (design_highpass(0.12, num_taps=63), 0.12, False),
        (design_bandpass((0.10, 0.20), num_taps=63), (0.10, 0.20), False),
        (design_bandstop((0.10, 0.20), num_taps=63), (0.10, 0.20), True),
    )
    for design, cutoff, pass_zero in designs:
        reference = scipy_signal.firwin(
            design.taps.size,
            cutoff,
            pass_zero=pass_zero,
            window="hann",
            scale=True,
            fs=design.sample_rate,
        )
        audit.allclose(
            design.taps,
            reference,
            atol=3e-14,
            rtol=3e-14,
            message=f"FIR design {design.kind}",
            section="fir_design",
        )

    for case in range(120):
        length = int(rng.integers(1, 1800))
        numerator_length = int(rng.integers(1, 12))
        denominator_length = int(rng.integers(1, 8))
        source = rng.normal(size=length)
        numerator = rng.normal(size=numerator_length)
        denominator = np.concatenate(([float(rng.uniform(0.5, 2.0))], rng.normal(scale=0.08, size=denominator_length - 1)))
        if case % 4 == 0:
            source = source + 1j * rng.normal(size=length)
            numerator = numerator + 1j * rng.normal(scale=0.05, size=numerator_length)
        audit.allclose(
            lfilter(numerator, denominator, source).samples,
            scipy_signal.lfilter(numerator, denominator, source),
            atol=4e-13,
            rtol=4e-13,
            message=f"recursive filter {case}",
            section="recursive",
        )

    for case in range(160):
        source = rng.normal(size=int(rng.integers(1, 1600)))
        kernel = rng.normal(size=int(rng.integers(1, 190)))
        if case % 3 == 0:
            source = source + 1j * rng.normal(size=source.size)
        if case % 5 == 0:
            kernel = kernel + 1j * rng.normal(size=kernel.size)
        reference = scipy_signal.fftconvolve(source, kernel, mode="full")
        audit.allclose(
            fft_convolve(source, kernel),
            reference,
            atol=3e-12,
            rtol=3e-12,
            message=f"FFT convolution {case}",
            section="fft_convolution",
        )
        audit.allclose(
            overlap_add_convolve(source, kernel, block_length=int(rng.choice((1, 7, 31, 127)))),
            scipy_signal.oaconvolve(source, kernel, mode="full"),
            atol=4e-12,
            rtol=4e-12,
            message=f"overlap-add convolution {case}",
            section="fft_convolution",
        )

    for case in range(80):
        length = int(rng.integers(64, 1800))
        sample_rate = float(rng.choice((800.0, 1000.0, 4096.0)))
        source = rng.normal(size=length)
        ours = periodogram(source, sample_rate=sample_rate, window="hann", detrend="constant")
        symmetric_window = scipy_signal.get_window("hann", length, fftbins=False)
        frequencies, power = scipy_signal.periodogram(
            source,
            fs=sample_rate,
            window=symmetric_window,
            detrend="constant",
            scaling="density",
            return_onesided=True,
        )
        audit.allclose(
            ours.frequencies,
            frequencies,
            atol=2e-12,
            rtol=2e-14,
            message=f"periodogram frequencies {case}",
            section="spectral",
        )
        audit.allclose(
            ours.power,
            power,
            atol=3e-14,
            rtol=3e-13,
            message=f"periodogram power {case}",
            section="spectral",
        )

    for case in range(80):
        length = int(rng.integers(300, 2200))
        frame_length = int(rng.choice((32, 64, 128, 256)))
        hop = int(rng.choice((frame_length // 4, frame_length // 2)))
        source = rng.normal(size=length)
        ours = welch_psd(
            source,
            sample_rate=1000.0,
            frame_length=frame_length,
            hop_length=hop,
            window="hann",
            detrend="constant",
        )
        frequencies, power = scipy_signal.welch(
            source,
            fs=1000.0,
            window="hann",
            nperseg=frame_length,
            noverlap=frame_length - hop,
            nfft=frame_length,
            detrend="constant",
            return_onesided=True,
            scaling="density",
            average="mean",
        )
        audit.allclose(
            ours.frequencies,
            frequencies,
            atol=2e-14,
            rtol=2e-14,
            message=f"Welch frequencies {case}",
            section="spectral",
        )
        audit.allclose(
            ours.power,
            power,
            atol=3e-14,
            rtol=3e-13,
            message=f"Welch power {case}",
            section="spectral",
        )

    for case in range(120):
        length = int(rng.integers(1, 500))
        up = int(rng.integers(1, 9))
        down = int(rng.integers(1, 9))
        source = rng.normal(size=length)
        if case % 5 == 0:
            source = source + 1j * rng.normal(size=length)
        taps = design_resample_filter(up, down)
        reduced_up = up // np.gcd(up, down)
        # SciPy multiplies user-supplied taps by the reduced up factor internally.
        reference = scipy_signal.resample_poly(
            source,
            up,
            down,
            window=taps / reduced_up,
            padtype="constant",
        )
        audit.allclose(
            resample_poly(source, up, down, taps=taps),
            reference,
            atol=2e-12,
            rtol=2e-12,
            message=f"rational resampling {case}",
            section="resampling",
        )

    return {
        "schema": "komorebi-math.block12-scipy-reference-audit",
        "package_version": __version__,
        "seed": 12122026,
        "passed": not audit.failures,
        "checks": audit.checks,
        "failures": audit.failures,
        "sections": dict(sorted(audit.sections.items())),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "counting_convention": "Each elementwise NumPy-SciPy comparison counts as one check.",
        "scope": "Convention-matched finite reference comparisons; not a claim of API identity.",
    }


def main() -> int:
    """Run the reference audit, persist JSON evidence, and return its status."""

    report = run()
    destination = Path(__file__).with_name("BLOCK12_SCIPY_REFERENCE_AUDIT.json")
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
