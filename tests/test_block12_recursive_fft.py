from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.exceptions import DimensionError, ValidationError
from komorebi_math.signals import (
    LinearFilter,
    OverlapAddConvolver,
    StreamingLinearFilter,
    certified_overlap_add_convolve,
    design_biquad,
    fft_convolve,
    lfilter,
    overlap_add_convolve,
    stream_lfilter,
)


def _difference_equation(
    numerator: np.ndarray,
    denominator: np.ndarray,
    signal: np.ndarray,
) -> np.ndarray:
    b = np.asarray(numerator) / denominator[0]
    a = np.asarray(denominator) / denominator[0]
    result = np.zeros(signal.size, dtype=np.result_type(b, a, signal))
    for index in range(signal.size):
        for offset, coefficient in enumerate(b):
            if index >= offset:
                result[index] += coefficient * signal[index - offset]
        for offset, coefficient in enumerate(a[1:], start=1):
            if index >= offset:
                result[index] -= coefficient * result[index - offset]
    return result


def test_linear_filter_matches_independent_difference_equation() -> None:
    rng = np.random.default_rng(12501)
    signal = rng.normal(size=257)
    numerator = np.array([0.3, -0.1, 0.2, 0.05])
    denominator = np.array([1.4, -0.35, 0.12])
    result = LinearFilter(numerator, denominator).process(signal)
    reference = _difference_equation(numerator, denominator, signal)
    assert np.allclose(result.samples, reference, atol=2e-15)
    assert result.final_state.shape == (3,)


def test_linear_filter_handles_complex_coefficients_and_state_continuation() -> None:
    rng = np.random.default_rng(12502)
    signal = rng.normal(size=300) + 1j * rng.normal(size=300)
    model = LinearFilter(
        np.array([0.4 + 0.2j, -0.1j, 0.05]),
        np.array([1.0, -0.2 + 0.1j, 0.04]),
    )
    first = model.process(signal[:137])
    second = model.process(signal[137:], initial_state=first.final_state)
    batch = model.process(signal)
    assert np.allclose(np.concatenate((first.samples, second.samples)), batch.samples, atol=2e-15)
    assert np.allclose(second.final_state, batch.final_state, atol=2e-15)


def test_lfilter_wrapper_and_streaming_chunks_match_batch() -> None:
    rng = np.random.default_rng(12503)
    signal = rng.normal(size=1000)
    model = LinearFilter([0.2, 0.4, 0.2], [1.0, -0.3, 0.08])
    wrapped = lfilter(model.numerator, model.denominator, signal)
    batch = model.process(signal)
    assert np.allclose(wrapped.samples, batch.samples)
    streamed = stream_lfilter(model, signal, (1, 17, 201, 3, 778))
    streamed.certificate.require()
    assert np.array_equal(streamed.value, batch.samples)


def test_streaming_linear_filter_reset_restores_zero_state() -> None:
    rng = np.random.default_rng(12504)
    signal = rng.normal(size=200)
    model = LinearFilter([1.0, 0.5], [1.0, -0.25])
    processor = StreamingLinearFilter(model)
    first = np.concatenate((processor.process(signal[:31]), processor.process(signal[31:])))
    assert processor.processed_samples == signal.size
    processor.reset()
    second = processor.process(signal)
    assert processor.processed_samples == signal.size
    assert np.array_equal(first, second)


@pytest.mark.parametrize("kind", ["lowpass", "highpass", "bandpass", "notch", "allpass"])
def test_biquad_designs_are_strictly_stable(kind: str) -> None:
    model = design_biquad(kind, 120.0, q=0.8, sample_rate=1000.0)
    model.stability_certificate().require()
    assert np.max(np.abs(model.poles)) < 1.0


def test_biquad_frequency_constraints() -> None:
    low = design_biquad("lowpass", 100.0, sample_rate=1000.0).response(
        points=1001, sample_rate=1000.0
    )
    high = design_biquad("highpass", 100.0, sample_rate=1000.0).response(
        points=1001, sample_rate=1000.0
    )
    notch = design_biquad("notch", 100.0, q=4.0, sample_rate=1000.0).response(
        points=1001, sample_rate=1000.0
    )
    allpass = design_biquad("allpass", 100.0, q=1.2, sample_rate=1000.0).response(
        points=1001, sample_rate=1000.0
    )
    assert low.magnitude[0] == pytest.approx(1.0, abs=1e-14)
    assert low.magnitude[-1] < 1e-14
    assert high.magnitude[0] < 1e-14
    assert high.magnitude[-1] == pytest.approx(1.0, abs=1e-14)
    notch_index = int(np.argmin(np.abs(notch.frequencies - 100.0)))
    assert notch.magnitude[notch_index] < 1e-13
    assert np.allclose(allpass.magnitude, 1.0, atol=2e-14)


def test_unstable_recursive_filter_fails_stability_certificate() -> None:
    model = LinearFilter([1.0], [1.0, -1.01])
    certificate = model.stability_certificate()
    assert not certificate.valid
    assert certificate.residuals["maximum_pole_radius"] == pytest.approx(1.01)


def test_recursive_filter_validation() -> None:
    with pytest.raises(ValidationError):
        LinearFilter([1.0], [0.0, 1.0])
    with pytest.raises(DimensionError):
        LinearFilter([1.0], [1.0, -0.2]).process([1.0], initial_state=[0.0, 0.0])
    with pytest.raises(ValidationError):
        design_biquad("lowpass", 500.0, sample_rate=1000.0)
    with pytest.raises(ValidationError):
        design_biquad("unknown", 100.0, sample_rate=1000.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", ["full", "causal", "same", "valid"])
def test_fft_convolution_matches_direct_cropping(mode: str) -> None:
    rng = np.random.default_rng(12505)
    signal = rng.normal(size=137)
    kernel = rng.normal(size=29)
    full = np.convolve(signal, kernel, mode="full")
    if mode == "full":
        reference = full
    elif mode == "causal":
        reference = full[: signal.size]
    elif mode == "same":
        reference = full[(kernel.size - 1) // 2 : (kernel.size - 1) // 2 + signal.size]
    else:
        reference = full[kernel.size - 1 : kernel.size - 1 + signal.size - kernel.size + 1]
    assert np.allclose(fft_convolve(signal, kernel, mode=mode), reference, atol=4e-14)


def test_fft_convolution_handles_complex_inputs() -> None:
    rng = np.random.default_rng(12506)
    signal = rng.normal(size=80) + 1j * rng.normal(size=80)
    kernel = rng.normal(size=17) - 1j * rng.normal(size=17)
    assert np.allclose(fft_convolve(signal, kernel), np.convolve(signal, kernel), atol=4e-14)


@pytest.mark.parametrize("block_length", [1, 2, 7, 31, 256])
def test_overlap_add_matches_direct_convolution(block_length: int) -> None:
    rng = np.random.default_rng(12507 + block_length)
    signal = rng.normal(size=311)
    kernel = rng.normal(size=47)
    result = certified_overlap_add_convolve(signal, kernel, block_length=block_length)
    result.certificate.require()
    assert np.allclose(result.value, np.convolve(signal, kernel), atol=5e-13)


def test_stateful_overlap_add_accepts_irregular_and_complex_chunks() -> None:
    rng = np.random.default_rng(12508)
    signal = rng.normal(size=501) + 1j * rng.normal(size=501)
    kernel = rng.normal(size=53)
    processor = OverlapAddConvolver(kernel, block_length=37)
    sizes = (1, 80, 7, 200, 3, 210)
    pieces: list[np.ndarray] = []
    offset = 0
    for size in sizes:
        pieces.append(processor.process(signal[offset : offset + size]))
        offset += size
    pieces.append(processor.flush())
    streamed = np.concatenate(pieces)
    assert np.allclose(streamed, np.convolve(signal, kernel), atol=5e-13)
    assert processor.processed_samples == signal.size
    assert processor.flush().size == 0
    with pytest.raises(ValidationError):
        processor.process(np.ones(1))


def test_overlap_add_empty_flush_and_reset() -> None:
    processor = OverlapAddConvolver(np.array([1.0, 2.0, 1.0]), block_length=4)
    assert processor.flush().size == 0
    processor.reset()
    result = np.concatenate((processor.process(np.arange(6.0)), processor.flush()))
    assert np.allclose(result, np.convolve(np.arange(6.0), np.array([1.0, 2.0, 1.0])))


def test_overlap_add_modes_and_validation() -> None:
    signal = np.arange(10.0)
    kernel = np.array([0.25, 0.5, 0.25])
    assert np.allclose(
        overlap_add_convolve(signal, kernel, block_length=3, mode="causal"),
        np.convolve(signal, kernel)[: signal.size],
    )
    with pytest.raises(DimensionError):
        fft_convolve([], kernel)
    with pytest.raises(ValidationError):
        OverlapAddConvolver(kernel, block_length=0)
