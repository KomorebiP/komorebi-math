# Production finite signal processing

The signal-processing namespace combines polyphase decomposition, filter banks, wavelet packets, singular packets, and jump recovery with a general finite digital signal-processing layer. The implementation is available from `komorebi_math.signals` and depends only on NumPy.

The word **finite** is controlling. Every routine operates on an explicitly supplied finite array or a finite ordered stream of chunks. Certificates verify algebraic identities, reconstruction residuals, declared parameter ranges, and batch-versus-streaming equivalence. They do not infer stochastic stationarity, asymptotic consistency, real-time deadlines, or application-specific model validity from one data record.

## Scope

The production layer contains ten connected components:

1. deterministic analysis windows and overlap diagnostics
2. finite framing, STFT analysis, inverse synthesis, and spectrograms
3. periodograms, Welch estimates, cross spectra, coherence, and transfer estimates
4. linear-phase FIR design, direct filtering, and causal streaming execution
5. integer upsampling and downsampling with rational polyphase resampling
6. finite LMS, normalized LMS, and recursive least-squares adaptation
7. multilevel Haar decomposition and coefficient shrinkage
8. chunk-invariant streaming STFT analysis
9. finite FIR/IIR recursion, continuation state, and biquad design
10. direct FFT and stateful overlap-add convolution

These components are additive. The earlier packet, polyphase, lifting, singular-recovery, and nonlinear signal APIs remain available.

## Windows and finite overlap geometry

`get_window` constructs boxcar, Hann, Hamming, Blackman, Bartlett, Kaiser, and Tukey windows under explicit periodic or symmetric conventions. An array supplied by the caller is accepted as a custom window only when its length and entries are valid.

```python
from komorebi_math.signals import (
    canonical_dual_window,
    get_window,
    overlap_add_certificate,
    window_metrics,
)

window = get_window("hann", 128, periodic=True)
metrics = window_metrics(window, sample_rate=1024.0)
assert metrics.coherent_gain == 0.5

overlap_add_certificate(
    window,
    32,
    squared=True,
    require_constant=True,
).require()

dual = canonical_dual_window(window, 32).require()
```

For a window `w` of length `L` and hop `H`, `overlap_profile` computes the periodic residue-class profile

\[
P_r = \sum_{n \equiv r \pmod H} w_n
\]

or its squared counterpart. `overlap_add_certificate` can require positivity only or constant overlap. `canonical_dual_window` uses the finite residue-class denominator to construct a synthesis window whose cross-overlap profile is one whenever that denominator is nonzero.

The dual-window certificate is a finite identity. It does not assert a frame theorem for an unspecified infinite signal space.

## Framing, STFT, and inverse synthesis

`frame_signal` defines one uniform finite framing convention. The last frame is included only when complete unless `pad_end=True`. Centering and boundary padding are explicit metadata.

```python
import numpy as np
from komorebi_math.signals import stft

rng = np.random.default_rng(12001)
signal = rng.normal(size=1001)

transform = stft(
    signal,
    sample_rate=8000.0,
    frame_length=127,
    hop_length=31,
    fft_length=256,
    window="hann",
    center=True,
    pad_end=True,
)

reconstructed = transform.inverse().require()
transform.roundtrip_certificate(signal).require()
assert np.allclose(reconstructed, signal, atol=2e-15)
```

The STFT stores all metadata required by normalized overlap-add synthesis. For analysis window `w`, synthesis window `v`, frame index `m`, and hop `H`, reconstruction divides by the represented finite denominator

\[
D[n] = \sum_m w[n-mH]v[n-mH].
\]

A reconstruction certificate fails when this denominator vanishes on a required sample. Constant framewise detrending is deliberately marked noninvertible because the removed means are not retained.

Real input supports one-sided Fourier coordinates. Complex input requires a two-sided transform.
Complex analysis windows also require two-sided coordinates. Inverse and roundtrip certificates check finite coefficients and synthesis output in addition to overlap coverage. A numerical overflow cannot pass a certificate merely because both an error and its reference scale become infinite.

## Spectrograms and spectral estimates

`spectrogram` derives magnitude, power, spectrum, or density values from the same finite STFT record. `periodogram` and `welch_psd` expose finite power-spectrum records with frequency coordinates, scaling metadata, integration helpers, and nonnegativity certificates.

```python
import numpy as np
from komorebi_math.signals import periodogram, welch_psd

sample_rate = 1024.0
index = np.arange(1024)
signal = 2.0 * np.sin(2.0 * np.pi * 91.0 * index / sample_rate)

line = periodogram(
    signal,
    sample_rate=sample_rate,
    window="boxcar",
    detrend="none",
)
assert line.dominant_frequency() == 91.0
line.certificate.require()

estimate = welch_psd(
    signal,
    sample_rate=sample_rate,
    frame_length=256,
    hop_length=128,
    average="mean",
)
estimate.certificate.require()
```

For a deterministic relation `y = g x`, `cross_spectrum` reports input and output power, cross power, magnitude-squared coherence, and the regularized H1 estimate

\[
\widehat H_1(f) = \frac{S_{yx}(f)}{S_{xx}(f)+\varepsilon}.
\]

```python
import numpy as np

time = np.arange(1024) / 1000.0
signal = np.sin(2 * np.pi * 60.0 * time)
from komorebi_math.signals import cross_spectrum

report = cross_spectrum(
    signal,
    -2.25 * signal,
    frame_length=256,
    hop_length=128,
    detrend="none",
)
report.certificate.require()
```

Coherence is clipped to its finite numerical range. A high observed coherence is a property of the supplied record and estimator. It is not a causal-identification theorem.

`parseval_certificate` separately checks the finite DFT energy identity for real or complex records.
Coherence normalization avoids multiplying the two auto spectra before normalization, preserving the supplied-record coherence when those finite spectra are small enough that their product would underflow.

## FIR design and filtering

`design_fir` implements odd-length real linear-phase windowed-sinc designs. Convenience constructors are available for lowpass, highpass, bandpass, and bandstop filters.

```python
import numpy as np
from komorebi_math.signals import design_lowpass, fir_filter

lowpass = design_lowpass(
    100.0,
    num_taps=101,
    sample_rate=1000.0,
    window=("kaiser", 8.0),
)
lowpass.certificate(tolerance=1e-9).require()

filtered = fir_filter(np.arange(1000.0), lowpass, mode="causal")
response = lowpass.response(points=1001)
assert lowpass.group_delay_samples == 50.0
```

The finite modes are:

- `full`: complete finite convolution
- `causal`: first `len(signal)` samples of complete convolution
- `same`: centered signal-length slice
- `valid`: samples with complete kernel overlap

`StreamingFIR` retains exactly `num_taps - 1` prior input samples. Its `process` method emits one causal output per accepted input sample. `flush` emits the remaining full-convolution tail once.
FIR arithmetic promotes fixed-width integer data to a floating or complex computation type before products and sums. A complex history remains complex when a subsequent chunk contains only real samples.

```python
import numpy as np

time = np.arange(1024) / 1000.0
signal = np.sin(2 * np.pi * 60.0 * time)
from komorebi_math.signals import design_lowpass, stream_fir_filter

lowpass = design_lowpass(100.0, num_taps=101, sample_rate=1000.0)

result = stream_fir_filter(
    signal,
    lowpass,
    chunk_sizes=(113, 287, 19, 605),
)
result.certificate.require()
```

The stream certificate compares the chunked output with the direct causal finite convolution. It does not make a wall-clock latency claim.

## Recursive FIR and IIR filtering

`LinearFilter` normalizes a finite rational transfer function and executes it in direct-form II transposed form. The returned `FilterOutput` contains both samples and the exact finite continuation state required by the next chunk.
The computation type includes the supplied initial state. Real input chunks and real filter coefficients therefore preserve any imaginary continuation state from preceding complex samples.

```python
import numpy as np

time = np.arange(1024) / 1000.0
signal = np.sin(2 * np.pi * 60.0 * time)
from komorebi_math.signals import design_biquad, stream_lfilter

biquad = design_biquad("lowpass", 120.0, q=0.8, sample_rate=1000.0)
biquad.stability_certificate().require()
streamed = stream_lfilter(biquad, signal, (1, 17, 300, 706))
streamed.certificate.require()
```

The stability certificate checks that every numerically computed pole lies strictly inside the unit disk by the declared tolerance. This is a finite coefficient-level BIBO stability diagnostic. It does not certify robustness to coefficient quantization not represented in the supplied model.

## FFT and overlap-add convolution

`fft_convolve` computes finite convolution through zero-padded FFT multiplication. `OverlapAddConvolver` processes bounded blocks, carries the exact overlap tail, and emits the remaining tail once at flush.

```python
import numpy as np

time = np.arange(1024) / 1000.0
signal = np.sin(2 * np.pi * 60.0 * time)
from komorebi_math.signals import design_lowpass, certified_overlap_add_convolve

lowpass = design_lowpass(100.0, num_taps=101, sample_rate=1000.0)

result = certified_overlap_add_convolve(
    signal,
    lowpass.taps,
    block_length=256,
)
result.certificate.require()
```

The certificate compares overlap-add output with direct finite convolution under the same crop convention. It establishes numerical equality within tolerance, not an unconditional speedup.

## Rational polyphase resampling

The resampling convention is explicit:

1. reduce `up/down` by their greatest common divisor
2. insert `up - 1` zeros between samples
3. apply the declared finite anti-imaging and anti-aliasing FIR
4. compensate the symmetric filter delay
5. retain every `down`-th sample
6. return exactly `ceil(N up / down)` samples

```python
import numpy as np
from komorebi_math.signals import resample_poly_result

signal = np.linspace(0.0, 1.0, 257)
result = resample_poly_result(signal, 5, 3)
result.certificate.require()
assert result.signal.size == 429
```

`RationalResamplingPlan.explicit_reference` evaluates the corresponding zero-insertion and direct-convolution construction. The certificate compares the polyphase implementation with that independent finite reference.

The default filter is a finite windowed-sinc design. It is not presented as a universally optimal resampling filter.

## Adaptive filters

The adaptive layer provides finite LMS, normalized LMS, and RLS recursions. Each result stores output, error, final coefficients, complete coefficient history, and finite diagnostics.

```python
import numpy as np
from komorebi_math.signals import fir_filter, nlms_filter

rng = np.random.default_rng(1205)
source = rng.normal(size=3000)
target = np.array([0.7, -0.35, 0.15])
desired = fir_filter(source, target, mode="causal")

identified = nlms_filter(source, desired, 3, step_size=0.8)
identified.certificate.require()
assert np.allclose(identified.final_weights, target, atol=1e-8)
```

Complex signals use conjugate inner products and coefficient updates. Parameter-range validation and finite-recursion checks are explicit. Persistent excitation, stochastic independence, and convergence for a chosen step size or forgetting factor remain unchecked assumptions in the result certificate.

## Multilevel Haar denoising

`haar_decompose` repeatedly applies the normalized finite Haar filter bank. A decomposition certificate verifies reconstruction and normalized energy conservation.
Haar pair arithmetic and energy diagnostics promote fixed-width integer data before summation. Reconstruction and energy use separate relative scales, so the numerical certificate has the same meaning for an otherwise identical scaled signal. `FiniteFilterBank.report` and `PeriodicFilterBank.frame_bounds` include unobserved input directions: an analysis operator with fewer rows than input coordinates has lower frame bound zero.

`wavelet_denoise` supports:

- hard thresholding
- soft thresholding
- nonnegative-garrote thresholding
- fixed thresholds
- universal thresholds
- finite SURE threshold selection
- common or level-dependent noise estimates

```python
import numpy as np
from komorebi_math.signals import denoising_risk_certificate, wavelet_denoise

rng = np.random.default_rng(1206)
clean = np.repeat(np.array([0.0, 2.0, -1.0, 1.5]), 128)
noisy = clean + rng.normal(scale=0.3, size=clean.size)

result = wavelet_denoise(
    noisy,
    levels=4,
    rule="universal",
    mode="soft",
)
result.certificate.require()
denoising_risk_certificate(clean, noisy, result.signal).require()
```

The shrinkage certificate verifies finite reconstruction and coefficient-magnitude monotonicity. The observed-risk certificate requires a supplied clean reference. Universal and SURE rules do not establish application-specific optimality without their noise-model assumptions.

The scalar and vector jump-recovery APIs reject nonfinite samples and nonfinite grid declarations. Vector jump contrasts promote integer data before subtraction, preventing unsigned downward jumps from wrapping to small positive contrasts. Their guarantees remain conditional on the declared smoothness, noise, jump-size, and separation hypotheses.

## Streaming STFT

`StreamingSTFT` maintains only the overlap and unframed samples required by the next finite frame. `process` emits every complete new frame. `flush(pad_end=True)` emits at most one terminal padded frame, exactly matching `frame_signal`.

```python
import numpy as np
from komorebi_math.signals import stream_stft

rng = np.random.default_rng(1207)
signal = rng.normal(size=1000)

result = stream_stft(
    signal,
    chunk_sizes=(123, 456, 421),
    frame_length=128,
    hop_length=32,
    fft_length=256,
)
result.certificate.require()
```

The certificate compares coefficients and time coordinates against a batch STFT with identical parameters. Chunk boundaries do not alter the represented finite transform.

## Verification

Run the installed digital signal-processing checks:

```powershell
komorebi verify dsp
```

The source tests cover numerical conventions, chunk invariance, finite
reconstruction, and comparison with convention-matched reference calculations.

## Explicit boundaries

The implemented scope includes the biquad families described above. General
high-order IIR design, arbitrary multirate filter-bank compilation, continuous
wavelet transforms, audio-file codecs, real-time acquisition, adaptive-filter
convergence theorems, and a hardware-validated GPU DSP path are outside this API.
