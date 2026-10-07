# Block 12 release

Version `0.12.0a1` adds the production finite signal-processing layer while retaining every earlier execution block.

## New production modules

- `komorebi_math.signals.windows`
- `komorebi_math.signals.time_frequency`
- `komorebi_math.signals.spectral`
- `komorebi_math.signals.filtering`
- `komorebi_math.signals.resampling`
- `komorebi_math.signals.adaptive`
- `komorebi_math.signals.denoise`
- `komorebi_math.signals.streaming`
- `komorebi_math.signals.recursive`
- `komorebi_math.signals.fft_convolution`

## Main capabilities

Block 12 implements deterministic windows and canonical duals, finite STFT analysis and normalized inverse synthesis, spectrograms, periodogram and Welch estimates, cross spectra, coherence and H1 transfer estimates, odd-length linear-phase FIR design, direct and streaming FIR execution, rational polyphase resampling, LMS/NLMS/RLS adaptation, multilevel Haar shrinkage, chunk-invariant streaming STFT analysis, finite FIR/IIR recursion with biquad design, and FFT overlap-add convolution.

Every new numerical theorem boundary is represented by a certificate or an explicit limitation. The code does not infer stationarity, causality, adaptive convergence, denoising optimality, or real-time performance from finite execution. The source checkpoint passes 648 tests, the installed DSP verifier passes 11 of 11 checks, the cumulative verifier passes 132 of 132 checks, and the deterministic randomized DSP audit passes 815,779 of 815,779 scalar checks. A separate optional SciPy audit passes 520,920 of 520,920 convention-matched reference comparisons.

## Compatibility

The broader package remains alpha. The committed research API manifest now includes `komorebi_math.signals` and contains 699 symbols. The curated beta namespace remains at 83 symbols with strict reflected signatures.

## Commands

```powershell
python -m pip install .\komorebi_math-0.12.0a1-py3-none-any.whl
komorebi verify dsp
komorebi verify block12
komorebi api check --strict-signatures
komorebi api check --beta
python .\examples\production_dsp.py
.\examples\Test-KomorebiBlock12.ps1
```

## Release boundaries

The base wheel remains NumPy-only. No CuPy or CUDA signal kernel is claimed. The implementation provides finite offline and chunked execution, not a hard real-time scheduler. It includes five classical biquad families but not a general high-order IIR or state-space design framework. Hosted Linux and Windows CI are defined but are reported as executed only when their external jobs actually run. Local static tools are reported separately according to availability.
