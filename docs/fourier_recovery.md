# Fourier recovery of periodic jump packets

`komorebi_math.signals.fourier_recovery` makes the finite unknown-edge reconstruction machinery available as a NumPy-only library API. It extracts the high-mode Fourier, Hankel, annihilator, and periodic integration steps from the Signal Reconstruction Final Edition of 6 October 2026.

The target is a real periodic signal `f = s + j`. Its smooth component `s` must have Fourier degree at most the supplied `background_degree`. Its jump component `j` has finitely many distinct active locations, real amplitudes that sum to zero, and mean zero. These are model assumptions. The library does not infer them from arbitrary data.

For a period `T`, use normalized Fourier coefficients

\[
\widehat f(n)=\frac1T\int_0^T f(x)e^{-2\pi i n x/T}\,dx.
\]

Above the declared background degree, the derivative moments satisfy

\[
m_n=2\pi i n\widehat f(n)
=\sum_{\ell=1}^L a_\ell z_\ell^n,
\qquad z_\ell=e^{-2\pi i\tau_\ell/T}.
\]

The factor `2π i n` holds for every period under this normalization. It does not become `T i n` unless the Fourier convention also changes.

```python
import numpy as np
from komorebi_math.signals.fourier_recovery import recover_fourier_jumps

period = 2 * np.pi
locations = np.array([0.7, 2.2, 4.8])
amplitudes = np.array([2.0, -0.75, -1.25])
background_degree = 3
maximum_jumps = 5

# These modes lie above the background's bandlimit.
measurements = {
    n: np.sum(amplitudes * np.exp(-1j * n * locations)) / (2j * np.pi * n)
    for n in range(background_degree + 1,
                   background_degree + 1 + 2 * maximum_jumps)
}
result = recover_fourier_jumps(
    measurements,
    background_degree=background_degree,
    maximum_jumps=maximum_jumps,
    period=period,
)
result.certificate.require()
assert result.order == 3
assert np.allclose(result.packet.locations, locations)
assert np.allclose(result.packet.amplitudes, amplitudes)
reconstructed_edge = result.packet.evaluate(np.linspace(0, period, 1000))
```

The API requires `2 * maximum_jumps` consecutive positive Fourier modes starting at `background_degree + 1`. A missing mode produces a validation error. A decimated sample subset cannot substitute for this block. Polyphase offset lifting is a distinct algorithm and is not silently inferred here.

`hankel_moment_matrix` constructs `H[i,j] = m[i+j]`. The order gate counts singular values above `rank_tolerance * largest_singular_value`. A supplied `known_order` must not exceed the observed numerical rank at that threshold. The selected leading Hankel solve gives a monic annihilating polynomial. Polynomial roots give candidate nodes, which are projected to the unit circle. An all-moment least-squares fit gives candidate amplitudes. Their imaginary parts and total sum are checked before canonical real, mean-zero synthesis.

The zero-order branch returns an empty `PeriodicJumpPacket`. Its evaluation is identically zero. An explicitly declared zero order with a substantial nonzero moment block fails the numerical certificate. Negative partners supplied for the requested positive-mode block must satisfy the real Fourier conjugacy condition. Measurements outside that block and its negative partners are not checked or used.

`PeriodicJumpPacket` reduces locations modulo the period, sorts them, copies its arrays, and keeps them read-only. Active locations must be distinct. Active amplitudes must be nonzero and have zero total within the supplied tolerance. A correction within that tolerance is removed before integration. Evaluation uses the right-continuous representative and subtracts its piecewise-constant mean using a closed formula evaluated in floating-point arithmetic. Its zero Fourier coefficient is zero.

The certificate verifies finite moment-fit, recurrence, node-circle, real-amplitude, and zero-sum consistency. It records the Hankel and Vandermonde condition numbers. The target's bandlimit, finite jump model, correctness of the rank threshold, and suitability of the noise and conditioning remain unchecked assumptions. A passing residual certifies consistency of the represented finite model. It does not certify the true number or locations of edges in an arbitrary noisy signal, a uniform stability theorem, or a model-free image method.

This module does not include the original plotting and benchmarking dependencies, the largest-singular-ratio order heuristic, multi-offset phase lifting, greedy image dictionary selection, or the infinite packet image frontier. It makes no dedicated Lean claim. The source review explicitly separates those obligations and records the corrected mean-zero periodic perturbation bounds.

Provenance: the recovered original is `sources/unknown_edge/universal_unknown_edge_recovery.py` in `Komorebi_Signal_Reconstruction_Final_Edition_2026-10-06`. Its SHA-256 is `966ef2f7a61e7b2a6c86c78b9a1d227dcdce4d43125ef65e135eba661b416c8a`. The recovered `reviews/SIGNAL_SCOPE_REVIEW.md` supplies the model and implementation boundaries. The integration adds explicit finite validation, copied evidence, general-period normalization, zero-order handling, and observed conjugacy checks.

Run the focused regressions with:

```bash
python -m pytest -q tests/test_fourier_recovery.py
```
