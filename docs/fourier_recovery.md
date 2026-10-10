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
result.fit_certificate.require()
assert result.resolution_status == "candidate_only"
assert not result.certificate.valid
assert result.order == 3
assert np.allclose(result.packet.locations, locations)
assert np.allclose(result.packet.amplitudes, amplitudes)
reconstructed_edge = result.packet.evaluate(np.linspace(0, period, 1000))
```

Automatic order selection returns a candidate. The fit can be inspected and
plotted, but its inferred jump count is not authenticated. `certificate.require()`
and `require_recovery()` reject that candidate even when its finite fit passes.

If the exact number of active jumps is independently known, supply it explicitly:

```python
declared = recover_fourier_jumps(
    measurements, background_degree=background_degree,
    maximum_jumps=maximum_jumps, known_order=3, period=period,
)
packet = declared.require_recovery()
assert declared.resolution_status == "conditional"
```

This approval is conditional on the supplied count being the true count, the
background bandlimit, and suitable noise and conditioning. Setting `known_order`
to a previously inferred `result.order` does not independently establish those
assumptions and must not be treated as a validation step.

The API requires `2 * maximum_jumps` consecutive positive Fourier modes starting at `background_degree + 1`. A missing mode produces a validation error. A decimated sample subset cannot substitute for this block. Polyphase offset lifting is a distinct algorithm and is not silently inferred here.

`hankel_moment_matrix` constructs `H[i,j] = m[i+j]`. The order gate counts singular values above `rank_tolerance * largest_singular_value`. A supplied `known_order` must equal the observed numerical rank at that threshold. Both a larger and a smaller declaration raise `AssumptionError`; a loose fit tolerance cannot bypass this rank agreement. The selected leading Hankel solve gives a monic annihilating polynomial. Polynomial roots give candidate nodes, which are projected to the unit circle. An all-moment least-squares fit gives candidate amplitudes. Their imaginary parts and total sum are checked before canonical real, mean-zero synthesis.

The zero-order branch returns an empty `PeriodicJumpPacket`. Its evaluation is identically zero. An inferred zero order is still a candidate. A declared zero order with nonzero observed rank raises `AssumptionError`. Negative partners supplied for the requested positive-mode block must satisfy the real Fourier conjugacy condition. Measurements outside that block and its negative partners are not checked or used.

`PeriodicJumpPacket` reduces locations modulo the period, sorts them, copies its arrays, and keeps them read-only. Active locations must be distinct. Active amplitudes must be nonzero and have zero total within the supplied tolerance. A correction within that tolerance is removed before integration. Evaluation uses the right-continuous representative and subtracts its piecewise-constant mean using a closed formula evaluated in floating-point arithmetic. Its zero Fourier coefficient is zero.

`fit_certificate` verifies finite moment-fit, recurrence, node-circle, real-amplitude, and zero-sum consistency. It records the Hankel and Vandermonde condition numbers. `certificate` is the separate conditional recovery approval: it is valid only when this fit passes and an exact order was supplied and matched the numerical rank. `resolution_status` is `candidate_only` for a passing inferred-order fit, `conditional` for a passing declared-order fit, and `invalid` for a failed fit or approval. `declared_order` records the caller's declaration. `require_recovery()` returns the packet only after conditional approval.

The target's bandlimit, finite jump model, truth of the supplied count, and suitability of noise and conditioning remain unchecked assumptions. A passing residual certifies consistency of the represented finite model. It does not certify a uniform parameter-error bound or a model-free image method.

The distinction matters for nearby jumps. A three-jump signal with two locations
only `5e-8` apart can have a passing two-jump fit, yet differ by about `0.7`
between those nearby edges. The fitted two-jump Hankel and Vandermonde matrices
can be well conditioned because their condition numbers describe the merged
candidate. The default result now withholds recovery approval. Declaring the
true three-jump order fails the rank gate at the default threshold. Merely
lowering that threshold can produce a spurious root rather than recover the
hidden edge. Trustworthy higher-precision data or more informative measurements
are required to address that loss of resolution.

Migration: code that used `result.certificate.require()` after automatic order
selection now receives `ValidationError`. Use `result.fit_certificate.require()`
for explicit candidate inspection, or supply independently known target order
and use `result.require_recovery()` for conditional recovery.

This module does not include the original plotting and benchmarking dependencies, the largest-singular-ratio order heuristic, multi-offset phase lifting, greedy image dictionary selection, or the infinite packet image frontier. It makes no dedicated Lean claim. The source review explicitly separates those obligations and records the corrected mean-zero periodic perturbation bounds.

Provenance: the recovered original is `sources/unknown_edge/universal_unknown_edge_recovery.py` in `Komorebi_Signal_Reconstruction_Final_Edition_2026-10-06`. Its SHA-256 is `966ef2f7a61e7b2a6c86c78b9a1d227dcdce4d43125ef65e135eba661b416c8a`. The recovered `reviews/SIGNAL_SCOPE_REVIEW.md` supplies the model and implementation boundaries. The integration adds explicit finite validation, copied evidence, general-period normalization, zero-order handling, and observed conjugacy checks.

Run the focused regressions with:

```bash
python -m pytest -q tests/test_fourier_recovery.py
```
