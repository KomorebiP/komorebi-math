# Block 6 release

Version `0.6.0a1` implements the arithmetic and sequence execution layer.

## Local domains

The package provides modular rings, prime fields, truncated local domains, exact valuations, factorial and binomial valuations, digit formulas, and Carmichael exponents for prime powers.

## Hurwitz and standard window coordinates

`HurwitzSeries` implements divided power convolution, exact finite inversion, and exponential modes. `StandardWindow` computes the compressed inverse attached to a prime and a divisor of the residual unit group order. The positive tail is certified periodic, and the corresponding rational generating function is available directly.

## Euler coordinates

`EulerLocalCoordinates` exposes one interface for every prime. Odd primes use alternating odd power sums and the odd inverse kernel. The dyadic prime uses its separate integral source and kernel. Reconstruction, inverse reconstruction, row valuations, and finite truncation congruences are executable.

## Mahler transfer

`MahlerSeries` supports exact finite analysis, synthesis, differences, affine subsequences, and polyphase refinement. The affine kernel has an exact composition certificate. Prime power refinement exposes its residual sparse structure. Scalar and matrix unit modes provide finite Mahler coordinates with explicit local valuation gains.

## Recurrence packets

`LinearRecurrence` supports exact terms, fast modular terms, companion matrices, residue packets, and modular orbit periods. Residue packets execute the jump matrix rather than repeatedly sampling the parent recurrence.

## Sequence execution

The sequence layer contains lazy exact objects, built in classical families, transforms, recurrence detection, finite automata, polynomial coefficient recurrences, and execution compilation. Declared catalog models are exact. A model inferred from a finite prefix is labeled as a finite detection result and does not become an infinite theorem.

## Verification boundary

The release validates exact finite identities and algorithms. It does not claim a complete arbitrary precision local field implementation, universal recurrence discovery, automatic recognition of every holonomic sequence, or historical novelty for the underlying classical formulas.
