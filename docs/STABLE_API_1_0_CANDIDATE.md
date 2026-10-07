# Stable API 1.0 Candidate

## API tiers

Komorebi Math separates availability from compatibility commitment.

| Tier | Namespace | Compatibility policy |
|---|---|---|
| Stable candidate | `komorebi_math.stable`, `komorebi_math.v1`, `komorebi_math.api.v1` | strict exported signatures plus committed semantics and persistence |
| Beta | `komorebi_math.beta` | additive surface with strict signatures |
| Research | top-level and domain namespaces | broad executable research surface; no complete 1.x freeze promise |

A symbol is not stable merely because it can be imported from the top-level package.

## Stable workflow

The candidate surface is selected to support one complete finite workflow:

```python
from komorebi_math.stable import (
    AnalysisSynthesisSystem,
    OperatorGrammar,
    OperatorRealization,
    PacketSystem,
    PolyphaseRefinement,
    check_stable_contract,
    dumps_portable,
    signal_polyphase_analyze,
)
```

The intended progression is

\[
\text{grammar and realization}
\to
\text{coordinate analysis}
\to
\text{packet components}
\to
\text{polyphase refinement}
\to
\text{certificate}
\to
\text{portable record}.
\]

## Structural rule

The manifest `public_api_stable_0_16.json` records:

- exported namespace and name
- data, class, function, or module kind
- implementation module
- qualified name
- reflected callable signature

The stable checker rejects removed exports, broad kind changes, and signature changes. The Block 16 deterministic-snapshot gate also rejects uncommitted additions.

## Semantic rule

The stable manifest does not certify mathematical meaning. `stable_semantics_0_16.json` supplies committed examples whose canonical outputs are checked on every release. Any deliberate semantic change requires an explicit fixture migration and release review.

## Persistence rule

Portable records are part of the compatibility contract when their codecs appear in `portable_codec_contract_0_16.json`. A codec version increase requires adjacent migrations. A Python class rename or move cannot be hidden behind the same tag without an explicit compatibility decision.

## Deprecation rule

After `1.0.0b1`, a stable symbol should not be removed within the 1.x line without a documented deprecation interval. A deprecation record should provide:

- the version in which deprecation began
- the replacement, when one exists
- the planned removal release
- the reason for the change

Actual removal should normally occur only in the next major release.

## What is not promised

The 1.0 candidate does not promise:

- stability of all 1,179 research exports
- constant performance across hardware
- universal analytic convergence from finite calculations
- compatibility of arbitrary extension-owned codecs
- availability of optional JAX, PyTorch, or CuPy backends
- native Windows JAX GPU execution

These boundaries are intentional.

## Verification

```powershell
komorebi api check --stable
komorebi verify stable
komorebi release qualify
```
