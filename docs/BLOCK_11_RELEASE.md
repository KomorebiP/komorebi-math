# Block 11 release

Version `0.11.0a1` completes the resumable persistence, semantic regression, finite profiling, and curated beta-contract block.

## Added

* chunked packet-jet streams with atomic progress commits
* pause and resume for live standard-basis capture
* chunked persistence for existing packet-jet snapshots
* bounded chunk iteration, coefficient selection, partial evaluation, and guarded materialization
* manifest, chunk, source, metadata, and stream-content SHA-256 validation
* explicit adjacent portable codec migrations
* packet-jet snapshot codec version 2 and a version 1 to version 2 migration
* five committed behavioral golden cases under a closed evaluator registry
* exact packet-jet structural complexity counts
* environment-scoped runtime and traced-memory observations
* an 83-symbol `komorebi_math.beta` namespace with strict signature checks
* research and beta API manifests for version 0.11
* Block 11 CLI commands, installed verification, examples, PowerShell smoke coverage, and randomized audit
* an actual CUDA validation program and a separate self-hosted GPU workflow definition

## Compatibility

The broader 528-symbol research surface remains additive pre-beta. The complete package version remains alpha. The beta contract applies only to `komorebi_math.beta`.

Portable packet-jet snapshots now encode as codec version 2. Version 1 payloads remain readable through a registered deterministic migration. Future codec versions are rejected until the corresponding implementation and migration path exist.

## Validation evidence

The final source suite passed **545/545** tests. The installed full environment passed **121/121** verification checks, the required-dependency-only environment passed **119/119**, all **5/5** golden cases passed, and the independent audit passed **11,738/11,738** checks. The committed manifests cover **528** research symbols and **83** beta symbols with zero compatibility differences.

Both successive wheel builds and normalized source-distribution builds were byte-identical. The extracted source distribution passed all **545** tests, and its rebuilt wheel was byte-identical to the release wheel. The wheel contains **211** files with no forbidden payload entries. The source distribution contains **427** members.

## Validation boundary

The local release environment executed Linux source, package, exact arithmetic, JAX, and CPU PyTorch validation. It did not execute Windows, hosted CI, PowerShell, Ruff, mypy, MkDocs, CuPy, or CUDA device paths. Those states are recorded as not executed rather than passed.

Digest validation provides integrity, not authenticity. Runtime observations are local measurements, not speed claims.

## Continuation

The next package block should prioritize actual CI evidence on every declared platform, device-backed CUDA validation, signed artifact provenance, provider identity for cross-process direct resumes, and performance work driven by recorded profiles rather than assumed bottlenecks.
