# Block 18: Additive Research Frame Algebra

Block 18 advances the package to `1.0.0b2` and adds frame algebra only to the research tier. It does not reopen the public 1.0 contract frozen by Block 17.

## Mathematical scope

Given an analysis map `A`, synthesis map `S`, and signal-space operator `T`, Block 18 studies the induced coefficient-space operator `A T S` and the frame projection `A S`. The research layer supports closure checks, coefficient-space composition, and reconstruction-aware identities for redundant systems. Exact identities are conditioned on the corresponding reconstruction and compatibility hypotheses; numerical checks retain explicit tolerances.

## Frozen compatibility gates

The following Block 17 invariants remain exact:

- 131 stable exports, with identical names, kinds, signatures, and alias identities
- 83 beta exports
- 51 registered portable codecs and their migration contract
- 10 stable semantic fixtures

No Block 18 frame-algebra symbol is stable or beta. Stable behavior and portable decoding are unchanged. Research compatibility is one-way additive: every symbol in the saved `1.0.0b1` research baseline must remain compatible, while new research symbols may be appended.

## Baseline and release status

The five `*_1_0_0b1.json` snapshots preserve the pre-Block-18 API, beta, stable, codec, and semantic baselines. The `1.0.0b1` wheel is retained under `release/predecessors` for installed-artifact compatibility checks.

`1.0.0b2` is a research beta, not a release candidate. Promotion of frame algebra requires an explicit public-tier review and must not be inferred from its presence in the research namespace.
