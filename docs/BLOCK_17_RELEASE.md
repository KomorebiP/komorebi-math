# Block 17: Public Beta and Upgrade Closure

Block 17 changes the package version from `0.16.0a2` to `1.0.0b1`. The mathematical implementation remains the Block 16 freeze candidate. The new block closes release engineering around that implementation.

## Added guarantees

- Exact stable API continuity from `0.16.0a2`
- Supported migration from the curated `0.15.0a2` stable surface
- Digest-checked portable records produced by both predecessor releases
- One documentation record for every stable symbol
- Digest-checked platform evidence with an explicit inheritance rule
- Separate beta and release-candidate readiness modes
- Executable predecessor-wheel upgrade tests
- Dedicated source, wheel, GPU, upgrade, and distribution runners

## Stable API result

The stable namespace remains 131 symbols. Comparison with `public_api_stable_0_16.json` must report:

```text
added             0
removed           0
kind changed      0
signature changed 0
```

No new mathematical feature is promoted to stable in this block.

## Evidence rule

Beta publication may use predecessor platform evidence only when all of the following are exact:

1. stable API continuity
2. stable semantic continuity
3. portable codec continuity
4. committed predecessor record reconstruction

Release-candidate qualification requires the current beta on every required platform family and cannot inherit predecessor execution.

## Release status

`1.0.0b1` is suitable for public beta testing. It is not represented as the final stable release. The next release is `1.0.0rc1` after exact Windows qualification of this beta.
