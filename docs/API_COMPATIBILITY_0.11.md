# API compatibility contract for 0.11

## Two stability surfaces

Version `0.11.0a1` deliberately separates the research API from a smaller compatibility surface.

| Surface | Namespaces | Policy |
|---|---|---|
| Research API | `komorebi_math`, `komorebi_math.nonlinear`, `komorebi_math.export` | `pre-beta-additive` |
| Curated beta API | `komorebi_math.beta` | `beta-additive-strict-signatures` |

The complete package remains an alpha release. Only the names exported by `komorebi_math.beta` receive the stricter Block 11 signature contract.

## Committed manifests

The release includes two canonical manifests:

* `public_api_0_11.json`, containing 528 structural symbol records
* `public_api_beta_0_11.json`, containing 83 curated symbol records

Each record contains the exported namespace, name, broad kind, implementation module, qualified name, and reflected signature when available.

The release manifests have these SHA-256 digests:

```text
research: f06b211b78e2c30dee297d912d59b4137b3da15faebb52d72ceb91e435a90bcb
beta:     842389ae182e3847670a34d58ea6b85a5f3eda7279872cecb6f61b4c56cecdb4
```

## Compatibility rules

| Change | Research surface | Beta surface |
|---|---:|---:|
| Add an exported name | pass | pass |
| Remove an exported name | fail | fail |
| Change broad symbol kind | fail | fail |
| Change reflected signature | reported, strict mode fails | fail |
| Change implementation module | recorded | recorded |
| Change documented behavior | requires semantic review | requires semantic review and fixture review |

The beta surface is additive rather than frozen. New names may be added. Existing signatures may not change without an explicit compatibility decision and manifest update.

## Commands

```powershell
komorebi api check --strict-signatures
komorebi api check --beta
komorebi api snapshot --output .\research-api.json
komorebi api snapshot --beta --output .\beta-api.json
```

The snapshot command refuses overwrite unless `--force` is supplied.

## Behavioral fixtures

Structural reflection cannot detect a function that keeps its signature but changes its mathematical result. Block 11 therefore adds a closed behavioral fixture with five deterministic cases:

1. exact packet-snapshot contraction
2. third-order sparse symmetric contraction
3. projection-cached packet execution
4. packet-snapshot codec migration from version 1 to version 2
5. streamed exact packet snapshot

Run it with:

```powershell
komorebi golden
```

The fixture digest is:

```text
614e852098c971dc7700892324fa5e2b8ebc823c3dc3cbec7d7b3dea1ec5003b
```

The fixture is a regression boundary for the committed cases. It is not an exhaustive semantic proof.

## Release procedure

1. Complete intended exports.
2. Run unit tests and installed verification.
3. Run the golden fixture.
4. Generate candidate research and beta manifests.
5. Review additions, removals, kind changes, and signature changes.
6. Review every expected-value change in the behavioral fixture.
7. Add a deprecation path before removing a public research name where practical.
8. Commit the reviewed manifests and fixtures.
9. Run strict checks against the installed wheel and extracted source distribution.

## What the contract does not prove

Neither manifest proves numerical stability, exception compatibility, asymptotic complexity, thread safety, persistence compatibility, or mathematical equivalence. Persistence versions, golden fixtures, focused tests, and theorem-specific certificates cover separate parts of that problem.
