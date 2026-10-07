# Public API compatibility contract for 0.10

## Stability level

Version `0.10.0a1` declares `pre-beta-additive` structural stability for the committed public manifest. This is stronger than no policy and weaker than a stable 1.0 semantic contract.

The manifest records every name in the explicit `__all__` lists of the top-level, nonlinear, and export namespaces. Each entry contains:

* exported namespace and name
* broad kind: class, function, module, or data
* implementation module
* qualified name
* reflected callable signature when available

## Compatibility rules

| Change | Default check | Strict-signature check |
|---|---:|---:|
| Add an exported name | pass | pass |
| Remove an exported name | fail | fail |
| Change broad symbol kind | fail | fail |
| Change reflected signature | report | fail |
| Change implementation module | record in new manifest | record in new manifest |
| Change mathematical behavior | not certified | not certified |

A default check supports additive alpha development while detecting the most disruptive changes. Release packaging also runs a strict check against the just-generated release manifest.

## Commands

```powershell
komorebi api check
komorebi api check --strict-signatures
komorebi api snapshot --output .\candidate.json
```

The snapshot command refuses to overwrite an existing file unless `--force` is supplied.

## Release procedure

1. Complete the intended public imports and aliases.
2. Run the full tests and installed verification.
3. Generate a candidate manifest.
4. Review every added, removed, kind-changed, and signature-changed symbol.
5. Add deprecation aliases before removing a public name where practical.
6. Commit the reviewed manifest with the release.
7. Run the default API check on all supported Python versions.
8. Run the strict check on the release build environment.

## Non-goals

Reflection cannot establish mathematical equivalence, numerical stability, exception compatibility, serialization compatibility, or performance. Those properties require focused tests, schema policies, and theorem-specific certificates. The API certificate states these unchecked assumptions explicitly.
