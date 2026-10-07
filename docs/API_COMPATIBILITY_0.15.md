# API Compatibility in 0.15.0a2

Version `0.15.0a2` retains the complete research API while adding a smaller 1.0-candidate surface.

## Namespaces

* `komorebi_math.stable`
* `komorebi_math.v1`
* `komorebi_math.api.v1`

The latter two are aliases of the curated stable namespace.

## Manifest policy

`public_api_stable_0_15.json` records every stable export, its kind, defining module, qualified name, and signature. Compatibility checks are strict for both symbol removal and signature change.

The complete research namespace remains governed by its own manifest. Addition to the broad package does not automatically promote a symbol to the stable namespace.

## Promotion rule

A symbol should enter the stable namespace only when it has:

1. a precise finite semantic contract
2. validation for invalid inputs and boundary cases
3. persistence behavior when the object is serializable
4. source and installed-wheel tests
5. documentation and at least one executable example
6. a signature intended to survive the 1.0 freeze

## Block 15 stable core

The first candidate surface emphasizes certificates, packets, finite polyphase coordinates, refinement plans, towers, rectangular lattice refinement, exact arithmetic refinement, persistence, and compatibility checking.
