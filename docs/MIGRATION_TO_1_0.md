# Migration from `1.0.0b1` to 1.0

## Select the intended API tier

Research code may continue importing from the broad package namespaces. Code intended to survive the 1.x line should migrate to one of the equivalent versioned aliases:

```python
from komorebi_math import stable
from komorebi_math import v1
from komorebi_math.api import v1
```

Use one style consistently within a project.

## Polyphase names

The canonical verb-first functions remain:

```python
from komorebi_math.stable import refine_polyphase, coarsen_polyphase
```

Readable aliases are also retained:

```python
from komorebi_math.stable import polyphase_refine, polyphase_coarsen
```

They are identity aliases, not duplicate implementations.

## Analysis--synthesis terminology

For square invertible systems, `projector()` remains mathematically accurate. For redundant frames, migrate new code to:

```python
system.component(index)
system.components(groups)
```

The `projector()` compatibility alias remains available, but it does not promise idempotence in a redundant system.

## Portable records

Before upgrading, retain the original portable payloads. Block 16 loads the committed `0.15` polyphase refinement and tower fixtures. Future codec version increases must supply adjacent migrations. Do not edit the `__komorebi_type__` or `__version__` fields manually.

## Verification sequence

```powershell
komorebi api check --stable
komorebi release contract
komorebi verify stable
komorebi verify block16
komorebi release qualify --output .\qualification.json --force
```

A project should treat any failing stable contract as an upgrade blocker rather than silently accepting changed output.
