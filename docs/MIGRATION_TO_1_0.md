# Upgrading applications

## Select an API tier

Applications that need the 1.x compatibility contract should import the stable
surface through one of its equivalent aliases:

```python
from komorebi_math import stable
from komorebi_math import v1
from komorebi_math.api import v1
```

Use one style consistently. Broad domain imports remain available under their
documented scope. See the [stable API contract](STABLE_API.md).

## Polyphase names

The canonical functions are `refine_polyphase` and `coarsen_polyphase`.
`polyphase_refine` and `polyphase_coarsen` are identity aliases with the same
implementation.

```python
from komorebi_math.stable import refine_polyphase, coarsen_polyphase
```

## Analysis and synthesis

For a square invertible system, `projector()` describes an algebraic projector.
A redundant frame instead has additive reconstruction components, which need
not be idempotent. Prefer `component()` and `components()` in new code unless
idempotence is separately established.

## Portable records

Retain original portable payloads when upgrading. The committed predecessor
contract covers the supported `0.15.0a2` and `0.16.0a2` records and checks their
producer versions, file/payload digests, reconstructed types and available
certificate execution. Codec changes require complete adjacent migrations.
Do not edit `__komorebi_type__` or `__version__` manually.

These checks cover declared portable schemas. Pickles, undocumented internal
objects and arbitrary third-party codecs are outside the upgrade contract.

## Verify an upgrade

```bash
python -m komorebi_math api check --stable
python -m komorebi_math release contract
python -m komorebi_math release upgrade
python -m komorebi_math verify all
python -m komorebi_math release check
```

A failing stable or persistence contract requires investigation before adopting
the new package. Compare the complete exception, certificate, interpreter and
dependency versions with the previous working environment.
