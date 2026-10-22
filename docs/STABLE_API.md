# Stable API contract

`komorebi_math.stable`, `komorebi_math.v1` and `komorebi_math.api.v1` expose
131 object-identical symbols for the 1.x line. They support a complete finite
workflow: construct operators and seeds, analyze coordinates, decompose data
into packets, refine representations, certify results, save portable records
and render mathematical documents.

## Compatibility promise

A stable symbol retains its exported name, broad kind, documented signature
and documented finite mathematical behavior throughout the 1.x line.
Persistence promises also retain the declared codec identity and provide
adjacent migrations when a codec version changes.

Additive stable names may appear in a later minor release. Removal or an
incompatible change requires the [deprecation process](DEPRECATION_POLICY.md)
and normally a major release. A correctness fix still has to state the
corrected behavior and its effect on existing applications.

```python
import komorebi_math.stable as stable
import komorebi_math.v1 as v1
import komorebi_math.api.v1 as api_v1

assert stable.__all__ == v1.__all__ == api_v1.__all__
```

## Three independent checks

The structural manifest records exported names, kinds, implementation
identities and callable signatures. Behavioral fixtures check committed
mathematical examples. The persistence contract checks record tags, represented
Python types, codec versions and migrations. Passing one does not replace the
other two.

```bash
python -m komorebi_math api check --stable
python -m komorebi_math verify stable
python -m komorebi_math release check
```

Programmatically:

```python
from komorebi_math.stable import check_stable_api, check_stable_contract

check_stable_api().certificate.require()
check_stable_contract().certificate.require()
```

## Broader domain APIs

An object being importable from the top-level package or a domain module does
not automatically place it under the stable contract. Explicit research APIs
remain available with their documented scope. The separate
`komorebi_math.beta` compatibility namespace is retained for existing
applications, with its own signature checks.

The stable contract does not promise constant performance, availability of
optional native backends, compatibility of arbitrary extension codecs or global
analytic conclusions from finite calculations. See the
[platform policy](SUPPORTED_PLATFORMS.md) and
[persistence guide](PERSISTENCE_SCHEMAS.md).
