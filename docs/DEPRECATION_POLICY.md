# Deprecation policy

A stable removal or incompatible change requires a documented transition.
Ordinary removals follow a deprecation interval and normally take effect in
the next major release. A correctness or security defect that cannot safely
remain requires an explicit compatibility decision and explanation.

A deprecated callable remains usable during the announced period, preserves
its behavior or delegates transparently, and emits
`KomorebiDeprecationWarning`. Its record identifies the first warning version,
a replacement when available, the planned removal release and the reason.

```python
from komorebi_math import deprecated


def replacement(value: int) -> int:
    return value + 1


@deprecated(since="1.1", replacement="replacement", remove_in="2.0")
def legacy(value: int) -> int:
    return replacement(value)
```

The decorator exposes `__komorebi_deprecation__` for tooling. Spelling aliases
may remain without warnings when they have unambiguous identical behavior.
Preferred device names use `CuPy`; existing `Cupy` implementation spellings
remain compatibility aliases.

Persistence versions have a separate obligation: supported older records must
remain decodable or have an explicit migration. An API alias does not permit
silent reinterpretation of stored data.

Before a removal, review the affected symbols, replacement, warning/removal
versions, persistence and CLI surfaces, migration tests and API manifest
differences. No existing stable symbol is scheduled for removal in 1.x.
