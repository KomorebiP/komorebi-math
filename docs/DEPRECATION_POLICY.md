# Deprecation policy

## Alpha policy

Beginning with `0.10.0a1`, public removals should use an explicit deprecation period unless the name creates a correctness, security, or packaging defect that cannot safely remain.

A deprecated callable should:

* remain callable during the announced period
* preserve its documented behavior or delegate transparently
* emit `KomorebiDeprecationWarning`
* identify the first deprecated version
* name the replacement when one exists
* state the planned removal version when known
* expose a `__komorebi_deprecation__` record for tooling

```python
from komorebi_math import deprecated

@deprecated(since="0.10", replacement="new_function", remove_in="1.0")
def old_function(value: int) -> int:
    return new_function(value)
```

## Alias policy

Spelling and capitalization aliases may be retained without a warning when they correct an internal naming inconsistency and have no behavioral ambiguity. Block 10 retains `Cupy*` implementation names while exporting preferred `CuPy*` aliases. A future removal requires a separately documented deprecation decision.

## Schema policy

Persistence codec versions are not governed solely by Python callable deprecation. A persisted version must remain decodable or receive an explicit migration tool. A Python API alias does not authorize silent reinterpretation of stored data.

## Removal review

Before a removal, the release audit must record:

1. the deprecated symbol
2. the replacement or reason for removal
3. the first warning version
4. the announced removal version
5. affected persistence or CLI surfaces
6. test coverage for the migration
7. public API manifest differences

No Block 10 public symbol is currently scheduled for removal.
