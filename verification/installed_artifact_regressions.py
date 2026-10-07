"""Three regressions, deliberately executed without importing a source checkout."""
from __future__ import annotations
import argparse
from fractions import Fraction
import importlib.metadata
import json
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-installed', action='store_true')
    args = parser.parse_args()
    import numpy as np
    import komorebi_math
    from komorebi_math.coordinates.families import fibonacci_recurrence_model
    from komorebi_math.export.persistence import from_portable_data, to_portable_data
    path = Path(komorebi_math.__file__).resolve()
    if args.require_installed and not path.is_relative_to(Path(sys.prefix).resolve()):
        raise SystemExit(f'wrong import location: {path}; expected within {sys.prefix}')
    results = []

    def recurrence() -> None:
        model = fibonacci_recurrence_model()
        values = (3, -1, 4, 1, 5, -9, 2, 6)
        restored = model.synthesize(model.analyze(values).coordinates())
        if restored != tuple(Fraction(v) for v in values):
            raise ValueError('recurrence reconstruction is not exact')

    def mapping() -> None:
        value = {Fraction(1, 3): ('alpha', 7)}
        if from_portable_data(to_portable_data(value)) != value:
            raise ValueError('direct portable mapping round-trip failed')

    def array() -> None:
        value = np.arange(12, dtype=np.int16).reshape(3, 4)
        restored = from_portable_data(to_portable_data(value))
        if not (np.array_equal(restored, value) and restored.dtype == value.dtype and restored.shape == value.shape):
            raise ValueError('direct portable ndarray round-trip changed values, dtype or shape')

    for name, probe in [('recurrence_synthesize', recurrence), ('direct_mapping_roundtrip', mapping),
                        ('direct_ndarray_roundtrip', array)]:
        try:
            probe()
            results.append({'name': name, 'passed': True})
        except Exception as exc:
            results.append({'name': name, 'passed': False, 'exception': type(exc).__name__, 'message': str(exc)})
    report = {'package_version': komorebi_math.__version__, 'imported_from': str(path),
              'python_prefix': sys.prefix, 'require_installed': args.require_installed,
              'numpy_version': importlib.metadata.version('numpy'), 'tests': results,
              'passed': all(r['passed'] for r in results)}
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
