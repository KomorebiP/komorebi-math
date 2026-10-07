from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.package_root.resolve()))

    import numpy as np
    import komorebi_math
    from komorebi_math import Certificate, Exactness, PacketSystem
    from komorebi_math.exceptions import ValidationError
    from komorebi_math.export.persistence import dumps_portable, loads_portable
    from komorebi_math.nonlinear.derivatives import PolynomialDerivativeProvider
    from komorebi_math.nonlinear.jets import PacketJetFamily
    from komorebi_math.nonlinear.snapshots import snapshot_packet_jet
    from komorebi_math.nonlinear.streaming import (
        load_packet_jet_snapshot_stream,
        open_packet_jet_snapshot_stream,
        save_packet_jet_snapshot_stream,
    )

    assertions = 0
    checks: list[str] = []

    def check(condition: bool, label: str) -> None:
        nonlocal assertions
        assertions += 1
        if not condition:
            raise AssertionError(label)
        checks.append(label)

    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)

    def rehash(envelope: dict) -> str:
        envelope['payload_sha256'] = hashlib.sha256(canonical(envelope['payload']).encode('utf-8')).hexdigest()
        return json.dumps(envelope)

    def pairs(envelope: dict) -> list[list[object]]:
        return envelope['payload']['data']['data']

    def pair(entries, key):
        return next(entry for entry in entries if entry[0] == key)

    fresh = Certificate(True, Exactness.EXACT, residuals={'r': 0.0}, theorem_id='AUDIT')
    check(fresh.valid is True, 'fresh certificate is live-valid')
    check(fresh.current_valid is True, 'fresh certificate current_valid is true')
    check(fresh.revalidation_required is False, 'fresh certificate needs no revalidation')

    restored = loads_portable(dumps_portable(fresh))
    check(restored.recorded_valid is True, 'persisted record retains recorded validity')
    check(restored.valid is False, 'persisted record is not live-valid')
    check(restored.current_valid is False, 'persisted record current_valid is false')
    check(restored.revalidation_required is True, 'persisted record requires revalidation')
    try:
        restored.require()
    except ValidationError:
        pass
    else:
        raise AssertionError('persisted record require() must fail')
    assertions += 1; checks.append('persisted record require() refuses historical evidence')

    second = loads_portable(dumps_portable(restored))
    check(second.recorded_valid is True, 're-persistence preserves original recorded validity')
    check(second.valid is False and second.revalidation_required, 're-persistence cannot revive live validity')

    legacy = json.loads(dumps_portable(fresh))
    legacy['payload']['__version__'] = 1
    ps = pairs(legacy)
    recorded = pair(ps, 'recorded_valid')[1]
    ps[:] = [entry for entry in ps if entry[0] != 'record_kind']
    pair(ps, 'recorded_valid')[0] = 'valid'
    pair(ps, 'valid')[1] = recorded
    legacy_restored = loads_portable(rehash(legacy))
    check(legacy_restored.recorded_valid is True, 'v1 migration retains recorded validity')
    check(legacy_restored.valid is False and legacy_restored.revalidation_required, 'v1 migration creates historical record')

    malformed = json.loads(dumps_portable(fresh))
    pairs(malformed).append(['unexpected', 1])
    try:
        loads_portable(rehash(malformed))
    except ValidationError:
        pass
    else:
        raise AssertionError('unknown certificate field must be rejected')
    assertions += 1; checks.append('unknown certificate field rejected')

    malformed = json.loads(dumps_portable(fresh))
    residuals = pair(pairs(malformed), 'residuals')[1]
    residuals['data'][0][1] = 'nan'
    try:
        loads_portable(rehash(malformed))
    except ValidationError:
        pass
    else:
        raise AssertionError('nonfinite residual coercion must be rejected')
    assertions += 1; checks.append('string NaN residual rejected')

    for bad in ('inf', 'nan', -1.0, True):
        malformed = json.loads(dumps_portable(Certificate(True, Exactness.NUMERICAL, tolerance=1e-9)))
        pair(pairs(malformed), 'tolerance')[1] = bad
        try:
            loads_portable(rehash(malformed))
        except ValidationError:
            pass
        else:
            raise AssertionError(f'invalid tolerance accepted: {bad!r}')
        assertions += 1; checks.append(f'invalid tolerance rejected: {bad!r}')

    digest_bad = json.loads(dumps_portable({'x': 1}))
    digest_bad['payload_sha256'] = '0' * 64
    try:
        loads_portable(json.dumps(digest_bad))
    except ValidationError:
        pass
    else:
        raise AssertionError('digest mismatch must be rejected')
    assertions += 1; checks.append('digest mismatch rejected')

    packets = PacketSystem([np.eye(1)], names=('p0',))
    provider = PolynomialDerivativeProvider({1: np.array([[2.0]])}, source_dimension=1, target_dimension=1)
    jet = PacketJetFamily(provider, np.zeros(1), packets, packets, 1).jet('p0', ('p0',))
    snapshot = snapshot_packet_jet(jet)
    loaded_snapshot = loads_portable(dumps_portable(snapshot))
    check(loaded_snapshot.coefficients == snapshot.coefficients, 'snapshot coefficients survive persistence')
    check(loaded_snapshot.capture_certificate.recorded_valid is True, 'snapshot capture record retains historical validity')
    check(loaded_snapshot.capture_certificate.valid is False, 'snapshot capture record is not live-valid')

    with tempfile.TemporaryDirectory() as td:
        destination = Path(td) / 'audit.kmstream'
        save_packet_jet_snapshot_stream(snapshot, destination, chunk_size=1)
        reader = open_packet_jet_snapshot_stream(destination, verify_chunks=True)
        integrity = reader.verify()
        check(integrity.current_valid is True, 'fresh stream integrity certificate is live-valid')
        loaded = load_packet_jet_snapshot_stream(destination)
        check(loaded.capture_certificate.recorded_valid is True, 'stream-loaded capture retains recorded validity')
        check(loaded.capture_certificate.valid is False, 'stream-loaded capture is historical not live')

    result = {
        'schema': 'komorebi-math.repair-r4-block7-persistence-audit',
        'schema_version': 1,
        'package_version': komorebi_math.__version__,
        'passed': True,
        'assertions': assertions,
        'checks': checks,
        'nonclaims': [
            'SHA-256 payload integrity is not authentication or authorship.',
            'The audit does not rerun arbitrary certificate producers.',
            'Custom codec decoders and migration functions remain trusted application code.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
