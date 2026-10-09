"""Bounded independent stress controls for evidence, persistence, and quotients.

The table census covers every commutative idempotent labelled operation through
four elements. Partition intersections, rather than the production union-find,
provide the independent least-congruence oracle.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from fractions import Fraction
from itertools import combinations, permutations, product
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Barrier

import numpy as np
import pytest

from komorebi_math.certificates import Certificate, ComputationResult, Exactness
from komorebi_math.core.semilattice import FiniteJoinSemilattice, SemilatticeQuotient
from komorebi_math.exceptions import ValidationError
import komorebi_math.export.persistence as portable
from komorebi_math.persistence_contract import PortableCodecContract, PortableCodecContractEntry


STRESS_METRICS = {}


def _partitions(size):
    def visit(prefix):
        if len(prefix) == size:
            yield tuple(prefix)
        else:
            for label in range(max(prefix, default=-1) + 2):
                yield from visit((*prefix, label))
    yield from visit(())


def _canonical(labels):
    return tuple(min(index for index, label in enumerate(labels) if label == labels[element])
                 for element in range(len(labels)))


def _is_congruence(table, labels):
    size = len(table)
    return all(labels[table[left][context]] == labels[table[right][context]]
               for left, right, context in product(range(size), repeat=3)
               if labels[left] == labels[right])


def _candidate_tables(size):
    pairs = tuple(combinations(range(size), 2))
    for values in product(range(size), repeat=len(pairs)):
        table = [[element if element == other else None for other in range(size)]
                 for element in range(size)]
        for (left, right), value in zip(pairs, values):
            table[left][right] = table[right][left] = value
        yield tuple(tuple(row) for row in table)


@pytest.mark.parametrize("size,expected", [(1, 1), (2, 2), (3, 9), (4, 76)])
def test_exhaustive_tables_partitions_relation_sets_and_witness_mutations(size, expected):
    candidates = accepted = partition_checks = closures = mutations = relabelings = 0
    all_partitions = tuple(_partitions(size))
    pairs = tuple(combinations(range(size), 2))
    for table in _candidate_tables(size):
        candidates += 1
        associative = all(table[table[a][b]][c] == table[a][table[b][c]]
                          for a, b, c in product(range(size), repeat=3))
        if not associative:
            with pytest.raises(ValidationError, match="associative"):
                FiniteJoinSemilattice(table)
            continue
        accepted += 1
        source = FiniteJoinSemilattice(table)
        congruences = []
        for labels in all_partitions:
            partition_checks += 1
            if _is_congruence(table, labels):
                congruences.append(labels)
                supplied = source.quotient(labels)
                assert supplied.partition == _canonical(labels)
                assert "finite least-congruence derivation" not in supplied.certificate().assumptions_checked
                quotient = supplied.quotient
                for left, right in product(range(size), repeat=2):
                    assert supplied.projection[table[left][right]] == quotient.join(
                        supplied.projection[left], supplied.projection[right])
            else:
                with pytest.raises(ValidationError):
                    source.quotient(labels)
        for mask in range(1 << len(pairs)):
            relations = tuple(pair for index, pair in enumerate(pairs) if mask & (1 << index))
            containing = tuple(labels for labels in congruences
                               if all(labels[left] == labels[right] for left, right in relations))
            oracle = tuple(min(right for right in range(size)
                               if all(labels[left] == labels[right] for labels in containing))
                           for left in range(size))
            result = source.generated_congruence(relations)
            closures += 1
            assert result.partition == oracle
            assert result.certificate().current_valid
            # Input ordering, pair orientation, and redundant generators cannot
            # change the independently computed least congruence.
            alternative = tuple((right, left) for left, right in reversed(relations)) + relations
            assert source.generated_congruence(alternative).partition == oracle
            if result.derivations:
                witnesses = list(result.derivations)
                first = witnesses[0]
                witnesses[0] = (first[0], first[1], 0, first[3])  # circular parent
                forged = SemilatticeQuotient(source, result.partition, result.relations, tuple(witnesses))
                assert not forged.certificate().valid
                with pytest.raises(ValidationError):
                    _ = forged.quotient
                mutations += 1
        for permutation in permutations(range(size)):
            inverse = tuple(permutation.index(element) for element in range(size))
            renamed_table = tuple(tuple(permutation[table[inverse[left]][inverse[right]]]
                                         for right in range(size)) for left in range(size))
            renamed = FiniteJoinSemilattice(renamed_table)
            relations = ((0, size - 1),)
            original = source.generated_congruence(relations)
            observed = renamed.generated_congruence(tuple((permutation[a], permutation[b]) for a, b in relations))
            for left, right in product(range(size), repeat=2):
                assert (original.partition[left] == original.partition[right]) == (
                    observed.partition[permutation[left]] == observed.partition[permutation[right]])
            relabelings += 1
    assert accepted == expected
    STRESS_METRICS[f"semilattice_size_{size}"] = {
        "candidates": candidates, "accepted": accepted, "partition_checks": partition_checks,
        "relation_sets": closures, "forged_witness_rejections": mutations, "relabelings": relabelings,
    }


@pytest.mark.parametrize("shape", [(), (1,), (1, 1)])
@pytest.mark.parametrize("field", ["residual", "tolerance"])
def test_mutable_array_scalar_cannot_approve_evidence(shape, field):
    value = np.zeros(shape)
    arguments = {"residuals": {"identity": value}} if field == "residual" else {"tolerance": value}
    certificate = Certificate(True, Exactness.NUMERICAL, **arguments)
    assert not certificate.current_valid
    value[...] = np.nan
    assert not ComputationResult("unused", certificate).valid
    with pytest.raises(ValidationError):
        certificate.require()


@pytest.mark.parametrize("value", [2**54 + 1, 10**1000, -(10**1000), Fraction(1, 3), Fraction(10**1000, 3)])
def test_exact_finite_evidence_roundtrip_never_rounds(value):
    original = Certificate(True, Exactness.EXACT, {"exact": value})
    assert original.current_valid
    first = portable.loads_portable(portable.dumps_portable(original))
    second = portable.loads_portable(portable.dumps_portable(first))
    for restored in (first, second):
        assert restored.residuals["exact"] == value
        assert restored.recorded_valid and not restored.current_valid
        with pytest.raises(ValidationError, match="revalidation"):
            restored.require()


def test_immutable_decimal_and_extended_numpy_scalar_are_finite_without_float64_overflow():
    assert Certificate(True, Exactness.NUMERICAL, {"finite": Decimal("1e4000")}).current_valid
    value = np.longdouble(np.finfo(np.longdouble).max)
    assert np.isfinite(value)
    assert Certificate(True, Exactness.NUMERICAL, {"finite": value}).current_valid
    assert not Certificate(True, Exactness.NUMERICAL, {"invalid": np.longdouble("nan")}).current_valid


@pytest.mark.parametrize("valid", [True, False])
@pytest.mark.parametrize("exactness", tuple(Exactness))
def test_persisted_records_stay_historical_under_repeated_roundtrip(valid, exactness):
    certificate = Certificate(valid, exactness, {"identity": 0.0}, assumptions_unchecked=("caller condition",))
    for _ in range(5):
        certificate = portable.loads_portable(portable.dumps_portable(certificate))
        assert certificate.recorded_valid is valid
        assert not certificate.valid and not certificate.current_valid
        assert certificate.revalidation_required
        with pytest.raises(ValidationError, match="revalidation"):
            certificate.require()


@pytest.mark.parametrize("dtype,value", [
    ("float16", 1e300), ("float32", 1e300), ("complex64", 1e300),
    ("int8", 256), ("uint8", -1), ("int8", 1.1), ("int16", True),
])
def test_declared_dtype_cannot_overflow_to_infinity_or_truncate_integer_data(dtype, value):
    payload = {"__komorebi_type__": "numpy.ndarray", "__version__": 1,
               "data": {"dtype": dtype, "shape": [1], "data": [value]}}
    with pytest.raises(ValidationError):
        portable.from_portable_data(payload)


@pytest.mark.parametrize("value", [np.longdouble("1.5"), np.clongdouble("1+2j")])
def test_unsupported_extended_numpy_precision_is_explicit(value):
    if not isinstance(value.item(), np.generic):
        assert portable.from_portable_data(portable.to_portable_data(value)) == value.item()
    else:
        with pytest.raises(ValidationError, match="extended NumPy scalar precision"):
            portable.to_portable_data(value)


def test_seeded_dtype_shape_and_nested_payload_roundtrips():
    rng = np.random.default_rng(20261008)
    count = 0
    for dtype in ("int8", "int64", "uint16", "float16", "float32", "float64", ">f8", "complex64", "complex128", "bool"):
        for _ in range(16):
            shape = tuple(int(value) for value in rng.integers(0, 5, size=int(rng.integers(0, 4))))
            size = int(np.prod(shape)) if shape else 1
            kind = np.dtype(dtype).kind
            if kind in "iub":
                array = rng.integers(0, 100, size=size).astype(dtype).reshape(shape)
            else:
                values = rng.normal(size=size)
                if kind == "c":
                    values = values + 1j * rng.normal(size=size)
                array = values.astype(dtype).reshape(shape)
            original = {"array": array, Fraction(2, 7): (b"\x00\xff", frozenset((1, 4)), Fraction(1, 3))}
            restored = portable.loads_portable(portable.dumps_portable(original))
            assert restored["array"].dtype == array.dtype
            assert restored["array"].shape == array.shape
            assert np.array_equal(restored["array"], array)
            assert restored[Fraction(2, 7)] == original[Fraction(2, 7)]
            count += 1
    STRESS_METRICS["seeded_portable_arrays"] = {"seed": 20261008, "cases": count, "rank_range": [0, 3], "axis_extent_range": [0, 4]}


@pytest.mark.parametrize("version", [True, False, 1.0, 1.25, "1", None])
def test_codec_contract_versions_are_strict_json_integers(version):
    entry = {"tag": "local.example", "version": version, "implementation_type": "local.Example"}
    with pytest.raises(ValidationError, match="positive integers"):
        PortableCodecContractEntry.from_dict(entry)
    contract = {"package_version": "1", "entries": [],
                "schema": "komorebi-math.portable-codec-contract", "schema_version": version}
    with pytest.raises(ValidationError, match="schema version"):
        PortableCodecContract.from_dict(contract)


def test_concurrent_exclusive_save_has_one_winner_and_cleans_temporary_files(tmp_path, monkeypatch):
    portable.portable_codecs()
    original_dump = portable.dumps_portable
    repetitions = 8
    workers = 8
    for repetition in range(repetitions):
        destination = tmp_path / f"race-{repetition}.json"
        barrier = Barrier(workers)
        def synchronized_dump(value, **kwargs):
            content = original_dump(value, **kwargs)
            barrier.wait(timeout=15)
            return content
        monkeypatch.setattr(portable, "dumps_portable", synchronized_dump)
        def save(value):
            try:
                portable.save_portable(value, destination)
                return True, value
            except ValidationError as error:
                assert "refusing to overwrite" in str(error)
                return False, value
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(save, range(workers)))
        winners = [value for successful, value in outcomes if successful]
        assert len(winners) == 1
        assert portable.load_portable(destination) == winners[0]
        assert not tuple(tmp_path.glob("*.tmp"))
    STRESS_METRICS["concurrent_exclusive_save"] = {"repetitions": repetitions, "writers_each": workers, "successful_writers_each": 1}


def test_failed_serialization_preserves_preexisting_destination(tmp_path):
    destination = tmp_path / "existing.json"
    portable.save_portable({"value": 1}, destination)
    original = destination.read_bytes()
    with pytest.raises(ValidationError):
        portable.save_portable(float("nan"), destination, overwrite=True)
    assert destination.read_bytes() == original


def test_simulated_windows_exclusive_publish_uses_rename_without_requiring_links(tmp_path, monkeypatch):
    calls = []
    def simulated_windows_rename(source, destination):
        # Model the no-replacement collision boundary without requiring hard
        # links from the host filesystem. Native concurrent tests qualify the
        # actual OS publication primitive separately.
        with Path(destination).open("xb") as stream:
            stream.write(Path(source).read_bytes())
        os.unlink(source)
        calls.append((source, destination))
    def forbidden_link(*args, **kwargs):
        raise AssertionError("Windows exclusive publication must not require hard links")
    monkeypatch.setattr(portable.os, "rename", simulated_windows_rename)
    monkeypatch.setattr(portable.os, "link", forbidden_link)
    temporary = tmp_path / "prepared.tmp"
    destination = tmp_path / "published.json"
    temporary.write_bytes(b"complete file")
    portable._publish_exclusive(temporary, destination, windows=True)
    assert calls == [(temporary, destination)]
    assert destination.read_bytes() == b"complete file"
    assert not temporary.exists()
    temporary.write_bytes(b"replacement")
    with pytest.raises(ValidationError, match="refusing to overwrite"):
        portable._publish_exclusive(temporary, destination, windows=True)
    assert destination.read_bytes() == b"complete file"


def test_simulated_posix_exclusive_publish_uses_links_and_preserves_existing_file(tmp_path, monkeypatch):
    calls = []
    def simulated_link(source, destination):
        with Path(destination).open("xb") as stream:
            stream.write(Path(source).read_bytes())
        calls.append((source, destination))
    def forbidden_rename(*args, **kwargs):
        raise AssertionError("POSIX rename would replace an existing destination")
    monkeypatch.setattr(portable.os, "link", simulated_link)
    monkeypatch.setattr(portable.os, "rename", forbidden_rename)
    temporary = tmp_path / "prepared.tmp"
    destination = tmp_path / "published.json"
    temporary.write_bytes(b"complete file")
    portable._publish_exclusive(temporary, destination, windows=False)
    assert calls == [(temporary, destination)]
    assert destination.read_bytes() == b"complete file"
    assert not temporary.exists()
    temporary.write_bytes(b"replacement")
    with pytest.raises(ValidationError, match="refusing to overwrite"):
        portable._publish_exclusive(temporary, destination, windows=False)
    assert destination.read_bytes() == b"complete file"


def test_cross_process_snapshot_load_requires_fresh_comparison(tmp_path):
    code = '''
import json, sys
import numpy as np
from komorebi_math import PacketSystem
from komorebi_math.exceptions import ValidationError
from komorebi_math.export.persistence import save_portable, load_portable
from komorebi_math.nonlinear import PolynomialDerivativeProvider, PacketJetFamily, snapshot_packet_jet
provider = PolynomialDerivativeProvider({2: np.array([[[2.0]]])}, source_dimension=1, target_dimension=1)
packets = PacketSystem((np.eye(1),), names=("all",))
family = PacketJetFamily(provider, np.array([0.2]), packets, packets, 2)
jet = family.jet("all", ("all", "all"))
if sys.argv[1] == "write":
    snapshot = snapshot_packet_jet(jet)
    snapshot.capture_certificate.require()
    save_portable(snapshot, sys.argv[2])
else:
    snapshot = load_portable(sys.argv[2])
    assert snapshot.capture_certificate.recorded_valid
    assert not snapshot.capture_certificate.current_valid
    try: snapshot.capture_certificate.require()
    except ValidationError: pass
    else: raise AssertionError("historical capture approved")
    comparison = snapshot.comparison_certificate(jet, (np.array([2.0]), np.array([3.0])))
    comparison.require()
    assert snapshot.evaluate((np.array([2.0]), np.array([3.0])))[0] == 12.0
    print(json.dumps({"fresh_comparison": comparison.current_valid, "historical_capture": snapshot.capture_certificate.current_valid}))
'''
    source = Path(__file__).resolve().parents[1] / "src"
    environment = {**os.environ, "PYTHONPATH": str(source)}
    path = str(tmp_path / "snapshot.json")
    for phase in ("write", "read"):
        outcome = subprocess.run([sys.executable, "-c", code, phase, path], env=environment,
                                 capture_output=True, text=True, timeout=30)
        assert outcome.returncode == 0, outcome.stderr
    assert json.loads(outcome.stdout) == {"fresh_comparison": True, "historical_capture": False}


def test_tamper_and_recomputed_digest_do_not_approve_loaded_certificates():
    original = Certificate(False, Exactness.NUMERICAL, {"identity": 1.0})
    envelope = json.loads(portable.dumps_portable(original))
    pairs = envelope["payload"]["data"]["data"]
    for pair in pairs:
        if pair[0] == "recorded_valid":
            pair[1] = True
    with pytest.raises(ValidationError, match="SHA256 mismatch"):
        portable.loads_portable(json.dumps(envelope))
    # A digest is an integrity check, not authentication. Even a matching digest
    # cannot promote the recorded validity claim into fresh evidence.
    from hashlib import sha256
    canonical = json.dumps(envelope["payload"], sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    envelope["payload_sha256"] = sha256(canonical.encode()).hexdigest()
    restored = portable.loads_portable(json.dumps(envelope))
    assert restored.recorded_valid and not restored.current_valid
    with pytest.raises(ValidationError, match="revalidation"):
        restored.require()


@pytest.mark.parametrize("field,value", [
    ("record_kind", "fresh_certificate"), ("recorded_valid", 1),
    ("recorded_valid", "true"), ("exactness", "unrecognized"), ("exactness", None),
    ("residuals", []), ("residuals", {1: 0.0}), ("residuals", {"r": True}),
    ("residuals", {"r": complex(0, 1)}), ("residuals", {"r": np.array(0.0)}),
    ("tolerance", True), ("tolerance", -1), ("tolerance", complex(0, 1)),
    ("assumptions_checked", ["list"]), ("assumptions_checked", (1,)),
    ("assumptions_unchecked", "not a tuple"), ("assumptions_unchecked", (False,)),
    ("diagnostics", []), ("theorem_id", 123), ("message", 123),
])
def test_malformed_certificate_records_are_rejected(field, value):
    original = Certificate(True, Exactness.EXACT, {"identity": 0})
    encoded = portable.to_portable_data(original)
    mapping = encoded["data"]["data"]
    for pair in mapping:
        if pair[0] == field:
            pair[1] = portable.to_portable_data(value)
    with pytest.raises(ValidationError):
        portable.from_portable_data(encoded)


def test_fresh_process_concurrent_codec_initialization():
    code = '''
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from fractions import Fraction
from komorebi_math.export.persistence import dumps_portable, loads_portable, portable_codecs
barrier = Barrier(16)
def exercise(index):
    barrier.wait(timeout=15)
    value = {"value": Fraction(index + 1, 17), "tree": ((index,),)}
    for _ in range(10):
        assert loads_portable(dumps_portable(value)) == value
    return len(portable_codecs())
with ThreadPoolExecutor(max_workers=16) as pool:
    observed = list(pool.map(exercise, range(16)))
assert len(set(observed)) == 1
print(observed[0])
'''
    source = Path(__file__).resolve().parents[1] / "src"
    outcome = subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": str(source)},
                             capture_output=True, text=True, timeout=30)
    assert outcome.returncode == 0, outcome.stderr
    assert int(outcome.stdout) > 0
    STRESS_METRICS["fresh_registry_threads"] = {"workers": 16, "roundtrips_each": 10}
