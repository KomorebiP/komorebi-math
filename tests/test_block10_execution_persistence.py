from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pytest

from komorebi_math import Exactness, PacketSystem
from komorebi_math.certificates import Certificate
from komorebi_math.deformation import LiftBranch
from komorebi_math.exceptions import BackendUnavailableError, ValidationError
from komorebi_math.export import (
    dumps_portable,
    from_serializable,
    load_portable,
    loads_portable,
    portable_digest,
    register_portable_codec,
    save_portable,
    to_serializable,
)
from komorebi_math.nonlinear import (
    CuPyPacketJetExecutor,
    CuPyPacketJetSnapshotExecutor,
    FiniteLiftTree,
    LiftTreeNode,
    OrderedRootedTree,
    PacketJetFamily,
    PolynomialDerivativeProvider,
    SparseSymmetricTensor,
    compile_packet_jet_plan,
    cupy_runtime_status,
    execute_packet_jet_plan,
    packet_jet_support,
    snapshot_packet_jet,
)
from komorebi_math.nonlinear.filtered import (
    FilteredSpectralSequenceResult,
    SpectralSequenceEntry,
    SpectralSequencePage,
)


def coordinate_family() -> tuple[PacketJetFamily, tuple[np.ndarray, np.ndarray]]:
    tensor = np.zeros((2, 2, 2), dtype=float)
    tensor[0, 0, 0] = 2.0
    tensor[1, 1, 1] = -3.0
    provider = PolynomialDerivativeProvider({2: tensor}, source_dimension=2, target_dimension=2)
    packets = PacketSystem(
        (np.diag([1.0, 0.0]), np.diag([0.0, 1.0])),
        names=("x", "y"),
    )
    family = PacketJetFamily(provider, np.array([0.2, -0.4]), packets, packets, 2)
    return family, (np.array([2.0, 3.0]), np.array([-1.0, 4.0]))


def test_projection_cached_plan_prunes_and_reconstructs() -> None:
    family, directions = coordinate_family()
    support = packet_jet_support(family)
    support.certificate.require()
    plan = compile_packet_jet_plan(family, support)
    plan.certificate.require()
    assert plan.full_address_count == 8
    assert plan.active_address_count == 2
    assert plan.pruned_address_count == 6
    assert plan.derivative_call_count == 2
    assert plan.naive_derivative_call_count == 8
    assert plan.derivative_call_reduction == pytest.approx(0.75)

    result = execute_packet_jet_plan(plan, family, directions)
    result.certificate.require()
    assert result.derivative_calls == 2
    assert result.pruned_blocks == 6
    assert np.allclose(result.reconstructed, result.direct)
    assert len(result.evaluations) == 2


def test_projection_cached_plan_without_support_is_conservative() -> None:
    family, directions = coordinate_family()
    plan = compile_packet_jet_plan(family)
    assert plan.active_address_count == 8
    assert plan.pruned_address_count == 0
    assert plan.derivative_call_count == 4
    result = execute_packet_jet_plan(plan, family, directions)
    result.certificate.require()
    assert len(result.evaluations) == 8


def test_projection_cached_plan_guard_and_family_mismatch() -> None:
    family, directions = coordinate_family()
    with pytest.raises(ValidationError, match="above max_addresses"):
        compile_packet_jet_plan(family, max_addresses=7)
    other_packets = PacketSystem((np.eye(2),), names=("all",))
    other_family = PacketJetFamily(family.provider, family.point, other_packets, other_packets, 2)
    plan = compile_packet_jet_plan(family)
    with pytest.raises(ValidationError, match="does not match"):
        execute_packet_jet_plan(plan, other_family, directions)


def test_packet_jet_snapshot_reconstructs_fixed_block() -> None:
    family, directions = coordinate_family()
    jet = family.jet("x", ("x", "x"))
    snapshot = snapshot_packet_jet(jet)
    snapshot.capture_certificate.require()
    assert snapshot.order == 2
    assert snapshot.nnz == 1
    assert np.allclose(snapshot.evaluate(directions), jet.evaluate(directions))
    snapshot.comparison_certificate(jet, directions).require()
    dense = snapshot.to_dense()
    assert dense.shape == (2, 2, 2)
    assert dense[0, 0, 0] == pytest.approx(2.0)


def test_packet_jet_snapshot_evaluation_guard() -> None:
    family, _ = coordinate_family()
    jet = family.jet("x", ("x", "x"))
    with pytest.raises(ValidationError, match="exceeding"):
        snapshot_packet_jet(jet, max_evaluations=3)


def spectral_result() -> FilteredSpectralSequenceResult:
    certificate = Certificate(valid=True, exactness=Exactness.EXACT, message="finite spectral page")
    entry = SpectralSequenceEntry(1, 0, 1, 1, 2, 1, 0)
    page = SpectralSequencePage(1, (entry,), 2, 1, certificate)
    return FilteredSpectralSequenceResult((page,), {1: 1}, {1: 1}, certificate)


def persistence_cases() -> tuple[object, ...]:
    family, _ = coordinate_family()
    snapshot = snapshot_packet_jet(family.jet("x", ("x", "x")))
    tree = OrderedRootedTree((OrderedRootedTree.leaf(), OrderedRootedTree.leaf()))
    tensor = SparseSymmetricTensor.from_mapping(2, 2, 1, {(0, 0, 1): Fraction(3, 5)})
    lift_tree = FiniteLiftTree(
        3,
        (
            (LiftTreeNode((1,), 1, None),),
            (LiftTreeNode((1,), 2, 0),),
        ),
    )
    return (
        tree,
        snapshot,
        tensor,
        LiftBranch(((1,), (1, 4))),
        lift_tree,
        spectral_result(),
        {"fraction": Fraction(2, 7), "array": np.arange(6).reshape(2, 3)},
    )


@pytest.mark.parametrize("value", persistence_cases())
def test_portable_roundtrip_for_block10_types(value: object) -> None:
    encoded = dumps_portable(value)
    decoded = loads_portable(encoded)
    assert dumps_portable(decoded) == encoded


def test_portable_digest_is_deterministic_for_mapping_order() -> None:
    assert portable_digest({"a": 1, "b": 2}) == portable_digest({"b": 2, "a": 1})


def test_portable_payload_tampering_is_detected() -> None:
    envelope = json.loads(dumps_portable({"value": 4}))
    envelope["payload"]["data"][0][1] = 5
    with pytest.raises(ValidationError, match="SHA256 mismatch"):
        loads_portable(json.dumps(envelope))


def test_portable_unknown_codec_and_version_are_rejected() -> None:
    envelope = json.loads(dumps_portable({"value": 4}))
    envelope["payload"] = {
        "__komorebi_type__": "unknown.codec",
        "__version__": 1,
        "data": None,
    }
    from hashlib import sha256

    canonical = json.dumps(envelope["payload"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    envelope["payload_sha256"] = sha256(canonical.encode()).hexdigest()
    with pytest.raises(ValidationError, match="unknown portable codec"):
        loads_portable(json.dumps(envelope))


def test_portable_file_save_load_and_overwrite_guard(tmp_path) -> None:
    path = tmp_path / "state.km.json"
    value = persistence_cases()[1]
    assert save_portable(value, path) == path
    decoded = load_portable(path)
    assert dumps_portable(decoded) == dumps_portable(value)
    with pytest.raises(ValidationError, match="refusing to overwrite"):
        save_portable(value, path)
    save_portable(value, path, overwrite=True)
    assert not tuple(tmp_path.glob("*.tmp"))


def test_safe_export_marker_decoder_roundtrip() -> None:
    value = {
        "fraction": Fraction(5, 9),
        "complex": 2 - 3j,
        "array": np.array([[1, 2], [3, 4]], dtype=np.int64),
    }
    decoded = from_serializable(to_serializable(value))
    assert decoded["fraction"] == Fraction(5, 9)
    assert decoded["complex"] == 2 - 3j
    assert np.array_equal(decoded["array"], value["array"])


def test_safe_export_decoder_never_instantiates_arbitrary_type() -> None:
    payload = {"__type__": "dangerous.module.Class", "value": 3}
    assert from_serializable(payload) == payload


@dataclass(frozen=True)
class CustomPortableValue:
    value: int


def test_explicit_custom_portable_codec() -> None:
    register_portable_codec(
        CustomPortableValue,
        tag="tests.custom_portable_value",
        version=1,
        encoder=lambda value: {"value": value.value},
        decoder=lambda data: CustomPortableValue(int(data["value"])),
        replace=True,
    )
    value = CustomPortableValue(17)
    assert loads_portable(dumps_portable(value)) == value


def test_cupy_contract_reports_unavailable_hardware_honestly() -> None:
    family, _ = coordinate_family()
    snapshot = snapshot_packet_jet(family.jet("x", ("x", "x")))
    status = cupy_runtime_status()
    if not status.runtime_available:
        with pytest.raises(BackendUnavailableError):
            CuPyPacketJetSnapshotExecutor(snapshot)
        with pytest.raises(BackendUnavailableError):
            CuPyPacketJetExecutor(family)


def test_portable_json_rejects_nonfinite_constants() -> None:
    with pytest.raises(ValidationError, match="nonfinite constant"):
        loads_portable('{"schema": NaN}')


def test_portable_json_rejects_duplicate_object_keys() -> None:
    with pytest.raises(ValidationError, match="duplicate object key"):
        loads_portable('{"schema": "first", "schema": "second"}')


def test_malformed_builtin_payload_raises_validation_error() -> None:
    from hashlib import sha256

    envelope = json.loads(dumps_portable(Fraction(1, 2)))
    envelope["payload"]["data"]["denominator"] = 0
    canonical = json.dumps(
        envelope["payload"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    envelope["payload_sha256"] = sha256(canonical.encode()).hexdigest()
    with pytest.raises(ValidationError, match="Fraction data is invalid"):
        loads_portable(json.dumps(envelope))
