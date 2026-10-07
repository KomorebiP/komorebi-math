"""Installed verification for the Block 10 scalable nonlinear runtime."""

from __future__ import annotations

import json
from fractions import Fraction

import numpy as np

from ..certificates import Exactness
from ..compatibility import check_public_api
from ..core import PacketSystem
from ..exceptions import ValidationError
from ..export import dumps_portable, loads_portable
from ..nonlinear import (
    AutomaticDifferentiationProvider,
    PacketJetFamily,
    PolynomialDerivativeProvider,
    SparsePolynomialDerivativeProvider,
    SparseSymmetricTensor,
    compile_packet_jet_plan,
    cupy_runtime_status,
    execute_packet_jet_plan,
    packet_jet_support,
    snapshot_packet_jet,
)
from .core import VerificationCheck, VerificationReport, _capture


def _coordinate_family() -> tuple[PacketJetFamily, tuple[np.ndarray, np.ndarray]]:
    tensor = np.zeros((2, 2, 2), dtype=float)
    tensor[0, 0, 0] = 2.0
    tensor[1, 1, 1] = -3.0
    provider = PolynomialDerivativeProvider({2: tensor}, source_dimension=2, target_dimension=2)
    packets = PacketSystem(
        (np.diag([1.0, 0.0]), np.diag([0.0, 1.0])),
        names=("x", "y"),
    )
    family = PacketJetFamily(provider, np.array([0.25, -0.5]), packets, packets, 2)
    directions = (np.array([2.0, 3.0]), np.array([-1.0, 4.0]))
    return family, directions


def run_scalable_nonlinear_verification() -> VerificationReport:
    """Run the installed Block 10 hardening and scalability checks."""

    checks: list[VerificationCheck] = []

    def sparse_check() -> None:
        dense_tensors = {
            0: np.array([Fraction(1, 3)], dtype=object),
            1: np.array([[Fraction(2), Fraction(-1)]], dtype=object),
            2: np.array(
                [[[Fraction(4), Fraction(3)], [Fraction(3), Fraction(-2)]]],
                dtype=object,
            ),
        }
        dense = PolynomialDerivativeProvider(dense_tensors, exactness=Exactness.EXACT)
        sparse = SparsePolynomialDerivativeProvider.from_dense_provider(dense)
        point = np.array([Fraction(2, 5), Fraction(-1, 7)], dtype=object)
        directions = (
            np.array([Fraction(3, 4), Fraction(2, 9)], dtype=object),
            np.array([Fraction(-5, 6), Fraction(1, 8)], dtype=object),
        )
        sparse.storage_certificate((point,), directions).require()
        assert np.array_equal(sparse.value(point), dense.value(point))
        assert sparse.stored_entry_count < sparse.dense_entry_count

    checks.append(_capture("sparse symmetric derivative storage", sparse_check))

    def execution_check() -> None:
        family, directions = _coordinate_family()
        support = packet_jet_support(family)
        support.certificate.require()
        plan = compile_packet_jet_plan(family, support)
        plan.certificate.require()
        result = execute_packet_jet_plan(plan, family, directions)
        result.certificate.require()
        assert result.derivative_calls == 2
        assert result.pruned_blocks == 6
        assert np.allclose(result.reconstructed, result.direct)

    checks.append(_capture("projection-cached packet-jet execution", execution_check))

    def snapshot_persistence_check() -> None:
        family, directions = _coordinate_family()
        jet = family.jet("x", ("x", "x"))
        snapshot = snapshot_packet_jet(jet)
        snapshot.capture_certificate.require()
        snapshot.comparison_certificate(jet, directions).require()
        encoded = dumps_portable(snapshot)
        decoded = loads_portable(encoded)
        assert dumps_portable(decoded) == encoded
        assert np.allclose(decoded.evaluate(directions), jet.evaluate(directions))

        envelope = json.loads(encoded)
        envelope["payload_sha256"] = "0" * 64
        try:
            loads_portable(json.dumps(envelope))
        except ValidationError:
            pass
        else:  # pragma: no cover - defensive invariant
            raise AssertionError("portable persistence accepted a modified digest")

    checks.append(_capture("digest-checked nonlinear persistence", snapshot_persistence_check))

    def generic_ad_check() -> None:
        def function(value: np.ndarray) -> np.ndarray:
            return np.array([value[0] ** 3 + value[0] * value[1]])

        def jvp(function_, point, direction):
            step = 1e-6
            return function_(point), (
                function_(point + step * direction) - function_(point - step * direction)
            ) / (2 * step)

        provider = AutomaticDifferentiationProvider(function, jvp, 2, 1, max_order=1)
        point = np.array([0.7, -0.2])
        direction = np.array([0.4, 1.3])
        expected = (3 * point[0] ** 2 + point[1]) * direction[0] + point[0] * direction[1]
        assert np.allclose(provider.derivative(1, point, (direction,)), [expected], atol=1e-8)

    checks.append(_capture("framework-neutral directional AD adapter", generic_ad_check))

    def api_check() -> None:
        report = check_public_api()
        report.certificate.require()
        assert not report.removed
        assert not report.kind_changed

    checks.append(_capture("committed public API manifest", api_check))

    def gpu_boundary_check() -> None:
        status = cupy_runtime_status()
        if status.runtime_available:
            assert status.installed and status.device_count > 0
        else:
            assert status.device_count == 0
            assert status.error

    checks.append(_capture("honest optional CuPy runtime boundary", gpu_boundary_check))

    def sparse_tensor_persistence_check() -> None:
        tensor = SparseSymmetricTensor.from_mapping(
            3,
            3,
            1,
            {(0, 0, 1, 2): Fraction(7, 11)},
        )
        restored = loads_portable(dumps_portable(tensor))
        assert restored == tensor
        restored.certificate().require()

    checks.append(_capture("sparse tensor portable roundtrip", sparse_tensor_persistence_check))

    return VerificationReport(tuple(checks))


__all__ = ["run_scalable_nonlinear_verification"]
