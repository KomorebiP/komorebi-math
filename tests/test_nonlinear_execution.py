from __future__ import annotations

import numpy as np
import pytest

from komorebi_math import PacketSystem
from komorebi_math.exceptions import BackendUnavailableError, ValidationError
from komorebi_math.nonlinear.derivatives import PolynomialDerivativeProvider
from komorebi_math.nonlinear.execution import (
    CuPyPacketJetExecutor,
    CuPyPolynomialDerivativeProvider,
    compile_packet_jet_plan,
    execute_packet_jet_plan,
    execute_packet_jets,
    nonlinear_cupy_available,
    packet_jet_benchmark_contract,
)
from komorebi_math.nonlinear.gpu import cupy_runtime_status
from komorebi_math.nonlinear.jets import PacketJetAddress, PacketJetFamily, packet_jet_support


def _coordinate_family() -> PacketJetFamily:
    packets = PacketSystem(
        [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])],
        names=("x", "y"),
    )
    quadratic = np.zeros((2, 2, 2))
    quadratic[0, 0, 0] = 2.0
    quadratic[1, 1, 1] = 2.0
    provider = PolynomialDerivativeProvider(
        {2: quadratic},
        source_dimension=2,
        target_dimension=2,
        name="separable quadratic",
    )
    return PacketJetFamily(provider, np.zeros(2), packets, packets, 2)


def test_projection_cached_plan_reduces_target_repeated_derivative_calls() -> None:
    family = _coordinate_family()
    plan = compile_packet_jet_plan(family)

    assert plan.full_address_count == 8
    assert plan.active_address_count == 8
    assert plan.derivative_call_count == 4
    assert plan.naive_derivative_call_count == 8
    assert plan.derivative_call_reduction == 0.5
    plan.certificate.require()

    directions = (np.array([2.0, 3.0]), np.array([-1.0, 4.0]))
    result = execute_packet_jet_plan(plan, family, directions)
    result.certificate.require()
    assert result.derivative_calls == 4
    assert result.input_projections == 4
    assert result.output_projections == 8
    assert np.allclose(result.reconstructed, result.direct)
    assert len(result.evaluations) == 8


def test_certified_support_prunes_zero_blocks_without_changing_result() -> None:
    family = _coordinate_family()
    support = packet_jet_support(family)
    support.certificate.require()
    plan = compile_packet_jet_plan(family, support)

    assert plan.support_complete
    assert plan.active_address_count == 2
    assert plan.pruned_address_count == 6
    assert plan.derivative_call_count == 2
    assert set(plan.active_addresses) == {
        PacketJetAddress("x", ("x", "x")),
        PacketJetAddress("y", ("y", "y")),
    }

    directions = (np.array([2.0, 3.0]), np.array([-1.0, 4.0]))
    result = execute_packet_jets(family, directions, support=support)
    result.certificate.require()
    assert result.pruned_blocks == 6
    assert result.derivative_calls == 2
    assert np.allclose(result.reconstructed, [-4.0, 24.0])
    assert np.allclose(result.direct, [-4.0, 24.0])


def test_incomplete_support_keeps_untested_blocks_active() -> None:
    family = _coordinate_family()
    support = packet_jet_support(family, max_evaluations=1)
    plan = compile_packet_jet_plan(family, support)

    assert not plan.support_complete
    assert plan.active_address_count == len(support.nonzero) + len(support.untested)
    assert plan.pruned_address_count == len(support.zero)
    assert plan.certificate.assumptions_unchecked


def test_execution_plan_validates_family_and_safety_limit() -> None:
    family = _coordinate_family()
    plan = compile_packet_jet_plan(family)
    other_packets = PacketSystem(
        [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])],
        names=("left", "right"),
    )
    other = PacketJetFamily(
        family.provider,
        family.point,
        other_packets,
        other_packets,
        family.order,
    )
    with pytest.raises(ValidationError, match="does not match"):
        execute_packet_jet_plan(plan, other, (np.ones(2), np.ones(2)))
    with pytest.raises(ValidationError, match="above max_addresses"):
        compile_packet_jet_plan(family, max_addresses=7)


def test_gpu_boundary_is_explicit_without_hardware_claims() -> None:
    status = cupy_runtime_status()
    assert status.installed is nonlinear_cupy_available()
    if not nonlinear_cupy_available():
        with pytest.raises(BackendUnavailableError, match="not installed"):
            CuPyPolynomialDerivativeProvider({1: np.eye(2)})
        with pytest.raises(BackendUnavailableError, match="not installed"):
            CuPyPacketJetExecutor(_coordinate_family())
    else:  # pragma: no cover - depends on the external runtime
        assert status.installed


def test_packet_jet_benchmark_contract_records_matched_workload() -> None:
    family = _coordinate_family()
    contract = packet_jet_benchmark_contract(family, backend="numpy")
    parameters = dict(contract.case_signature)
    assert contract.backend == "numpy"
    assert parameters["address_count"] == 8
    assert "active packet-address values" in contract.output_contract
