from __future__ import annotations

import numpy as np

from komorebi_math import PacketSystem
from komorebi_math.nonlinear import NonlinearPacketMap, NonlinearRefinementDefect


def packets() -> PacketSystem:
    return PacketSystem([np.diag([1, 0]), np.diag([0, 1])])


def test_nonlinear_packet_coordinate_conjugacy() -> None:
    system = packets()
    function = lambda x: np.array([x[0] ** 2 + x[1], x[0] * x[1]])
    packet_map = NonlinearPacketMap(function, system, system)
    point = np.array([2.0, 3.0])
    packet_map.certificate(point).require()
    assert np.allclose(packet_map(system.analyze(point)).reconstruct(), function(point))


def test_nonlinear_iteration() -> None:
    system = packets()
    packet_map = NonlinearPacketMap(lambda x: 0.5 * np.asarray(x), system, system)
    point = system.analyze(np.array([4.0, 8.0]))
    result = packet_map.iterate(point, 3)
    assert np.allclose(result.reconstruct(), [0.5, 1.0])


def test_nonlinear_refinement_defect_and_fixed_point_bound() -> None:
    defect = NonlinearRefinementDefect(
        fine_map=lambda x: np.asarray(x),
        coarse_map=lambda y: 0.5 * np.asarray(y),
        coarsen_source=np.array([[1.0, 1.0]]),
        coarsen_target=np.array([[1.0, 1.0]]),
    )
    point = np.array([2.0, 4.0])
    assert np.allclose(defect.evaluate(point), [3.0])
    assert abs(defect.fixed_point_bound(point, coarse_contraction=0.5) - 6.0) < 1e-12


def test_iterated_orbit_bound() -> None:
    defect = NonlinearRefinementDefect(
        fine_map=lambda x: 0.5 * np.asarray(x),
        coarse_map=lambda y: 0.5 * np.asarray(y),
        coarsen_source=np.eye(1),
        coarsen_target=np.eye(1),
    )
    assert defect.iterated_orbit_bound(np.array([2.0]), iterations=5, coarse_lipschitz=0.5) == 0.0


def test_nonlinear_composition_defect_identity() -> None:
    defect = NonlinearRefinementDefect(
        fine_map=lambda x: np.array([x[0] ** 2 + x[1], x[0] - x[1]]),
        coarse_map=lambda y: np.array([y[0] ** 2]),
        coarsen_source=np.array([[1.0, 0.5]]),
        coarsen_target=np.array([[1.0, -0.25]]),
    )
    result = defect.composition_identity(
        outer_fine=lambda z: np.array([z[0] + z[1], z[0] * z[1]]),
        outer_coarse=lambda w: w**2 + 0.3 * w,
        outer_coarsen_target=np.array([[1.0, 0.4]]),
        point=np.array([0.3, -0.8]),
    )
    result.certificate.require()


def test_fixed_point_bound_rejects_nonfixed_point() -> None:
    defect = NonlinearRefinementDefect(
        fine_map=lambda x: 0.5 * np.asarray(x),
        coarse_map=lambda y: 0.5 * np.asarray(y),
        coarsen_source=np.eye(1),
        coarsen_target=np.eye(1),
    )
    import pytest
    from komorebi_math.exceptions import ValidationError

    with pytest.raises(ValidationError, match="not a fine fixed point"):
        defect.fixed_point_bound(np.array([2.0]), coarse_contraction=0.5)
