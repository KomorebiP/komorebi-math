"""Installed verification suite for advanced nonlinear calculus."""

from __future__ import annotations

import numpy as np

from ..certificates import Exactness
from ..core import PacketSystem
from ..dynamics._linalg import matrix_exponential
from ..nonlinear import (
    CallableDerivativeProvider,
    FilteredCorrectionProblem,
    FilteredVectorSpace,
    FiniteAbelianGroup,
    FiniteFilteredCochainComplex,
    HilbertObstructionProjector,
    KatoTransport,
    LyapunovSchmidtReduction,
    MovingPacketSystem,
    NonlinearSchurElimination,
    PacketJetAddress,
    PacketJetFamily,
    PolynomialDerivativeProvider,
    PolynomialFixedPointProblem,
    divided_square_packets,
    faa_di_bruno,
    majorant_convergence,
    packet_jet_support,
    packet_taylor,
    set_partitions,
    verify_finite_filtered_branch,
)
from .core import VerificationCheck, VerificationReport, _capture


def run_advanced_nonlinear_verification() -> VerificationReport:
    """Run the installed Block 9 nonlinear verification suite."""

    checks: list[VerificationCheck] = []

    def composition_check() -> None:
        outer = PolynomialDerivativeProvider(
            {2: np.array([[[2]]], dtype=object)},
            exactness=Exactness.EXACT,
        )
        inner = PolynomialDerivativeProvider(
            {
                1: np.array([[1]], dtype=object),
                2: np.array([[[2]]], dtype=object),
            },
            exactness=Exactness.EXACT,
        )
        result = faa_di_bruno(
            outer,
            inner,
            np.array([2], dtype=object),
            (np.array([3], dtype=object), np.array([4], dtype=object)),
        )
        result.certificate.require()
        assert int(result.value[0]) == 888
        assert tuple(len(set_partitions(order)) for order in range(6)) == (1, 1, 2, 5, 15, 52)

    checks.append(_capture("Faà di Bruno composition", composition_check))

    def jet_check() -> None:
        tensor = np.zeros((2, 2, 2), dtype=float)
        tensor[0, 0, 0] = 2.0
        tensor[1, 1, 1] = 2.0
        provider = PolynomialDerivativeProvider({2: tensor}, source_dimension=2, target_dimension=2)
        packets = PacketSystem(
            (np.diag([1.0, 0.0]), np.diag([0.0, 1.0])),
            names=("x", "y"),
        )
        family = PacketJetFamily(provider, np.array([1.0, 2.0]), packets, packets, 2)
        directions = (np.array([2.0, 3.0]), np.array([-1.0, 4.0]))
        family.reconstruction_certificate(directions).require()
        support = packet_jet_support(family)
        support.certificate.require()
        assert set(support.nonzero) == {
            PacketJetAddress("x", ("x", "x")),
            PacketJetAddress("y", ("y", "y")),
        }
        packet_taylor(provider, np.array([1.0, 2.0]), np.array([0.2, -0.1]), packets, packets, 2).certificate.require()

    checks.append(_capture("multilinear packet jets", jet_check))

    def tree_check() -> None:
        problem = PolynomialFixedPointProblem(
            np.array([0.1]),
            {2: lambda values: np.array([values[0][0] * values[1][0]])},
        )
        expansion = problem.tree_expansion(4, max_trees=100_000)
        expansion.certificate.require()
        assert np.allclose(expansion.value, expansion.picard_value)
        majorant = majorant_convergence(0.1, {2: 1.0}, 0.2)
        majorant.certificate.require()
        assert majorant.picard_tail_bound(4) > 0

    checks.append(_capture("rooted nonlinear tree expansion", tree_check))

    def graded_check() -> None:
        group = FiniteAbelianGroup.cyclic(3)
        components = {
            "0": np.array([1.0, 2.0]),
            "1": np.array([-0.5, 0.25]),
            "2": np.array([0.3, -0.7]),
        }
        group.certificate().require()
        result = divided_square_packets(
            group,
            components,
            differential=lambda value: 3.0 * value,
            bracket=lambda left, right: 2.0 * left * right,
            divided_square=lambda value: value * value,
        )
        result.certificate.require()
        assert np.allclose(result.reconstructed, result.direct)

    checks.append(_capture("graded packet Maurer-Cartan identities", graded_check))

    def correction_check() -> None:
        projector = HilbertObstructionProjector([[1.0, 0.0], [0.0, 0.0]])
        obstructed = projector.solve([0.0, 1.0])
        obstructed.certificate.require()
        assert not obstructed.vanishes
        provider = CallableDerivativeProvider(
            lambda point: np.array([point[1], point[0] ** 2]),
            {
                1: lambda point, directions: np.array(
                    [directions[0][1], 2 * point[0] * directions[0][0]]
                ),
                2: lambda point, directions: np.array(
                    [0.0, 2 * directions[0][0] * directions[1][0]]
                ),
            },
            source_dimension=2,
            target_dimension=2,
        )
        reduction = LyapunovSchmidtReduction(provider, [0.0, 0.0], tolerance=1e-12)
        reduction.base_derivative_certificate().require()
        reduced = reduction.solve_range([0.3])
        reduced.certificate.require()
        assert np.allclose(np.abs(reduced.reduced_obstruction), [0.09], atol=1e-12)

    checks.append(_capture("obstruction and Lyapunov-Schmidt reduction", correction_check))

    def schur_check() -> None:
        retained = CallableDerivativeProvider(
            lambda point: np.array([3 * point[0] + 2 * point[1]]),
            {1: lambda point, directions: np.array([3 * directions[0][0] + 2 * directions[0][1]])},
            source_dimension=2,
            target_dimension=1,
        )
        hidden = CallableDerivativeProvider(
            lambda point: np.array([point[1] - 4 * point[0]]),
            {1: lambda point, directions: np.array([directions[0][1] - 4 * directions[0][0]])},
            source_dimension=2,
            target_dimension=1,
        )
        reduction = NonlinearSchurElimination(retained, hidden, 1, 1, [0.0], [0.0])
        first = reduction.first_derivatives()
        first.certificate.require()
        assert np.allclose(first.implicit_derivative, [[4.0]])
        assert np.allclose(first.reduced_derivative, [[11.0]])
        solved = reduction.solve_hidden([0.2])
        solved.certificate.require()
        assert np.allclose(solved.hidden, [0.8])

    checks.append(_capture("nonlinear Schur elimination", schur_check))

    def filtered_lifting_check() -> None:
        source = FilteredVectorSpace.from_weights([0, 1], prime=5)
        target = FilteredVectorSpace.from_weights([0, 1], prime=5)
        result = FilteredCorrectionProblem(source, target, np.eye(2, dtype=int), [0, 1], 1).solve()
        result.certificate.require()
        assert result.vanishes
        branch = verify_finite_filtered_branch(
            source,
            target,
            lambda point: (point[0] - 1, point[1] - 2),
            ((0, 0), (1, 0), (1, 2)),
        )
        branch.certificate.require()
        assert branch.exact_solution

    checks.append(_capture("finite filtered lifting", filtered_lifting_check))

    def spectral_sequence_check() -> None:
        c0 = FilteredVectorSpace(
            1,
            {0: np.eye(1, dtype=int), 1: np.zeros((1, 0), dtype=int), 2: np.zeros((1, 0), dtype=int)},
            prime=5,
        )
        c1 = FilteredVectorSpace.from_weights([1], prime=5)
        complex_ = FiniteFilteredCochainComplex(
            {0: 1, 1: 1},
            {0: [[1]]},
            {0: c0, 1: c1},
            prime=5,
        )
        result = complex_.spectral_sequence()
        result.certificate.require()
        assert result.pages[1].entry(0, 0).outgoing_rank == 1
        assert result.pages[2].total_dimension == 0

    checks.append(_capture("filtered obstruction spectral sequence", spectral_sequence_check))

    def transport_check() -> None:
        generator = 0.4 * np.array([[0, -1], [1, 0]], dtype=complex)
        base = np.diag([1, 0]).astype(complex)

        def packets(time: float):
            rotation = matrix_exponential(time * generator)
            first = rotation @ base @ rotation.conj().T
            return first, np.eye(2) - first

        def derivatives(time: float):
            return tuple(generator @ value - value @ generator for value in packets(time))

        system = MovingPacketSystem(packets, derivatives)
        system.derivative_certificate(0.3, step=1e-5).certificate.require()
        transport = KatoTransport(system)
        result = transport.refinement_certificate(0.0, 1.0, coarse_steps=32, tolerance=1e-6)
        result.certificate.require()

    checks.append(_capture("shared smooth packet transport", transport_check))

    return VerificationReport(tuple(checks))


__all__ = ["run_advanced_nonlinear_verification"]
