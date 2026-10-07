"""Adversarial closure tests for the Block 18 coefficient-frame algebra.

The fixtures in this module deliberately include oblique redundant systems.
They prevent the implementation from silently replacing a left inverse by an
orthogonal/Parseval assumption, and they distinguish coefficient-space block
composition from the generally false one-index physical-component product.
"""

from __future__ import annotations

from fractions import Fraction
import inspect

import numpy as np
import pytest

import komorebi_math.api.v1 as api_v1
import komorebi_math.stable as stable
import komorebi_math.v1 as v1
from komorebi_math.compatibility import check_stable_api, create_stable_api_manifest
from komorebi_math.coordinates import (
    AnalysisSynthesisCoefficientBlocks,
    AnalysisSynthesisGauge,
    AnalysisSynthesisMap,
    AnalysisSynthesisSystem,
    CoefficientPartition,
    ExactAnalysisSynthesisMap,
    ExactAnalysisSynthesisSystem,
)
from komorebi_math.coordinates.exact import (
    identity_matrix,
    matrix_multiply,
)
from komorebi_math.exceptions import ValidationError
from komorebi_math.stable_semantics import run_stable_semantics_fixture


NUMERICAL_TOLERANCE = 2e-12


def _assert_close(left: object, right: object, *, tolerance: float = NUMERICAL_TOLERANCE) -> None:
    np.testing.assert_allclose(
        np.asarray(left),
        np.asarray(right),
        atol=tolerance,
        rtol=tolerance,
    )


def _oblique_system(*, source: str = "physical", coordinates: str = "oblique") -> AnalysisSynthesisSystem:
    analysis = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    synthesis = np.array(
        [
            [0.5, -0.5, 0.5],
            [-0.25, 0.75, 0.25],
        ]
    )
    system = AnalysisSynthesisSystem(
        analysis,
        synthesis,
        source=source,
        coordinates=coordinates,
    )
    system.certificate().require()
    return system


def _canonical_system(
    analysis: object,
    *,
    source: str,
    coordinates: str,
) -> AnalysisSynthesisSystem:
    system = AnalysisSynthesisSystem.canonical_dual(
        np.asarray(analysis, dtype=float),
        source=source,
        coordinates=coordinates,
    )
    system.certificate().require()
    return system


def test_oblique_projection_and_gauge_decomposition_are_closed() -> None:
    system = _oblique_system()
    gauge = AnalysisSynthesisGauge(system)
    projection = gauge.coefficient_projection()
    gauge_projection = np.eye(system.coordinate_dimension) - projection

    _assert_close(projection @ projection, projection)
    _assert_close(gauge_projection @ gauge_projection, gauge_projection)
    _assert_close(projection @ gauge_projection, np.zeros_like(projection))
    _assert_close(gauge_projection @ projection, np.zeros_like(projection))
    _assert_close(projection + gauge_projection, np.eye(3))
    _assert_close(projection @ system.analysis, system.analysis)
    _assert_close(system.synthesis @ projection, system.synthesis)
    assert np.linalg.matrix_rank(projection) == system.source_dimension
    assert np.linalg.norm(projection - projection.conj().T) > 0.4

    assert isinstance(gauge, AnalysisSynthesisGauge)

    coefficient = np.array([3.0, -4.0, 5.0])
    kernel_vector = np.array([-2.0, -1.0, 1.0])
    canonical = gauge.canonical_coefficients(coefficient)
    discarded = gauge.gauge_part(coefficient)

    _assert_close(system.synthesis @ kernel_vector, np.zeros(2))
    _assert_close(projection @ kernel_vector, np.zeros(3))
    _assert_close(canonical + discarded, coefficient)
    _assert_close(system.synthesis @ discarded, np.zeros(2))
    _assert_close(projection @ discarded, np.zeros(3))
    _assert_close(gauge.canonical_coefficients(coefficient + kernel_vector), canonical)
    assert gauge.gauge_equivalent(coefficient, coefficient + kernel_vector)
    assert not gauge.gauge_equivalent(coefficient, coefficient + np.array([1.0, 0.0, 0.0]))


def test_coefficient_metric_and_dual_system_respect_the_quotient() -> None:
    system = _oblique_system()
    gauge = AnalysisSynthesisGauge(system)
    projection = gauge.coefficient_projection()
    metric = gauge.coefficient_metric()

    _assert_close(metric, system.synthesis.conj().T @ system.synthesis)
    _assert_close(metric, metric.conj().T)
    _assert_close(projection.conj().T @ metric, metric)
    _assert_close(metric @ projection, metric)
    eigenvalues = np.linalg.eigvalsh(metric)
    assert np.min(eigenvalues) >= -NUMERICAL_TOLERANCE
    assert np.linalg.matrix_rank(metric) == system.source_dimension

    left = np.array([2.0, -1.0, 4.0])
    right = np.array([-3.0, 5.0, 1.0])
    coefficient_pairing = left.conj() @ metric @ right
    source_pairing = (system.synthesis @ left).conj() @ (system.synthesis @ right)
    _assert_close(coefficient_pairing, source_pairing)

    dual = gauge.dual()
    dual.certificate().require()
    _assert_close(dual.analysis, system.synthesis.conj().T)
    _assert_close(dual.synthesis, system.analysis.conj().T)
    _assert_close(dual.coefficient_projection(), projection.conj().T)
    double_dual = dual.dual()
    _assert_close(double_dual.analysis, system.analysis)
    _assert_close(double_dual.synthesis, system.synthesis)


def test_map_lift_reconstructs_descends_gauge_and_has_only_physical_support() -> None:
    domain = _oblique_system(source="domain", coordinates="domain frame")
    codomain = _canonical_system(
        [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [2.0, 1.0]],
        source="codomain",
        coordinates="codomain frame",
    )
    operator = np.array([[2.0, -1.0], [3.0, 4.0]])
    mapping = AnalysisSynthesisMap(domain, codomain, operator)
    mapping.certificate().require()

    lift = mapping.coefficient_lift()
    domain_projection = AnalysisSynthesisGauge(domain).coefficient_projection()
    codomain_projection = AnalysisSynthesisGauge(codomain).coefficient_projection()
    domain_gauge = np.eye(domain.coordinate_dimension) - domain_projection
    codomain_gauge = np.eye(codomain.coordinate_dimension) - codomain_projection

    assert lift.shape == (codomain.coordinate_dimension, domain.coordinate_dimension)
    _assert_close(lift, codomain.analysis @ operator @ domain.synthesis)
    _assert_close(codomain_projection @ lift @ domain_projection, lift)
    _assert_close(codomain.synthesis @ lift @ domain.analysis, operator)
    _assert_close(mapping.reconstruct(), operator)
    _assert_close(lift @ domain_gauge, np.zeros_like(lift))
    _assert_close(codomain_gauge @ lift, np.zeros_like(lift))

    coefficient = np.array([1.0, -3.0, 2.0])
    kernel_vector = np.array([-2.0, -1.0, 1.0])
    output = mapping.apply_coefficients(coefficient)
    _assert_close(output, lift @ coefficient)
    _assert_close(mapping.apply_coefficients(coefficient + kernel_vector), output)
    _assert_close(codomain.synthesis @ output, operator @ domain.synthesis @ coefficient)


def test_lifted_identity_is_projection_not_redundant_coordinate_identity() -> None:
    system = _oblique_system()
    gauge = AnalysisSynthesisGauge(system)
    identity = AnalysisSynthesisMap(system, system, np.eye(system.source_dimension))
    projection = gauge.coefficient_projection()

    _assert_close(identity.coefficient_lift(), projection)
    assert not np.allclose(projection, np.eye(system.coordinate_dimension))
    coefficient = np.array([3.0, -1.0, 2.0])
    _assert_close(identity.apply_coefficients(coefficient), gauge.canonical_coefficients(coefficient))
    canonical = system.analyze(np.array([2.0, -4.0]))
    _assert_close(identity.apply_coefficients(canonical), canonical)


def test_composition_across_different_middle_presentations_uses_transition_bridge() -> None:
    domain = _oblique_system(source="domain", coordinates="domain frame")
    middle_left = _canonical_system(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        source="middle",
        coordinates="middle frame with three coefficients",
    )
    middle_right = _canonical_system(
        [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [2.0, 1.0]],
        source="middle",
        coordinates="middle frame with four coefficients",
    )
    codomain = _canonical_system(
        [[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]],
        source="codomain",
        coordinates="codomain frame",
    )
    first_operator = np.array([[2.0, -1.0], [3.0, 4.0]])
    second_operator = np.array([[1.0, 2.0], [-2.0, 1.0]])
    first = AnalysisSynthesisMap(domain, middle_left, first_operator)
    second = AnalysisSynthesisMap(middle_right, codomain, second_operator)

    transition = AnalysisSynthesisGauge(middle_left).transition_to(middle_right)
    assert transition.shape == (middle_right.coordinate_dimension, middle_left.coordinate_dimension)
    _assert_close(middle_right.synthesis @ transition @ middle_left.analysis, np.eye(2))
    _assert_close(
        AnalysisSynthesisGauge(middle_right).coefficient_projection()
        @ transition
        @ AnalysisSynthesisGauge(middle_left).coefficient_projection(),
        transition,
    )

    composite = first.then(second)
    composite.certificate().require()
    expected_operator = second_operator @ first_operator
    expected_lift = second.coefficient_lift() @ transition @ first.coefficient_lift()
    _assert_close(composite.reconstruct(), expected_operator)
    _assert_close(composite.coefficient_lift(), expected_lift)


def test_transition_identity_is_projection_and_transitions_are_transitive() -> None:
    first = _canonical_system(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        source="middle",
        coordinates="first",
    )
    second = _canonical_system(
        [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [2.0, 1.0]],
        source="middle",
        coordinates="second",
    )
    third = _canonical_system(
        [[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [-1.0, 2.0], [1.0, 1.0]],
        source="middle",
        coordinates="third",
    )

    first_gauge = AnalysisSynthesisGauge(first)
    _assert_close(first_gauge.transition_to(first), first_gauge.coefficient_projection())
    _assert_close(
        AnalysisSynthesisGauge(second).transition_to(third)
        @ first_gauge.transition_to(second),
        first_gauge.transition_to(third),
    )
    incompatible = _canonical_system(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        source="different physical space",
        coordinates="same dimensions are not an identification",
    )
    with pytest.raises(ValidationError, match="source|physical|space|compatible"):
        first_gauge.transition_to(incompatible)


def test_physical_and_dual_adjoints_are_distinct_outside_parseval_systems() -> None:
    domain = _canonical_system(
        [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
        source="domain",
        coordinates="non-Parseval domain",
    )
    codomain = _canonical_system(
        [[2.0, 0.0], [0.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
        source="codomain",
        coordinates="non-Parseval codomain",
    )
    operator = np.array([[1.0, 2.0], [-3.0, 4.0]])
    mapping = AnalysisSynthesisMap(domain, codomain, operator)
    lift = mapping.coefficient_lift()

    physical_adjoint = mapping.adjoint()
    dual_adjoint = mapping.dual_adjoint()
    lift_sharp = physical_adjoint.coefficient_lift()
    domain_metric = AnalysisSynthesisGauge(domain).coefficient_metric()
    codomain_metric = AnalysisSynthesisGauge(codomain).coefficient_metric()

    _assert_close(lift.conj().T @ codomain_metric, domain_metric @ lift_sharp)
    _assert_close(dual_adjoint.coefficient_lift(), lift.conj().T)
    assert np.linalg.norm(lift.conj().T - lift_sharp) > 2.0
    _assert_close(physical_adjoint.reconstruct(), operator.conj().T)
    _assert_close(physical_adjoint.adjoint().reconstruct(), operator)


def test_parseval_endpoints_recover_raw_coefficient_adjoint() -> None:
    scale = 1.0 / np.sqrt(2.0)
    analysis = scale * np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    domain = AnalysisSynthesisSystem(analysis, analysis.conj().T, source="domain")
    codomain = AnalysisSynthesisSystem(analysis, analysis.conj().T, source="codomain")
    operator = np.array([[1.0 + 2.0j, -3.0j], [4.0, 2.0 - 1.0j]])
    mapping = AnalysisSynthesisMap(domain, codomain, operator)
    _assert_close(
        mapping.adjoint().coefficient_lift(),
        mapping.coefficient_lift().conj().T,
    )


def test_coefficient_blocks_assemble_and_multiply_by_the_ordinary_block_law() -> None:
    system = _oblique_system()
    partition = CoefficientPartition(3, ((0, 2), (1,)))
    first = AnalysisSynthesisMap(system, system, np.array([[2.0, -1.0], [3.0, 4.0]]))
    second = AnalysisSynthesisMap(system, system, np.array([[1.0, 2.0], [-2.0, 1.0]]))
    first_blocks = first.blocks(partition, partition)
    second_blocks = second.blocks(partition, partition)

    assert isinstance(first_blocks, AnalysisSynthesisCoefficientBlocks)
    _assert_close(first_blocks.assemble(), first.coefficient_lift())
    _assert_close(second_blocks.assemble(), second.coefficient_lift())
    composite_lift = second.coefficient_lift() @ first.coefficient_lift()

    for target_index, target_group in enumerate(partition.groups):
        for source_index, source_group in enumerate(partition.groups):
            product = np.zeros((len(target_group), len(source_group)))
            for middle_index in range(len(partition.groups)):
                product = product + (
                    second_blocks.blocks[target_index][middle_index]
                    @ first_blocks.blocks[middle_index][source_index]
                )
            expected = composite_lift[np.ix_(target_group, source_group)]
            _assert_close(product, expected)


def test_redundant_physical_components_do_not_obey_one_index_block_multiplication() -> None:
    system = _oblique_system()
    components = system.components()
    assert all(np.linalg.norm(component @ component - component) > 0.15 for component in components)

    first = np.array([[2.0, -1.0], [3.0, 4.0]])
    second = np.array([[1.0, 2.0], [-2.0, 1.0]])
    source_index = 0
    target_index = 1
    naive = sum(
        (components[target_index] @ second @ components[middle])
        @ (components[middle] @ first @ components[source_index])
        for middle in range(len(components))
    )
    expected = components[target_index] @ second @ first @ components[source_index]
    assert np.linalg.norm(naive - expected) > 0.8

    corrected = sum(
        (components[target_index] @ second @ components[left_middle])
        @ (components[right_middle] @ first @ components[source_index])
        for left_middle in range(len(components))
        for right_middle in range(len(components))
    )
    _assert_close(corrected, expected)


def test_canonical_dual_uses_one_rank_and_pseudoinverse_threshold() -> None:
    inconsistent = np.diag([1.0, 9e-16])
    assert np.linalg.matrix_rank(inconsistent) == 2
    _assert_close(np.linalg.pinv(inconsistent) @ inconsistent, np.diag([1.0, 0.0]))
    with pytest.raises(ValidationError, match="rank|singular|left inverse|reconstruction"):
        AnalysisSynthesisSystem.canonical_dual(inconsistent)


def test_partition_validation_rejects_loss_overlap_and_coercion() -> None:
    fine = CoefficientPartition(4, ((0,), (1,), (2, 3)))
    coarse = CoefficientPartition(4, ((0, 1), (2, 3)))
    crossing = CoefficientPartition(4, ((0, 2), (1, 3)))
    assert fine.refines(coarse)
    assert coarse.coarsens(fine)
    assert not coarse.refines(crossing)
    assert not crossing.refines(coarse)

    with pytest.raises(ValidationError):
        CoefficientPartition(3, ((0,), (1,)))
    with pytest.raises(ValidationError):
        CoefficientPartition(3, ((0, 1), (1, 2)))
    with pytest.raises(ValidationError):
        CoefficientPartition(3, ((0,), (), (1, 2)))
    with pytest.raises(ValidationError):
        CoefficientPartition(3, ((0,), (True,), (2,)))
    with pytest.raises(ValidationError):
        CoefficientPartition(3, ((0,), (1.5,), (2,)))


def test_exact_system_map_gauge_composition_and_bridge_are_literal_equalities() -> None:
    analysis = ((1, 0), (0, 1), (1, 1))
    system = ExactAnalysisSynthesisSystem.canonical_dual(
        analysis,
        source="exact physical",
        coordinates="exact frame",
    )
    assert system.synthesis == (
        (Fraction(2, 3), Fraction(-1, 3), Fraction(1, 3)),
        (Fraction(-1, 3), Fraction(2, 3), Fraction(1, 3)),
    )
    projection = system.coefficient_projection()
    assert matrix_multiply(projection, projection) == projection

    coefficient = (3, -4, 5)
    kernel = (-1, -1, 1)
    assert system.gauge_equivalent(coefficient, tuple(a + b for a, b in zip(coefficient, kernel, strict=True)))
    canonical = system.canonical_coefficients(coefficient)
    gauge_part = system.gauge_part(coefficient)
    assert tuple(a + b for a, b in zip(canonical, gauge_part, strict=True)) == tuple(
        Fraction(value) for value in coefficient
    )

    first_operator = ((2, -1), (3, 4))
    second_operator = ((1, 2), (-2, 1))
    first = ExactAnalysisSynthesisMap(system, system, first_operator)
    second = ExactAnalysisSynthesisMap(system, system, second_operator)
    first.certificate().require()
    assert first.reconstruct() == first_operator
    assert matrix_multiply(
        matrix_multiply(projection, first.coefficient_lift()),
        projection,
    ) == first.coefficient_lift()
    composite = first.then(second)
    assert composite.reconstruct() == matrix_multiply(second_operator, first_operator)
    assert composite.coefficient_lift() == matrix_multiply(
        second.coefficient_lift(), first.coefficient_lift()
    )

    other = ExactAnalysisSynthesisSystem.canonical_dual(
        ((1, 0), (0, 1), (1, -1), (2, 1)),
        source="exact physical",
        coordinates="other exact frame",
    )
    transition = system.transition_to(other)
    assert matrix_multiply(
        matrix_multiply(other.synthesis, transition),
        system.analysis,
    ) == identity_matrix(2)


def test_stable_surface_and_committed_semantics_remain_frozen() -> None:
    assert len(stable.__all__) == 131
    assert stable.__all__ == v1.__all__ == api_v1.__all__
    for name in stable.__all__:
        assert getattr(stable, name) is getattr(v1, name)
        assert getattr(stable, name) is getattr(api_v1, name)

    block18_names = {
        "AnalysisSynthesisGauge",
        "AnalysisSynthesisMap",
        "AnalysisSynthesisCoefficientBlocks",
        "CoefficientPartition",
        "ExactAnalysisSynthesisSystem",
        "ExactAnalysisSynthesisMap",
    }
    assert block18_names.isdisjoint(stable.__all__)
    assert str(inspect.signature(AnalysisSynthesisSystem)) == (
        "(analysis: 'np.ndarray', synthesis: 'np.ndarray', source: 'str' = 'source', "
        "coordinates: 'str' = 'coordinates', tolerance: 'float' = 1e-12, "
        "theorem: 'str' = 'KM-AS-01') -> None"
    )
    manifest = create_stable_api_manifest()
    assert len(manifest.symbols) == 131
    assert check_stable_api().compatible
    semantics = run_stable_semantics_fixture()
    semantics.certificate.require()
    assert semantics.passed
    assert len(semantics.results) == 10
