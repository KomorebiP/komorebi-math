from __future__ import annotations

import numpy as np
import pytest

import komorebi_math
from komorebi_math import stable
from komorebi_math.coordinates import AnalysisSynthesisBlocks, AnalysisSynthesisSystem
from komorebi_math.exceptions import DimensionError, ValidationError
from komorebi_math.export.persistence import dumps_portable, loads_portable, portable_codecs


def test_block16_version() -> None:
    assert komorebi_math.__version__ == "1.0.0b4"


def test_square_analysis_synthesis_roundtrip_and_projectors() -> None:
    matrix = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    system = AnalysisSynthesisSystem.from_invertible(matrix, source="samples", coordinates="Hadamard")
    system.certificate().require()
    value = np.array([3.0, -2.0])
    assert np.allclose(system.analyze(value), matrix @ value)
    assert np.allclose(system.reconstruct(value), value)
    assert np.allclose(sum(system.components()), np.eye(2))
    assert system.square
    assert not system.redundant


def test_redundant_canonical_dual_is_a_left_inverse() -> None:
    analysis = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
    system = AnalysisSynthesisSystem.canonical_dual(analysis)
    assert system.redundant
    system.certificate().require()
    assert np.allclose(system.frame_operator(), np.eye(2))
    system.reconstruction_certificate(np.array([2.0, 5.0])).require()


def test_rank_deficient_canonical_dual_is_rejected() -> None:
    with pytest.raises(ValidationError, match="full column rank"):
        AnalysisSynthesisSystem.canonical_dual(np.array([[1.0, 2.0], [2.0, 4.0]]))


def test_invalid_synthesis_shape_is_rejected() -> None:
    with pytest.raises(DimensionError, match="synthesis shape"):
        AnalysisSynthesisSystem(np.eye(2), np.ones((3, 2)))


def test_operator_blocks_reconstruct_for_grouped_coordinates() -> None:
    analysis = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    system = AnalysisSynthesisSystem.canonical_dual(analysis)
    operator = np.array([[2.0, -1.0], [3.0, 4.0]])
    blocks = system.operator_blocks(operator, groups=((0, 2), (1,)))
    assert isinstance(blocks, AnalysisSynthesisBlocks)
    blocks.certificate().require()
    assert np.allclose(blocks.reconstruct(), operator)


def test_coordinate_groups_must_form_a_partition() -> None:
    system = AnalysisSynthesisSystem.from_invertible(np.eye(3))
    with pytest.raises(ValidationError, match="partition"):
        system.components(((0,), (1,)))
    with pytest.raises(ValidationError, match="disjoint"):
        system.components(((0, 1), (1, 2)))


def test_analysis_synthesis_portable_roundtrip() -> None:
    system = AnalysisSynthesisSystem.canonical_dual(
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    )
    restored = loads_portable(dumps_portable(system))
    assert isinstance(restored, AnalysisSynthesisSystem)
    restored.certificate().require()
    assert np.array_equal(restored.analysis, system.analysis)
    assert np.allclose(restored.synthesis, system.synthesis)
    assert "coordinates.analysis_synthesis" in {codec.tag for codec in portable_codecs()}


def test_analysis_synthesis_is_on_stable_versioned_surface() -> None:
    import komorebi_math.api.v1 as api_v1
    import komorebi_math.v1 as v1

    assert stable.AnalysisSynthesisSystem is AnalysisSynthesisSystem
    assert v1.AnalysisSynthesisSystem is AnalysisSynthesisSystem
    assert api_v1.AnalysisSynthesisSystem is AnalysisSynthesisSystem
    assert stable.OperatorGrammar is not None
    assert stable.CoordinateTransform is not None
