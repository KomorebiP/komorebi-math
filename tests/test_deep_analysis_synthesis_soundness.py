"""Adversarial witnesses for numerical reconstruction certificates."""
from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.coordinates import AnalysisSynthesisBlocks, AnalysisSynthesisSystem
from komorebi_math.exceptions import ValidationError


def test_integer_product_does_not_wrap_into_a_left_inverse() -> None:
    value = np.iinfo(np.int64).max
    system = AnalysisSynthesisSystem(np.array([[value]], dtype=np.int64), np.array([[value]], dtype=np.int64))
    assert int(value) * int(value) != 1
    assert not system.certificate().valid
    assert np.issubdtype(system.analysis.dtype, np.inexact)


def test_nonfinite_block_residual_cannot_pass_with_infinite_scale() -> None:
    system = AnalysisSynthesisSystem(np.eye(1), np.eye(1))
    blocks = AnalysisSynthesisBlocks(system, ((0,),), ((np.array([[-1e308]]),),), np.array([[1e308]]))
    with np.errstate(over="ignore", invalid="ignore"):
        certificate = blocks.certificate()
    assert not certificate.valid
    assert not np.isfinite(certificate.residuals["operator_block_reconstruction"])


def test_large_finite_identity_roundtrip_has_representable_norm() -> None:
    system = AnalysisSynthesisSystem(np.eye(1), np.eye(1))
    system.reconstruction_certificate([1e308]).require()
    system.operator_blocks([[1e308]]).certificate().require()


@pytest.mark.parametrize("groups", [((0,), (0,)), ((0,),), ((0.1,), (1,))])
def test_manually_constructed_block_groups_are_validated(groups) -> None:
    system = AnalysisSynthesisSystem(np.eye(2), np.eye(2))
    size = len(groups)
    blocks = tuple(tuple(np.zeros((2, 2)) for _ in range(size)) for _ in range(size))
    with pytest.raises(ValidationError):
        AnalysisSynthesisBlocks(system, groups, blocks, np.eye(2))


def test_arbitrary_summing_blocks_do_not_prove_the_declared_component_formula() -> None:
    system = AnalysisSynthesisSystem(np.eye(2), np.eye(2))
    zero = np.zeros((2, 2))
    blocks = AnalysisSynthesisBlocks(system, ((0,), (1,)), ((np.eye(2), zero), (zero, zero)), np.eye(2))
    assert np.array_equal(blocks.reconstruct(), blocks.operator)
    assert not blocks.certificate().valid


def test_integer_coordinate_arithmetic_is_numerical_and_does_not_wrap() -> None:
    maximum = np.iinfo(np.int64).max
    system = AnalysisSynthesisSystem(np.array([[2]], dtype=np.int64), np.array([[0.5]]))
    assert system.reconstruct(np.array([maximum], dtype=np.int64))[0] > 1e18
    system.reconstruction_certificate(np.array([maximum], dtype=np.int64)).require()
