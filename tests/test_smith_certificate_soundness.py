"""Check the full Smith witness rather than the transformation identity alone."""

from __future__ import annotations

import pytest

from komorebi_math.exceptions import ValidationError
from komorebi_math.refinement import SmithDecomposition, smith_decomposition

sp = pytest.importorskip("sympy")


@pytest.mark.parametrize(
    ("matrix", "diagonal", "left", "right", "factors"),
    [
        ([[1]], [[2]], [[2]], [[1]], (2,)),
        ([[1]], [[2]], [[1]], [[2]], (2,)),
        ([[1, 1], [0, 2]], [[1, 1], [0, 2]], sp.eye(2), sp.eye(2), (1, 2)),
        (sp.diag(2, 3), sp.diag(2, 3), sp.eye(2), sp.eye(2), (2, 3)),
        (sp.diag(0, 2), sp.diag(0, 2), sp.eye(2), sp.eye(2), (2,)),
        ([[sp.Rational(1, 2)]], [[1]], [[2]], [[1]], (1,)),
        ([[2]], [[1]], [[sp.Rational(1, 2)]], [[1]], (1,)),
        ([[2]], [[1]], [[1]], [[sp.Rational(1, 2)]], (1,)),
        ([[2]], [[2]], [[1]], [[1]], (3,)),
        ([[2]], [[2]], [[1]], [[1]], (2.0,)),
        ([[1]], [[1]], [[1]], [[1]], (True,)),
    ],
)
def test_identity_alone_does_not_certify_a_smith_decomposition(
    matrix, diagonal, left, right, factors
) -> None:
    decomposition = SmithDecomposition(
        sp.Matrix(matrix), sp.Matrix(diagonal), sp.Matrix(left), sp.Matrix(right), factors
    )
    assert decomposition.left * decomposition.matrix * decomposition.right == decomposition.diagonal
    certificate = decomposition.certificate()
    assert not certificate.valid
    with pytest.raises(ValidationError):
        certificate.require()


@pytest.mark.parametrize(
    ("diagonal", "left", "right"),
    [
        ([[1, 0]], [[1]], [[1]]),
        ([[1]], sp.eye(2), [[1]]),
        ([[1]], [[1]], sp.eye(2)),
    ],
)
def test_malformed_smith_witness_shapes_return_invalid_certificate(diagonal, left, right) -> None:
    certificate = SmithDecomposition(
        sp.Matrix([[1]]), sp.Matrix(diagonal), sp.Matrix(left), sp.Matrix(right), (1,)
    ).certificate()
    assert not certificate.valid
    assert certificate.residuals["failed_obligations"] > 0


@pytest.mark.parametrize(
    "matrix",
    [
        [[12, 6, 4], [3, 9, 6], [2, 16, 14]],
        [[-2]],
        [[2, 0, 0], [0, 4, 0]],
        sp.zeros(2, 3),
        sp.zeros(0, 2),
        sp.zeros(2, 0),
    ],
)
def test_produced_smith_witnesses_meet_every_obligation(matrix) -> None:
    result = smith_decomposition(matrix)
    certificate = result.certificate().require()
    assert certificate.residuals["failed_obligations"] == 0
    assert all(certificate.diagnostics["checks"].values())
