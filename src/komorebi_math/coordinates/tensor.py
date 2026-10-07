"""Transport of operators, forms, and tensor components between bases."""

from __future__ import annotations

from fractions import Fraction
from typing import Iterable, Literal, Sequence

from ..exceptions import DimensionError, ValidationError
from .exact import (
    Matrix,
    Scalar,
    Vector,
    as_matrix,
    as_vector,
    inverse_matrix,
    kronecker,
    matrix_multiply,
    matrix_shape,
    matrix_vector_multiply,
    transpose,
    zero_matrix,
)
from .transform import CoordinateTransform

Variance = Literal["contravariant", "covariant"]


def contragredient_transform(transform: CoordinateTransform) -> CoordinateTransform:
    """Return the induced transform on covector coordinates."""

    if not transform.square:
        raise DimensionError("covector transport requires a square coordinate change")
    matrix = transpose(inverse_matrix(transform.matrix))
    return CoordinateTransform(
        f"covectors in {transform.source}",
        f"covectors in {transform.target}",
        matrix,
        "CC.T1",
        "contragredient coordinate transform",
    )


def transport_covector(
    covector: Sequence[Scalar],
    transform: CoordinateTransform,
) -> Vector:
    """Transport covector components to the target coordinates."""

    return contragredient_transform(transform).apply(covector)


def transport_operator(
    operator: Iterable[Iterable[Scalar]],
    transform: CoordinateTransform,
) -> Matrix:
    r"""Return ``C A C^(-1)`` in target coordinates."""

    matrix = as_matrix(operator)
    rows, columns = matrix_shape(matrix)
    if not transform.square or rows != columns or rows != transform.source_dimension:
        raise DimensionError("operator transport needs matching square dimensions")
    return matrix_multiply(
        matrix_multiply(transform.matrix, matrix),
        inverse_matrix(transform.matrix),
    )


def transport_rectangular_operator(
    operator: Iterable[Iterable[Scalar]],
    domain_transform: CoordinateTransform,
    codomain_transform: CoordinateTransform,
) -> Matrix:
    r"""Transport a linear map by ``C_codomain A C_domain^(-1)``."""

    matrix = as_matrix(operator)
    rows, columns = matrix_shape(matrix)
    if not domain_transform.square or not codomain_transform.square:
        raise DimensionError("domain and codomain transforms must be square")
    if columns != domain_transform.source_dimension:
        raise DimensionError("operator domain does not match its coordinate transform")
    if rows != codomain_transform.source_dimension:
        raise DimensionError("operator codomain does not match its coordinate transform")
    return matrix_multiply(
        matrix_multiply(codomain_transform.matrix, matrix),
        inverse_matrix(domain_transform.matrix),
    )


def transport_bilinear_form(
    form: Iterable[Iterable[Scalar]],
    transform: CoordinateTransform,
) -> Matrix:
    r"""Return the target matrix ``C^(-T) G C^(-1)`` of a bilinear form."""

    matrix = as_matrix(form)
    rows, columns = matrix_shape(matrix)
    if not transform.square or rows != columns or rows != transform.source_dimension:
        raise DimensionError("bilinear-form transport needs matching square dimensions")
    inverse = inverse_matrix(transform.matrix)
    return matrix_multiply(matrix_multiply(transpose(inverse), matrix), inverse)


def tensor_coordinate_transform(
    transform: CoordinateTransform,
    variance: Sequence[Variance],
) -> CoordinateTransform:
    """Return the induced transform on flattened tensor components.

    Tensor axes are flattened in lexicographic order.  Contravariant axes use
    the original coordinate matrix.  Covariant axes use its contragredient.
    """

    if not transform.square:
        raise DimensionError("tensor transport requires a square coordinate transform")
    kinds = tuple(variance)
    if any(kind not in ("contravariant", "covariant") for kind in kinds):
        raise ValidationError("tensor variance entries must be contravariant or covariant")
    if not kinds:
        matrix: Matrix = ((Fraction(1),),)
    else:
        covariant = transpose(inverse_matrix(transform.matrix))
        matrix = ((Fraction(1),),)
        for kind in kinds:
            factor = transform.matrix if kind == "contravariant" else covariant
            matrix = kronecker(matrix, factor)
    rank = len(kinds)
    source = f"rank-{rank} tensor components in {transform.source}"
    target = f"rank-{rank} tensor components in {transform.target}"
    return CoordinateTransform(
        source,
        target,
        matrix,
        "CC.T2",
        f"tensor component transport with variance {kinds}",
    )


def transport_tensor_components(
    components: Sequence[Scalar],
    transform: CoordinateTransform,
    variance: Sequence[Variance],
) -> Vector:
    """Transport flattened tensor components between coordinates."""

    return tensor_coordinate_transform(transform, variance).apply(components)


def tensor_product_transform(
    left: CoordinateTransform,
    right: CoordinateTransform,
) -> CoordinateTransform:
    """Return the Kronecker product of two coordinate transforms."""

    return CoordinateTransform(
        f"{left.source} tensor {right.source}",
        f"{left.target} tensor {right.target}",
        kronecker(left.matrix, right.matrix),
        "CC.T3",
        "tensor-product coordinate transform",
    )


def direct_sum_transform(
    left: CoordinateTransform,
    right: CoordinateTransform,
) -> CoordinateTransform:
    """Return the block-diagonal direct sum of two coordinate transforms."""

    left_rows, left_columns = matrix_shape(left.matrix)
    right_rows, right_columns = matrix_shape(right.matrix)
    output = [
        [Fraction(0) for _ in range(left_columns + right_columns)]
        for _ in range(left_rows + right_rows)
    ]
    for row in range(left_rows):
        for column in range(left_columns):
            output[row][column] = left.matrix[row][column]
    for row in range(right_rows):
        for column in range(right_columns):
            output[left_rows + row][left_columns + column] = right.matrix[row][column]
    return CoordinateTransform(
        f"{left.source} direct sum {right.source}",
        f"{left.target} direct sum {right.target}",
        tuple(tuple(row) for row in output),
        "CC.T4",
        "direct-sum coordinate transform",
    )


def coordinate_pairing(
    covector: Sequence[Scalar],
    vector: Sequence[Scalar],
) -> Fraction:
    """Return the exact natural pairing of coordinate components."""

    left = as_vector(covector)
    right = as_vector(vector)
    if len(left) != len(right):
        raise DimensionError("pairing dimensions do not agree")
    return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))


def pairing_invariant(
    covector: Sequence[Scalar],
    vector: Sequence[Scalar],
    transform: CoordinateTransform,
) -> bool:
    """Return whether vector and covector transport preserves their pairing."""

    source_pairing = coordinate_pairing(covector, vector)
    target_vector = transform.apply(vector)
    target_covector = transport_covector(covector, transform)
    return coordinate_pairing(target_covector, target_vector) == source_pairing


__all__ = [
    "Variance",
    "contragredient_transform",
    "coordinate_pairing",
    "direct_sum_transform",
    "pairing_invariant",
    "tensor_coordinate_transform",
    "tensor_product_transform",
    "transport_bilinear_form",
    "transport_covector",
    "transport_operator",
    "transport_rectangular_operator",
    "transport_tensor_components",
]
