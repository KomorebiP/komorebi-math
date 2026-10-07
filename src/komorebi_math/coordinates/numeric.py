"""Numerical spectral and orthogonal coordinate transforms."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import Any

import numpy as np

from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError


def _array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise DimensionError(f"{name} must be a matrix")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise DimensionError(f"{name} must be nonempty")
    if not np.all(np.isfinite(array)):
        raise ValidationError(f"{name} must contain finite values")
    return np.array(array, copy=True)


@dataclass(frozen=True, slots=True)
class NumericCoordinateTransform:
    """A finite numerical matrix between coordinate systems."""

    source: str
    target: str
    matrix: np.ndarray
    theorem: str = "CC.NM1"
    description: str = ""
    tolerance: float = 1e-12

    def __post_init__(self) -> None:
        matrix = _array(self.matrix, name="coordinate matrix")
        if not str(self.source).strip() or not str(self.target).strip():
            raise ValidationError("numeric coordinate-system names must be nonempty")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValidationError("numeric coordinate tolerance must be positive and finite")
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "target", str(self.target))
        object.__setattr__(self, "theorem", str(self.theorem))
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(self, "tolerance", float(self.tolerance))

    @property
    def source_dimension(self) -> int:
        """Return the source dimension."""

        return int(self.matrix.shape[1])

    @property
    def target_dimension(self) -> int:
        """Return the target dimension."""

        return int(self.matrix.shape[0])

    @property
    def square(self) -> bool:
        """Return whether the transform is square."""

        return self.source_dimension == self.target_dimension

    def apply(self, coordinates: Any) -> np.ndarray:
        """Apply the numerical coordinate transform."""

        values = np.asarray(coordinates)
        if values.ndim != 1 or values.shape[0] != self.source_dimension:
            raise DimensionError("coordinate vector has the wrong dimension")
        if not np.all(np.isfinite(values)):
            raise ValidationError("coordinate vector must contain finite values")
        return self.matrix @ values

    def inverse(self) -> "NumericCoordinateTransform":
        """Return the numerical inverse of a square nonsingular transform."""

        if not self.square:
            raise DimensionError("only square numeric transforms can be inverted")
        try:
            inverse = np.linalg.inv(self.matrix)
        except np.linalg.LinAlgError as exc:
            raise ValidationError("numeric coordinate transform is singular") from exc
        return NumericCoordinateTransform(
            self.target,
            self.source,
            inverse,
            f"inverse({self.theorem})",
            f"inverse of {self.description or self.source + ' to ' + self.target}",
            self.tolerance,
        )

    def then(self, following: "NumericCoordinateTransform") -> "NumericCoordinateTransform":
        """Compose this transform with a following numerical transform."""

        if not isinstance(following, NumericCoordinateTransform):
            raise ValidationError("following must be a NumericCoordinateTransform")
        if self.target != following.source:
            raise ValidationError("numeric coordinate-system names do not compose")
        if self.target_dimension != following.source_dimension:
            raise DimensionError("numeric coordinate dimensions do not compose")
        return NumericCoordinateTransform(
            self.source,
            following.target,
            following.matrix @ self.matrix,
            f"{self.theorem} then {following.theorem}",
            "composed numerical coordinate transform",
            max(self.tolerance, following.tolerance),
        )

    def condition_number(self) -> float:
        """Return the two-norm condition number."""

        return float(np.linalg.cond(self.matrix))

    def certificate(self) -> Certificate:
        """Return a numerical inverse and conditioning certificate."""

        if not self.square:
            return Certificate(
                valid=True,
                exactness=Exactness.NUMERICAL,
                assumptions_checked=("finite numerical matrix",),
                diagnostics={
                    "source_dimension": self.source_dimension,
                    "target_dimension": self.target_dimension,
                    "square": False,
                },
                theorem_id=self.theorem,
                message="rectangular numerical coordinate transform validated",
            )
        try:
            inverse = np.linalg.inv(self.matrix)
        except np.linalg.LinAlgError:
            return Certificate(
                valid=False,
                exactness=Exactness.NUMERICAL,
                assumptions_checked=("finite numerical matrix",),
                diagnostics={"square": True, "singular": True},
                theorem_id=self.theorem,
                message="numeric coordinate transform is singular",
            )
        identity = np.eye(self.source_dimension, dtype=np.result_type(self.matrix, inverse))
        left_residual = float(np.linalg.norm(inverse @ self.matrix - identity, ord=np.inf))
        right_residual = float(np.linalg.norm(self.matrix @ inverse - identity, ord=np.inf))
        residual = max(left_residual, right_residual)
        return Certificate(
            valid=residual <= self.tolerance * max(1.0, self.condition_number()),
            exactness=Exactness.NUMERICAL,
            residuals={"inverse": residual},
            tolerance=self.tolerance,
            assumptions_checked=("finite matrix", "numerical nonsingularity"),
            diagnostics={
                "source_dimension": self.source_dimension,
                "target_dimension": self.target_dimension,
                "condition_number": self.condition_number(),
            },
            theorem_id=self.theorem,
            message="numerical coordinate transform validated",
        )


def discrete_fourier_transform(
    size: int,
    *,
    unitary: bool = True,
    inverse: bool = False,
    tolerance: float = 1e-12,
) -> NumericCoordinateTransform:
    """Return the finite discrete Fourier coordinate transform."""

    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise DimensionError("Fourier size must be a positive integer")
    sign = 1 if inverse else -1
    rows = np.arange(size).reshape(-1, 1)
    columns = np.arange(size).reshape(1, -1)
    matrix = np.exp(sign * 2j * pi * rows * columns / size)
    if unitary:
        matrix = matrix / sqrt(size)
    elif inverse:
        matrix = matrix / size
    source = f"{'frequency' if inverse else 'sample'} coordinates of length {size}"
    target = f"{'sample' if inverse else 'Fourier'} coordinates of length {size}"
    return NumericCoordinateTransform(
        source,
        target,
        matrix,
        "CC.S1",
        "unitary discrete Fourier transform" if unitary else "discrete Fourier transform",
        tolerance,
    )


def discrete_cosine_transform(
    size: int,
    *,
    orthonormal: bool = True,
    tolerance: float = 1e-12,
) -> NumericCoordinateTransform:
    """Return the type-II discrete cosine transform."""

    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise DimensionError("cosine-transform size must be a positive integer")
    k = np.arange(size).reshape(-1, 1)
    n = np.arange(size).reshape(1, -1)
    matrix = np.cos(pi * (n + 0.5) * k / size)
    if orthonormal:
        matrix[0, :] *= sqrt(1 / size)
        if size > 1:
            matrix[1:, :] *= sqrt(2 / size)
    return NumericCoordinateTransform(
        f"sample coordinates of length {size}",
        f"DCT-II coordinates of length {size}",
        matrix,
        "CC.S2",
        "orthonormal DCT-II" if orthonormal else "unnormalized DCT-II",
        tolerance,
    )


def walsh_hadamard_transform(
    size: int,
    *,
    orthonormal: bool = True,
    tolerance: float = 1e-12,
) -> NumericCoordinateTransform:
    """Return the Sylvester Walsh-Hadamard transform for a power-of-two size."""

    if isinstance(size, bool) or not isinstance(size, int) or size < 1 or size & (size - 1):
        raise DimensionError("Walsh-Hadamard size must be a positive power of two")
    matrix = np.array([[1.0]])
    while matrix.shape[0] < size:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    if orthonormal:
        matrix /= sqrt(size)
    return NumericCoordinateTransform(
        f"sample coordinates of length {size}",
        f"Walsh-Hadamard coordinates of length {size}",
        matrix,
        "CC.S3",
        "orthonormal Walsh-Hadamard transform" if orthonormal else "Walsh-Hadamard transform",
        tolerance,
    )


def haar_wavelet_transform(
    size: int,
    *,
    orthonormal: bool = True,
    tolerance: float = 1e-12,
) -> NumericCoordinateTransform:
    """Return a complete finite Haar wavelet transform."""

    if isinstance(size, bool) or not isinstance(size, int) or size < 1 or size & (size - 1):
        raise DimensionError("Haar size must be a positive power of two")
    matrix = np.eye(size)
    active = size
    transform = np.eye(size)
    while active > 1:
        stage = np.eye(size)
        scale = 1 / sqrt(2) if orthonormal else 1.0
        half = active // 2
        stage[:half, :active] = 0
        stage[half:active, :active] = 0
        for index in range(half):
            stage[index, 2 * index] = scale
            stage[index, 2 * index + 1] = scale
            stage[half + index, 2 * index] = scale
            stage[half + index, 2 * index + 1] = -scale
        transform = stage @ transform
        active = half
    matrix = transform
    return NumericCoordinateTransform(
        f"sample coordinates of length {size}",
        f"Haar wavelet coordinates of length {size}",
        matrix,
        "CC.S4",
        "complete finite Haar wavelet transform",
        tolerance,
    )


@dataclass(frozen=True, slots=True)
class ModalCoordinateSystem:
    """Numerical modal coordinates for a diagonalizable square matrix."""

    eigenvalues: np.ndarray
    synthesis: NumericCoordinateTransform
    analysis: NumericCoordinateTransform
    diagonal_residual: float

    def certificate(self) -> Certificate:
        """Return a numerical modal-decomposition certificate."""

        tolerance = max(self.analysis.tolerance, self.synthesis.tolerance)
        return Certificate(
            valid=self.diagonal_residual <= tolerance * max(1.0, self.synthesis.condition_number()),
            exactness=Exactness.NUMERICAL,
            residuals={"diagonalization": self.diagonal_residual},
            tolerance=tolerance,
            assumptions_checked=("square matrix", "complete numerical eigenvector basis"),
            diagnostics={
                "eigenvalues": tuple(complex(value) for value in self.eigenvalues),
                "condition_number": self.synthesis.condition_number(),
            },
            theorem_id="CC.S5",
            message="numerical modal coordinate system validated",
        )


def modal_coordinate_system(
    operator: Any,
    *,
    state_name: str = "state coordinates",
    modal_name: str = "modal coordinates",
    tolerance: float = 1e-10,
) -> ModalCoordinateSystem:
    """Construct numerical modal coordinates for a diagonalizable operator."""

    matrix = _array(operator, name="operator")
    if matrix.shape[0] != matrix.shape[1]:
        raise DimensionError("modal coordinates require a square operator")
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    if np.linalg.matrix_rank(eigenvectors, tol=tolerance) != matrix.shape[0]:
        raise ValidationError("operator does not have a complete numerical eigenvector basis")
    synthesis = NumericCoordinateTransform(
        modal_name,
        state_name,
        eigenvectors,
        "CC.S5",
        "modal synthesis from eigenvectors",
        tolerance,
    )
    analysis = synthesis.inverse()
    diagonal = np.diag(eigenvalues)
    residual = float(np.linalg.norm(analysis.matrix @ matrix @ synthesis.matrix - diagonal, ord=np.inf))
    return ModalCoordinateSystem(eigenvalues, synthesis, analysis, residual)


__all__ = [
    "ModalCoordinateSystem",
    "NumericCoordinateTransform",
    "discrete_cosine_transform",
    "discrete_fourier_transform",
    "haar_wavelet_transform",
    "modal_coordinate_system",
    "walsh_hadamard_transform",
]
