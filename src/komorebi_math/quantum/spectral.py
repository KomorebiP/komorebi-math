"""Finite spectral dimension and periodic lattice diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from math import pi
from typing import Any

import numpy as np

from ..exceptions import ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._validation import require_hermitian


def _periodic_shape(shape: Sequence[int]) -> tuple[int, ...]:
    """Validate and return an integer periodic lattice shape."""

    sizes: list[int] = []
    for raw in shape:
        value = int(raw)
        if isinstance(raw, (bool, np.bool_)) or value != raw or value < 2:
            raise ValidationError("every periodic lattice dimension must be an integer at least two")
        sizes.append(value)
    if not sizes:
        raise ValidationError("periodic lattice shape must be nonempty")
    return tuple(sizes)


def _low_mode(mode: Sequence[int], shape: tuple[int, ...]) -> tuple[int, ...]:
    """Validate a low frequency mode in the canonical half Brillouin zone."""

    if len(mode) != len(shape):
        raise ValidationError("mode and lattice shape dimensions do not agree")
    indices: list[int] = []
    for raw, size in zip(mode, shape, strict=True):
        value = int(raw)
        if isinstance(raw, (bool, np.bool_)) or value != raw:
            raise ValidationError("every mode coordinate must be an integer")
        if not 0 <= value <= size // 2:
            raise ValidationError("low mode coordinates must lie between zero and half the lattice size")
        indices.append(value)
    return tuple(indices)


def heat_trace(operator: Any, time: float) -> float:
    """Return ``Tr(exp(-time*operator))`` for a nonnegative Hermitian matrix."""

    value = require_hermitian(operator, name="spectral operator")
    t = float(time)
    if not np.isfinite(t) or t <= 0:
        raise ValidationError("time must be positive and finite")
    eigenvalues = np.linalg.eigvalsh(value)
    if float(np.min(eigenvalues)) < -1e-10:
        raise ValidationError("spectral operator must be nonnegative")
    return float(np.sum(np.exp(-t * np.maximum(eigenvalues, 0.0))))


@documented(
    ProvenanceRecord(
        identifier="PSC-VII-D59",
        title="Finite spectral dimension diagnostic",
        status=ClaimStatus.DEFINITION,
        source="Paper VII",
        assumptions=("finite nonnegative Hermitian operator", "positive diffusion time"),
        limitations=(
            "finite dimension forces the zero-time limit to zero",
            "the large-time limit is zero only when the operator has a zero eigenvalue",
        ),
        verification_suite="quantum.spectral",
    )
)
def spectral_dimension(operator: Any, time: float) -> float:
    """Return ``2*t*Tr(L exp(-tL))/Tr(exp(-tL))``."""

    value = require_hermitian(operator, name="spectral operator")
    t = float(time)
    if not np.isfinite(t) or t <= 0:
        raise ValidationError("time must be positive and finite")
    eigenvalues = np.linalg.eigvalsh(value)
    if float(np.min(eigenvalues)) < -1e-10:
        raise ValidationError("spectral operator must be nonnegative")
    clipped = np.maximum(eigenvalues, 0.0)
    # The common exp(-t*lambda_min) cancels. Keeping it in either sum would
    # produce 0/0 once the heat trace underflows, even for the scalar identity.
    minimum = float(np.min(clipped))
    gaps = clipped - minimum
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp(-t * gaps)
        mean = minimum + float(np.sum(gaps * weights) / np.sum(weights))
        result = float(2 * (t * mean))
    if not np.isfinite(result):
        raise ValidationError("spectral dimension exceeds finite floating-point range")
    return result


def periodic_lattice_eigenvalues(shape: Sequence[int], *, spacing: float | None = None) -> np.ndarray:
    """Return eigenvalues of the nearest neighbor periodic lattice Laplacian."""

    sizes = _periodic_shape(shape)
    h = 1 / max(sizes) if spacing is None else float(spacing)
    if not np.isfinite(h) or h <= 0:
        raise ValidationError("spacing must be positive and finite")
    values = []
    for index in product(*(range(size) for size in sizes)):
        values.append(4 / h**2 * sum(np.sin(pi * index[axis] / sizes[axis]) ** 2 for axis in range(len(sizes))))
    return np.sort(np.asarray(values, dtype=float))


def periodic_laplacian(shape: Sequence[int], *, spacing: float | None = None) -> np.ndarray:
    """Construct the finite nearest neighbor periodic lattice Laplacian."""

    sizes = _periodic_shape(shape)
    h = 1 / max(sizes) if spacing is None else float(spacing)
    if not np.isfinite(h) or h <= 0:
        raise ValidationError("spacing must be positive and finite")
    dimension = int(np.prod(sizes))
    matrix = np.zeros((dimension, dimension), dtype=float)

    def flat(index: tuple[int, ...]) -> int:
        return int(np.ravel_multi_index(index, sizes))

    for index in product(*(range(size) for size in sizes)):
        row = flat(index)
        matrix[row, row] = 2 * len(sizes) / h**2
        for axis, size in enumerate(sizes):
            for step in (-1, 1):
                neighbor = list(index)
                neighbor[axis] = (neighbor[axis] + step) % size
                matrix[row, flat(tuple(neighbor))] -= 1 / h**2
    return matrix


def low_mode_dispersion_bound(shape: Sequence[int], mode: Sequence[int], *, spacing: float | None = None) -> tuple[float, float, float]:
    """Return discrete eigenvalue, continuum eigenvalue, and quartic upper error bound."""

    sizes = _periodic_shape(shape)
    indices = _low_mode(mode, sizes)
    h = 1 / max(sizes) if spacing is None else float(spacing)
    if not np.isfinite(h) or h <= 0:
        raise ValidationError("spacing must be positive and finite")
    discrete = 4 / h**2 * sum(np.sin(pi * indices[axis] / sizes[axis]) ** 2 for axis in range(len(sizes)))
    continuum = sum((2 * pi * indices[axis] / (sizes[axis] * h)) ** 2 for axis in range(len(sizes)))
    bound = sum((2 * pi * indices[axis] / (sizes[axis] * h)) ** 4 * h**2 / 12 for axis in range(len(sizes)))
    return float(discrete), float(continuum), float(bound)


__all__ = [
    "heat_trace",
    "low_mode_dispersion_bound",
    "periodic_laplacian",
    "periodic_lattice_eigenvalues",
    "spectral_dimension",
]
