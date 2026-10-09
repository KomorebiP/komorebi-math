"""Fixed-mode four-spinor Dirac evolution from the Physics Block 44 source.

Units are c = hbar = 1. Fourier labels belong to the normalized unit three-torus,
so physical momentum is 2*pi*n. These numerical helpers do not construct an
unbounded operator, sampled-grid Hilbert space or infinite Fourier packet.
"""
from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np

from .._number_theory import require_integer
from ..exceptions import DimensionError, ValidationError

# Exact coordinate actions of Relativistic3DCliffordTransportU1.lean.
# Current source SHA256: 52288e2f60a23a3315ceaa2e1c13ef52d15a25bb64ca07c29dbedb73a415d403
_ACTIONS = (
    ((0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0)),
    ((0, 0, 0, -1j), (0, 0, 1j, 0), (0, -1j, 0, 0), (1j, 0, 0, 0)),
    ((0, 0, 1, 0), (0, 0, 0, -1), (1, 0, 0, 0), (0, -1, 0, 0)),
    ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, -1, 0), (0, 0, 0, -1)),
)


def _real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValidationError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (OverflowError, ValueError, TypeError) as exc:
        raise ValidationError(f"{name} must be representable as a finite real scalar") from exc
    if not np.isfinite(result):
        raise ValidationError(f"{name} must be a finite real scalar")
    return result


def _momentum(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=object)
    if vector.shape != (3,):
        raise DimensionError("physical momentum must have shape (3,)")
    return np.asarray([_real(entry, "momentum coordinate") for entry in vector], dtype=float)


def dirac_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return independent read-only snapshots of alpha_x, alpha_y, alpha_z, beta."""

    arrays = tuple(np.array(action, dtype=complex) for action in _ACTIONS)
    for array in arrays:
        array.setflags(write=False)
    return arrays


def central_mode_momentum(mode: Any, mesh: float = 0.0) -> np.ndarray:
    """Return continuum or central-difference momentum for a unit-torus Z^3 mode.

    The central symbol is p*sinc(mesh*p/pi), with value p at zero mesh.
    It is a continuum-torus multiplier; it does not impose a sampled Nyquist grid.
    """

    labels = np.asarray(mode, dtype=object)
    if labels.shape != (3,):
        raise DimensionError("mode must have shape (3,)")
    integers = tuple(require_integer(entry, name="mode coordinate") for entry in labels)
    spacing = _real(mesh, "mesh")
    try:
        momentum = 2 * np.pi * np.asarray(integers, dtype=float)
    except (OverflowError, ValueError) as exc:
        raise ValidationError("mode momentum exceeds finite floating-point range") from exc
    if not np.all(np.isfinite(momentum)):
        raise ValidationError("mode momentum exceeds finite floating-point range")
    if spacing == 0:
        return momentum
    with np.errstate(over="ignore", invalid="ignore"):
        phase = spacing * momentum
    if not np.all(np.isfinite(phase)):
        raise ValidationError("central-difference phase exceeds finite floating-point range")
    # sinc divides and then multiplies its argument by pi. At huge phases
    # that round trip can change the represented phase by many periods.
    # Retain its removable-value branch only where the argument is small.
    result = np.empty_like(momentum)
    small = np.abs(phase) <= 1e-4
    result[small] = momentum[small] * np.sinc(phase[small] / np.pi)
    result[~small] = np.sin(phase[~small]) / spacing
    if not np.all(np.isfinite(result)):
        raise ValidationError("central-difference momentum exceeds finite floating-point range")
    return result


def dirac_symbol(momentum: Any, mass: float = 1.0) -> np.ndarray:
    """Return the finite Hermitian matrix sum q_j*alpha_j + mass*beta."""

    vector = _momentum(momentum)
    coefficient = _real(mass, "mass")
    matrices = dirac_matrices()
    return sum((value * matrix for value, matrix in zip(vector, matrices[:3], strict=True)), coefficient * matrices[3])


def dirac_propagator(momentum: Any, time: float, mass: float = 1.0) -> np.ndarray:
    """Return exp(-i*time*H) using the finite Clifford scalar-square identity.

    For energy E>0 this is cos(tE)I - i*sin(tE)*H/E. Normalizing the real
    coefficients before constructing H/E avoids division of complex entries by
    a tiny E. At E=0 the result is the identity, including nonzero time.
    """

    vector = _momentum(momentum)
    coefficient = _real(mass, "mass")
    duration = _real(time, "time")
    energy = float(np.hypot.reduce(np.r_[coefficient, vector]))
    if not np.isfinite(energy):
        raise ValidationError("Dirac energy exceeds finite floating-point range")
    identity = np.eye(4, dtype=complex)
    if energy == 0:
        return identity
    phase = duration * energy
    if not np.isfinite(phase):
        raise ValidationError("Dirac evolution phase exceeds finite floating-point range")
    normalized = dirac_symbol(vector / energy, coefficient / energy)
    return np.cos(phase) * identity - 1j * np.sin(phase) * normalized


__all__ = ["dirac_matrices", "central_mode_momentum", "dirac_symbol", "dirac_propagator"]
