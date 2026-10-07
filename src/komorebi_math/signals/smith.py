"""Cyclic Cayley coordinates used as a secondary signal realization."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..exceptions import ValidationError

ArrayLike = np.ndarray | Sequence[complex] | complex


def gamma_from_impedance(impedance: ArrayLike) -> np.ndarray:
    """Map normalized impedance values to reflection coefficients by the Cayley transform."""

    value = np.asarray(impedance, dtype=complex)
    if np.any(np.isclose(value + 1.0, 0.0)):
        raise ZeroDivisionError("impedance minus one is divided by impedance plus one")
    return (value - 1.0) / (value + 1.0)


def impedance_from_gamma(gamma: ArrayLike) -> np.ndarray:
    """Invert the normalized impedance Cayley transform."""

    value = np.asarray(gamma, dtype=complex)
    if np.any(np.isclose(1.0 - value, 0.0)):
        raise ZeroDivisionError("Gamma equal to one is the open circuit pole")
    return (1.0 + value) / (1.0 - value)


def phase_mobius(impedance: ArrayLike, order: int, phase: int) -> np.ndarray:
    """Apply a cyclic disk rotation pulled back to normalized impedance coordinates."""

    if isinstance(order, bool) or int(order) != order or order < 1:
        raise ValidationError("order must be a positive integer")
    root = np.exp(2j * np.pi * (int(phase) % int(order)) / int(order))
    value = np.asarray(impedance, dtype=complex)
    numerator = (1.0 + root) * value + (1.0 - root)
    denominator = (1.0 - root) * value + (1.0 + root)
    if np.any(np.isclose(denominator, 0.0)):
        raise ZeroDivisionError("the phase Mobius map has a pole at the supplied point")
    return numerator / denominator


def smith_sensitivity(gamma: ArrayLike, *, regularization: float = 0.0) -> np.ndarray:
    """Return a regularized pointwise sensitivity for the impedance Cayley chart."""

    if regularization < 0:
        raise ValidationError("regularization must be nonnegative")
    value = np.asarray(gamma, dtype=complex)
    return 1.0 / (np.abs(1.0 - value) ** 2 + regularization) + 1.0 / (np.abs(1.0 + value) ** 2 + regularization)


__all__ = ["gamma_from_impedance", "impedance_from_gamma", "phase_mobius", "smith_sensitivity"]
