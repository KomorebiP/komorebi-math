"""Coefficient-prime-two group algebra diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...certificates import Certificate, Exactness


@dataclass(frozen=True, slots=True)
class DyadicSignDiagnostic:
    """Residual merging and delayed sign separation for ``C_2``."""

    rational_projectors_integral: bool
    residual_algebra_local: bool
    nilpotent_generator_square_zero: bool
    certificate: Certificate


def dyadic_sign_diagnostic() -> DyadicSignDiagnostic:
    """Return the exact integral and residual sign packet diagnostic."""

    # Over F_2, X = g - 1 satisfies X^2 = 0 in F_2[C_2].
    multiplication_by_x = np.array([[0, 0], [1, 0]], dtype=int)
    square_zero = not np.any((multiplication_by_x @ multiplication_by_x) % 2)
    return DyadicSignDiagnostic(
        rational_projectors_integral=False,
        residual_algebra_local=True,
        nilpotent_generator_square_zero=square_zero,
        certificate=Certificate(
            valid=square_zero,
            exactness=Exactness.EXACT,
            assumptions_checked=("group algebra of C2 over F2",),
            theorem_id="PSC-XII-DYADIC-SIGN",
            message="dyadic sign packets merge residually and are replaced by a nilpotent augmentation coordinate",
        ),
    )


__all__ = ["DyadicSignDiagnostic", "dyadic_sign_diagnostic"]
