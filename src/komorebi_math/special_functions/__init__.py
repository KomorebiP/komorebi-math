"""Audited machinery from the Special Functions I–VIII consolidation.

Exact rational scalar functions and numerical shape transforms are separate.
Importing this namespace does not load optional mpmath, SciPy, or SymPy.
"""

from .bernoulli import bernoulli_hurwitz, bernoulli_hurwitz_packet, bernoulli_packet_integer
from .shape import (
    beta_binomial_refinement,
    beta_logit_density,
    gamma_sech_ambiguity,
    log_gamma_fourier,
    log_gamma_window,
    meixner_characteristic,
    meixner_density,
    meixner_pollaczek,
    shape_affinity,
    shape_refinement_coefficient,
    shape_transfer_singular_value,
)
from .terminating import q_pochhammer, q_racah_lattice, q_racah_scalar, racah_scalar

__all__ = [
    "bernoulli_hurwitz", "bernoulli_hurwitz_packet", "bernoulli_packet_integer",
    "beta_binomial_refinement", "beta_logit_density", "gamma_sech_ambiguity",
    "log_gamma_fourier", "log_gamma_window", "meixner_characteristic", "meixner_density",
    "meixner_pollaczek", "q_pochhammer", "q_racah_lattice", "q_racah_scalar", "racah_scalar",
    "shape_affinity", "shape_refinement_coefficient", "shape_transfer_singular_value",
]
