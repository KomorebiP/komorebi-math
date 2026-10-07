"""Lazy exact sequences, automatic models, holonomic recurrences, and compilation."""

from .._planned import development_status
from ..registry import FeatureSpec
from .automatic import AutomaticSequence, base_digits, thue_morse_automaton
from .base import Sequence
from .catalog import (
    apery_sequence,
    bell_sequence,
    bernoulli_sequence,
    catalan_sequence,
    euler_zigzag_sequence,
    fibonacci_sequence,
    get_sequence,
    lucas_sequence,
    motzkin_sequence,
    partition_sequence,
    sequence_catalog,
    signed_euler_sequence,
    thue_morse_sequence,
)
from .compiler import CompiledSequence, compile_sequence, detect_linear_recurrence, recurrence_generating_function
from .holonomic import (
    HolonomicRecurrence,
    Polynomial,
    apery_recurrence,
    evaluate_polynomial,
    motzkin_recurrence,
)
from .transforms import (
    binomial_transform,
    binomial_transform_values,
    cauchy_product,
    finite_difference,
    inverse_binomial_transform,
    inverse_binomial_transform_values,
    partial_sums,
)

FEATURE_KEY = "sequences"


def status() -> FeatureSpec:
    """Return the registered implementation status for this domain."""

    return development_status(FEATURE_KEY)


def require() -> None:
    """Confirm that this package block is implemented."""

    return None


__all__ = [
    "FEATURE_KEY",
    "AutomaticSequence",
    "CompiledSequence",
    "HolonomicRecurrence",
    "Polynomial",
    "Sequence",
    "apery_recurrence",
    "apery_sequence",
    "base_digits",
    "bell_sequence",
    "bernoulli_sequence",
    "binomial_transform",
    "binomial_transform_values",
    "catalan_sequence",
    "cauchy_product",
    "compile_sequence",
    "detect_linear_recurrence",
    "euler_zigzag_sequence",
    "evaluate_polynomial",
    "fibonacci_sequence",
    "finite_difference",
    "get_sequence",
    "inverse_binomial_transform",
    "inverse_binomial_transform_values",
    "lucas_sequence",
    "motzkin_recurrence",
    "motzkin_sequence",
    "partial_sums",
    "partition_sequence",
    "recurrence_generating_function",
    "require",
    "sequence_catalog",
    "signed_euler_sequence",
    "status",
    "thue_morse_automaton",
    "thue_morse_sequence",
]
