"""Package specific exception hierarchy."""

from __future__ import annotations


class KomorebiMathError(Exception):
    """Base class for all public package exceptions."""


class ValidationError(KomorebiMathError, ValueError):
    """Raised when a mathematical object fails a declared validation rule."""


class DimensionError(ValidationError):
    """Raised when matrix, vector, or packet dimensions do not agree."""


class RelationError(ValidationError):
    """Raised when a grammar relation is malformed or violated."""


class BackendUnavailableError(KomorebiMathError):
    """Raised when an optional computation backend is not installed."""


class ExactnessError(KomorebiMathError):
    """Raised when an exact operation is requested from an inexact backend."""


class AssumptionError(KomorebiMathError):
    """Raised when a required mathematical assumption is not satisfied."""


class ResourceLimitError(ValidationError):
    """Raised before a computation exceeds an explicit resource contract."""


class FeatureNotImplementedError(NotImplementedError, KomorebiMathError):
    """Raised by a deliberately declared placeholder feature."""
