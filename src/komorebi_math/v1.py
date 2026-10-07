"""Convenience alias for the curated version 1 API."""

from . import stable as _stable
from .stable import *

__all__ = list(_stable.__all__)
