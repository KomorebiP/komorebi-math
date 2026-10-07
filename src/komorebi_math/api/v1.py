"""Version 1 compatibility alias for :mod:`komorebi_math.stable`."""

from .. import stable as _stable
from ..stable import *

__all__ = list(_stable.__all__)
