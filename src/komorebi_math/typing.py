"""Shared type aliases used by the public API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

Scalar: TypeAlias = int | float | complex
NumericArray: TypeAlias = npt.NDArray[np.generic]
MatrixInput: TypeAlias = Any
VectorInput: TypeAlias = Any
JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | Sequence["JSONValue"] | Mapping[str, "JSONValue"]
