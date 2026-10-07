"""Shared utilities for declared but unimplemented feature modules."""

from __future__ import annotations

from .exceptions import FeatureNotImplementedError
from .registry import FeatureSpec, FeatureStatus, feature


def development_status(key: str) -> FeatureSpec:
    return feature(key)


def require_implemented(key: str) -> None:
    spec = feature(key)
    if spec.status in {FeatureStatus.IMPLEMENTED, FeatureStatus.PARTIAL, FeatureStatus.EXPERIMENTAL}:
        return
    raise FeatureNotImplementedError(
        f"{spec.title} is declared as {spec.status.value} in {spec.module}. "
        "The namespace is reserved, but this block has not yet been migrated into the production wheel."
    )
