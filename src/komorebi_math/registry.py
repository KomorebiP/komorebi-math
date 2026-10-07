"""Feature registry for implemented and planned mathematical domains."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Iterable


class FeatureStatus(StrEnum):
    IMPLEMENTED = "implemented"
    PARTIAL = "partial"
    EXPERIMENTAL = "experimental"
    PLANNED = "planned"
    EXTERNAL = "external theorem required"


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    key: str
    module: str
    title: str
    status: FeatureStatus
    summary: str
    papers: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    public_objects: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


class FeatureRegistry:
    def __init__(self, features: Iterable[FeatureSpec] = ()) -> None:
        self._features: dict[str, FeatureSpec] = {}
        for feature in features:
            self.register(feature)

    def register(self, feature: FeatureSpec, *, replace: bool = False) -> None:
        if feature.key in self._features and not replace:
            raise ValueError(f"feature already registered: {feature.key}")
        self._features[feature.key] = feature

    def get(self, key: str) -> FeatureSpec:
        return self._features[key]

    def all(self) -> tuple[FeatureSpec, ...]:
        return tuple(self._features[key] for key in sorted(self._features))

    def by_status(self, status: FeatureStatus) -> tuple[FeatureSpec, ...]:
        return tuple(feature for feature in self.all() if feature.status is status)


FEATURES = FeatureRegistry()


def register_feature(feature: FeatureSpec, *, replace: bool = False) -> None:
    FEATURES.register(feature, replace=replace)


def feature(key: str) -> FeatureSpec:
    """Return one registered feature specification by key."""

    return FEATURES.get(key)


def features() -> tuple[FeatureSpec, ...]:
    """Return every registered feature specification in key order."""

    return FEATURES.all()
