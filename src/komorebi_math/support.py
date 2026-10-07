"""Machine-readable support policy and runtime environment inspection."""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import ValidationError

SUPPORT_MATRIX_SCHEMA = "komorebi-math.support-matrix"
SUPPORT_MATRIX_SCHEMA_VERSION = 1
DEFAULT_SUPPORT_MATRIX = Path(__file__).with_name("data") / "support_matrix_1_0.json"


def _distribution_version(*names: str) -> str | None:
    for name in names:
        try:
            return version(name)
        except PackageNotFoundError:
            continue
    return None


@dataclass(frozen=True, slots=True)
class ValidatedEnvironment:
    """One environment for which a recorded release suite completed."""

    identifier: str
    evidence: str
    platform: str
    python: str
    numpy: str
    suite: str
    result: str
    optional: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ValidatedEnvironment":
        required = {
            "identifier",
            "evidence",
            "platform",
            "python",
            "numpy",
            "suite",
            "result",
            "optional",
        }
        if set(value) != required:
            raise ValidationError("validated environment fields do not match the support schema")
        optional = value["optional"]
        if not isinstance(optional, Mapping):
            raise ValidationError("validated environment optional data must be an object")
        return cls(
            identifier=str(value["identifier"]),
            evidence=str(value["evidence"]),
            platform=str(value["platform"]),
            python=str(value["python"]),
            numpy=str(value["numpy"]),
            suite=str(value["suite"]),
            result=str(value["result"]),
            optional=dict(optional),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "evidence": self.evidence,
            "platform": self.platform,
            "python": self.python,
            "numpy": self.numpy,
            "suite": self.suite,
            "result": self.result,
            "optional": dict(self.optional),
        }


@dataclass(frozen=True, slots=True)
class SupportMatrix:
    """Declared and executed platform policy for one release line."""

    release_line: str
    python: Mapping[str, Any]
    base_dependencies: Mapping[str, str]
    optional_backends: Mapping[str, Any]
    validated_environments: tuple[ValidatedEnvironment, ...]
    unqualified_platforms: tuple[str, ...]
    schema: str = SUPPORT_MATRIX_SCHEMA
    schema_version: int = SUPPORT_MATRIX_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SupportMatrix":
        required = {
            "schema",
            "schema_version",
            "release_line",
            "python",
            "base_dependencies",
            "optional_backends",
            "validated_environments",
            "unqualified_platforms",
        }
        if set(value) != required:
            raise ValidationError("support matrix fields do not match the schema")
        if value["schema"] != SUPPORT_MATRIX_SCHEMA:
            raise ValidationError("unknown support matrix schema")
        if int(value["schema_version"]) != SUPPORT_MATRIX_SCHEMA_VERSION:
            raise ValidationError("unsupported support matrix schema version")
        environments = value["validated_environments"]
        if not isinstance(environments, list):
            raise ValidationError("validated environments must be a list")
        if any(not isinstance(entry, Mapping) for entry in environments):
            raise ValidationError("validated environments must contain objects")
        for field in ("python", "base_dependencies", "optional_backends"):
            if not isinstance(value[field], Mapping):
                raise ValidationError(f"support matrix {field} must be an object")
        unqualified = value["unqualified_platforms"]
        if not isinstance(unqualified, list):
            raise ValidationError("unqualified platforms must be a list")
        return cls(
            release_line=str(value["release_line"]),
            python=dict(value["python"]),
            base_dependencies={str(k): str(v) for k, v in value["base_dependencies"].items()},
            optional_backends=dict(value["optional_backends"]),
            validated_environments=tuple(
                ValidatedEnvironment.from_dict(entry) for entry in environments
            ),
            unqualified_platforms=tuple(str(item) for item in unqualified),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "release_line": self.release_line,
            "python": dict(self.python),
            "base_dependencies": dict(self.base_dependencies),
            "optional_backends": dict(self.optional_backends),
            "validated_environments": [entry.to_dict() for entry in self.validated_environments],
            "unqualified_platforms": list(self.unqualified_platforms),
        }


def load_support_matrix(path: str | Path | None = None) -> SupportMatrix:
    """Load the committed support policy."""

    source = DEFAULT_SUPPORT_MATRIX if path is None else Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValidationError(f"could not read support matrix {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError("support matrix is not valid JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValidationError("support matrix root must be an object")
    return SupportMatrix.from_dict(raw)


def runtime_environment(*, probe_ad: bool = True) -> dict[str, Any]:
    """Return a JSON-compatible snapshot without importing optional GPU runtimes."""

    result: dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "packages": {
            "scipy": _distribution_version("scipy"),
            "sympy": _distribution_version("sympy"),
            "mpmath": _distribution_version("mpmath"),
            "jax": _distribution_version("jax"),
            "torch": _distribution_version("torch"),
            "cupy": _distribution_version("cupy-cuda12x", "cupy-cuda13x", "cupy"),
        },
    }
    if probe_ad:
        from .nonlinear.autodiff import ad_framework_diagnostics

        result["automatic_differentiation"] = ad_framework_diagnostics()
    return result


__all__ = [
    "DEFAULT_SUPPORT_MATRIX",
    "SUPPORT_MATRIX_SCHEMA",
    "SUPPORT_MATRIX_SCHEMA_VERSION",
    "SupportMatrix",
    "ValidatedEnvironment",
    "load_support_matrix",
    "runtime_environment",
]
