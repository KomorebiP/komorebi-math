"""Deterministic public-API manifests and additive compatibility checks.

The compatibility layer deliberately distinguishes structural API stability from
mathematical semantics.  It can certify that exported names remain present and
retain their broad callable/data kind.  It cannot prove that two implementations
have identical mathematical behavior, so that limitation is reported explicitly
in every compatibility certificate.
"""

from __future__ import annotations

import inspect
import json
import types
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, EnumType
from functools import wraps
from hashlib import sha256
from importlib import import_module
from pathlib import Path, PurePath
from typing import Any, ParamSpec, TypeVar

import numpy as np

from ._version import __version__
from .certificates import Certificate, Exactness
from .exceptions import ValidationError

P = ParamSpec("P")
R = TypeVar("R")

PUBLIC_API_SCHEMA = "komorebi-math.public-api"
PUBLIC_API_SCHEMA_VERSION = 1
DEFAULT_PUBLIC_NAMESPACES = (
    "komorebi_math",
    "komorebi_math.nonlinear",
    "komorebi_math.export",
    "komorebi_math.signals",
    "komorebi_math.generating",
    "komorebi_math.coordinates",
)
DEFAULT_PUBLIC_API_MANIFEST = Path(__file__).with_name("data") / "public_api_1_0.json"
API_STABILITY_LEVEL = "pre-beta-additive"
BETA_PUBLIC_NAMESPACES = ("komorebi_math.beta",)
DEFAULT_BETA_API_MANIFEST = Path(__file__).with_name("data") / "public_api_beta_1_0.json"
BETA_API_STABILITY_LEVEL = "1.0-beta-additive-strict-signatures"
STABLE_PUBLIC_NAMESPACES = ("komorebi_math.stable",)
DEFAULT_STABLE_API_MANIFEST = Path(__file__).with_name("data") / "public_api_stable_1_0.json"
STABLE_API_STABILITY_LEVEL = "1.0-frozen-strict-signatures-semantics-and-persistence"


@dataclass(frozen=True, slots=True, order=True)
class PublicApiSymbol:
    """One exported public symbol in a deterministic API manifest."""

    namespace: str
    name: str
    kind: str
    implementation_module: str
    qualname: str
    signature: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible manifest data."""

        return {
            "namespace": self.namespace,
            "name": self.name,
            "kind": self.kind,
            "implementation_module": self.implementation_module,
            "qualname": self.qualname,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PublicApiSymbol":
        """Reconstruct one symbol from validated manifest data."""

        required = {
            "namespace",
            "name",
            "kind",
            "implementation_module",
            "qualname",
            "signature",
        }
        if set(value) != required:
            raise ValidationError("public API symbol fields do not match the schema")
        signature = value["signature"]
        if signature is not None and not isinstance(signature, str):
            raise ValidationError("public API signatures must be strings or null")
        return cls(
            namespace=str(value["namespace"]),
            name=str(value["name"]),
            kind=str(value["kind"]),
            implementation_module=str(value["implementation_module"]),
            qualname=str(value["qualname"]),
            signature=signature,
        )


@dataclass(frozen=True, slots=True)
class PublicApiManifest:
    """A deterministic snapshot of selected ``__all__`` namespaces."""

    package_version: str
    namespaces: tuple[str, ...]
    symbols: tuple[PublicApiSymbol, ...]
    schema: str = PUBLIC_API_SCHEMA
    schema_version: int = PUBLIC_API_SCHEMA_VERSION
    stability_level: str = API_STABILITY_LEVEL

    def __post_init__(self) -> None:
        if self.schema != PUBLIC_API_SCHEMA:
            raise ValidationError("unknown public API manifest schema")
        if int(self.schema_version) != PUBLIC_API_SCHEMA_VERSION:
            raise ValidationError("unsupported public API manifest schema version")
        namespaces = tuple(str(value) for value in self.namespaces)
        if not namespaces or len(set(namespaces)) != len(namespaces):
            raise ValidationError("public API manifest namespaces must be nonempty and unique")
        symbols = tuple(self.symbols)
        keys = tuple((symbol.namespace, symbol.name) for symbol in symbols)
        if len(set(keys)) != len(keys):
            raise ValidationError("public API manifest contains duplicate symbols")
        if tuple(sorted(symbols)) != symbols:
            raise ValidationError("public API manifest symbols must use deterministic ordering")
        if any(symbol.namespace not in namespaces for symbol in symbols):
            raise ValidationError("public API symbol lies outside the declared namespaces")
        object.__setattr__(self, "namespaces", namespaces)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "schema_version", PUBLIC_API_SCHEMA_VERSION)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical manifest mapping."""

        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "stability_level": self.stability_level,
            "package_version": self.package_version,
            "namespaces": list(self.namespaces),
            "symbols": [symbol.to_dict() for symbol in self.symbols],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PublicApiManifest":
        """Validate and reconstruct a public API manifest."""

        required = {
            "schema",
            "schema_version",
            "stability_level",
            "package_version",
            "namespaces",
            "symbols",
        }
        if set(value) != required:
            raise ValidationError("public API manifest fields do not match the schema")
        raw_namespaces = value["namespaces"]
        raw_symbols = value["symbols"]
        if not isinstance(raw_namespaces, list) or not isinstance(raw_symbols, list):
            raise ValidationError("public API manifest namespaces and symbols must be arrays")
        symbols = tuple(PublicApiSymbol.from_dict(item) for item in raw_symbols)
        return cls(
            package_version=str(value["package_version"]),
            namespaces=tuple(str(item) for item in raw_namespaces),
            symbols=symbols,
            schema=str(value["schema"]),
            schema_version=int(value["schema_version"]),
            stability_level=str(value["stability_level"]),
        )

    @property
    def digest(self) -> str:
        """Return the canonical SHA-256 digest of the manifest."""

        return sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def symbol_map(self) -> dict[tuple[str, str], PublicApiSymbol]:
        """Return symbols keyed by namespace and exported name."""

        return {(symbol.namespace, symbol.name): symbol for symbol in self.symbols}


@dataclass(frozen=True, slots=True)
class ApiCompatibilityReport:
    """Structural differences between a reference and current API manifest."""

    reference_digest: str
    current_digest: str
    added: tuple[tuple[str, str], ...]
    removed: tuple[tuple[str, str], ...]
    kind_changed: tuple[tuple[str, str, str, str], ...]
    signature_changed: tuple[tuple[str, str, str | None, str | None], ...]
    strict_signatures: bool
    certificate: Certificate

    @property
    def compatible(self) -> bool:
        """Return whether the selected compatibility policy passed."""

        return self.certificate.valid

    def to_dict(self) -> dict[str, Any]:
        """Return report data suitable for CLI and export use."""

        return {
            "compatible": self.compatible,
            "reference_digest": self.reference_digest,
            "current_digest": self.current_digest,
            "added": self.added,
            "removed": self.removed,
            "kind_changed": self.kind_changed,
            "signature_changed": self.signature_changed,
            "strict_signatures": self.strict_signatures,
            "certificate": self.certificate.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class DeprecationRecord:
    """Machine-readable metadata attached by :func:`deprecated`."""

    since: str
    replacement: str | None
    remove_in: str | None
    reason: str | None


class KomorebiDeprecationWarning(DeprecationWarning):
    """Warning category for explicit Komorebi Math API deprecations."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _public_kind(value: Any) -> str:
    if inspect.isclass(value):
        return "class"
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isbuiltin(value):
        return "function"
    if inspect.ismodule(value):
        return "module"
    return "data"


@dataclass(frozen=True, slots=True)
class _SignatureDefault:
    text: str

    def __repr__(self) -> str:
        return self.text


def _public_signature(value: Any) -> str | None:
    if not (callable(value) or inspect.isclass(value)):
        return None
    # EnumType's inherited constructor has the same member-lookup behavior
    # across the supported interpreters, but inspect.signature reports its
    # functional enum-creation parameters on Python 3.11 and ``(*values)`` on
    # 3.12+. Keep the committed form for that stdlib constructor only; an
    # explicit signature or any custom metaclass still needs reflection.
    if (
        type(value) is EnumType
        and not any(
            "__signature__" in base.__dict__
            for base in value.__mro__
            if base is not Enum
        )
    ):
        return "(*values)"
    # ``types.GenericAlias`` objects such as ``tuple[int, ...]`` are exported
    # intentionally as lightweight public coordinate aliases.  Their runtime
    # constructor signature is supplied by the interpreter rather than by the
    # project and changed between supported Python releases.  Python 3.11 and
    # 3.12 commonly report ``(*args, **kwargs)`` while Python 3.13 reports the
    # underlying tuple constructor as ``(iterable=(), /)``.  Treat these
    # aliases as variadic constructors so an API snapshot remains portable
    # across the declared 3.11--3.13 matrix.
    if isinstance(value, types.GenericAlias):
        return "(*args, **kwargs)"
    try:
        signature = inspect.signature(value)
        # NumPy 2 added the scalar type to float64 repr. Keep the committed
        # representation on NumPy 1.26 without changing the callable default
        # or concealing actual changes to its type or value.
        parameters = [
            parameter.replace(
                default=_SignatureDefault(f"np.float64({float(parameter.default)!r})")
            )
            if type(parameter.default) is np.float64
            else parameter
            for parameter in signature.parameters.values()
        ]
        return str(signature.replace(parameters=parameters))
    except (TypeError, ValueError):
        return None


def _public_symbol(namespace: str, name: str, value: Any) -> PublicApiSymbol:
    # ``Path`` is an operating-system factory.  The same public constant is a
    # ``PosixPath`` on Linux and a ``WindowsPath`` on Windows.  Recording that
    # concrete implementation in a supposedly portable API manifest makes an
    # otherwise identical release tree produce different snapshots on each
    # platform.  Normalize all concrete path objects to the public
    # ``pathlib.Path`` abstraction before recording their identity.
    if isinstance(value, PurePath):
        implementation_module = "pathlib"
        qualname = "Path"
    else:
        implementation_module = str(getattr(value, "__module__", type(value).__module__))
        qualname = str(
            getattr(value, "__qualname__", getattr(value, "__name__", type(value).__qualname__))
        )
    return PublicApiSymbol(
        namespace=namespace,
        name=name,
        kind=_public_kind(value),
        implementation_module=implementation_module,
        qualname=qualname,
        signature=_public_signature(value),
    )


def create_public_api_manifest(
    namespaces: Sequence[str] = DEFAULT_PUBLIC_NAMESPACES,
    *,
    package_version: str | None = None,
) -> PublicApiManifest:
    """Inspect selected namespaces and build a deterministic ``__all__`` manifest."""

    normalized = tuple(str(namespace).strip() for namespace in namespaces)
    if not normalized or any(not namespace for namespace in normalized):
        raise ValidationError("public API namespaces must be nonempty")
    if len(set(normalized)) != len(normalized):
        raise ValidationError("public API namespaces must be unique")
    symbols: list[PublicApiSymbol] = []
    for namespace in normalized:
        module = import_module(namespace)
        exported = getattr(module, "__all__", None)
        if exported is None:
            raise ValidationError(f"public API namespace {namespace!r} does not define __all__")
        names = tuple(str(name) for name in exported)
        if len(set(names)) != len(names):
            raise ValidationError(f"public API namespace {namespace!r} contains duplicate exports")
        for name in names:
            if not hasattr(module, name):
                raise ValidationError(f"public API export {namespace}.{name} is missing")
            symbols.append(_public_symbol(namespace, name, getattr(module, name)))
    return PublicApiManifest(
        package_version=__version__ if package_version is None else str(package_version),
        namespaces=normalized,
        symbols=tuple(sorted(symbols)),
    )


def create_beta_api_manifest(
    *,
    package_version: str | None = None,
) -> PublicApiManifest:
    """Build the curated beta namespace manifest with strict signatures."""

    observed = create_public_api_manifest(
        BETA_PUBLIC_NAMESPACES,
        package_version=package_version,
    )
    return PublicApiManifest(
        package_version=observed.package_version,
        namespaces=observed.namespaces,
        symbols=observed.symbols,
        stability_level=BETA_API_STABILITY_LEVEL,
    )


def create_stable_api_manifest(
    *,
    package_version: str | None = None,
) -> PublicApiManifest:
    """Build the curated 1.0-candidate namespace manifest."""

    observed = create_public_api_manifest(
        STABLE_PUBLIC_NAMESPACES,
        package_version=package_version,
    )
    return PublicApiManifest(
        package_version=observed.package_version,
        namespaces=observed.namespaces,
        symbols=observed.symbols,
        stability_level=STABLE_API_STABILITY_LEVEL,
    )


def load_public_api_manifest(
    path: str | Path | None = None,
) -> PublicApiManifest:
    """Load and validate one committed public API manifest."""

    source = DEFAULT_PUBLIC_API_MANIFEST if path is None else Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValidationError(f"could not read public API manifest {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError("public API manifest is not valid JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValidationError("public API manifest root must be an object")
    return PublicApiManifest.from_dict(raw)


def load_beta_api_manifest(
    path: str | Path | None = None,
) -> PublicApiManifest:
    """Load and validate the committed curated beta API manifest."""

    source = DEFAULT_BETA_API_MANIFEST if path is None else Path(path)
    manifest = load_public_api_manifest(source)
    if manifest.namespaces != BETA_PUBLIC_NAMESPACES:
        raise ValidationError("beta API manifest covers the wrong namespace set")
    if manifest.stability_level != BETA_API_STABILITY_LEVEL:
        raise ValidationError("beta API manifest declares the wrong stability level")
    return manifest


def load_stable_api_manifest(
    path: str | Path | None = None,
) -> PublicApiManifest:
    """Load and validate the committed 1.0-candidate API manifest."""

    source = DEFAULT_STABLE_API_MANIFEST if path is None else Path(path)
    manifest = load_public_api_manifest(source)
    if manifest.namespaces != STABLE_PUBLIC_NAMESPACES:
        raise ValidationError("stable API manifest covers the wrong namespace set")
    if manifest.stability_level != STABLE_API_STABILITY_LEVEL:
        raise ValidationError("stable API manifest declares the wrong stability level")
    return manifest


def save_public_api_manifest(
    manifest: PublicApiManifest,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write one canonical public API manifest with deterministic newlines."""

    if not isinstance(manifest, PublicApiManifest):
        raise ValidationError("save_public_api_manifest requires a PublicApiManifest")
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise ValidationError(f"refusing to overwrite public API manifest {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        manifest.to_dict(),
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    destination.write_text(content, encoding="utf-8", newline="\n")
    return destination


def compare_public_api(
    reference: PublicApiManifest,
    current: PublicApiManifest | None = None,
    *,
    strict_signatures: bool = False,
) -> ApiCompatibilityReport:
    """Compare manifests under an additive compatibility policy.

    Additions are compatible.  Removed exports and broad kind changes are not.
    Signature changes are always reported and become failures only when
    ``strict_signatures`` is true.
    """

    if not isinstance(reference, PublicApiManifest):
        raise ValidationError("reference must be a PublicApiManifest")
    observed = create_public_api_manifest(reference.namespaces) if current is None else current
    if not isinstance(observed, PublicApiManifest):
        raise ValidationError("current must be a PublicApiManifest")
    if observed.namespaces != reference.namespaces:
        raise ValidationError("reference and current manifests must cover identical namespaces")

    expected_map = reference.symbol_map()
    observed_map = observed.symbol_map()
    expected_keys = set(expected_map)
    observed_keys = set(observed_map)
    added = tuple(sorted(observed_keys - expected_keys))
    removed = tuple(sorted(expected_keys - observed_keys))
    common = sorted(expected_keys & observed_keys)
    kind_changed = tuple(
        (namespace, name, expected_map[(namespace, name)].kind, observed_map[(namespace, name)].kind)
        for namespace, name in common
        if expected_map[(namespace, name)].kind != observed_map[(namespace, name)].kind
    )
    signature_changed = tuple(
        (
            namespace,
            name,
            expected_map[(namespace, name)].signature,
            observed_map[(namespace, name)].signature,
        )
        for namespace, name in common
        if expected_map[(namespace, name)].signature != observed_map[(namespace, name)].signature
    )
    valid = not removed and not kind_changed and (not strict_signatures or not signature_changed)
    certificate = Certificate(
        valid=valid,
        exactness=Exactness.EXACT,
        residuals={
            "removed_exports": float(len(removed)),
            "kind_changes": float(len(kind_changed)),
            "signature_changes": float(len(signature_changed)),
        },
        assumptions_checked=(
            "selected namespaces define explicit __all__ exports",
            "export presence and broad symbol kind",
            "deterministic callable signature text",
        ),
        assumptions_unchecked=(
            "behavioral and mathematical semantic equivalence",
            "compatibility of undocumented implementation details",
        ),
        diagnostics={
            "stability_level": reference.stability_level,
            "added": len(added),
            "removed": len(removed),
            "kind_changed": len(kind_changed),
            "signature_changed": len(signature_changed),
            "strict_signatures": bool(strict_signatures),
        },
        theorem_id="KM-API-010",
        message=(
            "public API is structurally compatible with the committed manifest"
            if valid
            else "public API is incompatible with the committed manifest"
        ),
    )
    return ApiCompatibilityReport(
        reference_digest=reference.digest,
        current_digest=observed.digest,
        added=added,
        removed=removed,
        kind_changed=kind_changed,
        signature_changed=signature_changed,
        strict_signatures=bool(strict_signatures),
        certificate=certificate,
    )


def check_public_api(
    path: str | Path | None = None,
    *,
    strict_signatures: bool = False,
) -> ApiCompatibilityReport:
    """Compare the installed package with one committed API manifest."""

    reference = load_public_api_manifest(path)
    return compare_public_api(reference, strict_signatures=strict_signatures)


def check_beta_api(
    path: str | Path | None = None,
) -> ApiCompatibilityReport:
    """Check the curated beta namespace under strict signature compatibility."""

    reference = load_beta_api_manifest(path)
    current = create_beta_api_manifest()
    return compare_public_api(
        reference,
        current,
        strict_signatures=True,
    )


def check_stable_api(
    path: str | Path | None = None,
) -> ApiCompatibilityReport:
    """Check the 1.0-candidate namespace under strict signature compatibility."""

    reference = load_stable_api_manifest(path)
    current = create_stable_api_manifest()
    return compare_public_api(
        reference,
        current,
        strict_signatures=True,
    )


def warn_deprecated(
    name: str,
    *,
    since: str,
    replacement: str | None = None,
    remove_in: str | None = None,
    reason: str | None = None,
    stacklevel: int = 2,
) -> None:
    """Emit one structured Komorebi deprecation warning."""

    normalized_name = str(name).strip()
    normalized_since = str(since).strip()
    if not normalized_name or not normalized_since:
        raise ValidationError("deprecated names and since versions must be nonempty")
    clauses = [f"{normalized_name} is deprecated since {normalized_since}"]
    if replacement:
        clauses.append(f"use {replacement} instead")
    if remove_in:
        clauses.append(f"planned removal is {remove_in}")
    if reason:
        clauses.append(str(reason))
    warnings.warn("; ".join(clauses), KomorebiDeprecationWarning, stacklevel=stacklevel)


def deprecated(
    *,
    since: str,
    replacement: str | None = None,
    remove_in: str | None = None,
    reason: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a callable with explicit versioned deprecation metadata."""

    normalized_since = str(since).strip()
    if not normalized_since:
        raise ValidationError("a deprecation since version is required")
    record = DeprecationRecord(normalized_since, replacement, remove_in, reason)

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        if not callable(function):
            raise ValidationError("deprecated can decorate only callables")

        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            warn_deprecated(
                function.__qualname__,
                since=record.since,
                replacement=record.replacement,
                remove_in=record.remove_in,
                reason=record.reason,
                stacklevel=2,
            )
            return function(*args, **kwargs)

        setattr(wrapped, "__komorebi_deprecation__", record)
        return wrapped

    return decorate


__all__ = [
    "API_STABILITY_LEVEL",
    "BETA_API_STABILITY_LEVEL",
    "BETA_PUBLIC_NAMESPACES",
    "DEFAULT_STABLE_API_MANIFEST",
    "DEFAULT_BETA_API_MANIFEST",
    "DEFAULT_PUBLIC_API_MANIFEST",
    "DEFAULT_PUBLIC_NAMESPACES",
    "STABLE_API_STABILITY_LEVEL",
    "STABLE_PUBLIC_NAMESPACES",
    "PUBLIC_API_SCHEMA",
    "PUBLIC_API_SCHEMA_VERSION",
    "ApiCompatibilityReport",
    "DeprecationRecord",
    "KomorebiDeprecationWarning",
    "PublicApiManifest",
    "PublicApiSymbol",
    "check_beta_api",
    "check_public_api",
    "check_stable_api",
    "compare_public_api",
    "create_beta_api_manifest",
    "create_public_api_manifest",
    "create_stable_api_manifest",
    "deprecated",
    "load_beta_api_manifest",
    "load_public_api_manifest",
    "load_stable_api_manifest",
    "save_public_api_manifest",
    "warn_deprecated",
]
