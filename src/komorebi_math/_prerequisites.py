"""Private certificate prerequisite composition helpers."""
from __future__ import annotations

from typing import Any


def packet_certificate_dependencies(**systems: Any) -> tuple[bool, tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    """Recompute named packet prerequisites and expose their evidence.

    This helper is intentionally private. It composes existing PacketSystem
    certificates; it does not turn unresolved or external assumptions into proof.
    """
    cache: dict[int, Any] = {}
    valid = True
    checked: list[str] = []
    unchecked: list[str] = []
    diagnostics: dict[str, Any] = {}
    for label, system in systems.items():
        key = id(system)
        certificate = cache.get(key)
        if certificate is None:
            certificate = system.certificate()
            cache[key] = certificate
        valid = bool(valid and certificate.valid)
        checked.append(f"{label} packet axioms")
        unchecked.extend(f"{label} packet axioms: {item}" for item in certificate.assumptions_unchecked)
        diagnostics[f"{label}_packet_certificate"] = certificate.to_dict()
    return valid, tuple(checked), tuple(unchecked), diagnostics
