"""Explicit gate reports for conditional Langlands completion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from ...certificates import Certificate, ConditionalResult, Exactness


class CompletionGate(StrEnum):
    """The five formal gates in pointwise automorphic completion."""

    RESIDUAL = "residual"
    PATCHING = "patching"
    COMPARISON = "comparison"
    FIBER = "fiber"
    REALIZATION = "realization"


@dataclass(frozen=True, slots=True)
class GateStatus:
    """One gate value with a human-readable explanation."""

    satisfied: bool
    detail: str
    external: bool = True


@dataclass(frozen=True, slots=True)
class AutomorphicCompletionReport:
    """A five-gate conditional automorphic completion report."""

    gates: Mapping[CompletionGate | str, GateStatus]

    def __post_init__(self) -> None:
        normalized = {CompletionGate(key): value for key, value in self.gates.items()}
        missing = set(CompletionGate) - set(normalized)
        if missing:
            from ...exceptions import ValidationError

            raise ValidationError(f"completion report is missing gates: {sorted(gate.value for gate in missing)}")
        object.__setattr__(self, "gates", normalized)

    @property
    def complete(self) -> bool:
        """Return whether every gate is satisfied."""

        return all(status.satisfied for status in self.gates.values())

    @property
    def unmet_gates(self) -> tuple[CompletionGate, ...]:
        """Return the unmet gates in canonical order."""

        return tuple(gate for gate in CompletionGate if not self.gates[gate].satisfied)

    @property
    def external_gates(self) -> tuple[CompletionGate, ...]:
        """Return gates that depend on imported external arithmetic theorems."""

        return tuple(gate for gate in CompletionGate if self.gates[gate].external)

    def result(self) -> ConditionalResult[bool]:
        """Return the conditional completion conclusion and its assumptions."""

        checked = tuple(f"{gate.value}: {self.gates[gate].detail}" for gate in CompletionGate if self.gates[gate].satisfied)
        unchecked = tuple(f"{gate.value}: {self.gates[gate].detail}" for gate in CompletionGate if not self.gates[gate].satisfied)
        certificate = Certificate(
            valid=self.complete,
            exactness=Exactness.CONDITIONAL,
            assumptions_checked=checked,
            assumptions_unchecked=unchecked,
            diagnostics={
                "unmet_gates": tuple(gate.value for gate in self.unmet_gates),
                "external_gates": tuple(gate.value for gate in self.external_gates),
            },
            theorem_id="PSC-XII-FIVE-GATE-COMPLETION",
            message="all automorphic completion gates are supplied" if self.complete else "automorphic completion gates remain unmet",
        )
        return ConditionalResult(self.complete, certificate)


@dataclass(frozen=True, slots=True)
class CompatiblePropagationReport:
    """One-member propagation through an already supplied compatible system."""

    common_spine: bool
    semisimple: bool
    one_member_automorphic: bool
    local_global_compatibility: bool

    def result(self) -> ConditionalResult[bool]:
        """Return the conditional propagation conclusion."""

        values = {
            "common Frobenius spine": self.common_spine,
            "semisimple compatible members": self.semisimple,
            "one automorphic member": self.one_member_automorphic,
            "local global compatibility": self.local_global_compatibility,
        }
        complete = all(values.values())
        certificate = Certificate(
            valid=complete,
            exactness=Exactness.CONDITIONAL,
            assumptions_checked=tuple(name for name, value in values.items() if value),
            assumptions_unchecked=tuple(name for name, value in values.items() if not value),
            theorem_id="PSC-XII-ONE-MEMBER-PROPAGATION",
            message="automorphy propagates inside the supplied compatible system" if complete else "compatible propagation assumptions remain",
        )
        return ConditionalResult(complete, certificate)


__all__ = [
    "AutomorphicCompletionReport",
    "CompatiblePropagationReport",
    "CompletionGate",
    "GateStatus",
]
