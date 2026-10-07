"""Finite Clifford tower words, depth, and conditional descent plans."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

from ..._number_theory import require_integer, require_prime
from ...certificates import Certificate, ConditionalResult, Exactness
from ...exceptions import ValidationError


class CliffordMode(StrEnum):
    """The two prime-cyclic Clifford alternatives."""

    INDUCTION = "I"
    EXTENSION = "E"


@dataclass(frozen=True, slots=True)
class CliffordStep:
    """One validated prime-cyclic step in a declared Clifford tower."""

    parent_order: int
    child_order: int
    parent_degree: int
    child_degree: int
    mode: CliffordMode | str
    attached_descent: bool = False
    label: str = ""

    def __post_init__(self) -> None:
        parent_order = require_integer(self.parent_order, name="parent_order")
        child_order = require_integer(self.child_order, name="child_order")
        parent_degree = require_integer(self.parent_degree, name="parent_degree")
        child_degree = require_integer(self.child_degree, name="child_degree")
        if min(parent_order, child_order, parent_degree, child_degree) < 1:
            raise ValidationError("Clifford orders and degrees must be positive")
        if parent_order % child_order:
            raise ValidationError("child group order must divide the parent group order")
        quotient = parent_order // child_order
        require_prime(quotient, name="prime cyclic quotient order")
        mode = CliffordMode(self.mode)
        if mode is CliffordMode.INDUCTION and parent_degree != quotient * child_degree:
            raise ValidationError("induction step degree must multiply by the prime index")
        if mode is CliffordMode.EXTENSION and parent_degree != child_degree:
            raise ValidationError("extension step degree must remain unchanged")
        object.__setattr__(self, "parent_order", parent_order)
        object.__setattr__(self, "child_order", child_order)
        object.__setattr__(self, "parent_degree", parent_degree)
        object.__setattr__(self, "child_degree", child_degree)
        object.__setattr__(self, "mode", mode)

    @property
    def quotient_prime(self) -> int:
        """Return the prime cyclic quotient order."""

        return self.parent_order // self.child_order

    @property
    def nonabelian_extension_gate(self) -> bool:
        """Return whether this step is a higher-rank extension gate."""

        return self.mode is CliffordMode.EXTENSION and self.parent_degree > 1


@dataclass(frozen=True, slots=True)
class CliffordTowerReport:
    """One declared Clifford word and its unresolved descent gates."""

    word: str
    depth: int
    induction_steps: tuple[int, ...]
    extension_steps: tuple[int, ...]
    unresolved_extension_steps: tuple[int, ...]
    certificate: Certificate


class CliffordTower:
    """A finite prime-cyclic Clifford tower supplied with dimension data."""

    def __init__(self, steps: Iterable[CliffordStep]) -> None:
        self.steps = tuple(steps)
        if not self.steps:
            raise ValidationError("a Clifford tower requires at least one step")
        for index in range(len(self.steps) - 1):
            if self.steps[index].child_order != self.steps[index + 1].parent_order:
                raise ValidationError("Clifford group orders do not form a chain")
            if self.steps[index].child_degree != self.steps[index + 1].parent_degree:
                raise ValidationError("Clifford representation degrees do not form a chain")

    @property
    def word(self) -> str:
        """Return the Clifford word in induction and extension letters."""

        return "".join(step.mode.value for step in self.steps)

    @property
    def depth(self) -> int:
        """Return the number of nonabelian extension gates."""

        return sum(step.nonabelian_extension_gate for step in self.steps)

    def report(self) -> CliffordTowerReport:
        """Return the word, depth, and unresolved attached descent steps."""

        induction = tuple(index for index, step in enumerate(self.steps) if step.mode is CliffordMode.INDUCTION)
        extensions = tuple(index for index, step in enumerate(self.steps) if step.mode is CliffordMode.EXTENSION)
        unresolved = tuple(
            index
            for index, step in enumerate(self.steps)
            if step.nonabelian_extension_gate and not step.attached_descent
        )
        return CliffordTowerReport(
            word=self.word,
            depth=self.depth,
            induction_steps=induction,
            extension_steps=extensions,
            unresolved_extension_steps=unresolved,
            certificate=Certificate(
                valid=True,
                exactness=Exactness.EXACT,
                assumptions_checked=("declared prime-cyclic order chain", "declared Clifford degree alternative"),
                assumptions_unchecked=("existence of the declared representation restriction data",),
                diagnostics={"word": self.word, "depth": self.depth, "unresolved": unresolved},
                theorem_id="PSC-LANG-CLIFFORD-DEPTH",
                message="Clifford tower data are internally consistent",
            ),
        )

    def automorphic_plan(self) -> ConditionalResult[tuple[str, ...]]:
        """Return the formal induction and descent plan with explicit open gates."""

        operations: list[str] = []
        unchecked: list[str] = []
        for index, step in reversed(tuple(enumerate(self.steps))):
            if step.mode is CliffordMode.INDUCTION:
                operations.append(f"cyclic automorphic induction at step {index}")
            elif step.parent_degree == 1:
                operations.append(f"class field character extension at step {index}")
            elif step.attached_descent:
                operations.append(f"attached higher-rank cyclic descent at step {index}")
            else:
                operations.append(f"unresolved higher-rank cyclic descent at step {index}")
                unchecked.append(f"attached descent exists at Clifford step {index}")
        certificate = Certificate(
            valid=not unchecked,
            exactness=Exactness.CONDITIONAL,
            assumptions_checked=("finite Clifford word", "cyclic induction steps", "rank-one extension steps"),
            assumptions_unchecked=tuple(unchecked),
            diagnostics={"word": self.word, "depth": self.depth},
            theorem_id="PSC-LANG-SOLVABLE-COMPLETION",
            message="all Clifford gates are supplied" if not unchecked else "higher-rank cyclic descent gates remain",
        )
        return ConditionalResult(tuple(operations), certificate)


@dataclass(frozen=True, slots=True)
class VirtualCharacterTerm:
    """One coefficient in a virtual character resolution."""

    coefficient: int
    label: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficient", require_integer(self.coefficient, name="coefficient"))


@dataclass(frozen=True, slots=True)
class VirtualCharacterResolution:
    """A virtual monomial resolution with a positivity diagnostic."""

    terms: tuple[VirtualCharacterTerm, ...]

    @property
    def has_negative_terms(self) -> bool:
        """Return whether the resolution uses subtraction."""

        return any(term.coefficient < 0 for term in self.terms)

    @property
    def positive(self) -> bool:
        """Return whether every coefficient is nonnegative."""

        return not self.has_negative_terms

    def l_function_expression(self) -> str:
        """Return a symbolic virtual product expression."""

        factors = [f"L({term.label})^{term.coefficient}" for term in self.terms if term.coefficient]
        return " * ".join(factors) if factors else "1"


__all__ = [
    "CliffordMode",
    "CliffordStep",
    "CliffordTower",
    "CliffordTowerReport",
    "VirtualCharacterResolution",
    "VirtualCharacterTerm",
]
