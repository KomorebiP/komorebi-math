"""Finite graded nonlinear operations and packet Maurer--Cartan equations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import factorial
from typing import Any

import numpy as np

from .._residuals import value_audit, finite_tolerance
from .._prerequisites import packet_certificate_dependencies
from ..certificates import Certificate, Exactness
from ..core.packets import PacketSystem
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented

GradedOperationCallable = Callable[[tuple[Any, ...]], Any]
LinearOperation = Callable[[Any], Any]
BracketOperation = Callable[[Any, Any], Any]


def _add(left: Any | None, right: Any) -> Any:
    if left is None:
        return right
    try:
        return left + right
    except TypeError:
        return np.asarray(left, dtype=object) + np.asarray(right, dtype=object)


def _subtract(left: Any, right: Any) -> Any:
    try:
        return left - right
    except TypeError:
        return np.asarray(left, dtype=object) - np.asarray(right, dtype=object)


def _zero_like(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return np.zeros_like(np.asarray(value))
    try:
        return 0 * value
    except TypeError:
        return np.zeros_like(np.asarray(value))


def _norm(value: Any) -> float:
    try:
        return float(np.linalg.norm(np.asarray(value, dtype=complex).reshape(-1)))
    except (TypeError, ValueError):
        entries = np.asarray(value, dtype=object).reshape(-1)
        return float(sum(abs(complex(entry)) ** 2 for entry in entries) ** 0.5)


def _scale(value: Any, numerator: int, denominator: int, *, exact: bool) -> Any:
    scalar: Any = Fraction(numerator, denominator) if exact else numerator / denominator
    return scalar * value


def _combine_exactness(*values: Exactness) -> Exactness:
    if all(value is Exactness.EXACT for value in values):
        return Exactness.EXACT
    if any(value is Exactness.CONDITIONAL for value in values):
        return Exactness.CONDITIONAL
    if any(value is Exactness.HEURISTIC for value in values):
        return Exactness.HEURISTIC
    if any(value is Exactness.PLANNED for value in values):
        return Exactness.PLANNED
    return Exactness.NUMERICAL


@documented(
    ProvenanceRecord(
        identifier="PSC-III-D17",
        title="Finite abelian grading group",
        status=ClaimStatus.DEFINITION,
        source="Paper III runtime",
        assumptions=("finite explicit operation table",),
        verification_suite="nonlinear.graded",
    )
)
class FiniteAbelianGroup:
    """A finite abelian group represented by an explicit addition table."""

    def __init__(
        self,
        labels: Sequence[str],
        addition: Mapping[tuple[str, str], str],
        *,
        zero: str,
        validate: bool = True,
    ) -> None:
        self.labels = tuple(str(label) for label in labels)
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValidationError("group labels must be nonempty and unique")
        self.zero = str(zero)
        if self.zero not in self.labels:
            raise ValidationError("group identity must be one of the labels")
        self._addition = {
            (str(left), str(right)): str(value)
            for (left, right), value in addition.items()
        }
        expected = {(left, right) for left in self.labels for right in self.labels}
        if set(self._addition) != expected:
            raise ValidationError("the addition table must contain every ordered label pair exactly once")
        if any(value not in self.labels for value in self._addition.values()):
            raise ValidationError("the addition table is not closed on the label set")
        if validate:
            self.certificate().require()

    @classmethod
    def cyclic(cls, order: int, *, prefix: str = "") -> "FiniteAbelianGroup":
        size = int(order)
        if isinstance(order, bool) or size != order or size < 1:
            raise ValidationError("cyclic group order must be a positive integer")
        labels = tuple(f"{prefix}{index}" for index in range(size))
        table = {
            (labels[left], labels[right]): labels[(left + right) % size]
            for left in range(size)
            for right in range(size)
        }
        return cls(labels, table, zero=labels[0])

    def add(self, left: str, right: str) -> str:
        try:
            return self._addition[(str(left), str(right))]
        except KeyError as exc:
            raise ValidationError("group label is outside the finite grading group") from exc

    def sum(self, labels: Sequence[str]) -> str:
        total = self.zero
        for label in labels:
            total = self.add(total, str(label))
        return total

    def inverse(self, label: str) -> str:
        value = str(label)
        for candidate in self.labels:
            if self.add(value, candidate) == self.zero:
                return candidate
        raise ValidationError("the explicit table has no inverse for a group label")

    def certificate(self) -> Certificate:
        identity_failures = 0
        commutativity_failures = 0
        associativity_failures = 0
        inverse_failures = 0
        for left in self.labels:
            if self.add(self.zero, left) != left or self.add(left, self.zero) != left:
                identity_failures += 1
            if not any(
                self.add(left, candidate) == self.zero and self.add(candidate, left) == self.zero
                for candidate in self.labels
            ):
                inverse_failures += 1
            for right in self.labels:
                if self.add(left, right) != self.add(right, left):
                    commutativity_failures += 1
                for third in self.labels:
                    if self.add(self.add(left, right), third) != self.add(left, self.add(right, third)):
                        associativity_failures += 1
        valid = not any((identity_failures, inverse_failures, commutativity_failures, associativity_failures))
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT,
            residuals={},
            assumptions_checked=("closure", "identity", "inverses", "commutativity", "associativity"),
            diagnostics={
                "order": len(self.labels),
                "identity_failures": identity_failures,
                "inverse_failures": inverse_failures,
                "commutativity_failures": commutativity_failures,
                "associativity_failures": associativity_failures,
            },
            theorem_id="PSC-III-D17",
            message="valid finite abelian group table" if valid else "invalid finite abelian group table",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": self.labels,
            "zero": self.zero,
            "addition": {
                f"{left}+{right}": value
                for (left, right), value in self._addition.items()
            },
        }


@dataclass(frozen=True, slots=True)
class GradingCertificate:
    """Finite certificate that a multilinear operation respects the grading."""


    arity: int
    tested_input_labels: int
    forbidden_residual: float
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="PSC-III-T18",
        title="Graded packet convolution",
        status=ClaimStatus.PROVED,
        source="Paper III",
        assumptions=("finite group grading", "grade-respecting multilinear operation"),
        verification_suite="nonlinear.graded",
    )
)
class GradedPacketOperation:
    """A multilinear operation tested against a finite packet grading."""

    def __init__(
        self,
        group: FiniteAbelianGroup,
        packets: PacketSystem,
        arity: int,
        operation: GradedOperationCallable,
        *,
        exactness: Exactness = Exactness.NUMERICAL,
        tolerance: float | None = None,
        name: str = "graded packet operation",
    ) -> None:
        n = int(arity)
        if isinstance(arity, bool) or n != arity or n < 1:
            raise ValidationError("graded operation arity must be a positive integer")
        if tuple(packets.names) != tuple(group.labels):
            raise ValidationError("packet names must equal the grading group labels in order")
        if not callable(operation):
            raise ValidationError("graded operation must be callable")
        self.group = group
        self.packets = packets
        self.arity = n
        self.operation = operation
        self.exactness = exactness
        self.tolerance = finite_tolerance(packets.tolerance if tolerance is None else tolerance)
        self.name = str(name)

    def _validate_output(self, value: Any) -> Any:
        vector = self.packets.backend.as_vector(value)
        if self.packets.backend.shape(vector)[0] != self.packets.dimension:
            raise DimensionError("graded operation output dimension does not match the packet system")
        return vector

    def evaluate(self, arguments: Sequence[Any]) -> Any:
        values = tuple(arguments)
        if len(values) != self.arity:
            raise ValidationError(f"arity {self.arity} operation requires exactly {self.arity} arguments")
        vectors = tuple(self.packets.backend.as_vector(value) for value in values)
        if any(self.packets.backend.shape(vector)[0] != self.packets.dimension for vector in vectors):
            raise DimensionError("graded operation argument dimension does not match the packet system")
        return self._validate_output(self.operation(vectors))

    def grading_certificate(self, *, max_evaluations: int = 100_000) -> GradingCertificate:
        """Exhaustively test forbidden output packets on packet-range bases."""

        limit = int(max_evaluations)
        if isinstance(max_evaluations, bool) or limit != max_evaluations or limit < 1:
            raise ValidationError("max_evaluations must be a positive integer")
        bases: list[tuple[Any, ...]] = []
        for projector in self.packets.projectors:
            if self.packets.backend.name == "sympy":
                bases.append(tuple(projector.columnspace()))
            else:
                matrix = np.asarray(projector, dtype=complex)
                u, singular_values, _ = np.linalg.svd(matrix)
                threshold = self.tolerance if singular_values.size == 0 else max(
                    self.tolerance,
                    self.tolerance * float(singular_values[0]),
                )
                rank = int(np.sum(singular_values > threshold))
                bases.append(tuple(u[:, index] for index in range(rank)))
        audit = value_audit(exact=self.exactness is Exactness.EXACT and self.packets.backend.exact, tolerance=self.tolerance)
        residual = 0.0
        evaluations = 0
        complete = True
        tested_labels = 0
        for input_indices in product(range(len(self.packets)), repeat=self.arity):
            tested_labels += 1
            expected_name = self.group.sum(tuple(self.packets.names[index] for index in input_indices))
            expected_index = self.packets.index(expected_name)
            for arguments in product(*(bases[index] for index in input_indices)):
                if evaluations >= limit:
                    complete = False
                    break
                evaluations += 1
                value = self.evaluate(arguments)
                for output_index, projector in enumerate(self.packets.projectors):
                    if output_index == expected_index:
                        continue
                    forbidden = self.packets.backend.matmul(projector, value)
                    residual = max(residual, audit.check_value(f"forbidden:{evaluations}:{output_index}", forbidden))
            if not complete:
                break
        backend_exactness = Exactness.EXACT if self.packets.backend.exact else Exactness.NUMERICAL
        exactness = _combine_exactness(self.exactness, backend_exactness)
        if not complete:
            exactness = Exactness.CONDITIONAL
        parents_valid, parent_checked, parent_unchecked, parent_diagnostics = packet_certificate_dependencies(
            grading=self.packets
        )
        valid = parents_valid and audit.valid
        return GradingCertificate(
            self.arity,
            tested_labels,
            residual,
            Certificate(
                valid=valid,
                exactness=exactness,
                residuals={"forbidden_packet_output": residual},
                tolerance=None if exactness is Exactness.EXACT else self.tolerance,
                assumptions_checked=parent_checked + ("packet-range bases", "forbidden output packets", "tested label tuples"),
                assumptions_unchecked=parent_unchecked + ("global multilinearity of supplied operation",) + audit.unresolved +
                    (("unseen label tuples due to evaluation limit",) if not complete else ()),
                diagnostics={
                    **parent_diagnostics,
                    "arity": self.arity,
                    "residual_decisions": audit.decisions,
                    "basis_evaluations": evaluations,
                    "tested_label_tuples": tested_labels,
                    "complete": complete,
                },
                theorem_id="PSC-III-T18",
                message="operation respects the finite grading" if valid and complete else (
                    "tested portion respects the finite grading" if valid else "operation violates the finite grading"
                ),
            ),
        )

    def packet_convolution(self, arguments: Sequence[Any], output_label: str) -> "GradedConvolutionResult":
        """Compute one output packet by label convolution and compare directly."""

        values = tuple(self.packets.backend.as_vector(value) for value in arguments)
        if len(values) != self.arity:
            raise ValidationError(f"arity {self.arity} operation requires exactly {self.arity} arguments")
        output_name = str(output_label)
        output_index = self.packets.index(output_name)
        decompositions = tuple(self.packets.analyze(value).values for value in values)
        total = self.packets.backend.zeros(self.packets.dimension, 1)
        if self.packets.backend.name != "sympy":
            total = np.asarray(total).reshape(-1)
        terms: list[GradedConvolutionTerm] = []
        for indices in product(range(len(self.packets)), repeat=self.arity):
            labels = tuple(self.packets.names[index] for index in indices)
            if self.group.sum(labels) != output_name:
                continue
            raw = self.evaluate(tuple(decompositions[position][index] for position, index in enumerate(indices)))
            projected = self.packets.backend.matmul(self.packets.projectors[output_index], raw)
            total = self.packets.backend.add(total, projected)
            terms.append(GradedConvolutionTerm(labels, projected))
        direct = self.packets.backend.matmul(
            self.packets.projectors[output_index],
            self.evaluate(values),
        )
        exactness = _combine_exactness(
            self.exactness,
            Exactness.EXACT if self.packets.backend.exact else Exactness.NUMERICAL,
        )
        audit = value_audit(exact=exactness is Exactness.EXACT, tolerance=self.tolerance)
        residual = audit.check_value("graded_convolution_reconstruction", self.packets.backend.subtract(direct, total))
        parents_valid, parent_checked, parent_unchecked, parent_diagnostics = packet_certificate_dependencies(
            grading=self.packets
        )
        valid = parents_valid and audit.valid
        return GradedConvolutionResult(
            output_name,
            total,
            direct,
            tuple(terms),
            Certificate(
                valid=valid,
                exactness=exactness,
                residuals={"graded_convolution_reconstruction": residual},
                tolerance=None if exactness is Exactness.EXACT else self.tolerance,
                assumptions_checked=parent_checked + ("complete input packets", "output label sum enumeration"),
                assumptions_unchecked=parent_unchecked + ("operation respects the grading globally",) + audit.unresolved,
                diagnostics={**parent_diagnostics, "arity": self.arity, "retained_terms": len(terms), "residual_decisions": audit.decisions},
                theorem_id="PSC-III-T18",
                message="graded convolution reconstructs the output packet" if valid else "graded convolution reconstruction failed",
            ),
        )


@dataclass(frozen=True, slots=True)
class GradedConvolutionTerm:
    """One labeled input contribution to a graded packet convolution."""


    input_labels: tuple[str, ...]
    value: Any


@dataclass(frozen=True, slots=True)
class GradedConvolutionResult:
    """Packet-resolved graded convolution and direct reconstruction result."""


    output_label: str
    value: Any
    direct_value: Any
    terms: tuple[GradedConvolutionTerm, ...]
    certificate: Certificate


@dataclass(frozen=True, slots=True)
class MaurerCartanResult:
    """Finite packet Maurer--Cartan equations and reconstruction certificate."""


    equations: dict[str, Any]
    reconstructed: Any
    direct: Any
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="PSC-III-T19",
        title="Packet Maurer--Cartan equations",
        status=ClaimStatus.PROVED,
        source="Paper III",
        assumptions=("grading-preserving differential", "grade-additive bracket", "two invertible"),
        verification_suite="nonlinear.graded",
    )
)
def maurer_cartan_packets(
    group: FiniteAbelianGroup,
    components: Mapping[str, Any],
    differential: LinearOperation,
    bracket: BracketOperation,
    *,
    exactness: Exactness = Exactness.NUMERICAL,
    tolerance: float = 1e-10,
) -> MaurerCartanResult:
    """Assemble the finite packet equations for ``d eta + [eta,eta]/2``."""

    if set(components) != set(group.labels):
        raise ValidationError("Maurer--Cartan components must contain every group label exactly once")
    ordered = {label: components[label] for label in group.labels}
    sample = next(iter(ordered.values()))
    exact = exactness is Exactness.EXACT
    equations: dict[str, Any] = {}
    for output in group.labels:
        value = differential(ordered[output])
        for left in group.labels:
            for right in group.labels:
                if group.add(left, right) == output:
                    value = _add(value, _scale(bracket(ordered[left], ordered[right]), 1, 2, exact=exact))
        equations[output] = value
    reconstructed = _zero_like(sample)
    eta = _zero_like(sample)
    for label in group.labels:
        reconstructed = _add(reconstructed, equations[label])
        eta = _add(eta, ordered[label])
    direct = _add(differential(eta), _scale(bracket(eta, eta), 1, 2, exact=exact))
    audit = value_audit(exact=exact, tolerance=tolerance)
    residual = audit.check_value("reconstruction", _subtract(reconstructed, direct))
    valid = audit.valid
    return MaurerCartanResult(
        equations,
        reconstructed,
        direct,
        Certificate(
            valid=valid,
            exactness=exactness,
            residuals={"packet_maurer_cartan_reconstruction": residual},
            tolerance=None if exact else tolerance,
            assumptions_checked=("finite group label convolution", "packet equation reconstruction at supplied components"),
            assumptions_unchecked=("differential is linear", "bracket is bilinear and grade respecting") + audit.unresolved,
            diagnostics={"group_order": len(group.labels), "residual_decisions": audit.decisions},
            theorem_id="PSC-III-T19",
            message="packet Maurer--Cartan equations reconstruct the full equation" if valid else "Maurer--Cartan packet reconstruction failed",
        ),
    )


@dataclass(frozen=True, slots=True)
class HigherMaurerCartanResult:
    """Truncated higher packet equations through a declared maximum order."""


    equations: dict[str, Any]
    reconstructed: Any
    direct: Any
    max_order: int
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="PSC-III-P20",
        title="Finite truncated packet equations for higher operations",
        status=ClaimStatus.PROVED,
        source="Paper III",
        assumptions=("finite truncation", "grade-respecting multilinear operations", "factorials invertible"),
        limitations=("does not certify convergence of an infinite Maurer--Cartan series",),
        verification_suite="nonlinear.graded",
    )
)
def higher_maurer_cartan_packets(
    group: FiniteAbelianGroup,
    components: Mapping[str, Any],
    operations: Mapping[int, GradedOperationCallable],
    *,
    exactness: Exactness = Exactness.NUMERICAL,
    tolerance: float = 1e-10,
) -> HigherMaurerCartanResult:
    """Assemble a finite truncation ``sum_r ell_r(eta^r)/r!`` by labels."""

    if set(components) != set(group.labels):
        raise ValidationError("higher Maurer--Cartan components must contain every group label")
    normalized: dict[int, GradedOperationCallable] = {}
    for raw_order, operation in operations.items():
        order = int(raw_order)
        if isinstance(raw_order, bool) or order != raw_order or order < 1:
            raise ValidationError("higher operation orders must be positive integers")
        if not callable(operation):
            raise ValidationError("higher operations must be callable")
        normalized[order] = operation
    if not normalized:
        raise ValidationError("at least one higher operation is required")
    exact = exactness is Exactness.EXACT
    ordered = {label: components[label] for label in group.labels}
    sample = next(iter(ordered.values()))
    equations = {label: _zero_like(sample) for label in group.labels}
    eta = _zero_like(sample)
    for value in ordered.values():
        eta = _add(eta, value)
    direct = _zero_like(sample)
    for order, operation in sorted(normalized.items()):
        direct = _add(direct, _scale(operation((eta,) * order), 1, factorial(order), exact=exact))
        for labels in product(group.labels, repeat=order):
            output = group.sum(labels)
            term = operation(tuple(ordered[label] for label in labels))
            equations[output] = _add(
                equations[output],
                _scale(term, 1, factorial(order), exact=exact),
            )
    reconstructed = _zero_like(sample)
    for value in equations.values():
        reconstructed = _add(reconstructed, value)
    audit = value_audit(exact=exact, tolerance=tolerance)
    residual = audit.check_value("reconstruction", _subtract(reconstructed, direct))
    valid = audit.valid
    return HigherMaurerCartanResult(
        equations,
        reconstructed,
        direct,
        max(normalized),
        Certificate(
            valid=valid,
            exactness=exactness,
            residuals={"higher_packet_reconstruction": residual},
            tolerance=None if exact else tolerance,
            assumptions_checked=("finite higher-operation truncation", "label convolution"),
            assumptions_unchecked=(
                "operations are multilinear and grade respecting",
                "convergence of any omitted infinite tail",
            ) + audit.unresolved,
            diagnostics={"orders": tuple(sorted(normalized)), "group_order": len(group.labels), "residual_decisions": audit.decisions},
            theorem_id="PSC-III-P20",
            message="truncated higher packet equations reconstruct the full truncation" if valid else "higher packet reconstruction failed",
        ),
    )


@dataclass(frozen=True, slots=True)
class DividedSquareResult:
    """Packet divided-square equations and direct reconstruction result."""


    equations: dict[str, Any]
    reconstructed: Any
    direct: Any
    certificate: Certificate


def divided_square_packets(
    group: FiniteAbelianGroup,
    components: Mapping[str, Any],
    differential: LinearOperation,
    bracket: BracketOperation,
    divided_square: LinearOperation,
    *,
    exactness: Exactness = Exactness.NUMERICAL,
    tolerance: float = 1e-10,
) -> DividedSquareResult:
    """Assemble ``d eta + q(eta)`` from divided-square packet identities."""

    if set(components) != set(group.labels):
        raise ValidationError("divided-square components must contain every group label")
    ordered = {label: components[label] for label in group.labels}
    sample = next(iter(ordered.values()))
    equations: dict[str, Any] = {}
    for output in group.labels:
        value = differential(ordered[output])
        for label in group.labels:
            if group.add(label, label) == output:
                value = _add(value, divided_square(ordered[label]))
        for left_index, left in enumerate(group.labels):
            for right in group.labels[left_index + 1 :]:
                if group.add(left, right) == output:
                    value = _add(value, bracket(ordered[left], ordered[right]))
        equations[output] = value
    eta = _zero_like(sample)
    reconstructed = _zero_like(sample)
    for label in group.labels:
        eta = _add(eta, ordered[label])
        reconstructed = _add(reconstructed, equations[label])
    direct = _add(differential(eta), divided_square(eta))
    exact = exactness is Exactness.EXACT
    audit = value_audit(exact=exact, tolerance=tolerance)
    residual = audit.check_value("reconstruction", _subtract(reconstructed, direct))
    valid = audit.valid
    return DividedSquareResult(
        equations,
        reconstructed,
        direct,
        Certificate(
            valid=valid,
            exactness=exactness,
            residuals={"divided_square_packet_reconstruction": residual},
            tolerance=None if exact else tolerance,
            assumptions_checked=("ordered cross-label pairs", "finite label convolution"),
            assumptions_unchecked=("declared divided-square polarization identity", "grading compatibility") + audit.unresolved,
            diagnostics={"residual_decisions": audit.decisions},
            theorem_id="PSC-III-P21",
            message="divided-square packet equations reconstruct the full equation" if valid else "divided-square packet reconstruction failed",
        ),
    )


__all__ = [
    "DividedSquareResult",
    "FiniteAbelianGroup",
    "GradedConvolutionResult",
    "GradedConvolutionTerm",
    "GradedPacketOperation",
    "GradingCertificate",
    "HigherMaurerCartanResult",
    "MaurerCartanResult",
    "divided_square_packets",
    "higher_maurer_cartan_packets",
    "maurer_cartan_packets",
]
