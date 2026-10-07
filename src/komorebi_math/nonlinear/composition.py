"""Higher chain rules and packet Faà di Bruno composition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Any

import numpy as np

from .._residuals import value_audit, finite_tolerance
from .._prerequisites import packet_certificate_dependencies
from ..certificates import Certificate, Exactness
from ..core.packets import PacketSystem
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .derivatives import DerivativeProvider
from .jets import PacketJetAddress, PacketJetFamily, PacketJetSupport, PacketLabel

Partition = tuple[tuple[int, ...], ...]


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


def _add(left: Any | None, right: Any) -> Any:
    if left is None:
        return right
    return left + right


def _norm(value: Any) -> float:
    try:
        return float(np.linalg.norm(np.asarray(value, dtype=complex).reshape(-1)))
    except (TypeError, ValueError):
        array = np.asarray(value, dtype=object).reshape(-1)
        return float(sum(abs(complex(entry)) ** 2 for entry in array) ** 0.5)


@lru_cache(maxsize=None)
def set_partitions(order: int) -> tuple[Partition, ...]:
    """Return deterministic set partitions of ``range(order)``.

    Blocks and entries are ordered by their least elements.  The representation
    is canonical and suitable for serialization and exact term counting.
    """

    n = int(order)
    if isinstance(order, bool) or n != order or n < 0:
        raise ValidationError("partition order must be a nonnegative integer")
    if n == 0:
        return ((),)
    previous = set_partitions(n - 1)
    output: list[Partition] = []
    new_item = n - 1
    for partition in previous:
        output.append(partition + ((new_item,),))
        for block_index in range(len(partition)):
            blocks = list(partition)
            blocks[block_index] = blocks[block_index] + (new_item,)
            output.append(tuple(blocks))
    output.sort(key=lambda partition: (len(partition), partition))
    return tuple(output)


def stirling_second(order: int, blocks: int) -> int:
    """Return an exact Stirling number of the second kind."""

    n = int(order)
    k = int(blocks)
    if n < 0 or k < 0:
        return 0
    table = [[0] * (k + 1) for _ in range(n + 1)]
    table[0][0] = 1
    for row in range(1, n + 1):
        for column in range(1, min(row, k) + 1):
            table[row][column] = table[row - 1][column - 1] + column * table[row - 1][column]
    return table[n][k] if k <= n else 0


def labeled_composition_term_count(order: int, intermediate_packets: int) -> int:
    """Return ``sum_k S(order,k) * intermediate_packets**k``."""

    n = int(order)
    packets = int(intermediate_packets)
    if isinstance(order, bool) or n != order or n < 1:
        raise ValidationError("composition order must be a positive integer")
    if isinstance(intermediate_packets, bool) or packets != intermediate_packets or packets < 1:
        raise ValidationError("intermediate packet count must be a positive integer")
    return sum(stirling_second(n, blocks) * packets**blocks for blocks in range(1, n + 1))


@dataclass(frozen=True, slots=True)
class FaaDiBrunoTerm:
    """One classical or packet-labeled composition term."""

    partition: Partition
    intermediate_packets: tuple[str, ...]
    value: Any
    pruned: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "intermediate_packets": self.intermediate_packets,
            "pruned": self.pruned,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class FaaDiBrunoResult:
    """A higher composition value with term and validation records."""

    value: Any
    terms: tuple[FaaDiBrunoTerm, ...]
    raw_term_count: int
    evaluated_term_count: int
    pruned_term_count: int
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="PSC-III-T10",
        title="Finite dimensional Faà di Bruno formula",
        status=ClaimStatus.CLASSICAL,
        source="Classical higher chain rule used in Paper III",
        assumptions=("declared derivatives through the requested order",),
        verification_suite="nonlinear.composition",
    )
)
def faa_di_bruno(
    outer: DerivativeProvider,
    inner: DerivativeProvider,
    point: Any,
    directions: Sequence[Any],
    *,
    direct_provider: DerivativeProvider | None = None,
    tolerance: float = 1e-9,
    retain_terms: bool = True,
) -> FaaDiBrunoResult:
    """Evaluate the directional higher chain rule for ``outer(inner(x))``."""

    tolerance = finite_tolerance(tolerance)
    direction_tuple = tuple(directions)
    order = len(direction_tuple)
    if order < 1:
        raise ValidationError("Faà di Bruno evaluation requires at least one direction")
    if inner.target_dimension != outer.source_dimension:
        raise DimensionError("inner target and outer source dimensions do not match")
    if not all(
        inner.supports_order(value) and outer.supports_order(value)
        for value in range(1, order + 1)
    ):
        raise ValidationError("both providers must supply derivatives through the requested order")
    intermediate_point = inner.value(point)
    terms: list[FaaDiBrunoTerm] = []
    total: Any | None = None
    for partition in set_partitions(order):
        inner_values = tuple(
            inner.derivative(len(block), point, tuple(direction_tuple[index] for index in block))
            for block in partition
        )
        value = outer.derivative(len(partition), intermediate_point, inner_values)
        total = _add(total, value)
        if retain_terms:
            terms.append(FaaDiBrunoTerm(partition, (), value))
    if total is None:  # pragma: no cover - order is positive
        raise ArithmeticError("composition formula produced no terms")

    exactness = _combine_exactness(outer.exactness, inner.exactness)
    audit = value_audit(exact=exactness is Exactness.EXACT, tolerance=tolerance)
    residual = 0.0
    assumptions_unchecked = ["global correctness of declared derivative providers"]
    if direct_provider is not None:
        if direct_provider.source_dimension != inner.source_dimension or direct_provider.target_dimension != outer.target_dimension:
            raise DimensionError("direct composed provider has incompatible dimensions")
        if not direct_provider.supports_order(order):
            raise ValidationError("direct composed provider does not supply the requested order")
        direct = direct_provider.derivative(order, point, direction_tuple)
        try:
            difference = direct - total
        except TypeError:
            difference = np.asarray(direct, dtype=object).reshape(-1) - np.asarray(total, dtype=object).reshape(-1)
        exactness = _combine_exactness(exactness, direct_provider.exactness)
        audit = value_audit(exact=exactness is Exactness.EXACT, tolerance=tolerance)
        residual = audit.check_value("direct_composition_derivative", difference)
        # Agreement at this probe does not prove either provider globally correct.
    valid = audit.valid
    return FaaDiBrunoResult(
        total,
        tuple(terms),
        len(set_partitions(order)),
        len(set_partitions(order)),
        0,
        Certificate(
            valid=valid,
            exactness=exactness,
            residuals={"direct_composition_derivative": residual} if direct_provider is not None else {},
            tolerance=None if exactness is Exactness.EXACT else tolerance,
            assumptions_checked=("set partition enumeration", "higher chain rule assembly"),
            assumptions_unchecked=tuple(assumptions_unchecked) + audit.unresolved,
            diagnostics={"order": order, "partition_count": len(set_partitions(order)), "residual_decisions": audit.decisions},
            theorem_id="PSC-III-T10",
            message="Faà di Bruno composition assembled" if valid else "Faà di Bruno comparison failed",
        ),
    )


def _support_zero(
    support: Mapping[int, PacketJetSupport] | None,
    order: int,
    output: str,
    inputs: tuple[str, ...],
) -> bool:
    if support is None or order not in support:
        return False
    address = PacketJetAddress(output, inputs)
    return address in support[order].zero


@documented(
    ProvenanceRecord(
        identifier="PSC-III-T11A",
        title="Packet Faà di Bruno formula",
        status=ClaimStatus.PROVED,
        source="Paper III",
        assumptions=("complete finite packet systems", "declared higher derivatives"),
        verification_suite="nonlinear.composition",
    )
)
def packet_faa_di_bruno(
    outer: DerivativeProvider,
    inner: DerivativeProvider,
    point: Any,
    directions: Sequence[Any],
    source: PacketSystem,
    intermediate: PacketSystem,
    target: PacketSystem,
    *,
    input_packets: Sequence[PacketLabel],
    output_packet: PacketLabel,
    inner_support: Mapping[int, PacketJetSupport] | None = None,
    outer_support: Mapping[int, PacketJetSupport] | None = None,
    tolerance: float | None = None,
    retain_terms: bool = True,
    max_terms: int = 1_000_000,
) -> FaaDiBrunoResult:
    """Evaluate one packet jet of a composition through decorated partitions."""

    direction_tuple = tuple(directions)
    input_tuple = tuple(input_packets)
    order = len(direction_tuple)
    if order < 1 or len(input_tuple) != order:
        raise ValidationError("packet composition requires one input packet per direction")
    if source.backend.name != intermediate.backend.name or source.backend.name != target.backend.name:
        raise ValidationError("all packet systems in a composition must use one backend")
    if inner.source_dimension != source.dimension or inner.target_dimension != intermediate.dimension:
        raise DimensionError("inner provider dimensions do not match source and intermediate packets")
    if outer.source_dimension != intermediate.dimension or outer.target_dimension != target.dimension:
        raise DimensionError("outer provider dimensions do not match intermediate and target packets")
    output_index = target.index(str(output_packet)) if not isinstance(output_packet, int) else int(output_packet)
    input_indices = tuple(source.index(str(label)) if not isinstance(label, int) else int(label) for label in input_tuple)
    if output_index < 0 or output_index >= len(target) or any(index < 0 or index >= len(source) for index in input_indices):
        raise ValidationError("packet label is outside its packet system")
    raw_count = labeled_composition_term_count(order, len(intermediate))
    limit = int(max_terms)
    if isinstance(max_terms, bool) or limit != max_terms or limit < 1:
        raise ValidationError("max_terms must be a positive integer")
    if raw_count > limit:
        raise ValidationError(
            f"packet composition has {raw_count} labeled terms, above max_terms={limit}; "
            "supply structural support or raise the explicit limit"
        )

    tol = max(source.tolerance, intermediate.tolerance, target.tolerance) if tolerance is None else float(tolerance)
    inner_point = inner.value(point)
    outer_families = {
        blocks: PacketJetFamily(outer, inner_point, intermediate, target, blocks)
        for blocks in range(1, order + 1)
    }
    inner_families = {
        block_order: PacketJetFamily(inner, point, source, intermediate, block_order)
        for block_order in range(1, order + 1)
    }
    target_name = target.names[output_index]
    terms: list[FaaDiBrunoTerm] = []
    total: Any | None = None
    evaluated = 0
    pruned = 0
    for partition in set_partitions(order):
        for labels in product(range(len(intermediate)), repeat=len(partition)):
            label_names = tuple(intermediate.names[index] for index in labels)
            outer_inputs = label_names
            if _support_zero(outer_support, len(partition), target_name, outer_inputs):
                pruned += 1
                if retain_terms:
                    terms.append(FaaDiBrunoTerm(partition, label_names, None, True, "zero outer packet jet"))
                continue
            inner_values: list[Any] = []
            zero_inner = False
            for label, block in zip(labels, partition, strict=True):
                block_input_names = tuple(source.names[input_indices[index]] for index in block)
                if _support_zero(inner_support, len(block), intermediate.names[label], block_input_names):
                    zero_inner = True
                    break
                block_directions = tuple(direction_tuple[index] for index in block)
                block_inputs = tuple(input_indices[index] for index in block)
                inner_values.append(
                    inner_families[len(block)].jet(label, block_inputs).evaluate(block_directions)
                )
            if zero_inner:
                pruned += 1
                if retain_terms:
                    terms.append(FaaDiBrunoTerm(partition, label_names, None, True, "zero inner packet jet"))
                continue
            value = outer_families[len(partition)].jet(output_index, labels).evaluate(tuple(inner_values))
            evaluated += 1
            total = _add(total, value)
            if retain_terms:
                terms.append(FaaDiBrunoTerm(partition, label_names, value))

    if total is None:
        total = target.backend.zeros(target.dimension, 1)
        if target.backend.name != "sympy":
            total = np.asarray(total).reshape(-1)

    projected_directions = tuple(
        source.backend.matmul(source.projectors[index], source.backend.as_vector(direction))
        for index, direction in zip(input_indices, direction_tuple, strict=True)
    )
    classical = faa_di_bruno(
        outer,
        inner,
        point,
        projected_directions,
        tolerance=tol,
        retain_terms=False,
    )
    projected_classical = target.backend.matmul(
        target.projectors[output_index], target.backend.as_vector(classical.value)
    )
    packet_value = target.backend.as_vector(total)
    exactness = _combine_exactness(
        classical.certificate.exactness,
        Exactness.EXACT if source.backend.exact and intermediate.backend.exact and target.backend.exact else Exactness.NUMERICAL,
    )
    audit = value_audit(exact=exactness is Exactness.EXACT, tolerance=tol)
    residual = audit.check_value("packet_composition_reconstruction", target.backend.subtract(projected_classical, packet_value))
    parents_valid, parent_checked, parent_unchecked, parent_diagnostics = packet_certificate_dependencies(
        source=source, intermediate=intermediate, target=target
    )
    valid = parents_valid and audit.valid
    return FaaDiBrunoResult(
        total,
        tuple(terms),
        raw_count,
        evaluated,
        pruned,
        Certificate(
            valid=valid,
            exactness=exactness,
            residuals={"packet_composition_reconstruction": residual},
            tolerance=None if exactness is Exactness.EXACT else tol,
            assumptions_checked=parent_checked + (
                "set partition enumeration",
                "complete intermediate packet identity",
                "input and output packet projection",
                "declared zero supports used only for pruning",
            ),
            assumptions_unchecked=parent_unchecked + ("global correctness of declared derivative providers",) + audit.unresolved,
            diagnostics={
                **parent_diagnostics,
                "order": order,
                "residual_decisions": audit.decisions,
                "raw_terms": raw_count,
                "evaluated_terms": evaluated,
                "pruned_terms": pruned,
                "intermediate_packet_count": len(intermediate),
            },
            theorem_id="PSC-III-T11A",
            message="packet Faà di Bruno formula reconstructs the projected derivative" if valid else "packet composition reconstruction failed",
        ),
    )


@documented(
    ProvenanceRecord(
        identifier="PSC-III-D12",
        title="Composed derivative provider",
        status=ClaimStatus.DEFINITION,
        source="Paper III runtime",
        assumptions=("compatible finite dimensional derivative providers",),
        verification_suite="nonlinear.composition",
    )
)
class ComposedDerivativeProvider:
    """Expose a composition as another derivative provider."""

    def __init__(self, outer: DerivativeProvider, inner: DerivativeProvider, *, name: str | None = None) -> None:
        if inner.target_dimension != outer.source_dimension:
            raise DimensionError("inner target and outer source dimensions do not match")
        self.outer = outer
        self.inner = inner
        self.source_dimension = inner.source_dimension
        self.target_dimension = outer.target_dimension
        self.exactness = _combine_exactness(outer.exactness, inner.exactness)
        self.name = name or f"{outer.name} after {inner.name}"

    def supports_order(self, order: int) -> bool:
        n = int(order)
        if isinstance(order, bool) or n != order or n < 1:
            return False
        return all(self.inner.supports_order(value) and self.outer.supports_order(value) for value in range(1, n + 1))

    def value(self, point: Any) -> Any:
        return self.outer.value(self.inner.value(point))

    def derivative(self, order: int, point: Any, directions: Sequence[Any]) -> Any:
        n = int(order)
        direction_tuple = tuple(directions)
        if len(direction_tuple) != n:
            raise ValidationError(f"order {n} requires exactly {n} directions")
        if not self.supports_order(n):
            raise ValidationError("component providers do not supply all required derivative orders")
        return faa_di_bruno(
            self.outer,
            self.inner,
            point,
            direction_tuple,
            retain_terms=False,
        ).value


__all__ = [
    "ComposedDerivativeProvider",
    "FaaDiBrunoResult",
    "FaaDiBrunoTerm",
    "Partition",
    "faa_di_bruno",
    "labeled_composition_term_count",
    "packet_faa_di_bruno",
    "set_partitions",
    "stirling_second",
]
