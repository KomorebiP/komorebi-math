"""Path expansions for packet transfer blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._residuals import ResidualAudit
from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import ValidationError
from .graphs import TransferGraph
from .operators import PacketOperator


@dataclass(frozen=True, slots=True)
class TransferPath:
    vertices: tuple[int, ...]
    product: Any

    @property
    def length(self) -> int:
        return len(self.vertices) - 1


def path_contributions(
    operator: PacketOperator,
    *,
    source: int,
    target: int,
    length: int,
    tolerance: float | None = None,
) -> tuple[TransferPath, ...]:
    """Return every nonzero structural path contribution of fixed length.

    "Nonzero structural" refers to the selected edges, not the full product.
    A product of nonzero blocks may vanish; such path products are retained.
    Numerical structural support uses the stated tolerance.
    """

    if not operator.is_endomorphism():
        raise ValidationError("path expansions require a packet endomorphism")
    graph = TransferGraph.from_packet_operator(operator, tolerance=tolerance)
    contributions: list[TransferPath] = []
    for path in graph.paths(source, target, length=length):
        if length == 0:
            product = operator.backend.eye(operator.source.dimension)
            product = operator.backend.matmul(operator.source.projectors[source], product)
        else:
            product = operator.blocks[path[1]][path[0]]
            for step in range(1, length):
                product = operator.backend.matmul(
                    operator.blocks[path[step + 1]][path[step]], product
                )
        contributions.append(TransferPath(path, product))
    return tuple(contributions)


def block_from_paths(
    operator: PacketOperator,
    *,
    source: int,
    target: int,
    length: int,
    tolerance: float | None = None,
) -> ComputationResult[Any]:
    """Reconstruct one block of an operator power from packet paths."""

    contributions = path_contributions(
        operator,
        source=source,
        target=target,
        length=length,
        tolerance=tolerance,
    )
    total = operator.backend.zeros(operator.target.dimension, operator.source.dimension)
    for contribution in contributions:
        total = operator.backend.add(total, contribution.product)
    expected = operator.power(length).blocks[target][source]
    tol = operator.source.tolerance if tolerance is None else tolerance
    audit = ResidualAudit(operator.backend, tol)
    residual = audit.check("path_sum", operator.backend.subtract(total, expected))
    valid = audit.valid
    certificate = Certificate(
        valid=valid,
        exactness=Exactness.EXACT if operator.backend.exact else Exactness.NUMERICAL,
        residuals={"path_sum": residual},
        tolerance=None if operator.backend.exact else tol,
        assumptions_checked=("packet endomorphism", "fixed path length"),
        assumptions_unchecked=audit.unresolved,
        diagnostics={"path_count": len(contributions), "length": length, "residual_decisions": audit.decisions},
        theorem_id="PSC-I-T29",
        message="path expansion agrees with the operator power" if valid else "path expansion mismatch",
    )
    return ComputationResult(total, certificate)
