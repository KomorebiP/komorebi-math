"""Matrix realizations of typed operator grammars."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..backends import LinearAlgebraBackend, get_backend
from .._snapshots import backend_matrix_snapshot
from ..certificates import Certificate, Exactness
from ..config import get_settings
from ..exceptions import DimensionError, RelationError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .grammar import OperatorGrammar, OperatorRelation, OperatorWord


@dataclass(frozen=True, slots=True)
class RelationResidual:
    name: str
    residual: float
    satisfied: bool


@documented(
    ProvenanceRecord(
        identifier="PSC-I-T01",
        title="Typed realization and quotient path algebra correspondence",
        status=ClaimStatus.PROVED,
        source="Paper I",
        assumptions=("finite dimensional matrix realization",),
        verification_suite="core.realization",
    )
)
class OperatorRealization:
    """Assign vector space dimensions and matrices to a typed grammar."""

    def __init__(
        self,
        grammar: OperatorGrammar,
        spaces: Mapping[str, int],
        operators: Mapping[str, Any],
        *,
        backend: str | LinearAlgebraBackend | None = None,
        validate_relations: bool = True,
        tolerance: float | None = None,
    ) -> None:
        self.grammar = grammar
        self.backend = get_backend(backend)
        self.tolerance = get_settings().tolerance if tolerance is None else float(tolerance)
        self.spaces = {name: int(dimension) for name, dimension in spaces.items()}
        if set(self.spaces) != set(grammar.vertices):
            missing = set(grammar.vertices) - set(self.spaces)
            extra = set(self.spaces) - set(grammar.vertices)
            raise ValidationError(f"space keys must match grammar vertices, missing={missing}, extra={extra}")
        if any(dimension <= 0 for dimension in self.spaces.values()):
            raise DimensionError("every realized space must have positive dimension")
        if set(operators) != {arrow.name for arrow in grammar.arrows}:
            required = {arrow.name for arrow in grammar.arrows}
            missing = required - set(operators)
            extra = set(operators) - required
            raise ValidationError(f"operator keys must match grammar arrows, missing={missing}, extra={extra}")

        self.operators: dict[str, Any] = {}
        for arrow in grammar.arrows:
            matrix = backend_matrix_snapshot(self.backend, operators[arrow.name])
            expected = (self.spaces[arrow.target], self.spaces[arrow.source])
            if self.backend.shape(matrix) != expected:
                raise DimensionError(
                    f"operator {arrow.name!r} has shape {self.backend.shape(matrix)}, expected {expected}"
                )
            self.operators[arrow.name] = matrix
        if validate_relations and grammar.relations:
            certificate = self.relation_certificate()
            if not certificate.valid:
                raise RelationError(certificate.message)

    def evaluate(self, word: OperatorWord) -> Any:
        if word.is_identity:
            return self.backend.eye(self.spaces[word.source])
        reconstructed = self.grammar.word(*word.arrows)
        if reconstructed.source != word.source or reconstructed.target != word.target:
            raise ValidationError("word endpoints do not agree with the grammar")
        current = self.backend.eye(self.spaces[word.source])
        for arrow_name in word.arrows:
            current = self.backend.matmul(self.operators[arrow_name], current)
        return current

    def relation_matrix(self, relation: OperatorRelation) -> Any:
        rows = self.spaces[relation.target]
        cols = self.spaces[relation.source]
        total = self.backend.zeros(rows, cols)
        for coefficient, word in relation.terms:
            total = self.backend.add(total, self.backend.scale(coefficient, self.evaluate(word)))
        return total

    def relation_residuals(self) -> tuple[RelationResidual, ...]:
        residuals: list[RelationResidual] = []
        for relation in self.grammar.relations:
            matrix = self.relation_matrix(relation)
            value = self.backend.norm(matrix)
            satisfied = self.backend.is_zero(matrix, tolerance=self.tolerance)
            residuals.append(RelationResidual(relation.name, value, satisfied))
        return tuple(residuals)

    def relation_certificate(self) -> Certificate:
        residual_records = self.relation_residuals()
        valid = all(record.satisfied for record in residual_records)
        residuals = {record.name: record.residual for record in residual_records}
        exactness = Exactness.EXACT if self.backend.exact else Exactness.NUMERICAL
        return Certificate(
            valid=valid,
            exactness=exactness,
            residuals=residuals,
            tolerance=None if self.backend.exact else self.tolerance,
            assumptions_checked=("operator shapes", "typed path composition", "declared relations"),
            diagnostics={"backend": self.backend.name, "relation_count": len(residual_records)},
            theorem_id="PSC-I-T01",
            message="all grammar relations are satisfied" if valid else "one or more grammar relations fail",
        )

    def path_operators(
        self,
        *,
        source: str | None = None,
        target: str | None = None,
        max_length: int = 1,
        include_identities: bool = True,
    ) -> dict[OperatorWord, Any]:
        return {
            word: self.evaluate(word)
            for word in self.grammar.paths(
                source=source,
                target=target,
                max_length=max_length,
                include_identities=include_identities,
            )
        }

    def loop_operators(self, vertex: str, *, max_length: int) -> tuple[Any, ...]:
        return tuple(
            self.evaluate(word)
            for word in self.grammar.paths(
                source=vertex,
                target=vertex,
                max_length=max_length,
                include_identities=True,
            )
        )
