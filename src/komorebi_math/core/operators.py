"""Packet resolved linear operators."""

from __future__ import annotations

from .._residuals import ResidualAudit, finite_tolerance
from .._snapshots import backend_matrix_snapshot

from dataclasses import dataclass
from typing import Any

from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .packets import PacketCoordinates, PacketSystem


@documented(
    ProvenanceRecord(
        identifier="PSC-I-T27",
        title="Transfer block algebra",
        status=ClaimStatus.PROVED,
        source="Paper I",
        assumptions=("finite source and target packet systems",),
        verification_suite="core.operators",
    )
)
@dataclass(frozen=True, slots=True)
class PacketOperator:
    """An operator represented by all target by source packet blocks."""

    source: PacketSystem
    target: PacketSystem
    blocks: tuple[tuple[Any, ...], ...]

    def __post_init__(self) -> None:
        if self.source.backend.name != self.target.backend.name:
            raise ValidationError("source and target packet systems must use one backend")
        if len(self.blocks) != len(self.target):
            raise DimensionError("block row count does not match target packet count")
        if any(len(row) != len(self.source) for row in self.blocks):
            raise DimensionError("block column count does not match source packet count")
        expected = (self.target.dimension, self.source.dimension)
        normalized_rows: list[tuple[Any, ...]] = []
        for row in self.blocks:
            normalized_row: list[Any] = []
            for block in row:
                snapshot = backend_matrix_snapshot(self.backend, block)
                if self.backend.shape(snapshot) != expected:
                    raise DimensionError(f"embedded block shape must be {expected}")
                normalized_row.append(snapshot)
            normalized_rows.append(tuple(normalized_row))
        object.__setattr__(self, "blocks", tuple(normalized_rows))

    @property
    def backend(self):
        return self.source.backend

    @classmethod
    def from_operator(
        cls,
        operator: Any,
        source: PacketSystem,
        target: PacketSystem | None = None,
    ) -> "PacketOperator":
        target_system = source if target is None else target
        if source.backend.name != target_system.backend.name:
            raise ValidationError("source and target packet systems must use one backend")
        backend = source.backend
        matrix = backend.as_matrix(operator)
        expected = (target_system.dimension, source.dimension)
        if backend.shape(matrix) != expected:
            raise DimensionError(f"operator shape must be {expected}")
        blocks = tuple(
            tuple(
                backend.matmul(
                    target_projector,
                    backend.matmul(matrix, source_projector),
                )
                for source_projector in source.projectors
            )
            for target_projector in target_system.projectors
        )
        return cls(source, target_system, blocks)

    def reconstruct(self) -> Any:
        total = self.backend.zeros(self.target.dimension, self.source.dimension)
        for row in self.blocks:
            for block in row:
                total = self.backend.add(total, block)
        return total

    def certificate(self) -> Certificate:
        source_certificate = self.source.certificate()
        target_certificate = self.target.certificate()
        matrix = self.reconstruct()
        reconstructed = PacketOperator.from_operator(matrix, self.source, self.target)
        tolerance = max(finite_tolerance(self.source.tolerance), finite_tolerance(self.target.tolerance))
        audit = ResidualAudit(self.backend, tolerance)
        residual = max(audit.check(f"block:{j}:{i}", self.backend.subtract(left, right))
                       for j, (left_row, right_row) in enumerate(zip(self.blocks, reconstructed.blocks, strict=True))
                       for i, (left, right) in enumerate(zip(left_row, right_row, strict=True)))
        valid = source_certificate.valid and target_certificate.valid and audit.valid
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT if self.backend.exact else Exactness.NUMERICAL,
            residuals={"block_reprojection": residual},
            tolerance=None if self.backend.exact else tolerance,
            assumptions_checked=("source packet axioms", "target packet axioms", "block shapes"),
            assumptions_unchecked=source_certificate.assumptions_unchecked + target_certificate.assumptions_unchecked + audit.unresolved,
            diagnostics={"source_packets": len(self.source), "target_packets": len(self.target),
                         "source_certificate": source_certificate.to_dict(),
                         "target_certificate": target_certificate.to_dict(), "residual_decisions": audit.decisions},
            theorem_id="PSC-I-T27",
            message="valid packet operator" if valid else "parent packet or block-support checks failed or are unresolved",
        )

    def apply(self, vector: Any) -> Any:
        value = self.source.backend.as_vector(vector)
        if self.source.backend.shape(value)[0] != self.source.dimension:
            raise DimensionError("input vector dimension does not match source packets")
        return self.backend.matmul(self.reconstruct(), value)

    def apply_coordinates(self, coordinates: PacketCoordinates) -> PacketCoordinates:
        if not coordinates.system.equivalent(self.source):
            raise ValidationError("coordinates belong to a different source packet system")
        outputs: list[Any] = []
        for row in self.blocks:
            total = self.backend.scale(0, self.backend.matmul(row[0], coordinates.values[0]))
            for block, value in zip(row, coordinates.values, strict=True):
                total = self.backend.add(total, self.backend.matmul(block, value))
            outputs.append(total)
        return PacketCoordinates(self.target, tuple(outputs))

    def compose(self, right: "PacketOperator") -> "PacketOperator":
        """Return ``self`` after ``right``."""
        if not right.target.equivalent(self.source):
            raise ValidationError("the intermediate packet systems are not equivalent")
        output_rows: list[tuple[Any, ...]] = []
        for gamma in range(len(self.target)):
            row: list[Any] = []
            for alpha in range(len(right.source)):
                total = self.backend.zeros(self.target.dimension, right.source.dimension)
                for beta in range(len(self.source)):
                    total = self.backend.add(
                        total,
                        self.backend.matmul(self.blocks[gamma][beta], right.blocks[beta][alpha]),
                    )
                row.append(total)
            output_rows.append(tuple(row))
        return PacketOperator(right.source, self.target, tuple(output_rows))

    def adjoint(self) -> "PacketOperator":
        """Re-resolve the ambient adjoint in the original, reversed systems.

        For oblique projectors, merely adjointing each embedded block gives a
        representation in the *adjointed* systems, not these original systems.
        Reprojection preserves the existing packet names and return-system API.
        """
        return PacketOperator.from_operator(
            self.backend.adjoint(self.reconstruct()), self.target, self.source)

    def tensor(self, other: "PacketOperator") -> "PacketOperator":
        source = self.source.tensor(other.source)
        target = self.target.tensor(other.target)
        blocks: list[tuple[Any, ...]] = []
        for beta in range(len(self.target)):
            for delta in range(len(other.target)):
                row: list[Any] = []
                for alpha in range(len(self.source)):
                    for gamma in range(len(other.source)):
                        row.append(self.backend.kron(self.blocks[beta][alpha], other.blocks[delta][gamma]))
                blocks.append(tuple(row))
        return PacketOperator(source, target, tuple(blocks))

    def is_endomorphism(self) -> bool:
        return self.source.equivalent(self.target)

    def power(self, exponent: int) -> "PacketOperator":
        if exponent < 0:
            raise ValueError("packet operator powers must be nonnegative")
        if not self.is_endomorphism():
            raise ValidationError("only packet endomorphisms can be raised to a power")
        if exponent == 0:
            return PacketOperator.from_operator(self.backend.eye(self.source.dimension), self.source)
        result = self
        for _ in range(1, exponent):
            result = result.compose(self)
        return result

    def block_norms(self) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(self.backend.norm(block) for block in row) for row in self.blocks)

    def is_block_diagonal(self, *, tolerance: float | None = None) -> bool:
        if not self.is_endomorphism() or len(self.source) != len(self.target):
            return False
        tol = max(self.source.tolerance, self.target.tolerance) if tolerance is None else tolerance
        return all(
            beta == alpha or self.backend.is_zero(self.blocks[beta][alpha], tolerance=tol)
            for beta in range(len(self.target))
            for alpha in range(len(self.source))
        )

    def support_mask(self, *, tolerance: float | None = None) -> tuple[tuple[bool, ...], ...]:
        tol = max(self.source.tolerance, self.target.tolerance) if tolerance is None else tolerance
        return tuple(
            tuple(not self.backend.is_zero(block, tolerance=tol) for block in row)
            for row in self.blocks
        )
