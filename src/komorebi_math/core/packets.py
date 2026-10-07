"""Exact finite packet systems and packet coordinates."""

from __future__ import annotations

from .._residuals import ResidualAudit, finite_tolerance
from .._snapshots import backend_matrix_snapshot

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..backends import LinearAlgebraBackend, get_backend
from ..certificates import Certificate, Exactness
from ..config import get_settings
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented


@dataclass(frozen=True, slots=True)
class PacketCoordinates:
    system: "PacketSystem"
    values: tuple[Any, ...]

    def reconstruct(self) -> Any:
        return self.system.synthesize(self.values)

    def support(self, *, tolerance: float | None = None) -> tuple[str, ...]:
        tol = self.system.tolerance if tolerance is None else tolerance
        return tuple(
            name
            for name, value in zip(self.system.names, self.values, strict=True)
            if not self.system.backend.is_zero(value, tolerance=tol)
        )

    def __getitem__(self, key: int | str) -> Any:
        if isinstance(key, int):
            return self.values[key]
        return self.values[self.system.index(key)]

    def as_dict(self) -> dict[str, Any]:
        return dict(zip(self.system.names, self.values, strict=True))


@documented(
    ProvenanceRecord(
        identifier="PSC-I-T20",
        title="Exact finite packet coordinates",
        status=ClaimStatus.PROVED,
        source="Paper I",
        assumptions=("finite complete family of pairwise annihilating idempotents",),
        verification_suite="core.packets",
    )
)
class PacketSystem:
    """A complete finite family of pairwise annihilating idempotents."""

    def __init__(
        self,
        projectors: Sequence[Any],
        *,
        names: Sequence[str] | None = None,
        backend: str | LinearAlgebraBackend | None = None,
        tolerance: float | None = None,
        validate: bool | None = None,
    ) -> None:
        if not projectors:
            raise ValidationError("a packet system requires at least one projector")
        self.backend = get_backend(backend)
        self.tolerance = finite_tolerance(
            get_settings().tolerance if tolerance is None else tolerance
        )
        self.projectors = tuple(backend_matrix_snapshot(self.backend, projector) for projector in projectors)
        shape = self.backend.shape(self.projectors[0])
        if len(shape) != 2 or shape[0] != shape[1]:
            raise DimensionError("packet projectors must be square")
        if any(self.backend.shape(projector) != shape for projector in self.projectors):
            raise DimensionError("all packet projectors must have one common size")
        self.dimension = shape[0]
        if names is None:
            self.names = tuple(f"P{index}" for index in range(len(self.projectors)))
        else:
            self.names = tuple(str(name) for name in names)
            if len(self.names) != len(self.projectors):
                raise ValidationError("packet names and projectors must have the same length")
            if len(set(self.names)) != len(self.names):
                raise ValidationError("packet names must be unique")
        should_validate = get_settings().validate_on_construction if validate is None else validate
        if should_validate:
            self.certificate().require()

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc

    def certificate(self) -> Certificate:
        audit = ResidualAudit(self.backend, self.tolerance)
        idempotence = annihilation = 0.0
        for index, projector in enumerate(self.projectors):
            residual = self.backend.subtract(self.backend.matmul(projector, projector), projector)
            idempotence = max(idempotence, audit.check(f"idempotence:{index}", residual))
            for other_index, other in enumerate(self.projectors):
                if index != other_index:
                    annihilation = max(annihilation, audit.check(
                        f"annihilation:{index}:{other_index}", self.backend.matmul(projector, other)))
        total = self.backend.zeros(self.dimension, self.dimension)
        for projector in self.projectors:
            total = self.backend.add(total, projector)
        completeness = audit.check("completeness", self.backend.subtract(total, self.backend.eye(self.dimension)))
        valid = audit.valid
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT if self.backend.exact else Exactness.NUMERICAL,
            residuals={"idempotence": idempotence, "annihilation": annihilation, "completeness": completeness},
            tolerance=None if self.backend.exact else self.tolerance,
            assumptions_checked=("common dimension", "idempotence", "annihilation", "completeness"),
            assumptions_unchecked=audit.unresolved,
            diagnostics={"backend": self.backend.name, "ambient_dimension": self.dimension,
                         "packet_count": len(self.projectors), "residual_decisions": audit.decisions},
            theorem_id="PSC-I-T20",
            message="valid packet system" if valid else "packet axioms failed or could not be established",
        )

    def hilbert_certificate(self) -> Certificate:
        packet = self.certificate()
        audit = ResidualAudit(self.backend, self.tolerance)
        self_adjoint = max(audit.check(
            f"self_adjointness:{index}",
            self.backend.subtract(projector, self.backend.adjoint(projector)))
            for index, projector in enumerate(self.projectors))
        valid = packet.valid and audit.valid
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT if self.backend.exact else Exactness.NUMERICAL,
            residuals={**packet.residuals, "self_adjointness": self_adjoint},
            tolerance=None if self.backend.exact else self.tolerance,
            assumptions_checked=packet.assumptions_checked + ("self-adjointness",),
            assumptions_unchecked=packet.assumptions_unchecked + audit.unresolved,
            diagnostics={"packet_certificate": packet.to_dict(), "residual_decisions": audit.decisions},
            theorem_id="PSC-I-T20",
            message="orthogonal Hilbert packet system" if valid else "packet or self-adjointness checks failed or are unresolved",
        )

    def analyze(self, vector: Any) -> PacketCoordinates:
        value = self.backend.as_vector(vector)
        if self.backend.shape(value)[0] != self.dimension:
            raise DimensionError("vector dimension does not match the packet system")
        coordinates = tuple(self.backend.matmul(projector, value) for projector in self.projectors)
        return PacketCoordinates(self, coordinates)

    def synthesize(self, coordinates: PacketCoordinates | Sequence[Any] | Mapping[str, Any]) -> Any:
        if isinstance(coordinates, PacketCoordinates):
            if not coordinates.system.equivalent(self):
                raise ValidationError("packet coordinates belong to a different packet system")
            values = coordinates.values
        elif isinstance(coordinates, Mapping):
            if set(coordinates) != set(self.names):
                raise ValidationError("coordinate keys must match packet names")
            values = tuple(coordinates[name] for name in self.names)
        else:
            values = tuple(coordinates)
        if len(values) != len(self.projectors):
            raise ValidationError("coordinate count does not match packet count")
        vectors = tuple(self.backend.as_vector(value) for value in values)
        if any(self.backend.shape(vector)[0] != self.dimension for vector in vectors):
            raise DimensionError("every embedded packet coordinate must have the ambient dimension")
        total = self.backend.scale(0, vectors[0])
        for vector in vectors:
            total = self.backend.add(total, vector)
        return total

    def support(self, vector: Any, *, tolerance: float | None = None) -> tuple[str, ...]:
        return self.analyze(vector).support(tolerance=tolerance)

    def packet_dimensions(self) -> dict[str, int]:
        return {
            name: self.backend.rank(projector, tolerance=self.tolerance)
            for name, projector in zip(self.names, self.projectors, strict=True)
        }

    def aggregate(
        self,
        groups: Mapping[str, Iterable[int | str]],
        *,
        validate: bool = True,
    ) -> "PacketSystem":
        used: list[int] = []
        aggregate_projectors: list[Any] = []
        aggregate_names: list[str] = []
        for aggregate_name, members in groups.items():
            indices = [member if isinstance(member, int) else self.index(member) for member in members]
            if not indices:
                raise ValidationError("every aggregate packet requires at least one member")
            used.extend(indices)
            total = self.backend.scale(0, self.projectors[indices[0]])
            for index in indices:
                total = self.backend.add(total, self.projectors[index])
            aggregate_projectors.append(total)
            aggregate_names.append(str(aggregate_name))
        if sorted(used) != list(range(len(self.projectors))):
            raise ValidationError("aggregate groups must partition all packet indices exactly once")
        return PacketSystem(
            aggregate_projectors,
            names=aggregate_names,
            backend=self.backend,
            tolerance=self.tolerance,
            validate=validate,
        )

    def tensor(self, other: "PacketSystem") -> "PacketSystem":
        if self.backend.name != other.backend.name:
            raise ValidationError("tensor packet systems must use one backend")
        projectors: list[Any] = []
        names: list[str] = []
        for left_name, left in zip(self.names, self.projectors, strict=True):
            for right_name, right in zip(other.names, other.projectors, strict=True):
                projectors.append(self.backend.kron(left, right))
                names.append(f"{left_name}⊗{right_name}")
        return PacketSystem(
            projectors,
            names=names,
            backend=self.backend,
            tolerance=max(self.tolerance, other.tolerance),
        )

    def dual(self) -> "PacketSystem":
        return PacketSystem(
            [self.backend.transpose(projector) for projector in self.projectors],
            names=[f"{name}*" for name in self.names],
            backend=self.backend,
            tolerance=self.tolerance,
        )

    def equivalent(self, other: "PacketSystem", *, tolerance: float | None = None) -> bool:
        if self.backend.name != other.backend.name:
            return False
        if self.dimension != other.dimension or len(self.projectors) != len(other.projectors):
            return False
        if self.names != other.names:
            return False
        tol = max(self.tolerance, other.tolerance) if tolerance is None else tolerance
        return all(
            self.backend.is_zero(self.backend.subtract(left, right), tolerance=tol)
            for left, right in zip(self.projectors, other.projectors, strict=True)
        )


    def transport(self, intertwiner: Any) -> "PacketSystem":
        """Conjugate every packet projector by an invertible coordinate map."""
        matrix = self.backend.as_matrix(intertwiner)
        if self.backend.shape(matrix) != (self.dimension, self.dimension):
            raise DimensionError("coordinate transport must be an invertible square matrix")
        inverse = self.backend.inverse(matrix)
        return PacketSystem(
            [
                self.backend.matmul(matrix, self.backend.matmul(projector, inverse))
                for projector in self.projectors
            ],
            names=self.names,
            backend=self.backend,
            tolerance=self.tolerance,
        )

    def restrict(self, inclusion: Any, left_inverse: Any) -> "PacketSystem":
        """Restrict packets to an invariant subspace with declared splitting maps.

        ``inclusion`` maps the smaller space into the ambient space and
        ``left_inverse`` maps back. They must satisfy ``left_inverse @ inclusion = I``.
        """
        j = self.backend.as_matrix(inclusion)
        r = self.backend.as_matrix(left_inverse)
        if self.backend.shape(j)[0] != self.dimension:
            raise DimensionError("subspace inclusion has the wrong ambient dimension")
        subdimension = self.backend.shape(j)[1]
        if self.backend.shape(r) != (subdimension, self.dimension):
            raise DimensionError("left inverse has incompatible shape")
        split_residual = self.backend.subtract(
            self.backend.matmul(r, j), self.backend.eye(subdimension)
        )
        if not self.backend.is_zero(split_residual, tolerance=self.tolerance):
            raise ValidationError("left_inverse @ inclusion must be the identity")
        compressed = [
            self.backend.matmul(r, self.backend.matmul(projector, j))
            for projector in self.projectors
        ]
        for projector, small in zip(self.projectors, compressed, strict=True):
            residual = self.backend.subtract(
                self.backend.matmul(projector, j),
                self.backend.matmul(j, small),
            )
            if not self.backend.is_zero(residual, tolerance=self.tolerance):
                raise ValidationError("the declared subspace is not invariant under every packet")
        return PacketSystem(
            compressed,
            names=self.names,
            backend=self.backend,
            tolerance=self.tolerance,
        )

    def quotient(self, quotient_map: Any, right_inverse: Any) -> "PacketSystem":
        """Descend packets through an invariant quotient with a declared section."""
        q = self.backend.as_matrix(quotient_map)
        s = self.backend.as_matrix(right_inverse)
        if self.backend.shape(q)[1] != self.dimension:
            raise DimensionError("quotient map has the wrong ambient dimension")
        quotient_dimension = self.backend.shape(q)[0]
        if self.backend.shape(s) != (self.dimension, quotient_dimension):
            raise DimensionError("right inverse has incompatible shape")
        split_residual = self.backend.subtract(
            self.backend.matmul(q, s), self.backend.eye(quotient_dimension)
        )
        if not self.backend.is_zero(split_residual, tolerance=self.tolerance):
            raise ValidationError("quotient_map @ right_inverse must be the identity")
        descended = [
            self.backend.matmul(q, self.backend.matmul(projector, s))
            for projector in self.projectors
        ]
        for projector, small in zip(self.projectors, descended, strict=True):
            residual = self.backend.subtract(
                self.backend.matmul(q, projector),
                self.backend.matmul(small, q),
            )
            if not self.backend.is_zero(residual, tolerance=self.tolerance):
                raise ValidationError("the quotient kernel is not invariant under every packet")
        return PacketSystem(
            descended,
            names=self.names,
            backend=self.backend,
            tolerance=self.tolerance,
        )

    def block_diagonal_projection(self, operator: Any) -> Any:
        matrix = self.backend.as_matrix(operator)
        if self.backend.shape(matrix) != (self.dimension, self.dimension):
            raise DimensionError("operator dimension does not match packet system")
        total = self.backend.zeros(self.dimension, self.dimension)
        for projector in self.projectors:
            total = self.backend.add(
                total,
                self.backend.matmul(projector, self.backend.matmul(matrix, projector)),
            )
        return total

    def __len__(self) -> int:
        return len(self.projectors)
