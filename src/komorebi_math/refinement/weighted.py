"""Weighted packet transfers and filtered inverse expansions."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import inf
from typing import Any

from ..backends import LinearAlgebraBackend, get_backend
from .._residuals import ResidualAudit
from .._snapshots import backend_matrix_snapshot
from ..certificates import Certificate, ComputationResult, Exactness
from ..core.packets import PacketSystem
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .._number_theory import require_prime
from .smith import rational_padic_valuation


@dataclass(frozen=True, slots=True)
class WeightedTransferEdge:
    """One packet supported transfer with a positive filtration weight."""

    source: int
    target: int
    weight: int
    operator: Any
    name: str = ""

    def __post_init__(self) -> None:
        if self.source < 0 or self.target < 0:
            raise ValueError("packet indices must be nonnegative")
        if self.weight <= 0:
            raise ValueError("transfer weights must be positive")


@dataclass(frozen=True, slots=True)
class WeightedPath:
    """A directed path together with its total filtration weight."""

    vertices: tuple[int, ...]
    edges: tuple[int, ...]
    weight: int

    @property
    def length(self) -> int:
        return len(self.edges)


@dataclass(frozen=True, slots=True)
class FilteredInverseExpansion:
    """A finite truncation of the weighted inverse word expansion."""

    value: Any
    precision: int
    included_paths: tuple[WeightedPath, ...]
    omitted_minimum_weight: int
    certificate: Certificate

    def require(self) -> Any:
        self.certificate.require()
        return self.value


@documented(
    ProvenanceRecord(
        identifier="PSC-II-T10",
        title="Weighted inverse expansion",
        status=ClaimStatus.PROVED,
        source="Paper II",
        assumptions=("invertible packet diagonal", "positive declared transfer weights"),
        verification_suite="refinement.weighted",
    )
)
class WeightedTransferGraph:
    """A packet diagonal operator plus weighted directed transfer blocks."""

    def __init__(
        self,
        packet_system: PacketSystem,
        diagonal: Any,
        edges: tuple[WeightedTransferEdge, ...] | list[WeightedTransferEdge],
        *,
        backend: str | LinearAlgebraBackend | None = None,
        prime: int | None = None,
        validate_weights: bool = True,
    ) -> None:
        self.packet_system = packet_system
        self.backend = packet_system.backend if backend is None else get_backend(backend)
        if self.backend.name != packet_system.backend.name:
            raise ValidationError("weighted transfer graph and packet system must use one backend")
        self.diagonal = backend_matrix_snapshot(self.backend, diagonal)
        expected = (packet_system.dimension, packet_system.dimension)
        if self.backend.shape(self.diagonal) != expected:
            raise DimensionError(f"diagonal operator shape must be {expected}")
        self.edges = tuple(
            WeightedTransferEdge(
                edge.source, edge.target, edge.weight,
                backend_matrix_snapshot(self.backend, edge.operator), edge.name
            )
            for edge in edges
        )
        self.prime = None if prime is None else require_prime(prime)
        self._diagonal_inverse = self.backend.inverse(self.diagonal)
        self._validate_structure(validate_weights=validate_weights)
        if validate_weights and self.prime is not None and self.backend.exact:
            for label, matrix in (("diagonal", self.diagonal), ("diagonal inverse", self._diagonal_inverse)):
                minimum = min((rational_padic_valuation(entry, self.prime) for entry in matrix), default=0)
                if minimum < 0:
                    raise ValidationError(f"{label} does not lie in filtration level zero")

    def _validate_structure(self, *, validate_weights: bool) -> None:
        tolerance = self.packet_system.tolerance
        diagonal_residual = self.backend.subtract(
            self.diagonal,
            self.packet_system.block_diagonal_projection(self.diagonal),
        )
        if not self.backend.is_zero(diagonal_residual, tolerance=tolerance):
            raise ValidationError("declared diagonal operator is not packet diagonal")
        for edge in self.edges:
            if edge.source >= len(self.packet_system) or edge.target >= len(self.packet_system):
                raise ValueError("transfer edge references an unknown packet")
            matrix = self.backend.as_matrix(edge.operator)
            if self.backend.shape(matrix) != self.backend.shape(self.diagonal):
                raise DimensionError("every transfer edge must use the ambient operator shape")
            supported = self.backend.matmul(
                self.packet_system.projectors[edge.target],
                self.backend.matmul(matrix, self.packet_system.projectors[edge.source]),
            )
            if not self.backend.is_zero(self.backend.subtract(matrix, supported), tolerance=tolerance):
                raise ValidationError("a transfer edge is not supported on its declared source and target packets")
            if validate_weights and self.prime is not None and self.backend.exact:
                minimum = inf
                for entry in matrix:
                    minimum = min(minimum, rational_padic_valuation(entry, self.prime))
                if minimum < edge.weight:
                    raise ValidationError(
                        f"edge {edge.name or (edge.source, edge.target)} has valuation {minimum}, below declared weight {edge.weight}"
                    )

    @property
    def dimension(self) -> int:
        return self.packet_system.dimension

    @property
    def operator(self) -> Any:
        total = self.backend.copy(self.diagonal)
        for edge in self.edges:
            total = self.backend.add(total, self.backend.as_matrix(edge.operator))
        return total

    @property
    def diagonal_inverse(self) -> Any:
        return self.backend.copy(self._diagonal_inverse)

    def adjacency(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        rows: list[list[tuple[int, int]]] = [[] for _ in range(len(self.packet_system))]
        for index, edge in enumerate(self.edges):
            rows[edge.source].append((edge.target, index))
        return tuple(tuple(row) for row in rows)

    def paths_below_weight(
        self,
        precision: int,
        *,
        source: int | None = None,
        target: int | None = None,
        max_paths: int | None = 1_000_000,
    ) -> tuple[WeightedPath, ...]:
        """Enumerate all nonempty directed paths of total weight below ``precision``."""

        if precision <= 0:
            return ()
        if max_paths is not None and max_paths <= 0:
            raise ValueError("maximum path count must be positive or None")
        packet_count = len(self.packet_system)
        if source is not None and not 0 <= source < packet_count:
            raise IndexError("source packet is outside the graph")
        if target is not None and not 0 <= target < packet_count:
            raise IndexError("target packet is outside the graph")
        adjacency = self.adjacency()
        starts = range(packet_count) if source is None else (source,)
        output: list[WeightedPath] = []
        generated = 0
        stack: list[tuple[int, tuple[int, ...], tuple[int, ...], int]] = [
            (start, (start,), (), 0) for start in starts
        ]
        while stack:
            vertex, vertices, edge_indices, weight = stack.pop()
            for next_vertex, edge_index in adjacency[vertex]:
                edge = self.edges[edge_index]
                next_weight = weight + edge.weight
                if next_weight >= precision:
                    continue
                next_vertices = vertices + (next_vertex,)
                next_edges = edge_indices + (edge_index,)
                path = WeightedPath(next_vertices, next_edges, next_weight)
                generated += 1
                if max_paths is not None and generated > max_paths:
                    raise ValueError(
                        f"weighted path enumeration exceeds the declared limit {max_paths}"
                    )
                if target is None or next_vertex == target:
                    output.append(path)
                stack.append((next_vertex, next_vertices, next_edges, next_weight))
        output.sort(key=lambda path: (path.weight, path.length, path.vertices, path.edges))
        return tuple(output)

    def minimum_path_weight(self, source: int, target: int) -> int | float:
        """Return the least total transfer weight between two packets."""

        packet_count = len(self.packet_system)
        if not 0 <= source < packet_count or not 0 <= target < packet_count:
            raise IndexError("packet index is outside the graph")
        if source == target:
            return 0
        adjacency = self.adjacency()
        distances = [inf] * packet_count
        distances[source] = 0
        queue: list[tuple[int, int]] = [(0, source)]
        while queue:
            distance, vertex = heappop(queue)
            if distance != distances[vertex]:
                continue
            if vertex == target:
                return distance
            for next_vertex, edge_index in adjacency[vertex]:
                candidate = distance + self.edges[edge_index].weight
                if candidate < distances[next_vertex]:
                    distances[next_vertex] = candidate
                    heappush(queue, (candidate, next_vertex))
        return inf

    def _normalized_edge(self, edge_index: int) -> Any:
        return self.backend.matmul(self._diagonal_inverse, self.backend.as_matrix(self.edges[edge_index].operator))

    def path_operator(self, path: WeightedPath) -> Any:
        """Return ``A_last ... A_first D_inverse`` for one path."""

        product = self.backend.copy(self._diagonal_inverse)
        for edge_index in path.edges:
            product = self.backend.matmul(self._normalized_edge(edge_index), product)
        return product

    def truncated_inverse(
        self,
        precision: int,
        *,
        max_paths: int | None = 1_000_000,
    ) -> FilteredInverseExpansion:
        """Return the inverse word expansion below one filtration depth.

        With an exact prime valuation backend, the method verifies the stated
        congruence against the ordinary inverse.  Without such a backend, the
        returned certificate is conditional on the declared weights defining
        a complete multiplicative filtration.  The conditional path does not
        require the evaluated finite matrix to be invertible.
        """

        if precision <= 0:
            raise ValueError("precision must be positive")
        total = self.backend.copy(self._diagonal_inverse)
        paths = self.paths_below_weight(precision, max_paths=max_paths)
        for path in paths:
            contribution = self.path_operator(path)
            total = self.backend.add(total, self.backend.scale((-1) ** path.length, contribution))

        residuals: dict[str, float] = {}
        diagnostics: dict[str, Any] = {
            "included_paths": len(paths),
            "precision": precision,
            "omitted_weight_lower_bound": precision,
        }
        assumptions_unchecked: tuple[str, ...] = ()
        if self.prime is not None and self.backend.exact:
            exact_inverse = self.backend.inverse(self.operator)
            difference = self.backend.subtract(exact_inverse, total)
            residual_norm = self.backend.norm(difference)
            minimum = min(
                (rational_padic_valuation(entry, self.prime) for entry in difference),
                default=inf,
            )
            valid = minimum >= precision
            exactness = Exactness.EXACT
            residuals["ordinary_matrix_residual"] = residual_norm
            diagnostics["minimum_omitted_valuation"] = minimum
            diagnostics["ordinary_inverse_available"] = True
        else:
            valid = True
            exactness = Exactness.CONDITIONAL
            assumptions_unchecked = (
                "declared edge weights realize a complete multiplicative filtration",
            )
            try:
                ordinary_inverse = self.backend.inverse(self.operator)
            except Exception:
                diagnostics["ordinary_inverse_available"] = False
            else:
                difference = self.backend.subtract(ordinary_inverse, total)
                residuals["ordinary_matrix_residual"] = self.backend.norm(difference)
                diagnostics["ordinary_inverse_available"] = True

        certificate = Certificate(
            valid=bool(valid),
            exactness=exactness,
            residuals=residuals,
            tolerance=None,
            assumptions_checked=("packet support", "positive path weights", "invertible packet diagonal"),
            assumptions_unchecked=assumptions_unchecked,
            diagnostics=diagnostics,
            theorem_id="PSC-II-T10",
            message=(
                f"inverse is correct modulo filtration depth {precision}"
                if exactness is Exactness.EXACT and valid
                else "weighted inverse truncation is available under the declared filtration assumptions"
                if valid
                else "weighted inverse truncation failed the declared valuation check"
            ),
        )
        return FilteredInverseExpansion(total, precision, paths, precision, certificate)

    def exact_inverse_if_acyclic(
        self,
        *,
        max_paths: int | None = 1_000_000,
    ) -> ComputationResult[Any]:
        """Return the finite path inverse when the packet graph is acyclic."""

        if not self.is_acyclic():
            raise ValidationError("the packet transfer graph contains a directed cycle")
        max_length = max(len(self.packet_system) - 1, 0)
        total = self.backend.copy(self._diagonal_inverse)
        all_paths: list[WeightedPath] = []
        # Any path in an acyclic graph has at most packet_count - 1 edges.
        precision = 1 + sum(sorted((edge.weight for edge in self.edges), reverse=True)[:max_length])
        for path in self.paths_below_weight(precision, max_paths=max_paths):
            if path.length <= max_length:
                all_paths.append(path)
                total = self.backend.add(
                    total,
                    self.backend.scale((-1) ** path.length, self.path_operator(path)),
                )
        exact = self.backend.inverse(self.operator)
        audit = ResidualAudit(self.backend, self.packet_system.tolerance)
        residual = audit.check("inverse_residual", self.backend.subtract(total, exact))
        valid = audit.valid
        return ComputationResult(
            total,
            Certificate(
                valid=valid,
                exactness=Exactness.EXACT if self.backend.exact else Exactness.NUMERICAL,
                residuals={"inverse_residual": residual},
                tolerance=None if self.backend.exact else self.packet_system.tolerance,
                assumptions_checked=("acyclic transfer graph", "invertible packet diagonal"),
                assumptions_unchecked=audit.unresolved,
                diagnostics={"paths": len(all_paths), "maximum_length": max_length, "residual_decisions": audit.decisions},
                theorem_id="PSC-II-C11",
                message="finite acyclic path inverse" if valid else "finite path sum did not reproduce the inverse",
            ),
        )

    def is_acyclic(self) -> bool:
        packet_count = len(self.packet_system)
        adjacency = self.adjacency()
        state = [0] * packet_count

        def visit(vertex: int) -> bool:
            state[vertex] = 1
            for next_vertex, _ in adjacency[vertex]:
                if state[next_vertex] == 1:
                    return False
                if state[next_vertex] == 0 and not visit(next_vertex):
                    return False
            state[vertex] = 2
            return True

        return all(state[vertex] != 0 or visit(vertex) for vertex in range(packet_count))

    def weighted_adjacency(self, variable: Any) -> Any:
        """Return the scalar weighted adjacency matrix ``sum variable**weight``."""

        if self.backend.name == "sympy":
            import sympy as sp

            matrix = sp.zeros(len(self.packet_system))
        else:
            import numpy as np

            matrix = np.zeros((len(self.packet_system), len(self.packet_system)), dtype=complex)
        for edge in self.edges:
            matrix[edge.target, edge.source] += variable**edge.weight
        return matrix

    def transfer_series_matrix(self, variable: Any) -> Any:
        """Return ``(I - A(variable))**-1`` for the finite weighted graph."""

        adjacency = self.weighted_adjacency(variable)
        if self.backend.name == "sympy":
            import sympy as sp

            return (sp.eye(len(self.packet_system)) - adjacency).inv()
        import numpy as np

        return np.linalg.inv(np.eye(len(self.packet_system), dtype=complex) - adjacency)


__all__ = [
    "FilteredInverseExpansion",
    "WeightedPath",
    "WeightedTransferEdge",
    "WeightedTransferGraph",
]
