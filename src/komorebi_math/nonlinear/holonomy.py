"""Finite matrix transport, gauges, and holonomy certificates."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite
from typing import Any, Iterable, Mapping

from ..backends import LinearAlgebraBackend, get_backend
from .._residuals import ResidualAudit, finite_tolerance
from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented


@dataclass(frozen=True, slots=True)
class TransportEdge:
    """An invertible transport from ``source`` to ``target``."""

    source: str
    target: str
    matrix: Any
    name: str = ""

    def reversed_name(self) -> str:
        return self.name or f"{self.target}->{self.source}"


@dataclass(frozen=True, slots=True)
class FundamentalHolonomy:
    """One non tree edge and its based cycle holonomy."""

    edge_index: int
    source: str
    target: str
    matrix: Any


@dataclass(frozen=True, slots=True)
class IntertwiningResult:
    """Vertex intertwiners propagated along a spanning tree."""

    intertwiners: Mapping[str, Any]
    edge_residuals: Mapping[str, float]
    cycle_residuals: Mapping[str, float]
    certificate: Certificate


@dataclass(frozen=True, slots=True)
class GaugeResidualEnvelope:
    """Condition-number transport bounds for numerical holonomy residuals.

    If a based holonomy residual is ``E = H - I`` and the root gauge is ``g``,
    then the transformed residual is ``g E g^{-1}``. For the Frobenius norm,

    ``||E||_F / kappa_2(g) <= ||g E g^{-1}||_F <= kappa_2(g) ||E||_F``.

    The envelope records this coordinate sensitivity explicitly. It is a
    numerical conditioning diagnostic, not an invariant distance on a full
    ``GL_n`` gauge quotient.
    """

    root: str
    condition_number: float
    source_residuals: Mapping[str, float]
    lower_bounds: Mapping[str, float]
    upper_bounds: Mapping[str, float]
    metric: str = "frobenius"
    gauge_regime: str = "full_gl"


def _numeric_root_metric(value: Any, *, dimension: int, tolerance: float) -> Any:
    """Validate one positive-definite Hermitian metric for numerical use."""

    import numpy as np

    metric = np.asarray(value, dtype=complex)
    if metric.shape != (dimension, dimension):
        raise DimensionError("root metric has the wrong fiber dimension")
    if not np.all(np.isfinite(metric)):
        raise ValidationError("root metric must contain finite values")
    scale = max(1.0, float(np.linalg.norm(metric, ord=2)))
    if float(np.linalg.norm(metric - metric.conj().T, ord="fro")) > tolerance * scale:
        raise ValidationError("root metric must be Hermitian within the graph tolerance")
    hermitian = 0.5 * (metric + metric.conj().T)
    eigenvalues = np.linalg.eigvalsh(hermitian)
    if eigenvalues.size == 0 or float(eigenvalues[0]) <= tolerance * scale:
        raise ValidationError(
            "root metric must be positive definite and numerically resolved at the graph tolerance"
        )
    return hermitian


_GLUING_GAUGE_REGIMES = frozenset({"coordinate", "unitary", "full_gl"})
_TRANSFORM_GAUGE_REGIMES = frozenset({"full_gl", "unitary"})


def _normalize_gauge_regime(value: str, *, transform: bool = False) -> str:
    regime = str(value).strip().lower()
    allowed = _TRANSFORM_GAUGE_REGIMES if transform else _GLUING_GAUGE_REGIMES
    if regime not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValidationError(f"gauge_regime must be one of: {choices}")
    return regime


@documented(
    ProvenanceRecord(
        identifier="PSC-III-D38",
        title="Finite matrix transport system",
        status=ClaimStatus.DEFINITION,
        source="Paper III",
        assumptions=("connected finite graph", "invertible edge transports", "one model fiber dimension"),
        verification_suite="nonlinear.holonomy",
    )
)
class TransportGraph:
    """A connected finite graph with invertible matrix transports."""

    def __init__(
        self,
        vertices: Iterable[str],
        edges: Iterable[TransportEdge],
        *,
        backend: str | LinearAlgebraBackend | None = None,
        tolerance: float = 1e-10,
    ) -> None:
        self.vertices = tuple(dict.fromkeys(str(vertex) for vertex in vertices))
        if not self.vertices:
            raise ValueError("transport graph requires at least one vertex")
        self.vertex_set = frozenset(self.vertices)
        self.backend = get_backend(backend)
        self.tolerance = finite_tolerance(tolerance)
        self.edges = tuple(edges)
        if not self.edges and len(self.vertices) > 1:
            raise ValidationError("a multivertex transport graph needs edges")
        dimensions: set[int] = set()
        for edge in self.edges:
            if edge.source not in self.vertex_set or edge.target not in self.vertex_set:
                raise ValidationError("transport edge references an unknown vertex")
            matrix = self.backend.as_matrix(edge.matrix)
            rows, columns = self.backend.shape(matrix)
            if rows != columns:
                raise DimensionError("transport matrices must be square")
            dimensions.add(rows)
            self.backend.inverse(matrix)
        if len(dimensions) > 1:
            raise DimensionError("all transport matrices must use one model fiber dimension")
        self.dimension = next(iter(dimensions), 1)
        if len(self._reachable(self.vertices[0])) != len(self.vertices):
            raise ValidationError("transport graph must be connected")

    def _oriented_adjacency(self) -> dict[str, list[tuple[str, int, bool]]]:
        adjacency = {vertex: [] for vertex in self.vertices}
        for index, edge in enumerate(self.edges):
            adjacency[edge.source].append((edge.target, index, False))
            adjacency[edge.target].append((edge.source, index, True))
        return adjacency

    def _reachable(self, root: str) -> frozenset[str]:
        adjacency = self._oriented_adjacency()
        seen = {root}
        stack = [root]
        while stack:
            vertex = stack.pop()
            for neighbor, _, _ in adjacency[vertex]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        return frozenset(seen)

    def edge_label(self, edge_index: int) -> str:
        """Return a stable unique label for one transport edge."""

        edge = self.edges[edge_index]
        return f"{edge.name or 'edge'}[{edge_index}]"

    def oriented_matrix(self, edge_index: int, *, reverse: bool = False) -> Any:
        matrix = self.backend.as_matrix(self.edges[edge_index].matrix)
        return self.backend.inverse(matrix) if reverse else matrix

    def spanning_tree(self, root: str | None = None) -> tuple[dict[str, tuple[str, int, bool] | None], tuple[int, ...]]:
        root_vertex = self.vertices[0] if root is None else root
        if root_vertex not in self.vertex_set:
            raise KeyError("unknown root vertex")
        adjacency = self._oriented_adjacency()
        parent: dict[str, tuple[str, int, bool] | None] = {root_vertex: None}
        tree_edges: list[int] = []
        queue = [root_vertex]
        while queue:
            vertex = queue.pop(0)
            for neighbor, edge_index, reverse in adjacency[vertex]:
                if neighbor in parent:
                    continue
                parent[neighbor] = (vertex, edge_index, reverse)
                tree_edges.append(edge_index)
                queue.append(neighbor)
        return parent, tuple(tree_edges)

    def tree_transports(self, root: str | None = None) -> dict[str, Any]:
        """Return transports from the root fiber to every vertex along one spanning tree."""

        root_vertex = self.vertices[0] if root is None else root
        parent, _ = self.spanning_tree(root_vertex)
        transports: dict[str, Any] = {root_vertex: self.backend.eye(self.dimension)}
        unresolved = set(self.vertices) - {root_vertex}
        while unresolved:
            progress = False
            for vertex in tuple(unresolved):
                relation = parent[vertex]
                if relation is None:
                    raise ArithmeticError("nonroot vertex has no tree parent")
                parent_vertex, edge_index, reverse = relation
                if parent_vertex not in transports:
                    continue
                step = self.oriented_matrix(edge_index, reverse=reverse)
                transports[vertex] = self.backend.matmul(step, transports[parent_vertex])
                unresolved.remove(vertex)
                progress = True
            if not progress:
                raise ArithmeticError("failed to propagate spanning tree transports")
        return transports

    @documented(
        ProvenanceRecord(
            identifier="PSC-III-T39",
            title="Spanning tree normalization",
            status=ClaimStatus.PROVED,
            source="Paper III",
            assumptions=("connected transport graph",),
            verification_suite="nonlinear.holonomy",
        )
    )
    def tree_gauge(self, root: str | None = None) -> dict[str, Any]:
        return {vertex: self.backend.inverse(matrix) for vertex, matrix in self.tree_transports(root).items()}

    def gauge_transform(
        self,
        gauges: Mapping[str, Any],
        *,
        gauge_regime: str = "full_gl",
    ) -> "TransportGraph":
        regime = _normalize_gauge_regime(gauge_regime, transform=True)
        if set(gauges) != set(self.vertices):
            raise ValidationError("a gauge matrix is required at every vertex")
        gauge_matrices = {
            vertex: self.backend.as_matrix(gauges[vertex])
            for vertex in self.vertices
        }
        for vertex, matrix in gauge_matrices.items():
            if self.backend.shape(matrix) != (self.dimension, self.dimension):
                raise DimensionError(f"gauge at vertex {vertex!r} has the wrong fiber dimension")
            self.backend.inverse(matrix)
            if regime == "unitary":
                unitarity_residual = self.backend.subtract(
                    self.backend.matmul(matrix, self.backend.adjoint(matrix)),
                    self.backend.eye(self.dimension),
                )
                if not bool(self.backend.is_zero(unitarity_residual, tolerance=self.tolerance)):
                    raise ValidationError(
                        f"gauge at vertex {vertex!r} is not unitary within the graph tolerance"
                    )
        transformed: list[TransportEdge] = []
        for edge in self.edges:
            left = gauge_matrices[edge.target]
            right_inverse = self.backend.inverse(gauge_matrices[edge.source])
            matrix = self.backend.matmul(left, self.backend.matmul(self.backend.as_matrix(edge.matrix), right_inverse))
            transformed.append(TransportEdge(edge.source, edge.target, matrix, edge.name))
        return TransportGraph(self.vertices, transformed, backend=self.backend, tolerance=self.tolerance)

    @documented(
        ProvenanceRecord(
            identifier="PSC-III-T40",
            title="Fundamental cycle holonomy criterion",
            status=ClaimStatus.PROVED,
            source="Paper III",
            assumptions=("connected finite graph",),
            verification_suite="nonlinear.holonomy",
        )
    )
    def fundamental_holonomies(self, root: str | None = None) -> tuple[FundamentalHolonomy, ...]:
        root_vertex = self.vertices[0] if root is None else root
        transports = self.tree_transports(root_vertex)
        _, tree_edges = self.spanning_tree(root_vertex)
        tree_set = set(tree_edges)
        output: list[FundamentalHolonomy] = []
        for index, edge in enumerate(self.edges):
            if index in tree_set:
                continue
            holonomy = self.backend.matmul(
                self.backend.inverse(transports[edge.target]),
                self.backend.matmul(self.backend.as_matrix(edge.matrix), transports[edge.source]),
            )
            output.append(FundamentalHolonomy(index, edge.source, edge.target, holonomy))
        return tuple(output)

    def gluing_certificate(
        self,
        root: str | None = None,
        *,
        gauge_regime: str = "coordinate",
    ) -> Certificate:
        regime = _normalize_gauge_regime(gauge_regime)
        if not self.backend.exact and regime == "full_gl":
            raise ValidationError(
                "a numerical tolerance certificate is not gauge invariant under unrestricted full_gl similarity; "
                "use gauge_regime='coordinate', gauge_regime='unitary', or an exact backend"
            )
        identity = self.backend.eye(self.dimension)
        audit = ResidualAudit(self.backend, self.tolerance)
        residuals = {
            self.edge_label(item.edge_index): audit.check(
                self.edge_label(item.edge_index), self.backend.subtract(item.matrix, identity)
            )
            for item in self.fundamental_holonomies(root)
        }
        valid = audit.valid
        if self.backend.exact:
            invariance_scope = "exact identity is invariant under full_gl similarity"
            residual_semantics = "exact zero predicate"
            message = (
                "all fundamental holonomies are trivial"
                if valid
                else "one or more fundamental holonomies are nontrivial"
            )
        elif regime == "unitary":
            invariance_scope = "Frobenius residual is invariant under unitary similarity"
            residual_semantics = "unitary-gauge-invariant Frobenius tolerance"
            message = (
                "all fundamental holonomies are within the unitary-invariant numerical tolerance"
                if valid
                else "one or more fundamental holonomies exceed the unitary-invariant numerical tolerance"
            )
        else:
            invariance_scope = "no gauge-invariance claim beyond the chosen rooted coordinates"
            residual_semantics = "coordinate-dependent Frobenius tolerance"
            message = (
                "all fundamental holonomies are within the coordinate tolerance"
                if valid
                else "one or more fundamental holonomies exceed the coordinate tolerance"
            )
        return Certificate(
            valid=bool(valid),
            exactness=Exactness.EXACT if self.backend.exact else Exactness.NUMERICAL,
            residuals=residuals,
            tolerance=None if self.backend.exact else self.tolerance,
            assumptions_checked=(
                "spanning tree normalization",
                "fundamental cycle basis",
                invariance_scope,
            ),
            assumptions_unchecked=audit.unresolved,
            diagnostics={
                "betti_number": len(self.edges) - len(self.vertices) + 1,
                "residual_decisions": audit.decisions,
                "gauge_regime": regime,
                "residual_semantics": residual_semantics,
                "gauge_invariance": invariance_scope,
            },
            theorem_id="PSC-III-T40",
            message=message,
        )

    def gluing_residual_envelope(
        self,
        root_gauge: Any,
        *,
        root: str | None = None,
    ) -> GaugeResidualEnvelope:
        """Bound Frobenius residual distortion under one full-GL root gauge.

        Fundamental holonomies transform by simultaneous conjugation with the
        root gauge. For numerical backends, this method records the sharp
        condition-number envelope implied by submultiplicativity. It does not
        turn a coordinate-dependent tolerance into a gauge-invariant one.
        """

        if self.backend.exact:
            raise ValidationError(
                "gluing_residual_envelope is a numerical conditioning diagnostic; "
                "exact backends should use exact flatness"
            )
        root_vertex = self.vertices[0] if root is None else root
        if root_vertex not in self.vertex_set:
            raise KeyError("unknown root vertex")
        matrix = self.backend.as_matrix(root_gauge)
        if self.backend.shape(matrix) != (self.dimension, self.dimension):
            raise DimensionError("root gauge has the wrong fiber dimension")
        self.backend.inverse(matrix)

        import numpy as np

        numeric = np.asarray(matrix, dtype=complex)
        condition_number = float(np.linalg.cond(numeric, p=2))
        if isfinite(condition_number):
            condition_number = max(1.0, condition_number)

        identity = self.backend.eye(self.dimension)
        source_residuals: dict[str, float] = {}
        lower_bounds: dict[str, float] = {}
        upper_bounds: dict[str, float] = {}
        for item in self.fundamental_holonomies(root_vertex):
            label = self.edge_label(item.edge_index)
            residual = float(self.backend.norm(self.backend.subtract(item.matrix, identity)))
            source_residuals[label] = residual
            if residual == 0.0:
                lower_bounds[label] = 0.0
                upper_bounds[label] = 0.0
            elif isfinite(condition_number):
                lower_bounds[label] = residual / condition_number
                upper_bounds[label] = residual * condition_number
            else:
                lower_bounds[label] = 0.0
                upper_bounds[label] = inf

        return GaugeResidualEnvelope(
            root=root_vertex,
            condition_number=condition_number,
            source_residuals=source_residuals,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

    def transport_root_metric(
        self,
        root_metric: Any,
        root_gauge: Any,
    ) -> Any:
        """Transport a fiber metric under a full-GL coordinate reparameterization.

        If coefficient vectors transform as ``v' = g v``, the Hermitian metric
        matrix must transform as ``Q' = g^{-*} Q g^{-1}`` in order to represent
        the same abstract inner product. This is coordinate covariance, not a
        claim that an unrestricted active GL gauge preserves a fixed metric.
        """

        if self.backend.name != "numpy":
            raise ValidationError(
                "metric transport currently requires the NumPy numerical backend"
            )

        import numpy as np

        metric = _numeric_root_metric(
            root_metric,
            dimension=self.dimension,
            tolerance=self.tolerance,
        )
        gauge = np.asarray(self.backend.as_matrix(root_gauge), dtype=complex)
        if gauge.shape != (self.dimension, self.dimension):
            raise DimensionError("root gauge has the wrong fiber dimension")
        try:
            inverse = np.linalg.inv(gauge)
        except np.linalg.LinAlgError as exc:
            raise ValidationError("root gauge must be invertible") from exc
        transformed = inverse.conj().T @ metric @ inverse
        transformed = 0.5 * (transformed + transformed.conj().T)
        transformed.setflags(write=False)
        return transformed

    def metric_gluing_certificate(
        self,
        root_metric: Any,
        root: str | None = None,
    ) -> Certificate:
        """Certify numerical flatness relative to a transported fiber metric.

        For ``E = H - I`` and a positive-definite Hermitian root metric ``Q``,
        the weighted Hilbert--Schmidt residual is

        ``||E||_{HS,Q}^2 = tr(Q^{-1} E^* Q E)``.

        Under the paired full-GL coordinate change

        ``H' = g H g^{-1}``, ``Q' = g^{-*} Q g^{-1}``,

        this residual is unchanged in exact arithmetic. The certificate is
        therefore coordinate-covariant under full GL only when the metric is
        transported with the representation. It is not an invariant of the
        bare conjugacy class of ``H`` alone.
        """

        if self.backend.name != "numpy" or self.backend.exact:
            raise ValidationError(
                "metric_gluing_certificate currently requires the NumPy numerical backend"
            )

        import numpy as np

        root_vertex = self.vertices[0] if root is None else root
        if root_vertex not in self.vertex_set:
            raise KeyError("unknown root vertex")
        metric = _numeric_root_metric(
            root_metric,
            dimension=self.dimension,
            tolerance=self.tolerance,
        )
        metric_condition = float(np.linalg.cond(metric, p=2))
        identity = np.eye(self.dimension, dtype=complex)
        residuals: dict[str, float] = {}
        for item in self.fundamental_holonomies(root_vertex):
            label = self.edge_label(item.edge_index)
            holonomy = np.asarray(item.matrix, dtype=complex)
            error = holonomy - identity
            energy_matrix = error.conj().T @ metric @ error
            weighted_square = float(
                np.real(np.trace(np.linalg.solve(metric, energy_matrix)))
            )
            roundoff = self.tolerance * max(
                1.0,
                float(np.linalg.norm(energy_matrix, ord="fro")),
            )
            if weighted_square < -roundoff:
                raise ValidationError(
                    "metric-weighted holonomy residual became negative beyond roundoff"
                )
            residuals[label] = float(np.sqrt(max(0.0, weighted_square)))

        valid = all(value <= self.tolerance for value in residuals.values())
        return Certificate(
            valid=bool(valid),
            exactness=Exactness.NUMERICAL,
            residuals=residuals,
            tolerance=self.tolerance,
            assumptions_checked=(
                "spanning tree normalization",
                "fundamental cycle basis",
                "positive-definite Hermitian root metric",
                "full-GL coordinate changes transport the root metric covariantly",
            ),
            diagnostics={
                "betti_number": len(self.edges) - len(self.vertices) + 1,
                "gauge_regime": "full_gl_metric_covariant",
                "residual_semantics": "root-metric Hilbert-Schmidt tolerance",
                "metric_condition_number": metric_condition,
                "gauge_invariance": (
                    "invariant under paired H -> g H g^-1 and "
                    "Q -> g^{-*} Q g^-1 coordinate changes"
                ),
            },
            theorem_id="PSC-III-T40-METRIC-01",
            message=(
                "all fundamental holonomies are within the metric-covariant numerical tolerance"
                if valid
                else "one or more fundamental holonomies exceed the metric-covariant numerical tolerance"
            ),
        )

    @documented(
        ProvenanceRecord(
            identifier="PSC-III-T41",
            title="Fundamental holonomy intertwining criterion",
            status=ClaimStatus.PROVED,
            source="Paper III",
            assumptions=("two transport systems on the same graph", "root intertwiner"),
            verification_suite="nonlinear.holonomy",
        )
    )
    def intertwine(
        self,
        other: "TransportGraph",
        root_intertwiner: Any,
        *,
        root: str | None = None,
    ) -> IntertwiningResult:
        if self.vertices != other.vertices or len(self.edges) != len(other.edges):
            raise ValidationError("transport systems must use the same ordered graph")
        if self.backend.name != other.backend.name or self.dimension != other.dimension:
            raise ValidationError("transport systems must use one backend and fiber dimension")
        for left, right in zip(self.edges, other.edges, strict=True):
            if (left.source, left.target) != (right.source, right.target):
                raise ValidationError("transport edge orientations do not agree")
        root_vertex = self.vertices[0] if root is None else root
        source_paths = self.tree_transports(root_vertex)
        target_paths = other.tree_transports(root_vertex)
        root_map = self.backend.as_matrix(root_intertwiner)
        if self.backend.shape(root_map) != (self.dimension, self.dimension):
            raise DimensionError("root intertwiner has the wrong fiber dimension")
        intertwiners = {
            vertex: self.backend.matmul(
                target_paths[vertex],
                self.backend.matmul(root_map, self.backend.inverse(source_paths[vertex])),
            )
            for vertex in self.vertices
        }
        audit = ResidualAudit(self.backend, self.tolerance)
        edge_residuals: dict[str, float] = {}
        for index, (source_edge, target_edge) in enumerate(zip(self.edges, other.edges, strict=True)):
            residual = self.backend.subtract(
                self.backend.matmul(intertwiners[source_edge.target], self.backend.as_matrix(source_edge.matrix)),
                self.backend.matmul(self.backend.as_matrix(target_edge.matrix), intertwiners[source_edge.source]),
            )
            edge_residuals[self.edge_label(index)] = audit.check("edge:" + self.edge_label(index), residual)

        source_holonomy = self.fundamental_holonomies(root_vertex)
        target_holonomy = other.fundamental_holonomies(root_vertex)
        cycle_residuals: dict[str, float] = {}
        for source_item, target_item in zip(source_holonomy, target_holonomy, strict=True):
            residual = self.backend.subtract(
                self.backend.matmul(root_map, source_item.matrix),
                self.backend.matmul(target_item.matrix, root_map),
            )
            cycle_residuals[self.edge_label(source_item.edge_index)] = audit.check("cycle:" + self.edge_label(source_item.edge_index), residual)
        valid = audit.valid
        certificate = Certificate(
            valid=bool(valid),
            exactness=Exactness.EXACT if self.backend.exact else Exactness.NUMERICAL,
            residuals={
                **{f"edge:{name}": value for name, value in edge_residuals.items()},
                **{f"cycle:{name}": value for name, value in cycle_residuals.items()},
            },
            tolerance=None if self.backend.exact else self.tolerance,
            assumptions_checked=("common graph", "tree propagated vertex intertwiners", "fundamental cycle tests"),
            assumptions_unchecked=audit.unresolved,
            diagnostics={"root": root_vertex, "cycles": len(cycle_residuals), "residual_decisions": audit.decisions},
            theorem_id="PSC-III-T41",
            message="global packet transports intertwine" if valid else "transport intertwining obstruction is nonzero",
        )
        return IntertwiningResult(intertwiners, edge_residuals, cycle_residuals, certificate)


__all__ = [
    "FundamentalHolonomy",
    "GaugeResidualEnvelope",
    "IntertwiningResult",
    "TransportEdge",
    "TransportGraph",
]
