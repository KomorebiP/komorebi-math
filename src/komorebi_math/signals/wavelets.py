"""Finite Haar packet trees, admissible prunings, and exact best bases."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Iterable, Mapping, TypeAlias

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .filterbanks import HaarFilterBank

PacketPath: TypeAlias = str | tuple[int, ...]
CostFunction: TypeAlias = Callable[..., float]


def _strict_depth(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValidationError("depth must be a nonnegative integer")
    result = int(value)
    if result != value or result < 0:
        raise ValidationError("depth must be a nonnegative integer")
    return result


def _path_string(path: PacketPath) -> str:
    if isinstance(path, str):
        if any(symbol not in "01" for symbol in path):
            raise ValidationError("packet paths contain only zero and one")
        return path
    values = tuple(int(value) for value in path)
    if any(value not in (0, 1) for value in values):
        raise ValidationError("packet paths contain only zero and one")
    return "".join(str(value) for value in values)


def _path_tuple(path: PacketPath) -> tuple[int, ...]:
    return tuple(int(symbol) for symbol in _path_string(path))


def _split(coefficients: np.ndarray, normalized: bool) -> tuple[np.ndarray, np.ndarray]:
    return HaarFilterBank(normalized=normalized).analyze(coefficients)


def _merge(low: np.ndarray, high: np.ndarray, normalized: bool) -> np.ndarray:
    return HaarFilterBank(normalized=normalized).synthesize((low, high))


@documented(
    ProvenanceRecord(
        identifier="PSC-V-D12",
        title="Finite Haar packet tree",
        status=ClaimStatus.DEFINITION,
        source="Paper V",
        assumptions=("finite signal length divisible by two to the tree depth",),
        verification_suite="signals.wavelets",
    )
)
@dataclass(frozen=True, slots=True)
class HaarPacketTree:
    """All Haar packet coefficients through one finite binary depth."""

    signal: np.ndarray
    depth: int
    normalized: bool
    _nodes: dict[str, np.ndarray]

    @classmethod
    def analyze(cls, signal: Any, depth: int, *, normalized: bool = True) -> "HaarPacketTree":
        values = np.asarray(signal)
        if values.ndim != 1 or values.size == 0:
            raise DimensionError("Haar packet analysis requires a nonempty one dimensional signal")
        tree_depth = _strict_depth(depth)
        if values.size % (2**tree_depth):
            raise DimensionError("signal length must be divisible by two to the tree depth")
        dtype = np.result_type(values.dtype, float) if normalized else values.dtype
        root = values.astype(dtype, copy=True)
        nodes: dict[str, np.ndarray] = {"": root}
        for level in range(tree_depth):
            for path in tuple(path for path in nodes if len(path) == level):
                low, high = _split(nodes[path], normalized)
                nodes[path + "0"] = low
                nodes[path + "1"] = high
        return cls(root.copy(), tree_depth, bool(normalized), nodes)

    @classmethod
    def decompose(cls, signal: Any, depth: int, *, normalized: bool = True) -> "HaarPacketTree":
        """Alias for :meth:`analyze`."""

        return cls.analyze(signal, depth, normalized=normalized)

    def __post_init__(self) -> None:
        object.__setattr__(self, "signal", np.asarray(self.signal))
        object.__setattr__(self, "depth", _strict_depth(self.depth))
        object.__setattr__(self, "_nodes", {str(path): np.asarray(value) for path, value in self._nodes.items()})

    @property
    def nodes(self) -> dict[str, np.ndarray]:
        """Return a shallow copy of the path to coefficient mapping."""

        return dict(self._nodes)

    @property
    def coefficients(self) -> dict[str, np.ndarray]:
        """Compatibility alias for :attr:`nodes`."""

        return self.nodes

    def coefficient(self, path: PacketPath) -> np.ndarray:
        key = _path_string(path)
        try:
            return self._nodes[key]
        except KeyError as exc:
            raise ValidationError(f"unknown packet path: {key}") from exc

    def paths_at_depth(self, depth: int) -> tuple[str, ...]:
        level = _strict_depth(depth)
        if level > self.depth:
            raise ValidationError("requested depth exceeds the tree depth")
        return tuple(sorted(path for path in self._nodes if len(path) == level))

    @property
    def full_leaves(self) -> tuple[str, ...]:
        return self.paths_at_depth(self.depth)

    def validate_pruning(self, leaves: Iterable[PacketPath]) -> tuple[str, ...]:
        selected = tuple(sorted({_path_string(path) for path in leaves}))
        selected_set = set(selected)
        if not selected:
            raise ValidationError("a pruning must select at least one packet")
        for path in selected:
            if path not in self._nodes:
                raise ValidationError(f"unknown packet path: {path}")
            for length in range(len(path)):
                if path[:length] in selected_set:
                    raise ValidationError("a pruning cannot contain an ancestor and its descendant")

        def covered(path: str) -> bool:
            if path in selected_set:
                return True
            if len(path) == self.depth:
                return False
            return covered(path + "0") and covered(path + "1")

        if not covered(""):
            raise ValidationError("selected packets do not form a complete pruning")
        return selected

    @documented(
        ProvenanceRecord(
            identifier="PSC-V-T13",
            title="Exact reconstruction on a pruned Haar packet tree",
            status=ClaimStatus.PROVED,
            source="Paper V",
            assumptions=("complete finite pruning", "Haar local splits"),
            verification_suite="signals.wavelets",
        )
    )
    def reconstruct(self, leaves: Iterable[PacketPath] | None = None) -> np.ndarray:
        selected = set(self.full_leaves if leaves is None else self.validate_pruning(leaves))

        def recurse(path: str) -> np.ndarray:
            if path in selected:
                return self._nodes[path]
            return _merge(recurse(path + "0"), recurse(path + "1"), self.normalized)

        return recurse("")

    def energy(self, leaves: Iterable[PacketPath] | None = None) -> float:
        selected = self.full_leaves if leaves is None else self.validate_pruning(leaves)
        return float(sum(np.vdot(self._nodes[path], self._nodes[path]).real for path in selected))

    def reconstruction_certificate(
        self,
        leaves: Iterable[PacketPath] | None = None,
        *,
        tolerance: float = 1e-11,
    ) -> ComputationResult[np.ndarray]:
        selected = self.full_leaves if leaves is None else self.validate_pruning(leaves)
        reconstructed = self.reconstruct(selected)
        residual = float(np.linalg.norm(reconstructed - self.signal))
        energy_residual = abs(float(np.vdot(self.signal, self.signal).real) - self.energy(selected))
        valid = residual <= tolerance and (not self.normalized or energy_residual <= tolerance)
        return ComputationResult(
            reconstructed,
            Certificate(
                valid=valid,
                exactness=Exactness.NUMERICAL,
                residuals={"reconstruction": residual, "energy": energy_residual},
                tolerance=tolerance,
                assumptions_checked=("complete pruning", "finite Haar tree"),
                theorem_id="PSC-V-T13",
                message="pruned Haar reconstruction verified" if valid else "pruned Haar reconstruction failed",
            ),
        )

    def certificate(self, *, tolerance: float = 1e-11) -> Certificate:
        return self.reconstruction_certificate(tolerance=tolerance).certificate

    def prune(self, leaves: Iterable[PacketPath]) -> "PrunedPacketCoefficients":
        selected = self.validate_pruning(leaves)
        return PrunedPacketCoefficients(self, selected)


@dataclass(frozen=True, slots=True)
class PrunedPacketCoefficients:
    """Coefficients selected by one complete pruning of a Haar packet tree."""

    tree: HaarPacketTree
    leaves: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "leaves", self.tree.validate_pruning(self.leaves))

    @property
    def coefficients(self) -> dict[str, np.ndarray]:
        return {path: self.tree.coefficient(path) for path in self.leaves}

    def reconstruct(self) -> np.ndarray:
        return self.tree.reconstruct(self.leaves)

    def energy(self) -> float:
        return self.tree.energy(self.leaves)

    def certificate(self, *, tolerance: float = 1e-11) -> Certificate:
        return self.tree.reconstruction_certificate(self.leaves, tolerance=tolerance).certificate


@dataclass(frozen=True, slots=True)
class BestBasisResult:
    """Exact additive best basis result for a finite packet tree."""

    leaves: tuple[str, ...]
    cost: float
    decisions: dict[str, str]
    margins: dict[str, float]
    certificate: Certificate

    def reconstruct(self, tree: HaarPacketTree) -> np.ndarray:
        return tree.reconstruct(self.leaves)

    def pruning(self, tree: HaarPacketTree) -> PrunedPacketCoefficients:
        return tree.prune(self.leaves)


def _evaluate_cost(cost: Mapping[str, float] | CostFunction, coefficients: np.ndarray, path: str) -> float:
    if isinstance(cost, Mapping):
        if path not in cost:
            raise ValidationError(f"no cost was supplied for packet path {path!r}")
        value = float(cost[path])
    else:
        try:
            value = float(cost(coefficients, path))
        except TypeError:
            value = float(cost(coefficients))
    if not np.isfinite(value):
        raise ValidationError("best basis node costs must be finite")
    return value


@documented(
    ProvenanceRecord(
        identifier="PSC-V-T14",
        title="Exact additive best basis recursion",
        status=ClaimStatus.PROVED,
        source="Paper V",
        assumptions=("finite packet tree", "additive node cost"),
        verification_suite="signals.wavelets",
    )
)
def best_basis(tree: HaarPacketTree, cost: Mapping[str, float] | CostFunction) -> BestBasisResult:
    """Return the exact minimum cost pruning of a finite Haar packet tree."""

    decisions: dict[str, str] = {}
    margins: dict[str, float] = {}

    def recurse(path: str) -> tuple[Fraction, tuple[str, ...]]:
        keep_float = _evaluate_cost(cost, tree.coefficient(path), path)
        keep_cost = Fraction.from_float(keep_float)
        if len(path) == tree.depth:
            decisions[path] = "keep"
            margins[path] = float("inf")
            return keep_cost, (path,)
        left_cost, left_leaves = recurse(path + "0")
        right_cost, right_leaves = recurse(path + "1")
        split_cost = left_cost + right_cost
        margins[path] = float(abs(keep_cost - split_cost))
        if keep_cost <= split_cost:
            decisions[path] = "keep"
            return keep_cost, (path,)
        decisions[path] = "split"
        return split_cost, left_leaves + right_leaves

    optimum, leaves = recurse("")
    selected = tree.validate_pruning(leaves)
    return BestBasisResult(
        selected,
        float(optimum),
        decisions,
        margins,
        Certificate(
            valid=True,
            exactness=Exactness.EXACT,
            assumptions_checked=(
                "finite binary tree",
                "additive local costs",
                "exact dyadic comparison of the supplied finite floating costs",
            ),
            diagnostics={
                "exact_cost_numerator": optimum.numerator,
                "exact_cost_denominator": optimum.denominator,
            },
            theorem_id="PSC-V-T14",
            message="exact additive best basis selected over the supplied binary floating costs",
        ),
    )


def entropy_cost(coefficients: np.ndarray, _: str = "") -> float:
    """Return the Shannon entropy of normalized packet energy."""

    energy = np.abs(np.asarray(coefficients)) ** 2
    total = float(np.sum(energy))
    if total == 0.0:
        return 0.0
    probabilities = energy[energy > 0] / total
    return float(-np.sum(probabilities * np.log(probabilities)))


def shannon_entropy_cost(coefficients: np.ndarray, path: str = "") -> float:
    """Compatibility alias for :func:`entropy_cost`."""

    return entropy_cost(coefficients, path)


def lp_cost(exponent: float) -> CostFunction:
    """Return an additive coefficient cost based on a positive power sum."""

    p = float(exponent)
    if not np.isfinite(p) or p <= 0:
        raise ValidationError("cost exponent must be positive and finite")

    def cost(coefficients: np.ndarray, _: str = "") -> float:
        return float(np.sum(np.abs(coefficients) ** p))

    return cost


def haar_packet_analyze(signal: Any, depth: int, *, normalized: bool = True) -> HaarPacketTree:
    """Build a complete finite Haar packet tree to the requested depth."""

    return HaarPacketTree.analyze(signal, depth, normalized=normalized)


def haar_packet_synthesize(
    tree_or_coefficients: HaarPacketTree | PrunedPacketCoefficients,
    leaves: Iterable[PacketPath] | None = None,
) -> np.ndarray:
    """Reconstruct a signal from a complete tree or a pruned packet coefficient set."""

    if isinstance(tree_or_coefficients, PrunedPacketCoefficients):
        if leaves is not None:
            raise ValidationError("leaves cannot be supplied with pruned coefficients")
        return tree_or_coefficients.reconstruct()
    return tree_or_coefficients.reconstruct(leaves)


def packet_energy(
    tree_or_coefficients: HaarPacketTree | PrunedPacketCoefficients,
    leaves: Iterable[PacketPath] | None = None,
) -> float:
    """Return total squared coefficient energy for a complete or pruned packet representation."""

    if isinstance(tree_or_coefficients, PrunedPacketCoefficients):
        if leaves is not None:
            raise ValidationError("leaves cannot be supplied with pruned coefficients")
        return tree_or_coefficients.energy()
    return tree_or_coefficients.energy(leaves)


__all__ = [
    "BestBasisResult",
    "CostFunction",
    "HaarPacketTree",
    "PacketPath",
    "PrunedPacketCoefficients",
    "best_basis",
    "entropy_cost",
    "haar_packet_analyze",
    "haar_packet_synthesize",
    "lp_cost",
    "packet_energy",
    "shannon_entropy_cost",
]
