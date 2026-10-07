"""Integral packet separation and lattice gluing modules."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Any

from ..certificates import Certificate, Exactness
from ..exceptions import BackendUnavailableError, DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .._number_theory import require_prime
from .smith import padic_valuation, rational_padic_valuation, smith_decomposition


def _sympy() -> Any:
    try:
        import sympy as sp
    except ImportError as exc:
        raise BackendUnavailableError("lattice packet operations require the 'exact' SymPy extra") from exc
    return sp


@dataclass(frozen=True, slots=True)
class GluingModule:
    """Invariant factors of ``lattice / split_lattice``."""

    invariant_factors: tuple[int, ...]
    prime: int

    def __post_init__(self) -> None:
        require_prime(self.prime)
        if any(int(value) <= 0 for value in self.invariant_factors):
            raise ValueError("gluing invariant factors must be positive")

    @property
    def prime_exponents(self) -> tuple[int, ...]:
        return tuple(int(padic_valuation(value, self.prime)) for value in self.invariant_factors if value != 1)

    @property
    def separation_weight(self) -> int:
        exponents = self.prime_exponents
        return max(exponents, default=0)

    @property
    def length(self) -> int:
        return sum(self.prime_exponents)

    @property
    def order(self) -> int:
        result = 1
        for factor in self.invariant_factors:
            result *= abs(int(factor))
        return result


@dataclass(frozen=True, slots=True)
class PrecisionDelay:
    """Necessary and sufficient uniform input delay for one rational projector."""

    weight: int
    prime: int
    projector_index: int | None = None

    def __post_init__(self) -> None:
        require_prime(self.prime)
        if self.weight < 0:
            raise ValueError("precision delay weight must be nonnegative")

    def input_depth(self, output_depth: int) -> int:
        if output_depth < 0:
            raise ValueError("output depth must be nonnegative")
        return output_depth + self.weight


@documented(
    ProvenanceRecord(
        identifier="PSC-II-D12",
        title="Lattice packet data",
        status=ClaimStatus.DEFINITION,
        source="Paper II",
        assumptions=("rational packet projectors", "full lattice basis"),
        verification_suite="refinement.lattices",
    )
)
class LatticePacketData:
    """A rational packet decomposition measured against one integral lattice."""

    def __init__(
        self,
        lattice_basis: Any,
        projectors: Any,
        *,
        prime: int,
        split_basis: Any | None = None,
    ) -> None:
        prime = require_prime(prime)
        sp = _sympy()
        self.lattice_basis = sp.Matrix(lattice_basis)
        if self.lattice_basis.rows != self.lattice_basis.cols:
            raise DimensionError("lattice basis must be square")
        if self.lattice_basis.det() == 0:
            raise ValidationError("lattice basis must be invertible")
        self.projectors = tuple(sp.Matrix(projector) for projector in projectors)
        if not self.projectors:
            raise ValueError("at least one packet projector is required")
        dimension = self.lattice_basis.rows
        if any(projector.shape != (dimension, dimension) for projector in self.projectors):
            raise DimensionError("projectors and lattice basis must have one ambient dimension")
        self.prime = prime
        self.split_basis = None if split_basis is None else sp.Matrix(split_basis)
        if self.split_basis is not None and self.split_basis.shape != self.lattice_basis.shape:
            raise DimensionError("split lattice basis must have the same shape as the ambient lattice basis")
        if self.split_basis is not None:
            inclusion = self.lattice_basis.inv() * self.split_basis
            if any(not entry.is_Integer for entry in inclusion):
                raise ValidationError("split lattice is not integrally contained in the ambient lattice")
        self.certificate().require()

    @property
    def dimension(self) -> int:
        return int(self.lattice_basis.rows)

    def certificate(self) -> Certificate:
        sp = _sympy()
        identity = sp.eye(self.dimension)
        total = sp.zeros(self.dimension)
        valid = True
        for index, projector in enumerate(self.projectors):
            valid &= projector * projector == projector
            total += projector
            for other_index, other in enumerate(self.projectors):
                if index != other_index:
                    valid &= projector * other == sp.zeros(self.dimension)
        valid &= total == identity
        if self.split_basis is not None:
            inclusion = self.inclusion_matrix()
            valid &= all(entry.is_Integer for entry in inclusion)
            valid &= inclusion.det() != 0
        return Certificate(
            valid=bool(valid),
            exactness=Exactness.EXACT,
            residuals={"packet_axioms": 0.0 if valid else 1.0},
            assumptions_checked=("rational packet axioms", "full lattice basis") + (("integral split inclusion",) if self.split_basis is not None else ()),
            theorem_id="PSC-II-D12",
            message="valid lattice packet data" if valid else "invalid rational packet or lattice data",
        )

    def projector_in_lattice_basis(self, index: int) -> Any:
        projector = self.projectors[index]
        return self.lattice_basis.inv() * projector * self.lattice_basis

    @documented(
        ProvenanceRecord(
            identifier="PSC-II-T13",
            title="Exact separation weight from the lattice action",
            status=ClaimStatus.PROVED,
            source="Paper II",
            assumptions=("rational projectors", "discrete valuation at the declared prime"),
            verification_suite="refinement.lattices",
        )
    )
    def projector_weights(self) -> tuple[int, ...]:
        """Return the exact prime denominator exponent of every packet projector."""

        weights: list[int] = []
        for index in range(len(self.projectors)):
            matrix = self.projector_in_lattice_basis(index)
            minimum = inf
            for entry in matrix:
                valuation = rational_padic_valuation(entry, self.prime)
                minimum = min(minimum, valuation)
            weights.append(max(0, -int(minimum)) if minimum != inf else 0)
        return tuple(weights)

    @property
    def separation_weight(self) -> int:
        return max(self.projector_weights(), default=0)

    def precision_delays(self) -> tuple[PrecisionDelay, ...]:
        return tuple(
            PrecisionDelay(weight, self.prime, index)
            for index, weight in enumerate(self.projector_weights())
        )

    def inclusion_matrix(self) -> Any:
        if self.split_basis is None:
            raise ValidationError("a split lattice basis is required for the gluing module")
        return self.lattice_basis.inv() * self.split_basis

    @documented(
        ProvenanceRecord(
            identifier="PSC-II-T13",
            title="Gluing module and exact separation weight",
            status=ClaimStatus.PROVED,
            source="Paper II",
            assumptions=("basis of the split lattice is supplied",),
            verification_suite="refinement.lattices",
        )
    )
    def gluing_module(self) -> GluingModule:
        inclusion = self.inclusion_matrix()
        if any(not entry.is_Integer for entry in inclusion):
            raise ValidationError("split lattice is not integrally contained in the ambient lattice")
        decomposition = smith_decomposition(inclusion)
        return GluingModule(decomposition.invariant_factors, self.prime)

    def gluing_weight_certificate(self) -> Certificate:
        gluing = self.gluing_module()
        direct = self.separation_weight
        valid = gluing.separation_weight == direct
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT,
            residuals={"weight_difference": float(abs(gluing.separation_weight - direct))},
            assumptions_checked=("Smith invariants", "projector denominator valuations"),
            diagnostics={
                "gluing_weight": gluing.separation_weight,
                "direct_weight": direct,
                "length": gluing.length,
            },
            theorem_id="PSC-II-T13",
            message="gluing exponent equals the exact separation weight" if valid else "split basis does not match the declared packet intersections",
        )


def _refinement_parent_map(coarse: LatticePacketData, fine: LatticePacketData) -> tuple[int, ...]:
    """Return the unique coarse parent of every nonzero fine projector."""

    sp = _sympy()
    zero = sp.zeros(coarse.dimension)
    parents: list[int] = []
    children: list[list[int]] = [[] for _ in coarse.projectors]
    for fine_index, fine_projector in enumerate(fine.projectors):
        if fine_projector == zero:
            # A zero packet carries no lattice or separation information.  It
            # may be omitted from every reconstruction sum.
            parents.append(-1)
            continue
        candidates = [
            coarse_index
            for coarse_index, coarse_projector in enumerate(coarse.projectors)
            if coarse_projector * fine_projector == fine_projector
            and fine_projector * coarse_projector == fine_projector
        ]
        if len(candidates) != 1:
            raise ValidationError(
                f"fine packet {fine_index} does not have a unique coarse parent"
            )
        parent = candidates[0]
        parents.append(parent)
        children[parent].append(fine_index)

    for coarse_index, coarse_projector in enumerate(coarse.projectors):
        total = sp.zeros(coarse.dimension)
        for fine_index in children[coarse_index]:
            total += fine.projectors[fine_index]
        if total != coarse_projector:
            raise ValidationError(
                f"fine packets do not reconstruct coarse packet {coarse_index}"
            )
    return tuple(parents)


def refinement_weight_monotonicity(coarse: LatticePacketData, fine: LatticePacketData) -> Certificate:
    """Certify exact weight monotonicity for an actual packet refinement."""

    if coarse.prime != fine.prime:
        raise ValidationError("coarse and fine lattice data must use one prime")
    if coarse.lattice_basis != fine.lattice_basis:
        raise ValidationError("coarse and fine packet systems must use the same lattice")
    parent_map = _refinement_parent_map(coarse, fine)
    valid = coarse.separation_weight <= fine.separation_weight
    return Certificate(
        valid=valid,
        exactness=Exactness.EXACT,
        residuals={"monotonicity_gap": float(fine.separation_weight - coarse.separation_weight)},
        assumptions_checked=("common lattice", "exact packet refinement"),
        diagnostics={"parent_map": parent_map},
        theorem_id="PSC-II-T15",
        message="separation weight is monotone under the exact refinement" if valid else "fine packet data has unexpectedly smaller separation weight",
    )


__all__ = [
    "GluingModule",
    "LatticePacketData",
    "PrecisionDelay",
    "refinement_weight_monotonicity",
]
