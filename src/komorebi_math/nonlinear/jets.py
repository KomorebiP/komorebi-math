"""Packet jets, derivative reconstruction, support, and Taylor data."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations, product
from math import factorial
from typing import Any

import numpy as np

from .._residuals import value_audit, finite_tolerance, _exact_status
from .._prerequisites import packet_certificate_dependencies
from ..certificates import Certificate, Exactness
from ..core.packets import PacketSystem
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .derivatives import DerivativeProvider

PacketLabel = int | str


def _index(system: PacketSystem, label: PacketLabel) -> int:
    index = int(label) if isinstance(label, int) and not isinstance(label, bool) else system.index(str(label))
    if index < 0 or index >= len(system):
        raise ValidationError("packet index is outside the packet system")
    return index


def _exactness(provider: DerivativeProvider, source: PacketSystem, target: PacketSystem) -> Exactness:
    if provider.exactness is Exactness.EXACT and source.backend.exact and target.backend.exact:
        return Exactness.EXACT
    if provider.exactness in (Exactness.CONDITIONAL, Exactness.HEURISTIC, Exactness.PLANNED):
        return provider.exactness
    return Exactness.NUMERICAL


def _packet_basis(system: PacketSystem, packet: int, *, tolerance: float) -> tuple[Any, ...]:
    projector = system.projectors[packet]
    if system.backend.name == "sympy":
        return tuple(projector.columnspace())
    matrix = np.asarray(projector, dtype=complex)
    u, singular_values, _ = np.linalg.svd(matrix)
    if singular_values.size == 0:
        return ()
    threshold = max(tolerance, tolerance * float(singular_values[0]))
    rank = int(np.sum(singular_values > threshold))
    return tuple(u[:, index] for index in range(rank))


@dataclass(frozen=True, slots=True)
class PacketJetAddress:
    """The output and ordered input packet addresses of one jet block."""

    output: str
    inputs: tuple[str, ...]

    @property
    def order(self) -> int:
        return len(self.inputs)

    def to_dict(self) -> dict[str, Any]:
        return {"order": self.order, "output": self.output, "inputs": self.inputs}


@documented(
    ProvenanceRecord(
        identifier="PSC-III-D06",
        title="Finite packet jet",
        status=ClaimStatus.DEFINITION,
        source="Paper III",
        assumptions=("finite packet systems", "declared finite dimensional derivative provider"),
        verification_suite="nonlinear.jets",
    )
)
@dataclass(frozen=True, slots=True)
class PacketJet:
    """One projected multilinear derivative block.

    Inputs are projected into the declared source packets before the derivative
    is evaluated, and the output is projected into the declared target packet.
    This permits ambient vectors to be supplied safely while preserving the
    mathematical restriction to packet subspaces.
    """

    provider: DerivativeProvider
    point: Any
    source: PacketSystem
    target: PacketSystem
    output_index: int
    input_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.source.backend.name != self.target.backend.name:
            raise ValidationError("source and target packet systems must use one backend")
        if self.provider.source_dimension != self.source.dimension:
            raise DimensionError("provider source dimension does not match the source packet system")
        if self.provider.target_dimension != self.target.dimension:
            raise DimensionError("provider target dimension does not match the target packet system")
        if not self.provider.supports_order(len(self.input_indices)):
            raise ValidationError("provider does not supply the requested jet order")
        if self.output_index < 0 or self.output_index >= len(self.target):
            raise ValidationError("output packet index is outside the target system")
        if any(index < 0 or index >= len(self.source) for index in self.input_indices):
            raise ValidationError("an input packet index is outside the source system")
        point = self.source.backend.as_vector(self.point)
        if self.source.backend.shape(point)[0] != self.source.dimension:
            raise DimensionError("jet base point has the wrong source dimension")

    @property
    def order(self) -> int:
        return len(self.input_indices)

    @property
    def address(self) -> PacketJetAddress:
        return PacketJetAddress(
            self.target.names[self.output_index],
            tuple(self.source.names[index] for index in self.input_indices),
        )

    @property
    def exactness(self) -> Exactness:
        return _exactness(self.provider, self.source, self.target)

    def evaluate(self, directions: Sequence[Any]) -> Any:
        direction_tuple = tuple(directions)
        if len(direction_tuple) != self.order:
            raise ValidationError(f"order {self.order} jet requires exactly {self.order} directions")
        projected: list[Any] = []
        for index, direction in zip(self.input_indices, direction_tuple, strict=True):
            vector = self.source.backend.as_vector(direction)
            if self.source.backend.shape(vector)[0] != self.source.dimension:
                raise DimensionError("jet direction has the wrong source dimension")
            projected.append(self.source.backend.matmul(self.source.projectors[index], vector))
        value = self.target.backend.as_vector(
            self.provider.derivative(self.order, self.point, tuple(projected))
        )
        if self.target.backend.shape(value)[0] != self.target.dimension:
            raise DimensionError("jet derivative has the wrong target dimension")
        return self.target.backend.matmul(self.target.projectors[self.output_index], value)

    __call__ = evaluate

    def norm_bound(self, derivative_norm: float) -> float:
        """Return the projector-product bound from the packet jet theorem."""

        bound = float(derivative_norm) * self.target.backend.norm(self.target.projectors[self.output_index])
        for index in self.input_indices:
            bound *= self.source.backend.norm(self.source.projectors[index])
        return float(bound)

    def symmetry_certificate(self, directions: Sequence[Any], *, tolerance: float | None = None) -> Certificate:
        """Check permutation symmetry with the correspondingly permuted address."""

        direction_tuple = tuple(directions)
        if len(direction_tuple) != self.order:
            raise ValidationError("symmetry validation requires one direction per input address")
        tol = max(self.source.tolerance, self.target.tolerance) if tolerance is None else float(tolerance)
        audit = value_audit(exact=self.exactness is Exactness.EXACT, tolerance=tol)
        reference = self.evaluate(direction_tuple)
        residual = 0.0
        for permutation in permutations(range(self.order)):
            permuted = PacketJet(
                self.provider,
                self.point,
                self.source,
                self.target,
                self.output_index,
                tuple(self.input_indices[index] for index in permutation),
            )
            candidate = permuted.evaluate(tuple(direction_tuple[index] for index in permutation))
            residual = max(
                residual,
                audit.check_value(f"symmetry:{permutation}", self.target.backend.subtract(candidate, reference)),
            )
        parents_valid, parent_checked, parent_unchecked, parent_diagnostics = packet_certificate_dependencies(
            source=self.source, target=self.target
        )
        valid = parents_valid and audit.valid
        return Certificate(
            valid=valid,
            exactness=self.exactness,
            residuals={"jet_symmetry": residual},
            tolerance=None if self.exactness is Exactness.EXACT else tol,
            assumptions_checked=parent_checked + ("finite paired permutation comparisons", "corresponding packet address permutation"),
            assumptions_unchecked=parent_unchecked + ("global correctness of user supplied derivatives",) + audit.unresolved,
            diagnostics={**parent_diagnostics, "residual_decisions": audit.decisions},
            theorem_id="PSC-III-D06",
            message="packet jet is symmetric under paired permutations" if valid else "packet jet symmetry check failed",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "address": self.address.to_dict(),
            "provider": self.provider.name,
            "exactness": self.exactness.value,
            "source_dimension": self.source.dimension,
            "target_dimension": self.target.dimension,
        }


@documented(
    ProvenanceRecord(
        identifier="PSC-III-T07",
        title="Exact reconstruction of finite packet derivatives",
        status=ClaimStatus.PROVED,
        source="Paper III",
        assumptions=("complete finite packet systems", "multilinear derivative"),
        verification_suite="nonlinear.jets",
    )
)
class PacketJetFamily:
    """Every packet block of one derivative order at one base point."""

    def __init__(
        self,
        provider: DerivativeProvider,
        point: Any,
        source: PacketSystem,
        target: PacketSystem,
        order: int,
    ) -> None:
        self.provider = provider
        self.point = point
        self.source = source
        self.target = target
        self.order = int(order)
        if isinstance(order, bool) or self.order != order or self.order < 1:
            raise ValidationError("jet order must be a positive integer")
        if source.backend.name != target.backend.name:
            raise ValidationError("source and target packet systems must use one backend")
        if provider.source_dimension != source.dimension or provider.target_dimension != target.dimension:
            raise DimensionError("provider dimensions do not match the packet systems")
        if not provider.supports_order(self.order):
            raise ValidationError("provider does not supply the requested jet order")
        point_vector = source.backend.as_vector(point)
        if source.backend.shape(point_vector)[0] != source.dimension:
            raise DimensionError("jet base point has the wrong source dimension")
        self.exactness = _exactness(provider, source, target)

    def jet(self, output: PacketLabel, inputs: Sequence[PacketLabel]) -> PacketJet:
        input_tuple = tuple(inputs)
        if len(input_tuple) != self.order:
            raise ValidationError(f"order {self.order} jet requires {self.order} input packet labels")
        return PacketJet(
            self.provider,
            self.point,
            self.source,
            self.target,
            _index(self.target, output),
            tuple(_index(self.source, label) for label in input_tuple),
        )

    def addresses(self) -> tuple[PacketJetAddress, ...]:
        return tuple(
            PacketJetAddress(self.target.names[output], tuple(self.source.names[index] for index in inputs))
            for output in range(len(self.target))
            for inputs in product(range(len(self.source)), repeat=self.order)
        )

    def reconstruct(self, directions: Sequence[Any]) -> Any:
        """Reconstruct the full derivative by summing every packet jet."""

        direction_tuple = tuple(directions)
        if len(direction_tuple) != self.order:
            raise ValidationError(f"order {self.order} reconstruction requires {self.order} directions")
        total = self.target.backend.zeros(self.target.dimension, 1)
        if self.target.backend.name != "sympy":
            total = np.asarray(total).reshape(-1)
        for output in range(len(self.target)):
            for inputs in product(range(len(self.source)), repeat=self.order):
                value = PacketJet(
                    self.provider,
                    self.point,
                    self.source,
                    self.target,
                    output,
                    tuple(inputs),
                ).evaluate(direction_tuple)
                total = self.target.backend.add(total, value)
        return total

    def reconstruction_certificate(
        self,
        directions: Sequence[Any],
        *,
        tolerance: float | None = None,
    ) -> Certificate:
        tol = max(self.source.tolerance, self.target.tolerance) if tolerance is None else float(tolerance)
        direct = self.target.backend.as_vector(
            self.provider.derivative(self.order, self.point, tuple(directions))
        )
        reconstructed = self.target.backend.as_vector(self.reconstruct(directions))
        audit = value_audit(exact=self.exactness is Exactness.EXACT, tolerance=tol)
        residual = audit.check_value("derivative_reconstruction", self.target.backend.subtract(direct, reconstructed))
        parents_valid, parent_checked, parent_unchecked, parent_diagnostics = packet_certificate_dependencies(
            source=self.source, target=self.target
        )
        valid = parents_valid and audit.valid
        return Certificate(
            valid=valid,
            exactness=self.exactness,
            residuals={"derivative_reconstruction": residual},
            tolerance=None if self.exactness is Exactness.EXACT else tol,
            assumptions_checked=parent_checked + ("finite packet sum and direct comparison at the supplied directions",),
            assumptions_unchecked=parent_unchecked + ("global multilinearity/correctness of supplied derivatives",) + audit.unresolved,
            diagnostics={
                **parent_diagnostics,
                "order": self.order,
                "residual_decisions": audit.decisions,
                "address_count": len(self.target) * len(self.source) ** self.order,
                "provider": self.provider.name,
            },
            theorem_id="PSC-III-T07",
            message="packet jets reconstruct the derivative" if valid else "packet jet reconstruction failed",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "provider": self.provider.name,
            "exactness": self.exactness.value,
            "source_packets": self.source.names,
            "target_packets": self.target.names,
            "address_count": len(self.target) * len(self.source) ** self.order,
        }


@dataclass(frozen=True, slots=True)
class PacketJetSupport:
    """Finite basis test for zero and nonzero packet derivative blocks."""

    order: int
    nonzero: tuple[PacketJetAddress, ...]
    zero: tuple[PacketJetAddress, ...]
    untested: tuple[PacketJetAddress, ...]
    evaluations: int
    certificate: Certificate

    @property
    def complete(self) -> bool:
        return not self.untested

    def contains(self, output: str, inputs: Sequence[str]) -> bool:
        address = PacketJetAddress(str(output), tuple(str(value) for value in inputs))
        return address in self.nonzero


@documented(
    ProvenanceRecord(
        identifier="PSC-III-C08",
        title="Finite packet jet support test",
        status=ClaimStatus.FINITE_CERTIFICATE,
        source="Paper III runtime",
        assumptions=("finite dimensional packet ranges", "basis test of a multilinear map"),
        verification_suite="nonlinear.jets",
    )
)
def packet_jet_support(
    family: PacketJetFamily,
    *,
    tolerance: float | None = None,
    max_evaluations: int = 100_000,
) -> PacketJetSupport:
    """Classify packet jet blocks by exhaustive packet-range basis tests.

    Multilinearity makes the classification exact over an exact backend.  A
    numerical backend obtains a tolerance-certified finite classification.
    ``max_evaluations`` prevents accidental combinatorial explosions; any
    remaining addresses are returned explicitly as untested. Undecidable or
    nonfinite exact probes also leave an address unclassified in ``untested``;
    a later witnessed nonzero probe can still establish nonzero support.
    """

    limit = int(max_evaluations)
    if isinstance(max_evaluations, bool) or limit != max_evaluations or limit < 1:
        raise ValidationError("max_evaluations must be a positive integer")
    tol = max(family.source.tolerance, family.target.tolerance) if tolerance is None else float(tolerance)
    tol = finite_tolerance(tol)
    undecided_probes: list[dict[str, Any]] = []
    bases = tuple(
        _packet_basis(family.source, packet, tolerance=tol)
        for packet in range(len(family.source))
    )
    nonzero: list[PacketJetAddress] = []
    zero: list[PacketJetAddress] = []
    untested: list[PacketJetAddress] = []
    evaluations = 0
    exhausted = False
    addresses = family.addresses()
    for address_index, address in enumerate(addresses):
        output = family.target.index(address.output)
        inputs = tuple(family.source.index(name) for name in address.inputs)
        block_nonzero = False
        block_unknown = False
        for directions in product(*(bases[index] for index in inputs)):
            if evaluations >= limit:
                exhausted = True
                break
            evaluations += 1
            value = family.jet(output, inputs).evaluate(directions)
            if family.target.backend.exact:
                status, witness = _exact_status(family.target.backend, value)
                if status == "nonzero":
                    block_nonzero = True
                    break
                if status != "zero":
                    block_unknown = True
                    undecided_probes.append({"address": address.to_dict(), "status": status, "witness": witness})
            elif not family.target.backend.is_zero(value, tolerance=tol):
                block_nonzero = True
                break
        if exhausted:
            untested.extend(addresses[address_index:])
            break
        if block_nonzero:
            nonzero.append(address)
        elif block_unknown:
            untested.append(address)
        else:
            zero.append(address)

    exactness = family.exactness if not untested else Exactness.CONDITIONAL
    parents_valid, parent_checked, parent_unchecked, parent_diagnostics = packet_certificate_dependencies(
        source=family.source, target=family.target
    )
    return PacketJetSupport(
        family.order,
        tuple(nonzero),
        tuple(zero),
        tuple(untested),
        evaluations,
        Certificate(
            valid=parents_valid,
            exactness=exactness,
            residuals={},
            tolerance=None if family.exactness is Exactness.EXACT else tol,
            assumptions_checked=parent_checked + ("packet range bases", "tested packet addresses"),
            assumptions_unchecked=parent_unchecked + ("global multilinearity/correctness of supplied derivatives",) +
                (("unclassified addresses due to evaluation limit or undecided/nonfinite exact values",) if untested else ()) ,
            diagnostics={
                **parent_diagnostics,
                "order": family.order,
                "nonzero_blocks": len(nonzero),
                "zero_blocks": len(zero),
                "untested_blocks": len(untested),
                "undecided_exact_probes": undecided_probes,
                "basis_evaluations": evaluations,
            },
            theorem_id="PSC-III-C08",
            message="packet jet support classified" if not untested else "packet jet support partially classified",
        ),
    )


@dataclass(frozen=True, slots=True)
class PacketTaylorResult:
    """A truncated Taylor value assembled from reconstructed packet jets."""

    approximation: Any
    observed: Any
    truncation_residual: float
    order_certificates: tuple[Certificate, ...]
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="PSC-III-C09",
        title="Finite packet Taylor reconstruction",
        status=ClaimStatus.FINITE_CERTIFICATE,
        source="Paper III",
        assumptions=("declared derivatives through the truncation order",),
        limitations=("no automatic bound on the omitted Taylor remainder",),
        verification_suite="nonlinear.jets",
    )
)
def packet_taylor(
    provider: DerivativeProvider,
    point: Any,
    increment: Any,
    source: PacketSystem,
    target: PacketSystem,
    order: int,
    *,
    tolerance: float | None = None,
) -> PacketTaylorResult:
    """Assemble a finite Taylor polynomial through packet jet reconstruction."""

    n = int(order)
    if isinstance(order, bool) or n != order or n < 0:
        raise ValidationError("Taylor order must be a nonnegative integer")
    point_vector = source.backend.as_vector(point)
    increment_vector = source.backend.as_vector(increment)
    if source.backend.shape(point_vector)[0] != source.dimension or source.backend.shape(increment_vector)[0] != source.dimension:
        raise DimensionError("Taylor point and increment must match the source dimension")
    approximation = target.backend.as_vector(provider.value(point_vector))
    certificates: list[Certificate] = []
    for derivative_order in range(1, n + 1):
        family = PacketJetFamily(provider, point_vector, source, target, derivative_order)
        certificate = family.reconstruction_certificate((increment_vector,) * derivative_order, tolerance=tolerance)
        certificates.append(certificate)
        term = target.backend.as_vector(family.reconstruct((increment_vector,) * derivative_order))
        scalar: Any = (
            Fraction(1, factorial(derivative_order))
            if target.backend.exact
            else 1 / factorial(derivative_order)
        )
        approximation = target.backend.add(approximation, target.backend.scale(scalar, term))
    shifted = source.backend.add(point_vector, increment_vector)
    observed = target.backend.as_vector(provider.value(shifted))
    residual = target.backend.norm(target.backend.subtract(observed, approximation))
    exactness = _exactness(provider, source, target)
    parents_valid, parent_checked, parent_unchecked, parent_diagnostics = packet_certificate_dependencies(
        source=source, target=target
    )
    representation_valid = parents_valid and all(certificate.valid for certificate in certificates)
    return PacketTaylorResult(
        approximation,
        observed,
        residual,
        tuple(certificates),
        Certificate(
            valid=representation_valid,
            exactness=exactness,
            residuals={"observed_truncation_residual": residual},
            tolerance=None if exactness is Exactness.EXACT else (max(source.tolerance, target.tolerance) if tolerance is None else tolerance),
            assumptions_checked=parent_checked + ("packet reconstruction of every retained derivative order",),
            assumptions_unchecked=parent_unchecked + ("Taylor remainder beyond the retained order", "global correctness of user supplied derivatives"),
            diagnostics={**parent_diagnostics, "order": n, "provider": provider.name},
            theorem_id="PSC-III-C09",
            message="packet Taylor polynomial assembled; truncation residual is diagnostic",
        ),
    )


__all__ = [
    "PacketJet",
    "PacketJetAddress",
    "PacketJetFamily",
    "PacketJetSupport",
    "PacketLabel",
    "PacketTaylorResult",
    "packet_jet_support",
    "packet_taylor",
]
