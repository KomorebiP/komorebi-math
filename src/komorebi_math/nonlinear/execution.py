"""Projection-cached packet-jet execution and optional CuPy kernels."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from math import factorial
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .._residuals import value_audit, finite_tolerance
from .._prerequisites import packet_certificate_dependencies
from ..certificates import Certificate, Exactness
from ..exceptions import BackendUnavailableError, DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .derivatives import PolynomialDerivativeProvider
from .jets import PacketJetAddress, PacketJetFamily, PacketJetSupport


def nonlinear_cupy_available() -> bool:
    """Report whether a CuPy module can be discovered."""

    return find_spec("cupy") is not None


@dataclass(frozen=True, slots=True)
class PacketJetInputGroup:
    """Active output packets sharing one ordered input-packet address."""

    inputs: tuple[int, ...]
    outputs: tuple[int, ...]

    def __post_init__(self) -> None:
        if any(index < 0 for index in self.inputs) or any(index < 0 for index in self.outputs):
            raise ValidationError("packet-jet plan indices must be nonnegative")
        if not self.outputs:
            raise ValidationError("a packet-jet input group requires at least one output")
        if tuple(sorted(set(self.outputs))) != self.outputs:
            raise ValidationError("packet-jet group outputs must be unique and sorted")


@documented(
    ProvenanceRecord(
        identifier="PSC-III-A10",
        title="Projection-cached packet-jet execution plan",
        status=ClaimStatus.PROVED,
        source="Paper III scalable runtime",
        assumptions=("finite packet systems", "declared zero packet blocks are certified"),
        verification_suite="nonlinear.execution",
    )
)
@dataclass(frozen=True, slots=True)
class PacketJetExecutionPlan:
    """A deterministic grouping of packet-jet blocks by projected inputs."""

    order: int
    source_names: tuple[str, ...]
    target_names: tuple[str, ...]
    groups: tuple[PacketJetInputGroup, ...]
    active_addresses: tuple[PacketJetAddress, ...]
    pruned_addresses: tuple[PacketJetAddress, ...]
    support_was_supplied: bool
    support_complete: bool
    certificate: Certificate

    @property
    def active_address_count(self) -> int:
        return len(self.active_addresses)

    @property
    def pruned_address_count(self) -> int:
        return len(self.pruned_addresses)

    @property
    def full_address_count(self) -> int:
        return self.active_address_count + self.pruned_address_count

    @property
    def derivative_call_count(self) -> int:
        return len(self.groups)

    @property
    def naive_derivative_call_count(self) -> int:
        return self.full_address_count

    @property
    def derivative_call_reduction(self) -> float:
        if self.naive_derivative_call_count == 0:
            return 1.0
        return 1.0 - self.derivative_call_count / self.naive_derivative_call_count

    def matches(self, family: PacketJetFamily) -> bool:
        return (
            family.order == self.order
            and family.source.names == self.source_names
            and family.target.names == self.target_names
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "source_names": self.source_names,
            "target_names": self.target_names,
            "groups": tuple(
                {"inputs": group.inputs, "outputs": group.outputs}
                for group in self.groups
            ),
            "active_addresses": tuple(address.to_dict() for address in self.active_addresses),
            "pruned_addresses": tuple(address.to_dict() for address in self.pruned_addresses),
            "support_was_supplied": self.support_was_supplied,
            "support_complete": self.support_complete,
            "derivative_calls": self.derivative_call_count,
            "naive_derivative_calls": self.naive_derivative_call_count,
            "derivative_call_reduction": self.derivative_call_reduction,
            "certificate": self.certificate.to_dict(),
        }


def compile_packet_jet_plan(
    family: PacketJetFamily,
    support: PacketJetSupport | None = None,
    *,
    max_addresses: int = 1_000_000,
) -> PacketJetExecutionPlan:
    """Compile a projection-cached plan, pruning only certified zero blocks."""

    limit = int(max_addresses)
    if isinstance(max_addresses, bool) or limit != max_addresses or limit < 1:
        raise ValidationError("max_addresses must be a positive integer")
    # Count with saturation so rejection neither constructs the Cartesian
    # product nor creates an enormous arbitrary-precision power first.
    address_count = len(family.target)
    base = min(len(family.source), limit + 1)
    exponent = family.order
    while exponent and address_count <= limit:
        if exponent & 1:
            address_count = min(limit + 1, address_count * base)
        exponent >>= 1
        if exponent:
            base = min(limit + 1, base * base)
    if address_count > limit:
        raise ValidationError(f"packet-jet family is above max_addresses={limit}")
    all_addresses = family.addresses()
    if len(all_addresses) > limit:
        raise ValidationError(
            f"packet-jet family has {len(all_addresses)} addresses above max_addresses={limit}"
        )

    support_supplied = support is not None
    support_complete = True
    zero: set[PacketJetAddress] = set()
    if support is not None:
        support.certificate.require()
        if support.order != family.order:
            raise ValidationError("packet-jet support order does not match the family")
        classified = set(support.nonzero) | set(support.zero) | set(support.untested)
        if classified != set(all_addresses):
            raise ValidationError("packet-jet support must partition every family address")
        zero = set(support.zero)
        support_complete = support.complete

    active_addresses = tuple(address for address in all_addresses if address not in zero)
    pruned_addresses = tuple(address for address in all_addresses if address in zero)
    grouped: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for address in active_addresses:
        inputs = tuple(family.source.index(name) for name in address.inputs)
        grouped[inputs].append(family.target.index(address.output))
    groups = tuple(
        PacketJetInputGroup(inputs, tuple(sorted(outputs)))
        for inputs, outputs in sorted(grouped.items())
    )
    active_from_groups = sum(len(group.outputs) for group in groups)
    valid = active_from_groups == len(active_addresses)
    certificate = Certificate(
        valid=valid,
        exactness=family.exactness,
        residuals={"address_partition": 0.0 if valid else 1.0},
        tolerance=None
        if family.exactness is Exactness.EXACT
        else max(family.source.tolerance, family.target.tolerance),
        assumptions_checked=(
            "all packet-jet addresses were enumerated",
            "only certified zero addresses were pruned",
            "active addresses were grouped by ordered input packet tuple",
        ),
        assumptions_unchecked=(
            "support classification is incomplete, so untested blocks remain active",
        )
        if support is not None and not support.complete
        else (),
        diagnostics={
            "order": family.order,
            "active_addresses": len(active_addresses),
            "pruned_addresses": len(pruned_addresses),
            "input_groups": len(groups),
            "naive_derivative_calls": len(all_addresses),
            "planned_derivative_calls": len(groups),
        },
        theorem_id="PSC-III-A10",
        message="packet-jet execution plan compiled" if valid else "packet-jet execution plan is inconsistent",
    )
    return PacketJetExecutionPlan(
        family.order,
        family.source.names,
        family.target.names,
        groups,
        active_addresses,
        pruned_addresses,
        support_supplied,
        support_complete,
        certificate,
    )


@dataclass(frozen=True, slots=True)
class PacketJetEvaluation:
    """One evaluated packet-jet address and its projected target value."""

    address: PacketJetAddress
    value: Any


@dataclass(frozen=True, slots=True)
class PacketJetBatchResult:
    """Result of one projection-cached packet-jet execution."""

    evaluations: tuple[PacketJetEvaluation, ...]
    reconstructed: Any
    direct: Any
    derivative_calls: int
    input_projections: int
    output_projections: int
    pruned_blocks: int
    backend: str
    returned_to_host: bool
    certificate: Certificate

    def as_mapping(self) -> dict[PacketJetAddress, Any]:
        return {evaluation.address: evaluation.value for evaluation in self.evaluations}

    def value(self, address: PacketJetAddress) -> Any:
        try:
            return self.as_mapping()[address]
        except KeyError as exc:
            raise KeyError(f"packet-jet address was not active: {address}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluations": tuple(
                {"address": evaluation.address.to_dict(), "value": evaluation.value}
                for evaluation in self.evaluations
            ),
            "reconstructed": self.reconstructed,
            "direct": self.direct,
            "derivative_calls": self.derivative_calls,
            "input_projections": self.input_projections,
            "output_projections": self.output_projections,
            "pruned_blocks": self.pruned_blocks,
            "backend": self.backend,
            "returned_to_host": self.returned_to_host,
            "certificate": self.certificate.to_dict(),
        }


def execute_packet_jet_plan(
    plan: PacketJetExecutionPlan,
    family: PacketJetFamily,
    directions: Sequence[Any],
    *,
    tolerance: float | None = None,
) -> PacketJetBatchResult:
    """Execute a packet-jet plan with cached input projections."""

    tol = finite_tolerance(max(family.source.tolerance, family.target.tolerance)
                           if tolerance is None else tolerance)
    plan.certificate.require()
    if not plan.matches(family):
        raise ValidationError("packet-jet plan does not match the supplied family")
    direction_tuple = tuple(directions)
    if len(direction_tuple) != family.order:
        raise ValidationError(f"order {family.order} execution requires {family.order} directions")
    source = family.source
    target = family.target
    projected: list[tuple[Any, ...]] = []
    for direction in direction_tuple:
        vector = source.backend.as_vector(direction)
        if source.backend.shape(vector)[0] != source.dimension:
            raise DimensionError("packet-jet direction has the wrong source dimension")
        projected.append(
            tuple(source.backend.matmul(projector, vector) for projector in source.projectors)
        )

    evaluations: list[PacketJetEvaluation] = []
    total = target.backend.zeros(target.dimension, 1)
    if target.backend.name != "sympy":
        total = np.asarray(total).reshape(-1)
    for group in plan.groups:
        arguments = tuple(
            projected[position][packet_index]
            for position, packet_index in enumerate(group.inputs)
        )
        raw = target.backend.as_vector(
            family.provider.derivative(family.order, family.point, arguments)
        )
        if target.backend.shape(raw)[0] != target.dimension:
            raise DimensionError("packet-jet derivative has the wrong target dimension")
        for output in group.outputs:
            value = target.backend.matmul(target.projectors[output], raw)
            address = PacketJetAddress(
                target.names[output],
                tuple(source.names[index] for index in group.inputs),
            )
            evaluations.append(PacketJetEvaluation(address, value))
            total = target.backend.add(total, value)

    order_index = {address: index for index, address in enumerate(plan.active_addresses)}
    evaluations.sort(key=lambda evaluation: order_index[evaluation.address])
    direct = target.backend.as_vector(
        family.provider.derivative(family.order, family.point, direction_tuple)
    )
    audit = value_audit(exact=family.exactness is Exactness.EXACT, tolerance=tol)
    residual = audit.check_value("planned_reconstruction",
                                target.backend.subtract(direct, target.backend.as_vector(total)))
    parents_valid, parent_checked, parent_unchecked, parent_diagnostics = packet_certificate_dependencies(
        source=source, target=target
    )
    valid = parents_valid and audit.valid
    certificate = Certificate(
        valid=valid,
        exactness=family.exactness,
        residuals={"planned_reconstruction": residual},
        tolerance=None if family.exactness is Exactness.EXACT else tol,
        assumptions_checked=parent_checked + (
            "source directions projected once per packet",
            "one derivative call per active input address",
            "target packets reconstruct the retained outputs",
            "pruned addresses were certified zero",
        ),
        assumptions_unchecked=parent_unchecked + ("global correctness and multilinearity of the derivative provider",
                               "supplied support remains valid for these inputs") + audit.unresolved,
        diagnostics={
            **parent_diagnostics,
            "order": family.order,
            "residual_decisions": audit.decisions,
            "active_blocks": len(evaluations),
            "pruned_blocks": len(plan.pruned_addresses),
            "derivative_calls": len(plan.groups),
            "naive_derivative_calls": plan.naive_derivative_call_count,
            "input_projections": family.order * len(source),
            "output_projections": len(evaluations),
            "provider": family.provider.name,
            "backend": source.backend.name,
        },
        theorem_id="PSC-III-A10",
        message="projection-cached packet jets reconstruct the derivative" if valid else "planned packet-jet reconstruction failed",
    )
    return PacketJetBatchResult(
        tuple(evaluations),
        total,
        direct,
        len(plan.groups),
        family.order * len(source),
        len(evaluations),
        len(plan.pruned_addresses),
        source.backend.name,
        True,
        certificate,
    )


def execute_packet_jets(
    family: PacketJetFamily,
    directions: Sequence[Any],
    *,
    support: PacketJetSupport | None = None,
    tolerance: float | None = None,
    max_addresses: int = 1_000_000,
) -> PacketJetBatchResult:
    """Compile and execute one projection-cached packet-jet family."""

    plan = compile_packet_jet_plan(family, support, max_addresses=max_addresses)
    return execute_packet_jet_plan(plan, family, directions, tolerance=tolerance)


@runtime_checkable
class CupyDeviceDerivativeProvider(Protocol):
    """Optional protocol consumed by :class:`CupyPacketJetExecutor`."""

    source_dimension: int
    target_dimension: int
    max_order: int
    exactness: Exactness
    name: str
    device_backend: str

    def supports_order(self, order: int) -> bool:
        """Return whether one derivative order is available."""

    def device_derivative(self, order: int, point: Any, directions: Sequence[Any]) -> Any:
        """Evaluate a directional derivative without returning it to the host."""


class CupyPolynomialDerivativeProvider:
    """Dense polynomial derivatives resident on one CuPy device.

    The class is importable without CuPy.  Construction fails explicitly when
    CuPy is absent, and no GPU validation claim is made until construction and
    execution succeed on an actual device.
    """

    device_backend = "cupy"
    exactness = Exactness.NUMERICAL

    def __init__(
        self,
        tensors: Mapping[int, Any],
        *,
        source_dimension: int | None = None,
        target_dimension: int | None = None,
        tolerance: float = 1e-10,
        name: str = "CuPy polynomial derivative provider",
    ) -> None:
        if not nonlinear_cupy_available():
            raise BackendUnavailableError("CuPy is not installed")
        import cupy as cp

        dense = PolynomialDerivativeProvider(
            tensors,
            source_dimension=source_dimension,
            target_dimension=target_dimension,
            exactness=Exactness.NUMERICAL,
            tolerance=tolerance,
            name=name,
        )
        if any(np.asarray(tensor).dtype == object for tensor in dense.tensors.values()):
            raise ValidationError("CuPy polynomial tensors must use numerical dtypes")
        self._cp = cp
        self.tensors = {order: cp.asarray(tensor) for order, tensor in dense.tensors.items()}
        self.source_dimension = dense.source_dimension
        self.target_dimension = dense.target_dimension
        self.max_order = dense.degree
        self.tolerance = dense.tolerance
        self.name = dense.name

    @property
    def degree(self) -> int:
        return self.max_order

    def supports_order(self, order: int) -> bool:
        try:
            normalized = int(order)
        except (TypeError, ValueError):
            return False
        return (
            not isinstance(order, bool)
            and normalized == order
            and 1 <= normalized <= self.max_order
        )

    def _vector(self, value: Any, *, name: str) -> Any:
        array = self._cp.asarray(value)
        if array.ndim == 2 and 1 in array.shape:
            array = array.reshape(-1)
        if array.ndim != 1 or int(array.shape[0]) != self.source_dimension:
            raise DimensionError(f"{name} must have dimension {self.source_dimension}")
        return array

    def _contract(self, tensor: Any, vectors: Sequence[Any]) -> Any:
        result = tensor
        for vector in vectors:
            result = self._cp.tensordot(result, vector, axes=(1, 0))
        return result

    def device_value(self, point: Any) -> Any:
        x = self._vector(point, name="point")
        total = self._cp.zeros(self.target_dimension, dtype=self._cp.result_type(x.dtype, *[tensor.dtype for tensor in self.tensors.values()]))
        for order, tensor in self.tensors.items():
            total = total + self._contract(tensor, (x,) * order) / factorial(order)
        return total

    def value(self, point: Any) -> np.ndarray:
        return self._cp.asnumpy(self.device_value(point))

    def device_derivative(self, order: int, point: Any, directions: Sequence[Any]) -> Any:
        derivative_order = int(order)
        if not self.supports_order(derivative_order):
            raise ValidationError(
                f"CuPy polynomial derivative order must lie between 1 and {self.max_order}"
            )
        x = self._vector(point, name="point")
        direction_tuple = tuple(self._vector(direction, name="direction") for direction in directions)
        if len(direction_tuple) != derivative_order:
            raise ValidationError(
                f"order {derivative_order} requires exactly {derivative_order} directions"
            )
        total = self._cp.zeros(self.target_dimension, dtype=self._cp.result_type(x.dtype, *[tensor.dtype for tensor in self.tensors.values()]))
        for tensor_order, tensor in self.tensors.items():
            if tensor_order < derivative_order:
                continue
            remaining = tensor_order - derivative_order
            total = total + self._contract(
                tensor,
                (*direction_tuple, *((x,) * remaining)),
            ) / factorial(remaining)
        return total

    def derivative(self, order: int, point: Any, directions: Sequence[Any]) -> np.ndarray:
        return self._cp.asnumpy(self.device_derivative(order, point, directions))

    def synchronize(self) -> None:
        self._cp.cuda.Stream.null.synchronize()


class CupyPacketJetExecutor:
    """Device-resident packet projections and polynomial jet reconstruction."""

    name = "cupy"

    def __init__(
        self,
        family: PacketJetFamily,
        plan: PacketJetExecutionPlan | None = None,
    ) -> None:
        if not nonlinear_cupy_available():
            raise BackendUnavailableError("CuPy is not installed")
        if family.source.backend.name != "numpy" or family.target.backend.name != "numpy":
            raise ValidationError("CuPy packet-jet execution currently requires NumPy packet systems")
        if not isinstance(family.provider, CupyDeviceDerivativeProvider):
            raise ValidationError("CuPy execution requires a CupyDeviceDerivativeProvider")
        import cupy as cp

        self._cp = cp
        self.family = family
        self.plan = compile_packet_jet_plan(family) if plan is None else plan
        self.plan.certificate.require()
        if not self.plan.matches(family):
            raise ValidationError("packet-jet plan does not match the CuPy family")
        self._source_projectors = tuple(cp.asarray(projector) for projector in family.source.projectors)
        self._target_projectors = tuple(cp.asarray(projector) for projector in family.target.projectors)
        self._point = cp.asarray(family.point)

    def execute(
        self,
        directions: Sequence[Any],
        *,
        return_host: bool = True,
        input_on_device: bool = False,
        tolerance: float | None = None,
    ) -> PacketJetBatchResult:
        cp = self._cp
        direction_tuple = tuple(directions)
        if len(direction_tuple) != self.family.order:
            raise ValidationError(
                f"order {self.family.order} execution requires {self.family.order} directions"
            )
        arrays = tuple(direction if input_on_device else cp.asarray(direction) for direction in direction_tuple)
        if any(array.ndim != 1 or int(array.shape[0]) != self.family.source.dimension for array in arrays):
            raise DimensionError("CuPy packet-jet directions have the wrong source dimension")
        projected = tuple(
            tuple(projector @ direction for projector in self._source_projectors)
            for direction in arrays
        )
        evaluations: list[PacketJetEvaluation] = []
        total = cp.zeros(self.family.target.dimension, dtype=cp.result_type(*[array.dtype for array in arrays]))
        provider = self.family.provider
        for group in self.plan.groups:
            arguments = tuple(
                projected[position][packet_index]
                for position, packet_index in enumerate(group.inputs)
            )
            raw = provider.device_derivative(self.family.order, self._point, arguments)
            for output in group.outputs:
                value = self._target_projectors[output] @ raw
                address = PacketJetAddress(
                    self.family.target.names[output],
                    tuple(self.family.source.names[index] for index in group.inputs),
                )
                evaluations.append(PacketJetEvaluation(address, value))
                total = total + value
        direct = provider.device_derivative(self.family.order, self._point, arrays)
        residual = float(cp.asnumpy(cp.linalg.norm(direct - total)))
        tol = max(self.family.source.tolerance, self.family.target.tolerance) if tolerance is None else float(tolerance)
        valid = residual <= tol
        order_index = {address: index for index, address in enumerate(self.plan.active_addresses)}
        evaluations.sort(key=lambda evaluation: order_index[evaluation.address])
        if return_host:
            evaluations = [
                PacketJetEvaluation(evaluation.address, cp.asnumpy(evaluation.value))
                for evaluation in evaluations
            ]
            returned_total: Any = cp.asnumpy(total)
            returned_direct: Any = cp.asnumpy(direct)
        else:
            returned_total = total
            returned_direct = direct
        certificate = Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"device_reconstruction": residual},
            tolerance=tol,
            assumptions_checked=(
                "CuPy installed",
                "packet projectors transferred to the active device",
                "device derivative provider used without host round trips inside the kernel",
                "device reconstruction compared with the direct device derivative",
            ),
            assumptions_unchecked=(
                "cross-backend agreement with a CPU reference",
                "performance comparison under a matched synchronized timing contract",
            ),
            diagnostics={
                "active_blocks": len(evaluations),
                "pruned_blocks": len(self.plan.pruned_addresses),
                "derivative_calls": len(self.plan.groups),
                "input_on_device": bool(input_on_device),
                "returned_to_host": bool(return_host),
            },
            theorem_id="PSC-III-A10G",
            message="CuPy packet-jet reconstruction completed" if valid else "CuPy packet-jet reconstruction failed",
        )
        return PacketJetBatchResult(
            tuple(evaluations),
            returned_total,
            returned_direct,
            len(self.plan.groups),
            self.family.order * len(self.family.source),
            len(evaluations),
            len(self.plan.pruned_addresses),
            "cupy",
            bool(return_host),
            certificate,
        )

    def synchronize(self) -> None:
        self._cp.cuda.Stream.null.synchronize()

    def hardware_certificate(self) -> Certificate:
        """Return device metadata only after a CUDA device was queried."""

        cp = self._cp
        device_id = int(cp.cuda.runtime.getDevice())
        properties = cp.cuda.runtime.getDeviceProperties(device_id)
        raw_name = properties.get("name", b"unknown")
        device_name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
        return Certificate(
            valid=True,
            exactness=Exactness.NUMERICAL,
            assumptions_checked=("CuPy runtime loaded", "active CUDA device queried"),
            assumptions_unchecked=("matched performance benchmark",),
            diagnostics={
                "device_id": device_id,
                "device_name": device_name,
                "compute_capability": (
                    int(properties.get("major", 0)),
                    int(properties.get("minor", 0)),
                ),
                "total_global_memory": int(properties.get("totalGlobalMem", 0)),
            },
            theorem_id="PSC-III-A10G",
            message="CUDA device is available for packet-jet execution",
        )


def packet_jet_benchmark_contract(
    family: PacketJetFamily,
    *,
    backend: str,
    timing_scope: str = "kernel",
) -> Any:
    """Create a matched benchmark contract without importing algorithms eagerly."""

    from ..algorithms.benchmarks import BenchmarkContract

    return BenchmarkContract.create(
        {
            "order": family.order,
            "source_dimension": family.source.dimension,
            "target_dimension": family.target.dimension,
            "source_packets": len(family.source),
            "target_packets": len(family.target),
            "address_count": len(family.addresses()),
            "provider": family.provider.name,
        },
        "full directional derivative plus active packet-address values",
        timing_scope,
        backend,
    )


# Preferred public spelling.  The original ``Cupy*`` names remain aliases so
# serialized references and early Block 10 development snapshots stay usable.
CuPyDeviceDerivativeProvider = CupyDeviceDerivativeProvider
CuPyPolynomialDerivativeProvider = CupyPolynomialDerivativeProvider
CuPyPacketJetExecutor = CupyPacketJetExecutor


__all__ = [
    "CuPyDeviceDerivativeProvider",
    "CuPyPacketJetExecutor",
    "CuPyPolynomialDerivativeProvider",
    "CupyDeviceDerivativeProvider",
    "CupyPacketJetExecutor",
    "CupyPolynomialDerivativeProvider",
    "PacketJetBatchResult",
    "PacketJetEvaluation",
    "PacketJetExecutionPlan",
    "PacketJetInputGroup",
    "compile_packet_jet_plan",
    "execute_packet_jet_plan",
    "execute_packet_jets",
    "nonlinear_cupy_available",
    "packet_jet_benchmark_contract",
]
