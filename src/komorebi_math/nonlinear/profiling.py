"""Operation counts and observational profiles for finite packet-jet runtimes.

The deterministic estimates in this module are exact finite counts.  Wall-clock
and memory observations are environment dependent and are therefore reported as
heuristic measurements with the interpreter, platform, repetition count, and
scope recorded.  No timing result is promoted to a hardware-independent
complexity or speed claim.
"""

from __future__ import annotations

import gc
import platform
import statistics
import sys
import tempfile
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from ..certificates import Certificate, Exactness
from ..exceptions import ValidationError
from ..export.persistence import dumps_portable
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .execution import (
    PacketJetExecutionPlan,
    compile_packet_jet_plan,
    execute_packet_jet_plan,
)
from .jets import PacketJet, PacketJetFamily, PacketJetSupport
from .snapshots import PacketJetSnapshot, snapshot_packet_jet
from .streaming import save_packet_jet_snapshot_stream

T = TypeVar("T")


def _positive_integer(value: Any, *, name: str) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a positive integer") from exc
    if isinstance(value, bool) or normalized != value or normalized < 1:
        raise ValidationError(f"{name} must be a positive integer")
    return normalized


def _nonnegative_integer(value: Any, *, name: str) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a nonnegative integer") from exc
    if isinstance(value, bool) or normalized != value or normalized < 0:
        raise ValidationError(f"{name} must be a nonnegative integer")
    return normalized


@dataclass(frozen=True, slots=True)
class PacketJetComplexityEstimate:
    """Exact finite counts implied by packet and ambient dimensions."""

    order: int
    source_dimension: int
    target_dimension: int
    source_packets: int
    target_packets: int
    family_addresses: int
    active_addresses: int
    pruned_addresses: int
    input_groups: int
    naive_derivative_calls: int
    planned_derivative_calls: int
    projected_input_vectors: int
    output_projections: int
    dense_basis_evaluations: int
    dense_tensor_elements: int
    certificate: Certificate

    @property
    def derivative_call_reduction(self) -> float:
        if self.naive_derivative_calls == 0:
            return 1.0
        return 1.0 - self.planned_derivative_calls / self.naive_derivative_calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "source_dimension": self.source_dimension,
            "target_dimension": self.target_dimension,
            "source_packets": self.source_packets,
            "target_packets": self.target_packets,
            "family_addresses": self.family_addresses,
            "active_addresses": self.active_addresses,
            "pruned_addresses": self.pruned_addresses,
            "input_groups": self.input_groups,
            "naive_derivative_calls": self.naive_derivative_calls,
            "planned_derivative_calls": self.planned_derivative_calls,
            "derivative_call_reduction": self.derivative_call_reduction,
            "projected_input_vectors": self.projected_input_vectors,
            "output_projections": self.output_projections,
            "dense_basis_evaluations": self.dense_basis_evaluations,
            "dense_tensor_elements": self.dense_tensor_elements,
            "certificate": self.certificate.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RuntimeProfileSample:
    """One observed duration and Python allocation peak."""

    duration_ns: int
    traced_peak_bytes: int

    def __post_init__(self) -> None:
        if self.duration_ns < 0 or self.traced_peak_bytes < 0:
            raise ValidationError("runtime profile observations must be nonnegative")


@dataclass(frozen=True, slots=True)
class RuntimeStageProfile:
    """Repeated observations for one precisely named runtime stage."""

    name: str
    scope: str
    warmup: int
    samples: tuple[RuntimeProfileSample, ...]

    @property
    def repetitions(self) -> int:
        return len(self.samples)

    @property
    def minimum_duration_ns(self) -> int:
        return min(sample.duration_ns for sample in self.samples)

    @property
    def median_duration_ns(self) -> int:
        return int(statistics.median(sample.duration_ns for sample in self.samples))

    @property
    def maximum_duration_ns(self) -> int:
        return max(sample.duration_ns for sample in self.samples)

    @property
    def median_peak_bytes(self) -> int:
        return int(statistics.median(sample.traced_peak_bytes for sample in self.samples))

    @property
    def maximum_peak_bytes(self) -> int:
        return max(sample.traced_peak_bytes for sample in self.samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scope": self.scope,
            "warmup": self.warmup,
            "repetitions": self.repetitions,
            "minimum_duration_ns": self.minimum_duration_ns,
            "median_duration_ns": self.median_duration_ns,
            "maximum_duration_ns": self.maximum_duration_ns,
            "median_peak_bytes": self.median_peak_bytes,
            "maximum_peak_bytes": self.maximum_peak_bytes,
            "samples": tuple(
                {
                    "duration_ns": sample.duration_ns,
                    "traced_peak_bytes": sample.traced_peak_bytes,
                }
                for sample in self.samples
            ),
        }


@dataclass(frozen=True, slots=True)
class PacketJetProfileReport:
    """Exact counts paired with environment-specific runtime observations."""

    estimate: PacketJetComplexityEstimate
    stages: tuple[RuntimeStageProfile, ...]
    environment: Mapping[str, Any]
    certificate: Certificate

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate.to_dict(),
            "stages": tuple(stage.to_dict() for stage in self.stages),
            "environment": dict(self.environment),
            "certificate": self.certificate.to_dict(),
        }


def estimate_packet_jet_complexity(
    family: PacketJetFamily,
    support: PacketJetSupport | None = None,
    *,
    plan: PacketJetExecutionPlan | None = None,
    max_addresses: int = 1_000_000,
) -> PacketJetComplexityEstimate:
    """Return exact finite operation counts for a packet-jet family."""

    compiled = compile_packet_jet_plan(
        family,
        support,
        max_addresses=max_addresses,
    ) if plan is None else plan
    compiled.certificate.require()
    if not compiled.matches(family):
        raise ValidationError("packet-jet complexity plan does not match the family")
    family_addresses = len(family.target) * (len(family.source) ** family.order)
    dense_basis = family.source.dimension**family.order
    dense_elements = family.target.dimension * dense_basis
    valid = family_addresses == compiled.full_address_count
    certificate = Certificate(
        valid=valid,
        exactness=Exactness.EXACT,
        residuals={
            "address_count_mismatch": float(
                abs(family_addresses - compiled.full_address_count)
            )
        },
        assumptions_checked=(
            "finite packet counts and derivative order",
            "complete family address enumeration",
            "execution-plan grouping and pruning certificate",
            "ambient standard-basis cardinality",
        ),
        assumptions_unchecked=(
            "wall-clock cost per derivative call",
            "backend-specific kernel fusion, caching, and transfer overhead",
        ),
        diagnostics={
            "support_supplied": support is not None,
            "support_complete": compiled.support_complete,
        },
        theorem_id="PSC-III-A61",
        message=(
            "finite packet-jet complexity counts agree"
            if valid
            else "finite packet-jet complexity count mismatch"
        ),
    )
    return PacketJetComplexityEstimate(
        order=family.order,
        source_dimension=family.source.dimension,
        target_dimension=family.target.dimension,
        source_packets=len(family.source),
        target_packets=len(family.target),
        family_addresses=family_addresses,
        active_addresses=compiled.active_address_count,
        pruned_addresses=compiled.pruned_address_count,
        input_groups=len(compiled.groups),
        naive_derivative_calls=compiled.naive_derivative_call_count,
        planned_derivative_calls=compiled.derivative_call_count,
        projected_input_vectors=family.order * len(family.source),
        output_projections=compiled.active_address_count,
        dense_basis_evaluations=dense_basis,
        dense_tensor_elements=dense_elements,
        certificate=certificate,
    )


def profile_operation(
    name: str,
    operation: Callable[[], T],
    *,
    repetitions: int = 5,
    warmup: int = 1,
    scope: str = "Python call including allocations and synchronization performed by the callable",
) -> RuntimeStageProfile:
    """Observe one callable without interpreting timings as asymptotic evidence."""

    count = _positive_integer(repetitions, name="profile repetitions")
    warm = _nonnegative_integer(warmup, name="profile warmup")
    if not callable(operation):
        raise ValidationError("profile operation must be callable")
    normalized_name = str(name).strip()
    normalized_scope = str(scope).strip()
    if not normalized_name or not normalized_scope:
        raise ValidationError("profile name and scope must be nonempty")
    for _ in range(warm):
        operation()
    samples: list[RuntimeProfileSample] = []
    for _ in range(count):
        gc.collect()
        inherited_trace = tracemalloc.is_tracing()
        if inherited_trace:
            baseline, _ = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
        else:
            baseline = 0
            tracemalloc.start()
        start = time.perf_counter_ns()
        try:
            operation()
        finally:
            stop = time.perf_counter_ns()
            _, peak = tracemalloc.get_traced_memory()
            if not inherited_trace:
                tracemalloc.stop()
        samples.append(
            RuntimeProfileSample(stop - start, max(0, int(peak) - int(baseline)))
        )
    return RuntimeStageProfile(normalized_name, normalized_scope, warm, tuple(samples))


@documented(
    ProvenanceRecord(
        identifier="PSC-III-A61",
        title="Finite packet-jet execution profile",
        status=ClaimStatus.HEURISTIC,
        source="streaming nonlinear runtime",
        assumptions=("fixed local process and supplied finite workload",),
        limitations=(
            "timings are environment dependent",
            "tracemalloc observes Python-managed allocations rather than total device or process memory",
        ),
        verification_suite="nonlinear.profiling",
    )
)
def profile_packet_jet_runtime(
    family: PacketJetFamily,
    directions: Sequence[Any],
    *,
    support: PacketJetSupport | None = None,
    jet: PacketJet | None = None,
    repetitions: int = 5,
    warmup: int = 1,
    max_addresses: int = 1_000_000,
    max_snapshot_evaluations: int = 100_000,
    include_stream_write: bool = False,
    stream_chunk_size: int = 1024,
) -> PacketJetProfileReport:
    """Profile selected packet-jet stages under one explicit local contract."""

    direction_tuple = tuple(directions)
    plan = compile_packet_jet_plan(family, support, max_addresses=max_addresses)
    plan.certificate.require()
    estimate = estimate_packet_jet_complexity(family, support, plan=plan)
    estimate.certificate.require()

    stages: list[RuntimeStageProfile] = [
        profile_operation(
            "address_enumeration",
            family.addresses,
            repetitions=repetitions,
            warmup=warmup,
            scope="complete in-memory enumeration of family packet addresses",
        ),
        profile_operation(
            "plan_compilation",
            lambda: compile_packet_jet_plan(
                family,
                support,
                max_addresses=max_addresses,
            ),
            repetitions=repetitions,
            warmup=warmup,
            scope="address enumeration, certified pruning, and grouping into input tuples",
        ),
        profile_operation(
            "plan_execution",
            lambda: execute_packet_jet_plan(plan, family, direction_tuple),
            repetitions=repetitions,
            warmup=warmup,
            scope="input projections, grouped provider calls, output projections, and reconstruction check",
        ),
    ]

    captured: PacketJetSnapshot | None = None
    if jet is not None:
        if jet.order != family.order:
            raise ValidationError("profiled packet jet order does not match the family")
        stages.append(
            profile_operation(
                "snapshot_capture",
                lambda: snapshot_packet_jet(
                    jet,
                    max_evaluations=max_snapshot_evaluations,
                ),
                repetitions=repetitions,
                warmup=warmup,
                scope="complete standard-basis capture of one fixed packet-jet block",
            )
        )
        captured = snapshot_packet_jet(
            jet,
            max_evaluations=max_snapshot_evaluations,
        )
        stages.append(
            profile_operation(
                "portable_serialization",
                lambda: dumps_portable(captured, indent=None),
                repetitions=repetitions,
                warmup=warmup,
                scope="canonical portable JSON construction and payload SHA-256",
            )
        )
        if include_stream_write:
            size = _positive_integer(stream_chunk_size, name="profile stream chunk size")

            def write_stream() -> None:
                with tempfile.TemporaryDirectory() as directory:
                    save_packet_jet_snapshot_stream(
                        captured,
                        Path(directory) / "snapshot.kmstream",
                        chunk_size=size,
                    )

            stages.append(
                profile_operation(
                    "stream_snapshot_write",
                    write_stream,
                    repetitions=repetitions,
                    warmup=warmup,
                    scope="chunk JSON encoding, per-chunk digesting, fsync, manifest updates, and promotion",
                )
            )

    environment = {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "timer": "time.perf_counter_ns",
        "memory_observer": "tracemalloc Python allocation peak",
        "repetitions": _positive_integer(repetitions, name="profile repetitions"),
        "warmup": _nonnegative_integer(warmup, name="profile warmup"),
    }
    certificate = Certificate(
        valid=True,
        exactness=Exactness.HEURISTIC,
        assumptions_checked=(
            "all exact finite operation counts passed their structural certificate",
            "every stage used the same supplied family and direction tuple",
            "timing and allocation scopes are recorded per stage",
        ),
        assumptions_unchecked=(
            "scheduler isolation, fixed CPU frequency, cache state, and thermal stability",
            "cross-platform or cross-device comparability",
            "native and device allocations not visible to tracemalloc",
        ),
        diagnostics={
            "stages": len(stages),
            "snapshot_profiled": captured is not None,
            "stream_write_profiled": bool(include_stream_write and captured is not None),
        },
        theorem_id="PSC-III-A61",
        message="local packet-jet runtime observations recorded without a portable speed claim",
    )
    return PacketJetProfileReport(
        estimate,
        tuple(stages),
        environment,
        certificate,
    )


__all__ = [
    "PacketJetComplexityEstimate",
    "PacketJetProfileReport",
    "RuntimeProfileSample",
    "RuntimeStageProfile",
    "estimate_packet_jet_complexity",
    "profile_operation",
    "profile_packet_jet_runtime",
]
