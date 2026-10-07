"""Structured operators, packet compilation, precision planning, and scheduling."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..registry import FeatureSpec

from .benchmarks import (
    BenchmarkComparison,
    BenchmarkContract,
    BenchmarkResult,
    TimingScope,
    benchmark_callable,
    compare_benchmarks,
)
from .compiler import (
    DecisionMargin,
    TreeNode,
    TreePlan,
    combine_tree_objectives,
    compile_tree_budget,
    compile_tree_lagrangian,
    decision_margins,
    pareto_frontier,
    replace_tree_nodes,
)
from .complexity import (
    CostEstimate,
    arithmetic_intensity,
    block_diagonal_low_rank_flops,
    block_diagonal_low_rank_storage_entries,
    dense_apply_flops,
    packet_apply_flops,
    packet_cost_estimate,
    packet_dense_storage_entries,
    packet_streaming_bytes,
    roofline_time_lower_bound,
)
from .discovery import (
    BDLRDiscoveryResult,
    SupportDiscoveryResult,
    discover_block_diagonal_low_rank,
    discover_packet_support,
)
from .gpu import (
    CupyPacketExecutor,
    DeviceExecutionResult,
    NumpyPacketExecutor,
    cupy_available,
    packet_executor,
)
from .ir import Address, PacketIR
from .krylov import (
    BlockJacobiCertificate,
    ConjugateGradientResult,
    KrylovBasisResult,
    LanczosResult,
    block_jacobi_certificate,
    block_jacobi_preconditioner,
    conjugate_gradient,
    krylov_basis,
    lanczos_tridiagonal,
    power_perturbation_bound,
    preconditioned_condition_bound,
)
from .layout import PacketLayout
from .low_rank import (
    BlockDiagonalLowRankOperator,
    LowRankOperator,
    RecompressionResult,
    accumulated_recompression_bound,
    product_chain_error_bound,
    recompress_core,
    recompress_factors,
)
from .precision import (
    DTypeChoice,
    PrecisionAssignment,
    assign_discrete_precisions,
    clipped_precision_allocation,
    optimal_precision_tolerances,
    packet_error_bound,
    precision_certificate,
    quantization_error,
    quantize_array,
    quantize_by_packet,
)
from .scheduling import (
    BreakEvenResult,
    ScheduledTask,
    Task,
    TaskSchedule,
    amortization_threshold,
    batched_launch_count,
    break_even_analysis,
    schedule_unlimited,
    shape_classes,
    verify_schedule,
)
from .structured import (
    BlockDiagonalOperator,
    CirculantOperator,
    DenseOperator,
    DiagonalOperator,
    IdentityOperator,
    KroneckerOperator,
    StructuredOperator,
    ToeplitzOperator,
    WalshHadamardOperator,
    as_structured_operator,
)
from .support import (
    SupportGraph,
    inverse_support_if_acyclic,
    longest_path_vertices,
    schur_fill_support,
    support_polynomial,
    support_power,
    support_product,
    topological_levels,
    transitive_closure,
)

FEATURE_KEY = "algorithms"


def status() -> "FeatureSpec":
    """Return the registered feature specification for this namespace."""

    from ..registry import FeatureSpec, feature

    result: FeatureSpec = feature(FEATURE_KEY)
    return result


def require() -> None:
    """Confirm that the structured algorithms namespace is implemented."""

    return None


__all__ = [
    "Address",
    "BDLRDiscoveryResult",
    "BenchmarkComparison",
    "BenchmarkContract",
    "BenchmarkResult",
    "BlockDiagonalLowRankOperator",
    "BlockDiagonalOperator",
    "BlockJacobiCertificate",
    "BreakEvenResult",
    "CirculantOperator",
    "ConjugateGradientResult",
    "CostEstimate",
    "CupyPacketExecutor",
    "DTypeChoice",
    "DecisionMargin",
    "DenseOperator",
    "DeviceExecutionResult",
    "DiagonalOperator",
    "FEATURE_KEY",
    "IdentityOperator",
    "KroneckerOperator",
    "KrylovBasisResult",
    "LanczosResult",
    "LowRankOperator",
    "NumpyPacketExecutor",
    "PacketIR",
    "PacketLayout",
    "PrecisionAssignment",
    "RecompressionResult",
    "ScheduledTask",
    "StructuredOperator",
    "SupportDiscoveryResult",
    "SupportGraph",
    "Task",
    "TaskSchedule",
    "TimingScope",
    "ToeplitzOperator",
    "TreeNode",
    "TreePlan",
    "WalshHadamardOperator",
    "accumulated_recompression_bound",
    "amortization_threshold",
    "arithmetic_intensity",
    "as_structured_operator",
    "assign_discrete_precisions",
    "batched_launch_count",
    "benchmark_callable",
    "block_diagonal_low_rank_flops",
    "block_diagonal_low_rank_storage_entries",
    "block_jacobi_certificate",
    "block_jacobi_preconditioner",
    "break_even_analysis",
    "clipped_precision_allocation",
    "combine_tree_objectives",
    "compare_benchmarks",
    "compile_tree_budget",
    "compile_tree_lagrangian",
    "conjugate_gradient",
    "cupy_available",
    "decision_margins",
    "dense_apply_flops",
    "discover_block_diagonal_low_rank",
    "discover_packet_support",
    "inverse_support_if_acyclic",
    "krylov_basis",
    "lanczos_tridiagonal",
    "longest_path_vertices",
    "optimal_precision_tolerances",
    "packet_apply_flops",
    "packet_cost_estimate",
    "packet_dense_storage_entries",
    "packet_error_bound",
    "packet_executor",
    "packet_streaming_bytes",
    "pareto_frontier",
    "power_perturbation_bound",
    "precision_certificate",
    "preconditioned_condition_bound",
    "product_chain_error_bound",
    "quantization_error",
    "quantize_array",
    "quantize_by_packet",
    "recompress_core",
    "recompress_factors",
    "replace_tree_nodes",
    "require",
    "roofline_time_lower_bound",
    "schedule_unlimited",
    "schur_fill_support",
    "shape_classes",
    "status",
    "support_polynomial",
    "support_power",
    "support_product",
    "topological_levels",
    "transitive_closure",
    "verify_schedule",
]
